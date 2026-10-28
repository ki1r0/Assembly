"""One bounded, evidence-recording bolt-insertion episode."""

from __future__ import annotations

import json
import math
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from .agent import BoltHarnessAgent
from .evaluator import BoltSeatTolerances
from .executor import BoltSkillExecutor, make_nominal_bolt_motion_plan
from .harness import BoltHarnessLoop, BoltHarnessResult
from .measurement import (
    BoltSeatSamplingAdapter,
    BoltSeatSamplingCriteria,
    CapSupportContactCriteria,
)
from .recording import EpisodeRGBVideoRecorder


VIDEO_FPS = 12.0
MAX_HELD_SAMPLE_S = 0.3
MAX_RELEASE_SAMPLE_S = 1.0
_DIMENSION_FIELDS = {
    "bolt_diameter_m", "bolt_length_m", "hole_diameter_m", "seat_depth_m", "clearance_m",
}
_PUBLIC_GOAL = (
    "Pick up the initially unheld bolt, carry it to the selected cover socket, actively insert and "
    "seat it while maintaining the cap grasp, check held readiness, then release, retract, and "
    "verify that the assembly remains seated."
)
_SUCCESS_STATUSES = {"completed", "succeeded"}


@dataclass(frozen=True)
class BoltEpisodeResult:
    """Episode outcome; callers should propagate ``exit_code`` to their runner."""

    status: str
    exit_code: int
    task_success: bool
    video_finalized: bool
    model_evidence: bool
    run_dir: Path
    error: str | None = None
    harness_result: BoltHarnessResult | None = None


class _ScriptedDevelopmentAgent:
    """Fixed development sequence; never used as a VLM fallback."""

    def __init__(self, target_id: str) -> None:
        self.target_id = target_id
        self.trace_callback = None

    def next_tool_call(self, observation, task_spec, history, *, completed=False):
        calls = [item["call"] for item in history]
        skills = [item["skill"] for item in calls if item.get("tool") == "execute_skill"]
        checks = [item["result"] for item in history if item["call"].get("tool") == "check_task"]
        if "pick" not in skills:
            return self._skill("pick")
        if "transport" not in skills:
            return self._skill("transport")
        if "insert_and_seat" not in skills:
            return self._skill("insert_and_seat")
        if not checks:
            return {"tool": "check_task"}
        if not checks[0].get("completed", False):
            return {"tool": "stop"}
        if "release_and_retract" not in skills:
            return self._skill("release_and_retract")
        if len(checks) < 2:
            return {"tool": "check_task"}
        return {"tool": "stop"}

    def _skill(self, skill: str) -> dict[str, Any]:
        mode = "compliant" if skill == "insert_and_seat" else "default"
        return {
            "tool": "execute_skill", "skill": skill, "target_id": self.target_id,
            "mode": mode, "max_duration_s": 30.0,
        }


class _ReleaseTrackingExecutor:
    """Track only successful executor release calls for phase-specific evaluation."""

    def __init__(self, executor: BoltSkillExecutor) -> None:
        self._executor = executor
        self.release_completed = False
        self.completed_insertions = 0

    def execute_skill(self, skill: str, target_id: str, **arguments: Any) -> Mapping[str, Any]:
        result = self._executor.execute_skill(skill, target_id, **arguments)
        if isinstance(result, Mapping) and result.get("status") in _SUCCESS_STATUSES:
            if skill == "release_and_retract":
                self.release_completed = True
            elif skill == "insert_and_seat":
                self.completed_insertions += 1
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._executor, name)


def _exact_mapping(value: Any, name: str, expected: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        raise ValueError(f"{name} fields differ; missing={sorted(missing)}, unexpected={sorted(extra)}")
    return value


def _dataclass_from_settings(kind: type, value: Any, name: str):
    expected = {field.name for field in fields(kind)}
    return kind(**dict(_exact_mapping(value, name, expected)))


def _validated_settings(settings: Any, agent_mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if agent_mode not in {"vlm", "scripted"}:
        raise ValueError("agent_mode must be 'vlm' or 'scripted'")
    if not isinstance(settings, Mapping):
        raise TypeError("settings must be a mapping")

    source_task = settings.get("task_spec")
    if not isinstance(source_task, Mapping):
        raise TypeError("settings.task_spec must be a mapping")
    if "nominal_pose" in source_task:
        raise ValueError("task_spec.nominal_pose is excluded; exact final root coordinates are not public")
    task_id = source_task.get("task_id")
    target_id = source_task.get("target_id")
    socket_id = source_task.get("socket_id")
    if any(not isinstance(item, str) or not item.strip() for item in (task_id, target_id, socket_id)):
        raise ValueError("task_spec requires non-empty task_id, target_id, and socket_id")
    dimensions = source_task.get("nominal_dimensions")
    if not isinstance(dimensions, Mapping):
        raise TypeError("task_spec.nominal_dimensions must be a mapping of public CAD dimensions")
    public_dimensions = {}
    for key in _DIMENSION_FIELDS & dimensions.keys():
        value = dimensions[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"task_spec.nominal_dimensions.{key} must be finite numeric CAD data")
        public_dimensions[key] = float(value)
    if not public_dimensions:
        raise ValueError("task_spec.nominal_dimensions must include an allowlisted CAD dimension")
    direction = source_task.get("assembly_direction")
    if not isinstance(direction, Sequence) or isinstance(direction, (str, bytes)) or len(direction) != 3:
        raise ValueError("task_spec.assembly_direction must be a public three-value CAD direction")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in direction):
        raise ValueError("task_spec.assembly_direction must contain finite numbers")
    task_spec = {
        "task_id": task_id.strip(), "goal": _PUBLIC_GOAL, "target_id": target_id.strip(),
        "target_ids": [target_id.strip()], "socket_id": socket_id.strip(),
        "nominal_dimensions": public_dimensions,
        "assembly_direction": [float(item) for item in direction],
    }

    tolerances = _dataclass_from_settings(BoltSeatTolerances, settings.get("tolerances"), "settings.tolerances")
    sampling_source = settings.get("sampling_criteria")
    if not isinstance(sampling_source, Mapping):
        raise TypeError("settings.sampling_criteria must be calibrated explicitly")
    sampling_fields = {field.name for field in fields(BoltSeatSamplingCriteria)}
    _exact_mapping(sampling_source, "settings.sampling_criteria", sampling_fields)
    criteria_values = dict(sampling_source)
    criteria_values["cap_support"] = _dataclass_from_settings(
        CapSupportContactCriteria, criteria_values["cap_support"], "sampling_criteria.cap_support"
    )
    criteria = BoltSeatSamplingCriteria(**criteria_values)
    if tolerances.held_seat_dwell_s > MAX_HELD_SAMPLE_S:
        raise ValueError(f"held_seat_dwell_s exceeds the {MAX_HELD_SAMPLE_S}s sampling bound")
    if tolerances.stable_dwell_s > MAX_RELEASE_SAMPLE_S:
        raise ValueError(f"stable_dwell_s exceeds the {MAX_RELEASE_SAMPLE_S}s sampling bound")

    executor_source = _exact_mapping(
        settings.get("executor"), "settings.executor", {"max_gripper_force_n", "max_insertion_force_n"}
    )
    forces = {}
    for key, value in executor_source.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"settings.executor.{key} must be a calibrated positive finite value")
        forces[key] = float(value)

    model_config = None
    if agent_mode == "vlm":
        model_source = _exact_mapping(settings.get("model"), "settings.model", {"endpoint", "model"})
        endpoint, model_name = model_source["endpoint"], model_source["model"]
        if not isinstance(endpoint, str) or not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("settings.model requires a configured HTTP endpoint and model name")
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("model endpoint must be HTTP(S) without embedded credentials, query, or fragment")
        model_config = {"endpoint": endpoint, "model": model_name.strip()}

    snapshot = {
        "agent_mode": agent_mode,
        "model": model_config,
        "task_spec": task_spec,
        "executor": forces,
        "tolerances": asdict(tolerances),
        "sampling_criteria": asdict(criteria),
        "video_fps": VIDEO_FPS,
    }
    return {
        "task_spec": task_spec, "tolerances": tolerances, "criteria": criteria,
        "executor": forces, "model": model_config,
    }, snapshot


def _numpy(value: Any) -> np.ndarray:
    if callable(getattr(value, "detach", None)):
        value = value.detach()
    if callable(getattr(value, "cpu", None)):
        value = value.cpu()
    if callable(getattr(value, "numpy", None)):
        value = value.numpy()
    return np.asarray(value)


def _scalar(value: Any, name: str) -> float:
    values = _numpy(value)
    if values.size != 1:
        raise ValueError(f"{name} must contain exactly one value")
    result = float(values.reshape(-1)[0])
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vector(value: Any, name: str, size: int | None = None) -> list[float]:
    values = _numpy(value)
    if values.ndim >= 2:
        if values.shape[0] != 1:
            raise ValueError(f"{name} supports exactly one environment")
        values = values[0]
    result = [float(item) for item in values.reshape(-1)]
    if not result or (size is not None and len(result) != size) or any(not math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain {size or 'finite'} values")
    return result


def _rgb8(value: Any) -> np.ndarray:
    rgb = _numpy(value)
    if rgb.ndim == 4:
        if rgb.shape[0] != 1:
            raise ValueError("task RGB camera must contain exactly one environment")
        rgb = rgb[0]
    if rgb.ndim != 3 or rgb.shape[2] not in {3, 4}:
        raise ValueError("task_rgb_camera output['rgb'] must be HxWx3/4")
    rgb = rgb[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        if not np.isfinite(rgb).all() or np.any(rgb < 0.0) or np.any(rgb > 1.0):
            raise ValueError("floating RGB camera values must be finite and in [0, 1]")
        rgb = np.rint(rgb * 255.0).astype(np.uint8)
    elif rgb.dtype != np.uint8:
        raise ValueError("RGB camera output must be uint8 or normalized floating point")
    return np.ascontiguousarray(rgb)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if callable(getattr(value, "detach", None)):
        return _jsonable(_numpy(value))
    raise TypeError(f"cannot serialize evidence value of type {type(value).__name__}")


def _append_jsonl(stream, value: Mapping[str, Any]) -> None:
    stream.write(json.dumps(_jsonable(value), ensure_ascii=True, allow_nan=False) + "\n")
    stream.flush()


def _sim_time(env: Any) -> float:
    return _scalar(env._robot._data._sim_timestamp, "environment simulation timestamp")


def _zero_action(env: Any) -> Any:
    action = env.actions
    if tuple(action.shape) != (1, 6):
        raise ValueError("Factory executor environment actions must have shape (1, 6)")
    if callable(getattr(action, "new_zeros", None)):
        return action.new_zeros(action.shape)
    return np.zeros_like(action)


def _single_flag(value: Any, name: str) -> bool:
    array = _numpy(value)
    if array.size != 1:
        raise ValueError(f"{name} must contain one environment flag")
    return bool(array.reshape(-1)[0])


def _save_png(path: Path, rgb: np.ndarray) -> None:
    import imageio.v2 as imageio

    imageio.imwrite(path, rgb, format="png")


def run_bolt_episode(env: Any, settings: Mapping[str, Any], run_dir: str | Path, agent_mode: str) -> BoltEpisodeResult:
    """Run one already-reset episode without resetting or replacing model decisions.

    Required settings contain ``task_spec`` (task_id, target_id, socket_id,
    nominal_dimensions, assembly_direction), complete ``tolerances`` and
    ``sampling_criteria`` dataclass fields, and calibrated executor force caps.
    VLM mode additionally requires ``model.endpoint`` and ``model.model``.
    """
    root = Path(run_dir)
    private_dir = root / "private"
    video_path = root / "episode.mp4"
    output_paths = (
        root / "config.yaml", root / "public_trace.jsonl", root / "run_summary.json",
        root / "frames", video_path, video_path.with_suffix(".timestamps.json"), root / "help_request.json",
        private_dir / "evaluator_trace.jsonl", private_dir / "exception.txt",
    )
    root.mkdir(parents=True, exist_ok=True)
    if any(path.exists() for path in output_paths):
        return BoltEpisodeResult(
            "output_conflict", 1, False, False, False, root,
            "episode evidence path already exists; existing files were preserved",
        )

    started = time.monotonic()
    public_stream = private_stream = None
    recorder = None
    original_step = None
    had_instance_step = False
    previous_instance_step = None
    harness_result = None
    failure: BaseException | None = None
    failure_info = None
    error = None
    video_error = None
    frame_count = 0
    latest_frame_path: Path | None = None
    latest_frame_id: str | None = None
    latest_frame_index: int | None = None
    latest_status = None
    latest_measurement = None
    initial_unheld = False
    pickup_proven = False
    model_requests = 0
    model_responses = 0
    executor_proxy = None
    normalized = None
    snapshot = None
    help_outbox_written = False

    try:
        private_dir.mkdir(parents=True, exist_ok=True)
        public_stream = (root / "public_trace.jsonl").open("x", encoding="utf-8")
        private_stream = (private_dir / "evaluator_trace.jsonl").open("x", encoding="utf-8")
        normalized, snapshot = _validated_settings(settings, agent_mode)
        with (root / "config.yaml").open("x", encoding="utf-8") as config_stream:
            config_stream.write(json.dumps(snapshot, indent=2, ensure_ascii=True, allow_nan=False) + "\n")

        baseline = getattr(env, "wrench_baseline_stats", None)
        if not isinstance(baseline, Mapping) or baseline.get("stationary_zero_reference_only") is not True:
            raise RuntimeError("caller must supply an environment with a verified stationary wrench zero")
        if getattr(env, "num_envs", 1) != 1:
            raise ValueError("run_bolt_episode supports exactly one environment")
        for method in ("step", "get_public_executor_state", "public_wrench", "get_private_bolt_contact_source"):
            if not callable(getattr(env, method, None)):
                raise TypeError(f"environment must expose {method}()")
        if not hasattr(env, "episode_length_buf") or not hasattr(env, "max_episode_length"):
            raise TypeError("environment must expose DirectRLEnv episode timeout state to prevent implicit resets")
        step_dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        if not math.isfinite(step_dt) or step_dt <= 0:
            raise ValueError("environment Factory control step duration must be positive")
        if normalized["tolerances"].max_sample_gap_s < step_dt:
            raise ValueError("calibrated max_sample_gap_s is shorter than one environment control step")
        held_sample_steps = math.ceil(normalized["tolerances"].held_seat_dwell_s / step_dt)
        release_sample_steps = math.ceil(normalized["tolerances"].stable_dwell_s / step_dt)
        if held_sample_steps * step_dt > MAX_HELD_SAMPLE_S + 1e-12:
            raise ValueError("held-seat dwell cannot be sampled within the 0.3s bounded window")
        if release_sample_steps * step_dt > MAX_RELEASE_SAMPLE_S + 1e-12:
            raise ValueError("release dwell cannot be sampled within the 1.0s bounded window")

        sensors = getattr(getattr(env, "scene", None), "sensors", None)
        camera = sensors.get("task_rgb_camera") if isinstance(sensors, Mapping) else None
        if camera is None or not hasattr(camera, "frame"):
            raise RuntimeError("env.scene.sensors['task_rgb_camera'] with frame and RGB output is required")
        contact_source = env.get_private_bolt_contact_source()
        adapter = BoltSeatSamplingAdapter(
            env, normalized["tolerances"], normalized["criteria"], contact_source=contact_source
        )
        plan = make_nominal_bolt_motion_plan(
            max_gripper_force_n=normalized["executor"]["max_gripper_force_n"],
            max_insertion_force_n=normalized["executor"]["max_insertion_force_n"],
            held_seat_dwell_s=normalized["tolerances"].held_seat_dwell_s,
        )
        executor_proxy = _ReleaseTrackingExecutor(
            BoltSkillExecutor(env, target_id=normalized["task_spec"]["target_id"], plan=plan)
        )
        recorder = EpisodeRGBVideoRecorder(video_path, fps=VIDEO_FPS)
        frame_dir = root / "frames"
        frame_dir.mkdir()

        def trace_event(event: dict[str, Any]) -> None:
            nonlocal model_requests, model_responses
            if event.get("event") == "request":
                model_requests += 1
            elif event.get("event") == "response":
                model_responses += 1
            _append_jsonl(public_stream, {
                "time_utc": datetime.now(timezone.utc).isoformat(), **event,
            })

        def sample_private_evaluator() -> None:
            nonlocal latest_status, latest_measurement, pickup_proven
            previous_count = len(adapter._trace)
            latest_status = adapter.observe()
            if len(adapter._trace) != previous_count + 1:
                raise RuntimeError("measurement adapter did not append exactly one private sample")
            sample = adapter._trace[-1]
            latest_measurement = sample.measurement
            pickup_proven = bool(getattr(sample, "pickup_proven", False))
            if type(latest_status.seat_ready) is not bool or type(latest_status.task_success) is not bool:
                raise TypeError("measurement adapter status must expose only boolean projections")
            _append_jsonl(private_stream, {
                "sample_index": len(adapter._trace) - 1,
                "camera_frame_id": latest_frame_id,
                "sample": sample,
            })

        def capture_camera_frame() -> bool:
            nonlocal latest_frame_path, latest_frame_id, latest_frame_index, frame_count
            data = camera.data
            outputs = getattr(data, "output", None)
            if not isinstance(outputs, Mapping) or "rgb" not in outputs:
                raise RuntimeError("task_rgb_camera.data.output['rgb'] is unavailable")
            frame_index = int(_scalar(camera.frame, "task_rgb_camera.frame"))
            if latest_frame_index is not None and frame_index == latest_frame_index:
                return False
            if latest_frame_index is not None and frame_index != latest_frame_index + 1:
                raise RuntimeError("task_rgb_camera frame counter skipped; video evidence is incomplete")
            if frame_index <= 0:
                raise RuntimeError("task_rgb_camera has not produced an initial rendered frame")
            # Isaac Lab 2.3 SensorBase records acquisition time when camera.data refreshes.
            acquisition_time = getattr(camera, "_timestamp_last_update", None)
            if acquisition_time is None:
                raise RuntimeError("task_rgb_camera does not expose its SensorBase acquisition timestamp")
            acquisition_time_s = _scalar(acquisition_time, "task_rgb_camera._timestamp_last_update")
            rgb = _rgb8(outputs["rgb"])
            frame_id = f"task_rgb_camera-{frame_index:08d}"
            frame_path = frame_dir / f"{frame_id}.png"
            _save_png(frame_path, rgb)
            recorder.append(rgb, acquisition_time_s)
            latest_frame_index = frame_index
            latest_frame_id = frame_id
            latest_frame_path = frame_path
            frame_count += 1
            trace_event({
                "event": "camera_frame", "camera": "task_rgb_camera", "frame_id": frame_id,
                "camera_frame": frame_index, "timestamp_s": acquisition_time_s,
                "path": str(frame_path.relative_to(root)),
            })
            return True

        def observation() -> Mapping[str, Any]:
            if latest_frame_path is None or latest_frame_id is None:
                raise RuntimeError("no real task_rgb_camera frame is available for public observation")
            state = env.get_public_executor_state()
            if not isinstance(state, Mapping):
                raise TypeError("get_public_executor_state() must return a mapping")
            return {
                "observation_id": latest_frame_id,
                "timestamp": _sim_time(env),
                "frames": {"task_rgb_camera": str(latest_frame_path)},
                "tcp_pose": _vector(state["tcp_pose"], "tcp_pose", 7),
                "tcp_velocity": _vector(state["tcp_velocity"], "tcp_velocity", 6),
                "joint_positions": _vector(state["joint_positions"], "joint_positions"),
                "joint_velocities": _vector(state["joint_velocities"], "joint_velocities"),
                "wrench": _vector(env.public_wrench(), "public_wrench", 6),
                "gripper_width": _scalar(state["gripper_width_m"], "gripper_width_m"),
                "finger_bolt_contacts": list(state["finger_bolt_contacts"]),
            }

        def persist_help_request(request: Mapping[str, Any]) -> None:
            nonlocal help_outbox_written
            with (root / "help_request.json").open("x", encoding="utf-8") as request_stream:
                json.dump(request, request_stream, indent=2, ensure_ascii=True, allow_nan=False)
                request_stream.write("\n")
            help_outbox_written = True

        def ensure_not_at_timeout() -> None:
            current = _scalar(env.episode_length_buf, "episode_length_buf")
            limit = int(env.max_episode_length)
            if limit <= 1 or current >= limit - 2:
                raise RuntimeError("episode timeout is imminent; refusing a step that would implicitly reset env")

        original_step = env.step
        had_instance_step = "step" in getattr(env, "__dict__", {})
        previous_instance_step = getattr(env, "__dict__", {}).get("step")

        def observed_step(action: Any, *args: Any, **kwargs: Any) -> Any:
            ensure_not_at_timeout()
            result = original_step(action, *args, **kwargs)
            capture_camera_frame()
            sample_private_evaluator()
            if isinstance(result, tuple) and len(result) >= 5:
                if _single_flag(result[2], "terminated") or _single_flag(result[3], "truncated"):
                    raise RuntimeError("environment ended during the episode; no implicit reset is accepted")
            return result

        env.step = observed_step

        capture_camera_frame()
        sample_private_evaluator()
        initial_unheld = not bool(latest_measurement.physically_held)
        if not initial_unheld:
            raise RuntimeError("mandatory initial evaluator sample found the bolt already held")

        if agent_mode == "vlm":
            agent = BoltHarnessAgent(
                endpoint=normalized["model"]["endpoint"],
                model=normalized["model"]["model"],
                redact_images=True,
            )
        else:
            agent = _ScriptedDevelopmentAgent(normalized["task_spec"]["target_id"])

        last_insertion_sampled = 0
        release_sampled = False

        def check_task() -> Mapping[str, bool]:
            nonlocal last_insertion_sampled, release_sampled
            if executor_proxy.release_completed:
                if not latest_status.task_success and not release_sampled:
                    release_sampled = True
                    for _ in range(release_sample_steps):
                        if latest_status.task_success:
                            break
                        env.step(_zero_action(env))
                return {"task_success": bool(latest_status.task_success)}

            held = bool(latest_measurement.physically_held)
            if (
                held and not latest_status.seat_ready
                and executor_proxy.completed_insertions > last_insertion_sampled
            ):
                last_insertion_sampled = executor_proxy.completed_insertions
                for _ in range(held_sample_steps):
                    if latest_status.seat_ready or not latest_measurement.physically_held:
                        break
                    env.step(_zero_action(env))
                held = bool(latest_measurement.physically_held)
            return {"held": held, "seat_ready": bool(latest_status.seat_ready)}

        harness_result = BoltHarnessLoop(
            agent,
            executor_proxy,
            task_spec=normalized["task_spec"],
            observation_callback=observation,
            check_task_callback=check_task,
            trace_callback=trace_event,
            help_request_callback=persist_help_request,
        ).run()
        if harness_result.private_exception is not None:
            failure = harness_result.private_exception
            failure_info = (type(failure), failure, failure.__traceback__)
        if not harness_result.task_success and harness_result.status != "help_requested":
            error = harness_result.error or "evaluator did not confirm post-release task success"
    except BaseException as exc:
        failure = exc
        failure_info = (type(exc), exc, exc.__traceback__)
        error = "episode setup or execution failed"
        if public_stream is not None:
            try:
                _append_jsonl(public_stream, {"event": "episode_failure", "error": error})
            except Exception:
                pass
    finally:
        if original_step is not None:
            try:
                if had_instance_step:
                    env.step = previous_instance_step
                else:
                    delattr(env, "step")
            except Exception as exc:
                failure = failure or exc
                error = error or "environment step hook could not be restored"
        if recorder is not None:
            try:
                video_error = recorder.close()
                video_finalized = (
                    video_error is None and recorder.frame_count > 0
                    and video_path.is_file() and video_path.stat().st_size > 0
                )
                if not video_finalized:
                    video_error = video_error or "finalized video file is missing or empty"
            except Exception as exc:
                video_error = f"{type(exc).__name__}: {exc}"
                video_finalized = False
        else:
            video_finalized = False
        if executor_proxy is not None:
            executor = getattr(executor_proxy, "_executor", None)
            diagnostics = getattr(executor, "private_diagnostics", None)
            insertion_diagnostics = (
                diagnostics.get("insertion") if isinstance(diagnostics, Mapping) else None
            )
            if insertion_diagnostics is not None:
                try:
                    diagnostic_path = private_dir / "executor_diagnostics.json"
                    with diagnostic_path.open("x", encoding="utf-8") as diagnostic_stream:
                        diagnostic_stream.write(
                            json.dumps(
                                _jsonable(insertion_diagnostics),
                                ensure_ascii=True,
                                allow_nan=False,
                            )
                            + "\n"
                        )
                except Exception as exc:
                    failure = failure or exc
                    error = error or "private executor diagnostics could not be written"
        for stream in (public_stream, private_stream):
            if stream is not None:
                try:
                    stream.close()
                except Exception as exc:
                    failure = failure or exc
                    error = error or "trace file could not be finalized"
        if failure_info is None and failure is not None:
            failure_info = (type(failure), failure, failure.__traceback__)
        if failure_info is not None:
            private_dir.mkdir(parents=True, exist_ok=True)
            try:
                (private_dir / "exception.txt").write_text(
                    "".join(traceback.format_exception(*failure_info)), encoding="utf-8"
                )
            except Exception:
                error = error or "private exception evidence could not be written"
        if video_error is not None:
            error = error or "video finalization failed"
        model_evidence = agent_mode == "vlm" and model_requests > 0 and model_responses > 0
        task_success = bool(harness_result is not None and harness_result.task_success)
        model_requirement_met = agent_mode != "vlm" or model_evidence
        success = task_success and model_requirement_met and video_finalized and failure is None
        status = "success" if success else (
            harness_result.status if harness_result is not None and not task_success
            and (harness_result.status != "help_requested" or help_outbox_written) else "failed"
        )
        exit_code = 0 if success else 1
        summary = {
            "task": normalized["task_spec"]["task_id"] if normalized else None,
            "agent_mode": agent_mode,
            "model": normalized["model"]["model"] if normalized and normalized["model"] else None,
            "model_request_count": model_requests,
            "model_response_count": model_responses,
            "model_evidence": model_evidence,
            "starts_ungrasped": initial_unheld,
            "physical_cap_grasp": pickup_proven,
            "actively_seated_before_release": bool(harness_result and harness_result.seat_ready),
            "task_success": task_success,
            "help_requested": help_outbox_written,
            "help_delivery": "local_outbox" if help_outbox_written else None,
            "calls": harness_result.calls if harness_result else 0,
            "status": status,
            "exit_code": exit_code,
            "video_file": video_path.name,
            "video_frame_count": frame_count,
            "video_finalized": video_finalized,
            "sim_time_s": _summary_sim_time(env),
            "wall_time_s": time.monotonic() - started,
            "public_trace": "public_trace.jsonl",
            "private_evaluator_trace": "private/evaluator_trace.jsonl",
            "config_snapshot": "config.yaml" if snapshot is not None else None,
            "error": error,
            "video_error": video_error,
        }
        try:
            with (root / "run_summary.json").open("x", encoding="utf-8") as summary_stream:
                summary_stream.write(json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
        except Exception as exc:
            error = error or f"run summary could not be written: {type(exc).__name__}"
            exit_code = 1
            status = "failed"

    return BoltEpisodeResult(
        status=status, exit_code=exit_code, task_success=task_success,
        video_finalized=video_finalized, model_evidence=model_evidence,
        run_dir=root, error=error, harness_result=harness_result,
    )


def _summary_sim_time(env: Any) -> float | None:
    try:
        return _sim_time(env)
    except Exception:
        return None


__all__ = ["BoltEpisodeResult", "VIDEO_FPS", "run_bolt_episode"]
