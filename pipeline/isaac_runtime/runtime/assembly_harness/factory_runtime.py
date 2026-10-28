"""Default Isaac Factory adapter for the generic runtime assembly harness."""

from __future__ import annotations

import math
import logging
import json
from collections import deque
from collections.abc import Mapping
from pathlib import Path
import traceback
from typing import Any

import numpy as np
from .depth_io import save_depth

from runtime.bolt_harness.executor import MAX_SKILL_DURATION_S

from .connector import GoalResolutionError
from .executor import (
    AssemblyExecutor,
    ExecutorLimits,
    FactoryAssemblyDriver,
    _project_world_point_to_pixel,
)
from .checkpoint import (
    CHECKPOINT_STATUSES,
    CHECKPOINT_VERSION,
    load_checkpoint,
    validate_checkpoint,
    write_checkpoint as write_private_checkpoint,
)
from .perception import (
    build_public_observation,
    estimate_camera_features,
    filter_robot_overlapping_contour_features,
    point_cloud_world,
)
from .robot_geometry import (
    DEFAULT_DEPTH_MATCH_TOLERANCE_M,
    MAX_DEPTH_MATCH_TOLERANCE_M,
    MAX_ROBOT_CAMERA_SKEW_S,
    RobotLinkMesh,
    filter_robot_surface_samples,
    invalidate_robot_depth_matches,
    link_transforms_from_body_state,
    render_robot_depth,
    visible_robot_meshes_from_usd,
)

_LOGGER = logging.getLogger(__name__)
_MODEL_FRAME_HISTORY_SIZE = 8
_CAMERA_FRAME_WAIT_STEP_MARGIN = 2


def _numpy(value: Any) -> np.ndarray:
    if callable(getattr(value, "detach", None)):
        value = value.detach()
    if callable(getattr(value, "cpu", None)):
        value = value.cpu()
    if callable(getattr(value, "numpy", None)):
        value = value.numpy()
    return np.asarray(value)


def _snapshot_signature(snapshot: Any) -> tuple[Any, ...] | None:
    if snapshot is None:
        return None
    if not isinstance(snapshot, Mapping):
        return (type(snapshot).__name__, id(snapshot))
    scalar_fields = (
        "frame_id", "camera_timestamp_s", "robot_timestamp_s", "sim_step_counter",
        "camera_render_step_counter", "robot_fk_step_counter", "render_event_id",
    )
    transforms = snapshot.get("link_transforms_world")
    transform_values = ()
    if isinstance(transforms, Mapping):
        transform_values = tuple(
            (str(name), tuple(np.asarray(value, dtype=np.float64).reshape(-1).tolist()))
            for name, value in transforms.items()
        )
    return (
        id(snapshot),
        *(snapshot.get(field) for field in scalar_fields),
        transform_values,
    )


def _depth_residual_summary(
    observed_depth_m: Any,
    predicted_robot_depth_m: Any,
    pixel_xy: tuple[int, int],
    *,
    window_radius_px: int = 12,
) -> dict[str, Any] | None:
    observed = np.asarray(observed_depth_m, dtype=np.float32)
    predicted = np.asarray(predicted_robot_depth_m, dtype=np.float32)
    if observed.ndim != 2 or predicted.shape != observed.shape:
        return None
    x, y = pixel_xy
    height, width = observed.shape
    if not (0 <= x < width and 0 <= y < height):
        return None
    x0, x1 = max(0, x - window_radius_px), min(width, x + window_radius_px + 1)
    y0, y1 = max(0, y - window_radius_px), min(height, y + window_radius_px + 1)

    def summarize(obs: np.ndarray, robot: np.ndarray) -> dict[str, Any]:
        observed_valid = np.isfinite(obs)
        robot_valid = np.isfinite(robot)
        overlap = observed_valid & robot_valid
        residual = np.abs(obs[overlap].astype(np.float64) - robot[overlap].astype(np.float64))
        near_surface = residual[residual <= MAX_DEPTH_MATCH_TOLERANCE_M]
        quantiles = np.quantile(near_surface, (0.5, 0.9, 0.95, 0.99)) if len(near_surface) else ()
        return {
            "observed_without_robot_depth_pixels": int(np.count_nonzero(observed_valid & ~robot_valid)),
            "valid_overlap_pixels": int(len(residual)),
            "over_5mm_overlap_pixels": int(np.count_nonzero(residual > MAX_DEPTH_MATCH_TOLERANCE_M)),
            "within_5mm_abs_residual_quantiles_m": [float(value) for value in quantiles],
            "match_pixels_by_tolerance_mm": {
                str(tolerance_mm): int(np.count_nonzero(residual <= tolerance_mm / 1000.0))
                for tolerance_mm in (1.0, 1.5, 2.0, 3.0, 5.0)
            },
        }

    collision_observed = float(observed[y, x])
    collision_predicted = float(predicted[y, x])
    collision_residual = (
        abs(collision_observed - collision_predicted)
        if math.isfinite(collision_observed) and math.isfinite(collision_predicted)
        else None
    )
    return {
        "collision_abs_residual_m": collision_residual,
        "window_radius_px": window_radius_px,
        "collision_neighborhood": summarize(observed[y0:y1, x0:x1], predicted[y0:y1, x0:x1]),
        "whole_frame": summarize(observed, predicted),
    }


def _immutable_point_cloud(points: Any) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("cached RGB-D obstacle cloud must be an N-by-3 array")
    backing = np.ascontiguousarray(values).tobytes()
    return np.frombuffer(backing, dtype=np.float64).reshape((-1, 3))


def _strict_dataclass(kind: type, value: Any, name: str):
    from dataclasses import fields

    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    expected = {field.name for field in fields(kind)}
    if set(value) != expected:
        raise ValueError(f"{name} fields differ from its evaluator contract")
    return kind(**dict(value))


def _apply_bench_task_scene_config(cfg: Any, task_config: Mapping[str, Any]) -> None:
    assets = task_config.get("assets", {})
    if not isinstance(assets, Mapping):
        raise TypeError("benchmark task assets must be a mapping")
    cover_asset = assets.get("cover")
    if cover_asset is not None:
        cover_path = Path(str(cover_asset)).expanduser()
        if not cover_path.is_absolute():
            cover_path = Path(__file__).resolve().parents[2] / cover_path
        if not cover_path.is_file():
            raise FileNotFoundError(f"benchmark cover asset does not exist: {cover_path}")
        cfg.cover_usd_path = str(cover_path.resolve())

    task_runtime = task_config.get("runtime", {})
    if not isinstance(task_runtime, Mapping):
        raise TypeError("benchmark task runtime must be a mapping")
    cover_pose = task_runtime.get("cover_pose")
    if cover_pose is not None:
        if not isinstance(cover_pose, Mapping) or set(cover_pose) != {
            "position_m", "quaternion_wxyz",
        }:
            raise ValueError("runtime.cover_pose requires position_m and quaternion_wxyz")
        position = tuple(float(value) for value in cover_pose["position_m"])
        quaternion = tuple(float(value) for value in cover_pose["quaternion_wxyz"])
        if len(position) != 3 or len(quaternion) != 4 or not np.isfinite((*position, *quaternion)).all():
            raise ValueError("runtime.cover_pose must contain finite 3D position and WXYZ quaternion")
        quaternion_norm = float(np.linalg.norm(quaternion))
        if quaternion_norm <= 1e-12:
            raise ValueError("runtime.cover_pose quaternion must be non-zero")
        cfg.cover_pos_m = position
        cfg.cover_quat_wxyz = tuple(value / quaternion_norm for value in quaternion)


def _evaluator_from_settings(
    env: Any, settings: Mapping[str, Any], task_config: Mapping[str, Any] | None = None,
):
    from runtime.bolt_harness.evaluator import BoltSeatTolerances
    from runtime.bolt_harness.measurement import (
        BoltSeatGeometryConfig,
        BoltSeatSamplingAdapter,
        BoltSeatSamplingCriteria,
        CapSupportContactCriteria,
    )

    tolerances = _strict_dataclass(BoltSeatTolerances, settings.get("tolerances"), "tolerances")
    criteria_source = settings.get("sampling_criteria")
    if not isinstance(criteria_source, Mapping):
        raise TypeError("sampling_criteria must be a mapping")
    criteria_values = dict(criteria_source)
    criteria_values["cap_support"] = _strict_dataclass(
        CapSupportContactCriteria, criteria_values.get("cap_support"), "sampling_criteria.cap_support"
    )
    criteria = _strict_dataclass(BoltSeatSamplingCriteria, criteria_values, "sampling_criteria")
    source = env.get_private_bolt_contact_source()
    task_config = task_config or {}
    target = task_config.get("private_target_mapping", {})
    if not isinstance(target, Mapping):
        raise TypeError("private_target_mapping must be a mapping")
    evaluator_name = target.get(
        "evaluator", "runtime.bolt_harness.measurement.BoltSeatSamplingAdapter"
    )
    if evaluator_name not in {
        "bolt_seat",
        "runtime.bolt_harness.measurement.BoltSeatSamplingAdapter",
    }:
        raise ValueError(f"unsupported benchmark evaluator {evaluator_name!r}")
    geometry_source = target.get("geometry", {})
    if not isinstance(geometry_source, Mapping):
        raise TypeError("private_target_mapping.geometry must be a mapping")
    geometry = BoltSeatGeometryConfig(**dict(geometry_source))
    return BoltSeatSamplingAdapter(
        env, tolerances, criteria, contact_source=source, geometry_config=geometry,
    )


def derive_factory_robot_calibration(
    env: Any,
) -> tuple[
    dict[str, float],
    dict[str, str],
    dict[str, tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]],
]:
    """Measure conservative per-collider link bounds and the tree from composed USD."""
    import omni.usd

    return _derive_factory_robot_calibration_from_stage(
        env,
        omni.usd.get_context().get_stage(),
        robot_root_path="/World/envs/env_0/Robot",
    )


def _derive_factory_robot_calibration_from_stage(
    env: Any,
    stage: Any,
    *,
    robot_root_path: str,
) -> tuple[
    dict[str, float],
    dict[str, str],
    dict[str, tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]],
]:
    """Extract calibration from an existing stage, including USD instance proxies."""
    from pxr import Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(robot_root_path)
    if not root or not root.IsValid():
        raise RuntimeError("composed Factory robot USD root is unavailable")
    robot_prims = tuple(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0.0:
        raise RuntimeError("composed Factory robot USD has invalid metersPerUnit metadata")

    body_names = tuple(str(name) for name in env._robot.body_names)
    bodies: dict[str, Any] = {}
    for prim in robot_prims:
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName() in body_names:
            if prim.GetName() in bodies:
                raise RuntimeError(f"robot USD contains duplicate rigid-body name {prim.GetName()!r}")
            bodies[prim.GetName()] = prim
    if set(bodies) != set(body_names):
        raise RuntimeError("could not map every Factory articulation body to its composed USD prim")

    bounds_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    radii: dict[str, float] = {}
    collision_bounds_m: dict[
        str, tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]
    ] = {}
    body_paths = {str(prim.GetPath()): name for name, prim in bodies.items()}
    colliders_by_body: dict[str, list[Any]] = {name: [] for name in body_names}
    for collider in robot_prims:
        if not collider.HasAPI(UsdPhysics.CollisionAPI):
            continue
        owner = collider.GetParent()
        while owner and owner.IsValid() and str(owner.GetPath()) not in body_paths:
            owner = owner.GetParent()
        owner_name = body_paths.get(str(owner.GetPath())) if owner and owner.IsValid() else None
        if owner_name is not None:
            colliders_by_body[owner_name].append(collider)
    for body_name, body_prim in bodies.items():
        radius = 0.0
        body_bounds: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
        for collider in colliders_by_body[body_name]:
            local_bounds = (
                bounds_cache.ComputeRelativeBound(collider, body_prim)
                .ComputeAlignedRange()
            )
            local_minimum = meters_per_unit * np.asarray(local_bounds.GetMin(), dtype=np.float64)
            local_maximum = meters_per_unit * np.asarray(local_bounds.GetMax(), dtype=np.float64)
            if (
                not np.isfinite(local_minimum).all()
                or not np.isfinite(local_maximum).all()
                or not np.all(local_minimum < local_maximum)
            ):
                raise RuntimeError(
                    f"robot USD collider {collider.GetPath()} has invalid body-local bounds"
                )
            body_bounds.append((tuple(local_minimum), tuple(local_maximum)))
            for mask in range(8):
                local_corner = np.asarray([
                    local_maximum[axis] if mask & (1 << axis) else local_minimum[axis]
                    for axis in range(3)
                ])
                radius = max(radius, float(np.linalg.norm(local_corner)))
        if not body_bounds or not math.isfinite(radius) or radius <= 0.0:
            raise RuntimeError(f"robot USD has no measurable collision bound for body {body_name!r}")
        radii[body_name] = radius
        collision_bounds_m[body_name] = tuple(body_bounds)

    directed_edges: list[tuple[str, str]] = []
    for prim in robot_prims:
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        body0 = [body_paths.get(str(path)) for path in joint.GetBody0Rel().GetTargets()]
        body1 = [body_paths.get(str(path)) for path in joint.GetBody1Rel().GetTargets()]
        parents = [name for name in body0 if name is not None]
        children = [name for name in body1 if name is not None]
        if len(parents) == 1 and len(children) == 1:
            directed_edges.append((parents[0], children[0]))

    roots = set(body_names) - {child for _, child in directed_edges}
    if len(roots) != 1 or len(directed_edges) != len(body_names) - 1:
        raise RuntimeError("robot USD joints do not form one complete rooted articulation tree")
    root_name = next(iter(roots))
    parents_by_child: dict[str, str] = {}
    for parent, child in directed_edges:
        if child in parents_by_child or child == root_name:
            raise RuntimeError("robot USD articulation has an ambiguous parent link")
        parents_by_child[child] = parent
    visited = {root_name}
    while True:
        newly_visited = {
            child for child, parent in parents_by_child.items()
            if parent in visited and child not in visited
        }
        if not newly_visited:
            break
        visited.update(newly_visited)
    if visited != set(body_names):
        raise RuntimeError("robot USD articulation tree is disconnected")
    return radii, parents_by_child, collision_bounds_m


def _copy_native_sample(value: Any) -> Any:
    clone = getattr(value, "clone", None)
    if callable(clone):
        return clone()
    return np.array(_numpy(value), copy=True)


def _current_sim_step_counter(env: Any) -> int:
    raw = getattr(env, "_sim_step_counter", None)
    if raw is None:
        raise RuntimeError("Factory env does not expose its native simulation-step counter")
    values = _numpy(raw).astype(np.float64, copy=False).reshape(-1)
    if values.size != 1 or not np.isfinite(values).all() or values[0] != math.floor(values[0]):
        raise RuntimeError("Factory native simulation-step counter must be one finite integer")
    return int(values[0])


def _append_robot_fk_history_sample(history: Any, env: Any) -> None:
    robot = env._robot
    data = robot.data
    history.append(
        {
            "sim_step_counter": _current_sim_step_counter(env),
            "robot_timestamp": _copy_native_sample(data._sim_timestamp),
            "body_link_pos_w": _copy_native_sample(data.body_link_pos_w),
            "body_link_quat_w": _copy_native_sample(data.body_link_quat_w),
        }
    )


def _install_robot_fk_history_hook(env: Any, history: Any):
    """Retain native FK at every physics-scene update for camera-time lookup."""
    original_update = env.scene.update

    def update_and_record(*args: Any, **kwargs: Any):
        result = original_update(*args, **kwargs)
        _append_robot_fk_history_sample(history, env)
        return result

    env.scene.update = update_and_record
    _append_robot_fk_history_sample(history, env)
    return original_update


class FactoryAssemblyRuntime:
    """Public RGB-D/runtime bridge composed around the existing Factory env."""

    def __init__(
        self,
        env: Any,
        app: Any,
        *,
        output_dir: str | Path,
        observe_only: bool,
        robot_link_radii_m: Mapping[str, float],
        robot_link_parents: Mapping[str, str],
        robot_link_collision_bounds_m: Mapping[
            str, tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]
        ],
        robot_link_meshes: tuple[RobotLinkMesh, ...],
        robot_depth_range_m: tuple[float, float],
        camera_period_s: float,
        limits: ExecutorLimits,
        robot_self_filter_unavailable_reason: str | None = None,
        evaluator: Any = None,
        robot_fk_history: Any = None,
        original_scene_update: Any = None,
        resume_requires_fresh_sample: bool = False,
        sweep_margin_m: float = 0.005,
        ik_damping: float = 0.02,
    ) -> None:
        self.env = env
        self.app = app
        self.limits = limits
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.frame_dir = self.output_dir / "runtime_frames"
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.observe_only = bool(observe_only)
        self.robot_link_radii_m = dict(robot_link_radii_m)
        self.robot_link_parents = dict(robot_link_parents)
        self.robot_link_collision_bounds_m = dict(robot_link_collision_bounds_m)
        self.robot_body_names = tuple(str(name) for name in env._robot.body_names)
        self.robot_link_meshes = tuple(robot_link_meshes)
        if {mesh.link_name for mesh in self.robot_link_meshes} != set(self.robot_body_names):
            if not self.observe_only:
                raise RuntimeError("visible robot meshes must cover every Factory articulation body")
            self.robot_link_meshes = ()
            robot_self_filter_unavailable_reason = (
                robot_self_filter_unavailable_reason
                or "visible robot meshes do not cover every Factory articulation body"
            )
        self.robot_depth_near_m, self.robot_depth_far_m = map(float, robot_depth_range_m)
        self.camera_period_s = float(camera_period_s)
        if not math.isfinite(self.camera_period_s) or self.camera_period_s <= 0.0:
            raise ValueError("camera_period_s must be positive and finite")
        self.robot_depth_match_tolerance_m = DEFAULT_DEPTH_MATCH_TOLERANCE_M
        self._robot_self_filter_unavailable_reason = None
        self._report_robot_filter_unavailable(robot_self_filter_unavailable_reason)
        self._evaluator = evaluator
        self._last_private_status = None
        self._release_authorized = False
        self._resume_requires_fresh_sample = bool(resume_requires_fresh_sample)
        self._listeners: list[Any] = []
        self._last_frame = None
        self._camera_frame_history: deque[dict[str, Any]] = deque(maxlen=_MODEL_FRAME_HISTORY_SIZE)
        self._seen_camera_frame_ids: set[int] = set()
        self._camera_render_event_count = 0
        self._last_model_camera_timestamp_s: float | None = None
        self._robot_model_pose_snapshot = None
        self._robot_fk_history = robot_fk_history if robot_fk_history is not None else deque()
        self._original_scene_update = original_scene_update
        self._last_rgb_path: Path | None = None
        self._last_depth_path: Path | None = None
        self._private_obstacle_context = None
        self._camera_features_cache = None
        self._obstacle_map_cache = None
        self._private_collision_sequence = 0
        self._observation_count = 0
        self._pending_held_surface_track = None
        self._checkpoint_disabled_after_release = False
        self._original_step = None
        if not self.observe_only:
            if evaluator is None:
                raise RuntimeError("action mode requires the independent private evaluator")
            self._install_evaluator_step_hook()
            if not self._resume_requires_fresh_sample:
                self._sample_private_evaluator()
        else:
            self._install_observe_only_step_hook()

        driver = FactoryAssemblyDriver(
            env,
            observe_public=self.observe_public,
            step_observation=lambda: self.observe_public(include_feature_overlay=False),
            robot_link_radii_m=self.robot_link_radii_m,
            robot_link_parents=self.robot_link_parents,
            robot_link_collision_bounds_m=self.robot_link_collision_bounds_m,
            ik_damping=ik_damping,
            sweep_margin_m=sweep_margin_m,
            private_collision_recorder=self._write_private_collision_diagnostic,
        )
        self.driver = driver
        if self.observe_only:
            self.executor = None
        else:
            self.executor = AssemblyExecutor(
                driver,
                limits=limits,
                observation_provider=self.observe_public,
                private_progress_recorder=self._write_private_execution_progress,
            )

    @classmethod
    def from_args(cls, args: Any) -> "FactoryAssemblyRuntime":
        app = None
        env = None
        robot_fk_history = None
        original_scene_update = None
        try:
            import copy
            import yaml
            from isaaclab.app import AppLauncher

            settings_path = Path(args.env_config).expanduser()
            if not settings_path.is_absolute():
                settings_path = Path(__file__).resolve().parents[2] / settings_path
            settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
            if not isinstance(settings, Mapping):
                raise ValueError("environment calibration file must contain a mapping")

            bench_task = getattr(args, "bench_task_config", {}) or {}
            if not isinstance(bench_task, Mapping):
                raise TypeError("benchmark task config must be a mapping")

            app = AppLauncher(args).app
            import omni.usd

            # Isaac extensions such as isaacsim.core are registered by AppLauncher.
            from runtime.bolt_harness.env import BoltInsertionEnv
            from runtime.bolt_harness.env_cfg import make_env_cfg

            import torch

            cfg = make_env_cfg(device=args.device, condition="S0")
            _apply_bench_task_scene_config(cfg, bench_task)
            resume_state = None
            resume_state_path = getattr(args, "resume_state", None)
            if resume_state_path is not None:
                resume_state = load_checkpoint(resume_state_path)
            bench_instance = getattr(args, "bench_instance", None)
            if bench_instance is not None:
                if not isinstance(bench_instance, Mapping):
                    raise TypeError("benchmark instance must be a mapping")
                offset = np.asarray(bench_instance.get("bolt_reset_offset_m", (0.0, 0.0, 0.0)), dtype=np.float64)
                if (
                    offset.shape != (3,)
                    or not np.isfinite(offset).all()
                    or np.max(np.abs(offset[:2])) > 0.0100001
                    or abs(float(offset[2])) > 1e-9
                ):
                    raise ValueError("benchmark bolt offsets must be finite, within 10 mm XY, and zero Z")
                cfg.bolt_reset_pos_m = tuple(
                    float(base + delta) for base, delta in zip(cfg.bolt_reset_pos_m, offset)
                )
                cfg.bolt.init_state.pos = cfg.bolt_reset_pos_m
            allowed_env_settings = {
                "bolt_mass_kg", "wrench_warmup_steps", "wrench_stationary_sample_count",
                "task_translation_prop_gain", "task_rot_deriv_scale", "episode_length_s",
                "tcp_tracking_stop_error_m", "tcp_tracking_stop_orientation_rad",
                "tcp_linear_speed_stop_mps", "tcp_angular_speed_stop_radps",
            }
            env_settings = {key: settings[key] for key in allowed_env_settings if key in settings}
            if "bolt_mass_kg" in env_settings:
                cfg.bolt_mass_kg = float(env_settings["bolt_mass_kg"])
                cfg.bolt.spawn.mass_props.mass = cfg.bolt_mass_kg
            if "task_translation_prop_gain" in env_settings:
                cfg.task_translation_prop_gain = float(env_settings["task_translation_prop_gain"])
                cfg.ctrl.default_task_prop_gains[:3] = [cfg.task_translation_prop_gain] * 3
            if "task_rot_deriv_scale" in env_settings:
                cfg.task_rot_deriv_scale = float(env_settings["task_rot_deriv_scale"])
            for key in (
                "wrench_warmup_steps", "wrench_stationary_sample_count", "episode_length_s",
                "tcp_tracking_stop_error_m", "tcp_tracking_stop_orientation_rad",
                "tcp_linear_speed_stop_mps", "tcp_angular_speed_stop_radps",
            ):
                if key in env_settings:
                    setattr(cfg, key, env_settings[key])
            pose_tolerances = settings.get("pose_window_tolerances")
            cfg.pose_window_tolerances = copy.deepcopy(pose_tolerances) if pose_tolerances else None

            env = BoltInsertionEnv(cfg)
            env.reset(seed=getattr(args, "physics_seed", args.seed))
            physics_dt = float(cfg.sim.dt)
            camera_period = float(cfg.task_rgb_camera.update_period)
            if not args.observe_only:
                history_capacity = max(2, int(math.ceil(camera_period / physics_dt)) + 2)
                robot_fk_history = deque(maxlen=history_capacity)
                original_scene_update = _install_robot_fk_history_hook(env, robot_fk_history)
            # Calibrate the wrench zero in the reset, no-contact state. Restoring a
            # grasp first would make the existing no-contact safety check fail.
            _calibrate_stationary_wrench(env, torch)
            if resume_state is not None:
                _restore_factory_checkpoint(env, resume_state)
                if robot_fk_history is not None:
                    robot_fk_history.clear()
                    _append_robot_fk_history_sample(robot_fk_history, env)
            camera = env.scene.sensors.get("task_rgb_camera")
            if camera is None:
                raise RuntimeError("Factory env has no task RGB-D camera")
            if args.observe_only:
                radii, parents, collision_bounds = derive_factory_robot_calibration(env)
                robot_meshes = ()
                filter_unavailable_reason = "robot self-filter intentionally disabled in observe-only mode"
            else:
                radii, parents, collision_bounds = derive_factory_robot_calibration(env)
                robot_meshes = visible_robot_meshes_from_usd(
                    omni.usd.get_context().get_stage(),
                    robot_root_path="/World/envs/env_0/Robot",
                    body_names=env._robot.body_names,
                )
                filter_unavailable_reason = None
            evaluator = None if args.observe_only else _evaluator_from_settings(env, settings, bench_task)
            if resume_state is not None and evaluator is not None:
                evaluator.restore_checkpoint_progression(resume_state["semantic_progression"])
            control_dt = float(cfg.sim.dt) * int(cfg.decimation)
            executor_settings = settings.get("executor", {})
            max_contact_force = float(executor_settings.get("max_insertion_force_n", 3.0))
            max_gripper_force = float(executor_settings.get("max_gripper_force_n", 12.0))
            limits = ExecutorLimits(
                control_dt_s=control_dt,
                position_action_scale_m=(0.01, 0.01, 0.01),
                rotation_action_scale_rad=(0.1, 0.1, 0.1),
                max_contact_force_n=max_contact_force,
                max_gripper_force_n=max_gripper_force,
                max_goal_uncertainty_m=0.01,
                max_goal_axis_uncertainty_rad=0.15,
                max_rgbd_age_s=0.2,
                max_steps=min(2000, max(1, int(MAX_SKILL_DURATION_S / control_dt))),
            )
            return cls(
                env,
                app,
                output_dir=args.output_dir,
                observe_only=args.observe_only,
                robot_link_radii_m=radii,
                robot_link_parents=parents,
                robot_link_collision_bounds_m=collision_bounds,
                robot_link_meshes=robot_meshes,
                robot_depth_range_m=tuple(cfg.task_rgb_camera.spawn.clipping_range),
                camera_period_s=camera_period,
                limits=limits,
                robot_self_filter_unavailable_reason=filter_unavailable_reason,
                evaluator=evaluator,
                robot_fk_history=robot_fk_history,
                original_scene_update=original_scene_update,
                resume_requires_fresh_sample=resume_state is not None,
            )
        except BaseException as startup_error:
            _LOGGER.exception("Factory runtime startup failed")
            try:
                private_dir = Path(args.output_dir).expanduser().resolve() / "private"
                private_dir.mkdir(parents=True, exist_ok=True)
                (private_dir / "exception.txt").write_text(
                    "".join(traceback.format_exception(
                        type(startup_error), startup_error, startup_error.__traceback__,
                    )),
                    encoding="utf-8",
                )
            except Exception:
                _LOGGER.exception("Could not persist the Factory runtime startup exception")
            if env is not None:
                if original_scene_update is not None:
                    try:
                        env.scene.update = original_scene_update
                    except BaseException:
                        _LOGGER.exception("Factory scene cleanup failed while preserving startup error")
                try:
                    env.close()
                except BaseException:
                    _LOGGER.exception("Factory environment cleanup failed while preserving startup error")
            if app is not None:
                try:
                    app.close()
                except BaseException:
                    _LOGGER.exception("Isaac app cleanup failed while preserving startup error")
            raise

    def _install_evaluator_step_hook(self) -> None:
        self._original_step = self.env.step

        def observed_step(*args: Any, **kwargs: Any):
            current = _numpy(self.env.episode_length_buf).reshape(-1)
            if current.size != 1 or int(current[0]) >= int(self.env.max_episode_length) - 2:
                raise RuntimeError("refusing a step that could trigger an implicit Factory reset")
            result = self._original_step(*args, **kwargs)
            if isinstance(result, tuple) and len(result) >= 5:
                terminated = bool(_numpy(result[2]).any())
                truncated = bool(_numpy(result[3]).any())
                if terminated or truncated:
                    raise RuntimeError("Factory episode ended; implicit reset is not accepted")
            self._sample_private_evaluator()
            return result

        self.env.step = observed_step

    def _install_observe_only_step_hook(self) -> None:
        self._original_step = self.env.step

        def guarded_observation_step(*args: Any, **kwargs: Any):
            current = _numpy(self.env.episode_length_buf).reshape(-1)
            if current.size != 1 or int(current[0]) >= int(self.env.max_episode_length) - 2:
                raise RuntimeError("refusing an observation hold near the Factory episode reset boundary")
            result = self._original_step(*args, **kwargs)
            if isinstance(result, tuple) and len(result) >= 5:
                terminated = bool(_numpy(result[2]).any())
                truncated = bool(_numpy(result[3]).any())
                if terminated or truncated:
                    raise RuntimeError("Factory episode ended during an observe-only camera hold")
            if getattr(self.env, "_episode_ended", None) is not False:
                raise RuntimeError("Factory episode ended during an observe-only camera hold")
            return result

        self.env.step = guarded_observation_step

    def _sample_private_evaluator(self):
        status = self._evaluator.observe()
        if type(status.seat_ready) is not bool or type(status.task_success) is not bool:
            raise TypeError("private evaluator must return boolean status projections")
        self._release_authorized = status.seat_ready
        self._last_private_status = status
        self._resume_requires_fresh_sample = False
        return status

    def _report_robot_filter_unavailable(self, reason: str | None) -> None:
        normalized = str(reason).strip() if reason is not None else ""
        self._robot_self_filter_unavailable_reason = normalized or None
        if normalized:
            _LOGGER.warning("Factory robot self-filter unavailable: %s", normalized)

    def _report_robot_filter_available(self) -> None:
        if getattr(self, "_robot_self_filter_unavailable_reason", None) is not None:
            _LOGGER.info("Factory robot self-filter synchronized with camera acquisition again")
        self._robot_self_filter_unavailable_reason = None

    def _capture_frame(self):
        camera = self.env.scene.sensors.get("task_rgb_camera")
        last_frame = self._last_frame
        if last_frame is not None:
            sim_timestamp = _numpy(self.env._robot.data._sim_timestamp).astype(
                np.float64, copy=False,
            ).reshape(-1)
            if sim_timestamp.size != 1 or not np.isfinite(sim_timestamp).all():
                raise RuntimeError("camera scheduling requires one finite native simulation timestamp")
            elapsed_s = float(sim_timestamp[0]) - float(last_frame.timestamp_s)
            if elapsed_s < -1e-9:
                raise RuntimeError("simulation time regressed behind the last camera acquisition")
            if elapsed_s + 1e-9 < self.camera_period_s:
                return last_frame

        sim = getattr(self.env, "sim", None)
        render = getattr(sim, "render", None)
        if not callable(render):
            raise RuntimeError("Factory env simulation cannot render a synchronized RGB-D acquisition")
        # Bind the lazy annotator read to a render at this exact native physics step.
        render_step_counter = _current_sim_step_counter(self.env)
        render()
        if _current_sim_step_counter(self.env) != render_step_counter:
            raise RuntimeError("native physics advanced during the synchronized camera render")
        self._camera_render_event_count += 1
        render_event_id = self._camera_render_event_count
        camera.update(0.0, force_recompute=True)
        frame = self.env.get_public_rgbd_frame()
        if self._last_frame is not None:
            if frame.timestamp_s == self._last_frame.timestamp_s:
                return self._last_frame
            if frame.timestamp_s < self._last_frame.timestamp_s:
                raise RuntimeError("camera acquisition timestamps must increase strictly")
            if frame.frame_id == self._last_frame.frame_id:
                raise RuntimeError("one camera frame ID cannot refer to multiple acquisition timestamps")
        if self._last_frame is None or frame.frame_id != self._last_frame.frame_id:
            self._validate_camera_frame_order(frame)
            if self.observe_only:
                self._robot_model_pose_snapshot = None
                if not self._robot_self_filter_unavailable_reason:
                    self._report_robot_filter_unavailable(
                        "robot self-filter intentionally disabled in observe-only mode"
                    )
            elif not self.robot_link_meshes:
                self._report_robot_filter_unavailable("visible robot mesh mapping is unavailable")
            else:
                try:
                    self._cache_robot_model_pose(
                        frame.frame_id,
                        frame.timestamp_s,
                        sim_step_counter=render_step_counter,
                        render_event_id=render_event_id,
                    )
                except Exception as exc:
                    self._robot_model_pose_snapshot = None
                    reason = f"camera/FK acquisition-time mismatch or invalid FK: {type(exc).__name__}: {exc}"
                    self._report_robot_filter_unavailable(reason)
                    raise
                else:
                    self._report_robot_filter_available()
            import cv2

            rgb_path = self.frame_dir / f"rgb_{frame.frame_id:08d}.png"
            depth_path = self.frame_dir / f"depth_{frame.frame_id:08d}.npy"
            if not cv2.imwrite(str(rgb_path), frame.rgb[:, :, ::-1]):
                raise OSError("failed to persist the live Factory RGB frame")
            depth_path = save_depth(depth_path, frame.depth_m)
            self._last_frame = frame
            self._last_rgb_path = rgb_path
            self._last_depth_path = depth_path
            self._camera_frame_history.append({
                "frame_id": frame.frame_id,
                "timestamp_s": frame.timestamp_s,
                "rgb_path": str(rgb_path),
                "depth_path": str(depth_path),
            })
            self._seen_camera_frame_ids.add(frame.frame_id)
            sim_timestamp = float(frame.timestamp_s)
            for listener in tuple(self._listeners):
                listener(frame.rgb, sim_timestamp)
        return frame

    def _validate_camera_frame_order(self, frame: Any) -> None:
        if frame.frame_id in self._seen_camera_frame_ids:
            raise RuntimeError(f"camera frame ID {frame.frame_id} was reused")
        if self._camera_frame_history:
            previous = self._camera_frame_history[-1]
            if frame.timestamp_s <= previous["timestamp_s"]:
                raise RuntimeError("camera acquisition timestamps must increase strictly")

    def _attach_camera_frame_history(self, observation: dict[str, Any]) -> dict[str, Any]:
        history = tuple(self._camera_frame_history)
        if not history:
            raise RuntimeError("no real camera acquisition is available for the public observation")

        frames: dict[str, str] = {}
        sequence: list[dict[str, Any]] = []
        for index, entry in enumerate(history[:-1]):
            frame_key = f"task_camera_history_{index:02d}"
            frames[frame_key] = entry["rgb_path"]
            sequence.append({
                "frame_key": frame_key,
                "frame_id": entry["frame_id"],
                "timestamp_s": entry["timestamp_s"],
            })

        current_entry = history[-1]
        current_image = observation.get("frames", {}).get("task_camera_feature_overlay")
        current_key = (
            "task_camera_current_feature_overlay"
            if isinstance(current_image, str)
            else "task_camera_current"
        )
        frames[current_key] = current_image if isinstance(current_image, str) else current_entry["rgb_path"]
        sequence.append({
            "frame_key": current_key,
            "frame_id": current_entry["frame_id"],
            "timestamp_s": current_entry["timestamp_s"],
        })
        observation["frames"] = frames
        observation["frame_sequence"] = sequence
        return observation

    def observe_for_model(self) -> Mapping[str, Any]:
        """Return an eight-acquisition model view, advancing only through guarded stationary holds."""
        pending_track = getattr(self, "_pending_held_surface_track", None)
        self._pending_held_surface_track = None
        if self.driver is None:
            raise RuntimeError("cannot bootstrap model frames without the protected Factory hold driver")
        if not self._has_calibrated_wrench_baseline():
            raise RuntimeError("camera hold requires the existing calibrated stationary wrench baseline")

        observation = self.observe_public(include_feature_overlay=True)
        latest = self._camera_frame_history[-1]
        target_timestamp = self._last_model_camera_timestamp_s
        max_steps_per_frame = max(
            1,
            int(math.ceil(self.camera_period_s / self.limits.control_dt_s))
            + _CAMERA_FRAME_WAIT_STEP_MARGIN,
        )
        hold_step_budget = min(
            self.limits.max_steps,
            max_steps_per_frame * _MODEL_FRAME_HISTORY_SIZE,
        )
        hold_steps = 0

        while (
            len(self._camera_frame_history) < _MODEL_FRAME_HISTORY_SIZE
            or (target_timestamp is not None and latest["timestamp_s"] <= target_timestamp)
        ):
            previous_timestamp = latest["timestamp_s"]
            acquired = False
            for _ in range(max_steps_per_frame):
                if hold_steps >= hold_step_budget:
                    raise RuntimeError("bounded stationary-hold budget exhausted before eight fresh camera frames")
                self._stationary_hold_step()
                hold_steps += 1
                latest = self._camera_frame_history[-1]
                if latest["timestamp_s"] > previous_timestamp:
                    acquired = True
                    break
            if not acquired:
                raise RuntimeError("camera did not produce a new acquisition during the bounded stationary hold")

        observation = self.observe_public(include_feature_overlay=True)
        latest = self._camera_frame_history[-1]
        if len(observation.get("frame_sequence", ())) != _MODEL_FRAME_HISTORY_SIZE:
            raise RuntimeError("model observation did not contain exactly eight camera acquisitions")
        if target_timestamp is not None and latest["timestamp_s"] <= target_timestamp:
            raise RuntimeError("model observation must include a camera acquisition newer than the previous request")
        self._last_model_camera_timestamp_s = latest["timestamp_s"]
        if pending_track is not None:
            goal, previous_observation, moving_tcp_feature_pose = pending_track
            observation = self.executor.connector.carry_held_surface_track(
                goal,
                previous_observation,
                observation,
                moving_tcp_feature_pose=moving_tcp_feature_pose,
                max_match_distance_m=self.limits.max_feature_match_distance_m,
            )
        return observation

    def _has_calibrated_wrench_baseline(self) -> bool:
        baseline = getattr(self.env, "wrench_baseline_stats", None)
        return (
            isinstance(baseline, Mapping)
            and baseline.get("stationary_zero_reference_only") is True
        )

    def _stationary_hold_step(self) -> None:
        before = self._camera_hold_state()
        self.driver.hold()
        after = self._camera_hold_state()
        if before["gripper_target_joint_m"] != after["gripper_target_joint_m"]:
            raise RuntimeError("camera hold changed the existing gripper target")
        if abs(after["gripper_width_m"] - before["gripper_width_m"]) > self.limits.gripper_stability_tolerance_m:
            raise RuntimeError("gripper width changed during stationary camera hold")
        for side, was_contacted, is_contacted in zip(
            ("left", "right"), before["finger_bolt_contacts"], after["finger_bolt_contacts"],
        ):
            if was_contacted and not is_contacted:
                forces = _public_finger_contact_forces_n(self.env)
                force_detail = ""
                if forces is not None:
                    force_detail = f" (measured finger-to-bolt force: {forces[0 if side == 'left' else 1]:.3f} N)"
                raise RuntimeError(
                    f"{side} finger contact was lost during stationary camera hold{force_detail}"
                )

    def _camera_hold_state(self) -> dict[str, Any]:
        env = self.env
        if getattr(env, "_episode_ended", None) is not False:
            raise RuntimeError("camera hold refused because the Factory episode is ended or its state is unavailable")
        episode_steps = _numpy(env.episode_length_buf).reshape(-1)
        if (
            episode_steps.size != 1
            or int(episode_steps[0]) >= int(env.max_episode_length) - 2
        ):
            raise RuntimeError("camera hold refused near the Factory episode reset boundary")
        if not self._has_calibrated_wrench_baseline():
            raise RuntimeError("camera hold requires the existing calibrated stationary wrench baseline")

        state = env.get_public_executor_state()
        if not isinstance(state, Mapping) or state.get("motion_safety_armed") is not True:
            raise RuntimeError("camera hold requires native Factory motion safety to be armed")
        cfg = env.cfg
        try:
            tracking_values = _numpy(state["tcp_tracking_error_m"]).astype(np.float64, copy=False).reshape(-1)
            orientation_values = _numpy(state["tcp_orientation_tracking_error_rad"]).astype(
                np.float64, copy=False,
            ).reshape(-1)
            velocity = _numpy(state["tcp_velocity"]).astype(np.float64, copy=False).reshape(-1)
            width_values = np.asarray(state["gripper_width_m"], dtype=np.float64).reshape(-1)
            target_values = _numpy(env.gripper_target_joint_m).astype(np.float64, copy=False).reshape(-1)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("camera hold requires finite TCP and gripper state") from exc
        if (
            tracking_values.size != 1
            or orientation_values.size != 1
            or velocity.size != 6
            or width_values.size != 1
            or target_values.size != 1
            or not np.isfinite(tracking_values).all()
            or not np.isfinite(orientation_values).all()
            or not np.isfinite(velocity).all()
            or not np.isfinite(width_values).all()
            or not np.isfinite(target_values).all()
        ):
            raise RuntimeError("camera hold requires finite TCP tracking, orientation, velocity, and gripper state")
        tracking = float(tracking_values[0])
        orientation_error = float(orientation_values[0])
        if (
            tracking > float(cfg.tcp_tracking_stop_error_m)
            or orientation_error > float(cfg.tcp_tracking_stop_orientation_rad)
            or np.linalg.norm(velocity[:3]) > float(cfg.tcp_linear_speed_stop_mps)
            or np.linalg.norm(velocity[3:]) > float(cfg.tcp_angular_speed_stop_radps)
        ):
            raise RuntimeError("camera hold refused because current TCP tracking or velocity exceeds native limits")
        wrench = _numpy(env.public_wrench()).reshape(-1)
        if wrench.size != 6 or not np.isfinite(wrench).all():
            raise RuntimeError("camera hold requires a finite public wrench with its stationary zero reference")
        force_n = float(np.linalg.norm(wrench[:3]))
        if force_n > self.limits.max_contact_force_n:
            raise RuntimeError("camera hold stopped because the public force limit was reached")
        width = float(width_values[0])
        target_joint_m = float(target_values[0])
        contacts = tuple(state["finger_bolt_contacts"])
        if len(contacts) != 2 or any(type(value) is not bool for value in contacts):
            raise RuntimeError("camera hold requires current bilateral gripper contact state")
        if any(contacts) and 2.0 * target_joint_m > width + self.limits.gripper_stability_tolerance_m:
            raise RuntimeError("camera hold refused because the gripper target would open against finger contact")
        return {
            "gripper_width_m": width,
            "gripper_target_joint_m": target_joint_m,
            "finger_bolt_contacts": contacts,
        }

    def _cache_robot_model_pose(
        self,
        frame_id: int,
        camera_timestamp_s: float,
        *,
        sim_step_counter: int,
        render_event_id: int,
    ) -> None:
        if (
            isinstance(sim_step_counter, bool)
            or not isinstance(sim_step_counter, (int, np.integer))
            or sim_step_counter < 0
        ):
            raise ValueError("camera render step counter must be a non-negative integer")
        if (
            isinstance(render_event_id, bool)
            or not isinstance(render_event_id, (int, np.integer))
            or render_event_id <= 0
        ):
            raise ValueError("camera render event ID must be a positive integer")
        cached = self._robot_model_pose_snapshot
        if cached is not None and cached["frame_id"] == frame_id:
            if cached.get("sim_step_counter") != sim_step_counter:
                raise RuntimeError("camera frame ID is already bound to a different native physics step")
            if cached.get("render_event_id") != render_event_id:
                raise RuntimeError("camera frame ID is already bound to a different render event")
            if abs(cached["camera_timestamp_s"] - camera_timestamp_s) > MAX_ROBOT_CAMERA_SKEW_S:
                raise RuntimeError("one camera frame ID cannot refer to multiple capture timestamps")
            return

        history = tuple(getattr(self, "_robot_fk_history", ()))
        matching_steps = [
            sample for sample in history
            if sample.get("sim_step_counter") == sim_step_counter
        ]
        if not matching_steps:
            raise RuntimeError(
                "no post-scene-update robot FK sample matches the native camera render step "
                f"{sim_step_counter}"
            )
        sample = matching_steps[-1]
        timestamp_values = _numpy(sample["robot_timestamp"]).astype(np.float64, copy=False).reshape(-1)
        if timestamp_values.size != 1 or not np.isfinite(timestamp_values).all():
            raise RuntimeError("robot FK timestamp for the native render step is invalid")
        robot_timestamp_s = float(timestamp_values[0])
        transforms = link_transforms_from_body_state(
            self.robot_body_names,
            sample["body_link_pos_w"],
            sample["body_link_quat_w"],
        )
        self._robot_model_pose_snapshot = {
            "frame_id": int(frame_id),
            "camera_timestamp_s": float(camera_timestamp_s),
            "robot_timestamp_s": robot_timestamp_s,
            "sim_step_counter": sample.get("sim_step_counter"),
            "camera_render_step_counter": sim_step_counter,
            "robot_fk_step_counter": sample.get("sim_step_counter"),
            "render_event_id": render_event_id,
            "link_transforms_world": transforms,
        }

    def observe_public(self, *, include_feature_overlay: bool = True) -> Mapping[str, Any]:
        frame = self._capture_frame()
        if self._last_rgb_path is None or self._last_depth_path is None:
            raise RuntimeError("Factory RGB-D frame was not persisted")
        observation_id = f"factory-observation-{self._observation_count:08d}"
        self._observation_count += 1
        public_state = dict(self.env.get_public_assembly_state())
        robot_timestamp_value = public_state.get("robot_timestamp_s", public_state.get("robot_state_timestamp_s"))
        if robot_timestamp_value is None:
            robot_timestamp_value = self.env._robot.data._sim_timestamp
        robot_timestamp_values = _numpy(robot_timestamp_value).astype(np.float64, copy=False).reshape(-1)
        if robot_timestamp_values.size != 1 or not np.isfinite(robot_timestamp_values).all():
            raise RuntimeError("public robot state must expose one finite native simulation timestamp")
        robot_timestamp_s = float(robot_timestamp_values[0])
        executor_state = self.env.get_public_executor_state()
        finger_contacts = [bool(value) for value in executor_state["finger_bolt_contacts"]]
        finger_contact_force_n = _public_finger_contact_forces_n(self.env)
        public_state.update({
            "tcp_pose": _finite_list(public_state["tcp_pose"], 7, "tcp_pose"),
            "commanded_tcp_pose": _finite_list(executor_state["commanded_tcp_pose"][0], 7, "commanded_tcp_pose"),
            "tcp_velocity": _finite_list(public_state["tcp_velocity"], 6, "tcp_velocity"),
            "joint_positions": _finite_list(executor_state["joint_positions"][0, :7], 7, "joint_positions"),
            "joint_velocities": _finite_list(executor_state["joint_velocities"][0, :7], 7, "joint_velocities"),
            "finger_contacts": finger_contacts,
            "finger_bolt_contacts": finger_contacts,
            "grasped": all(finger_contacts),
            "robot_timestamp_s": robot_timestamp_s,
            "camera_observation_id": observation_id,
            "rgbd_source_observation_id": observation_id,
            "rgbd_source_timestamp_s": float(frame.timestamp_s),
            "rgbd_obstacle_map": self._obstacle_map(frame, observation_id),
        })
        if finger_contact_force_n is not None:
            public_state["finger_contact_force_n"] = finger_contact_force_n
        if getattr(self.env, "wrench_baseline_stats", None) is not None:
            public_state["wrench"] = _finite_list(public_state["wrench"], 6, "wrench")
        else:
            public_state.pop("wrench", None)
        observation = build_public_observation(
            frame,
            observation_id=observation_id,
            rgb_path=str(self._last_rgb_path),
            depth_path=str(self._last_depth_path),
            robot_state=public_state,
            include_planes=False,
            camera_features=self._camera_features_for_frame(frame),
            include_feature_overlay=include_feature_overlay,
        )
        return self._attach_camera_frame_history(observation)

    def _camera_features_for_frame(self, frame: Any):
        frame_key = (int(frame.frame_id), float(frame.timestamp_s))
        snapshot = getattr(self, "_robot_model_pose_snapshot", None)
        snapshot_signature = _snapshot_signature(snapshot)
        key = (*frame_key, snapshot_signature)
        cache = getattr(self, "_camera_features_cache", None)
        if cache is not None and cache.get("frame_key") == frame_key:
            if cache.get("snapshot_signature") != snapshot_signature:
                raise RuntimeError("cached RGB-D features cannot be rebound to a changed camera/FK snapshot")
            if cache["key"] == key:
                return cache["features"]

        robot_cache = getattr(self, "_obstacle_map_cache", None)
        robot_depth = None
        if robot_cache is not None and robot_cache.get("key") == frame_key:
            if robot_cache.get("snapshot_signature") != snapshot_signature:
                raise RuntimeError("camera feature raster does not match the obstacle-map FK snapshot")
            robot_depth = robot_cache.get("robot_depth_m")
        if not getattr(self, "observe_only", True) and robot_depth is None:
            raise RuntimeError("action-mode features require the synchronized robot depth raster")

        feature_frame = frame
        robot_matched_pixels = None
        if robot_depth is not None:
            feature_depth = invalidate_robot_depth_matches(
                frame,
                robot_depth,
                depth_tolerance_m=self.robot_depth_match_tolerance_m,
            )
            robot_matched_pixels = np.isfinite(frame.depth_m) & ~np.isfinite(feature_depth)
            feature_frame = type(frame)(
                frame_id=frame.frame_id,
                timestamp_s=frame.timestamp_s,
                rgb=frame.rgb,
                depth_m=feature_depth,
                K=frame.K,
                T_world_camera=frame.T_world_camera,
                pixel_convention=getattr(frame, "pixel_convention", None),
            )
        features = estimate_camera_features(
            feature_frame,
            observation_id=f"camera-cache-frame-{frame.frame_id}",
            include_planes=False,
            include_contours=True,
        )
        if robot_matched_pixels is not None:
            features = filter_robot_overlapping_contour_features(features, robot_matched_pixels)
        self._camera_features_cache = {
            "key": key,
            "frame_key": frame_key,
            "snapshot_signature": snapshot_signature,
            "features": features,
        }
        return features

    def _obstacle_map(self, frame: Any, observation_id: str) -> dict[str, Any]:
        key = (int(frame.frame_id), float(frame.timestamp_s))
        snapshot = getattr(self, "_robot_model_pose_snapshot", None)
        cache = getattr(self, "_obstacle_map_cache", None)
        if cache is not None and cache["key"] == key:
            if _snapshot_signature(snapshot) != cache["snapshot_signature"]:
                self._private_obstacle_context = None
                raise RuntimeError(
                    "cached RGB-D map cannot be rebound to a changed camera/FK acquisition snapshot"
                )
            self._set_private_obstacle_context(cache, observation_id)
            return self._public_obstacle_map(cache, observation_id)

        self._private_obstacle_context = None
        stride = 6
        height, width = frame.depth_m.shape
        # The pixel grid already bounds this cloud; avoid a second global thinning
        # that can erase small RGB-D contact patches needed by the executor.
        max_points = ((height + stride - 1) // stride) * ((width + stride - 1) // stride)
        sampled = point_cloud_world(frame, stride=stride, max_points=max_points)
        points = np.asarray(sampled["points_world_m"], dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise RuntimeError("RGB-D obstacle map must contain finite world xyz points")
        if len(points) < 16:
            raise RuntimeError("RGB-D obstacle map has too few valid points")
        robot_link_meshes = getattr(self, "robot_link_meshes", ())
        self_filtered = False
        predicted_robot_depth = None
        if (
            not getattr(self, "observe_only", False)
            and robot_link_meshes
            and snapshot is not None
            and snapshot["frame_id"] == frame.frame_id
            and snapshot.get("camera_render_step_counter") is not None
            and snapshot.get("robot_fk_step_counter") is not None
            and snapshot.get("render_event_id") is not None
        ):
            try:
                predicted_robot_depth = render_robot_depth(
                    robot_link_meshes,
                    snapshot["link_transforms_world"],
                    frame,
                    body_names=self.robot_body_names,
                    near_m=self.robot_depth_near_m,
                    far_m=self.robot_depth_far_m,
                )
                result = filter_robot_surface_samples(
                    frame,
                    sampled,
                    robot_link_meshes,
                    snapshot["link_transforms_world"],
                    body_names=self.robot_body_names,
                    camera_source_timestamp_s=frame.timestamp_s,
                    robot_state_timestamp_s=snapshot["robot_timestamp_s"],
                    camera_render_step_counter=snapshot["camera_render_step_counter"],
                    robot_fk_step_counter=snapshot["robot_fk_step_counter"],
                    render_event_id=snapshot["render_event_id"],
                    depth_tolerance_m=self.robot_depth_match_tolerance_m,
                    near_m=self.robot_depth_near_m,
                    far_m=self.robot_depth_far_m,
                    predicted_robot_depth_m=predicted_robot_depth,
                )
            except Exception as exc:
                if not getattr(self, "observe_only", False):
                    raise
                self._robot_model_pose_snapshot = None
                self._report_robot_filter_unavailable(
                    f"robot mesh-depth filtering unavailable: {type(exc).__name__}: {exc}"
                )
            else:
                if result.robot_self_filtered:
                    points = result.points_world_m
                    self_filtered = True
                    predicted_robot_depth.setflags(write=False)
                    self._report_robot_filter_available()
                elif not getattr(self, "observe_only", False):
                    raise RuntimeError("robot mesh-depth filter did not validate this RGB-D frame")
                else:
                    self._report_robot_filter_unavailable("robot mesh-depth filter did not validate this RGB-D frame")
        elif (
            getattr(self, "observe_only", False)
            and not getattr(self, "_robot_self_filter_unavailable_reason", None)
        ):
            self._report_robot_filter_unavailable("camera-time-matched robot FK snapshot is unavailable")

        cache = {
            "key": key,
            "frame": frame,
            "snapshot": snapshot,
            "snapshot_signature": _snapshot_signature(snapshot),
            "points_m": _immutable_point_cloud(points),
            "robot_depth_m": predicted_robot_depth if self_filtered else None,
            "robot_self_filtered": self_filtered,
        }
        self._obstacle_map_cache = cache
        self._set_private_obstacle_context(cache, observation_id)
        return self._public_obstacle_map(cache, observation_id)

    def _set_private_obstacle_context(self, cache: Mapping[str, Any], observation_id: str) -> None:
        if cache["robot_self_filtered"]:
            self._private_obstacle_context = {
                "observation_id": observation_id,
                "frame": cache["frame"],
                "snapshot": cache["snapshot"],
                "robot_depth_m": cache.get("robot_depth_m"),
            }
        else:
            self._private_obstacle_context = None

    @staticmethod
    def _public_obstacle_map(cache: Mapping[str, Any], observation_id: str) -> dict[str, Any]:
        return {
            "source_observation_id": observation_id,
            "frame": "world",
            "valid": len(cache["points_m"]) >= 16,
            "robot_self_filtered": cache["robot_self_filtered"],
            "timestamp_s": float(cache["frame"].timestamp_s),
            "points_m": cache["points_m"],
        }

    def _write_private_collision_diagnostic(self, sweep: Mapping[str, Any]) -> None:
        """Persist synchronized collision evidence outside model/public observations."""
        context = self._private_obstacle_context
        frame = context.get("frame") if isinstance(context, Mapping) else None
        snapshot = context.get("snapshot") if isinstance(context, Mapping) else None
        point = np.asarray(sweep.get("obstacle_point_world_m"), dtype=np.float64).reshape(-1)
        if point.shape != (3,) or not np.isfinite(point).all():
            point = np.full(3, np.nan)

        pixel = None
        predicted_depth_m = None
        observed_depth_m = None
        robot_depth = context.get("robot_depth_m") if isinstance(context, Mapping) else None
        if frame is not None and snapshot is not None and np.isfinite(point).all():
            projected = _project_world_point_to_pixel(
                point.tolist(), frame.K, frame.T_world_camera,
            )
            if projected is not None:
                pixel_xy = [int(math.floor(value + 0.5)) for value in projected]
                height, width = frame.depth_m.shape
                pixel = {"xy": pixel_xy, "continuous_xy": projected}
                if 0 <= pixel_xy[0] < width and 0 <= pixel_xy[1] < height:
                    observed = float(frame.depth_m[pixel_xy[1], pixel_xy[0]])
                    observed_depth_m = observed if math.isfinite(observed) else None
                    predicted = (
                        float(robot_depth[pixel_xy[1], pixel_xy[0]])
                        if isinstance(robot_depth, np.ndarray)
                        else float("nan")
                    )
                    predicted_depth_m = predicted if math.isfinite(predicted) else None

        self._private_collision_sequence += 1
        private_dir = self.output_dir / "private"
        private_dir.mkdir(parents=True, exist_ok=True)
        frame_id = int(frame.frame_id) if frame is not None else None
        camera_timestamp_s = float(frame.timestamp_s) if frame is not None else None
        transforms = (
            {
                name: np.asarray(matrix, dtype=np.float64).tolist()
                for name, matrix in snapshot["link_transforms_world"].items()
            }
            if snapshot is not None else None
        )
        collision_record = getattr(self.driver, "last_sweep_collision_diagnostic", None)
        payload = {
            "observation_id": context.get("observation_id") if isinstance(context, Mapping) else None,
            "camera_fk_sync": {
                "fresh_sim_render_before_camera_update": True,
                "camera_frame_id": frame_id,
                "render_event_id": (
                    int(snapshot["render_event_id"]) if snapshot is not None else None
                ),
                "camera_render_step_counter": (
                    int(snapshot["camera_render_step_counter"]) if snapshot is not None else None
                ),
                "robot_fk_step_counter": (
                    int(snapshot["robot_fk_step_counter"]) if snapshot is not None else None
                ),
                "camera_timestamp_s": camera_timestamp_s,
                "fk_robot_timestamp_s": (
                    float(snapshot["robot_timestamp_s"]) if snapshot is not None else None
                ),
                "clock_delta_s": (
                    float(snapshot["robot_timestamp_s"] - camera_timestamp_s)
                    if snapshot is not None else None
                ),
            },
            "camera_frame": {
                "frame_id": frame_id,
                "timestamp_s": camera_timestamp_s,
                "K": frame.K.tolist() if frame is not None else None,
                "T_world_camera": frame.T_world_camera.tolist() if frame is not None else None,
                "rgb_path": str(self.frame_dir / f"rgb_{frame_id:08d}.png") if frame_id is not None else None,
                "depth_path": str(self.frame_dir / f"depth_{frame_id:08d}.npy.gz") if frame_id is not None else None,
                "collision_pixel": pixel,
                "observed_depth_m": observed_depth_m,
                "predicted_robot_depth_m": predicted_depth_m,
                "depth_residual_summary": (
                    _depth_residual_summary(frame.depth_m, robot_depth, tuple(pixel["xy"]))
                    if frame is not None and isinstance(robot_depth, np.ndarray) and pixel is not None
                    else None
                ),
            },
            "selected_fk_snapshot": (
                {
                    "frame_id": int(snapshot["frame_id"]),
                    "camera_timestamp_s": float(snapshot["camera_timestamp_s"]),
                    "robot_timestamp_s": float(snapshot["robot_timestamp_s"]),
                    "camera_render_step_counter": int(snapshot["camera_render_step_counter"]),
                    "robot_fk_step_counter": int(snapshot["robot_fk_step_counter"]),
                    "render_event_id": int(snapshot["render_event_id"]),
                    "link_transforms_world": transforms,
                }
                if snapshot is not None else None
            ),
            "collision": {
                **dict(sweep),
                "sweep_diagnostic": dict(collision_record)
                if isinstance(collision_record, Mapping) else None,
            },
        }
        path = private_dir / f"collision_blocked_{self._private_collision_sequence:06d}.json"
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def _write_private_execution_progress(self, progress: Mapping[str, Any]) -> None:
        """Append executor telemetry only to the episode's private diagnostics."""
        private_dir = self.output_dir / "private"
        private_dir.mkdir(parents=True, exist_ok=True)
        with (private_dir / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(progress), allow_nan=False, separators=(",", ":")) + "\n")

    def write_checkpoint(self, call: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        """Persist an eligible physical state without exposing it to the model."""
        release_call = call.get("motion_regime") == "release" or call.get("tool") == "release"
        if release_call:
            self._checkpoint_disabled_after_release = True
        if (
            self.observe_only
            or result.get("status") not in CHECKPOINT_STATUSES
            or result.get("terminal") is not False
            or getattr(self, "_checkpoint_disabled_after_release", False)
        ):
            return
        diagnostics = result.get("execution_diagnostics", {})
        checkpoint = _capture_factory_checkpoint(
            self.env,
            {
                "tool": call.get("tool"),
                "status": result.get("status"),
                "terminal": result.get("terminal"),
                "driver_step_count": diagnostics.get("driver_step_count", 0)
                if isinstance(diagnostics, Mapping) else 0,
            },
            progression=self._evaluator.checkpoint_progression(),
        )
        write_private_checkpoint(self.output_dir / "private" / "checkpoint.json", checkpoint)

    def check_task(self) -> bool:
        if self._evaluator is None:
            raise RuntimeError("task evaluator is not available in observe-only mode")
        if self._last_private_status is None:
            if getattr(self, "_resume_requires_fresh_sample", False):
                return False
            self._sample_private_evaluator()
        return bool(self._last_private_status.task_success)

    def _action_rejection(self, observation: Mapping[str, Any]) -> dict[str, Any] | None:
        if getattr(self, "observe_only", False):
            return {
                "status": "proposal_only",
                "terminal": True,
                "feedback": "Observe-only Factory runtime never sends a model proposal to the motion executor.",
                "execution_diagnostics": {
                    "driver_step_count": 0,
                    "ik_call_count": 0,
                    "sweep_check_count": 0,
                },
            }
        sensors = observation.get("robot_state", {}).get("sensors", {})
        obstacle_map = sensors.get("rgbd_obstacle_map", {}) if isinstance(sensors, Mapping) else {}
        if not isinstance(obstacle_map, Mapping) or obstacle_map.get("robot_self_filtered") is not True:
            return {
                "status": "invalid_rgbd_map",
                "terminal": True,
                "feedback": (
                    "Action rejected: the public RGB-D map is not accurately robot-self-filtered; "
                    "no motion was sent to the executor."
                ),
                "execution_diagnostics": {
                    "driver_step_count": 0,
                    "ik_call_count": 0,
                    "sweep_check_count": 0,
                },
            }
        return None

    def execute_goal(self, goal: Mapping[str, Any], observation: Mapping[str, Any]):
        self._pending_held_surface_track = None
        if goal.get("motion_regime") == "release":
            self._checkpoint_disabled_after_release = True
        rejection = self._action_rejection(observation)
        if rejection is not None:
            return rejection
        resolved = None
        connector = getattr(self.executor, "connector", None)
        if goal.get("motion_regime") == "free_space" and connector is not None:
            try:
                resolved = connector.resolve(goal, observation)
            except (GoalResolutionError, TypeError, ValueError, ArithmeticError):
                pass
        result = self.executor.execute_goal(goal, observation)
        diagnostics = result.get("execution_diagnostics") if isinstance(result, Mapping) else None
        if (
            resolved is not None
            and isinstance(diagnostics, Mapping)
            and type(diagnostics.get("driver_step_count")) is int
            and diagnostics["driver_step_count"] > 0
        ):
            self._pending_held_surface_track = (
                {
                    "motion_regime": "free_space",
                    "constraints": [dict(item) for item in goal.get("constraints", ())],
                },
                observation,
                resolved.tcp_to_moving_feature_pose,
            )
        return result

    def execute_tcp_delta(self, call: Mapping[str, Any], observation: Mapping[str, Any]):
        self._pending_held_surface_track = None
        rejection = self._action_rejection(observation)
        if rejection is not None:
            return rejection
        return self.executor.execute_tcp_delta(call, observation)

    def subscribe_rgb_frames(self, listener):
        if not callable(listener):
            raise TypeError("RGB frame listener must be callable")
        self._listeners.append(listener)
        self._capture_frame()

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def close(self) -> None:
        if self._original_step is not None:
            self.env.step = self._original_step
            self._original_step = None
        if self._original_scene_update is not None:
            self.env.scene.update = self._original_scene_update
            self._original_scene_update = None
        try:
            self.env.close()
        finally:
            self.app.close()


def _finite_list(value: Any, size: int, name: str) -> list[float]:
    result = _numpy(value).reshape(-1)
    if len(result) != size or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return [float(item) for item in result]


def _public_finger_contact_forces_n(env: Any) -> list[float] | None:
    reader = getattr(env, "get_unfiltered_finger_contact", None)
    if not callable(reader):
        return None
    contacts = reader()
    if not isinstance(contacts, Mapping):
        raise TypeError("public bilateral grasp contact must be a mapping")
    return [
        _finite_list(contacts[f"{side}_force_n"], 1, f"{side} finger contact force")[0]
        for side in ("left", "right")
    ]


def _capture_factory_checkpoint(
    env: Any,
    provenance: Mapping[str, Any],
    *,
    progression: Mapping[str, Any],
) -> dict[str, Any]:
    robot = env._robot
    joint_positions = _finite_list(robot.data.joint_pos[0], int(robot.num_joints), "robot joint positions")
    joint_velocities = _finite_list(robot.data.joint_vel[0], int(robot.num_joints), "robot joint velocities")
    if len(joint_positions) != 9 or len(joint_velocities) != 9:
        raise RuntimeError("Factory Panda checkpoint requires 9 robot joint values")
    transforms = _numpy(env._bolt.root_physx_view.get_transforms())
    velocities = _numpy(env._bolt.root_physx_view.get_velocities())
    if transforms.shape != (1, 7) or velocities.shape != (1, 6):
        raise RuntimeError("Factory bolt checkpoint has an unexpected native root-state shape")
    raw_root_pose = _finite_list(transforms[0], 7, "bolt root pose")
    # PhysX exposes root transforms as position + XYZW; the checkpoint contract
    # and write_root_pose_to_sim use position + WXYZ.
    root_pose_w = raw_root_pose[:3] + [
        raw_root_pose[6], raw_root_pose[3], raw_root_pose[4], raw_root_pose[5],
    ]
    gripper_target_joint_m = _finite_list(
        getattr(env, "gripper_target_joint_m"), 1, "gripper target joint",
    )[0]
    checkpoint = {
        "format": "assembly_harness_checkpoint",
        "version": CHECKPOINT_VERSION,
        "robot": {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "gripper_target_joint_m": gripper_target_joint_m,
        },
        "bolt": {
            "root_pose_w": root_pose_w,
            "root_velocity_w": _finite_list(velocities[0], 6, "bolt root velocity"),
        },
        "provenance": dict(provenance),
        "semantic_progression": dict(progression),
    }
    return validate_checkpoint(checkpoint)


def _restore_factory_checkpoint(env: Any, checkpoint: Mapping[str, Any]) -> None:
    import torch

    checkpoint = validate_checkpoint(checkpoint)
    robot = env._robot
    if int(getattr(robot, "num_joints", -1)) != 9:
        raise RuntimeError("Factory Panda checkpoint requires a 9-joint robot")
    if int(getattr(env, "num_envs", 1)) != 1:
        raise RuntimeError("Factory checkpoint restore supports exactly one environment")
    device = getattr(env, "device", robot.data.joint_pos.device)
    env_ids = torch.tensor([0], dtype=torch.long, device=device)
    joint_positions = torch.as_tensor(
        checkpoint["robot"]["joint_positions"], dtype=robot.data.joint_pos.dtype, device=device,
    ).reshape(1, 9)
    joint_velocities = torch.as_tensor(
        checkpoint["robot"]["joint_velocities"], dtype=robot.data.joint_vel.dtype, device=device,
    ).reshape(1, 9)
    root_pose = torch.as_tensor(
        checkpoint["bolt"]["root_pose_w"], dtype=robot.data.joint_pos.dtype, device=device,
    ).reshape(1, 7)
    root_velocity = torch.as_tensor(
        checkpoint["bolt"]["root_velocity_w"], dtype=robot.data.joint_vel.dtype, device=device,
    ).reshape(1, 6)

    robot.write_joint_state_to_sim(joint_positions, joint_velocities, env_ids=env_ids)
    robot.set_joint_position_target(joint_positions, env_ids=env_ids)
    robot.set_joint_effort_target(torch.zeros_like(joint_positions), env_ids=env_ids)
    env.ctrl_target_joint_pos[env_ids] = joint_positions
    # reset() initializes this to the open target; restore the captured target
    # before the first resumed action so a held bolt is not released.
    env.gripper_target_joint_m = float(checkpoint["robot"]["gripper_target_joint_m"])
    env._bolt.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    env._bolt.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
    env._bolt.reset(env_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=0.0)
    env._compute_intermediate_values(0.0)
    current_timestamp = float(_numpy(env._robot._data._sim_timestamp).reshape(-1)[0])
    env._previous_tcp_pos[env_ids] = env.fingertip_midpoint_pos[env_ids]
    env._previous_tcp_quat[env_ids] = env.fingertip_midpoint_quat[env_ids]
    env.ee_linvel_fd[env_ids] = env.fingertip_midpoint_linvel[env_ids]
    env.ee_angvel_fd[env_ids] = env.fingertip_midpoint_angvel[env_ids]
    env.last_update_timestamp = current_timestamp
    env.last_velocity_sample_dt_s = 0.0
    env._wrench_raw_history.clear()
    env._tcp_position_history.clear()
    env._last_wrench_sample_timestamp = None
    env.wrench_assembly_smooth[env_ids] = env.wrench_assembly[env_ids]
    env.last_factory_control_snapshot = None
    env.commanded_tcp_pos[env_ids] = env.fingertip_midpoint_pos[env_ids]
    env.commanded_tcp_quat[env_ids] = env.fingertip_midpoint_quat[env_ids]
    _reset_factory_temporal_windows_after_restore(env)


def _reset_factory_temporal_windows_after_restore(env: Any) -> None:
    """Discard pre-restore native windows while preserving the native clock."""
    samples = getattr(env, "_private_physics_samples", None)
    if not isinstance(samples, list):
        raise RuntimeError("Factory environment does not expose its private native sample buffer")
    completed_steps = getattr(env, "_private_physics_completed_steps", None)
    if (
        isinstance(completed_steps, bool)
        or not isinstance(completed_steps, int)
        or completed_steps < 0
    ):
        raise RuntimeError("Factory environment has an invalid private physics-step counter")
    timestamp_values = _numpy(env._robot._data._sim_timestamp).reshape(-1)
    if timestamp_values.size != 1 or not np.isfinite(timestamp_values).all():
        raise RuntimeError("Factory environment has an invalid restored native timestamp")
    transforms = _numpy(env._bolt.root_physx_view.get_transforms())
    velocities = _numpy(env._bolt.root_physx_view.get_velocities())
    if transforms.shape != (1, 7) or velocities.shape != (1, 6):
        raise RuntimeError("Factory restored native bolt state has an unexpected shape")
    if not np.isfinite(transforms).all() or not np.isfinite(velocities).all():
        raise RuntimeError("Factory restored native bolt state is non-finite")

    samples[:] = [{
        "physics_step_index": completed_steps,
        "sim_timestamp_s": float(timestamp_values[0]),
        "capture_phase": "checkpoint_restore_anchor",
        "native_root_transform_raw": transforms[0].tolist(),
        "native_root_velocity_raw": velocities[0].tolist(),
        "contact_reports": [],
        "decoded_exact_cap_finger_contacts": {"left": [], "right": []},
    }]
    env._private_physics_last_sample_index = completed_steps
    env._private_pose_provider_last_step_index = completed_steps
    env._private_exact_cap_contact_ticks = {"left": [], "right": []}
    with env._private_contact_report_lock:
        env._private_pending_contact_reports.clear()
    env._private_synchronous_contact_report_polls.clear()
    env._private_synchronous_report_last_step_index = completed_steps


def _calibrate_stationary_wrench(env: Any, torch: Any) -> None:
    sample_count = int(env.cfg.wrench_stationary_sample_count)
    warmup_steps = int(env.cfg.wrench_warmup_steps)
    if sample_count < 1 or warmup_steps < sample_count:
        raise ValueError("Factory wrench warmup must cover its calibrated stationary sample window")
    joint_speeds: list[float] = []
    contact_trace: list[tuple[float, float]] = []
    for _ in range(warmup_steps):
        result = env.step(torch.zeros((1, 6), device=env.device))
        if isinstance(result, tuple) and len(result) >= 5:
            if bool(_numpy(result[2]).any() or _numpy(result[3]).any()):
                raise RuntimeError("Factory environment ended during stationary wrench warmup")
        joint_speeds.append(float(np.abs(_numpy(env._robot.data.joint_vel)).max(axis=1).max()))
        contacts = env.get_bilateral_grasp_contact()
        contact_trace.append((
            float(_numpy(contacts["left_force_n"])[0]),
            float(_numpy(contacts["right_force_n"])[0]),
        ))
    tcp_samples = list(env._tcp_position_history)[-sample_count:]
    if len(tcp_samples) != sample_count:
        raise RuntimeError("Factory TCP history is insufficient to calibrate the wrench zero")
    tcp = np.stack([_numpy(sample[0]) for sample in tcp_samples], axis=0)
    tcp_drift = float(np.linalg.norm(tcp - tcp[0], axis=-1).max())
    window_steps = max(1, math.ceil(sample_count / int(env.cfg.decimation)))
    stationary = (
        bool(env.motion_safety_armed)
        and tcp_drift <= float(env.cfg.wrench_stationary_tcp_drift_limit_m)
        and max(joint_speeds[-window_steps:]) <= float(env.cfg.wrench_stationary_joint_speed_limit_rad_s)
    )
    no_contacts = all(
        max(pair) < float(env.cfg.grasp_contact_min_force_n)
        for pair in contact_trace[-window_steps:]
    )
    if not stationary or not no_contacts:
        raise RuntimeError("Factory robot failed the calibrated stationary/no-contact wrench-zero checks")
    env.set_stationary_wrench_baseline(stationary=True, bilateral_contacts_clear=True)


__all__ = [
    "FactoryAssemblyRuntime",
    "derive_factory_robot_calibration",
]
