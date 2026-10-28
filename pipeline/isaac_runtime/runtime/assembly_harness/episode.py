"""One bounded, logged VLM assembly episode over a generic runtime adapter."""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from runtime.bolt_harness.recording import EpisodeRGBVideoRecorder

from .agent import AssemblyVLM
from .harness import AssemblyHarnessLoop, AssemblyHarnessResult


VIDEO_FPS = 12.0


@dataclass(frozen=True)
class AssemblyEpisodeResult:
    status: str
    exit_code: int
    completed: bool
    model_evidence: bool
    video_finalized: bool
    run_dir: Path
    error: str | None = None
    harness_result: AssemblyHarnessResult | None = None


def run_assembly_episode(
    *,
    agent: AssemblyVLM,
    executor: Any,
    task_goal: str,
    observation_callback: Callable[[], Mapping[str, Any]],
    check_task_callback: Callable[[], bool],
    frame_subscribe: Callable[[Callable[[np.ndarray, float], None]], Callable[[], None] | None],
    checkpoint_callback: Callable[[Mapping[str, Any], Mapping[str, Any]], None] | None = None,
    run_dir: str | Path,
    video_fps: float = VIDEO_FPS,
    max_calls: int = 48,
    observe_only: bool = False,
) -> AssemblyEpisodeResult:
    """Run a single episode and persist public model/action evidence plus video.

    ``frame_subscribe`` registers a listener that receives each new RGB array and
    its simulation timestamp. Model HTTP wait time is deliberately not represented
    as simulated motion.
    """
    root = Path(run_dir)
    video_path = root / "episode.mp4"
    trace_path = root / "public_trace.jsonl"
    summary_path = root / "run_summary.json"
    conflict_paths = (video_path, video_path.with_suffix(".timestamps.json"), trace_path, summary_path)
    root.mkdir(parents=True, exist_ok=True)
    if any(path.exists() for path in conflict_paths):
        return AssemblyEpisodeResult(
            "output_conflict", 1, False, False, False, root,
            "episode output already exists; existing evidence was preserved",
        )

    started = time.monotonic()
    recorder = EpisodeRGBVideoRecorder(video_path, fps=video_fps)
    trace_stream = None
    unsubscribe = None
    harness_result = None
    failure: BaseException | None = None
    error: str | None = None
    model_requests = 0
    model_responses = 0
    model_frame_sequences: list[list[dict[str, Any]]] = []
    model_frame_files: list[dict[str, str]] = []
    video_error: str | None = None
    last_sim_time_s: float | None = None

    def trace(event: dict[str, Any]) -> None:
        nonlocal model_requests, model_responses
        if event.get("event") == "model_request":
            model_requests += 1
            frame_sequence = event.get("frame_sequence")
            if isinstance(frame_sequence, list) and all(isinstance(item, Mapping) for item in frame_sequence):
                model_frame_sequences.append([dict(item) for item in frame_sequence])
                frame_files = event.get("frame_files")
                if isinstance(frame_files, Mapping) and all(
                    isinstance(key, str) and isinstance(path, str)
                    for key, path in frame_files.items()
                ):
                    model_frame_files.append(dict(frame_files))
        elif event.get("event") == "model_response":
            model_responses += 1
        if trace_stream is None:
            raise RuntimeError("public trace stream is not open")
        trace_stream.write(json.dumps(
            {"time_utc": datetime.now(timezone.utc).isoformat(), **event},
            ensure_ascii=True, allow_nan=False,
        ) + "\n")
        trace_stream.flush()

    def on_frame(rgb: np.ndarray, sim_time_s: float) -> None:
        nonlocal last_sim_time_s
        recorder.append(rgb, sim_time_s)
        last_sim_time_s = float(sim_time_s)

    previous_agent_trace = getattr(agent, "trace_callback", None)

    def model_trace(event: dict[str, Any]) -> None:
        if callable(previous_agent_trace):
            previous_agent_trace(event)
        trace(event)

    try:
        trace_stream = trace_path.open("x", encoding="utf-8")
        agent.trace_callback = model_trace
        unsubscribe = frame_subscribe(on_frame)
        if unsubscribe is not None and not callable(unsubscribe):
            raise TypeError("frame_subscribe must return a cleanup callable or None")
        harness = AssemblyHarnessLoop(
            agent,
            executor,
            task_goal=task_goal,
            observation_callback=observation_callback,
            check_task_callback=check_task_callback,
            trace_callback=trace,
            checkpoint_callback=checkpoint_callback,
            max_calls=max_calls,
            observe_only=observe_only,
        )
        harness_result = harness.run()
        if harness_result.exit_code != 0:
            error = harness_result.error or harness_result.status
            if harness_result.private_exception is not None:
                failure = harness_result.private_exception
    except BaseException as exc:
        failure = exc
        error = f"episode failed: {type(exc).__name__}"
        if trace_stream is not None:
            trace({"event": "episode_failure", "error": error})
    finally:
        if unsubscribe is not None:
            try:
                unsubscribe()
            except Exception as exc:
                failure = failure or exc
                error = error or "frame listener cleanup failed"
        try:
            video_error = recorder.close()
        except Exception as exc:
            video_error = f"{type(exc).__name__}: {exc}"
        if trace_stream is not None:
            try:
                trace_stream.close()
            except Exception as exc:
                failure = failure or exc
                error = error or "public trace could not be finalized"
        if failure is not None:
            try:
                private_dir = root / "private"
                private_dir.mkdir(parents=True, exist_ok=True)
                (private_dir / "exception.txt").write_text(
                    "".join(traceback.format_exception(type(failure), failure, failure.__traceback__)),
                    encoding="utf-8",
                )
            except Exception:
                error = error or "private exception evidence could not be written"
        try:
            agent.trace_callback = previous_agent_trace
        except Exception as exc:
            failure = failure or exc
            error = error or "agent trace callback could not be restored"

    model_evidence = model_requests > 0 and model_responses > 0
    evaluator_summary = None
    evaluator_summary_callback = getattr(executor, "official_evaluator_summary", None)
    if callable(evaluator_summary_callback):
        try:
            evaluator_summary = evaluator_summary_callback()
            if not isinstance(evaluator_summary, Mapping):
                raise TypeError("official evaluator summary must be a mapping")
        except Exception as exc:
            error = error or f"official evaluator summary failed: {type(exc).__name__}"
            failure = failure or exc
    video_finalized = (
        video_error is None and recorder.frame_count > 0 and video_path.is_file()
        and video_path.stat().st_size > 0
        and video_path.with_suffix(".timestamps.json").is_file()
        and video_path.with_suffix(".timestamps.json").stat().st_size > 0
    )
    completed = bool(harness_result is not None and harness_result.completed and harness_result.status == "success")
    success = completed and model_evidence and video_finalized and failure is None
    last_tool_event = harness_result.history[-1] if harness_result and harness_result.history else None
    parsed_tool_call = last_tool_event.get("call") if observe_only and last_tool_event else None
    parsed_tool_result = last_tool_event.get("result") if observe_only and last_tool_event else None
    observe_status = None
    if observe_only and model_evidence and video_finalized and failure is None and last_tool_event:
        tool = parsed_tool_call.get("tool") if isinstance(parsed_tool_call, Mapping) else None
        result_status = parsed_tool_result.get("status") if isinstance(parsed_tool_result, Mapping) else None
        observe_status = {
            ("execute_goal", "proposal_only"): "proposal_recorded",
            ("grasp", "proposal_only"): "proposal_recorded",
            ("stop", "stopped"): "model_stopped",
            ("observe", "completed"): "observation_recorded",
        }.get((tool, result_status))
    operation_succeeded = success or observe_status is not None
    if success:
        status = "success"
    elif observe_status is not None:
        status = observe_status
        error = None
    elif harness_result is not None and not completed:
        status = harness_result.status
    else:
        status = "failed"
    summary = {
        "task_goal": task_goal,
        "observe_only": observe_only,
        "model": getattr(agent, "model", None),
        "runtime_metadata": getattr(executor, "task_metadata", None),
        "official_evaluator": dict(evaluator_summary) if evaluator_summary is not None else None,
        "model_request_count": model_requests,
        "model_response_count": model_responses,
        "model_frame_sequences": model_frame_sequences,
        "model_frame_files": model_frame_files,
        "model_evidence": model_evidence,
        "calls": harness_result.calls if harness_result else 0,
        "operation_succeeded": operation_succeeded,
        "assembly_completed": completed,
        "completed": completed,
        "parsed_tool_call": parsed_tool_call,
        "parsed_tool_result": parsed_tool_result,
        "status": status,
        "exit_code": 0 if operation_succeeded else 1,
        "video_file": video_path.name,
        "video_fps": video_fps,
        "video_frame_count": recorder.frame_count,
        "video_finalized": video_finalized,
        "sim_time_s": last_sim_time_s,
        "wall_time_s": time.monotonic() - started,
        "public_trace": trace_path.name,
        "error": error,
        "video_error": video_error,
    }
    try:
        with summary_path.open("x", encoding="utf-8") as summary_stream:
            summary_stream.write(json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
    except Exception as exc:
        return AssemblyEpisodeResult(
            "failed", 1, completed, model_evidence, video_finalized, root,
            f"run summary could not be written: {type(exc).__name__}", harness_result,
        )
    return AssemblyEpisodeResult(
        status, 0 if operation_succeeded else 1, completed, model_evidence, video_finalized,
        root, error or video_error, harness_result,
    )


__all__ = ["AssemblyEpisodeResult", "VIDEO_FPS", "run_assembly_episode"]
