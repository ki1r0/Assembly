from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time
from typing import Any, Dict, List

import numpy as np
import omni.kit.app
import omni.timeline
import omni.ui as ui

from belief import BeliefManager, GhostVisualizer
from sensor import capture_rgb_uint8_async, get_latest_rgb_uint8, init_camera, save_last_frame, save_sent_frames
from sensor import cosmos_endpoint, test_cosmos_connection, StateMonitor
from agent import cognitive_worker
from control import FrankaControlPolicy
from memory import LongTermMemory, ShortTermMemory
from .config import (
    CAPTURE_FPS,
    COSMOS_MODEL,
    INQUIRY_FRAME_COUNT,
    INQUIRY_INTERVAL_SEC,
    MEM0_API_KEY,
    MIN_INFER_INTERVAL_SEC,
    STREAM_DEFAULT_PROMPT,
    TIMEOUT_SEC,
)
from .log import log_error, log_info, log_line, log_warn, render_log
from .paths import asset_browser_cache_dir
from .state import STATE


def _get_task_loop() -> asyncio.AbstractEventLoop:
    """Return an asyncio loop suitable for scheduling background tasks in Kit.

    Compatibility note: some Kit builds expose `IApp.get_async_loop()`, others do not.
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        pass
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        pass
    try:
        app = omni.kit.app.get_app()
        get_async_loop = getattr(app, "get_async_loop", None)
        if callable(get_async_loop):
            return get_async_loop()
    except Exception:
        pass
    raise RuntimeError("No available asyncio loop for scheduling task.")


def _get_recent_frames(last_seconds: float, max_frames: int) -> List[Any]:
    """Return up to max_frames from the recent window, evenly spread oldest->newest."""
    if not STATE.frame_buffer:
        return []
    items = list(STATE.frame_buffer)  # (ts, rgb)
    now = time.time()
    cutoff = now - max(0.0, float(last_seconds))
    recent_items = [(ts, rgb) for ts, rgb in items if ts >= cutoff]
    if not recent_items:
        return []
    max_frames = max(1, int(max_frames))
    if len(recent_items) <= max_frames:
        return [rgb for _ts, rgb in recent_items]

    # Evenly sample across the window so motion is visible to the VLM.
    n = len(recent_items)
    step = (n - 1) / float(max_frames - 1)
    idxs = []
    for i in range(max_frames):
        idx = int(round(i * step))
        idx = max(0, min(n - 1, idx))
        idxs.append(idx)

    # Keep index order monotonic and unique.
    dedup = []
    seen = set()
    for idx in idxs:
        if idx not in seen:
            seen.add(idx)
            dedup.append(idx)
    return [recent_items[i][1] for i in dedup]


def _clone_frame(frame: Any) -> Any:
    try:
        return frame.copy()
    except Exception:
        return frame


def _get_latest_buffer_frames(max_frames: int) -> List[Any]:
    """Return latest frames from buffer without a recency cutoff (oldest->newest)."""
    if not STATE.frame_buffer:
        return []
    max_frames = max(1, int(max_frames))
    items = list(STATE.frame_buffer)
    picked = items[-max_frames:]
    return [_clone_frame(rgb) for _ts, rgb in picked]


def _canon_name(name: str) -> str:
    return "".join(ch for ch in str(name or "").lower() if ch.isalnum())


def _collect_gt_interactable_names(max_names: int = 64) -> List[str]:
    """Collect interactable names from GT state monitor for initialization form."""
    names: List[str] = []
    sm = STATE.state_monitor
    if sm is None:
        return names

    try:
        poses = sm.get_current_poses()
        if isinstance(poses, dict) and poses:
            for pose in poses.values():
                n = str(getattr(pose, "name", "") or "").strip()
                if n:
                    names.append(n)
    except Exception:
        pass

    if not names:
        try:
            discovered = sm.auto_discover_objects()
            if discovered:
                sm.set_tracked_prims(discovered)
                for prim_path in discovered:
                    n = str(prim_path or "").rstrip("/").split("/")[-1].strip()
                    if n:
                        names.append(n)
        except Exception:
            pass

    if not names:
        for prim_path in list(getattr(sm, "_tracked_paths", []) or []):
            n = str(prim_path or "").rstrip("/").split("/")[-1].strip()
            if n:
                names.append(n)

    deduped: List[str] = []
    seen = set()
    for n in names:
        c = _canon_name(n)
        if not c or c in seen:
            continue
        seen.add(c)
        deduped.append(n)
        if len(deduped) >= max(1, int(max_names)):
            break
    return deduped


def _build_init_interactable_form(interactable_names: List[str]) -> Dict[str, Any]:
    objects: Dict[str, Any] = {}
    for name in list(interactable_names or []):
        key = str(name or "").strip()
        if not key:
            continue
        # Empty per-object form; model fills fields.
        objects[key] = {}
    return {
        "belief_state_update": {"objects": objects, "static_context": {}},
        "stm_observation": "",
        "reply": "",
        "action": {"type": "noop", "args": {}},
    }


def _get_frames_since(last_ts: float, max_frames: int) -> tuple[List[Any], float]:
    """Return up to max_frames newer than last_ts, sampled oldest->newest.

    Includes one anchor frame from <= last_ts (when available) so each request has temporal continuity
    with the previous request.
    """
    if not STATE.frame_buffer:
        return [], float(last_ts)
    items = list(STATE.frame_buffer)  # (ts, rgb)
    newer_items = [(ts, rgb) for ts, rgb in items if float(ts) > float(last_ts)]
    if not newer_items:
        return [], float(last_ts)

    # Add one anchor frame to preserve change across request boundaries.
    anchor = None
    if last_ts > 0.0:
        for ts, rgb in reversed(items):
            if float(ts) <= float(last_ts):
                anchor = (ts, rgb)
                break
    if anchor is not None:
        newer_items = [anchor] + newer_items

    max_frames = max(1, int(max_frames))
    if len(newer_items) <= max_frames:
        return [rgb for _ts, rgb in newer_items], float(newer_items[-1][0])

    # Evenly sample across all newly captured frames to preserve motion over the whole wait period.
    n = len(newer_items)
    step = (n - 1) / float(max_frames - 1)
    idxs = []
    for i in range(max_frames):
        idx = int(round(i * step))
        idx = max(0, min(n - 1, idx))
        idxs.append(idx)

    dedup = []
    seen = set()
    for idx in idxs:
        if idx not in seen:
            seen.add(idx)
            dedup.append(idx)
    sampled = [newer_items[i][1] for i in dedup]
    return sampled, float(newer_items[-1][0])


def _log_long(prefix: str, text: str, chunk: int = 400) -> None:
    if not text:
        log_line(prefix, "<empty response>")
        return
    for i in range(0, len(text), int(chunk)):
        log_line(prefix, text[i : i + int(chunk)])


def _extract_reply_text(text: str) -> str:
    """Best-effort extraction of `reply` from JSON-looking model output."""
    s = str(text or "").strip()
    if not s:
        return ""
    if s.startswith("{"):
        try:
            payload = json.loads(s)
            if isinstance(payload, dict):
                r = str(payload.get("reply") or "").strip()
                if r:
                    return r
        except Exception:
            pass
    return s


def _update_vlm_status(status: str, detail: str = "") -> None:
    """Update the VLM occupancy indicator in the UI and log."""
    label = STATE.vlm_status_label
    if label is None:
        return
    try:
        if status == "idle":
            label.text = "VLM: Idle"
            label.set_style({"color": 0xFF00CC00})  # green
        elif status == "busy":
            trigger = detail or STATE.vlm_busy_trigger or "auto"
            if trigger == "user":
                label.text = "VLM: Processing your question..."
                label.set_style({"color": 0xFF00AAFF})  # blue
            elif trigger == "grounding":
                label.text = "VLM: Grounding..."
                label.set_style({"color": 0xFFFFAA00})  # orange
            else:
                label.text = f"VLM: Busy ({trigger})"
                label.set_style({"color": 0xFFFF4444})  # red
        elif status == "queued":
            label.text = "VLM: Queued"
            label.set_style({"color": 0xFFFFCC00})  # yellow
    except Exception:
        pass


def _has_pending_inference() -> bool:
    """True when there is an in-flight or queued model request."""
    if bool(getattr(STATE, "vlm_busy", False)):
        return True
    q = getattr(STATE, "worker_in_q", None)
    if q is not None:
        try:
            if not q.empty():
                return True
        except Exception:
            pass
    return False


def _sync_vlm_status_indicator() -> None:
    """Keep UI indicator consistent with queue/in-flight state."""
    if STATE.manual_pending:
        _update_vlm_status("busy", "user")
        return
    if _has_pending_inference():
        _update_vlm_status("busy", STATE.vlm_busy_trigger or "auto")
        return
    _update_vlm_status("idle")


def _compact_belief_from_snapshot(snapshot: Dict[str, Any]) -> str:
    """Render a compact belief summary directly from belief snapshot."""
    if not isinstance(snapshot, dict):
        return "no objects tracked"
    objects = snapshot.get("objects")
    static_context = snapshot.get("static_context")
    if not isinstance(objects, dict) or not objects:
        if isinstance(static_context, dict) and static_context:
            return f"no objects tracked | background={len(static_context)}"
        return "no objects tracked"
    parts: List[str] = []
    for name, obj in objects.items():
        if not isinstance(obj, dict):
            continue
        status = str(obj.get("belief_status", "unknown"))
        stale = bool(obj.get("stale", False))
        conf = obj.get("confidence")
        container = str(obj.get("inferred_container", "")).strip()
        s = f"{name}: {status}"
        if container:
            s += f" in {container}"
        if stale:
            s += " (stale)"
        if conf is not None:
            try:
                s += f" conf={float(conf):.2f}"
            except Exception:
                pass
        parts.append(s)
    if not parts:
        if isinstance(static_context, dict) and static_context:
            return f"no objects tracked | background={len(static_context)}"
        return "no objects tracked"
    if isinstance(static_context, dict) and static_context:
        parts.append(f"background={len(static_context)}")
    return " | ".join(parts)[:600]


def _configure_quiet_kit_logging() -> None:
    """Reduce known-noisy Kit warning channels for this demo.

    This does *not* suppress our own logs (we render them in the UI), it only reduces terminal spam from
    common rendering/perf warnings.
    """
    try:
        import carb

        s = carb.settings.get_settings()
        # Keep global level as-is; tune only the noisy channels.
        s.set("/log/channels/omni.hydra", "error")
        s.set("/log/channels/omni.syntheticdata.plugin", "error")
        s.set("/log/channels/rtx.postprocessing.plugin", "error")
        s.set("/log/channels/omni.fabric.plugin", "error")
        s.set("/log/channels/omni.kit.window.property.*", "error")
        s.set("/log/channels/semantics.schema.property.*", "error")
        # Audio warnings are common in containers.
        s.set("/log/channels/carb.audio.*", "error")
    except Exception:
        pass


def _ensure_asset_browser_cache_dir() -> None:
    """Prevent noisy FileNotFoundError from isaacsim.asset.browser cache writes in some container layouts."""
    try:
        cache_dir = str(asset_browser_cache_dir())
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "isaacsim.asset.browser.cache.json")
        if not os.path.exists(cache_file):
            # Create a valid empty json file.
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write("{}\n")
    except Exception:
        pass


def _ensure_magic_assembly_once() -> None:
    """Initialize the magic assembly manager lazily for slash commands and startup."""
    if STATE.magic_assembly is not None:
        return

    from .magic_assembly import MagicAssemblyManager

    STATE.magic_assembly = MagicAssemblyManager(logger=lambda m: log_line("INFO ", m))
    try:
        created = STATE.magic_assembly.ensure_extra_hub_bolt_assets()
        if created.get("bolts", 0) or created.get("sockets", 0):
            log_line(
                "INFO ",
                "Extra hub-bolt assets ensured "
                f"(bolts={created.get('bolts', 0)}, sockets={created.get('sockets', 0)})",
            )
    except Exception as exc:
        log_line("WARN ", f"Failed to ensure extra hub-bolt assets: {exc}")
    try:
        created = STATE.magic_assembly.ensure_case_attachment_assets()
        if any(created.values()):
            log_line(
                "INFO ",
                "Case assets ensured "
                f"(bolts={created.get('bolts', 0)}, oils={created.get('oils', 0)}, "
                f"top_sockets={created.get('top_sockets', 0)}, "
                f"base_alias_sockets={created.get('base_alias_sockets', 0)})",
            )
    except Exception as exc:
        log_line("WARN ", f"Failed to ensure case attachment assets: {exc}")


def _ui_pose_resolver(child: str, step_id: str):
    """Pose resolver for the SequenceRunner, routed through the chat UI.

    A check_pose step cannot prompt synchronously: the runner executes on the
    Kit main thread, so blocking here would freeze the very thread that delivers
    the answer. Instead we log the question and return None, which tells the
    runner to pause the run (recording pending_pose). The operator types Y/N in
    the chat box; on_send() routes that to runner.answer_pose() and re-dispatches
    the paused command (STATE.pose_resume_cmd) to resume.
    """
    log_line("INFO ", f"[POSE?] Is '{child}' correctly posed? Reply Y / N "
                      f"(or /pose_yes // pose_no)")
    return None


def _parse_pose_answer(text: str):
    """Map chat input to a pose answer: True (yes), False (no), or None (neither)."""
    t = (text or "").strip().lower()
    if t in ("y", "yes", "/pose_yes"):
        return True
    if t in ("n", "no", "/pose_no"):
        return False
    return None


def _run_with_pose_pause(text: str, run_fn):
    """Invoke a runner run method, remembering the command so a deferred
    check_pose can be resumed after the operator answers.

    Sets STATE.pose_resume_cmd before running; clears it if the run did not
    pause (so a normal completion leaves no stale resume target). Returns the
    run method's own return value. Caller checks STATE.sequence_runner.pending_pose
    to decide whether to log a normal completion message."""
    STATE.pose_resume_cmd = text
    result = run_fn()
    sr = STATE.sequence_runner
    if sr is None or sr.pending_pose is None:
        STATE.pose_resume_cmd = None
    return result


def _ensure_sequence_runner_once() -> None:
    """Initialize the SequenceRunner lazily, after magic_assembly is ready."""
    if STATE.sequence_runner is not None:
        return
    _ensure_magic_assembly_once()
    if STATE.magic_assembly is None:
        log_line("WARN ", "[SEQ] cannot init SequenceRunner: magic_assembly not ready")
        return
    try:
        from assembly import SequenceRunner
        runner = SequenceRunner(
            magic_assembly=STATE.magic_assembly,
            log_fn=log_line,
        )
        runner.set_pose_resolver(_ui_pose_resolver)
        if runner.load():
            STATE.sequence_runner = runner
        else:
            log_line("WARN ", "[SEQ] SequenceRunner.load() failed")
    except Exception as exc:
        log_line("WARN ", f"[SEQ] SequenceRunner init error: {exc}")


def _load_instance_runner(variant: str) -> bool:
    """Rebuild the SequenceRunner pointed at a generated instance file
    (assembly/instances/<variant>.yaml) instead of the canonical sequence.
    Returns True on success."""
    from pathlib import Path

    _ensure_magic_assembly_once()
    if STATE.magic_assembly is None:
        log_line("WARN ", "[SEQ] cannot load instance: magic_assembly not ready")
        return False

    import assembly as _assembly_pkg
    inst_dir = Path(_assembly_pkg.__file__).parent / "instances"
    inst_path = inst_dir / f"{variant}.yaml"
    if not inst_path.exists():
        available = sorted(p.stem for p in inst_dir.glob("*.yaml"))
        log_line("WARN ", f"/load_instance: unknown variant {variant!r}. "
                          f"Available: {', '.join(available) or '(none)'}")
        return False

    try:
        from assembly import SequenceRunner
        runner = SequenceRunner(
            magic_assembly=STATE.magic_assembly,
            log_fn=log_line,
            sequence_path=inst_path,
        )
        runner.set_pose_resolver(_ui_pose_resolver)
        if not runner.load():
            log_line("WARN ", f"/load_instance: load() failed for {variant!r}")
            return False
        STATE.sequence_runner = runner
        log_line("INFO ", f"/load_instance: loaded {variant!r} "
                          f"({len(runner._steps)} steps) — run with /run_sequence")
        return True
    except Exception as exc:
        log_line("WARN ", f"/load_instance: error loading {variant!r}: {exc}")
        return False


def _current_stage_id(stage) -> str:
    """Return a stable identifier for the currently opened stage."""
    try:
        root = stage.GetRootLayer()
        return str(root.realPath or root.identifier or "")
    except Exception:
        return ""


def _capture_initial_scene_snapshot_once(force: bool = False) -> bool:
    """Remember each top-level prim's original parent and local transform."""
    import omni.usd
    from pxr import Gf, Usd, UsdGeom  # type: ignore

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        log_warn("Cannot capture initial scene snapshot: stage unavailable.")
        return False

    stage_id = _current_stage_id(stage)
    if (
        not force
        and STATE.initial_scene_snapshot
        and STATE.initial_scene_stage_id == stage_id
    ):
        return True

    root = stage.GetDefaultPrim()
    if root is None or not root.IsValid():
        log_warn("Cannot capture initial scene snapshot: defaultPrim unavailable.")
        return False

    snapshot: Dict[str, Dict[str, Any]] = {}
    tc = Usd.TimeCode.Default()
    for child in root.GetChildren():
        if child is None or not child.IsValid():
            continue
        name = child.GetName()
        if not name or name.startswith("socket_") or name.startswith("plug_"):
            continue
        try:
            xf = UsdGeom.Xformable(child)
            local = xf.GetLocalTransformation(tc)
            if isinstance(local, tuple):
                local = local[0]
            local = Gf.Matrix4d(local)
        except Exception:
            local = Gf.Matrix4d(1.0)
        snapshot[name] = {
            "path": str(child.GetPath()),
            "parent_path": str(child.GetParent().GetPath()),
            "local_matrix": Gf.Matrix4d(local),
        }

    STATE.initial_scene_snapshot = snapshot
    STATE.initial_scene_stage_id = stage_id
    log_line("INFO ", f"Captured initial scene snapshot for {len(snapshot)} top-level prim(s).")
    return True


def _restore_scene_to_initial_snapshot() -> bool:
    """Restore top-level prims to the parent/local transforms captured at startup."""
    from pxr import Gf, Sdf  # type: ignore

    if not _capture_initial_scene_snapshot_once():
        return False

    try:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
    except Exception:
        stage = None
    if stage is None:
        log_warn("/reset: stage unavailable.")
        return False

    try:
        _ensure_magic_assembly_once()
    except Exception as exc:
        log_warn(f"/reset: magic_assembly init failed: {exc}")
        return False

    ma = STATE.magic_assembly
    if ma is None:
        log_warn("/reset: magic_assembly unavailable.")
        return False

    restored = 0
    missing: List[str] = []
    failed: List[str] = []

    for name, spec in STATE.initial_scene_snapshot.items():
        current_path = ma._find_prim_path(stage, name)
        if current_path is None:
            missing.append(name)
            continue

        current_path = Sdf.Path(str(current_path))
        target_parent = Sdf.Path(str(spec.get("parent_path") or ""))
        if not target_parent.IsAbsolutePath():
            failed.append(name)
            continue

        if current_path.GetParentPath() != target_parent:
            moved_path = ma._reparent(stage, current_path, target_parent)
            if moved_path is None:
                failed.append(name)
                continue
            current_path = moved_path

        prim = stage.GetPrimAtPath(current_path)
        if prim is None or not prim.IsValid():
            missing.append(name)
            continue

        try:
            ma._set_xform_matrix(prim, Gf.Matrix4d(spec["local_matrix"]))
            ma._enable_rigid_body(prim)
            restored += 1
        except Exception:
            failed.append(name)

    try:
        ma._records.clear()
    except Exception:
        pass

    if restored:
        log_line("INFO ", f"/reset: restored {restored} top-level prim(s) to startup pose.")
    if missing:
        preview = ", ".join(missing[:8])
        extra = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
        log_line("WARN ", f"/reset: missing prims not restored: {preview}{extra}")
    if failed:
        preview = ", ".join(failed[:8])
        extra = "" if len(failed) <= 8 else f" (+{len(failed) - 8} more)"
        log_line("WARN ", f"/reset: failed to restore: {preview}{extra}")
    return restored > 0 and not failed


def _init_pipeline_once() -> None:
    """Initialize cognition + memory + (optional) robot control once per session."""
    print(f"[PIPELINE] _init_pipeline_once() called", flush=True)
    # Recreate LTM/worker when Mem0 key changes (common during interactive tuning).
    mem0_fp = (str(MEM0_API_KEY or "").strip() or "none")[-8:]
    need_worker_restart = False
    if getattr(STATE, "_ltm_key_fp", None) != mem0_fp:
        STATE._ltm_key_fp = mem0_fp
        if STATE.long_memory is not None:
            need_worker_restart = True
            STATE.long_memory = None
            log_line("INFO ", "Mem0 key change detected; reinitializing long-term memory.")

    if STATE.belief_manager is None:
        STATE.belief_manager = BeliefManager(initial_state={"objects": {}, "static_context": {}})
        log_line("INFO ", "BeliefManager initialized.")

    if STATE.short_memory is None:
        STATE.short_memory = ShortTermMemory(ttl_sec=6.0, logger=lambda m: log_line("INFO ", m))
        log_line("INFO ", "ShortTermMemory initialized.")

    if STATE.long_memory is None:
        STATE.long_memory = LongTermMemory(logger=lambda m: log_line("INFO ", m))
        log_line("INFO ", "LongTermMemory initialized.")

    if STATE.ghost_visualizer is None:
        # PXR-only (no omni.isaac.core dependency).
        STATE.ghost_visualizer = GhostVisualizer(logger=lambda m: log_line("INFO ", m))

    if STATE.robot_controller is None:
        # Safe to create even when no Franka is present; it self-disables and logs.
        STATE.robot_controller = FrankaControlPolicy(logger=lambda m: log_line("INFO ", m))

    _ensure_magic_assembly_once()

    if STATE.state_monitor is None:
        from .config import GT_TRACKED_PRIMS, GT_POSITION_THRESHOLD, GT_ORIENTATION_THRESHOLD, GT_COOLDOWN_SEC
        STATE.state_monitor = StateMonitor(
            tracked_prim_paths=GT_TRACKED_PRIMS if GT_TRACKED_PRIMS else None,
            position_threshold=GT_POSITION_THRESHOLD,
            orientation_threshold=GT_ORIENTATION_THRESHOLD,
            cooldown_sec=GT_COOLDOWN_SEC,
            logger=lambda m: log_line("INFO ", m)
        )
        # Force initial discovery and log results
        try:
            discovered = STATE.state_monitor.auto_discover_objects()
            if discovered:
                STATE.state_monitor.set_tracked_prims(discovered)
                log_line("INFO ", f"GT StateMonitor tracking {len(discovered)} objects: {discovered[:10]}")
            else:
                log_line("WARN ", "GT StateMonitor: no trackable objects found in scene")
        except Exception as exc:
            log_line("WARN ", f"GT StateMonitor discovery failed: {exc}")

    if STATE.worker_in_q is None or STATE.worker_out_q is None:
        STATE.worker_in_q = queue.Queue(maxsize=1)
        STATE.worker_out_q = queue.Queue()

    if need_worker_restart and STATE.worker_thread is not None and STATE.worker_thread.is_alive():
        try:
            _clear_worker_input_queue()
            STATE.worker_in_q.put_nowait(None)
            STATE.worker_thread.join(timeout=1.0)
        except Exception:
            pass
        STATE.worker_thread = None
        STATE.worker_in_q = queue.Queue(maxsize=1)
        STATE.worker_out_q = queue.Queue()
        log_line("INFO ", "Cognitive worker restarted for new memory backend configuration.")

    if STATE.worker_thread is None or (not STATE.worker_thread.is_alive()):
        STATE.worker_thread = threading.Thread(
            target=cognitive_worker,
            args=(STATE.belief_manager, STATE.short_memory, STATE.long_memory, STATE.worker_in_q, STATE.worker_out_q),
            name="cognitive_worker",
            daemon=True,
        )
        STATE.worker_thread.start()
        log_line("INFO ", "Cognitive worker thread started.")

    if STATE.worker_poll_task is None:
        loop = _get_task_loop()
        STATE.worker_poll_task = loop.create_task(_worker_poll_loop())


def _queue_put_latest(q: queue.Queue, item: Dict[str, Any]) -> None:
    """Keep at most one pending request; always replace with the latest."""
    if q is None:
        return
    try:
        while True:
            q.get_nowait()
            if hasattr(q, "task_done"):
                try:
                    q.task_done()
                except Exception:
                    pass
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        # Best-effort drop+retry once.
        try:
            q.get_nowait()
        except Exception:
            return
        try:
            q.put_nowait(item)
        except Exception:
            return


def _clear_worker_input_queue() -> None:
    if STATE.worker_in_q is None:
        return
    try:
        while True:
            STATE.worker_in_q.get_nowait()
            if hasattr(STATE.worker_in_q, "task_done"):
                try:
                    STATE.worker_in_q.task_done()
                except Exception:
                    pass
    except queue.Empty:
        pass


def _enqueue_inference(*, frames: List[Any], user_text: str, trigger: Dict[str, Any], force: bool) -> bool:
    """Queue an inference request to the worker (never blocks the Kit thread)."""
    print(f"[ENQUEUE] _enqueue_inference(frames={len(frames) if frames else 0}, trigger={trigger}, force={force})", flush=True)
    _init_pipeline_once()
    if STATE.worker_in_q is None:
        print(f"[ENQUEUE] worker_in_q is None, returning False", flush=True)
        return False

    now = time.time()
    trigger_type = str((trigger or {}).get("type") or "auto")

    # Drop malformed frame placeholders (None / wrong shape) before request packing.
    valid_frames: List[Any] = []
    dropped = 0
    for frame in list(frames or []):
        try:
            arr = np.asarray(frame)
            if arr.ndim == 3 and arr.size > 0:
                valid_frames.append(_clone_frame(arr))
            else:
                dropped += 1
        except Exception:
            dropped += 1
    if dropped:
        log_line("WARN ", f"Dropped {dropped} invalid frame(s) before enqueue.")

    # Backpressure: while one request is in-flight, drop auto heartbeats.
    # Manual/user requests bypass this when force=True (they preempt queued auto requests).
    if (not force) and STATE.vlm_busy and trigger_type != "user":
        return False

    # Rate-limit all non-forced triggers to prevent burst firing.
    # User triggers (force=True) bypass this entirely.
    if (not force) and (now - float(STATE.last_infer_time or 0.0) < float(MIN_INFER_INTERVAL_SEC)):
        return False
    STATE.last_infer_time = now

    # GT-change inquiries must include a full post-trigger frame packet.
    # Runtime contract: dispatch once 5 post-change frames are collected.
    if trigger_type == "gt_change" and len(valid_frames) < 5:
        print(f"[ENQUEUE] WARN: insufficient gt_change frames={len(valid_frames)}", flush=True)
        log_line("WARN ", "GT trigger requires 5 valid frames — deferring.")
        return False

    # Zero-frame guard.
    if not valid_frames:
        print(f"[ENQUEUE] WARN: 0 frames for trigger={trigger_type}!", flush=True)
        if trigger_type == "grounding":
            log_line("ERR  ", "Grounding aborted: no valid frame was captured.")
            return False
        # user/manual requests may still run text-only from belief+memory.
        log_line("WARN ", f"0 frames in {trigger_type} request — model will answer from belief only.")

    # Get current GT state from StateMonitor
    gt_state = {}
    if STATE.state_monitor is not None:
        try:
            current_poses = STATE.state_monitor.get_current_poses()
            if current_poses:
                gt_objects = {}
                for name, pose in current_poses.items():
                    gt_objects[pose.name] = {
                        "prim_path": pose.prim_path,
                        "position": list(pose.position),
                        "orientation": list(pose.orientation),
                        "timestamp": pose.timestamp
                    }
                gt_state = {"ground_truth_objects": gt_objects, "timestamp": now}
        except Exception as exc:
            log_line("WARN ", f"Failed to get GT state: {exc}")

    # Log GT state for debugging
    gt_obj_names = list(gt_state.get("ground_truth_objects", {}).keys()) if gt_state else []
    if gt_obj_names:
        log_line("INFO ", f"GT objects in request: {gt_obj_names}")
    else:
        log_line("INFO ", "GT state: empty (StateMonitor has no tracked objects)")

    # Include the full current GT interactable set in gt_change triggers so the
    # model can reason over all tracked objects, not only changed_objects.
    trigger_payload = dict(trigger or {})
    if trigger_type == "gt_change":
        all_interactables = list(gt_obj_names)
        if not all_interactables:
            try:
                snap = STATE.belief_manager.get_snapshot() if STATE.belief_manager else {}
                if isinstance(snap, dict):
                    all_interactables = [str(k) for k in (snap.get("objects") or {}).keys()]
            except Exception:
                all_interactables = []
        if all_interactables:
            trigger_payload["all_interactables"] = all_interactables

    req = {
        "frames": valid_frames,
        "user_text": str(user_text or ""),
        "trigger": trigger_payload,
        "gt_state": gt_state,
        "ts": now
    }
    req["run_id"] = int(STATE.run_id)

    _queue_put_latest(STATE.worker_in_q, req)
    STATE.vlm_busy = True
    STATE.vlm_busy_since = now
    STATE.vlm_busy_trigger = trigger_type

    t = trigger_type
    if t == "user":
        log_line("INFO ", f"Manual: queued for model (images={len(req['frames'])})")
        _update_vlm_status("busy", "user")
    else:
        log_line("INFO ", f"Auto: queued for model (trigger={t}, images={len(req['frames'])})")
        _update_vlm_status("busy", t)
    return True


def on_test_cosmos() -> None:
    try:
        test_cosmos_connection()
    except Exception as exc:
        log_error(str(exc))
        log_line("ERR  ", str(exc))


def on_init_cam() -> None:
    try:
        init_camera()
    except Exception as exc:
        log_error(str(exc))
        log_line("ERR  ", str(exc))


def on_save_frame() -> None:
    try:
        path = save_last_frame()
        log_line("INFO ", f"Saved last camera frame: {path}")
    except Exception as exc:
        log_error(str(exc))
        log_line("ERR  ", str(exc))


def on_initiate_cosmos() -> None:
    """Pre-simulation grounding: capture static frame and initialize belief state with semantic object names."""
    try:
        # Prevent duplicate initialization requests while one is in-flight.
        if getattr(STATE, "grounding_in_progress", False):
            log_line("WARN ", "Grounding already in progress.")
            return
        if _has_pending_inference() and str(getattr(STATE, "vlm_busy_trigger", "") or "") == "grounding":
            log_line("WARN ", "Grounding request already queued.")
            return

        log_line("INFO ", "Initiating Cosmos Grounding...")
        STATE.grounding_complete = False
        STATE.grounding_in_progress = True
        _update_vlm_status("busy", "grounding")
        if STATE.init_status_label is not None:
            try:
                STATE.init_status_label.text = "Initializing..."
                STATE.init_status_label.set_style({"color": 0xFFFFAA00})  # amber
            except Exception:
                pass

        # Acquire a single static frame for grounding.
        async def _capture_and_ground():
            try:
                import numpy as np
                _init_pipeline_once()

                rgb = None

                # Fast path 1: ring buffer already has frames (PLAY is running or was running).
                if STATE.frame_buffer:
                    items = list(STATE.frame_buffer)
                    rgb = _clone_frame(items[-1][1])
                    log_line("INFO ", "Grounding: using latest buffered frame.")

                # Fast path 2: a frame was previously captured (e.g. after Init Camera click).
                elif STATE.last_rgb is not None:
                    rgb = _clone_frame(STATE.last_rgb)
                    log_line("INFO ", "Grounding: using last captured frame (STATE.last_rgb).")

                # Slow path: capture a static frame before PLAY.
                # Run a bounded retry loop to avoid long hangs before PLAY.
                # Each capture attempt is timeout-limited; between attempts we let
                # the Kit loop advance so the renderer can warm up.
                else:
                    log_line("INFO ", "Grounding: no buffered frames — capturing pre-PLAY static frame...")
                    from sensor import capture_rgb_uint8_async

                    def _is_valid_frame(_rgb: Any) -> bool:
                        return bool(
                            _rgb is not None
                            and isinstance(_rgb, np.ndarray)
                            and _rgb.ndim == 3
                            and _rgb.size > 0
                            and int(np.max(_rgb)) > 1
                        )

                    max_attempts = 4
                    for attempt in range(1, max_attempts + 1):
                        try:
                            candidate = await asyncio.wait_for(capture_rgb_uint8_async(), timeout=10.0)
                        except asyncio.TimeoutError:
                            log_line("WARN ", f"Grounding capture attempt {attempt}/{max_attempts} timed out.")
                            candidate = None
                        except Exception as exc:
                            log_line("WARN ", f"Grounding capture attempt {attempt}/{max_attempts} failed: {exc}")
                            candidate = None

                        if _is_valid_frame(candidate):
                            rgb = candidate
                            log_line("INFO ", f"Grounding capture succeeded on attempt {attempt}/{max_attempts}.")
                            break

                        # Fallback probe: latest annotator frame without Replicator step.
                        try:
                            probe = get_latest_rgb_uint8()
                        except Exception:
                            probe = None
                        if _is_valid_frame(probe):
                            rgb = _clone_frame(probe)
                            log_line("INFO ", f"Grounding capture succeeded via latest-frame probe on attempt {attempt}/{max_attempts}.")
                            break

                        await omni.kit.app.get_app().next_update_async()

                    if rgb is None:
                        log_line("WARN ", "Grounding capture did not return a valid frame after retries.")

                # Threshold > 1 (not > 10) so a legitimately dark/night scene is accepted.
                frame_valid = (rgb is not None and isinstance(rgb, np.ndarray)
                               and rgb.ndim == 3 and rgb.size > 0
                               and np.max(rgb) > 1)
                log_line("INFO ", f"Grounding frame: shape={getattr(rgb, 'shape', 'N/A')}, "
                         f"max_pixel={int(np.max(rgb)) if frame_valid else 'N/A'}, valid={frame_valid}")
                if not frame_valid:
                    # Last-resort: the warm-up loop inside capture_rgb_uint8_async exhausted
                    # its steps and still returned black (e.g. renderer not ready).  If there
                    # is a cached frame from a prior capture, use it before failing hard.
                    if STATE.last_rgb is not None and isinstance(STATE.last_rgb, np.ndarray):
                        rgb = _clone_frame(STATE.last_rgb)
                        frame_valid = bool(rgb.ndim == 3 and rgb.size > 0 and np.max(rgb) > 1)
                        if frame_valid:
                            log_line("INFO ", "Grounding fallback: using last cached non-black frame.")

                if not frame_valid:
                    log_line("ERR  ", "Grounding failed: no valid frame captured. Initialization aborted.")
                    STATE.grounding_complete = False
                    STATE.grounding_in_progress = False
                    if STATE.init_status_label is not None:
                        try:
                            STATE.init_status_label.text = "Init failed (no frame)"
                            STATE.init_status_label.set_style({"color": 0xFFCC4444})
                        except Exception:
                            pass
                    _update_vlm_status("idle", "")
                    return

                # Build grounding prompt + prebuilt interactable form.
                interactable_names = _collect_gt_interactable_names(max_names=96)
                init_form = _build_init_interactable_form(interactable_names)
                if interactable_names:
                    log_line(
                        "INFO ",
                        f"Grounding interactables from GT ({len(interactable_names)}): {interactable_names[:12]}",
                    )
                else:
                    log_line("WARN ", "Grounding: no GT interactables found; model will infer from frame.")

                grounding_prompt = (
                    "INITIALIZATION MODE.\n"
                    "Single-frame setup grounding before PLAY.\n"
                    "Fill belief_state_update for all required interactable keys.\n"
                    "Do not remove or rename provided object keys.\n"
                    "Set temporal_change='initial_state'.\n"
                    "Return strict JSON only.\n"
                )

                # Enqueue grounding inference
                frames = [_clone_frame(rgb)]
                preplay_dir = _ensure_preplay_frames_dir()
                try:
                    paths = await asyncio.to_thread(save_sent_frames, "init_grounding", list(frames))
                    if paths:
                        log_line("INFO ", f"Grounding sent frame: {paths[0]}")
                        log_line("INFO ", f"Grounding frames dir: {preplay_dir}")
                except Exception as exc:
                    log_warn(f"Failed to save grounding frame: {exc}")
                queued = _enqueue_inference(
                    frames=frames,
                    user_text=grounding_prompt,
                    trigger={
                        "type": "grounding",
                        "mode": "initialization",
                        "init_interactables": list(interactable_names),
                        "init_belief_form": init_form,
                    },
                    force=True
                )

                if queued:
                    log_line("INFO ", f"Grounding inference queued (images={len(frames)}).")
                else:
                    log_line("WARN ", "Grounding inquiry could not be queued.")
                    STATE.grounding_in_progress = False
                    if STATE.init_status_label is not None:
                        try:
                            STATE.init_status_label.text = "Ready (no init)"
                            STATE.init_status_label.set_style({"color": 0xFF888888})
                        except Exception:
                            pass
                    _sync_vlm_status_indicator()

            except Exception as exc:
                log_error(str(exc))
                log_line("ERR  ", f"Grounding failed: {exc}")
                STATE.grounding_in_progress = False
                if STATE.init_status_label is not None:
                    try:
                        STATE.init_status_label.text = "Ready (no init)"
                        STATE.init_status_label.set_style({"color": 0xFF888888})
                    except Exception:
                        pass
                _sync_vlm_status_indicator()

        # Schedule grounding on the Kit async loop (always use Kit's loop, not the
        # Python default asyncio loop which may be a different, non-running loop).
        task = _get_task_loop().create_task(_capture_and_ground())
        setattr(STATE, "_grounding_task", task)

    except Exception as exc:
        log_error(str(exc))
        log_line("ERR  ", str(exc))


def _diagnose_physics_scene() -> None:
    """Check physics configuration and log diagnostics. Call after PLAY."""
    try:
        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[PHYS_DIAG] No USD stage loaded!", flush=True)
            log_line("WARN ", "Physics diag: no USD stage loaded")
            return

        physics_scenes = []
        dynamic_bodies = []
        kinematic_bodies = []
        colliders = []

        for prim in stage.Traverse():
            if not prim.IsValid():
                continue
            path_str = prim.GetPath().pathString

            # Check for PhysicsScene
            if prim.HasAPI(UsdPhysics.Scene) or prim.IsA(UsdPhysics.Scene):
                physics_scenes.append(path_str)

            # Check for rigid bodies
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rb = UsdPhysics.RigidBodyAPI(prim)
                kinematic_attr = rb.GetKinematicEnabledAttr()
                is_kinematic = False
                if kinematic_attr and kinematic_attr.HasAuthoredValue():
                    is_kinematic = bool(kinematic_attr.Get())
                if is_kinematic:
                    kinematic_bodies.append(path_str)
                else:
                    dynamic_bodies.append(path_str)

            # Check for collision API
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                colliders.append(path_str)

        # Check for gravity in physics scenes
        gravity_info = []
        for scene_path in physics_scenes:
            scene_prim = stage.GetPrimAtPath(scene_path)
            if scene_prim:
                try:
                    scene_api = UsdPhysics.Scene(scene_prim)
                    grav_attr = scene_api.GetGravityDirectionAttr()
                    grav_mag_attr = scene_api.GetGravityMagnitudeAttr()
                    grav_dir = grav_attr.Get() if grav_attr else "N/A"
                    grav_mag = grav_mag_attr.Get() if grav_mag_attr else "N/A"
                    gravity_info.append(f"{scene_path}: dir={grav_dir}, mag={grav_mag}")
                except Exception as exc:
                    gravity_info.append(f"{scene_path}: error reading gravity: {exc}")

        tl = omni.timeline.get_timeline_interface()
        time_now = tl.get_current_time()

        print(f"[PHYS_DIAG] === Physics Scene Diagnostics ===", flush=True)
        print(f"[PHYS_DIAG] PhysicsScene prims: {physics_scenes if physics_scenes else 'NONE FOUND!'}", flush=True)
        print(f"[PHYS_DIAG] Gravity: {gravity_info if gravity_info else 'N/A (no physics scene)'}", flush=True)
        print(f"[PHYS_DIAG] Dynamic rigid bodies ({len(dynamic_bodies)}): {dynamic_bodies[:15]}", flush=True)
        print(f"[PHYS_DIAG] Kinematic rigid bodies ({len(kinematic_bodies)}): {kinematic_bodies[:15]}", flush=True)
        print(f"[PHYS_DIAG] Collision prims: {len(colliders)}", flush=True)
        print(f"[PHYS_DIAG] Timeline: playing={tl.is_playing()}, stopped={tl.is_stopped()}, time={time_now:.4f}s", flush=True)

        log_line("INFO ", f"Physics: {len(physics_scenes)} scene(s), "
                 f"{len(dynamic_bodies)} dynamic, {len(kinematic_bodies)} kinematic bodies")
        if not physics_scenes:
            log_line("WARN ", "NO PhysicsScene prim found! Physics will NOT step. "
                     "Add a PhysicsScene to your USD stage.")
        if not dynamic_bodies:
            log_line("WARN ", "No dynamic rigid bodies found. Nothing will move under physics.")

    except Exception as exc:
        print(f"[PHYS_DIAG] Error: {exc}", flush=True)
        log_line("WARN ", f"Physics diagnostics failed: {exc}")


def _ensure_physics_scene_defaults() -> None:
    """Repair obviously invalid PhysicsScene gravity values in-place.

    Some stage variants show gravity as dir=(0,0,0), mag=-inf on first PLAY, which can lead
    to a non-advancing simulation. This function only patches clearly invalid values.
    """
    try:
        import math
        import omni.usd
        from pxr import Gf, UsdPhysics
        try:
            from pxr import PhysxSchema  # type: ignore
        except Exception:
            PhysxSchema = None

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        patched = 0
        for prim in stage.Traverse():
            if not prim.IsValid():
                continue
            if not (prim.HasAPI(UsdPhysics.Scene) or prim.IsA(UsdPhysics.Scene)):
                continue

            # Patch USD PhysicsScene attrs.
            scene_api = UsdPhysics.Scene(prim)
            dir_attr = scene_api.GetGravityDirectionAttr()
            mag_attr = scene_api.GetGravityMagnitudeAttr()

            cur_dir = dir_attr.Get() if dir_attr else None
            cur_mag = mag_attr.Get() if mag_attr else None

            dir_invalid = True
            try:
                if cur_dir is not None:
                    n = float(cur_dir[0]) ** 2 + float(cur_dir[1]) ** 2 + float(cur_dir[2]) ** 2
                    dir_invalid = n < 1e-8
            except Exception:
                dir_invalid = True

            mag_invalid = True
            try:
                if cur_mag is not None:
                    mag_f = float(cur_mag)
                    mag_invalid = (not math.isfinite(mag_f)) or (mag_f <= 0.0)
            except Exception:
                mag_invalid = True

            if dir_invalid and dir_attr:
                dir_attr.Set(Gf.Vec3f(0.0, 0.0, -1.0))
                patched += 1
            if mag_invalid and mag_attr:
                mag_attr.Set(9.81)
                patched += 1

            # Also patch PhysX scene API when present; some scenes read gravity from PhysX schema.
            if PhysxSchema is not None:
                try:
                    if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
                        physx_api = PhysxSchema.PhysxSceneAPI(prim)
                    else:
                        physx_api = PhysxSchema.PhysxSceneAPI.Apply(prim)

                    p_dir_attr = physx_api.GetGravityDirectionAttr()
                    p_mag_attr = physx_api.GetGravityMagnitudeAttr()
                    p_dir = p_dir_attr.Get() if p_dir_attr else None
                    p_mag = p_mag_attr.Get() if p_mag_attr else None

                    p_dir_invalid = True
                    try:
                        if p_dir is not None:
                            pn = float(p_dir[0]) ** 2 + float(p_dir[1]) ** 2 + float(p_dir[2]) ** 2
                            p_dir_invalid = pn < 1e-8
                    except Exception:
                        p_dir_invalid = True

                    p_mag_invalid = True
                    try:
                        if p_mag is not None:
                            pm = float(p_mag)
                            p_mag_invalid = (not math.isfinite(pm)) or (pm <= 0.0)
                    except Exception:
                        p_mag_invalid = True

                    if p_dir_invalid and p_dir_attr:
                        p_dir_attr.Set(Gf.Vec3f(0.0, 0.0, -1.0))
                        patched += 1
                    if p_mag_invalid and p_mag_attr:
                        p_mag_attr.Set(9.81)
                        patched += 1
                except Exception:
                    # Non-fatal; continue with USD physics attrs.
                    pass

        if patched > 0:
            log_line("INFO ", f"Patched invalid PhysicsScene gravity attrs ({patched} fields).")
    except Exception as exc:
        log_warn(f"PhysicsScene gravity patch skipped: {exc}")


def _ensure_sim_context_playing() -> bool:
    """Best-effort nudge for Isaac Lab SimulationContext when timeline is in PLAY."""
    try:
        from isaaclab.sim import SimulationContext

        sim_ctx = SimulationContext.instance()
        if sim_ctx is None:
            return False
        try:
            sim_ctx.play()
            return True
        except Exception:
            return False
    except Exception:
        return False


def _set_play_simulations(enabled: bool) -> None:
    """Best-effort toggle for Kit's simulation-playback flag."""
    try:
        import carb

        settings = carb.settings.get_settings()
        value = bool(enabled)
        for key in ("/app/player/playSimulations", "/app/player/playSimulation"):
            try:
                settings.set_bool(key, value)
            except Exception:
                try:
                    settings.set(key, value)
                except Exception:
                    pass
    except Exception:
        pass


def _startup_prime_sim_context() -> None:
    """Prime SimulationContext once so first PLAY starts stepping reliably."""
    try:
        from isaaclab.sim import SimulationContext

        sim_ctx = SimulationContext.instance()
        if sim_ctx is None:
            return
        sim_ctx.reset()
        log_line("INFO ", "Startup SimulationContext primed via reset().")
    except Exception as exc:
        log_warn(f"Startup SimulationContext prime skipped: {exc}")


async def _wait_for_sim_time_advance(start_time: float, *, max_updates: int = 120, min_delta: float = 5e-4) -> bool:
    """Wait for timeline simulation time to advance by at least min_delta."""
    tl = omni.timeline.get_timeline_interface()
    app = omni.kit.app.get_app()
    for _ in range(int(max_updates)):
        await app.next_update_async()
        try:
            if tl.is_playing() and (float(tl.get_current_time()) - float(start_time)) >= float(min_delta):
                return True
        except Exception:
            continue
    return False


def _cancel_run_bootstrap_task() -> None:
    task = getattr(STATE, "run_bootstrap_task", None)
    if task is not None:
        try:
            task.cancel()
        except Exception:
            pass
    STATE.run_bootstrap_task = None


async def _physics_monitor_task() -> None:
    """Monitor physics stepping after PLAY by checking timeline time advancement."""
    try:
        tl = omni.timeline.get_timeline_interface()

        # Wait 5 frames for physics to warm up.
        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()

        t0 = tl.get_current_time()

        # Wait 30 more frames (~0.5s at 60fps).
        for _ in range(30):
            if not STATE.playing:
                return
            await omni.kit.app.get_app().next_update_async()

        t1 = tl.get_current_time()
        dt = t1 - t0

        print(f"[PHYS_DIAG] Sim time: t0={t0:.4f}s -> t1={t1:.4f}s  (dt={dt:.4f}s over ~30 frames)", flush=True)
        if abs(dt) < 0.001:
            print(f"[PHYS_DIAG] WARNING: Simulation time NOT advancing! Physics is NOT stepping.", flush=True)
            log_line("WARN ", f"Simulation time not advancing ({t0:.4f}s -> {t1:.4f}s). "
                     "Physics may not be stepping. Check PhysicsScene prim.")
            # First-run recovery path in Isaac Lab: ensure SimulationContext is actively in PLAY.
            _set_play_simulations(True)
            recovered = False
            if _ensure_sim_context_playing():
                for _ in range(20):
                    if not STATE.playing:
                        return
                    await omni.kit.app.get_app().next_update_async()
                t2 = tl.get_current_time()
                dt2 = t2 - t1
                if abs(dt2) >= 0.001:
                    log_line("INFO ", f"Physics recovered after SimulationContext.play() (dt={dt2:.3f}s).")
                    recovered = True
            if not recovered:
                # Last resort for first-run dead start: reset context then play again.
                try:
                    from isaaclab.sim import SimulationContext

                    sim_ctx = SimulationContext.instance()
                    if sim_ctx is not None:
                        sim_ctx.reset()
                        sim_ctx.play()
                        _set_play_simulations(True)
                        for _ in range(20):
                            if not STATE.playing:
                                return
                            await omni.kit.app.get_app().next_update_async()
                        t3 = tl.get_current_time()
                        dt3 = t3 - t1
                        if abs(dt3) >= 0.001:
                            log_line("INFO ", f"Physics recovered after SimulationContext.reset()+play() (dt={dt3:.3f}s).")
                        else:
                            log_line("WARN ", "Physics still not stepping after reset+play recovery attempt.")
                except Exception as exc:
                    log_warn(f"Physics recovery via SimulationContext reset failed: {exc}")
        else:
            print(f"[PHYS_DIAG] OK: Simulation time advancing ({dt:.4f}s over ~30 frames).", flush=True)
            log_line("INFO ", f"Physics stepping OK: sim time advanced {dt:.3f}s")

        # Log object positions if StateMonitor is active.
        sm = STATE.state_monitor
        if sm is not None and sm._tracked_paths:
            poses_t1 = sm.get_current_poses()
            for path, pose in list(poses_t1.items())[:5]:
                print(f"[PHYS_DIAG] Object '{pose.name}': pos=({pose.position[0]:.4f}, {pose.position[1]:.4f}, {pose.position[2]:.4f})", flush=True)

            # Wait another second and compare.
            for _ in range(60):
                if not STATE.playing:
                    return
                await omni.kit.app.get_app().next_update_async()

            poses_t2 = sm.get_current_poses()
            for path, pose2 in list(poses_t2.items())[:5]:
                pose1 = poses_t1.get(path)
                if pose1 is None:
                    continue
                import math as _math
                disp = _math.sqrt(sum((a - b) ** 2 for a, b in zip(pose1.position, pose2.position)))
                print(f"[PHYS_DIAG] Object '{pose2.name}' after +1s: "
                      f"pos=({pose2.position[0]:.4f}, {pose2.position[1]:.4f}, {pose2.position[2]:.4f}), "
                      f"displacement={disp:.6f}m", flush=True)
                if disp < 0.0001:
                    log_line("INFO ", f"Object '{pose2.name}' stationary (disp={disp:.6f}m) — may be at rest")

    except asyncio.CancelledError:
        pass
    except Exception as exc:
        print(f"[PHYS_DIAG] Monitor error: {exc}", flush=True)


def on_play() -> None:
    """User-facing PLAY that starts physics/animation."""
    try:
        tl = omni.timeline.get_timeline_interface()
        print(f"[PLAY] on_play() called. timeline.is_playing()={tl.is_playing()}, is_stopped()={tl.is_stopped()}", flush=True)
        if getattr(STATE, "grounding_in_progress", False):
            log_line("WARN ", "Grounding still in progress — belief state may be empty. Click 'Initiate Cosmos' first for accurate tracking.")
        log_line("PLAY ", "on_play() -> timeline.play()")
        t0 = float(tl.get_current_time())
        _ensure_physics_scene_defaults()
        _set_play_simulations(True)
        tl.play()
        _ensure_sim_context_playing()
        print(f"[PLAY] timeline.play() returned. is_playing()={tl.is_playing()}", flush=True)

        async def _ensure_play_event_received() -> None:
            """Safety net: if Kit drops the timeline PLAY event, start the run ourselves."""
            try:
                app = omni.kit.app.get_app()
                # Give the timeline event a few frames to propagate.
                for _ in range(30):
                    await app.next_update_async()
                    if STATE.playing:
                        return  # _start_run was called by the timeline hook — all good.
                # Timeline PLAY event was lost; start run directly.
                if tl.is_playing() and not STATE.playing:
                    print("[PLAY] Fallback _start_run() (PLAY event not observed)", flush=True)
                    _start_run()
            except Exception:
                pass

        loop = _get_task_loop()
        loop.create_task(_ensure_play_event_received())
    except Exception as exc:
        print(f"[PLAY] ERROR in on_play(): {exc}", flush=True)
        log_error(str(exc))
        log_line("ERR  ", str(exc))


def _reset_episode_runtime(reason: str) -> None:
    """Stop run tasks and reset simulation state to initial pose."""
    print(f"[RESET] _reset_episode_runtime(reason={reason})", flush=True)
    # De-bounce duplicated callbacks (toolbar + UI stop can both fire).
    now = time.time()
    last_ts = float(getattr(STATE, "_last_reset_ts", 0.0))
    if now - last_ts < 0.25:
        print(f"[RESET] Debounced (last reset {now - last_ts:.3f}s ago)", flush=True)
        return
    setattr(STATE, "_last_reset_ts", now)

    _end_run(reason=reason)
    # Invalidate stale in-flight worker replies from previous run.
    STATE.run_id += 1
    _clear_worker_input_queue()
    STATE.vlm_busy = False
    STATE.vlm_busy_since = 0.0
    STATE.vlm_busy_trigger = ""
    STATE.manual_pending = False
    STATE.pause_inquiry_until = 0.0
    STATE.last_auto_infer_ts = 0.0
    _update_vlm_status("idle")
    try:
        STATE.frame_buffer.clear()
    except Exception:
        pass

    try:
        from isaaclab.sim import SimulationContext

        sim_ctx = SimulationContext.instance()
        if sim_ctx is not None:
            # Full reset restores all object states to their initial authored poses.
            sim_ctx.reset()
            log_line("INFO ", "Episode reset via SimulationContext.reset().")
    except Exception as exc:
        log_warn(f"Episode reset not available in this workflow: {exc}")


def on_clear() -> None:
    STATE.log_lines.clear()
    render_log()
    log_info("Cleared logs.")


def _queue_combine_batch(
    command_name: str,
    combines: list[tuple[str, str, str, str]],
) -> bool:
    """Queue and execute a fixed list of combine commands immediately."""
    ma = STATE.magic_assembly
    if ma is None:
        log_line("WARN ", f"/{command_name}: magic_assembly not initialised")
        return True

    from .magic_assembly import AssemblyCommand

    total = len(combines)
    for index, (child, parent, plug, socket) in enumerate(combines, start=1):
        ma.enqueue(AssemblyCommand(
            action="combine",
            child_name=child,
            parent_name=parent,
            plug_name=plug,
            socket_name=socket,
            callback=lambda ok, msg, child=child, parent=parent, plug=plug, socket=socket, index=index, total=total: log_line(
                "INFO " if ok else "WARN ",
                f"/{command_name}[{index}/{total}] "
                f"({child!r} -> {parent!r}, plug={plug!r}, socket={socket!r}) "
                f"{'OK' if ok else 'FAILED: ' + msg}",
            ),
        ))
        log_line(
            "INFO ",
            f"/{command_name} queued[{index}/{total}]: "
            f"{child!r} -> {parent!r} @ plug={plug!r}, socket={socket!r}",
        )

    try:
        n = ma.execute_pending()
        log_line("INFO ", f"/{command_name}: executed {n} pending assembly command(s)")
    except Exception as exc:
        log_line("WARN ", f"/{command_name}: execute_pending failed: {exc}")
    return True


def _resolve_existing_part_name(candidates: List[str]) -> str:
    ma = STATE.magic_assembly
    if ma is None:
        return candidates[0]
    stage = ma._stage_fn()
    if stage is None:
        return candidates[0]
    for name in candidates:
        try:
            if ma._find_prim_path(stage, name) is not None:
                return name
        except Exception:
            continue
    return candidates[0]


def _hub_bolt_part_name(index: int, side: str) -> str:
    side_norm = str(side or "").strip().lower()
    if side_norm not in {"top", "base"}:
        raise ValueError(f"unexpected hub-bolt side: {side!r}")
    return _resolve_existing_part_name(
        [f"M6_Hub_Bolt_{int(index):02d}_{side_norm}", f"M6_Hub_Bolt_{int(index):02d}"]
    )


def _ensure_case_attachment_assets_logged(command_name: str) -> None:
    ma = STATE.magic_assembly
    if ma is None:
        return
    try:
        created = ma.ensure_case_attachment_assets()
        if any(created.values()):
            log_line(
                "INFO ",
                f"/{command_name} ensured case assets "
                f"(bolts={created.get('bolts', 0)}, oils={created.get('oils', 0)}, "
                f"top_sockets={created.get('top_sockets', 0)}, "
                f"base_alias_sockets={created.get('base_alias_sockets', 0)})",
            )
    except Exception as exc:
        log_line("WARN ", f"/{command_name}: failed to ensure case assets: {exc}")


def combine_casing_top() -> bool:
    """Shortcut for assembling the top-side hub covers and hub bolts onto Casing_Top."""
    top_output = _resolve_existing_part_name(["Hub_Cover_Output_Top", "Hub_Cover_Output"])
    top_input = _resolve_existing_part_name(["Hub_Cover_Input_Top", "Hub_Cover_Input"])
    top_small = _resolve_existing_part_name(["Hub_Cover_Small_Top", "Hub_Cover_Small"])
    batch = [
        (top_output, "Casing_Top", "plug_main", "socket_hub_output"),
        (top_input, "Casing_Top", "plug_main", "socket_hub_input"),
        (top_small, "Casing_Top", "plug_main", "socket_hub_small"),
    ]
    batch.extend(
        (
            _hub_bolt_part_name(index, "top"),
            "Casing_Top",
            "plug_main",
            f"socket_bolt_hub_{index}",
        )
        for index in (1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14)
    )
    return _queue_combine_batch("combine_casing_top", batch)


def combine_casing_base() -> bool:
    """Shortcut for installing the three ready shaft subassemblies and base-side covers onto Casing_Base."""
    _ensure_case_attachment_assets_logged("combine_casing_base")
    base_output = _resolve_existing_part_name(["Hub_Cover_Output_Base", "Hub_Cover_Output"])
    base_small_01 = _resolve_existing_part_name(
        ["Hub_Cover_Small_Base_01", "Hub_Cover_Small_Base", "Hub_Cover_Small"]
    )
    base_small_02 = _resolve_existing_part_name(
        ["Hub_Cover_Small_Base_02", "Hub_Cover_Small_Base", "Hub_Cover_Small"]
    )
    return _queue_combine_batch(
        "combine_casing_base",
        [
            ("Transfer_Shaft", "Casing_Base", "plug_main", "socket_gear_transfer"),
            ("Input_Shaft", "Casing_Base", "plug_main", "socket_gear_input"),
            ("Output_Shaft", "Casing_Base", "plug_main", "socket_gear_output"),
            (base_output, "Casing_Base", "plug_main", "socket_hub_output"),
            (base_small_01, "Casing_Base", "plug_main", "socket_hub_small_1"),
            (base_small_02, "Casing_Base", "plug_main", "socket_hub_small_2"),
        ]
        + [
            (
                _hub_bolt_part_name(index, "base"),
                "Casing_Base",
                "plug_main",
                f"socket_bolt_hub_{index}",
            )
            for index in (1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14)
        ],
    )


def combine_base_shafts() -> bool:
    """Shortcut for assembling the three shafts into Casing_Base."""
    return _queue_combine_batch(
        "combine_base_shafts",
        [
            ("Transfer_Shaft", "Casing_Base", "plug_main", "socket_gear_transfer"),
            ("Input_Shaft", "Casing_Base", "plug_main", "socket_gear_input"),
            ("Output_Shaft", "Casing_Base", "plug_main", "socket_gear_output"),
        ],
    )


def combine_bolt_hub() -> bool:
    """Shortcut for assembling the authored M6 hub bolts onto Casing_Top and the casing nuts onto their bolts."""
    ma = STATE.magic_assembly
    if ma is None:
        log_line("WARN ", "/combine_bolt_hub: magic_assembly not initialised")
        return True

    try:
        created = ma.ensure_extra_hub_bolt_assets()
        if created.get("bolts", 0) or created.get("sockets", 0):
            log_line(
                "INFO ",
                "/combine_bolt_hub ensured extra assets "
                f"(bolts={created.get('bolts', 0)}, sockets={created.get('sockets', 0)})",
            )
    except Exception as exc:
        log_line("WARN ", f"/combine_bolt_hub: failed to ensure extra assets: {exc}")

    return _queue_combine_batch(
        "combine_bolt_hub",
        [
            *[
                (
                    _hub_bolt_part_name(index, "top"),
                    "Casing_Top",
                    "plug_main",
                    f"socket_bolt_hub_{index}",
                )
                for index in (1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14)
            ],
            ("M10_Casing_Nut_01", "M10_Casing_Bolt_01", "plug_main", None),
            ("M10_Casing_Nut_02", "M10_Casing_Bolt_02", "plug_main", None),
            ("M10_Casing_Nut_03", "M10_Casing_Bolt_03", "plug_main", None),
            ("M10_Casing_Nut_04", "M10_Casing_Bolt_04", "plug_main", None),
            ("M10_Casing_Nut_05", "M10_Casing_Bolt_05", "plug_main", None),
            ("M10_Casing_Nut_06", "M10_Casing_Bolt_06", "plug_main", None),
        ],
    )


def combine_casing_bolt() -> bool:
    """Shortcut for assembling the six M10 casing bolts from the Casing_Top side and their matching casing nuts."""
    _ensure_case_attachment_assets_logged("combine_casing_bolt")
    return _queue_combine_batch(
        "combine_casing_bolt",
        [
            ("M10_Casing_Bolt_01", "Casing_Top", "plug_main", "socket_bolt_casing_1"),
            ("M10_Casing_Bolt_02", "Casing_Top", "plug_main", "socket_bolt_casing_2"),
            ("M10_Casing_Bolt_03", "Casing_Top", "plug_main", "socket_bolt_casing_3"),
            ("M10_Casing_Bolt_04", "Casing_Top", "plug_main", "socket_bolt_casing_4"),
            ("M10_Casing_Bolt_05", "Casing_Top", "plug_main", "socket_bolt_casing_5"),
            ("M10_Casing_Bolt_06", "Casing_Top", "plug_main", "socket_bolt_casing_6"),
            ("M10_Casing_Nut_01", "M10_Casing_Bolt_01", "plug_main", None),
            ("M10_Casing_Nut_02", "M10_Casing_Bolt_02", "plug_main", None),
            ("M10_Casing_Nut_03", "M10_Casing_Bolt_03", "plug_main", None),
            ("M10_Casing_Nut_04", "M10_Casing_Bolt_04", "plug_main", None),
            ("M10_Casing_Nut_05", "M10_Casing_Bolt_05", "plug_main", None),
            ("M10_Casing_Nut_06", "M10_Casing_Bolt_06", "plug_main", None),
        ],
    )


def combine_accessories() -> bool:
    """Shortcut for oil-level indicators and breather plug on the casing base."""
    _ensure_case_attachment_assets_logged("combine_accessories")
    return _queue_combine_batch(
        "combine_accessories",
        [
            ("Oil_Level_Indicator", "Casing_Base", "plug_main", "socket_oil_1"),
            ("Oil_Level_Indicator_02", "Casing_Base", "plug_main", "socket_oil_2"),
            ("Breather_Plug", "Casing_Base", "plug_main", "socket_breather"),
        ],
    )


def _combine_bearing_subassembly(command_name: str) -> bool:
    """Queue a hardcoded bearing subassembly batch when explicitly enabled."""
    from .magic_assembly import BEARING_SUBASSEMBLY_BATCHES

    combines = BEARING_SUBASSEMBLY_BATCHES.get(command_name, [])
    if not combines:
        log_line("WARN ", f"/{command_name} disabled after revert.")
        return True
    return _queue_combine_batch(command_name, combines)


def combine_input_shaft() -> bool:
    """Shortcut for the input-shaft bearing subassembly."""
    return _combine_bearing_subassembly("combine_input_shaft")


def combine_transfer_shaft() -> bool:
    """Shortcut for the transfer-shaft bearing subassembly."""
    return _combine_bearing_subassembly("combine_transfer_shaft")


def combine_output_shaft() -> bool:
    """Shortcut for the output-shaft bearing subassembly."""
    return _combine_bearing_subassembly("combine_output_shaft")


def _try_slash_command(text: str) -> bool:
    """Parse and execute a /command entered in the text box.

    Supported commands  (categorization per docs/commands_description.md)
    ---------------------------------------------------------------------
    Basic — Combination:
      /combine("partA", "partB", "plug_name", "socket_name")
      /separate("part_name")

    Basic — Movement:
      /move("part_name", "point")               — teleport part to `point`
      /move("part_name", "src", "dest")         — teleport from src to dest
                                                  (src is sanity-checked vs current)
        Points: named location ("center", "side", "table", "floor", "current", ...),
                a literal (x, y, z) tuple, or "current".
      /pick_up("part_name", height_m=0.15)      — translate part +Z
      /put_down("part_name", drop_m=0.15)       — translate part -Z then auto-seat on table
      /hover("part_name", "table"|"floor"|z, margin=0.005)
                                                — drop part along Z so its bottom rests on a surface
      /flip("part_name", "x"|"y")               — rotate part 180° about world X or Y, then auto-seat
      /upright("part_name", "x"|"y")            — rotate part 90° about world X (default) or Y, then auto-seat
      /rotate("part_name", angle_deg=45, "z")   — rotate part by angle about axis

    Basic — Operational:
      /undo                — reverse the most recent recorded action
      /reset               — restore all top-level scene parts to startup parent/pose
      /assemblies          — list all active part attachments

    Shortcuts:
      /focus("part_name")  — shortcut for /move(part, "current", "center")
      /unfocus("part_name") — inverse of /focus (restores pre-focus pose)
      /flip_casing_base    — rotate Casing_Base 180° so inside faces upward
    /combine_casing_top  — combine the top-side hub covers and M6 hub bolts onto Casing_Top
    /combine_casing_base — install the 3 shaft subassemblies and 3 base-side hub covers onto Casing_Base
    /combine_base_shafts — combine the 3 shafts into Casing_Base
    /combine_bolt_hub    — combine the authored M6 hub bolts onto Casing_Top
    /combine_casing_bolt — combine the 6 M10 casing bolts onto Casing_Top and attach their 6 casing nuts
    /combine_accessories — combine oil indicators and breather plug onto Casing_Base
    /combine_output_shaft — combine the two output-shaft bearings with hardcoded offsets
    Returns True if the text was recognised and handled (so on_send()
    should NOT forward it to the VLM), False otherwise.
    """
    import ast, re

    text = text.strip()
    if not text.startswith("/"):
        return False

    try:
        _ensure_magic_assembly_once()
    except Exception as exc:
        log_line("WARN ", f"/command: magic_assembly init failed: {exc}")

    if re.fullmatch(r"/reset\s*", text, re.IGNORECASE):
        _reset_episode_runtime("slash_reset")
        _restore_scene_to_initial_snapshot()
        return True

    # ── /assemblies ──────────────────────────────────────────────────────
    if re.fullmatch(r"/assemblies\s*", text, re.IGNORECASE):
        ma = STATE.magic_assembly
        if ma is None:
            log_line("WARN ", "/assemblies: magic_assembly not initialised")
            return True
        pairs = ma.list_assemblies()
        if pairs:
            log_line("INFO ", f"/assemblies ({len(pairs)} active):")
            for child, parent in pairs:
                log_line("INFO ", f"  {child}  →  {parent}")
        else:
            log_line("INFO ", "/assemblies: no active attachments")
        return True

    # ── /undo ──────────────────────────────────────────────────────────────
    if re.fullmatch(r"/undo\s*", text, re.IGNORECASE):
        ma = STATE.magic_assembly
        if ma is None:
            log_line("WARN ", "/undo: magic_assembly not initialised")
            return True
        ok = ma.undo()
        log_line("INFO " if ok else "WARN ", f"/undo: {'OK' if ok else 'nothing to undo or failed'}")
        return True

    # ── /flip_casing_base ──────────────────────────────────────────────────
    if re.fullmatch(r"/flip_casing_base\s*", text, re.IGNORECASE):
        ma = STATE.magic_assembly
        if ma is None:
            log_line("WARN ", "/flip_casing_base: magic_assembly not initialised")
            return True
        ok = ma.flip_casing_base()
        log_line("INFO " if ok else "WARN ", f"/flip_casing_base: {'OK' if ok else 'FAILED'}")
        return True

    # ── sequence runner commands ───────────────────────────────────────────
    if re.fullmatch(r"/sequence_status\s*", text, re.IGNORECASE):
        _ensure_sequence_runner_once()
        sr = STATE.sequence_runner
        if sr is None:
            log_line("WARN ", "/sequence_status: runner not available")
        else:
            log_line("INFO ", f"/sequence_status: {sr.format_status()}")
        return True

    if re.fullmatch(r"/step_next\s*", text, re.IGNORECASE):
        _ensure_sequence_runner_once()
        sr = STATE.sequence_runner
        if sr is None:
            log_line("WARN ", "/step_next: runner not available")
        else:
            result = _run_with_pose_pause(text, lambda: sr.step_next())
            if sr.pending_pose is not None:
                pass  # paused for a pose answer; prompt already logged
            elif result is None:
                log_line("INFO ", "/step_next: sequence complete (or not started)")
            else:
                status = "OK" if result.success else f"FAILED — {result.failure_reason}"
                log_line("INFO " if result.success else "WARN ",
                         f"/step_next: [{result.step_id}] {result.human_label} — {status}")
        return True

    if re.fullmatch(r"/step_prev\s*", text, re.IGNORECASE):
        _ensure_sequence_runner_once()
        sr = STATE.sequence_runner
        if sr is None:
            log_line("WARN ", "/step_prev: runner not available")
        else:
            ok = sr.step_prev()
            log_line("INFO " if ok else "WARN ", f"/step_prev: {'OK' if ok else 'FAILED'}")
        return True

    if re.fullmatch(r"/validate_state\s*", text, re.IGNORECASE):
        _ensure_sequence_runner_once()
        sr = STATE.sequence_runner
        if sr is None:
            log_line("WARN ", "/validate_state: runner not available")
        else:
            sr.validate_state()  # logs results internally
        return True

    if re.fullmatch(r"/reset_sequence\s*", text, re.IGNORECASE):
        _ensure_sequence_runner_once()
        sr = STATE.sequence_runner
        if sr is None:
            log_line("WARN ", "/reset_sequence: runner not available")
        else:
            sr.reset()
            log_line("INFO ", "/reset_sequence: task state cleared")
        return True

    m_inst = re.fullmatch(r"/load_instance\s*\(\s*['\"]([\w./-]+)['\"]\s*\)\s*", text, re.IGNORECASE)
    if m_inst:
        _load_instance_runner(m_inst.group(1))
        return True

    m_run = re.fullmatch(r"/run_sequence(?:\s*\(\s*(['\"]?)(\w*)\1\s*\))?\s*", text, re.IGNORECASE)
    if m_run:
        _ensure_sequence_runner_once()
        sr = STATE.sequence_runner
        if sr is None:
            log_line("WARN ", "/run_sequence: runner not available")
        else:
            n = _run_with_pose_pause(text, lambda: sr.run_sequence(delay_s=0.3))
            if sr.pending_pose is None:
                log_line("INFO ", f"/run_sequence: {n} steps completed")
        return True

    m_step = re.fullmatch(r"/run_step\s*\(\s*['\"]([^'\"]+)['\"]\s*\)\s*", text, re.IGNORECASE)
    if m_step:
        _ensure_sequence_runner_once()
        sr = STATE.sequence_runner
        step_id = m_step.group(1)
        if sr is None:
            log_line("WARN ", f"/run_step: runner not available")
        else:
            result = _run_with_pose_pause(text, lambda: sr.jump_to_step(step_id))
            if sr.pending_pose is not None:
                pass  # paused for a pose answer; prompt already logged
            elif result is None:
                log_line("WARN ", f"/run_step({step_id!r}): step not found")
            else:
                status = "OK" if result.success else f"FAILED — {result.failure_reason}"
                log_line("INFO " if result.success else "WARN ",
                         f"/run_step: [{result.step_id}] {result.human_label} — {status}")
        return True

    # ── DAG queries: ready steps, streams, groups ──────────────────────────
    m_ready = re.fullmatch(
        r"/ready(?:\s*\(\s*(?:stream\s*=\s*)?['\"]?(\w*)['\"]?\s*\))?\s*",
        text, re.IGNORECASE)
    if m_ready:
        _ensure_sequence_runner_once()
        sr = STATE.sequence_runner
        if sr is None:
            log_line("WARN ", "/ready: runner not available")
        else:
            stream_filter = m_ready.group(1) or None
            ready = sr.get_ready_steps(stream=stream_filter)
            suffix = f" (stream={stream_filter!r})" if stream_filter else ""
            if not ready:
                log_line("INFO ", f"/ready{suffix}: no steps ready")
            else:
                log_line("INFO ", f"/ready{suffix}: {len(ready)} step(s) ready")
                for sid in ready[:20]:
                    s_stream = sr.step_stream(sid) or "-"
                    s_group = sr.step_group(sid) or "-"
                    log_line("INFO ", f"  [{sid}] stream={s_stream} group={s_group}")
                if len(ready) > 20:
                    log_line("INFO ", f"  ...and {len(ready) - 20} more")
        return True

    if re.fullmatch(r"/streams\s*", text, re.IGNORECASE):
        _ensure_sequence_runner_once()
        sr = STATE.sequence_runner
        if sr is None:
            log_line("WARN ", "/streams: runner not available")
        else:
            for stream_id in sr._streams:
                p = sr.stream_progress(stream_id)
                par = ", ".join(p.get("can_parallel_with", [])) or "(none)"
                log_line("INFO ",
                         f"  [{stream_id}] {p['completed']}/{p['total']} done, "
                         f"{p['ready']} ready  ||  parallel-with: {par}")
        return True

    if re.fullmatch(r"/groups\s*", text, re.IGNORECASE):
        _ensure_sequence_runner_once()
        sr = STATE.sequence_runner
        if sr is None:
            log_line("WARN ", "/groups: runner not available")
        else:
            for gid, spec in sr._groups.items():
                log_line("INFO ", f"  [{gid}] stream={spec.get('stream', '-')}  "
                                  f"members={len(spec.get('members', []))}  "
                                  f"-- {spec.get('description', '')}")
        return True

    m_runstream = re.fullmatch(r"/run_stream\s*\(\s*['\"]([^'\"]+)['\"]\s*\)\s*", text, re.IGNORECASE)
    if m_runstream:
        _ensure_sequence_runner_once()
        sr = STATE.sequence_runner
        sid = m_runstream.group(1)
        if sr is None:
            log_line("WARN ", "/run_stream: runner not available")
        else:
            results = _run_with_pose_pause(text, lambda: sr.run_stream(sid, delay_s=0.2))
            if sr.pending_pose is None:
                ok = sum(1 for r in results if r.success)
                log_line("INFO ", f"/run_stream({sid!r}): {ok}/{len(results)} succeeded")
        return True

    m_rungroup = re.fullmatch(r"/run_group\s*\(\s*['\"]([^'\"]+)['\"]\s*\)\s*", text, re.IGNORECASE)
    if m_rungroup:
        _ensure_sequence_runner_once()
        sr = STATE.sequence_runner
        gid = m_rungroup.group(1)
        if sr is None:
            log_line("WARN ", "/run_group: runner not available")
        else:
            results = _run_with_pose_pause(text, lambda: sr.run_group(gid, delay_s=0.1))
            if sr.pending_pose is None:
                ok = sum(1 for r in results if r.success)
                log_line("INFO ", f"/run_group({gid!r}): {ok}/{len(results)} succeeded")
        return True

    # ── batch combine shortcuts ────────────────────────────────────────────
    if re.fullmatch(r"/combine_casing_top\s*", text, re.IGNORECASE):
        return combine_casing_top()

    if re.fullmatch(r"/combine_casing_base\s*", text, re.IGNORECASE):
        return combine_casing_base()

    if re.fullmatch(r"/combine_base_shafts\s*", text, re.IGNORECASE):
        return combine_base_shafts()

    if re.fullmatch(r"/combine_bolt_hub\s*", text, re.IGNORECASE):
        return combine_bolt_hub()

    if re.fullmatch(r"/combine_casing_bolt\s*", text, re.IGNORECASE):
        return combine_casing_bolt()

    if re.fullmatch(r"/combine_accessories\s*", text, re.IGNORECASE):
        return combine_accessories()

    if re.fullmatch(r"/combine_input_shaft\s*", text, re.IGNORECASE):
        return combine_input_shaft()

    if re.fullmatch(r"/combine_transfer_shaft\s*", text, re.IGNORECASE):
        return combine_transfer_shaft()

    if re.fullmatch(r"/combine_output_shaft\s*", text, re.IGNORECASE):
        return combine_output_shaft()

    # ── /combine or /separate ─────────────────────────────────────────────
    m = re.fullmatch(r"/(\w+)\s*\((.*)\)\s*", text, re.DOTALL)
    if not m:
        # Unrecognised slash command — warn but still intercept (don't hit VLM).
        log_line(
            "WARN ",
            f"Unknown /command: {text!r}  "
            f"(sequence: /run_sequence, /step_next, /step_prev, /run_step('id'), "
            f"/sequence_status, /validate_state, /reset_sequence, "
            f"/load_instance('variant')  |  "
            f"DAG: /ready, /ready('stream_id'), /streams, /groups, "
            f"/run_stream('id'), /run_group('id')  |  "
            f"assembly: /combine, /separate, /focus, /reset, /assemblies, "
            f"/combine_casing_top, /combine_casing_base, /combine_base_shafts, "
            f"/combine_bolt_hub, /combine_casing_bolt, /combine_accessories, "
            f"/combine_output_shaft)",
        )
        return True

    cmd_name = m.group(1).lower()
    args_src  = m.group(2).strip()

    # Safely parse the argument list with ast — no eval().
    try:
        args = ast.literal_eval(f"[{args_src}]")
    except Exception as exc:
        log_line("WARN ", f"/command parse error: {exc}  — args must be quoted strings")
        return True

    ma = STATE.magic_assembly
    if ma is None:
        log_line("WARN ", "/command: magic_assembly not initialised")
        return True

    if cmd_name == "combine":
        if len(args) != 4:
            log_line("WARN ", "/combine requires 4 arguments: partA, partB, plug, socket")
            return True
        child = str(args[0]).strip()
        parent = str(args[1]).strip()
        plug = str(args[2]).strip()
        socket = str(args[3]).strip()
        if not child or not parent or not plug or not socket:
            log_line("WARN ", "/combine arguments cannot be empty")
            return True
        if parent in {"Casing_Top", "Casing_Base"}:
            try:
                created = ma.ensure_case_attachment_assets()
                if any(created.values()):
                    log_line(
                        "INFO ",
                        f"/combine prep: created case assets {created}",
                    )
            except Exception as exc:
                log_line("WARN ", f"/combine prep failed: {exc}")
        from .magic_assembly import AssemblyCommand
        ma.enqueue(AssemblyCommand(
            action="combine",
            child_name=child,
            parent_name=parent,
            plug_name=plug,
            socket_name=socket,
            callback=lambda ok, msg: log_line(
                "INFO " if ok else "WARN ",
                f"/combine({child!r} → {parent!r}, plug={plug!r}, socket={socket!r}) "
                f"{'OK' if ok else 'FAILED: ' + msg}",
            ),
        ))
        log_line(
            "INFO ",
            f"/combine queued: {child!r} → {parent!r} @ plug={plug!r}, socket={socket!r}",
        )

    elif cmd_name == "separate":
        if len(args) < 1:
            log_line("WARN ", "/separate requires 1 argument: part_name")
            return True
        part = str(args[0])
        from .magic_assembly import AssemblyCommand
        ma.enqueue(AssemblyCommand(
            action="separate",
            part_name=part,
            callback=lambda ok, msg: log_line(
                "INFO " if ok else "WARN ",
                f"/separate({part!r}) {'OK' if ok else 'FAILED: ' + msg}",
            ),
        ))
        log_line("INFO ", f"/separate queued: {part!r}")

    elif cmd_name == "focus":
        if len(args) < 1:
            log_line("WARN ", "/focus requires 1 argument: part_name")
            return True
        part = str(args[0])
        from .magic_assembly import AssemblyCommand
        ma.enqueue(AssemblyCommand(
            action="focus",
            part_name=part,
            callback=lambda ok, msg: log_line(
                "INFO " if ok else "WARN ",
                f"/focus({part!r}) {'OK' if ok else 'FAILED: ' + msg}",
            ),
        ))
        log_line("INFO ", f"/focus queued: {part!r}")

    elif cmd_name == "unfocus":
        if len(args) < 1:
            log_line("WARN ", "/unfocus requires 1 argument: part_name")
            return True
        part = str(args[0])
        ok = ma.unfocus(part)
        log_line("INFO " if ok else "WARN ", f"/unfocus({part!r}) {'OK' if ok else 'FAILED'}")
        return True

    elif cmd_name == "move":
        if len(args) < 2:
            log_line(
                "WARN ",
                "/move requires part_name and at least 1 point. "
                "Usage: /move(\"part\", \"dest\") or /move(\"part\", \"src\", \"dest\"). "
                "Points: named location, (x,y,z) tuple, or \"current\"."
            )
            return True
        part = str(args[0])
        if len(args) == 2:
            point_1, point_2 = args[1], None
        else:
            point_1, point_2 = args[1], args[2]
        ok = ma.move(part, point_1, point_2)
        log_line(
            "INFO " if ok else "WARN ",
            f"/move({part!r}, {point_1!r}"
            + (f", {point_2!r}" if point_2 is not None else "")
            + f") {'OK' if ok else 'FAILED'}",
        )
        return True

    elif cmd_name == "upright":
        if len(args) < 1:
            log_line("WARN ", "/upright requires 1 argument: part_name (optional 2nd: axis 'x' or 'y')")
            return True
        part = str(args[0])
        axis = str(args[1]) if len(args) >= 2 else "x"
        # Auto-seat onto the table afterwards: standing a flat part changes its
        # lowest point, so re-settle it on the surface (folded into one undo).
        ok = ma.upright(part, axis=axis, seat=True)
        log_line("INFO " if ok else "WARN ",
                 f"/upright({part!r}, axis={axis!r}, seat) {'OK' if ok else 'FAILED'}")
        return True

    elif cmd_name == "flip":
        if len(args) < 1:
            log_line("WARN ", "/flip requires 1 argument: part_name (optional 2nd: axis 'x' or 'y')")
            return True
        part = str(args[0])
        axis = str(args[1]) if len(args) >= 2 else "x"
        # Auto-seat onto the table afterwards: flipping changes which face is
        # down, so re-settle it on the surface (folded into one undo).
        ok = ma.flip(part, axis=axis, seat=True)
        log_line("INFO " if ok else "WARN ",
                 f"/flip({part!r}, axis={axis!r}, seat) {'OK' if ok else 'FAILED'}")
        return True

    elif cmd_name == "rotate":
        if len(args) < 1:
            log_line("WARN ",
                     "/rotate requires 1 argument: part_name (optional: angle_deg=45, axis='z')")
            return True
        part = str(args[0])
        try:
            angle = float(args[1]) if len(args) >= 2 else 45.0
        except (TypeError, ValueError):
            log_line("WARN ", f"/rotate: angle must be a number, got {args[1]!r}")
            return True
        axis = str(args[2]) if len(args) >= 3 else "z"
        ok = ma.rotate(part, angle_deg=angle, axis=axis)
        log_line("INFO " if ok else "WARN ",
                 f"/rotate({part!r}, {angle}°, axis={axis!r}) {'OK' if ok else 'FAILED'}")
        return True

    elif cmd_name == "pick_up":
        if len(args) < 1:
            log_line("WARN ", "/pick_up requires 1 argument: part_name (optional 2nd: height_m=0.15)")
            return True
        part = str(args[0])
        try:
            height = float(args[1]) if len(args) >= 2 else 0.15
        except (TypeError, ValueError):
            log_line("WARN ", f"/pick_up: height must be a number, got {args[1]!r}")
            return True
        ok = ma.pick_up(part, height=height)
        log_line("INFO " if ok else "WARN ",
                 f"/pick_up({part!r}, +{height}m) {'OK' if ok else 'FAILED'}")
        return True

    elif cmd_name == "put_down":
        if len(args) < 1:
            log_line("WARN ", "/put_down requires 1 argument: part_name (optional 2nd: drop_m=0.15)")
            return True
        part = str(args[0])
        try:
            drop = float(args[1]) if len(args) >= 2 else 0.15
        except (TypeError, ValueError):
            log_line("WARN ", f"/put_down: drop must be a number, got {args[1]!r}")
            return True
        # Auto-seat onto the table afterwards so the part rests exactly on the
        # surface regardless of drop distance (folded into one undo).
        ok = ma.put_down(part, drop=drop, seat=True)
        log_line("INFO " if ok else "WARN ",
                 f"/put_down({part!r}, -{drop}m, seat) {'OK' if ok else 'FAILED'}")
        return True

    elif cmd_name == "hover":
        if len(args) < 1:
            log_line("WARN ", "/hover requires 1 argument: part_name "
                              "(optional 2nd: surface='table'|'floor'|z, 3rd: margin=0.005)")
            return True
        part = str(args[0])
        surface = args[1] if len(args) >= 2 else "table"
        try:
            margin = float(args[2]) if len(args) >= 3 else 0.005
        except (TypeError, ValueError):
            log_line("WARN ", f"/hover: margin must be a number, got {args[2]!r}")
            return True
        ok = ma.hover(part, surface=surface, margin=margin)
        log_line("INFO " if ok else "WARN ",
                 f"/hover({part!r}, surface={surface!r}, margin={margin}m) {'OK' if ok else 'FAILED'}")
        return True

    else:
        log_line(
            "WARN ",
            f"Unknown /command: /{cmd_name}  "
            f"(combination: /combine, /separate  |  "
            f"movement: /move, /pick_up, /put_down, /hover, /flip, /upright, /rotate  |  "
            f"operational: /undo, /reset, /assemblies  |  "
            f"shortcuts: /focus, /unfocus, /flip_casing_base, /combine_casing_top, "
            f"/combine_casing_base, /combine_base_shafts, /combine_bolt_hub, "
            f"/combine_casing_bolt, /combine_accessories)",
        )
        return True

    # Slash commands run on the main Kit thread, so we can drain the queue
    # immediately — no need to wait for PLAY / _robot_update_loop.
    try:
        n = ma.execute_pending()
        log_line("INFO ", f"/command: executed {n} pending assembly command(s)")
    except Exception as exc:
        log_line("WARN ", f"/command: execute_pending failed: {exc}")

    return True


def on_send() -> None:
    """Queue a user message for the cognition worker (non-blocking)."""
    user_text = (STATE.input_model.as_string or "").strip() if STATE.input_model else ""
    if not user_text:
        return

    # A paused check_pose takes priority: interpret this input as the Y/N answer
    # and resume the command that paused (see _ui_pose_resolver). Anything that
    # isn't a yes/no answer is rejected so the operator must resolve the pose
    # before issuing other commands.
    sr = STATE.sequence_runner
    if sr is not None and sr.pending_pose is not None:
        pending = sr.pending_pose
        answer = _parse_pose_answer(user_text)
        if STATE.input_model:
            STATE.input_model.set_value("")
        if answer is None:
            log_line("WARN ", "Pose check pending — please reply Y (yes) or N (no).")
            return
        log_line("YOU  ", user_text)
        sr.answer_pose(answer)
        log_line("INFO ", f"[POSE] '{pending.get('child')}' -> "
                          + ("correct" if answer
                             else "INCORRECT — conditional flip will fire"))
        resume_cmd = STATE.pose_resume_cmd
        STATE.pose_resume_cmd = None
        if resume_cmd:
            _try_slash_command(resume_cmd)
        return

    # Handle /slash-commands locally — never forward them to the VLM.
    if user_text.startswith("/"):
        log_line("YOU  ", user_text)
        if STATE.input_model:
            STATE.input_model.set_value("")
        _try_slash_command(user_text)
        return

    if STATE.manual_pending:
        log_line("WARN ", "Manual inquiry already in-flight; please wait.")
        _sync_vlm_status_indicator()
        return

    log_line("YOU  ", user_text)
    if STATE.input_model:
        STATE.input_model.set_value("")

    attach = True
    if STATE.attach_latest_model is not None:
        try:
            attach = bool(STATE.attach_latest_model.as_bool)
        except Exception:
            attach = True

    async def _send_manual_with_frames(frames_to_send: List[Any]) -> None:
        # Save frames used for manual prompts for traceability (no blocking on disk I/O).
        if frames_to_send:
            try:
                paths = await asyncio.to_thread(save_sent_frames, "manual", list(frames_to_send))
                if paths:
                    log_line("INFO ", f"Manual sent frames: {', '.join(paths)}")
            except Exception as exc:
                log_warn(f"Failed to save manual frames: {exc}")

        # Block auto inquiries until the manual response arrives.
        STATE.manual_pending = True
        STATE.pause_inquiry_until = time.time() + 120.0  # released when manual response arrives
        queued = _enqueue_inference(frames=frames_to_send, user_text=user_text, trigger={"type": "user"}, force=True)
        if not queued:
            STATE.manual_pending = False
            STATE.pause_inquiry_until = 0.0
            log_line("WARN ", "Manual inquiry rejected: model became busy before enqueue.")
            _sync_vlm_status_indicator()
            return

    frames: List[Any] = []
    manual_frame_count = max(2, int(INQUIRY_FRAME_COUNT))
    if attach:
        # Manual prompts use the latest buffered frames so the model can observe the current scene.
        frames = _get_latest_buffer_frames(manual_frame_count)
        if frames:
            log_line("INFO ", f"Manual using {len(frames)} latest buffered frame(s).")

    loop = _get_task_loop()

    if attach and not frames:
        # On STOP/reset the run buffer is intentionally cleared. Capture a fresh frame
        # so manual questions still use current visual context instead of text-only memory.
        log_line("INFO ", "Manual prompt has no buffered frames; capturing a fresh frame.")

        async def _capture_then_send() -> None:
            live_frames: List[Any] = []
            try:
                rgb = await capture_rgb_uint8_async()
                if rgb is not None:
                    live_frames = [_clone_frame(rgb)]
                    log_line("INFO ", "Manual captured a fresh frame (images=1).")
            except Exception as exc:
                log_warn(f"Manual fresh-frame capture failed: {exc}")

            if not live_frames and STATE.last_rgb is not None:
                live_frames = [_clone_frame(STATE.last_rgb)]
                log_line("INFO ", "Manual fallback to latest cached camera frame (images=1).")

            if not live_frames:
                log_line("WARN ", "Manual prompt has no visual frame; sending text-only.")

            await _send_manual_with_frames(live_frames)

        loop.create_task(_capture_then_send())
        return

    loop.create_task(_send_manual_with_frames(frames))


def on_step_next() -> None:
    """Shortcut button: send the /step_next slash command (same path as typing it)."""
    cmd = "/step_next"
    log_line("YOU  ", cmd)
    _try_slash_command(cmd)


def _bind_input_enter_to_send(field) -> None:
    """Best-effort: map Enter in the command box to the same path as the Send button."""

    def _submit(*_args, **_kwargs) -> None:
        try:
            on_send()
        except Exception as exc:
            log_warn(f"Enter-to-send failed: {exc}")

    owners = [STATE.input_model, getattr(field, "model", None), field]
    method_names = (
        "add_end_edit_fn",
        "set_end_edit_fn",
        "subscribe_end_edit_fn",
        "set_on_end_edit_fn",
        "add_submit_fn",
        "set_submit_fn",
        "add_accepted_fn",
        "set_accepted_fn",
    )

    for owner in owners:
        if owner is None:
            continue
        for method_name in method_names:
            fn = getattr(owner, method_name, None)
            if not callable(fn):
                continue
            try:
                hook = fn(_submit)
                if hook is not None:
                    STATE._kit_subs.append(hook)
                setattr(STATE, "_input_enter_binding", f"{owner.__class__.__name__}.{method_name}")
                return
            except Exception:
                continue

    log_warn("Enter-to-send shortcut unavailable in this Isaac build; use the Send button.")


async def _capture_loop() -> None:
    """Capture RGB frames into the RAM ring buffer (does not write to disk).

    Rate-limited by wall-clock interval (1/CAPTURE_FPS).
    Duplicate frames are rejected via a pixel fingerprint check rather than
    sim-time gating, because Isaac Sim's timeline current_time can wrap or
    go backwards during bootstrap and early physics warmup.
    """
    interval = 1.0 / max(0.1, float(CAPTURE_FPS))
    count = 0
    skipped_static = 0
    last_log = time.time()
    tl = omni.timeline.get_timeline_interface()
    last_fingerprint = None
    wall_start = time.time()
    # Brief warmup: skip the very first 0.3s to let the renderer produce a
    # valid first frame after PLAY.
    warmup_sec = 0.3
    print(f"[CAPTURE_LOOP] Starting. interval={interval:.3f}s, CAPTURE_FPS={CAPTURE_FPS}, STATE.playing={STATE.playing}", flush=True)
    iteration = 0
    while STATE.playing:
        iteration += 1
        try:
            # Do not trigger Replicator stepping while timeline is stopped/paused.
            if not tl.is_playing():
                await omni.kit.app.get_app().next_update_async()
                continue

            # Brief warmup to let first rendered frame stabilize.
            if (time.time() - wall_start) < warmup_sec:
                await asyncio.sleep(0.05)
                continue

            if iteration <= 3:
                print(f"[CAPTURE_LOOP] iter={iteration} calling capture_rgb_uint8_async()...", flush=True)
            rgb = await capture_rgb_uint8_async()
            ts = time.time()
            if iteration <= 3:
                print(f"[CAPTURE_LOOP] iter={iteration} got frame shape={rgb.shape if rgb is not None else None}", flush=True)
            if rgb is None:
                await asyncio.sleep(interval)
                continue
            # Freeze the frame in RAM; some backends reuse the same underlying buffer.
            try:
                frozen = rgb.copy()
            except Exception:
                frozen = rgb

            # Skip visually identical frames (pixel fingerprint dedup).
            try:
                sample = frozen[::32, ::32, :]
                fingerprint = (int(np.sum(sample)), int(np.mean(sample)))
            except Exception:
                fingerprint = None
            if last_fingerprint is not None and fingerprint == last_fingerprint:
                skipped_static += 1
                if (ts - last_log) >= 1.0:
                    log_line(
                        "INFO ",
                        f"Captured frames: {count}/s (buffer={len(STATE.frame_buffer)}), skipped_static={skipped_static}/s",
                    )
                    count = 0
                    skipped_static = 0
                    last_log = ts
                await asyncio.sleep(interval)
                continue

            STATE.frame_buffer.append((ts, frozen))
            last_fingerprint = fingerprint
            count += 1
            if (ts - last_log) >= 1.0:
                log_line(
                    "INFO ",
                    f"Captured frames: {count}/s (buffer={len(STATE.frame_buffer)}), skipped_static={skipped_static}/s",
                )
                count = 0
                skipped_static = 0
                last_log = ts
        except asyncio.CancelledError:
            print(f"[CAPTURE_LOOP] Cancelled after {iteration} iterations", flush=True)
            break
        except Exception as exc:
            print(f"[CAPTURE_LOOP] ERROR at iter={iteration}: {exc}", flush=True)
            log_error(str(exc))
            log_line("ERR  ", str(exc))
            break
        await asyncio.sleep(interval)
    print(f"[CAPTURE_LOOP] Exited. Total iterations={iteration}, STATE.playing={STATE.playing}", flush=True)


async def _worker_poll_loop() -> None:
    """Poll worker outputs and update belief/ghost/robot from the main thread.

    This loop runs for the whole UI session (not only while timeline is playing),
    so manual prompts still receive/log model replies when simulation is paused.
    """
    while True:
        try:
            while True:
                msg = STATE.worker_out_q.get_nowait() if STATE.worker_out_q is not None else None
                if not msg:
                    break

                # Accept worker outputs even if run_id mismatches (worker was still processing when STOP was pressed)
                # Only drop if the output is truly ancient (>30s old)
                try:
                    msg_run_id = int(msg.get("run_id", STATE.run_id))
                    msg_ts = float(msg.get("ts", 0))
                    age = time.time() - msg_ts if msg_ts > 0 else 999
                except Exception:
                    msg_run_id = int(STATE.run_id)
                    age = 0

                if msg_run_id != int(STATE.run_id) and age > 30.0:
                    log_line("WARN ", f"Dropped ancient worker output (age={age:.1f}s, run_id={msg_run_id})")
                    continue

                status = msg.get("status")
                if status != "Done":
                    STATE.vlm_busy = False
                    STATE.vlm_busy_since = 0.0
                    STATE.vlm_busy_trigger = ""
                    _update_vlm_status("idle")
                    err = str(msg.get("error") or "")
                    log_line("WARN ", f"Worker status={status}: {err}")
                    # Avoid spamming a dead/OOM model server with immediate retries.
                    if "HTTP 500" in err or "EngineDeadError" in err or "out of memory" in err.lower():
                        STATE.pause_inquiry_until = time.time() + 8.0
                        log_line("WARN ", "Auto inquiry paused for 8s after model error.")
                    continue

                # Calculate response time
                response_time = time.time() - STATE.vlm_busy_since if STATE.vlm_busy_since > 0 else 0
                STATE.vlm_busy = False
                STATE.vlm_busy_since = 0.0
                STATE.vlm_busy_trigger = ""
                _update_vlm_status("idle")

                # Check if this was a grounding request
                trigger = msg.get("trigger", {})
                trigger_type = str((trigger or {}).get("type") or "")
                is_grounding = trigger_type == "grounding" and trigger.get("mode") == "initialization"
                is_manual = trigger_type == "user"

                # When a manual request is pending, suppress display of auto responses
                # that were in-flight before the manual was sent.
                if STATE.manual_pending and not is_manual and not is_grounding:
                    log_line("INFO ", f"Auto response suppressed (manual pending, {response_time:.1f}s)")
                    _update_vlm_status("busy", "user")  # still waiting for manual
                    # Still update belief state silently from auto response
                    snapshot = STATE.belief_manager.get_snapshot() if STATE.belief_manager else {}
                    if STATE.short_memory is not None and isinstance(snapshot, dict):
                        try:
                            STATE.short_memory.update_objects_from_belief(snapshot, now=time.time())
                        except Exception:
                            pass
                    continue

                # Clear manual_pending and release auto inquiry pause when manual response arrives.
                if is_manual:
                    STATE.manual_pending = False
                    STATE.pause_inquiry_until = time.time() + float(INQUIRY_INTERVAL_SEC)
                    _update_vlm_status("idle")
                elif not is_grounding:
                    # Throttle completion-driven auto chaining so indicator can reflect true idle gaps.
                    STATE.pause_inquiry_until = max(
                        float(STATE.pause_inquiry_until or 0.0),
                        time.time() + float(INQUIRY_INTERVAL_SEC),
                    )

                if is_grounding:
                    log_line("INFO ", f"Grounding complete ({response_time:.1f}s) — belief state initialised.")
                    STATE.grounding_complete = True
                    STATE.grounding_in_progress = False

                    # Log the classified objects so the user can verify the split.
                    _bu = msg.get("belief_update") or {}
                    if isinstance(_bu, dict):
                        _active = list((_bu.get("objects") or {}).keys())
                        _static = list((_bu.get("static_context") or {}).keys())
                        if _active:
                            log_line("INFO ", f"Active (Interactable): {_active}")
                        if _static:
                            log_line("INFO ", f"Static (Structural):   {_static[:10]}")

                    if STATE.init_status_label is not None:
                        try:
                            STATE.init_status_label.text = "Initialization Complete. Ready for Play."
                            STATE.init_status_label.set_style({"color": 0xFF00CC00})  # green
                        except Exception:
                            pass

                else:
                    log_line("INFO ", f"Worker response received ({response_time:.1f}s)")

                reply = str(msg.get("reply") or "")
                action = msg.get("action") if isinstance(msg.get("action"), dict) else {}
                stm_observation = str(msg.get("stm_observation") or "")
                ltm_snips = msg.get("ltm_snippets") if isinstance(msg.get("ltm_snippets"), list) else []
                snapshot = STATE.belief_manager.get_snapshot() if STATE.belief_manager else {}

                # Defensive sync: if worker produced canonical belief but STM didn't ingest it, repair here.
                if STATE.short_memory is not None and isinstance(snapshot, dict):
                    try:
                        STATE.short_memory.update_objects_from_belief(snapshot, now=time.time())
                    except Exception as exc:
                        log_line("WARN ", f"ShortMemory sync failed: {exc}")

                # Always log model output to terminal
                raw_output = msg.get("raw") or {}
                belief_update = msg.get("belief_update") or {}

                # Also print to stdout so it appears in terminal regardless of UI
                print(f"[WORKER] stm={stm_observation[:200]!r}", flush=True)
                print(f"[WORKER] reply={reply[:200]!r}", flush=True)
                if isinstance(raw_output, dict):
                    raw_reply = str(raw_output.get("reply") or "")
                    raw_meta = raw_output.get("meta", {})
                    print(f"[WORKER] raw.reply={raw_reply[:200]!r} | raw.meta={raw_meta}", flush=True)

                # --- Three-Stage Display: STM (perception) then Reply (belief-based) ---
                if stm_observation and stm_observation not in ("no observation",):
                    log_line("SEE  ", stm_observation)

                # Find the best reply text from multiple sources
                display_reply = ""
                if reply and reply not in ("<empty response>", ""):
                    display_reply = reply
                elif isinstance(raw_output, dict):
                    raw_reply = str(raw_output.get("reply") or "")
                    if raw_reply and raw_reply not in ("<empty response>", ""):
                        display_reply = raw_reply

                if display_reply:
                    display_reply = _extract_reply_text(display_reply)
                    # Keep UI concise: one AI entry per worker response.
                    last_reply = str(getattr(STATE, "_last_logged_ai_reply", "") or "")
                    last_ts = float(getattr(STATE, "_last_logged_ai_ts", 0.0) or 0.0)
                    now_ts = time.time()
                    if not (display_reply == last_reply and (now_ts - last_ts) < 2.0):
                        log_line("AI   ", display_reply)
                        setattr(STATE, "_last_logged_ai_reply", display_reply)
                        setattr(STATE, "_last_logged_ai_ts", now_ts)
                    STATE.last_vlm_response = display_reply
                else:
                    # Build summary from belief_update objects
                    obj_parts = []
                    bu_objs = belief_update.get("objects", {}) if isinstance(belief_update, dict) else {}
                    if isinstance(bu_objs, dict):
                        for oname, odata in list(bu_objs.items())[:5]:
                            if isinstance(odata, dict):
                                status = odata.get("belief_status", "?")
                                conf = odata.get("confidence", "?")
                                change = odata.get("temporal_change", "")
                                obj_parts.append(f"{oname}: {status} (conf={conf}, {change})")
                    if obj_parts:
                        summary = "Observed: " + " | ".join(obj_parts)
                        log_line("AI   ", summary)
                        STATE.last_vlm_response = summary
                    else:
                        # Log parse error if present in meta
                        parse_err = ""
                        if isinstance(raw_output, dict):
                            meta = raw_output.get("meta", {})
                            if isinstance(meta, dict):
                                parse_err = str(meta.get("parse_error", ""))
                        if parse_err:
                            log_line("WARN ", f"Model error: {parse_err[:300]}")
                        log_line("AI   ", "<no model output - check terminal for [VLM] logs>")

                # Log belief update and action
                if belief_update:
                    log_line("INFO ", f"Belief: {json.dumps(belief_update, ensure_ascii=True)[:500]}")
                if action and action.get("type") != "noop":
                    log_line("INFO ", f"Action: {json.dumps(action, ensure_ascii=True)}")

                # Update UI labels.
                if STATE.action_label is not None:
                    try:
                        STATE.action_label.text = json.dumps(action or {}, ensure_ascii=True, indent=2)[:2000]
                    except Exception:
                        STATE.action_label.text = str(action)

                if STATE.belief_label is not None and STATE.short_memory is not None:
                    belief_text = STATE.short_memory.compact_summary()
                    # Fallback to raw belief snapshot if STM looks empty.
                    if belief_text.startswith("no objects tracked"):
                        belief_text = _compact_belief_from_snapshot(snapshot)
                    STATE.belief_label.text = belief_text

                if STATE.ltm_label is not None:
                    STATE.ltm_label.text = "\n".join([str(s) for s in ltm_snips][:5])

                gv = STATE.ghost_visualizer
                if gv is not None and getattr(gv, "enabled", False):
                    try:
                        gv.sync_ghosts(snapshot)
                    except Exception as exc:
                        log_line("WARN ", f"Ghost sync failed: {exc}")

                # Execute robot action on the main thread (non-blocking, persistent target).
                rc = STATE.robot_controller
                if isinstance(action, dict) and action:
                    a_type = str(action.get("type") or "").strip().lower()
                    # --- Magic assembly actions (kinematic snap) ---
                    if a_type in ("combine", "separate"):
                        ma = STATE.magic_assembly
                        if ma is not None:
                            from .magic_assembly import AssemblyCommand
                            args = action.get("args") or {}
                            if a_type == "combine":
                                child = str(args.get("child") or args.get("child_name") or args.get("partA") or "").strip()
                                parent = str(args.get("parent") or args.get("parent_name") or args.get("partB") or "").strip()
                                plug = str(args.get("plug") or args.get("plug_name") or "").strip()
                                socket = str(args.get("socket") or args.get("socket_name") or "").strip()
                                if not (child and parent and plug and socket):
                                    log_line(
                                        "WARN ",
                                        "combine ignored: requires args {partA/child, partB/parent, plug, socket}",
                                    )
                                    _sync_vlm_status_indicator()
                                    await asyncio.sleep(0.05)
                                    continue
                                cmd = AssemblyCommand(
                                    action="combine",
                                    child_name=child,
                                    parent_name=parent,
                                    plug_name=plug,
                                    socket_name=socket,
                                )
                            else:
                                cmd = AssemblyCommand(
                                    action="separate",
                                    part_name=str(args.get("part") or args.get("part_name") or ""),
                                )
                            ma.enqueue(cmd)
                            log_line("INFO ", f"Assembly queued: {a_type} {args}")
                        else:
                            log_line("WARN ", "magic_assembly not initialized — combine/separate ignored")
                    # --- Physical robot arm actions ---
                    elif rc is not None and getattr(rc, "enabled", False) and a_type and a_type != "noop":
                        rc.set_action(action)
                        if STATE.short_memory is not None:
                            STATE.short_memory.set_last_action(action, status="running")
                _sync_vlm_status_indicator()
        except queue.Empty:
            # Watchdog: if worker has been busy for too long without output, log it
            if STATE.vlm_busy and STATE.vlm_busy_since > 0:
                busy_duration = time.time() - STATE.vlm_busy_since
                if busy_duration > 30.0:  # 30 second timeout
                    now_ts = time.time()
                    last_warn_ts = float(getattr(STATE, "_worker_busy_warn_ts", 0.0) or 0.0)
                    # Throttle repetitive watchdog warnings to keep the terminal readable.
                    if (now_ts - last_warn_ts) >= 30.0:
                        log_line("WARN ", f"Worker busy for {busy_duration:.1f}s without response - may be stuck")
                        setattr(STATE, "_worker_busy_warn_ts", now_ts)
                    if busy_duration > 60.0:  # 1 minute - force release
                        STATE.vlm_busy = False
                        STATE.vlm_busy_since = 0.0
                        STATE.vlm_busy_trigger = ""
                        _update_vlm_status("idle")
                        log_line("WARN ", "Worker watchdog: released stuck busy flag")
                        setattr(STATE, "_worker_busy_warn_ts", 0.0)
            else:
                if getattr(STATE, "_worker_busy_warn_ts", 0.0):
                    setattr(STATE, "_worker_busy_warn_ts", 0.0)
                _sync_vlm_status_indicator()
        except asyncio.CancelledError:
            break
        except Exception as exc:
            log_error(str(exc))
            log_line("ERR  ", str(exc))
            break

        await asyncio.sleep(0.05)


async def _inquiry_loop() -> None:
    """Watchdog loop — keeps VLM busy-flag healthy; does NOT auto-enqueue frames.

    Inference is now purely event-driven:
      • GT change    : _robot_update_loop() fires when a tracked object moves > threshold.
      • User message : on_send() fires when the human types in the chat.
      • Test button  : on_test_cosmos() fires on demand.

    This loop's only job is to release a stale busy-flag when the worker thread
    crashes or times out without sending a reply, so the pipeline never gets stuck.
    """
    print(f"[INQUIRY] _inquiry_loop() starting (watchdog-only). STATE.playing={STATE.playing}", flush=True)
    while STATE.playing:
        tick = 0.1
        try:
            _sync_vlm_status_indicator()

            # Release stale busy flag if worker got stuck/crashed without a callback.
            if STATE.vlm_busy:
                stale_threshold = max(30.0, float(TIMEOUT_SEC) * 1.5)
                if (time.time() - float(STATE.vlm_busy_since or 0.0)) > stale_threshold:
                    STATE.vlm_busy = False
                    STATE.vlm_busy_since = 0.0
                    STATE.vlm_busy_trigger = ""
                    log_line("WARN ", "Watchdog: released stale in-flight VLM request.")
                    _sync_vlm_status_indicator()

        except asyncio.CancelledError:
            break
        except Exception as exc:
            log_error(str(exc))
            log_line("ERR  ", str(exc))
            break

        await asyncio.sleep(tick)


async def _robot_update_loop() -> None:
    """Apply robot targets and monitor GT state every Kit tick while playing (main thread only)."""
    rc = STATE.robot_controller
    sm = STATE.state_monitor
    last_status = None
    _frame_count = 0
    _last_diag_time = time.time()
    _pending_gt_trigger = None  # {"changed_objects": [...], "max_displacement": float, "event_ts": float, "capture_since_ts": float}
    _last_gt_wait_log = 0.0
    _last_gt_event_ts = 0.0
    while STATE.playing:
        _frame_count += 1
        # Periodic physics time check (every ~120 frames / ~2s).
        if _frame_count % 120 == 0:
            try:
                _tl = omni.timeline.get_timeline_interface()
                _now = time.time()
                _elapsed = _now - _last_diag_time
                print(f"[PHYS_TICK] frame={_frame_count}, sim_time={_tl.get_current_time():.3f}s, "
                      f"is_playing={_tl.is_playing()}, wall_elapsed={_elapsed:.1f}s", flush=True)
                _last_diag_time = _now
            except Exception:
                pass
        try:
            # Drain the magic assembly command queue (main-thread USD writes).
            if STATE.magic_assembly is not None:
                STATE.magic_assembly.execute_pending()

            # Update robot controller if enabled
            if rc is not None and getattr(rc, "enabled", False):
                status = rc.update()
                # Keep task/action status synchronized in STM.
                if last_status and last_status.status == "running" and status.status in ("done", "error"):
                    # Update STM status for downstream prompting / UI.
                    if STATE.short_memory is not None:
                        try:
                            snap = STATE.short_memory.get_snapshot()
                            last_action = {}
                            if isinstance(snap.get("task"), dict) and isinstance(snap["task"].get("last_action"), dict):
                                last_action = snap["task"]["last_action"]
                            STATE.short_memory.set_last_action(last_action, status=status.status)
                        except Exception:
                            pass
                    log_line("INFO ", f"Robot action finished: {status.status} ({status.detail})")
                last_status = status

            # Monitor GT state changes and trigger inference
            if sm is not None:
                trigger_event = sm.update()
                if trigger_event is not None:
                    # Skip if displacement is negligible (initial baseline or noise)
                    if trigger_event.max_displacement >= 0.001:
                        event_ts = float(getattr(trigger_event, "timestamp", 0.0) or time.time())
                        # De-duplicate repeated monitor callbacks for the same event timestamp.
                        if event_ts > (_last_gt_event_ts + 1e-9):
                            _last_gt_event_ts = event_ts
                            changed_names = ", ".join(trigger_event.changed_objects) if trigger_event.changed_objects else "objects"
                            log_line(
                                "INFO ",
                                f"GT trigger: {trigger_event.changed_objects} changed (displacement={trigger_event.max_displacement:.3f}m)",
                            )
                            # Save/update pending trigger (accumulate changed objects across firings).
                            if _pending_gt_trigger is None:
                                _pending_gt_trigger = {
                                    "changed_objects": list(trigger_event.changed_objects or []),
                                    "max_displacement": float(trigger_event.max_displacement),
                                    "event_ts": event_ts,
                                    # Start post-trigger frame capture now; dispatch only after 5 frames.
                                    "capture_since_ts": event_ts,
                                }
                            else:
                                merged_objects = list(
                                    dict.fromkeys(
                                        list(_pending_gt_trigger.get("changed_objects", []))
                                        + list(trigger_event.changed_objects or [])
                                    )
                                )
                                _pending_gt_trigger["changed_objects"] = merged_objects
                                _pending_gt_trigger["max_displacement"] = max(
                                    float(_pending_gt_trigger.get("max_displacement", 0.0)),
                                    float(trigger_event.max_displacement),
                                )
                                _pending_gt_trigger["event_ts"] = max(
                                    float(_pending_gt_trigger.get("event_ts", event_ts)),
                                    event_ts,
                                )
                                changed_names = ", ".join(merged_objects) if merged_objects else "objects"

                # Try to dispatch pending GT trigger with the latest buffered frames.
                _GT_FRAME_COUNT = 5
                if _pending_gt_trigger is not None:
                    # Never enqueue while another request is in-flight/queued.
                    if _has_pending_inference() or STATE.manual_pending or (time.time() < float(STATE.pause_inquiry_until or 0.0)):
                        await omni.kit.app.get_app().next_update_async()
                        continue

                    changed_objects = list(_pending_gt_trigger.get("changed_objects", []))
                    changed_names = ", ".join(changed_objects) if changed_objects else "objects"
                    capture_since_ts = float(_pending_gt_trigger.get("capture_since_ts", 0.0))

                    # Capture strictly post-trigger consecutive frames; dispatch once we have 5.
                    buffered_items = list(STATE.frame_buffer)
                    post_trigger_items = [(ts, rgb) for ts, rgb in buffered_items if float(ts) >= capture_since_ts]
                    if len(post_trigger_items) >= _GT_FRAME_COUNT:
                        frames = [_clone_frame(rgb) for _ts, rgb in post_trigger_items[-_GT_FRAME_COUNT:]]
                        log_line(
                            "INFO ",
                            f"GT trigger dispatched ({len(post_trigger_items)} post-trigger buffered, sending {len(frames)})",
                        )
                        try:
                            paths = await asyncio.to_thread(save_sent_frames, "auto_gt", list(frames))
                            if paths:
                                log_line("INFO ", f"Auto sent frames: {', '.join(paths)}")
                        except Exception as exc:
                            log_warn(f"Failed to save GT-triggered frames: {exc}")

                        queued = _enqueue_inference(
                            frames=frames,
                            user_text=f"Objects moved: {changed_names}. Describe what changed and update belief states.",
                            trigger={
                                "type": "gt_change",
                                "changed_objects": changed_objects,
                                "max_displacement": float(_pending_gt_trigger.get("max_displacement", 0.0)),
                                "timestamp": float(_pending_gt_trigger.get("event_ts", time.time())),
                            },
                            force=False,
                        )
                        if queued:
                            _pending_gt_trigger = None
                    else:
                        now = time.time()
                        if now - _last_gt_wait_log > 1.0:
                            log_line(
                                "INFO ",
                                f"GT trigger waiting for 5 post-trigger frames (have {len(post_trigger_items)}).",
                            )
                            _last_gt_wait_log = now
        except asyncio.CancelledError:
            break
        except Exception as exc:
            log_warn(f"Robot/GT update failed: {exc}")
            # Don't break - keep the loop running so GT monitoring continues
        await omni.kit.app.get_app().next_update_async()


def start_capture_and_inquiry() -> None:
    print(f"[CAPTURE] start_capture_and_inquiry() called. capture_task={STATE.capture_task}, inquiry_task={STATE.inquiry_task}", flush=True)
    if STATE.capture_task is not None or STATE.inquiry_task is not None:
        print(f"[CAPTURE] Tasks already exist, returning early", flush=True)
        return
    loop = _get_task_loop()

    print(f"[CAPTURE] Creating _capture_loop, _inquiry_loop, _robot_update_loop tasks", flush=True)
    STATE.capture_task = loop.create_task(_capture_loop())
    STATE.inquiry_task = loop.create_task(_inquiry_loop())
    if STATE.robot_update_task is None:
        STATE.robot_update_task = loop.create_task(_robot_update_loop())
    print(f"[CAPTURE] Tasks created successfully", flush=True)


def stop_capture_and_inquiry() -> None:
    if STATE.capture_task is not None:
        STATE.capture_task.cancel()
        STATE.capture_task = None
    if STATE.inquiry_task is not None:
        STATE.inquiry_task.cancel()
        STATE.inquiry_task = None
    if STATE.robot_update_task is not None:
        STATE.robot_update_task.cancel()
        STATE.robot_update_task = None


def _set_run_frames_dir() -> None:
    pending = str(getattr(STATE, "pending_run_frames_dir", "") or "").strip()
    if pending:
        run_dir = pending
        os.makedirs(run_dir, exist_ok=True)
        STATE.pending_run_frames_dir = ""
        log_line("INFO ", f"Run frames dir (reused from init): {run_dir}")
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(STATE.sent_frames_root, ts)
        os.makedirs(run_dir, exist_ok=True)
        log_line("INFO ", f"Run frames dir: {run_dir}")
    STATE.sent_frames_dir = run_dir


def _ensure_preplay_frames_dir() -> str:
    """Create/reuse a pre-play sent_frames dir so init + run share one folder."""
    pending = str(getattr(STATE, "pending_run_frames_dir", "") or "").strip()
    if pending:
        os.makedirs(pending, exist_ok=True)
        STATE.sent_frames_dir = pending
        return pending
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(STATE.sent_frames_root, ts)
    os.makedirs(run_dir, exist_ok=True)
    STATE.pending_run_frames_dir = run_dir
    STATE.sent_frames_dir = run_dir
    return run_dir


async def _bootstrap_run_start(run_id_snapshot: int) -> None:
    """Ensure timeline/physics is really advancing before starting capture/inquiry tasks."""
    tl = omni.timeline.get_timeline_interface()
    try:
        if not tl.is_playing():
            _set_play_simulations(True)
            tl.play()

        start_t = float(tl.get_current_time())
        started = await _wait_for_sim_time_advance(start_t, max_updates=90, min_delta=5e-4)
        if not started:
            # Guard: suppress timeline STOP events during recovery so _end_run is not
            # triggered by our internal stop→play cycles.
            STATE._bootstrap_recovery = True
            try:
                for attempt in (1, 2, 3):
                    log_line("WARN ", f"PLAY latched but simulation time did not advance; recovery attempt {attempt}/3.")
                    try:
                        tl.stop()
                        await omni.kit.app.get_app().next_update_async()
                        await omni.kit.app.get_app().next_update_async()
                    except Exception:
                        pass

                    _set_play_simulations(True)
                    _ensure_physics_scene_defaults()

                    if attempt >= 2:
                        # Harder nudge for first-run dead start in some Isaac Lab sessions.
                        try:
                            from isaaclab.sim import SimulationContext

                            sim_ctx = SimulationContext.instance()
                            if sim_ctx is not None:
                                sim_ctx.reset()
                        except Exception:
                            pass

                    _ensure_sim_context_playing()
                    tl.play()
                    # Give more frames on later attempts for physics to initialise.
                    wait_updates = 120 if attempt < 3 else 180
                    started = await _wait_for_sim_time_advance(float(tl.get_current_time()), max_updates=wait_updates, min_delta=5e-4)
                    if started:
                        break
            finally:
                STATE._bootstrap_recovery = False

        if (not started) or (run_id_snapshot != int(STATE.run_id)) or (not STATE.playing):
            log_line("WARN ", "Simulation did not start stepping on this PLAY. Run bootstrap aborted.")
            STATE.playing = False
            STATE.run_active = False
            try:
                tl.stop()
            except Exception:
                pass
            _set_play_simulations(False)
            return

        # Replicator products can be invalidated after STOP/PLAY cycles; refresh once per run.
        try:
            init_camera()
        except Exception as exc:
            log_line("WARN ", f"Camera re-init during run bootstrap failed: {exc}")

        print(f"[START_RUN] Bootstrap OK for run_id={run_id_snapshot}; starting capture/inquiry tasks", flush=True)
        start_capture_and_inquiry()
        _diagnose_physics_scene()
        loop = _get_task_loop()
        loop.create_task(_physics_monitor_task())
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        log_line("ERR  ", f"Run bootstrap failed: {exc}")
        STATE.playing = False
        STATE.run_active = False
        _set_play_simulations(False)
    finally:
        # Clear handle once task exits.
        if getattr(STATE, "run_bootstrap_task", None) is asyncio.current_task():
            STATE.run_bootstrap_task = None


def _start_run() -> None:
    print(f"[START_RUN] _start_run() called. STATE.playing={STATE.playing}", flush=True)
    if STATE.playing:
        print(f"[START_RUN] Already playing, returning early", flush=True)
        return
    _set_play_simulations(True)
    STATE.run_id += 1
    STATE.playing = True
    STATE.run_active = True
    STATE.run_start_time = time.time()
    STATE.last_auto_infer_ts = float(STATE.run_start_time)
    _set_run_frames_dir()

    # Repair invalid stage gravity attrs before starting capture/agent loops.
    _ensure_physics_scene_defaults()

    # Initialize cognition pipeline eagerly (StateMonitor, memory, worker, etc.)
    # so GT tracking and robot control are active from the first frame.
    try:
        _init_pipeline_once()
    except Exception as exc:
        print(f"[START_RUN] _init_pipeline_once() failed: {exc}", flush=True)
        log_warn(f"Pipeline init failed: {exc}")

    print(f"[START_RUN] STATE.playing={STATE.playing}, run_id={STATE.run_id}. Bootstrapping run start...", flush=True)
    _cancel_run_bootstrap_task()
    loop = _get_task_loop()
    STATE.run_bootstrap_task = loop.create_task(_bootstrap_run_start(int(STATE.run_id)))

    print(f"[START_RUN] bootstrap task created: {STATE.run_bootstrap_task}", flush=True)
    log_line("INFO ", f"Run #{STATE.run_id} started.")


def _end_run(*, reason: str) -> None:
    print(f"[END_RUN] _end_run(reason={reason}). STATE.playing={STATE.playing}", flush=True)
    if not STATE.playing:
        print(f"[END_RUN] Not playing, returning early", flush=True)
        return
    STATE.playing = False
    STATE.run_active = False
    _set_play_simulations(False)
    _cancel_run_bootstrap_task()
    stop_capture_and_inquiry()
    if STATE.run_start_time:
        duration = time.time() - STATE.run_start_time
        log_line("INFO ", f"Run duration: {duration:.2f}s (ended via {reason})")

    # Store an episode record asynchronously (avoid blocking UI/timeline callback thread).
    if STATE.long_memory is not None:
        snap = STATE.short_memory.get_snapshot() if STATE.short_memory is not None else {}
        belief = STATE.belief_manager.get_snapshot() if STATE.belief_manager is not None else {}
        payload = {"task": "scene_episode", "last_belief": belief, "stm": snap}

        async def _store_episode_async() -> None:
            try:
                await asyncio.to_thread(STATE.long_memory.end_episode, payload)
            except Exception as exc:
                log_warn(f"Failed to store episode in LTM: {exc}")

        loop = _get_task_loop()
        loop.create_task(_store_episode_async())


def on_end_run() -> None:
    """User-facing STOP: pause timeline and reset episode to initial state."""
    print("[STOP] on_end_run() called from UI", flush=True)
    try:
        tl = omni.timeline.get_timeline_interface()
        # Explicit STOP to match user expectation and force timeline state transition.
        tl.stop()
    except Exception:
        pass
    # Perform reset immediately so STOP semantics are deterministic.
    _reset_episode_runtime("ui_stop")


def on_reset_episode() -> None:
    """Backward-compatible alias for STOP/reset."""
    on_end_run()


def _install_timeline_run_hooks() -> None:
    """Start/stop the run on Timeline events without polling."""
    print(f"[HOOKS] _install_timeline_run_hooks() called. Already installed={getattr(STATE, '_timeline_hooks_installed', False)}", flush=True)
    if getattr(STATE, "_timeline_hooks_installed", False):
        return
    setattr(STATE, "_timeline_hooks_installed", True)
    print(f"[HOOKS] Installing timeline event subscription...", flush=True)

    timeline = omni.timeline.get_timeline_interface()
    stream = timeline.get_timeline_event_stream()

    def _on_timeline_event(e):
        try:
            et = int(getattr(e, "type", -1))
        except Exception:
            et = -1

        if et == int(omni.timeline.TimelineEventType.PLAY):
            print(f"[TIMELINE] PLAY (bootstrap_recovery={STATE._bootstrap_recovery})", flush=True)
            if not STATE._bootstrap_recovery:
                _start_run()
            else:
                print(f"[TIMELINE] PLAY suppressed (bootstrap recovery in progress)", flush=True)
        elif et == int(omni.timeline.TimelineEventType.PAUSE):
            print(f"[TIMELINE] PAUSE -> STOP", flush=True)
            # Convert PAUSE into full STOP semantics so reset is deterministic.
            try:
                timeline.stop()
            except Exception:
                pass
            if not STATE._bootstrap_recovery:
                _reset_episode_runtime("timeline_pause")
            else:
                print(f"[TIMELINE] PAUSE suppressed (bootstrap recovery in progress)", flush=True)
        elif et == int(omni.timeline.TimelineEventType.STOP):
            print(f"[TIMELINE] STOP", flush=True)
            if not STATE._bootstrap_recovery:
                _reset_episode_runtime("timeline_stop")
            else:
                print(f"[TIMELINE] STOP suppressed (bootstrap recovery in progress)", flush=True)
        # else: ignore tick/update events (type=4 etc.)

    try:
        sub = stream.create_subscription_to_pop(_on_timeline_event)
        STATE._kit_subs.append(sub)
    except Exception as exc:
        log_warn(f"Failed to subscribe to timeline events: {exc}")
        return

    # If the user hit PLAY before hooks installed, sync once.
    try:
        if timeline.is_playing():
            _start_run()
    except Exception:
        pass


def build_ui() -> None:
    if STATE.window is not None:
        STATE.window.visible = True
        return

    STATE.window = ui.Window("HeadCam Terminal Chat", width=760, height=640)

    with STATE.window.frame:
        with ui.VStack(spacing=6):
            # Top controls: compact fixed height.
            with ui.Frame(height=120):
                with ui.VStack(spacing=6):
                    ui.Label("Free chat + agent loop. Frames captured in RAM; model runs off-thread (sim never blocks).")

                    with ui.HStack(height=28, spacing=8):
                        ui.Button("Init Camera", clicked_fn=on_init_cam, height=28)
                        ui.Button("Initiate Cosmos", clicked_fn=on_initiate_cosmos, height=28, tooltip="Pre-simulation: VLM identifies objects and creates initial belief state")
                        # Init status: "Ready (no init)" → "Initializing..." → "Ready ✓"
                        STATE.init_status_label = ui.Label("Ready (no init)", width=130, alignment=ui.Alignment.LEFT)
                        STATE.init_status_label.set_style({"color": 0xFF888888})  # grey = not yet initialised
                        ui.Button("Test Cosmos", clicked_fn=on_test_cosmos, height=28)
                        STATE.play_button = ui.Button("PLAY", clicked_fn=on_play, height=28)
                        ui.Button("STOP", clicked_fn=on_end_run, height=28)
                        ui.Spacer()
                        ui.Button("Clear", clicked_fn=on_clear, height=28)

                    with ui.HStack(height=22, spacing=8):
                        STATE.attach_latest_model = ui.SimpleBoolModel(True)
                        ui.CheckBox(STATE.attach_latest_model, width=18, height=18)
                        ui.Label("Attach recent frames to manual prompts")

                    ui.Label(f"Camera Prim: {STATE.active_camera_prim_path}", word_wrap=True)
                    ui.Label(f"Cosmos Endpoint: {cosmos_endpoint() or '<disabled>'}", word_wrap=True)
                    ui.Label(f"Cosmos Model: {COSMOS_MODEL}", word_wrap=True)

            ui.Separator()

            with ui.HStack(height=20, spacing=8):
                ui.Label("Terminal:")
                ui.Spacer()
                STATE.vlm_status_label = ui.Label("VLM: Idle", width=260, alignment=ui.Alignment.RIGHT)
                STATE.vlm_status_label.set_style({"color": 0xFF00CC00})  # green

            # Use a larger fixed terminal area for better visibility.
            STATE.log_frame = ui.ScrollingFrame(height=300)
            with STATE.log_frame:
                STATE.log_label = ui.Label("", word_wrap=True)

            ui.Separator()

            # Bottom status + input panel: compact fixed height.
            with ui.Frame(height=145):
                with ui.VStack(spacing=6):
                    ui.Label("Last action:")
                    STATE.action_label = ui.Label("{}", word_wrap=True, height=24)

                    ui.Label("Belief (compact):")
                    STATE.belief_label = ui.Label("", word_wrap=True, height=20)

                    ui.Label("Long-term memory (top snippets):")
                    STATE.ltm_label = ui.Label("", word_wrap=True, height=24)

                    ui.Separator()

                    ui.Label("Type anything:")
                    STATE.input_model = ui.SimpleStringModel("")
                    input_field = ui.StringField(STATE.input_model, height=30)
                    _bind_input_enter_to_send(input_field)

                    with ui.HStack(height=30, spacing=8):
                        ui.Button("Send", clicked_fn=on_send, height=30)
                        ui.Button("Next step", clicked_fn=on_step_next, height=30,
                                  tooltip="Shortcut for /step_next — execute the next ready assembly step")
                        ui.Spacer()


def _install_startup_autostop_hooks() -> None:
    """Install hooks to keep the stage static on launch until the user clicks PLAY.

    In Isaac Lab "python" experiences, the timeline can sometimes resume PLAY from persistent settings,
    or get toggled to PLAY after a stage finishes loading. We force STOP around stage-open time so the
    initial state is stable/static and the demo only proceeds after an explicit user PLAY.
    """
    if getattr(STATE, "_startup_hooks_installed", False):
        return
    setattr(STATE, "_startup_hooks_installed", True)

    # Best-effort: mark stage as opened if one already exists (script-editor use case).
    try:
        import omni.usd

        if omni.usd.get_context().get_stage() is not None:
            STATE.startup_stage_opened = True
    except Exception:
        pass

    try:
        import omni.usd

        ctx = omni.usd.get_context()
        stream = ctx.get_stage_event_stream()
        stage_event_type = getattr(omni.usd, "StageEventType", None)
        opened_type = None
        if stage_event_type is not None and hasattr(stage_event_type, "OPENED"):
            try:
                opened_type = int(stage_event_type.OPENED)
            except Exception:
                opened_type = stage_event_type.OPENED
        if opened_type is None:
            # Fallback: rely on the startup timeline guard instead of guessing stage event types.
            raise RuntimeError("omni.usd.StageEventType.OPENED not available")

        timeline = omni.timeline.get_timeline_interface()

        def _force_show_play_button() -> None:
            # Keep compatibility with existing startup guards if the toolbar is present.
            try:
                import omni.kit.widget.toolbar

                toolbar = omni.kit.widget.toolbar.get_instance()
                play_button_group = toolbar._builtin_tools._play_button_group  # type: ignore[attr-defined]
                if play_button_group is not None:
                    play_btn = play_button_group._play_button  # type: ignore[attr-defined]
                    play_btn.visible = True
                    play_btn.enabled = True
            except Exception:
                pass

        def _on_stage_event(e):
            if getattr(STATE, "_startup_stage_event_handled", False):
                return
            try:
                et = int(getattr(e, "type", -1))
            except Exception:
                et = -1
            if et != int(opened_type):
                return

            STATE.startup_stage_opened = True
            setattr(STATE, "_startup_stage_event_handled", True)
            try:
                if timeline.is_playing():
                    timeline.stop()
                    _set_play_simulations(False)
                    log_warn(
                        "Timeline PLAY detected during startup/stage-load; forced STOP. Click PLAY once the stage finishes loading."
                    )
                else:
                    # Ensure STOPPED state (not just paused) for clean physics init.
                    timeline.stop()
                    _set_play_simulations(False)
                _force_show_play_button()
            except Exception:
                pass

        sub = stream.create_subscription_to_pop(_on_stage_event)
        STATE._kit_subs.append(sub)

        # Panel PLAY/STOP controls are authoritative for this demo UI.
    except Exception as exc:
        log_warn(f"Failed to install stage event hook for auto-play suppression: {exc}")


def run() -> None:
    loop = _get_task_loop()

    _configure_quiet_kit_logging()
    _ensure_asset_browser_cache_dir()

    # Install stage/timeline hooks immediately so we don't miss early stage-open events.
    _install_startup_autostop_hooks()
    # Best-effort: force STOP immediately on script start.
    # STOP (not PAUSE) ensures timeline is at t=0 with clean physics state.
    try:
        omni.timeline.get_timeline_interface().stop()
        _set_play_simulations(False)
    except Exception:
        pass
    # Hide built-in timeline controls to avoid PLAY/PAUSE ambiguity; use panel PLAY/STOP.
    try:
        import omni.kit.widget.toolbar

        toolbar = omni.kit.widget.toolbar.get_instance()
        play_button_group = toolbar._builtin_tools._play_button_group  # type: ignore[attr-defined]
        if play_button_group is not None:
            play_button_group.visible = False
            play_button_group.enabled = False
    except Exception:
        pass

    async def _startup_timeline_guard(max_duration_sec: float = 15.0, stable_updates: int = 15) -> None:
        """Prevent unintended auto-play during startup/stage-load.

        Exits early once the stage is opened and the timeline has remained STOPPED for a short
        number of Kit updates. This avoids the "must click PLAY twice" regression while still
        catching late auto-play triggers after the stage loads.
        """
        timeline = omni.timeline.get_timeline_interface()
        warned = False
        stable = 0
        deadline = time.time() + float(max_duration_sec)

        # Force STOP once immediately (best-effort).
        # Use stop() (not pause()) so timeline resets to t=0 with clean physics.
        try:
            timeline.stop()
            _set_play_simulations(False)
        except Exception:
            pass

        while time.time() < deadline:
            try:
                if timeline.is_playing():
                    stable = 0
                    timeline.stop()
                    _set_play_simulations(False)
                    if not warned:
                        log_warn("Timeline PLAY detected during startup; forced STOP. Click PLAY once the stage finishes loading.")
                        warned = True
                else:
                    if STATE.startup_stage_opened:
                        stable += 1
                        if stable >= int(stable_updates):
                            break
            except Exception:
                pass
            await omni.kit.app.get_app().next_update_async()

        STATE.startup_autostop_done = True
        print(f"[STARTUP] Timeline guard finished. startup_stage_opened={STATE.startup_stage_opened}, autostop_done=True", flush=True)

    async def _physx_warmup() -> None:
        """Brief PLAY→STOP cycle to force PhysX scene initialization.

        Without this, the first user PLAY may report is_playing()=True
        but simulation time stays at 0 and objects don't move.
        timeline.stop() afterwards reverts all USD transforms to their
        authored initial state, so the scene is clean for the real PLAY.

        This runs BEFORE timeline hooks are installed, so _start_run()
        and _reset_episode_runtime() are NOT triggered.
        """
        try:
            timeline = omni.timeline.get_timeline_interface()
            print("[STARTUP] PhysX warm-up: PLAY...", flush=True)
            _set_play_simulations(True)
            timeline.play()
            # Give PhysX ~10 frames to initialize the scene.
            app = omni.kit.app.get_app()
            for _ in range(10):
                await app.next_update_async()
            # STOP resets timeline to t=0 and reverts USD transforms.
            timeline.stop()
            _set_play_simulations(False)
            for _ in range(5):
                await app.next_update_async()
            print(f"[STARTUP] PhysX warm-up done. is_stopped={timeline.is_stopped()}, "
                  f"time={timeline.get_current_time():.4f}", flush=True)
        except Exception as exc:
            print(f"[STARTUP] PhysX warm-up failed (non-fatal): {exc}", flush=True)

    async def _startup_sequence() -> None:
        print(f"[STARTUP] _startup_sequence() starting. Calling _startup_timeline_guard()...", flush=True)
        await _startup_timeline_guard()
        print(f"[STARTUP] Guard done. Running PhysX warm-up...", flush=True)
        await _physx_warmup()
        _startup_prime_sim_context()
        print(f"[STARTUP] Installing timeline run hooks...", flush=True)
        _install_timeline_run_hooks()
        print(f"[STARTUP] Timeline hooks installed. Ready for PLAY.", flush=True)

    # Kick startup sequence (do not await; keep UI responsive).
    loop.create_task(_startup_sequence())
    build_ui()
    _capture_initial_scene_snapshot_once()
    render_log()

    log_info("UI ready. Click 'Init Camera' once, then click PLAY to start the sim loop.")
    log_info(f"Cosmos endpoint: {cosmos_endpoint() or '<disabled>'}")
    log_info(f"Cosmos model: {COSMOS_MODEL}")
    if not COSMOS_MODEL or COSMOS_MODEL == "your-model-name-here":
        log_warn("COSMOS_MODEL is not set. Update COSMOS_MODEL or set COSMOS_MODEL env var.")

    # Best-effort cleanup on Kit shutdown.
    try:
        app = omni.kit.app.get_app()

        def _on_shutdown(_e=None):
            try:
                if STATE.worker_in_q is not None:
                    STATE.worker_in_q.put_nowait(None)
            except Exception:
                pass
            try:
                if STATE.worker_thread is not None and STATE.worker_thread.is_alive():
                    STATE.worker_thread.join(timeout=1.0)
            except Exception:
                pass

        app.get_shutdown_event_stream().create_subscription_to_pop(_on_shutdown)
    except Exception:
        pass
