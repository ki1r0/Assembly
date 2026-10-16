"""Public, bounded model interface for the bolt-insertion harness."""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from hrc_m1.observer import _frame_data_uri
from .executor import MAX_SKILL_DURATION_S


MAX_NUDGE_DURATION_S = 1.0
MAX_RETRACT_DURATION_S = 5.0
_MAX_FRAMES = 8
_MAX_HISTORY = 32
_MAX_TEXT_LENGTH = 512
_DIRECTIONS = {"x_positive", "x_negative", "y_positive", "y_negative", "z_positive", "z_negative"}
_TOOL_STATUSES = {
    "accepted", "completed", "failed", "force_limit", "invalid_action", "motion_timeout", "stalled",
    "running", "stopped", "succeeded", "timeout", "unknown",
}
_SUCCESSFUL_STATUSES = {"completed", "succeeded"}
_OBSERVATION_FIELDS = {
    "timestamp", "joint_positions", "joint_velocities", "tcp_pose", "tcp_velocity", "wrench",
    "gripper_width", "camera_intrinsics", "camera_extrinsics", "finger_bolt_contacts",
}
_TASK_TEXT_FIELDS = {"task_id", "goal", "target_id", "socket_id"}
_DIMENSION_FIELDS = {"bolt_diameter_m", "bolt_length_m", "hole_diameter_m", "seat_depth_m", "clearance_m"}
_LIMIT_FIELDS = {"nudge_max_duration_s", "retract_max_duration_s", "skill_max_duration_s"}
_TOOL_LIMITS = {
    "nudge_max_duration_s": MAX_NUDGE_DURATION_S,
    "retract_max_duration_s": MAX_RETRACT_DURATION_S,
    "skill_max_duration_s": MAX_SKILL_DURATION_S,
}
_SKILL_MODES = {
    "pick": "default",
    "transport": "default",
    "insert_and_seat": "compliant",
    "release_and_retract": "default",
}
_SYSTEM_PROMPT = (
    "You control one bolt-insertion task through the listed tools. The exact, case-sensitive `execute_skill.skill` "
    "enum strings are `pick`, `transport`, `insert_and_seat`, and `release_and_retract`; use one string exactly, "
    "with no explanation appended. Their meanings are separate guidance: pick up the bolt; move it to the socket "
    "while retaining the grasp; insert and seat while retaining the grasp; release and retract. Required order is "
    "`pick`, `transport`, `insert_and_seat`, `check_task`, then `release_and_retract` only when "
    "`public_progress.release_authorized` is true, "
    "then `check_task` again. "
    "The public check result is phase-labeled. Before release, `release_authorized: true` permits "
    "release_and_retract only; it does not finish the run. After successful release_and_retract, only a fresh "
    "`task_success: true` completes the task and permits the run to finish. A pre-release true remains valid "
    "until physical action; repeating check_task without intervening physical action adds no evidence. "
    "Release is permitted only while `release_authorized` is true. This describes the evidence and does not "
    "prescribe a next action. "
    "`check_task` is a standalone tool, never an `execute_skill.skill`; call it exactly as `{\"tool\":\"check_task\"}`. "
    "Its phase-labeled boolean result is read-only context, never a call argument. "
    "A stop never proves completion. Return exactly one JSON object with a `tool` key and only that tool's "
    "listed arguments. `tool` is exactly a listed tool name; for a skill call use `tool`=`execute_skill` and "
    "put `pick` (or another listed skill) in a separate `skill` field, never combine them. "
    f"For `execute_skill` only, `max_duration_s` is optional: omit it to use the executor default "
    f"of {MAX_SKILL_DURATION_S:g} seconds (or a lower public task limit). This bounds the real skill trajectory; pickup, transport, and "
    "insertion take physical motion and are not instantaneous. If you provide a budget, it is used unchanged "
    "and must allow the full motion within the stated limit. "
    "Use `public_progress` and tool history when choosing: do not repeat a successfully completed skill unless "
    "a public observation or later tool result gives a concrete recovery reason. Never invent such a reason; "
    "the progress summary does not prescribe the next action. "
    "Use only the supplied public images, task, observation and tool history. Never infer or invent a check result, "
    "return code, or request other data. `send_help_request` is an optional model-selected terminal action: it "
    "ends this run after local outbox persistence, invokes no helper, and does not claim a human received it. "
    "Describe only public symptoms and leave the cause unverified. Cite exact `observation:<observation_id>` or "
    "`tool_history:<zero-based-index>` references from the supplied context. A bounded `insert_and_seat` "
    "`force_limit` or `stalled` is returned insertion feedback, not an automatic help trigger; choose the next "
    "action from public state."
)
_TOOL_SCHEMA = {
    "observe": {"arguments": {}},
    "execute_skill": {
        "arguments": {"skill": sorted(_SKILL_MODES), "target_id": "one of task_spec.target_ids",
                      "mode": "must equal mode_by_skill[skill]",
                      "max_duration_s": f"optional; omitted uses executor default {MAX_SKILL_DURATION_S:g}, capped by a lower public task limit; if supplied, 0 < value <= the effective limit"},
        "skill_descriptions": {
            "pick": "Pick up the initially unheld bolt.",
            "transport": "Move the bolt to the selected socket while retaining the grasp.",
            "insert_and_seat": "Actively insert and seat the bolt while retaining the grasp.",
            "release_and_retract": "Release and retract after release_authorized=true.",
        },
        "mode_by_skill": _SKILL_MODES,
        "release_and_retract_requires_release_authorized": True,
    },
    "nudge": {
        "arguments": {"frame": ["assembly", "world"], "direction": sorted(_DIRECTIONS),
                      "step_class": ["coarse", "fine", "contact"], "max_duration_s": f"0 < value <= {MAX_NUDGE_DURATION_S}"}
    },
    "retract": {
        "arguments": {"direction": sorted(_DIRECTIONS), "distance_class": ["short", "full"],
                      "max_duration_s": f"0 < value <= {MAX_RETRACT_DURATION_S}"}
    },
    "check_task": {
        "arguments": {},
        "returns_read_only": {
            "pre_release": {"release_authorized": "Boolean; true permits release_and_retract only."},
            "post_release": {"task_success": "Boolean; only a fresh true completes the task."},
        },
        "semantics": "The public result is phase-labeled. Before release, release_authorized=true permits release_and_retract only and does not finish the run. After successful release_and_retract, only a fresh task_success=true completes the task and permits the run to finish. Repeating check_task without intervening physical action adds no evidence.",
    },
    "send_help_request": {
        "arguments": {
            "target": "one of task_spec.target_ids",
            "operation": "non-empty requested assistance description",
            "allowed_scope": "non-empty requested scope",
            "desired_postconditions": "1 to 8 non-empty public outcomes",
            "evidence_refs": "1 to 8 exact observation:<id> or tool_history:<index> references",
            "observed_problem": "non-empty model-reported public symptoms; cause unverified",
        },
        "semantics": "Terminal request persisted to the local outbox only; no helper executes and task_success remains false.",
    },
    "stop": {"arguments": {}, "semantics": "Stop without claiming task completion."},
}


class BoltAgentError(RuntimeError):
    """Model transport or response failure; no fallback action is substituted."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > _MAX_TEXT_LENGTH:
        raise ValueError(f"{name} must be a non-empty string of at most {_MAX_TEXT_LENGTH} characters")
    return value.strip()


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _numeric_tree(value: Any, name: str, *, depth: int = 0) -> Any:
    if depth > 2:
        raise ValueError(f"{name} is nested too deeply")
    if isinstance(value, (list, tuple)):
        if not value or len(value) > 64:
            raise ValueError(f"{name} must contain 1 to 64 numeric values")
        return [_numeric_tree(item, name, depth=depth + 1) for item in value]
    return _number(value, name)


def _pose(value: Any, name: str) -> list[float]:
    pose = _numeric_tree(value, name)
    if not isinstance(pose, list) or len(pose) != 7 or any(not isinstance(item, float) for item in pose):
        raise ValueError(f"{name} must be [x, y, z, qw, qx, qy, qz]")
    return pose


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed object")
    return value


def _project_observation(value: Any) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    source = _mapping(value, "observation")
    result: dict[str, Any] = {}
    if "observation_id" in source:
        result["observation_id"] = _text(source["observation_id"], "observation_id")
    for key in _OBSERVATION_FIELDS & source.keys():
        if key == "finger_bolt_contacts":
            contacts = source[key]
            if (not isinstance(contacts, Sequence) or isinstance(contacts, (str, bytes))
                    or len(contacts) != 2 or any(type(contact) is not bool for contact in contacts)):
                raise ValueError("observation.finger_bolt_contacts must contain two booleans")
            result[key] = list(contacts)
        else:
            result[key] = _pose(source[key], key) if key == "tcp_pose" else _numeric_tree(source[key], key)

    frames = _mapping(source.get("frames", {}), "observation.frames")
    if not frames or len(frames) > _MAX_FRAMES:
        raise ValueError(f"observation.frames must contain 1 to {_MAX_FRAMES} images")
    images: list[tuple[str, str]] = []
    for name, source_image in frames.items():
        camera = _text(name, "camera name")
        if not isinstance(source_image, str):
            raise ValueError(f"frame {camera!r} must be a path or data URI")
        uri = _frame_data_uri(source_image)
        if not isinstance(uri, str) or not uri.startswith("data:image/"):
            raise ValueError(f"frame {camera!r} could not be converted to an image data URI")
        images.append((camera, uri))
    result["frames"] = [name for name, _ in images]

    estimates = source.get("object_estimates", ())
    if not isinstance(estimates, Sequence) or isinstance(estimates, (str, bytes)) or len(estimates) > 16:
        raise ValueError("observation.object_estimates must be a sequence of at most 16 estimates")
    if estimates:
        public_estimates = []
        for estimate in estimates:
            item = _mapping(estimate, "object estimate")
            public: dict[str, Any] = {}
            for key in ("target_id", "class_name"):
                if key in item:
                    public[key] = _text(item[key], f"object_estimate.{key}")
            if "pose" in item:
                public["pose"] = _pose(item["pose"], "object_estimate.pose")
            if "confidence" in item:
                confidence = _number(item["confidence"], "object_estimate.confidence")
                if not 0 <= confidence <= 1:
                    raise ValueError("object_estimate.confidence must be in [0, 1]")
                public["confidence"] = confidence
            public_estimates.append(public)
        result["object_estimates"] = public_estimates
    return result, images


def _project_task(value: Any) -> tuple[dict[str, Any], set[str], dict[str, float]]:
    source = _mapping(value, "task_spec")
    result: dict[str, Any] = {}
    for key in _TASK_TEXT_FIELDS & source.keys():
        result[key] = _text(source[key], f"task_spec.{key}")
    targets = source.get("target_ids", ())
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)) or len(targets) > 16:
        raise ValueError("task_spec.target_ids must be a sequence of at most 16 ids")
    target_ids = {_text(target, "task_spec.target_ids item") for target in targets}
    if "target_id" in result:
        target_ids.add(result["target_id"])
    if target_ids:
        result["target_ids"] = sorted(target_ids)
    if "nominal_pose" in source:
        result["nominal_pose"] = _pose(source["nominal_pose"], "task_spec.nominal_pose")
    if "assembly_direction" in source:
        result["assembly_direction"] = _numeric_tree(source["assembly_direction"], "task_spec.assembly_direction")
        if (not isinstance(result["assembly_direction"], list) or len(result["assembly_direction"]) != 3
                or any(not isinstance(item, float) for item in result["assembly_direction"])):
            raise ValueError("task_spec.assembly_direction must contain 3 numeric values")

    dimensions = _mapping(source.get("nominal_dimensions", {}), "task_spec.nominal_dimensions")
    public_dimensions = {
        key: _number(dimensions[key], f"nominal_dimensions.{key}")
        for key in _DIMENSION_FIELDS & dimensions.keys()
    }
    if public_dimensions:
        result["nominal_dimensions"] = public_dimensions

    supplied_limits = _mapping(source.get("tool_limits", {}), "task_spec.tool_limits")
    limits = dict(_TOOL_LIMITS)
    for key in _LIMIT_FIELDS & supplied_limits.keys():
        value = _number(supplied_limits[key], f"tool_limits.{key}")
        if value <= 0:
            raise ValueError(f"tool_limits.{key} must be positive")
        limits[key] = min(limits[key], value)
    result["tool_limits"] = limits
    return result, target_ids, limits


def _duration(value: Any, name: str, maximum: float) -> float:
    duration = _number(value, name)
    if not 0 < duration <= maximum:
        raise ValueError(f"{name} must be in (0, {maximum}]")
    return duration


def _validate_tool_call(
    value: Any,
    *,
    completed: bool,
    target_ids: set[str],
    limits: Mapping[str, float],
    available_evidence_refs: set[str] | None = None,
    enforce_completion: bool = True,
) -> dict[str, Any]:
    call = _mapping(value, "tool call")
    tool = call.get("tool")
    if not isinstance(tool, str) or tool not in _TOOL_SCHEMA:
        raise ValueError(f"unknown tool: {tool!r}")
    fields = {
        "observe": {"tool"}, "check_task": {"tool"}, "stop": {"tool"},
        "nudge": {"tool", "frame", "direction", "step_class", "max_duration_s"},
        "retract": {"tool", "direction", "distance_class", "max_duration_s"},
        "execute_skill": {"tool", "skill", "target_id", "mode", "max_duration_s"},
        "send_help_request": {
            "tool", "target", "operation", "allowed_scope", "desired_postconditions",
            "evidence_refs", "observed_problem",
        },
    }[tool]
    if tool == "execute_skill":
        required_fields = fields - {"max_duration_s"}
        if not required_fields <= set(call) or not set(call) <= fields:
            raise ValueError(
                "execute_skill tool call must contain tool, skill, target_id, and mode; "
                "max_duration_s is optional"
            )
    elif set(call) != fields:
        raise ValueError(f"{tool} tool call must contain exactly: {', '.join(sorted(fields))}")
    result = {"tool": tool}
    if tool == "nudge":
        if not isinstance(call["frame"], str) or call["frame"] not in {"assembly", "world"}:
            raise ValueError("nudge.frame must be 'assembly' or 'world'")
        if not isinstance(call["direction"], str) or call["direction"] not in _DIRECTIONS:
            raise ValueError("nudge.direction is not an allowed axis direction")
        if not isinstance(call["step_class"], str) or call["step_class"] not in {"coarse", "fine", "contact"}:
            raise ValueError("nudge.step_class must be coarse, fine, or contact")
        result.update(frame=call["frame"], direction=call["direction"], step_class=call["step_class"],
                      max_duration_s=_duration(call["max_duration_s"], "nudge.max_duration_s", limits["nudge_max_duration_s"]))
    elif tool == "retract":
        if not isinstance(call["direction"], str) or call["direction"] not in _DIRECTIONS:
            raise ValueError("retract.direction is not an allowed axis direction")
        if not isinstance(call["distance_class"], str) or call["distance_class"] not in {"short", "full"}:
            raise ValueError("retract.distance_class must be short or full")
        result.update(direction=call["direction"], distance_class=call["distance_class"],
                      max_duration_s=_duration(call["max_duration_s"], "retract.max_duration_s", limits["retract_max_duration_s"]))
    elif tool == "execute_skill":
        skill = call["skill"]
        if not isinstance(skill, str) or skill not in _SKILL_MODES:
            raise ValueError(f"unsupported execute_skill skill: {skill!r}")
        target_id = _text(call["target_id"], "execute_skill.target_id")
        if target_id not in target_ids:
            raise ValueError("execute_skill.target_id must be declared by task_spec")
        if call["mode"] != _SKILL_MODES[skill]:
            raise ValueError(f"execute_skill mode for {skill} must be {_SKILL_MODES[skill]!r}")
        if enforce_completion and skill == "release_and_retract" and completed is not True:
            raise ValueError("release_and_retract requires release_authorized=True from check_task")
        result.update(skill=skill, target_id=target_id, mode=call["mode"],
                      max_duration_s=_duration(call.get("max_duration_s", limits["skill_max_duration_s"]),
                                                "execute_skill.max_duration_s", limits["skill_max_duration_s"]))
    elif tool == "send_help_request":
        target = _text(call["target"], "send_help_request.target")
        if target not in target_ids:
            raise ValueError("send_help_request.target must be declared by task_spec")
        postconditions = call["desired_postconditions"]
        if (not isinstance(postconditions, Sequence) or isinstance(postconditions, (str, bytes))
                or not 1 <= len(postconditions) <= 8):
            raise ValueError("send_help_request.desired_postconditions must contain 1 to 8 items")
        postconditions = [_text(item, "send_help_request.desired_postconditions item") for item in postconditions]
        refs = call["evidence_refs"]
        if (not isinstance(refs, Sequence) or isinstance(refs, (str, bytes))
                or not 1 <= len(refs) <= 8):
            raise ValueError("send_help_request.evidence_refs must contain 1 to 8 items")
        refs = [_text(item, "send_help_request.evidence_refs item") for item in refs]
        if len(set(refs)) != len(refs):
            raise ValueError("send_help_request.evidence_refs must be unique")
        if not set(refs) <= (available_evidence_refs or set()):
            raise ValueError("send_help_request.evidence_refs must cite supplied public observation/history")
        result.update(
            target=target,
            operation=_text(call["operation"], "send_help_request.operation"),
            allowed_scope=_text(call["allowed_scope"], "send_help_request.allowed_scope"),
            desired_postconditions=postconditions,
            evidence_refs=refs,
            observed_problem=_text(call["observed_problem"], "send_help_request.observed_problem"),
        )
    return result


def _project_history(value: Any, *, completed: bool, target_ids: set[str], limits: Mapping[str, float]) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("tool_history must be a sequence")
    history = []
    release_completed = False
    for event in value[-_MAX_HISTORY:]:
        item = _mapping(event, "tool_history item")
        call = _validate_tool_call(
            item.get("call"), completed=completed, target_ids=target_ids, limits=limits,
            enforce_completion=False,
        )
        result = _mapping(item.get("result"), "tool_history result")
        if call["tool"] == "check_task":
            if set(result) != {"completed"} or type(result["completed"]) is not bool:
                raise ValueError("check_task history result must contain only a boolean completed field")
            if release_completed:
                public_result = {"phase": "post_release", "task_success": result["completed"]}
            else:
                public_result = {"phase": "pre_release", "release_authorized": result["completed"]}
        else:
            status = result.get("status")
            if not isinstance(status, str) or status not in _TOOL_STATUSES:
                raise ValueError("tool result status is not public or recognized")
            public_result = {"status": status}
            if "observation" in result:
                public_result["observation"] = _project_observation(result["observation"])[0]
        history.append({"call": call, "result": public_result})
        if (call["tool"] == "execute_skill" and call.get("skill") == "release_and_retract"
                and public_result.get("status") in _SUCCESSFUL_STATUSES):
            release_completed = True
    return history


def _available_evidence_refs(observation: Mapping[str, Any], history: Sequence[Mapping[str, Any]]) -> set[str]:
    refs = set()

    def add_observation(item: Any) -> None:
        if isinstance(item, Mapping) and isinstance(item.get("observation_id"), str):
            refs.add(f"observation:{item['observation_id']}")

    add_observation(observation)
    for index, event in enumerate(history):
        refs.add(f"tool_history:{index}")
        result = event.get("result") if isinstance(event, Mapping) else None
        if isinstance(result, Mapping):
            add_observation(result.get("observation"))
    return refs


def _project_public_progress(history: Sequence[Mapping[str, Any]], *, current_check_signal: bool) -> dict[str, Any]:
    completed_skills = []
    seen_skills = set()
    for event in history:
        call = event["call"]
        result = event["result"]
        skill = call.get("skill")
        if (call["tool"] == "execute_skill" and result.get("status") in _SUCCESSFUL_STATUSES
                and skill not in seen_skills):
            completed_skills.append(skill)
            seen_skills.add(skill)

    check_phase = "post_release" if "release_and_retract" in seen_skills else "pre_release"
    latest_check = next(
        (event["result"] for event in reversed(history) if event["call"]["tool"] == "check_task"),
        None,
    )
    progress: dict[str, Any] = {
        "completed_skills": completed_skills,
        "check_phase": check_phase,
        "release_authorized": current_check_signal if check_phase == "pre_release" else None,
        "task_success": current_check_signal if check_phase == "post_release" else None,
        "latest_boolean_check_result": latest_check,
        "last_call": None,
        "last_result": None,
    }
    if history:
        last = history[-1]
        call = last["call"]
        progress["last_call"] = {"tool": call["tool"]}
        if call["tool"] == "execute_skill":
            progress["last_call"]["skill"] = call["skill"]
        progress["last_result"] = dict(last["result"])
    return progress


def parse_tool_call(
    content: str,
    *,
    completed: bool = False,
    task_spec: Mapping[str, Any] | None = None,
    available_evidence_refs: set[str] | None = None,
) -> dict[str, Any]:
    """Parse JSON or a whole JSON fence; unwrap one JSON-string layer only if it contains an object."""
    if not isinstance(content, str):
        raise ValueError("model tool response must be a JSON string")
    decoded = _decode_tool_json(content)
    if isinstance(decoded, str):
        try:
            decoded = json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise ValueError("model response string does not contain a JSON object") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError("model tool response must be a JSON object")
    if type(completed) is not bool:
        raise ValueError("completed must be a boolean")
    _, target_ids, limits = _project_task(task_spec if task_spec is not None else {})
    return _validate_tool_call(
        decoded, completed=completed, target_ids=target_ids, limits=limits,
        available_evidence_refs=available_evidence_refs,
    )


def _decode_tool_json(content: str) -> Any:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[0].strip().lower() not in {"```", "```json"} or lines[-1].strip() != "```":
            raise ValueError("model tool response must be a whole JSON object or JSON fence")
        text = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model tool response is malformed JSON: {exc.msg}") from exc


def _tool_system_prompt(
    target_ids: set[str], validation_error: str | None = None,
    available_evidence_refs: set[str] | None = None,
    preferred_evidence_ref: str | None = None,
) -> str:
    examples = [{"tool": "observe"}]
    if target_ids:
        target_id = sorted(target_ids)[0]
        examples.extend(
            {"tool": "execute_skill", "skill": skill, "target_id": target_id, "mode": mode}
            for skill, mode in _SKILL_MODES.items()
        )
    examples.extend([
        {"tool": "nudge", "frame": "world", "direction": "x_positive",
         "step_class": "fine", "max_duration_s": MAX_NUDGE_DURATION_S},
        {"tool": "retract", "direction": "z_positive", "distance_class": "short",
         "max_duration_s": MAX_RETRACT_DURATION_S},
        {"tool": "check_task"},
    ])
    help_evidence_ref = None
    if available_evidence_refs:
        if preferred_evidence_ref in available_evidence_refs:
            help_evidence_ref = preferred_evidence_ref
        else:
            help_evidence_ref = min(available_evidence_refs)
    if target_ids and help_evidence_ref is not None:
        examples.append({
            "tool": "send_help_request", "target": sorted(target_ids)[0],
            "operation": "request assembly assistance", "allowed_scope": "specified assembly only",
            "desired_postconditions": ["bolt remains held and insertion can be safely reassessed"],
            "evidence_refs": [help_evidence_ref],
            "observed_problem": "Insertion has not completed; cause is unverified.",
        })
    examples.append({"tool": "stop"})
    prompt = (
        _SYSTEM_PROMPT
        + " Exact valid JSON tool-call shape examples only (not action recommendations; choose from public state; each line is one JSON object):\n"
        + "\n".join(json.dumps(example, ensure_ascii=True, separators=(",", ":")) for example in examples)
    )
    if validation_error is not None:
        diagnostic = json.dumps(validation_error.replace("\n", " ")[:240], ensure_ascii=True)
        prompt += (
            "\nYour previous response failed tool validation. Validation error (diagnostic only): "
            + diagnostic
            + ". Correct that schema error and return one valid tool-call JSON object matching one of the examples and listed fields."
        )
    return prompt


class BoltHarnessAgent:
    """OpenAI-compatible model client. Inputs are public data mappings, never an env."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        api_key_env: str = "HRC_M1_API_KEY",
        timeout_s: float = 30.0,
        trace_callback: Callable[[dict[str, Any]], None] | None = None,
        redact_images: bool = True,
    ) -> None:
        self.endpoint = _text(endpoint, "endpoint")
        self.model = _text(model, "model")
        self.api_key_env = _text(api_key_env, "api_key_env")
        self.timeout_s = _number(timeout_s, "timeout_s")
        if not 0 < self.timeout_s <= 120:
            raise ValueError("timeout_s must be in (0, 120]")
        if type(redact_images) is not bool:
            raise ValueError("redact_images must be a boolean")
        self.trace_callback = trace_callback
        self.redact_images = redact_images

    def _trace(self, event: dict[str, Any]) -> None:
        if self.trace_callback is not None:
            self.trace_callback(event)

    def _trace_request(self, payload: dict[str, Any], images: list[tuple[str, str]], attempt: int) -> None:
        frame_names = iter(name for name, _ in images)
        content = []
        for part in payload["messages"][1]["content"]:
            if part.get("type") == "image_url":
                url = f"frame://{next(frame_names)}" if self.redact_images else part["image_url"]["url"]
                content.append({"type": "image_url", "image_url": {"url": url}})
            else:
                content.append(dict(part))
        traced = {
            **payload,
            "response_format": dict(payload["response_format"]),
            "messages": [dict(payload["messages"][0]), {**payload["messages"][1], "content": content}],
        }
        self._trace({"event": "request", "attempt": attempt, "payload": traced})

    def next_tool_call(
        self,
        observation: Mapping[str, Any],
        task_spec: Mapping[str, Any],
        tool_history: Sequence[Mapping[str, Any]] = (),
        *,
        completed: bool = False,
    ) -> dict[str, Any]:
        """Send allowlisted public context and return one validated executor tool call.

        `observation` accepts observation_id, frames, numeric sensor/robot fields,
        two boolean finger contacts, and object_estimates. `task_spec` accepts
        task/target ids, goal, nominal geometry and optional numeric tool_limits.
        History entries are `{call, result}`; a check_task result is exactly
        `{completed: bool}`.
        If `trace_callback` is set, it receives request payload and raw response events;
        auth headers are never traced and image URLs are frame IDs by default.
        """
        if type(completed) is not bool:
            raise ValueError("completed must be a boolean")
        public_observation, images = _project_observation(observation)
        public_task, target_ids, limits = _project_task(task_spec)
        public_history = _project_history(tool_history, completed=completed, target_ids=target_ids, limits=limits)
        available_evidence_refs = _available_evidence_refs(public_observation, public_history)
        preferred_evidence_ref = (
            f"observation:{public_observation['observation_id']}"
            if "observation_id" in public_observation else None
        )
        public_progress = _project_public_progress(public_history, current_check_signal=completed)
        context = {
            "task_spec": public_task,
            "observation": public_observation,
            "tool_history": public_history,
            "public_progress": public_progress,
            "pose_format": "position [x,y,z] in metres; quaternion [w,x,y,z]",
            "tools": _TOOL_SCHEMA,
        }
        content: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(context, ensure_ascii=True)}]
        for camera, uri in images:
            content.extend([
                {"type": "text", "text": f"Public RGB frame: {camera}"},
                {"type": "image_url", "image_url": {"url": uri}},
            ])
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 256,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _tool_system_prompt(
                    target_ids, available_evidence_refs=available_evidence_refs,
                    preferred_evidence_ref=preferred_evidence_ref,
                )},
                {"role": "user", "content": content},
            ],
        }
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        for attempt in (1, 2):
            self._trace_request(payload, images, attempt)
            request = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    response_bytes = response.read()
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise BoltAgentError(f"model request failed: {type(exc).__name__}") from exc

            response_text = response_bytes.decode("utf-8", errors="replace")
            self._trace({"event": "response", "attempt": attempt, "raw_response": response_text})
            try:
                raw = json.loads(response_text)
                response_content = raw["choices"][0]["message"]["content"]
                return parse_tool_call(
                    response_content, completed=completed, task_spec=task_spec,
                    available_evidence_refs=available_evidence_refs,
                )
            except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
                if attempt == 2:
                    raise BoltAgentError("model returned an invalid tool response after one retry") from exc
                payload["messages"][0]["content"] = _tool_system_prompt(
                    target_ids, validation_error=str(exc), available_evidence_refs=available_evidence_refs,
                    preferred_evidence_ref=preferred_evidence_ref,
                )
        raise BoltAgentError("model returned no tool response")


__all__ = ["BoltAgentError", "BoltHarnessAgent", "parse_tool_call"]
