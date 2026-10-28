"""Task-agnostic VLM interface for runtime assembly goals."""

from __future__ import annotations

import base64
import json
import math
import os
import re
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import numpy as np

from .connector import has_observed_aperture_geometry, has_observed_grasp_contact_geometry


MAX_FEATURES = 64
MAX_HISTORY = 8
MAX_TEXT = 1024
MAX_CALLS = 48
_IMAGE_LIMIT_BYTES = 16 * 1024 * 1024
_MODEL_IMAGE_MAX_SIDE = 896
_MODEL_HISTORY_IMAGE_MAX_SIDE = 384
_MODEL_SEQUENCE_TOTAL_PIXEL_LIMIT = 2_100_000
_MODEL_LOCAL_CROP_MAX_COUNT = 6
_MODEL_LOCAL_CROP_MAX_SIDE = 384
_MODEL_LOCAL_CROP_IMAGE_SIDE = 280
_MODEL_IMAGE_MIN_SIDE = 64
_MODEL_IMAGE_MAX_ALLOWED_SIDE = 4096
_MODEL_OUTPUT_MIN_TOKENS = 1
_MODEL_OUTPUT_MAX_TOKENS = 768
_MAX_MOVE_TCP_STEPS = 2000
_MAX_MOVE_TCP_DURATION_S = 30.0
_DEFAULT_MOVE_TCP_STEPS = 200
_DEFAULT_MOVE_TCP_DURATION_S = 3.0
_COLLISION_RECOVERY_RETRY_TEMPERATURE = 0.8
_MIN_MOVE_TCP_TRANSLATION_M = 0.001
_MIN_MOVE_TCP_ROTATION_RAD = 0.02
_GRASP_RECOVERY_MIN_LATERAL_M = 0.01
_HELD_OBJECT_SURFACE_MAX_DISTANCE_M = 0.06
_MODEL_HISTORY_DUPLICATE_RESULT_FIELDS = frozenset({"resolved_goal", "compiled_goal"})
_MAX_LOCAL_PLANE_REFERENCE_EXTENT_M = 0.3
# shortcut: plane targets must stay within 0.3m of the held part in height; expand only for verified tall assemblies.
_MAX_HELD_PLANE_VERTICAL_OFFSET_M = 0.3
_REGIMES = {"free_space", "contact", "grasp", "release"}
_HOLLOW_VESSEL_MOVING_TARGET_RE = re.compile(
    r"\b(?:pick\s+up|put|place|move|lift|grasp|take)\s+(?:the\s+|a\s+|an\s+)?"
    r"(?:bowl|cup|mug|glass|container)\b",
    re.IGNORECASE,
)
_RELATIONS = {
    "coaxial": "Align moving and reference axes on the same line; parallel or antiparallel axis directions are equivalent, and the connector chooses the nearest sign to the current pose. Both features need position_m, axis, position_std_m, and axis_std_rad.",
    "parallel": "Make moving and reference axes parallel; both features need position_m, axis, position_std_m, and axis_std_rad.",
    "centered_on": "Place the moving feature position at the reference feature position; both need position_m and position_std_m.",
    "point_on_plane": "Project the moving point onto a reference plane, preserving its tangential offset; this does not align it to the reference center or prove contact. The plane may be virtual for a measured aperture. Both need position_m and position_std_m; the reference needs normal and axis_std_rad.",
    "remain_clear_of": "Maintain non-contact clearance. The moving feature needs position_m, position_std_m, and bounding_radius_m. An object reference also needs bounding_radius_m; a plane/surface reference instead needs normal, extent_m, and axis_std_rad, with its finite extent used for support clearance.",
}
_PLANE_NORMAL_KINDS = {
    "rgbd_plane_normal", "surface_patch_normal_not_hole_axis",
    "local_depth_surface_normal", "fitted_rim_surface_normal",
}
SUPPORTED_RELATIONS = tuple(sorted(_RELATIONS))
_OBSERVATION_STATE_FIELDS = {
    "joint_positions", "joint_velocities", "tcp_pose", "commanded_tcp_pose", "tcp_velocity", "gripper_width",
    "wrench", "camera_intrinsics", "camera_extrinsics", "camera_depth_path",
    "camera_depth_convention", "camera_frame_id", "finger_contacts", "finger_bolt_contacts",
    "finger_contact_force_n",
    "rgbd_obstacle_map", "gripper_width_m", "grasped", "camera_observation_id",
    "release_authorized", "rgbd_source_observation_id", "rgbd_source_timestamp_s", "robot_timestamp_s",
}
_TOOLS = {
    "observe": {"arguments": {"observation_id": "optional current observation_id"}},
    "grasp": {
        "arguments": {
            "observation_id": "the current observation_id",
            "target_feature_ref": "one current graspable_candidate_refs entry selected from the image",
        },
        "semantics": (
            "Request one geometry-checked grasp of the model-selected current feature; the adapter binds the unique "
            "current TCP feature. A graspable candidate only has RGB-D-supported finger-contact geometry; it does "
            "not identify the task object or make an interior container surface a valid grasp target. For a visibly "
            "hollow vessel, select its exterior wall, never the interior floor or empty opening."
        ),
    },
    "move_tcp": {
        "arguments": {
            "observation_id": "the current observation_id",
            "frame": "must be 'tcp'; deltas are expressed in the starting TCP frame",
            "translation_delta_m": (
                "three finite TCP-local translation components in meters; for grasp-collision recovery, one "
                "approach-axis retreat may be used to withdraw, but further recovery must move laterally around "
                "the blocker along a direction supported by visible clearance; the normal sweep checker decides safety"
            ),
            "rotation_vector_rad": "optional three finite TCP-local axis-angle components in radians; defaults to zero and norm is at most pi",
            "max_duration_s": "optional finite positive duration cap in seconds, at most 30; defaults to 3",
            "motion_budget": "optional integer control-step cap in [1, 2000]; defaults to 200 and is capped by executor limits",
        },
        "semantics": "Compute one target from the cited observation when public grasped=false; rotate local translation by the starting TCP orientation and right-multiply local rotation onto its wxyz quaternion. The executor caps steps by the model budget, duration, and calculated motion budget. This tool never commands the gripper.",
    },
    "execute_goal": {
        "arguments": {
            "observation_id": "the current observation_id",
            "intent": "a concise action objective grounded in the current task and observation",
            "constraints": [{
                "relation": {"type": "string", "enum": list(SUPPORTED_RELATIONS), "semantics": _RELATIONS},
                "moving_feature": "a feature_ref from the current observation",
                "reference_feature": "a feature_ref from the current observation",
            }],
            "motion_regime": {
                "enum": sorted(_REGIMES),
                "semantics": {
                    "free_space": "Bounded motion without intentional contact.",
                    "contact": "Bounded force-monitored motion. Solid-surface contact requires samples at the predicted contact point. Aperture entry instead requires observed rim geometry and coaxial plus point_on_plane constraints to the same aperture, without any collision exemptions; it ends at the entry plane, not an inferred hidden insertion depth.",
                    "grasp": "The moving_feature must be a TCP-local robot FK gripper feature; the reference_feature must be an observed object with geometry-derived grasp_width_m. Do not assume attachment before bilateral public contact verification.",
                    "release": "Release only when the executor reports its native authorization condition.",
                },
            },
            "stop_condition": "a bounded condition for ending this operation",
        },
        "semantics": "Submit one runtime relation goal. The connector resolves it from current public perception; do not supply guessed world coordinates or a task-specific route.",
    },
    "check_task": {"arguments": {}, "returns": {"completed": "boolean only"}},
    "finish": {"arguments": {}, "semantics": "End only after check_task returned completed=true."},
    "stop": {"arguments": {}, "semantics": "Stop without claiming task completion."},
}
_SYSTEM_PROMPT = (
    "You select bounded manipulation goals from the current task and public observations. "
    "You may observe, request a geometry-checked grasp only when a current graspable candidate is available, "
    "submit one bounded TCP-local move_tcp adjustment only when public grasped=false, submit one relation-based "
    "execute_goal, check the independent task-completion boolean, finish after completed=true, or stop. "
    "No action sequence or task phase is prescribed. "
    "Every execute_goal must name the current observation_id and cite feature_ref values present in "
    "that observation. Express the desired relation, not guessed coordinates, hidden simulator state, "
    "or a prewritten route. The connector is responsible for resolving the relation using current "
    "perception and the executor's normal safety protections. Choose only goals supported by the "
    "visible images, public robot state, current feature estimates, task text, and prior public tool feedback. "
    "Manipulate only the object named by the task, not a visually similar distractor. Before grasping, "
    "visually match the candidate to the named object using the full scene and current crop; a feature's "
    "graspability or geometry is never evidence of semantic identity. If the match is uncertain, do not guess. "
    "A graspable candidate certifies local finger-contact geometry only, not semantic target identity or approach clearance. "
    "When the task object is a visibly hollow vessel such as a bowl or cup, grasp its exterior wall; never target the interior floor or empty opening. "
    "After any `collision_blocked` result, re-plan from the next fresh public RGB-D observation and current "
    "TCP/feature geometry. Never repeat the same physical grasp target by changing only observation or feature "
    "IDs. After a grasp approach is collision-blocked, do not submit another grasp until a bounded repositioning "
    "action has actually moved the TCP to a visibly different, clear approach and a fresh observation is available. "
    "Before the first grasp, inspect the visible corridor from the current TCP to the selected target. A grasp call "
    "does not plan a route around obstacles; it executes the selected geometric approach under the sweep checker. "
    "If a nearby object occupies that corridor, first choose a bounded lateral `move_tcp` preposition at a visibly "
    "clear height, use the current TCP-local axes to express it, refresh observation, and only then reassess grasp. "
    "Use the blocker pixel only as a landmark in the current image; when the image supports it, choose a lateral "
    "free-space waypoint around the visible obstruction rather than moving farther along the blocked approach. "
    "For that recovery, use any world or TCP-local clearance direction in collision feedback only as a candidate "
    "hint, never as proof that the route is clear. If the feedback includes an example TCP-local translation and "
    "fresh RGB-D shows room, use that exact vector as a bounded trial; otherwise choose a meaningful bounded move "
    "along the hinted direction (near 0.01 m is preferable to a 1-2 mm nudge) or retreat along the approach axis; "
    "do not repeat a direction rejected with zero motion, and let the normal sweep checker decide whether a candidate "
    "move is safe. "
    "A grasp-collision `example_0.01m_retreat_translation_delta_m` is parallel to the blocked approach: it may be "
    "used once to withdraw if current RGB-D shows room, but it is not a lateral route and must not be repeated "
    "instead of moving around the blocker. After that withdrawal, choose a visibly clear cross-approach direction; "
    "if feedback supplies an `example_0.02m_lateral_trial_translation_delta_m`, evaluate it as a candidate only "
    "when the fresh image shows room. It is not clearance proof; the same sweep, IK, and force checks decide safety. "
    "`move_tcp` translations are TCP-local: use `current_capabilities.tcp_local_axes_world` to determine which "
    "local component moves laterally in the world; do not assume local x/y are image-horizontal or world axes. "
    "A single-geometry clearance hint cannot establish that a route is clear of all obstacles; use the fresh image "
    "to judge whether its direction has room, then let the full sweep checker decide. Reassess from the automatically "
    "refreshed observation after each bounded move. Keep normal safety checks; do not invent coordinates or use "
    "contact/grasp until current evidence supports it. If no grounded recovery is supported, stop. "
    "Treat the zero-referenced public wrench only as secondary net load/contact evidence alongside visual changes; "
    "do not infer force direction, bilateral grasp, or task completion from it. Bilateral finger contacts define "
    "grasp state, and executor force-stop results are authoritative. Per-finger `finger_contact_force_n` values "
    "are direct measured finger-to-object contact forces; use them with the contact booleans as contact evidence, "
    "not as proof of lift or retention. "
    "When `frame_sequence` is present, the images are chronological distinct camera acquisitions; only "
    "the last feature-overlay frame corresponds to current feature refs and pixel geometry. Use earlier "
    "frames as temporal context, but never apply current refs to earlier frames. "
    "The only task-completion evaluator result is the boolean returned by check_task; "
    "release_authorized is a separate native release permission, not a completion result. A fresh "
    "observation is taken automatically after each tool call; do not call observe twice in a row. Use observe "
    "only to resolve a specific relevant public fact that is missing or stale; do not "
    "substitute a different target "
    "when the task specifies a visual relation. Return one raw, flat "
    "JSON object with a top-level tool field; do not use Markdown fences or nested arguments."
)
_CAPABILITY_GUIDANCE = (
    " Read the top-level `current_capabilities` before the feature list; it is derived only from current "
    "public state and geometry. To request a grasp, choose one `graspable_candidate_refs` entry with "
    "the `grasp` tool; the adapter requires exactly one current TCP ref and compiles that request to "
    "`centered_on` with `motion_regime=grasp`. In `execute_goal`, a grasp uses a TCP ref as moving_feature "
    "and a candidate ref as reference_feature. Request grasp only when `grasped` is explicitly false; "
    "if `grasped` is true, do not use `grasp` or `motion_regime=grasp`. "
    "`contact_geometry_unverified_refs` have a width estimate but current RGB-D does not support both "
    "predicted finger contact sites; they are not grasp candidates. Never pass one, or any non-candidate "
    "feature, to the `grasp` tool. If `graspable_candidate_refs` is empty, a grasp call cannot be valid. "
    "For a task that moves a hollow vessel itself, `aperture_interior_candidate_refs` lists current patches inside "
    "its observed opening and these are not valid exterior grasp targets. Choose from the remaining "
    "`graspable_candidate_refs`; for other tasks, use `grasp_candidate_aperture_relations` as geometry only, not "
    "object identity. "
    "Do not repurpose another feature as a grasp target. A graspable-candidate label confirms only local RGB-D "
    "finger-contact geometry, not semantic identity. For a visibly hollow or concave container such as a bowl, "
    "select a current candidate on the exterior wall with visible finger clearance; never grasp the interior floor "
    "or empty center of its opening, even if either appears in the candidate list. If no exterior candidate can be "
    "identified from the current image and overlay, observe for useful evidence or stop instead of guessing. "
    "When `grasped` is explicitly false and the current image and visible "
    "target geometry support a bounded TCP correction, you may use one `move_tcp` adjustment "
    "to improve the opening pose, then reassess from the fresh observation. Do not guess a fixed angle or "
    "position, repeat an unchanged adjustment, or move a held object with this tool. Otherwise observe only if "
    "a new acquisition can add relevant visual evidence; if not, stop and report incomplete. Do not guess "
    "geometry or repeat a rejected grasp "
    "without new visual evidence. "
    "true or unknown is not permission to open the gripper. `grasped=false` means no world-frame object "
    "is held. If `grasped` is true, inspect `held_object_surface_candidate_refs`: these are current "
    "non-self-filtered depth patches near the measured TCP, not semantic identities or object axes. Visually confirm "
    "the candidate lies on the held part; proximity alone is not identity evidence. Also inspect "
    "`tracked_held_surface_refs`: these are last-seen public RGB-D surfaces carried by measured TCP motion while "
    "bilateral finger-to-object contacts and positive per-finger forces persist; they are estimates, not current image evidence. "
    "A tracked surface may be used only for bounded `free_space` positioning, never for contact, release, axis evidence, "
    "or seating. Prefer a current visible held surface when available. When `grasped` is true, every "
    "`execute_goal.constraints[].moving_feature` must be one of "
    "`current_capabilities.held_object_surface_candidate_refs` or `current_capabilities.tracked_held_surface_refs` "
    "for `free_space` moves, "
    "never a `tcp_feature_refs` value or unrelated plane feature; if both lists are empty, stop. "
    "Contact and release require a current visible held surface. "
    "Use `aperture_image_relations` to compare public image bearings relative to the largest detected aperture candidate; "
    "these measurements do not label the task target, so choose the feature_ref that matches the task description. "
    "`reference_feature` is the target, not the held part. For aperture entry, use the "
    "visually selected value from `aperture_reference_refs`, and constrain one aperture per goal. All constraints "
    "in one goal must use the same moving_feature; "
    "do not combine a TCP-moving constraint with a held-part-moving constraint. "
    "An aperture center can be a correct localization target even when it has no solid depth samples. "
    "Use measured rim geometry to estimate its entry plane and axis. Approach and align with clearance "
    "from public geometry, then refresh evidence; do not aim at the rim merely to obtain solid samples. "
    "For solid held-part contact, use `point_on_plane` with that moving feature and a visible, task-relevant solid "
    "surface from `plane_reference_refs` at the expected contact point; for every `point_on_plane` relation, "
    "the reference must be in `plane_reference_refs` or `aperture_reference_refs`. The latter are virtual "
    "entry planes, not solid contact surfaces. Force-monitored aperture entry needs both `coaxial` and "
    "`point_on_plane` to the same current aperture and the same held-part moving feature; `coaxial` needs a "
    "reliable measured moving-part axis. Make non-contact approach/alignment decisions separately and refresh "
    "observation before entry. Surface-patch PCA is not a part axis; if no reliable axis is visible, do not "
    "propose contact entry. It grants no wall-contact exemption and no hidden insertion depth. "
    "Broad workspace/table planes are excluded and are not "
    "insertion targets. Circular-candidate centers are not solid surface references; an empty aperture center is not a "
    "contact surface, and `centered_on` alone does not authorize contact. "
    "These refs indicate geometry availability only: visually select the task surface and exclude unrelated surfaces. Do not treat a "
    "surface patch's PCA axis as the axis of the held part or target. Exclude unrelated robot, table, and "
    "fixture surfaces. Release "
    "requires `release_authorized` true; authorization is not completion."
)


class AssemblyAgentError(RuntimeError):
    """Model transport or tool-response failure; no fallback action is substituted."""


def _text(value: Any, name: str, *, limit: int = MAX_TEXT) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be a non-empty string of at most {limit} characters")
    return value.strip()


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _bounded_int(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _numeric_vector(value: Any, name: str, size: int) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != size:
        raise ValueError(f"{name} must contain exactly {size} finite numbers")
    return [_number(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _tcp_local_axes_world(tcp_pose: Any) -> dict[str, list[float]]:
    pose = np.asarray(_numeric_vector(tcp_pose, "robot_state.sensors.tcp_pose", 7), dtype=np.float64)
    quaternion = pose[3:]
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-9:
        raise ValueError("robot_state.sensors.tcp_pose quaternion must have non-zero norm")
    w, x, y, z = quaternion / norm
    rotation = np.asarray((
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    ), dtype=np.float64)
    return {
        axis: np.round(rotation[:, index], 6).tolist()
        for index, axis in enumerate(("x", "y", "z"))
    }


def _json_value(value: Any, name: str, *, depth: int = 0) -> Any:
    if depth > 8:
        raise ValueError(f"{name} is nested too deeply")
    if value is None or type(value) is bool or isinstance(value, str):
        if isinstance(value, str) and len(value) > MAX_TEXT:
            raise ValueError(f"{name} contains an overlong string")
        return value
    if isinstance(value, (int, float)):
        return _number(value, name)
    if isinstance(value, Mapping):
        if len(value) > 128 or any(not isinstance(key, str) for key in value):
            raise ValueError(f"{name} must be a bounded string-keyed object")
        forbidden = [
            key for key in value
            if any(token in key.casefold() for token in ("ground_truth", "evaluator", "private"))
            or "gt" in key.casefold().split("_")
        ]
        if forbidden:
            raise ValueError(f"{name} contains a non-public field")
        return {key: _json_value(item, f"{name}.{key}", depth=depth + 1) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) > 512:
            raise ValueError(f"{name} is too long")
        return [_json_value(item, name, depth=depth + 1) for item in value]
    raise ValueError(f"{name} contains a non-JSON value")


def _frame_data_uri(
    value: Any,
    name: str,
    *,
    coordinate_maps: dict[str, dict[str, Any]] | None = None,
    max_side: int = _MODEL_IMAGE_MAX_SIDE,
) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"frame {name!r} must be an image path or data URI")
    if value.startswith("data:image/"):
        if len(value) > _IMAGE_LIMIT_BYTES * 2:
            raise ValueError(f"frame {name!r} exceeds the image size bound")
        return value
    path = Path(value)
    if not path.is_file():
        raise ValueError(f"frame {name!r} is not an existing image path or data URI")
    if path.stat().st_size > _IMAGE_LIMIT_BYTES:
        raise ValueError(f"frame {name!r} exceeds the image size bound")
    suffix = path.suffix.lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(suffix)
    if mime is None:
        raise ValueError(f"frame {name!r} must have a PNG or JPEG extension")
    image_bytes = path.read_bytes()
    try:
        import cv2
    except ImportError:
        pass
    else:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            height, width = image.shape[:2]
            model_width, model_height = width, height
            longest_side = max(width, height)
            if longest_side > max_side:
                scale = max_side / longest_side
                model_width = max(1, round(width * scale))
                model_height = max(1, round(height * scale))
                resized = cv2.resize(
                    image,
                    (model_width, model_height),
                    interpolation=cv2.INTER_AREA,
                )
                success, encoded = cv2.imencode(suffix, resized)
                if not success:
                    raise ValueError(f"frame {name!r} could not be resized for model input")
                image_bytes = encoded.tobytes()
            if coordinate_maps is not None:
                coordinate_maps[name] = {
                    "source_size_px": [width, height],
                    "model_image_size_px": [model_width, model_height],
                    "scale_xy": [model_width / width, model_height / height],
                }
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"


def _project_frame_sequence(value: Mapping[str, Any], frames: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    raw_sequence = value.get("frame_sequence")
    if raw_sequence is None:
        return None
    if not isinstance(raw_sequence, Sequence) or isinstance(raw_sequence, (str, bytes)) or len(raw_sequence) != 8:
        raise ValueError("observation.frame_sequence must contain exactly eight camera acquisitions")

    sequence: list[dict[str, Any]] = []
    frame_keys: set[str] = set()
    frame_ids: set[tuple[type, Any]] = set()
    timestamps: set[float] = set()
    previous_timestamp = -math.inf
    for index, item in enumerate(raw_sequence):
        if not isinstance(item, Mapping) or set(item) != {"frame_key", "frame_id", "timestamp_s"}:
            raise ValueError("each frame_sequence entry must contain frame_key, frame_id, and timestamp_s")
        frame_key = _text(item["frame_key"], "frame_sequence.frame_key", limit=128)
        frame_id = item["frame_id"]
        if isinstance(frame_id, bool) or not isinstance(frame_id, (str, int)):
            raise ValueError("frame_sequence.frame_id must be a non-empty string or integer")
        if isinstance(frame_id, int):
            if frame_id < 0:
                raise ValueError("frame_sequence.frame_id must be non-negative")
        else:
            frame_id = _text(frame_id, "frame_sequence.frame_id", limit=160)
        identity = (type(frame_id), frame_id)
        if identity in frame_ids:
            raise ValueError("frame_sequence.frame_id values must be unique")
        frame_ids.add(identity)

        timestamp_s = _number(item["timestamp_s"], "frame_sequence.timestamp_s")
        if timestamp_s < 0 or timestamp_s <= previous_timestamp or timestamp_s in timestamps:
            raise ValueError("frame_sequence timestamps must be unique and strictly chronological")
        timestamps.add(timestamp_s)
        previous_timestamp = timestamp_s
        if frame_key in frame_keys:
            raise ValueError("frame_sequence.frame_key values must be unique")
        frame_keys.add(frame_key)
        is_overlay = frame_key.endswith("_feature_overlay")
        if (index == 7) != is_overlay:
            raise ValueError("only the latest frame_sequence entry may be the current feature overlay")
        sequence.append({"frame_key": frame_key, "frame_id": frame_id, "timestamp_s": timestamp_s})

    if frame_keys != set(frames):
        raise ValueError("frame_sequence keys must match all eight observation.frames keys")
    frame_paths = [frames[item["frame_key"]] for item in sequence]
    if any(not isinstance(frames[item["frame_key"]], str)
           or frames[item["frame_key"]].startswith("data:image/") for item in sequence):
        raise ValueError("sequenced frames must be persisted image paths so resize maps can be recorded")
    if len(set(frame_paths)) != 8:
        raise ValueError("frame_sequence requires eight distinct persisted image paths")

    latest = sequence[-1]
    state = value.get("robot_state", {})
    sensors = state.get("sensors", {}) if isinstance(state, Mapping) else {}
    if isinstance(sensors, Mapping):
        current_frame_id = sensors.get("camera_frame_id")
        if current_frame_id is not None and current_frame_id != latest["frame_id"]:
            raise ValueError("latest frame_sequence frame_id must match the current observation camera_frame_id")
        current_timestamp = value.get("timestamp")
        if current_timestamp is not None and abs(_number(current_timestamp, "observation.timestamp") - latest["timestamp_s"]) > 1e-6:
            raise ValueError("latest frame_sequence timestamp must match the current observation timestamp")
    return sequence


def project_observation(
    value: Any,
    *,
    model_image_max_side: int = _MODEL_IMAGE_MAX_SIDE,
    model_history_image_max_side: int = _MODEL_HISTORY_IMAGE_MAX_SIDE,
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """Project the perception adapter output onto a small public allowlist."""
    model_image_max_side = _bounded_int(
        model_image_max_side,
        "model_image_max_side",
        minimum=_MODEL_IMAGE_MIN_SIDE,
        maximum=_MODEL_IMAGE_MAX_ALLOWED_SIDE,
    )
    model_history_image_max_side = _bounded_int(
        model_history_image_max_side,
        "model_history_image_max_side",
        minimum=_MODEL_IMAGE_MIN_SIDE,
        maximum=_MODEL_IMAGE_MAX_ALLOWED_SIDE,
    )
    if model_history_image_max_side > model_image_max_side:
        raise ValueError("model_history_image_max_side must not exceed model_image_max_side")
    if not isinstance(value, Mapping):
        raise TypeError("observation must be a public mapping")
    observation_id = _text(value.get("observation_id"), "observation_id", limit=160)
    result: dict[str, Any] = {"observation_id": observation_id}
    if "timestamp" in value:
        result["timestamp"] = _number(value["timestamp"], "observation.timestamp")

    frames = value.get("frames")
    if not isinstance(frames, Mapping) or not 1 <= len(frames) <= 8:
        raise ValueError("observation.frames must contain one to eight public RGB frames")
    frame_sequence = _project_frame_sequence(value, frames)
    images = []
    frame_paths = {}
    image_coordinate_maps: dict[str, dict[str, Any]] = {}
    camera_order = (
        [item["frame_key"] for item in frame_sequence]
        if frame_sequence is not None else list(frames)
    )
    latest_overlay = frame_sequence[-1]["frame_key"] if frame_sequence is not None else None
    for camera in camera_order:
        frame = frames[camera]
        camera = _text(camera, "frame camera", limit=128)
        frame_paths[camera] = _text(frame, f"frame path {camera}", limit=MAX_TEXT)
        max_side = (
            model_image_max_side
            if frame_sequence is None or camera == latest_overlay else model_history_image_max_side
        )
        images.append((camera, _frame_data_uri(
            frame, camera, coordinate_maps=image_coordinate_maps, max_side=max_side,
        )))
    overlay_images = [item for item in images if item[0].endswith("_feature_overlay")]
    if overlay_images and frame_sequence is None:
        images = overlay_images
    result["frames"] = frame_paths
    if frame_sequence is not None:
        result["frame_sequence"] = frame_sequence
        if set(image_coordinate_maps) != set(camera_order):
            raise ValueError("all sequenced images require recorded model resize maps")
        total_pixels = sum(
            size[0] * size[1]
            for item in image_coordinate_maps.values()
            for size in [item["model_image_size_px"]]
        )
        if total_pixels > _MODEL_SEQUENCE_TOTAL_PIXEL_LIMIT:
            raise ValueError("resized frame_sequence exceeds the total model image pixel budget")
    if image_coordinate_maps:
        result["image_coordinate_maps_by_camera"] = image_coordinate_maps
    for key in ("depth", "calibration"):
        if key in value:
            field_value = value[key]
            if not isinstance(field_value, Mapping):
                raise TypeError(f"observation.{key} must be a mapping")
            result[key] = _json_value(field_value, f"observation.{key}")

    state = value.get("robot_state", {})
    if not isinstance(state, Mapping):
        raise TypeError("observation.robot_state must be a mapping")
    nested_sensors = state.get("sensors", {})
    if not isinstance(nested_sensors, Mapping):
        raise TypeError("observation.robot_state.sensors must be a mapping")
    raw_sensors = {key: state[key] for key in _OBSERVATION_STATE_FIELDS & state.keys()}
    raw_sensors.update({key: nested_sensors[key] for key in _OBSERVATION_STATE_FIELDS & nested_sensors.keys()})
    public_sensors = {}
    for key, item in raw_sensors.items():
        if key == "rgbd_obstacle_map":
            public_sensors[key] = _public_obstacle_map(item, observation_id)
        elif key in {"finger_bolt_contacts", "finger_contacts"}:
            public_sensors["finger_contacts"] = _json_value(item, "robot_state.sensors.finger_contacts")
        elif key == "finger_contact_force_n":
            forces = _numeric_vector(item, "robot_state.sensors.finger_contact_force_n", 2)
            if any(force < 0.0 for force in forces):
                raise ValueError("robot_state.sensors.finger_contact_force_n must be non-negative")
            public_sensors[key] = forces
        elif key == "gripper_width_m":
            public_sensors["gripper_width"] = _number(item, "robot_state.sensors.gripper_width_m")
        elif key != "gripper_width":
            public_sensors[key] = _json_value(item, f"robot_state.sensors.{key}")
    if "finger_contacts" in public_sensors:
        contacts = public_sensors["finger_contacts"]
        if not isinstance(contacts, list) or len(contacts) != 2 or any(type(item) is not bool for item in contacts):
            raise ValueError("robot_state.sensors.finger_contacts must contain two booleans")
    result["robot_state"] = {"sensors": public_sensors}

    features = value.get("features", ())
    if not isinstance(features, Sequence) or isinstance(features, (str, bytes)) or len(features) > MAX_FEATURES:
        raise ValueError(f"observation.features must contain at most {MAX_FEATURES} current estimates")
    public_features = []
    refs = set()
    for feature in features:
        if not isinstance(feature, Mapping):
            raise TypeError("each feature estimate must be a mapping")
        allowed = {
            "feature_ref", "kind", "description", "frame", "position", "axis", "geometry",
            "uncertainty", "source_observation_id",
        }
        public = {key: feature[key] for key in allowed if key in feature}
        ref = _text(public.get("feature_ref"), "feature.feature_ref", limit=160)
        if ref in refs:
            raise ValueError("feature_ref values must be unique within an observation")
        refs.add(ref)
        source_id = _text(public.get("source_observation_id"), "feature.source_observation_id", limit=160)
        if source_id != observation_id:
            raise ValueError("feature estimates must cite the current source_observation_id")
        item = {
            "feature_ref": ref,
            "source_observation_id": source_id,
        }
        for key in ("kind", "description", "frame"):
            if key in public:
                item[key] = _text(public[key], f"feature.{key}", limit=MAX_TEXT)
        source_uncertainty = public.get("uncertainty", {})
        if not isinstance(source_uncertainty, Mapping):
            raise TypeError("feature.uncertainty must be a mapping")
        geometry = dict(public.get("geometry", {})) if isinstance(public.get("geometry", {}), Mapping) else None
        if geometry is None:
            raise TypeError("feature.geometry must be a mapping")
        if "position_m" not in geometry:
            if "position" in geometry:
                geometry["position_m"] = geometry["position"]
            elif "position" in public:
                geometry["position_m"] = public["position"]
        # The connector requires an angular uncertainty whenever it consumes an
        # axis or normal. Keep unqualified estimates public but do not promote them.
        if "axis_std_rad" in source_uncertainty:
            if "axis" not in geometry:
                if "axis_unit" in geometry:
                    geometry["axis"] = geometry["axis_unit"]
                elif "principal_axis_unit" in geometry:
                    geometry["axis"] = geometry["principal_axis_unit"]
            elif "axis" not in geometry and "axis" in public:
                geometry["axis"] = public["axis"]
            if "normal" not in geometry:
                for key in ("normal_unit", "surface_normal_unit"):
                    if key in geometry:
                        geometry["normal"] = geometry[key]
                        break
        else:
            if "axis" in geometry:
                geometry["axis_unit"] = geometry.pop("axis")
            if "normal" in geometry:
                geometry["normal_unit"] = geometry.pop("normal")
        item["geometry"] = _json_value(geometry, "feature.geometry")

        uncertainty = dict(source_uncertainty)
        if "position_std_m" not in uncertainty:
            calibrated_error = uncertainty.get("calibrated_measurement_error_m")
            if calibrated_error is not None:
                uncertainty["position_std_m"] = _number(
                    calibrated_error, "feature.uncertainty.calibrated_measurement_error_m",
                )
        position_std = uncertainty.get("position_std_m")
        if isinstance(position_std, Sequence) and not isinstance(position_std, (str, bytes)):
            if len(position_std) != 3:
                raise ValueError("feature.uncertainty.position_std_m vector must have three values")
            axes_std = [_number(value, "feature.uncertainty.position_std_m") for value in position_std]
            if any(value < 0.0 for value in axes_std):
                raise ValueError("feature.uncertainty.position_std_m values must be non-negative")
            uncertainty["position_std_axes_m"] = axes_std
            uncertainty["position_std_m"] = math.sqrt(sum(value * value for value in axes_std))
        if uncertainty:
            item["uncertainty"] = _json_value(uncertainty, "feature.uncertainty")
        public_features.append(item)
    result["features"] = public_features
    return result, images


def _public_obstacle_map(value: Any, observation_id: str) -> dict[str, Any]:
    base_fields = frozenset({"source_observation_id", "frame", "valid", "points_m"})
    optional_fields = frozenset({"timestamp_s", "robot_self_filtered"})
    if not isinstance(value, Mapping) or not base_fields <= frozenset(value) or (
        frozenset(value) - base_fields - optional_fields
    ):
        raise ValueError("robot_state.sensors.rgbd_obstacle_map has an invalid public shape")
    if value.get("source_observation_id") != observation_id or value.get("frame") not in {"world", "world_m"}:
        raise ValueError("RGB-D obstacle map must cite the current observation in world coordinates")
    if value.get("valid") is not True:
        raise ValueError("RGB-D obstacle map must be marked valid")
    if "robot_self_filtered" in value and type(value["robot_self_filtered"]) is not bool:
        raise ValueError("RGB-D obstacle map robot_self_filtered must be a boolean")
    points = value.get("points_m")
    if isinstance(points, np.ndarray):
        if (
            points.dtype.kind not in "iuf"
            or (points.dtype.kind == "f" and points.dtype.itemsize > 8)
            or points.ndim != 2 or points.shape[1] != 3 or len(points) > 200_000
            or not np.isfinite(points).all()
        ):
            raise ValueError("RGB-D obstacle map points must be finite real numeric xyz triples within the public bound")
        normalized = points.tolist()
    elif not isinstance(points, Sequence) or isinstance(points, (str, bytes)) or len(points) > 200_000:
        raise ValueError("RGB-D obstacle map points exceed the public bound")
    else:
        normalized = []
        for point in points:
            if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) != 3:
                raise ValueError("RGB-D obstacle points must be xyz triples")
            normalized.append([_number(axis, "RGB-D obstacle point") for axis in point])
    result = {
        "source_observation_id": observation_id,
        "frame": value["frame"],
        "valid": True,
        "points_m": normalized,
    }
    if "timestamp_s" in value:
        result["timestamp_s"] = _number(value["timestamp_s"], "RGB-D obstacle map timestamp_s")
    if "robot_self_filtered" in value:
        result["robot_self_filtered"] = value["robot_self_filtered"]
    return result


def _model_local_crops(
    observation: Mapping[str, Any],
    *,
    max_count: int = _MODEL_LOCAL_CROP_MAX_COUNT,
) -> list[tuple[dict[str, Any], str]]:
    """Create a few bounded views of current feature regions without new acquisitions."""
    max_count = _bounded_int(
        max_count,
        "max_local_crops",
        minimum=0,
        maximum=_MODEL_LOCAL_CROP_MAX_COUNT,
    )
    sequence = observation.get("frame_sequence")
    maps = observation.get("image_coordinate_maps_by_camera")
    frames = observation.get("frames")
    if (
        not isinstance(sequence, Sequence) or len(sequence) != 8
        or not isinstance(maps, Mapping) or not isinstance(frames, Mapping)
    ):
        return []
    current = sequence[-1]
    frame_key = current["frame_key"]
    frame_path = frames.get(frame_key)
    current_map = maps.get(frame_key)
    if not isinstance(frame_path, str) or not isinstance(current_map, Mapping):
        return []
    source_size = current_map.get("source_size_px")
    if not isinstance(source_size, Sequence) or len(source_size) != 2:
        return []
    width, height = source_size
    if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in (width, height)):
        return []

    base_pixels = sum(
        size[0] * size[1]
        for item in maps.values()
        for size in [item.get("model_image_size_px", ())]
        if isinstance(item, Mapping) and isinstance(size, Sequence) and len(size) == 2
    )
    crop_pixels = _MODEL_LOCAL_CROP_IMAGE_SIDE ** 2
    available_crops = min(
        max_count,
        max(0, _MODEL_SEQUENCE_TOTAL_PIXEL_LIMIT - base_pixels) // crop_pixels,
    )
    if not available_crops:
        return []

    capabilities, _ = _capability_summary(observation)
    graspable_refs = set(capabilities["graspable_candidate_refs"])
    held_surface_refs = capabilities["held_object_surface_candidate_refs"]
    held_surface_ref = held_surface_refs[0] if held_surface_refs else None
    try:
        import cv2
    except ImportError:
        return []
    image = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    if image is None or image.shape[:2] != (height, width):
        return []
    suffix = Path(frame_path).suffix.lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(suffix)
    if mime is None:
        return []

    candidates = []
    if capabilities["grasped"] is True:
        aperture_features = []
        aperture_refs = set(capabilities["aperture_reference_refs"])
        for feature in observation.get("features", ()):
            if not isinstance(feature, Mapping) or feature.get("feature_ref") not in aperture_refs:
                continue
            geometry = feature.get("geometry")
            ellipse = geometry.get("image_ellipse_candidate_px") if isinstance(geometry, Mapping) else None
            bbox = geometry.get("pixel_bbox_xyxy") if isinstance(geometry, Mapping) else None
            if not isinstance(ellipse, Mapping) or not isinstance(bbox, Sequence) or len(bbox) != 4:
                continue
            try:
                center = [_number(value, "aperture ellipse center") for value in ellipse["center_xy"]]
                axes = [_number(value, "aperture ellipse axes") for value in ellipse["axes_px"]]
                feature_bbox = [_number(value, "aperture bbox") for value in bbox]
                position_std_m = _number(feature["uncertainty"]["position_std_m"], "aperture position_std_m")
            except (KeyError, TypeError, ValueError):
                continue
            if (
                len(center) != 2 or len(axes) != 2 or len(feature_bbox) != 4
                or min(axes) <= 0.0 or position_std_m < 0.0
            ):
                continue
            aperture_features.append((
                position_std_m, -float(np.prod(axes)), center[1], center[0],
                feature, center, axes, feature_bbox,
            ))
        for position_std_m, _, _, _, feature, center, axes, feature_bbox in sorted(
            aperture_features, key=lambda item: item[:4],
        ):
            side = min(min(width, height), max(256, math.ceil(2.8 * max(axes))))
            crop_x0 = max(0, min(width - side, round(center[0] - side / 2)))
            crop_y0 = max(0, min(height - side, round(center[1] - side / 2)))
            bbox = (crop_x0, crop_y0, crop_x0 + side, crop_y0 + side)
            candidates.append((
                0.5, position_std_m, bbox[1], bbox[0], bbox, feature["feature_ref"],
                feature_bbox, "aperture_neighborhood",
            ))

    for feature in observation.get("features", ()):
        if not isinstance(feature, Mapping):
            continue
        if capabilities["grasped"] is True and feature.get("feature_ref") != held_surface_ref:
            continue
        geometry = feature.get("geometry")
        bbox = geometry.get("pixel_bbox_xyxy") if isinstance(geometry, Mapping) else None
        if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
            continue
        if feature.get("kind") == "dark_aperture_candidate":
            # Keep apertures in the full-scene overlay; isolated crops lose task context.
            continue
        try:
            x0, y0, x1, y1 = (_number(value, "feature.geometry.pixel_bbox_xyxy") for value in bbox)
        except ValueError:
            continue
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(width), x1), min(float(height), y1)
        box_width, box_height = x1 - x0, y1 - y0
        if box_width <= 0 or box_height <= 0 or max(box_width, box_height) > _MODEL_LOCAL_CROP_MAX_SIDE:
            continue
        side = min(
            min(width, height),
            max(96, math.ceil(2 * max(box_width, box_height))),
        )
        center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
        crop_x0 = max(0, min(width - side, round(center_x - side / 2)))
        crop_y0 = max(0, min(height - side, round(center_y - side / 2)))
        crop_bbox = (crop_x0, crop_y0, crop_x0 + side, crop_y0 + side)
        feature_ref = feature["feature_ref"]
        is_grasp_candidate = feature_ref in graspable_refs
        priority = -1 if is_grasp_candidate else 0 if feature_ref == held_surface_ref else 1
        candidates.append((
            priority,
            -(box_width * box_height),
            crop_bbox[1],
            crop_bbox[0],
            crop_bbox,
            feature_ref,
            [x0, y0, x1, y1],
            "grasp_candidate" if is_grasp_candidate else "feature_region",
        ))

    candidates.sort(key=lambda item: item[:4])
    selected: list[tuple[int, int, int, int]] = []
    selected_crops: list[tuple[dict[str, Any], str]] = []
    for candidate in candidates:
        _, _, _, _, bbox, feature_ref, feature_bbox, crop_kind = candidate
        duplicate = False
        for other in selected:
            intersection = (
                max(0, min(bbox[2], other[2]) - max(bbox[0], other[0]))
                * max(0, min(bbox[3], other[3]) - max(bbox[1], other[1]))
            )
            smaller_area = min(
                (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                (other[2] - other[0]) * (other[3] - other[1]),
            )
            if smaller_area and intersection / smaller_area >= 0.75:
                duplicate = True
                break
        if duplicate and feature_ref not in graspable_refs:
            continue
        crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        resized = cv2.resize(
            crop, (_MODEL_LOCAL_CROP_IMAGE_SIDE, _MODEL_LOCAL_CROP_IMAGE_SIDE),
            interpolation=cv2.INTER_AREA,
        )
        success, encoded = cv2.imencode(suffix, resized)
        if not success:
            continue
        crop_bytes = encoded.tobytes()
        if len(crop_bytes) > _IMAGE_LIMIT_BYTES:
            continue
        selected.append(bbox)
        metadata = {
            "crop_key": f"local_crop_{len(selected_crops) + 1:02d}",
            "crop_kind": crop_kind,
            "feature_ref": feature_ref,
            "source_frame_key": frame_key,
            "source_frame_id": current["frame_id"],
            "source_bbox_xyxy": list(bbox),
            "feature_bbox_xyxy": feature_bbox,
            "source_image_size_px": [width, height],
            "model_image_size_px": [_MODEL_LOCAL_CROP_IMAGE_SIDE, _MODEL_LOCAL_CROP_IMAGE_SIDE],
            "crop_to_source_offset_xy": [bbox[0], bbox[1]],
            "crop_to_source_scale_xy": [
                (bbox[2] - bbox[0]) / _MODEL_LOCAL_CROP_IMAGE_SIDE,
                (bbox[3] - bbox[1]) / _MODEL_LOCAL_CROP_IMAGE_SIDE,
            ],
        }
        selected_crops.append((metadata, f"data:{mime};base64,{base64.b64encode(crop_bytes).decode('ascii')}"))
        if len(selected_crops) >= available_crops:
            break
    return selected_crops


def _feature_refs(observation: Mapping[str, Any]) -> set[str]:
    return {item["feature_ref"] for item in observation["features"]}


def _feature_role(
    feature: Mapping[str, Any], observation: Mapping[str, Any] | None = None,
) -> str:
    frame = feature.get("frame")
    if feature.get("kind") == "robot_tcp" and frame == "tcp":
        return "tcp_gripper"
    geometry = feature.get("geometry", {})
    if (
        isinstance(geometry, Mapping)
        and geometry.get("position_kind") == "kinematic_tcp_track_from_rgbd_surface"
    ):
        return "tracked_held_surface_candidate"
    width = geometry.get("grasp_width_m") if isinstance(geometry, Mapping) else None
    if (
        frame in {"world", "world_m"}
        and not isinstance(width, bool)
        and isinstance(width, (int, float))
        and math.isfinite(float(width))
        and width > 0.0
    ):
        if observation is not None and has_observed_grasp_contact_geometry(
            observation, str(feature.get("feature_ref", "")),
        ):
            return "graspable_candidate"
        return "contact_geometry_unverified"
    if frame in {"world", "world_m"}:
        return "world_feature"
    return "local_feature"


def _has_local_plane_extent(feature: Mapping[str, Any]) -> bool:
    geometry = feature.get("geometry", {})
    extent = geometry.get("extent_m") if isinstance(geometry, Mapping) else None
    if extent is None:
        return True
    if not isinstance(extent, Sequence) or isinstance(extent, (str, bytes)) or len(extent) != 3:
        return False
    try:
        values = np.asarray(extent, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        np.isfinite(values).all()
        and np.all(values >= 0.0)
        and float(np.max(values)) <= _MAX_LOCAL_PLANE_REFERENCE_EXTENT_M
    )


def _capability_summary(observation: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    features = observation.get("features", ())
    roles = {item["feature_ref"]: _feature_role(item, observation) for item in features}
    state = observation.get("robot_state", {})
    sensors = state.get("sensors", {}) if isinstance(state, Mapping) else {}
    sensors = sensors if isinstance(sensors, Mapping) else {}

    def public_bool(name: str) -> bool | None:
        value = sensors.get(name)
        return value if type(value) is bool else None

    held_surface_refs = []
    tracked_surface_refs = []
    has_tracked_surface = any(role == "tracked_held_surface_candidate" for role in roles.values())
    tcp_pose = sensors.get("tcp_pose")
    if public_bool("grasped") is True and isinstance(tcp_pose, Sequence) and len(tcp_pose) == 7:
        try:
            tcp_position = np.asarray(tcp_pose[:3], dtype=np.float64)
            if not np.isfinite(tcp_position).all():
                raise ValueError
        except (TypeError, ValueError):
            tcp_position = None
        if tcp_position is not None:
            nearby = []
            for feature in features:
                ref = feature.get("feature_ref")
                geometry = feature.get("geometry", {})
                position = geometry.get("position_m") if isinstance(geometry, Mapping) else None
                if (
                    roles.get(ref) not in {
                        "world_feature", "graspable_candidate", "tracked_held_surface_candidate",
                    }
                    or feature.get("kind") != "depth_surface_patch"
                    or not isinstance(position, Sequence)
                    or isinstance(position, (str, bytes))
                    or len(position) != 3
                ):
                    continue
                try:
                    distance = float(np.linalg.norm(np.asarray(position, dtype=np.float64) - tcp_position))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(distance) and distance <= _HELD_OBJECT_SURFACE_MAX_DISTANCE_M:
                    if has_tracked_surface and roles.get(ref) != "tracked_held_surface_candidate":
                        continue
                    nearby.append((distance, ref))
            nearby.sort()
            for _, ref in nearby:
                if roles.get(ref) == "tracked_held_surface_candidate":
                    tracked_surface_refs.append(ref)
                else:
                    held_surface_refs.append(ref)
                # shortcut: proximity labels candidates; replace with instance tracking when available.
                roles[ref] = (
                    "tracked_held_surface_candidate"
                    if ref in tracked_surface_refs else "held_object_surface_candidate"
                )

    held_z_values = []
    if public_bool("grasped") is True:
        for feature in features:
            if feature.get("feature_ref") not in held_surface_refs:
                continue
            geometry = feature.get("geometry", {})
            position = geometry.get("position_m") if isinstance(geometry, Mapping) else None
            try:
                xyz = np.asarray(position, dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if xyz.shape == (3,) and np.isfinite(xyz).all():
                held_z_values.append(float(xyz[2]))
    held_z_reference = float(np.median(held_z_values)) if held_z_values else None

    plane_refs = []
    for feature in features:
        if feature.get("frame") not in {"world", "world_m"}:
            continue
        if feature.get("kind") in {
            "circular_surface_candidate", "dark_aperture_candidate",
            "raised_circular_candidate", "recessed_circular_candidate",
        } or roles.get(feature.get("feature_ref")) == "tracked_held_surface_candidate":
            continue
        if not _has_local_plane_extent(feature):
            continue
        if held_z_reference is not None:
            position = feature.get("geometry", {}).get("position_m")
            try:
                reference_z = float(position[2])
            except (IndexError, TypeError, ValueError):
                continue
            if (
                not math.isfinite(reference_z)
                or abs(reference_z - held_z_reference) > _MAX_HELD_PLANE_VERTICAL_OFFSET_M
            ):
                continue
        try:
            _validate_relation_geometry("point_on_plane", feature, feature)
        except ValueError:
            continue
        plane_refs.append(feature["feature_ref"])

    aperture_features = [feature for feature in features if has_observed_aperture_geometry(feature)]
    aperture_centers = []
    for feature in aperture_features:
        geometry = feature.get("geometry", {})
        ellipse = geometry.get("image_ellipse_candidate_px") if isinstance(geometry, Mapping) else None
        if not isinstance(ellipse, Mapping):
            continue
        center = ellipse.get("center_xy")
        axes = ellipse.get("axes_px")
        if not isinstance(center, Sequence) or len(center) != 2 or not isinstance(axes, Sequence) or len(axes) != 2:
            continue
        try:
            cx, cy = (_number(value, "aperture image center") for value in center)
            area = _number(axes[0], "aperture image axis") * _number(axes[1], "aperture image axis")
        except ValueError:
            continue
        if area > 0:
            aperture_centers.append((area, feature, cx, cy))
    aperture_image_relations = []
    if aperture_centers:
        _, anchor, anchor_x, anchor_y = max(aperture_centers, key=lambda item: item[0])
        for _, feature, center_x, center_y in aperture_centers:
            if feature["feature_ref"] == anchor["feature_ref"]:
                continue
            dx, dy = center_x - anchor_x, center_y - anchor_y
            aperture_image_relations.append({
                "feature_ref": feature["feature_ref"],
                "anchor_feature_ref": anchor["feature_ref"],
                "delta_xy_px": [round(dx, 1), round(dy, 1)],
                "bearing_clockwise_from_image_up_deg": round(math.degrees(math.atan2(dx, -dy)) % 360.0, 1),
                "distance_px": round(math.hypot(dx, dy), 1),
            })

    grasp_candidate_aperture_relations = []
    for candidate in features:
        if roles.get(candidate.get("feature_ref")) != "graspable_candidate":
            continue
        geometry = candidate.get("geometry", {})
        centroid = geometry.get("pixel_centroid_xy") if isinstance(geometry, Mapping) else None
        bbox = geometry.get("pixel_bbox_xyxy") if isinstance(geometry, Mapping) else None
        if not isinstance(centroid, Sequence) or isinstance(centroid, (str, bytes)) or len(centroid) != 2:
            if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
                continue
            centroid = ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
        try:
            candidate_x, candidate_y = (_number(value, "grasp candidate pixel") for value in centroid)
        except ValueError:
            continue

        for _, aperture, aperture_x, aperture_y in aperture_centers:
            aperture_geometry = aperture.get("geometry", {})
            ellipse = aperture_geometry.get("image_ellipse_candidate_px")
            axes = ellipse.get("axes_px") if isinstance(ellipse, Mapping) else None
            angle = ellipse.get("angle_rad", 0.0) if isinstance(ellipse, Mapping) else 0.0
            if not isinstance(axes, Sequence) or isinstance(axes, (str, bytes)) or len(axes) != 2:
                continue
            try:
                axis_x, axis_y = (_number(value, "aperture ellipse axis") for value in axes)
                angle = _number(angle, "aperture ellipse angle")
            except ValueError:
                continue
            if min(axis_x, axis_y) <= 0.0:
                continue
            dx, dy = candidate_x - aperture_x, candidate_y - aperture_y
            local_x = math.cos(angle) * dx + math.sin(angle) * dy
            local_y = -math.sin(angle) * dx + math.cos(angle) * dy
            normalized_radius = math.sqrt((local_x / axis_x) ** 2 + (local_y / axis_y) ** 2)
            if math.isfinite(normalized_radius) and normalized_radius <= 2.5:
                grasp_candidate_aperture_relations.append({
                    "candidate_feature_ref": candidate["feature_ref"],
                    "aperture_feature_ref": aperture["feature_ref"],
                    "candidate_center_inside_ellipse": normalized_radius <= 1.0,
                    "normalized_ellipse_radius": round(normalized_radius, 3),
                })

    summary = {
        "tcp_feature_refs": [ref for ref, role in roles.items() if role == "tcp_gripper"],
        "graspable_candidate_refs": [ref for ref, role in roles.items() if role == "graspable_candidate"],
        "contact_geometry_unverified_refs": [
            ref for ref, role in roles.items() if role == "contact_geometry_unverified"
        ],
        "held_object_surface_candidate_refs": held_surface_refs,
        "tracked_held_surface_refs": tracked_surface_refs,
        "plane_reference_refs": plane_refs,
        "aperture_reference_refs": [feature["feature_ref"] for feature in features
                                    if has_observed_aperture_geometry(feature)],
        "aperture_image_relations": aperture_image_relations,
        "grasp_candidate_aperture_relations": grasp_candidate_aperture_relations,
        "grasped": public_bool("grasped"),
        "release_authorized": public_bool("release_authorized"),
    }
    return summary, roles


def _hollow_vessel_aperture_scope(
    task_goal: str,
    capabilities: Mapping[str, Any],
) -> tuple[set[str], set[str]]:
    if not _HOLLOW_VESSEL_MOVING_TARGET_RE.search(task_goal):
        return None, set()
    relations = capabilities.get("grasp_candidate_aperture_relations", ())
    interior_relations = [
        relation
        for relation in relations
        if isinstance(relation, Mapping)
        and relation.get("candidate_center_inside_ellipse") is True
        and isinstance(relation.get("candidate_feature_ref"), str)
    ]
    if not interior_relations:
        return set(), set()

    aperture_refs = {
        item["aperture_feature_ref"] for item in interior_relations
        if isinstance(item.get("aperture_feature_ref"), str)
    }
    interior_refs = {
        item["candidate_feature_ref"] for item in interior_relations
        if isinstance(item.get("candidate_feature_ref"), str)
    }
    return aperture_refs, interior_refs


def _scope_hollow_vessel_grasp_candidates(
    task_goal: str,
    capabilities: dict[str, Any],
    model_observation: dict[str, Any],
) -> None:
    """Mask interior patches across observed openings without guessing target identity."""
    aperture_refs, interior_refs = _hollow_vessel_aperture_scope(
        task_goal, capabilities,
    )
    if not interior_refs:
        return

    candidates = list(capabilities.get("graspable_candidate_refs", ()))
    capabilities["aperture_interior_candidate_refs"] = [
        ref for ref in candidates if ref in interior_refs
    ]
    remaining_candidates = [
        ref for ref in candidates if ref not in interior_refs
    ]
    shell_distance = {}
    for relation in capabilities.get("grasp_candidate_aperture_relations", ()):
        if (
            isinstance(relation, Mapping)
            and relation.get("aperture_feature_ref") in aperture_refs
            and relation.get("candidate_center_inside_ellipse") is False
            and relation.get("candidate_feature_ref") in remaining_candidates
        ):
            ref = relation["candidate_feature_ref"]
            try:
                distance = float(relation["normalized_ellipse_radius"]) - 1.0
            except (KeyError, TypeError, ValueError):
                continue
            shell_distance[ref] = min(shell_distance.get(ref, math.inf), max(0.0, distance))
    original_order = {ref: index for index, ref in enumerate(remaining_candidates)}
    remaining_candidates.sort(key=lambda ref: (shell_distance.get(ref, math.inf), original_order[ref]))
    capabilities["graspable_candidate_refs"] = remaining_candidates
    for feature in model_observation.get("features", ()):
        if isinstance(feature, dict) and feature.get("feature_ref") in interior_refs:
            feature["role"] = "aperture_interior_surface_not_exterior_grasp"


def _task_semantic_guidance(task_goal: str) -> str:
    """Add a public visual distinction for the bowl-on-plate task's vessel distractor."""
    normalized_goal = task_goal.casefold()
    if "bowl" not in normalized_goal or "plate" not in normalized_goal:
        return ""
    return (
        "\nTask-object identity check: this task moves the shallow, broad bowl onto the plate. "
        "The tall narrow bottle is a distractor, not the bowl. Visually identify the bowl in the "
        "current full-scene image, then choose only a current grasp candidate visibly supported on "
        "that bowl's exterior wall. The plate is the destination, not the grasp target. Candidate "
        "geometry proves contact feasibility only; if the bowl-to-feature match is unclear, observe "
        "or stop rather than grasping another object. Before a grasp, inspect whether the visible bottle "
        "occupies the TCP-to-bowl corridor. If it does, do not retry the same direct corridor: choose a "
        "materially different bounded route only where the current RGB-D view shows room, and let the "
        "normal full sweep, IK, and force checks validate it. Do not invent a route or fixed coordinates."
    )
def _validate_relation_geometry(relation: str, moving: Mapping[str, Any], reference: Mapping[str, Any]) -> None:
    moving_geometry = moving.get("geometry", {})
    reference_geometry = reference.get("geometry", {})
    moving_uncertainty = moving.get("uncertainty", {})
    reference_uncertainty = reference.get("uncertainty", {})
    required: dict[str, tuple[tuple[str, str], ...]] = {
        "coaxial": (("moving", "axis"), ("reference", "axis"),
                    ("moving", "position_std_m"), ("reference", "position_std_m"),
                    ("moving", "axis_std_rad"), ("reference", "axis_std_rad")),
        "parallel": (("moving", "axis"), ("reference", "axis"),
                     ("moving", "position_std_m"), ("reference", "position_std_m"),
                     ("moving", "axis_std_rad"), ("reference", "axis_std_rad")),
        "centered_on": (("moving", "position_std_m"), ("reference", "position_std_m")),
        "point_on_plane": (("moving", "position_std_m"), ("reference", "position_std_m"),
                           ("reference", "normal"), ("reference", "axis_std_rad")),
    }
    if relation == "remain_clear_of":
        plane_reference = (
            reference.get("kind") == "plane_candidate"
            or reference_geometry.get("normal_kind") in _PLANE_NORMAL_KINDS
        )
        required[relation] = (
            (("moving", "bounding_radius_m"), ("reference", "normal"), ("reference", "extent_m"),
             ("reference", "axis_std_rad"))
            if plane_reference else
            (("moving", "bounding_radius_m"), ("reference", "bounding_radius_m"))
        )
    for side, feature in (("moving", moving), ("reference", reference)):
        if "position_m" not in feature.get("geometry", {}):
            raise ValueError(f"constraint.relation {relation!r} requires {side} feature position_m")
        if "position_std_m" not in feature.get("uncertainty", {}):
            raise ValueError(f"constraint.relation {relation!r} requires {side} feature position_std_m")
    for side, key in required[relation]:
        source = moving_geometry if side == "moving" else reference_geometry
        if key.endswith("_std_m") or key == "axis_std_rad":
            source = moving_uncertainty if side == "moving" else reference_uncertainty
        if key not in source:
            raise ValueError(f"constraint.relation {relation!r} requires {side} feature {key}")


def _validate_grasp_geometry(
    constraints: Sequence[Mapping[str, str]],
    features: Mapping[str, Mapping[str, Any]],
    sensors: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> None:
    if sensors.get("grasped") is not False:
        if sensors.get("grasped") is True:
            raise ValueError(
                "grasp is unavailable while public grasped=true; do not re-grasp. Use the current "
                "held_object_surface_candidate_refs with motion_regime=free_space or contact, or stop"
            )
        raise ValueError("grasp requires explicit grasped=false")
    moving_refs = {item["moving_feature"] for item in constraints}
    if len(moving_refs) != 1:
        raise ValueError("grasp requires one TCP-local moving gripper feature")
    moving_ref = next(iter(moving_refs))
    if _feature_role(features[moving_ref]) != "tcp_gripper":
        raise ValueError("grasp moving_feature must be the current TCP-local robot FK feature")
    for constraint in constraints:
        reference = features[constraint["reference_feature"]]
        if _feature_role(reference, observation) != "graspable_candidate":
            raise ValueError(
                "grasp reference_feature must have current RGB-D support for both predicted finger contact sites"
            )
        width = reference.get("geometry", {}).get("grasp_width_m")
        if isinstance(width, bool) or not isinstance(width, (int, float)) or not math.isfinite(float(width)) or width <= 0:
            raise ValueError("grasp reference_feature requires a positive observed geometry.grasp_width_m")


def _model_observation(
    observation: Mapping[str, Any],
    *,
    image_cameras: Sequence[str] | None = None,
) -> dict[str, Any]:
    result = dict(observation)
    capabilities, roles = _capability_summary(result)
    if capabilities["grasped"] is True:
        capabilities["tcp_feature_refs"] = []
    image_maps = result.get("image_coordinate_maps_by_camera")
    frame_sequence = result.get("frame_sequence")
    if image_cameras is not None:
        selected_cameras = set(image_cameras)
        result["frames"] = {
            camera: path for camera, path in observation["frames"].items()
            if camera in selected_cameras
        }
        if isinstance(image_maps, Mapping):
            selected_maps = {
                camera: value for camera, value in image_maps.items()
                if camera in selected_cameras
            }
            if selected_maps:
                result["image_coordinate_maps_by_camera"] = selected_maps
            else:
                result.pop("image_coordinate_maps_by_camera", None)
            image_maps = selected_maps
    pixel_scale = None
    if isinstance(frame_sequence, Sequence) and frame_sequence:
        current_frame_key = frame_sequence[-1]["frame_key"]
        current_map = image_maps.get(current_frame_key) if isinstance(image_maps, Mapping) else None
        if not isinstance(current_map, Mapping):
            raise ValueError("current feature overlay requires its recorded image coordinate map")
        only_map = current_map
    elif isinstance(image_maps, Mapping) and len(image_maps) == 1:
        only_map = next(iter(image_maps.values()))
    else:
        only_map = None
    if isinstance(only_map, Mapping):
        scale = only_map.get("scale_xy")
        if isinstance(scale, Sequence) and not isinstance(scale, (str, bytes)) and len(scale) == 2:
            pixel_scale = (float(scale[0]), float(scale[1]))
    result.pop("depth", None)
    result.pop("calibration", None)
    frames = result["frames"]
    if isinstance(frame_sequence, Sequence):
        result["frames"] = [
            item["frame_key"] for item in frame_sequence if item["frame_key"] in frames
        ]
    else:
        result["frames"] = sorted(frames)
    if isinstance(result.get("depth"), Mapping):
        result["depth"] = sorted(result["depth"])
    state = result.get("robot_state")
    if isinstance(state, Mapping) and isinstance(state.get("sensors"), Mapping):
        state = dict(state)
        sensors = dict(state["sensors"])
        sensors.pop("rgbd_obstacle_map", None)
        for key in ("camera_intrinsics", "camera_extrinsics", "camera_depth_path", "camera_depth_convention"):
            sensors.pop(key, None)
        if sensors.get("commanded_tcp_pose") == sensors.get("tcp_pose"):
            sensors.pop("commanded_tcp_pose", None)
        state["sensors"] = sensors
        result["robot_state"] = state
    features = result.get("features")
    if isinstance(features, Sequence) and not isinstance(features, (str, bytes)):
        result["capabilities"] = capabilities
        model_features = []
        descriptions: list[dict[str, str]] = []
        uncertainty_notes: list[dict[str, str]] = []
        has_feature_overlay = image_cameras is not None and any(
            camera.endswith("_feature_overlay") for camera in image_cameras
        )
        for index, raw_feature in enumerate(features, 1):
            if capabilities["grasped"] is True and roles[raw_feature["feature_ref"]] == "tcp_gripper":
                continue
            feature = dict(raw_feature)
            if has_feature_overlay and roles[raw_feature["feature_ref"]] != "tracked_held_surface_candidate":
                feature["overlay_label"] = f"F{index:02d}"
            feature["role"] = roles[feature["feature_ref"]]
            kind = feature.get("kind")
            description = feature.pop("description", None)
            if isinstance(kind, str) and isinstance(description, str):
                item = {"kind": kind, "description": description}
                if item not in descriptions:
                    descriptions.append(item)
            feature.pop("source_observation_id", None)
            geometry = feature.get("geometry")
            if isinstance(geometry, Mapping):
                feature["geometry"] = {
                    key: geometry[key]
                    for key in (
                        "position_m", "axis", "normal", "extent_m", "bounding_radius_m",
                        "grasp_width_m", "pixel_bbox_xyxy", "pixel_centroid_xy",
                        "image_circle_candidate_px", "image_ellipse_candidate_px",
                        "center_minus_rim_depth_m", "rim_depth_m", "center_depth_m",
                        "center_expected_rim_depth_m", "rim_plane_fit_rms_m", "rgbd_shape_evidence",
                        "position_kind", "axis_kind", "normal_kind",
                    )
                    if key in geometry
                }
                if pixel_scale is not None:
                    sx, sy = pixel_scale
                    centroid = geometry.get("pixel_centroid_xy")
                    if isinstance(centroid, Sequence) and not isinstance(centroid, (str, bytes)) and len(centroid) == 2:
                        feature["geometry"]["model_image_pixel_centroid_xy"] = [
                            float(centroid[0]) * sx, float(centroid[1]) * sy,
                        ]
                        feature["geometry"].pop("pixel_centroid_xy", None)
                    bbox = geometry.get("pixel_bbox_xyxy")
                    if isinstance(bbox, Sequence) and not isinstance(bbox, (str, bytes)) and len(bbox) == 4:
                        feature["geometry"]["model_image_pixel_bbox_xyxy"] = [
                            float(bbox[0]) * sx, float(bbox[1]) * sy,
                            float(bbox[2]) * sx, float(bbox[3]) * sy,
                        ]
                        feature["geometry"].pop("pixel_bbox_xyxy", None)
                    ellipse = geometry.get("image_ellipse_candidate_px")
                    if isinstance(ellipse, Mapping):
                        model_ellipse = dict(ellipse)
                        center = ellipse.get("center_xy")
                        axes = ellipse.get("axes_px")
                        if (
                            isinstance(center, Sequence) and not isinstance(center, (str, bytes))
                            and len(center) == 2
                        ):
                            model_ellipse["center_xy"] = [
                                float(center[0]) * sx, float(center[1]) * sy,
                            ]
                        if (
                            isinstance(axes, Sequence) and not isinstance(axes, (str, bytes))
                            and len(axes) == 2
                        ):
                            model_ellipse["axes_px"] = [
                                float(axes[0]) * sx, float(axes[1]) * sy,
                            ]
                        feature["geometry"]["model_image_ellipse_candidate_px"] = model_ellipse
                        feature["geometry"].pop("image_ellipse_candidate_px", None)
            uncertainty = feature.get("uncertainty")
            if isinstance(uncertainty, Mapping):
                compact_uncertainty = {}
                for key, value in uncertainty.items():
                    if key in {"uncertainty_source", "error_note"} and isinstance(kind, str) and isinstance(value, str):
                        item = {"kind": kind, "note": value}
                        if item not in uncertainty_notes:
                            uncertainty_notes.append(item)
                    elif (
                        key == "rgbd_shape_evidence"
                        and isinstance(geometry, Mapping)
                        and value == geometry.get("rgbd_shape_evidence")
                    ):
                        # The same evidence is retained once in geometry rather than duplicated.
                        continue
                    else:
                        compact_uncertainty[key] = value
                feature["uncertainty"] = compact_uncertainty
            model_features.append(feature)
        result["features"] = model_features
        if descriptions:
            result["feature_descriptions_by_kind"] = descriptions
        if uncertainty_notes:
            result["feature_uncertainty_notes_by_kind"] = uncertainty_notes
    if image_cameras is not None:
        # Image labels in the multimodal message carry the frame keys and chronology.
        result.pop("frames", None)
    if pixel_scale is not None:
        # The model pixel coordinates above are already derived from these maps.
        result.pop("image_coordinate_maps_by_camera", None)
    return _compact_model_numbers(result)


def _compact_model_numbers(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, Mapping):
        return {key: _compact_model_numbers(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_compact_model_numbers(item) for item in value]
    return value


def _model_history(history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    sanitized = []
    for raw_event in history[-MAX_HISTORY:]:
        event = dict(raw_event)
        tool_result = event.get("result")
        if isinstance(tool_result, Mapping):
            tool_result = {
                key: value for key, value in tool_result.items()
                if key not in _MODEL_HISTORY_DUPLICATE_RESULT_FIELDS
            }
            if isinstance(tool_result.get("observation"), Mapping):
                tool_result["observation"] = {
                    "observation_id": tool_result["observation"].get("observation_id"),
                }
            event["result"] = tool_result
        sanitized.append(event)
    return _json_value(sanitized, "history")


def _strip_json_markdown_fence(text: str) -> str:
    match = re.fullmatch(r"\s*```json[ \t]*\r?\n(.*?)\r?\n```[ \t]*", text, re.DOTALL)
    return match.group(1) if match else text


def parse_tool_call(
    value: Any,
    *,
    observation: Mapping[str, Any],
    completed: bool,
    allowed_tools: Mapping[str, Any] | None = None,
    task_goal: str | None = None,
) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("model response must be a JSON object") from exc
    if not isinstance(value, Mapping) or type(completed) is not bool:
        raise ValueError("tool response must be an object and completed must be a boolean")
    tool = value.get("tool")
    if allowed_tools is not None and tool not in allowed_tools:
        if tool == "observe":
            raise ValueError("observe is unavailable immediately after observe; use the already refreshed observation")
        if tool == "grasp":
            capabilities, _ = _capability_summary(observation)
            if not capabilities.get("graspable_candidate_refs"):
                raise ValueError("grasp is unavailable: no current graspable candidates; do not select an unverified feature")
            raise ValueError("grasp is unavailable unless public grasped is explicitly false")
        if tool == "move_tcp":
            capabilities, _ = _capability_summary(observation)
            raise ValueError(
                "move_tcp is unavailable unless public grasped is explicitly false"
                if capabilities.get("grasped") is not False else
                "move_tcp is not available in this mode"
            )
        raise ValueError(f"unsupported tool in this mode: {tool!r}")
    if tool == "observe":
        if set(value) not in ({"tool"}, {"tool", "observation_id"}):
            raise ValueError("observe accepts only an optional current observation_id")
        if "observation_id" in value and _text(
            value["observation_id"], "observe.observation_id", limit=160,
        ) != observation.get("observation_id"):
            raise ValueError("observe must reference the current observation_id")
        return {"tool": "observe"}
    fields = {
        "check_task": {"tool"}, "finish": {"tool"}, "stop": {"tool"},
        "grasp": {"tool", "observation_id", "target_feature_ref"},
        "move_tcp": {
            "tool", "observation_id", "frame", "translation_delta_m", "rotation_vector_rad",
            "max_duration_s", "motion_budget",
        },
        "execute_goal": {
            "tool", "observation_id", "intent", "constraints", "motion_regime", "stop_condition",
        },
    }.get(tool)
    if fields is None:
        raise ValueError(f"unsupported tool: {tool!r}")
    if tool == "move_tcp":
        optional_fields = {"rotation_vector_rad", "max_duration_s", "motion_budget"}
        missing = sorted(fields - optional_fields - set(value))
        unexpected = sorted(set(value) - fields)
        if missing or unexpected:
            detail = []
            if missing:
                detail.append(f"missing required fields: {', '.join(missing)}")
            if unexpected:
                detail.append(f"unexpected fields: {', '.join(unexpected)}")
            raise ValueError(
                "move_tcp requires tool, observation_id, frame, and translation_delta_m; "
                "rotation_vector_rad, max_duration_s, and motion_budget are optional"
                + (f"; {'; '.join(detail)}" if detail else "")
            )
    elif set(value) != fields:
        unexpected = sorted(set(value) - fields)
        detail = f"; unexpected fields: {', '.join(unexpected)}" if unexpected else ""
        raise ValueError(
            f"{tool} must contain exactly: {', '.join(sorted(fields))}{detail}; "
            "feedback belongs to tool history, not the tool call"
        )
    if tool == "finish" and completed is not True:
        raise ValueError("finish requires the latest evaluator completed=true")
    if tool == "move_tcp":
        observation_id = _text(value["observation_id"], "move_tcp.observation_id", limit=160)
        if observation_id != observation.get("observation_id"):
            raise ValueError("move_tcp must reference the current observation_id")
        if value["frame"] != "tcp":
            raise ValueError("move_tcp.frame must be 'tcp'")
        translation = _numeric_vector(value["translation_delta_m"], "move_tcp.translation_delta_m", 3)
        rotation = _numeric_vector(
            value.get("rotation_vector_rad", [0.0, 0.0, 0.0]), "move_tcp.rotation_vector_rad", 3,
        )
        if float(np.linalg.norm(rotation)) > math.pi + 1e-9:
            raise ValueError("move_tcp.rotation_vector_rad norm must be at most pi")
        if (
            float(np.linalg.norm(translation)) <= _MIN_MOVE_TCP_TRANSLATION_M
            and float(np.linalg.norm(rotation)) <= _MIN_MOVE_TCP_ROTATION_RAD
        ):
            raise ValueError(
                "move_tcp delta is within executor tolerances; translation must exceed 1 mm or rotation must "
                "exceed 0.02 rad to produce motion"
            )
        duration = _number(
            value.get("max_duration_s", _DEFAULT_MOVE_TCP_DURATION_S), "move_tcp.max_duration_s",
        )
        if not 0.0 < duration <= _MAX_MOVE_TCP_DURATION_S:
            raise ValueError(f"move_tcp.max_duration_s must be in (0, {_MAX_MOVE_TCP_DURATION_S:g}]")
        budget = value.get("motion_budget", _DEFAULT_MOVE_TCP_STEPS)
        if type(budget) is not int or not 1 <= budget <= _MAX_MOVE_TCP_STEPS:
            raise ValueError(f"move_tcp.motion_budget must be an integer in [1, {_MAX_MOVE_TCP_STEPS}]")
        return {
            "tool": "move_tcp",
            "observation_id": observation_id,
            "frame": "tcp",
            "translation_delta_m": translation,
            "rotation_vector_rad": rotation,
            "max_duration_s": duration,
            "motion_budget": budget,
        }
    if tool == "grasp":
        observation_id = _text(value["observation_id"], "grasp.observation_id", limit=160)
        if observation_id != observation.get("observation_id"):
            raise ValueError("grasp must reference the current observation_id")
        capabilities, roles = _capability_summary(observation)
        tcp_refs = capabilities["tcp_feature_refs"]
        if len(tcp_refs) != 1:
            raise ValueError(f"grasp requires exactly one current TCP feature_ref; found {len(tcp_refs)}")
        target_ref = _text(value["target_feature_ref"], "grasp.target_feature_ref", limit=160)
        features = {item["feature_ref"]: item for item in observation["features"]}
        target = features.get(target_ref)
        if target is None:
            raise ValueError("grasp target_feature_ref must come from the current observation")
        if roles.get(target_ref) != "graspable_candidate":
            if roles.get(target_ref) == "contact_geometry_unverified":
                raise ValueError(
                    "current RGB-D does not support both predicted finger contact sites for grasp target_feature_ref"
                )
            if not capabilities["graspable_candidate_refs"]:
                raise ValueError(
                    "no current graspable candidates: public RGB-D does not validate bilateral contact geometry; "
                    "do not select another feature for grasp. Observe only if a new acquisition can add relevant "
                    "visual evidence; otherwise stop and report incomplete"
                )
            raise ValueError("grasp target_feature_ref must be a current graspable candidate")
        if isinstance(task_goal, str) and _HOLLOW_VESSEL_MOVING_TARGET_RE.search(task_goal):
            _, interior_refs = _hollow_vessel_aperture_scope(
                task_goal, capabilities,
            )
            if target_ref in interior_refs:
                raise ValueError(
                    "grasp target center lies inside the currently observed aperture associated with the hollow vessel named as "
                    "the task object; this is interior-surface geometry, not an exterior grasp target. Choose a "
                    "current candidate visibly on the vessel exterior or use a bounded, visually grounded move "
                    "and refresh the observation. Do not repeat this interior candidate."
                )
        constraint = [{
            "relation": "centered_on",
            "moving_feature": tcp_refs[0],
            "reference_feature": target_ref,
        }]
        _validate_relation_geometry("centered_on", features[tcp_refs[0]], target)
        sensors = observation.get("robot_state", {}).get("sensors", {})
        _validate_grasp_geometry(constraint, features, sensors, observation)
        return {
            "tool": "grasp",
            "observation_id": observation_id,
            "target_feature_ref": target_ref,
        }
    if tool != "execute_goal":
        return {"tool": tool}

    current_id = observation.get("observation_id")
    observation_id = _text(value["observation_id"], "execute_goal.observation_id", limit=160)
    if observation_id != current_id:
        raise ValueError("execute_goal must reference the current observation_id")
    constraints = value["constraints"]
    if not isinstance(constraints, Sequence) or isinstance(constraints, (str, bytes)):
        raise ValueError("execute_goal.constraints must contain one to eight relations")
    if not 1 <= len(constraints) <= 8:
        raise ValueError(
            f"execute_goal.constraints must contain one to eight relations; got {len(constraints)}. "
            "Select only the single visually task-matching target, not one relation per candidate feature"
        )
    refs = _feature_refs(observation)
    features = {item["feature_ref"]: item for item in observation["features"]}
    normalized_constraints = []
    for constraint in constraints:
        if not isinstance(constraint, Mapping) or set(constraint) != {
            "relation", "moving_feature", "reference_feature",
        }:
            raise ValueError("each constraint must contain relation, moving_feature, reference_feature")
        relation = constraint["relation"]
        if not isinstance(relation, str) or relation not in _RELATIONS:
            raise ValueError(f"constraint.relation must be one of {sorted(_RELATIONS)}")
        moving = _text(constraint["moving_feature"], "constraint.moving_feature", limit=160)
        reference = _text(constraint["reference_feature"], "constraint.reference_feature", limit=160)
        missing_refs = sorted({moving, reference} - refs)
        if missing_refs:
            stale_refs = ", ".join(f"...{ref[-40:]}" for ref in missing_refs)
            raise ValueError(
                f"goal feature refs {stale_refs} are absent from the current observation; "
                "refs are frame-scoped, so copy exact current feature_ref values"
            )
        if relation == "point_on_plane":
            capabilities, _ = _capability_summary(observation)
            if reference not in capabilities["plane_reference_refs"] + capabilities["aperture_reference_refs"]:
                raise ValueError(
                    "point_on_plane reference must be a current local plane_reference_refs entry; "
                    "broad workspace/table surfaces are not local task targets"
                )
        if relation in {"coaxial", "parallel"}:
            for side, feature in (("moving", features[moving]), ("reference", features[reference])):
                geometry = feature.get("geometry", {})
                if (
                    feature.get("kind") == "depth_surface_patch"
                    and geometry.get("axis_kind") == "largest_pca_extent_not_object_or_hole_axis"
                ):
                    raise ValueError(
                        f"constraint.relation {relation!r} cannot use the {side} depth-surface PCA axis as an object axis; "
                        "for non-contact alignment only, centered_on may target exactly one visually task-matching "
                        "aperture whose published position uncertainty is within controller precision. Do not reuse "
                        "a target rejected for uncertainty, including a re-bound reference to the same visible hole; "
                        "never combine multiple aperture centers. If no such target is clear, stop. Do not use contact "
                        "without a measured part axis"
                    )
        _validate_relation_geometry(relation, features[moving], features[reference])
        normalized_constraints.append({
            "relation": relation, "moving_feature": moving, "reference_feature": reference,
        })
    if len({constraint["moving_feature"] for constraint in normalized_constraints}) != 1:
        raise ValueError("all constraints in one goal must use the same moving_feature; select one moving feature")
    regime = value["motion_regime"]
    if not isinstance(regime, str) or regime not in _REGIMES:
        raise ValueError(f"motion_regime must be one of {sorted(_REGIMES)}")
    if regime == "grasp":
        sensors = observation.get("robot_state", {}).get("sensors", {})
        _validate_grasp_geometry(normalized_constraints, features, sensors, observation)
    elif regime == "contact":
        capabilities, roles = _capability_summary(observation)
        grasped = capabilities["grasped"]
        if grasped is True:
            moving_role = "held_object_surface_candidate"
        elif grasped is False:
            moving_role = "tcp_gripper"
        else:
            raise ValueError("contact requires explicit public grasped state")
        if not any(
            constraint["relation"] == "point_on_plane"
            and roles.get(constraint["moving_feature"]) == moving_role
            for constraint in normalized_constraints
        ):
            if grasped is True:
                held_refs = capabilities["held_object_surface_candidate_refs"]
                raise ValueError(
                    "held-object contact requires point_on_plane with moving_feature from current "
                    f"held_object_surface_candidate_refs {held_refs!r}, never robot_tcp. Use only a visible, task-relevant solid "
                    "contact surface, not an empty hole; otherwise choose grounded free_space or stop"
                )
            raise ValueError("contact requires point_on_plane from the current TCP or a verified held-object surface")
        if any(
            constraint["relation"] == "point_on_plane"
            and features[constraint["reference_feature"]].get("kind") in {
                "dark_aperture_candidate", "recessed_circular_candidate",
            }
            and not any(
                other["relation"] == "coaxial"
                and other["moving_feature"] == constraint["moving_feature"]
                and other["reference_feature"] == constraint["reference_feature"]
                for other in normalized_constraints
            )
            for constraint in normalized_constraints
        ):
            raise ValueError(
                "Aperture contact requires both coaxial and point_on_plane to the same aperture and moving feature. "
                "Coaxial needs a measured part axis, never depth-surface PCA. Without that axis, use non-contact "
                "alignment or stop."
            )
    elif regime == "free_space":
        capabilities, roles = _capability_summary(observation)
        if (
            capabilities["grasped"] is True
            and roles.get(normalized_constraints[0]["moving_feature"]) not in {
                "held_object_surface_candidate", "tracked_held_surface_candidate",
            }
        ):
            held_refs = capabilities["held_object_surface_candidate_refs"]
            tracked_refs = capabilities["tracked_held_surface_refs"]
            raise ValueError(
                "grasped=true free_space requires moving_feature from current held_object_surface_candidate_refs "
                f"{held_refs!r} or tracked_held_surface_refs {tracked_refs!r}; robot_tcp and unrelated "
                "world planes are not the held part. Do not use "
                "coaxial/parallel with surface-patch PCA axes"
            )
    else:
        capabilities, roles = _capability_summary(observation)
        if (
            capabilities["grasped"] is True
            and roles.get(normalized_constraints[0]["moving_feature"]) != "held_object_surface_candidate"
        ):
            raise ValueError(
                f"{regime} requires a currently visible held_object_surface_candidate_ref; "
                "kinematically tracked features only support free_space positioning"
            )
    return {
        "tool": "execute_goal",
        "observation_id": observation_id,
        "intent": _text(value["intent"], "execute_goal.intent"),
        "constraints": normalized_constraints,
        "motion_regime": regime,
        "stop_condition": _text(value["stop_condition"], "execute_goal.stop_condition"),
    }


def _tools_for_mode(observe_only: bool) -> dict[str, Any]:
    if observe_only:
        return {name: _TOOLS[name] for name in ("observe", "grasp", "move_tcp", "execute_goal", "stop")}
    return dict(_TOOLS)


def _tools_for_observation(observe_only: bool, observation: Mapping[str, Any]) -> dict[str, Any]:
    tools = _tools_for_mode(observe_only)
    capabilities, _ = _capability_summary(observation)
    grasped = capabilities.get("grasped")
    if grasped is not False:
        tools.pop("move_tcp", None)
    if grasped is not False or not capabilities.get("graspable_candidate_refs"):
        tools.pop("grasp", None)
    return tools


def compile_grasp_tool_call(
    value: Mapping[str, Any],
    *,
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile a typed grasp request into the existing relation-goal contract."""
    call = parse_tool_call(value, observation=observation, completed=False)
    if call["tool"] != "grasp":
        raise ValueError("only grasp tool calls can be compiled as a grasp goal")
    capabilities, _ = _capability_summary(observation)
    return {
        "tool": "execute_goal",
        "observation_id": call["observation_id"],
        "intent": "Grasp the model-selected current public feature using the measured TCP gripper.",
        "constraints": [{
            "relation": "centered_on",
            "moving_feature": capabilities["tcp_feature_refs"][0],
            "reference_feature": call["target_feature_ref"],
        }],
        "motion_regime": "grasp",
        "stop_condition": "Bilateral public contact is verified or a native executor safety limit is reached.",
    }


def _prompt(
    validation_error: str | None = None,
    *,
    observe_only: bool = False,
    feature_overlay: bool = False,
    available_tools: Sequence[str] | None = None,
) -> str:
    if observe_only:
        system_text = (
            "You are in read-only proposal mode. Review the supplied current public RGB image, "
            "feature estimates, and robot sensors. You may propose only actions listed in Available tools; "
            "the harness will validate and record them without executing them. There is no actuation, evaluator "
            "query, finish, or task-completion claim in this mode. "
            "Use the user's task text to identify the intended object and visual relation; do not replace "
            "it with a different target. Use the supplied fresh observation; observe only to resolve a "
            "specific missing or stale public fact. If grounding remains uncertain, stop."
        )
    else:
        system_text = _SYSTEM_PROMPT
    names = tuple(available_tools) if available_tools is not None else tuple(_tools_for_mode(observe_only))
    text = (
        system_text
        + "\nAvailable tools: " + ", ".join(names) + ". "
        + "Return exactly one raw JSON object with the tool name in a top-level `tool` field. "
        + "Do not return `{tool_name:{arguments:...}}`, do not add an `arguments` wrapper, and do not use Markdown fences. "
        + 'Simple form: {"tool":"stop"}. '
    )
    if "observe" in names:
        text += (
            'observe forms: {"tool":"observe"} or '
            '{"tool":"observe","observation_id":"current observation_id"}. '
        )
    if "grasp" in names:
        text += (
            'grasp form: {"tool":"grasp","observation_id":"current observation_id",'
            '"target_feature_ref":"one current graspable_candidate_refs value"}. A grasp call does not route around '
            'obstacles; inspect the visible TCP-to-target corridor first and preposition with a bounded move if needed. '
            'Before choosing, inspect each supplied current grasp-candidate crop, match its exact feature_ref to the full '
            'scene, and select only a candidate visibly on the named object and required surface. A graspable candidate '
            'proves local contact geometry, not semantic identity; for a hollow vessel use its visible exterior wall, '
            'never its interior floor or opening center. If no candidate clearly matches, observe or stop rather than '
            'guessing or choosing the nearest candidate. '
        )
    if "move_tcp" in names:
        text += (
            'move_tcp required fields: `tool`, current `observation_id`, `frame="tcp"`, and '
            '`translation_delta_m` (three TCP-local meter components). Optional `rotation_vector_rad` '
            '(three TCP-local radians) defaults to `[0,0,0]`; optional `max_duration_s` and `motion_budget` '
            'default to 3.0 seconds and 200 steps. '
            'Translation and rotation are TCP-local; '
            'rotate translation by the starting TCP orientation and apply the local rotation by right-multiplying '
            'its wxyz quaternion. Translation magnitude must exceed 1 mm or rotation must exceed 0.02 rad; '
            'smaller goals are already within executor tolerance. For collision recovery, use a meaningful bounded '
            'move along a collision-feedback clearance direction or retreat along the approach axis only when fresh '
            'RGB-D supports the route. If feedback includes example_0.01m_trial_translation_delta_m, use that exact '
            'vector as a candidate only when the image shows room; it is not auto-executed. Near 0.01 m is preferable '
            'to a 1-2 mm nudge. Any world/TCP-local '
            'hint is not proof of clearance; the normal full sweep checker may reject it without sending motion. '
            'Do not repeat a direction '
            'already rejected with zero motion. max_duration_s must be in (0, 30]; '
            'motion_budget is in [1, 2000]. '
            'The executor caps control steps by the model budget, duration, and calculated motion budget. '
            'This tool does not command the gripper or release. '
        )
    if "execute_goal" in names:
        text += (
            'execute_goal form: {"tool":"execute_goal","observation_id":"current observation_id",'
            '"intent":"...","constraints":[{"relation":"relation_name",'
            '"moving_feature":"current feature_ref","reference_feature":"current feature_ref"}],'
            '"motion_regime":"free_space|contact|grasp|release","stop_condition":"..."}. '
            "Keep relation as a plain string; include one to eight constraints. Relations: "
            + json.dumps(_RELATIONS, ensure_ascii=True, separators=(",", ":"))
            + " Motion regimes: free_space (bounded non-contact), contact (public force monitoring), "
            + "grasp (attachment needs bilateral contact), release (only with native release authorization)."
        )
    text += (
        " Choose only operations whose public-state preconditions are met. Feature refs identify visible "
        "regions; grasp-candidate labels describe geometric applicability, not semantic identity. A grasp "
        "candidate does not guarantee reachability or success. The executor handles low-level motion and "
        "microsteps. An unresolved later task requirement need not block a useful local action whose own "
        "public preconditions are met."
    )
    text += (
        " Feature geometry `model_image_pixel_*` coordinates match the resized image; `feature_ref` is an "
        "opaque ID, and any numeric suffix is not an image coordinate."
    )
    text += (
        " A feature `kind` is not a `feature_ref`. Select only exact `feature_ref` values present in the "
        "current observation feature list; refs are observation-scoped and must never be copied from older "
        "frames or tool history, even for the same-looking feature. A label absent from the current list is "
        "unavailable even if it appears in an image or prior feedback."
    )
    if feature_overlay:
        text += (
            " The supplied `*_feature_overlay` image preserves the full RGB scene and draws current "
            "feature labels. Match each visible F-number to the current feature entry with that exact "
            "`overlay_label`, then copy its exact `feature_ref`; do not infer the mapping from list order. "
            "Labels without a listed feature entry are unavailable. "
        )
    if observe_only:
        text += (
            "Use current image evidence and current feature_refs to ground the task-described objects and "
            "relations. Do not infer a feature_ref from kind, list order, or an ID suffix. Do not add action "
            "steps or unrelated constraints."
        )
    if not observe_only:
        text += (
            "Treat executor status and feedback as authoritative: do not repeat a rejected call unchanged; "
            "re-evaluate the current state and choose a corrected valid call or stop. These are generic tool "
            "semantics, not a task sequence. "
            'check_task and finish are flat forms: {"tool":"check_task"} and {"tool":"finish"}; '
            "finish requires completed=true."
        )
    text += _CAPABILITY_GUIDANCE
    if validation_error:
        text += (
            "\nPrevious response failed validation: " + validation_error[:240]
            + ". Return a corrected JSON object using only the listed fields; tool feedback is context, "
            "not an argument, and must not be echoed as a field."
        )
    return text


def _goal_motion_regime(call: Mapping[str, Any], result: Mapping[str, Any]) -> str | None:
    if call.get("tool") == "grasp":
        return "grasp"
    regime = call.get("motion_regime")
    if isinstance(regime, str):
        return regime
    compiled = result.get("compiled_goal")
    if isinstance(compiled, Mapping) and isinstance(compiled.get("motion_regime"), str):
        return compiled["motion_regime"]
    return None


def _rebound_feature_identity(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    parts = value.split(":")
    # Only strip an explicit observation/camera prefix; opaque IDs remain exact.
    if len(parts) >= 3 and parts[1].startswith("cam"):
        return ":".join(parts[2:])
    return value


def _physical_grasp_key(
    call: Mapping[str, Any], result: Mapping[str, Any],
) -> tuple[str, Any] | None:
    if _goal_motion_regime(call, result) != "grasp":
        return None
    target_ref = call.get("target_feature_ref")
    if isinstance(target_ref, str):
        identity = _rebound_feature_identity(target_ref)
        return ("grasp", (("centered_on", identity),)) if identity is not None else None

    goal: Mapping[str, Any] = call
    compiled = result.get("compiled_goal")
    if isinstance(compiled, Mapping):
        goal = compiled
    constraints = goal.get("constraints")
    if not isinstance(constraints, Sequence) or isinstance(constraints, (str, bytes)):
        return None
    target_constraints = []
    for item in constraints:
        if not isinstance(item, Mapping):
            return None
        relation = item.get("relation")
        reference = _rebound_feature_identity(item.get("reference_feature"))
        if not isinstance(relation, str) or reference is None:
            return None
        target_constraints.append((relation, reference))
    return ("grasp", tuple(sorted(target_constraints))) if target_constraints else None


def _related_collision_for_repeated(
    history: Sequence[Mapping[str, Any]],
    repeated_index: int,
    repeated_call: Mapping[str, Any],
    repeated_result: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
    feedback = repeated_result.get("feedback")
    if (
        not isinstance(feedback, str)
        or "geometrically rebinds to the same target as a prior zero-motion-step rejection" not in feedback
    ):
        return None
    repeated_key = _physical_grasp_key(repeated_call, repeated_result)
    if repeated_key is None:
        return None
    for candidate in reversed(history[:repeated_index]):
        if not isinstance(candidate, Mapping):
            continue
        result = candidate.get("result")
        call = candidate.get("call", candidate.get("tool_call"))
        if (
            not isinstance(call, Mapping)
            or not isinstance(result, Mapping)
        ):
            continue
        if _physical_grasp_key(call, result) != repeated_key:
            continue
        status = result.get("status")
        if status == "collision_blocked":
            return call, result
        # A newer same-target rejection owns the feedback; do not cross failure causes.
        if status != "repeated_no_progress":
            return None
    return None


def _collision_recovery_guidance(call: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    diagnostics = result.get("execution_diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    collision = result.get("sweep_collision_diagnostic")
    if isinstance(collision, Mapping):
        kind = collision.get("collision_kind")
        fraction = collision.get("sample_fraction")
        pixel = collision.get("obstacle_pixel_xy")
        detail = f"collision_kind={kind!r}, sample_fraction={fraction!r}, obstacle_pixel_xy={pixel!r}"
    else:
        feedback = result.get("feedback")
        detail = (
            f"executor_feedback={feedback!r}"
            if isinstance(feedback, str) and feedback
            else "collision diagnostic unavailable"
        )
    has_separation = (
        isinstance(result.get("feedback"), str)
        and "separation direction in TCP-local coordinates=" in result["feedback"]
    )
    separation_hint = (
        " This exact TCP-local translation was tested and collision_blocked without being sent; do not repeat it. "
        "Choose a different bounded recovery supported by the current view or stop if none is safe."
        if call.get("tool") == "move_tcp" and has_separation else
        " The feedback includes a candidate 5 mm TCP-local retreat derived from one colliding geometry and public "
        "RGB-D. This is only a local direction, not a route or a guarantee that other obstacles clear. Do not copy "
        "it blindly; if that exact move was blocked, choose a different visibly clear direction with useful lateral "
        "clearance, or stop. It is not executed automatically. Do not use the image pixel as a coordinate."
        if has_separation else ""
    )
    is_grasp = call.get("tool") == "grasp" or (
        call.get("tool") == "execute_goal" and call.get("motion_regime") == "grasp"
    )
    setup_note = (
        " The reported driver steps may include gripper setup; they are not evidence that the arm reached "
        "the target."
        if is_grasp and diagnostics.get("driver_step_count") != 0
        else ""
    )
    grasp_recovery_note = (
        " Because this was a grasp collision, do not retry any grasp or submit a grasp-regime `execute_goal` yet. "
        "First make a bounded `move_tcp` or `free_space` repositioning move that changes the approach around the "
        "visible obstruction, using the fresh image and current TCP frame; then reassess from the refreshed image. "
        "A tiny single-geometry retreat is not enough if it was rejected. Stop only if no grounded safe route is visible."
        if is_grasp else ""
    )
    return (
        "\nImmediate collision recovery: the previous goal was collision_blocked ("
        f"{detail}).{setup_note}{grasp_recovery_note}{separation_hint} Re-plan from the fresh public RGB-D observation and current TCP/feature "
        "geometry. Do not repeat the same relation and feature pair without new grounded evidence, and do not "
        "repeat the same physical grasp target by changing only observation or feature IDs. If current public "
        "feedback names an image pixel, use it only as a visual landmark in the current image, never as a "
        "metric or TCP coordinate. If current public geometry supports recovery, choose a bounded "
        "`free_space` approach/alignment action or permitted TCP-local `move_tcp` adjustment when `grasped=false` "
        "that routes laterally around the visible blocker; do not repeat a blocked 5 mm hint. Then reassess from the "
        "automatically refreshed observation. Do not invent coordinates, bypass normal safety checks, or use "
        "contact/grasp until current evidence supports it. If no grounded recovery is supported, stop."
    )


def _history_tcp_pose(result: Mapping[str, Any]) -> np.ndarray | None:
    observation = result.get("observation")
    robot_state = observation.get("robot_state") if isinstance(observation, Mapping) else None
    sensors = robot_state.get("sensors") if isinstance(robot_state, Mapping) else None
    pose = sensors.get("tcp_pose") if isinstance(sensors, Mapping) else None
    try:
        value = np.asarray(pose, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return value if value.shape == (7,) and np.isfinite(value).all() else None


def _world_direction_for_local_translation(
    delta_local: np.ndarray, tcp_pose: np.ndarray | None,
) -> np.ndarray | None:
    if tcp_pose is None:
        return None
    try:
        axes = _tcp_local_axes_world(tcp_pose.tolist())
        direction = sum(
            (float(delta_local[index]) * np.asarray(axes[axis], dtype=np.float64)
             for index, axis in enumerate(("x", "y", "z"))),
            start=np.zeros(3, dtype=np.float64),
        )
    except (KeyError, TypeError, ValueError):
        return None
    norm = float(np.linalg.norm(direction))
    return direction / norm if math.isfinite(norm) and norm > 1e-9 else None


def _grasp_recovery_lateral_progress_m(
    history: Sequence[Mapping[str, Any]], collision_index: int,
) -> float | None:
    collision_result = history[collision_index].get("result")
    if not isinstance(collision_result, Mapping):
        return None
    start_pose = _history_tcp_pose(collision_result)
    if start_pose is None:
        return None
    approach_axis = _grasp_recovery_approach_axis_world(history, collision_index)

    current_pose = start_pose
    for entry in history[collision_index + 1:]:
        if not isinstance(entry, Mapping):
            continue
        call = entry.get("call", entry.get("tool_call"))
        result = entry.get("result")
        diagnostics = result.get("execution_diagnostics") if isinstance(result, Mapping) else None
        if not isinstance(call, Mapping) or not isinstance(result, Mapping):
            continue
        is_reposition = call.get("tool") == "move_tcp" or (
            call.get("tool") == "execute_goal" and call.get("motion_regime") == "free_space"
        )
        steps = diagnostics.get("driver_step_count") if isinstance(diagnostics, Mapping) else None
        if (
            is_reposition and result.get("status") in {"completed", "motion_timeout"}
            and isinstance(steps, (int, float)) and steps > 0
        ):
            pose = _history_tcp_pose(result)
            if pose is not None:
                current_pose = pose
    displacement = current_pose[:3] - start_pose[:3]
    if approach_axis is None:
        return float(np.linalg.norm(displacement))
    lateral = displacement - float(np.dot(displacement, approach_axis)) * approach_axis
    return float(np.linalg.norm(lateral))


def _grasp_recovery_approach_axis_world(
    history: Sequence[Mapping[str, Any]], collision_index: int,
) -> np.ndarray | None:
    collision_result = history[collision_index].get("result")
    if not isinstance(collision_result, Mapping):
        return None
    start_pose = _history_tcp_pose(collision_result)
    resolved = collision_result.get("resolved_goal")
    target_geometry = resolved.get("target_geometry") if isinstance(resolved, Mapping) else None
    target_tcp = target_geometry.get("tcp_pose") if isinstance(target_geometry, Mapping) else None
    target_position = target_tcp.get("position_m") if isinstance(target_tcp, Mapping) else None
    try:
        vector = np.asarray(target_position, dtype=np.float64) - start_pose[:3]
    except (TypeError, ValueError):
        return None
    if vector.shape != (3,) or not np.isfinite(vector).all():
        return None
    length = float(np.linalg.norm(vector))
    return vector / length if length > 1e-6 else None


def _pending_grasp_collision_index(history: Sequence[Mapping[str, Any]]) -> int | None:
    collision_index = None
    for index, entry in enumerate(history):
        if not isinstance(entry, Mapping):
            continue
        call = entry.get("call", entry.get("tool_call"))
        result = entry.get("result")
        if isinstance(call, Mapping) and isinstance(result, Mapping):
            if result.get("status") == "collision_blocked" and _goal_motion_regime(call, result) == "grasp":
                collision_index = index
    if collision_index is None:
        return None
    progress = _grasp_recovery_lateral_progress_m(history, collision_index)
    if progress is not None and progress >= _GRASP_RECOVERY_MIN_LATERAL_M:
        return None
    return collision_index


def _grasp_recovery_lateral_candidate(
    history: Sequence[Mapping[str, Any]], observation: Mapping[str, Any] | None = None,
) -> list[float] | None:
    current_pose = None
    if isinstance(observation, Mapping):
        sensors = observation.get("robot_state", {}).get("sensors", {})
        raw_pose = sensors.get("tcp_pose") if isinstance(sensors, Mapping) else None
        try:
            pose = np.asarray(raw_pose, dtype=np.float64)
            if pose.shape == (7,) and np.isfinite(pose).all():
                current_pose = pose
        except (TypeError, ValueError):
            pass
    for index in range(len(history) - 1, -1, -1):
        entry = history[index]
        if not isinstance(entry, Mapping):
            continue
        call = entry.get("call", entry.get("tool_call"))
        result = entry.get("result")
        if not isinstance(call, Mapping) or not isinstance(result, Mapping):
            continue
        feedback = result.get("feedback")
        if not isinstance(feedback, str):
            continue
        if call.get("tool") == "move_tcp" and result.get("status") == "collision_blocked":
            candidate_pattern = r"example_0\.01m_trial_translation_delta_m=(\[[^\]]+\])"
        elif call.get("tool") == "grasp" and result.get("status") == "collision_blocked":
            candidate_pattern = r"example_0\.02m_lateral_trial_translation_delta_m=(\[[^\]]+\])"
        else:
            continue
        match = re.search(candidate_pattern, feedback)
        if match is None:
            continue
        try:
            delta = np.asarray(json.loads(match.group(1)), dtype=np.float64)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if delta.shape == (3,) and np.isfinite(delta).all() and float(np.linalg.norm(delta)) > 0.0:
            collision_pose = _history_tcp_pose(result)
            candidate_world = _world_direction_for_local_translation(delta, collision_pose)
            if candidate_world is None:
                return None

            source_diagnostics = result.get("execution_diagnostics")
            source_steps = (
                source_diagnostics.get("driver_step_count")
                if isinstance(source_diagnostics, Mapping) else None
            )
            if call.get("tool") == "move_tcp" and source_steps == 0:
                attempted_delta = np.asarray(call.get("translation_delta_m"), dtype=np.float64)
                attempted_world = (
                    _world_direction_for_local_translation(attempted_delta, collision_pose)
                    if attempted_delta.shape == (3,) and np.isfinite(attempted_delta).all()
                    else None
                )
                if (
                    attempted_world is not None
                    and float(np.dot(candidate_world, attempted_world)) >= 0.995
                ):
                    continue

            for later in history[index + 1:]:
                if not isinstance(later, Mapping):
                    continue
                later_call = later.get("call", later.get("tool_call"))
                later_result = later.get("result")
                diagnostics = (
                    later_result.get("execution_diagnostics")
                    if isinstance(later_result, Mapping) else None
                )
                if (
                    not isinstance(later_call, Mapping) or later_call.get("tool") != "move_tcp"
                    or not isinstance(later_result, Mapping)
                    or later_result.get("status") not in {"collision_blocked", "repeated_no_progress"}
                    or not isinstance(diagnostics, Mapping)
                    or diagnostics.get("driver_step_count") != 0
                ):
                    continue
                attempted_delta = np.asarray(later_call.get("translation_delta_m"), dtype=np.float64)
                if attempted_delta.shape != (3,) or not np.isfinite(attempted_delta).all():
                    continue
                attempted_world = _world_direction_for_local_translation(
                    attempted_delta, _history_tcp_pose(later_result),
                )
                if (
                    attempted_world is not None
                    and float(np.dot(candidate_world, attempted_world)) >= 0.995
                ):
                    break
            else:
                if current_pose is None:
                    return None
                try:
                    current_axes = _tcp_local_axes_world(current_pose.tolist())
                except (TypeError, ValueError):
                    return None
                # A one-point 10 mm sweep hint is only a direction; the normal full
                # sweep still validates the bounded recovery move.
                candidate_distance = max(float(np.linalg.norm(delta)), 0.02)
                candidate_local = np.asarray([
                    float(np.dot(candidate_world, np.asarray(current_axes[axis], dtype=np.float64)))
                    for axis in ("x", "y", "z")
                ]) * candidate_distance
                return [round(float(value), 5) for value in candidate_local]
    return None


def _collision_recovery_call_error(
    call: Mapping[str, Any], history: Sequence[Mapping[str, Any]],
    observation: Mapping[str, Any] | None = None,
) -> str | None:
    """Catch unusable/repeated directions only while recovering from a grasp collision."""
    collision_index = _pending_grasp_collision_index(history)
    if collision_index is None:
        return None
    tool = call.get("tool")
    if tool in {"stop", "observe", "check_task"}:
        return None
    if tool == "grasp" or (tool == "execute_goal" and call.get("motion_regime") == "grasp"):
        return (
            "a grasp approach was collision-blocked and the TCP has not achieved at least 10 mm of measured "
            "net lateral displacement around that approach; first make a bounded move_tcp or free_space "
            "reposition and obtain its fresh observation, or stop"
        )
    if tool == "execute_goal" and call.get("motion_regime") == "free_space":
        return None
    if tool != "move_tcp":
        return None

    delta = np.asarray(call.get("translation_delta_m"), dtype=np.float64)
    if delta.shape != (3,) or not np.isfinite(delta).all():
        return None
    candidate_norm = float(np.linalg.norm(delta))
    rotation = np.asarray(call.get("rotation_vector_rad", (0.0, 0.0, 0.0)), dtype=np.float64)
    if candidate_norm < 0.005 and float(np.linalg.norm(rotation)) <= 0.02:
        return (
            f"collision-recovery translation is only {candidate_norm * 1000.0:.1f} mm and is unlikely to create "
            "measurable separation from the existing sweep envelope; this request was not sent. If the fresh "
            "RGB-D view supports it, use a meaningful bounded trial near 0.01 m along the latest collision "
            "feedback's TCP-local candidate direction. The full sweep/IK/force checks still decide safety; otherwise stop."
        )
    current_pose = None
    if isinstance(observation, Mapping):
        sensors = observation.get("robot_state", {}).get("sensors", {})
        raw_pose = sensors.get("tcp_pose") if isinstance(sensors, Mapping) else None
        try:
            pose_value = np.asarray(raw_pose, dtype=np.float64)
            if pose_value.shape == (7,) and np.isfinite(pose_value).all():
                current_pose = pose_value
        except (TypeError, ValueError):
            pass
    collision_result = history[collision_index].get("result")
    collision_pose = _history_tcp_pose(collision_result) if isinstance(collision_result, Mapping) else None
    target_geometry = collision_result.get("resolved_goal", {}).get("target_geometry") if isinstance(collision_result, Mapping) else None
    target_tcp = target_geometry.get("tcp_pose") if isinstance(target_geometry, Mapping) else None
    try:
        target_position = np.asarray(target_tcp.get("position_m"), dtype=np.float64)
        approach_vector = target_position - collision_pose[:3] if collision_pose is not None else None
        approach_norm = float(np.linalg.norm(approach_vector)) if approach_vector is not None else 0.0
        approach_axis = approach_vector / approach_norm if approach_norm > 1e-9 else None
    except (TypeError, ValueError, AttributeError):
        approach_axis = None
    axial_retreat_already_used = False
    blocked_directions: list[tuple[str, np.ndarray]] = []
    for entry_index in range(collision_index + 1, len(history)):
        entry = history[entry_index]
        if not isinstance(entry, Mapping):
            continue
        previous_call = entry.get("call", entry.get("tool_call"))
        result = entry.get("result")
        diagnostics = result.get("execution_diagnostics") if isinstance(result, Mapping) else None
        if (
            not isinstance(previous_call, Mapping) or previous_call.get("tool") != "move_tcp"
            or not isinstance(result, Mapping) or not isinstance(diagnostics, Mapping)
        ):
            continue
        previous_delta = np.asarray(previous_call.get("translation_delta_m"), dtype=np.float64)
        if previous_delta.shape != (3,) or not np.isfinite(previous_delta).all():
            continue
        previous_norm = float(np.linalg.norm(previous_delta))
        if previous_norm <= 1e-9:
            continue
        previous_pose = next((
            _history_tcp_pose(prior.get("result"))
            for prior in reversed(history[collision_index:entry_index])
            if isinstance(prior, Mapping) and _history_tcp_pose(prior.get("result")) is not None
        ), None)
        if previous_pose is None:
            previous_pose = _history_tcp_pose(result)
        if previous_pose is None:
            previous_pose = current_pose
        driver_steps = diagnostics.get("driver_step_count")
        direction_pose = previous_pose
        if driver_steps == 0:
            direction_pose = _history_tcp_pose(result)
            if direction_pose is None:
                direction_pose = current_pose
        previous_world = _world_direction_for_local_translation(
            previous_delta, direction_pose,
        )
        previous_approach_dot = (
            float(np.dot(previous_world, approach_axis))
            if approach_axis is not None and previous_world is not None else None
        )
        if (
            previous_approach_dot is not None and previous_approach_dot <= -0.9
            and result.get("status") in {"completed", "motion_timeout", "collision_blocked"}
            and isinstance(driver_steps, (int, float)) and driver_steps > 0
        ):
            axial_retreat_already_used = True
        if (
            result.get("status") not in {"collision_blocked", "repeated_no_progress"}
            or driver_steps != 0
        ):
            continue
        frame = "world" if previous_world is not None else "tcp_local"
        blocked_ray = previous_world if previous_world is not None else previous_delta / previous_norm
        if not any(
            known_frame == frame and np.allclose(blocked_ray, known, rtol=0.0, atol=1e-3)
            for known_frame, known in blocked_directions
        ):
            blocked_directions.append((frame, blocked_ray))

    if candidate_norm <= 1e-9:
        return None
    requested_world = _world_direction_for_local_translation(delta, current_pose)
    direction_frame = "world" if requested_world is not None else "tcp_local"
    requested_direction = requested_world if requested_world is not None else delta / candidate_norm
    requested_approach_dot = (
        float(np.dot(requested_world, approach_axis))
        if approach_axis is not None and requested_world is not None else None
    )
    if requested_approach_dot is not None and requested_approach_dot >= 0.9:
        return (
            "this signed TCP move advances back toward the grasp target that was just collision-blocked; "
            "do not retry that approach. Move away from the target using the exact signed axial-retreat "
            "candidate from feedback, or choose a visibly clear lateral route and let the normal sweep decide"
        )
    if (
        axial_retreat_already_used and requested_approach_dot is not None
        and requested_approach_dot <= -0.9
    ):
        return (
            "the previous nonzero recovery move already used the signed direction away from the grasp target; that route "
            "does not create lateral clearance, so do not repeat it. Choose a visibly clear cross-approach "
            "direction from the current RGB-D frame, then let the normal sweep, IK, and force checks decide safety"
        )
    matching_direction = next((
        direction for frame, direction in blocked_directions
        if frame == direction_frame and float(np.dot(requested_direction, direction)) >= 0.995
    ), None)
    if matching_direction is not None:
        rejected_delta = [round(float(component), 6) for component in delta / candidate_norm]
        blocked_ray_list = [
            [round(float(component), 3) for component in direction]
            for frame, direction in blocked_directions if frame == direction_frame
        ]
        world_text = (
            f" world_direction={[round(float(component), 3) for component in requested_direction]} "
            f"and maps at the current TCP pose to world="
            f"{[round(float(component), 3) for component in requested_direction]}"
            if direction_frame == "world" else ""
        )
        return (
            "this TCP-local direction was rejected with zero motion by the RGB-D sweep or "
            "repeated_no_progress guard: "
            f"tcp_local_unit_direction={rejected_delta};{world_text} "
            f"blocked signed directions so far={blocked_ray_list} ({direction_frame} coordinates). "
            "Different TCP-local vectors can map to the same world route. Changing only a blocked direction's "
            "length is not a new route. Choose a materially different physical direction only if current RGB-D "
            "shows clearance; otherwise stop"
        )
    return None


def _recent_execution_guidance(history: Sequence[Mapping[str, Any]]) -> str:
    entry = None
    entry_index = None
    for index in range(len(history) - 1, -1, -1):
        candidate = history[index]
        if not isinstance(candidate, Mapping):
            continue
        result = candidate.get("result")
        diagnostics = result.get("execution_diagnostics") if isinstance(result, Mapping) else None
        if not isinstance(diagnostics, Mapping):
            continue
        status = result.get("status") if isinstance(result, Mapping) else None
        if (
            diagnostics.get("driver_step_count") != 0
            and status not in {"collision_blocked", "motion_timeout"}
        ):
            return ""
        entry = candidate
        entry_index = index
        break
    if entry is None:
        return ""
    call = entry.get("call", entry.get("tool_call"))
    result = entry.get("result")
    if not isinstance(call, Mapping) or not isinstance(result, Mapping):
        return ""
    status = result.get("status")
    if status == "motion_timeout":
        driver_steps = (result.get("execution_diagnostics") or {}).get("driver_step_count")
        tool = call.get("tool", "action")
        return (
            f"\nThe previous bounded {tool} action ended with motion_timeout after {driver_steps} driver steps. "
            "It was held and may have made partial motion; this is neither task completion nor proof that the target "
            "was reached. Re-check the newest public image, measured TCP pose, and finger-contact state. If the same "
            "task object and target relation are still visually supported, continue from the updated state using only "
            "fresh current feature refs; do not switch to a visually unrelated candidate just because the action "
            "timed out. If the target identity or route is no longer supported, choose another evidence-based action "
            "or stop. Normal sweep, IK, force, and action budgets remain in effect."
        )
    if status == "missing_contact_geometry" and call.get("motion_regime") == "contact":
        return (
            "\nImmediate correction: the previous contact goal had zero robot steps because current RGB-D "
            "did not support its expected contact point. Do not repeat that contact relation/target. Choose a "
            "different relation supported by the fresh observation. Inspect geometry_diagnostics for frames, "
            "timestamps, transformed goal point and support. A missing solid sample is unknown support, "
            "not evidence of free space or jam. A hole center is valid for localization, not solid contact; "
            "do not redirect the bolt to the rim. Use grounded clearance/alignment if supported, or stop."
        )
    if status == "collision_blocked":
        return _collision_recovery_guidance(call, result)
    if status == "uncertainty_too_high":
        feedback = result.get("feedback", "")
        if call.get("motion_regime") == "free_space" and "axis uncertainty" in str(feedback):
            return (
                "\nImmediate correction: the previous free-space goal was rejected because the moving-feature "
                "axis is less certain than controller precision; zero robot steps occurred. If both current "
                "position uncertainties fit the controller limit, a non-contact `centered_on` relation may "
                "localize one visually task-matching aperture without requiring an axis estimate. This does "
                "not align orientation or authorize contact/insertion. Otherwise select a different supported "
                "target or stop."
            )
        targets = sorted({
            item.get("reference_feature")
            for item in call.get("constraints", ())
            if isinstance(item, Mapping) and isinstance(item.get("reference_feature"), str)
        })
        target_text = ", ".join(targets) if targets else "the cited target"
        return (
            "\nImmediate correction: the previous goal was rejected before motion because the cited feature "
            "uncertainty exceeds controller precision. Do not reuse this target or a re-bound visual copy: "
            f"{target_text}. Select a different task-compatible current feature whose published uncertainty "
            "fits the relation, or stop if none does. The previous attempt executed zero robot steps."
        )
    if status == "repeated_no_progress":
        related_collision = _related_collision_for_repeated(
            history, entry_index, call, result,
        ) if entry_index is not None else None
        if related_collision is not None:
            return _collision_recovery_guidance(*related_collision)
        return (
            "\nImmediate correction: the previous physical target was blocked again with zero robot steps. "
            "Do not reuse it by rebinding or changing its feature ID; choose a different visually task-matching "
            "current feature, or stop if none is supported."
        )
    if status == "inconsistent_constraints" and "shared center" in str(result.get("feedback", "")):
        targets = sorted({
            item.get("reference_feature")
            for item in call.get("constraints", ())
            if isinstance(item, Mapping) and isinstance(item.get("reference_feature"), str)
        })
        target_text = ", ".join(targets) if targets else "the cited targets"
        return (
            "\nImmediate correction: the previous zero-step goal tried to center one held feature on multiple "
            "different references, which cannot share one target pose. Do not reuse this multi-target goal: "
            f"{target_text}. Visually select exactly one task-matching current reference and submit only that "
            "relation, or stop if the intended target cannot be distinguished."
        )
    if status == "ik_unreachable":
        return (
            f"\nImmediate correction: the previous goal was rejected as {status} before robot motion. "
            "Do not repeat it unchanged; choose a relation supported by current estimates or stop."
        )
    return ""


class AssemblyVLM:
    """OpenAI-compatible VLM client; only projected public observations enter requests."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_s: float = 45.0,
        trace_callback: Callable[[dict[str, Any]], None] | None = None,
        observe_only: bool = False,
        model_image_max_side: int = _MODEL_IMAGE_MAX_SIDE,
        model_history_image_max_side: int = _MODEL_HISTORY_IMAGE_MAX_SIDE,
        max_local_crops: int = _MODEL_LOCAL_CROP_MAX_COUNT,
        max_output_tokens: int = _MODEL_OUTPUT_MAX_TOKENS,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("endpoint must be a non-empty URL")
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("endpoint must be HTTP(S) without embedded credentials, query, or fragment")
        self.endpoint = endpoint
        self.model = _text(model, "model", limit=256)
        self.api_key_env = _text(api_key_env, "api_key_env", limit=128)
        self.timeout_s = _number(timeout_s, "timeout_s")
        if not 0 < self.timeout_s <= 120:
            raise ValueError("timeout_s must be in (0, 120]")
        self.model_image_max_side = _bounded_int(
            model_image_max_side,
            "model_image_max_side",
            minimum=_MODEL_IMAGE_MIN_SIDE,
            maximum=_MODEL_IMAGE_MAX_ALLOWED_SIDE,
        )
        self.model_history_image_max_side = _bounded_int(
            model_history_image_max_side,
            "model_history_image_max_side",
            minimum=_MODEL_IMAGE_MIN_SIDE,
            maximum=_MODEL_IMAGE_MAX_ALLOWED_SIDE,
        )
        if self.model_history_image_max_side > self.model_image_max_side:
            raise ValueError("model_history_image_max_side must not exceed model_image_max_side")
        self.max_local_crops = _bounded_int(
            max_local_crops,
            "max_local_crops",
            minimum=0,
            maximum=_MODEL_LOCAL_CROP_MAX_COUNT,
        )
        self.max_output_tokens = _bounded_int(
            max_output_tokens,
            "max_output_tokens",
            minimum=_MODEL_OUTPUT_MIN_TOKENS,
            maximum=_MODEL_OUTPUT_MAX_TOKENS,
        )
        self.trace_callback = trace_callback
        self.observe_only = bool(observe_only)

    def next_tool_call(
        self,
        task_goal: str,
        observation: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
        *,
        completed: bool,
    ) -> dict[str, Any]:
        if type(completed) is not bool:
            raise ValueError("completed must be a boolean")
        goal = _text(task_goal, "task_goal")
        public_observation, images = project_observation(
            observation,
            model_image_max_side=self.model_image_max_side,
            model_history_image_max_side=self.model_history_image_max_side,
        )
        public_history = _model_history(history)
        image_cameras = [camera for camera, _ in images]
        feature_overlay = any(camera.endswith("_feature_overlay") for camera in image_cameras)
        model_observation = _model_observation(public_observation, image_cameras=image_cameras)
        local_crops = _model_local_crops(public_observation, max_count=self.max_local_crops)
        local_crop_metadata = [metadata for metadata, _ in local_crops]
        if local_crop_metadata:
            model_observation["local_feature_crops"] = local_crop_metadata
        current_capabilities = model_observation.pop("capabilities", {})
        _scope_hollow_vessel_grasp_candidates(goal, current_capabilities, model_observation)
        available_tools = _tools_for_observation(self.observe_only, public_observation)
        if history and isinstance(history[-1], Mapping):
            previous_call = history[-1].get("call", history[-1].get("tool_call"))
            if isinstance(previous_call, Mapping) and previous_call.get("tool") == "observe":
                available_tools.pop("observe", None)
        available_tool_names = tuple(available_tools)
        execution_guidance = _recent_execution_guidance(history) + _task_semantic_guidance(goal)
        pending_collision_index = _pending_grasp_collision_index(history)
        if pending_collision_index is not None:
            lateral_progress = _grasp_recovery_lateral_progress_m(
                history, pending_collision_index,
            )
            progress_text = (
                "unknown" if lateral_progress is None
                else f"{lateral_progress * 1000.0:.1f} mm"
            )
            remaining_lateral_m = (
                None if lateral_progress is None
                else max(0.0, _GRASP_RECOVERY_MIN_LATERAL_M - lateral_progress)
            )
            remaining_text = (
                "unknown" if remaining_lateral_m is None
                else f"{remaining_lateral_m * 1000.0:.1f} mm"
            )
            next_move_lateral_m = (
                None if remaining_lateral_m is None
                else max(0.02, remaining_lateral_m + 0.005)
            )
            next_move_text = (
                "unknown" if next_move_lateral_m is None
                else f"{next_move_lateral_m * 1000.0:.1f} mm"
            )
            approach_axis = _grasp_recovery_approach_axis_world(
                history, pending_collision_index,
            )
            approach_axis_text = (
                "unknown" if approach_axis is None
                else str([round(float(value), 3) for value in approach_axis])
            )
            execution_guidance += (
                "\nGrasp-collision recovery is still pending: measured net TCP displacement perpendicular "
                "to the blocked approach is " + progress_text + "; remaining to the 10 mm recovery condition is "
                + remaining_text + ". The blocked approach unit vector in world coordinates is "
                + approach_axis_text + ". Do not retry this physical grasp target "
                "until a fresh, sweep-checked reposition reaches at least 10 mm of net lateral displacement. "
                "For the next bounded reposition, use the current TCP axes to choose a visibly clear route whose "
                "projected component perpendicular to that approach is at least " + next_move_text + "; do not spend "
                "repeated calls on 10 mm local nudges that project to only a few millimeters laterally. "
                "The reverse-approach example is only an axial retreat and cannot satisfy this lateral requirement; "
                "if feedback gives `example_0.01m_retreat_translation_delta_m`, copy its signed components exactly "
                "and never negate them; use that retreat at most once, then choose a visibly clear cross-approach direction from the "
                "current RGB-D. Prefer a bounded 20-40 mm lateral target when that corridor visibly has room; "
                "if the latest feedback gives an example_0.02m_lateral_trial_translation_delta_m and the current "
                "image shows room, evaluate that candidate before inventing another axis. A nonzero axial retreat "
                "may not be repeated. Keep the normal sweep, IK, and force checks. "
                "If no such route is visibly supported, stop rather than repeat the grasp."
            )
            lateral_candidate = _grasp_recovery_lateral_candidate(history, observation)
            if lateral_candidate is not None:
                execution_guidance += (
                    "\nThe latest unblocked public-RGB-D clearance direction is TCP-local "
                    f"delta_m={lateral_candidate}. If and only if the current image shows a clear corridor and "
                    "that world route has not been rejected, follow this signed direction; do not negate or mirror "
                    "any component. If feedback called it a 10 mm trial, a 20 mm bounded move along the same "
                    "direction may be considered, but the candidate is not clearance proof and still goes through the "
                    "unchanged sweep, IK, and force checks."
                )
            available_tools = {
                name: description for name, description in available_tools.items()
                if name in {"move_tcp", "observe", "stop"}
            }
            available_tool_names = tuple(available_tools)
        current_observation_id = model_observation.pop("observation_id")
        model_sensors = model_observation.get("robot_state", {}).get("sensors", {})
        if isinstance(model_sensors, dict):
            model_sensors.pop("grasped", None)
            model_sensors.pop("release_authorized", None)
            if isinstance(current_capabilities, dict) and model_sensors.get("tcp_pose") is not None:
                current_capabilities["tcp_local_axes_world"] = _tcp_local_axes_world(
                    model_sensors["tcp_pose"],
                )
        context = {
            "task_goal": goal,
            "current_capabilities": current_capabilities,
            "observation_id": current_observation_id,
            "observation": model_observation,
            "tool_history": public_history,
            "completed": completed,
        }
        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": json.dumps(context, ensure_ascii=True, allow_nan=False, separators=(",", ":")),
        }]
        frame_metadata = {
            item["frame_key"]: (index, item)
            for index, item in enumerate(public_observation.get("frame_sequence", ()), start=1)
        }
        for camera, uri in images:
            if camera in frame_metadata:
                index, metadata = frame_metadata[camera]
                role = "latest current feature overlay" if index == len(frame_metadata) else "historical raw frame; current refs do not apply"
                label = (
                    f"Chronological camera frame {index}/8: frame_key={camera}, "
                    f"frame_id={metadata['frame_id']}, capture_timestamp_s={metadata['timestamp_s']!r}; {role}."
                )
            else:
                label = f"Current public RGB frame: {camera}"
            content.extend([
                {"type": "text", "text": label},
                {"type": "image_url", "image_url": {"url": uri}},
            ])
        for metadata, uri in local_crops:
            crop_kind = metadata.get("crop_kind")
            crop_context = (
                "aperture neighborhood context crop" if crop_kind == "aperture_neighborhood"
                else "current grasp-candidate crop" if crop_kind == "grasp_candidate"
                else "local crop"
            )
            label = (
                f"{crop_context} {metadata['crop_key']} from the current acquisition (not a new frame): "
                f"feature_ref={metadata['feature_ref']}, source_frame_key={metadata['source_frame_key']}, "
                f"source_frame_id={metadata['source_frame_id']}, "
                f"source_bbox_xyxy={metadata['source_bbox_xyxy']} in original image pixels. "
                + ("This crop helps inspect local geometry but does not establish task-object identity."
                   if crop_kind == "grasp_candidate" else "")
            )
            content.extend([
                {"type": "text", "text": label},
                {"type": "image_url", "image_url": {"url": uri}},
            ])
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_output_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _prompt(
                    observe_only=self.observe_only, feature_overlay=feature_overlay,
                    available_tools=available_tool_names,
                ) + execution_guidance},
                {"role": "user", "content": content},
            ],
        }
        api_key = os.environ.get(self.api_key_env)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        pending_grasp_collision = pending_collision_index is not None
        max_response_attempts = 6 if pending_grasp_collision else 2
        for attempt in range(1, max_response_attempts + 1):
            decoded = None
            traced_payload = json.loads(json.dumps(payload))
            request_event = {
                "event": "model_request",
                "attempt": attempt,
                "image_count": len(images) + len(local_crops),
                "feature_count": len(public_observation["features"]),
                "payload": traced_payload,
            }
            if local_crop_metadata:
                request_event["local_feature_crops"] = local_crop_metadata
            if frame_sequence := public_observation.get("frame_sequence"):
                request_event["frame_sequence"] = frame_sequence
                request_event["frame_files"] = {
                    item["frame_key"]: public_observation["frames"][item["frame_key"]]
                    for item in frame_sequence
                }
            self._trace(request_event)
            request = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    raw_text = response.read().decode("utf-8", errors="replace")
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise AssemblyAgentError(f"VLM request failed: {type(exc).__name__}") from exc
            self._trace({"event": "model_response", "attempt": attempt, "raw_response": raw_text})
            try:
                response = json.loads(raw_text)
                content_text = response["choices"][0]["message"]["content"]
                if isinstance(content_text, str):
                    content_text = _strip_json_markdown_fence(content_text)
                decoded = json.loads(content_text) if isinstance(content_text, str) else content_text
                parsed_call = parse_tool_call(
                    decoded, observation=public_observation, completed=completed,
                    allowed_tools=available_tools,
                    task_goal=goal,
                )
                recovery_error = _collision_recovery_call_error(
                    parsed_call, history, observation=public_observation,
                )
                if recovery_error:
                    raise ValueError(recovery_error)
                return parsed_call
            except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
                if attempt == max_response_attempts:
                    raise AssemblyAgentError(
                        f"VLM returned an invalid tool response after {attempt - 1} retries: {exc}"
                    ) from exc
                payload["messages"][0]["content"] = _prompt(
                    str(exc), observe_only=self.observe_only, feature_overlay=feature_overlay,
                    available_tools=available_tool_names,
                ) + execution_guidance
                rejected_ref = (
                    decoded.get("target_feature_ref")
                    if isinstance(decoded, Mapping) and decoded.get("tool") == "grasp"
                    else None
                )
                interior_refs = current_capabilities.get("aperture_interior_candidate_refs", ())
                if (
                    isinstance(rejected_ref, str)
                    and rejected_ref in interior_refs
                    and "inside the currently observed aperture" in str(exc)
                ):
                    candidates = current_capabilities.get("graspable_candidate_refs", ())
                    payload["temperature"] = 0.2
                    payload["messages"].append({
                        "role": "user",
                        "content": (
                            "Correction: the proposed target_feature_ref " + rejected_ref
                            + " was rejected before execution because its observed center lies inside the bowl opening. "
                            "Do not use that reference again for this observation. The only current geometric grasp "
                            "candidate references are " + json.dumps(list(candidates), ensure_ascii=True)
                            + ". Inspect the supplied full-scene image, current feature overlay, and crops yourself; "
                            "choose one of those exact references only if it is visibly on the bowl's exterior wall. "
                            "The candidate list is not a recommendation, and the harness will not choose a point for you. "
                            "If none is visibly supported on the bowl exterior, use observe or stop; do not target the "
                            "opening, interior, plate, bottle, or another object."
                        ),
                    })
                if pending_grasp_collision:
                    payload["temperature"] = _COLLISION_RECOVERY_RETRY_TEMPERATURE
                    recovery_message = {
                        "role": "user",
                        "content": (
                            f"Recovery response rejected before execution: {exc}. The rejected proposal is not a "
                            "safe route. Do not repeat its signed direction by changing only length or a small "
                            "component. Compare the proposed world direction with every blocked signed direction "
                            "in the feedback. Inspect the current full RGB-D image and choose a materially different "
                            "lateral route only if the corridor is visibly clear; otherwise return {\"tool\":\"stop\"}. "
                            "A one-point clearance vector is only a candidate, not proof of a collision-free route; "
                            "the unchanged sweep, IK, and force checks remain authoritative."
                        ),
                    }
                    prior_recovery_message = next((
                        index for index in range(len(payload["messages"]) - 1, 1, -1)
                        if payload["messages"][index].get("role") == "user"
                        and isinstance(payload["messages"][index].get("content"), str)
                        and payload["messages"][index]["content"].startswith(
                            "Recovery response rejected before execution:"
                        )
                    ), None)
                    if prior_recovery_message is None:
                        payload["messages"].append(recovery_message)
                    else:
                        payload["messages"][prior_recovery_message] = recovery_message
        raise AssemblyAgentError("VLM returned no tool response")

    def _trace(self, event: dict[str, Any]) -> None:
        if self.trace_callback is not None:
            self.trace_callback(event)


__all__ = [
    "AssemblyAgentError", "AssemblyVLM", "MAX_CALLS", "SUPPORTED_RELATIONS",
    "compile_grasp_tool_call", "parse_tool_call", "project_observation",
]
