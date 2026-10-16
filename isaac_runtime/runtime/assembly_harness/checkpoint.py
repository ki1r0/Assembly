"""Private physical-state checkpoints for explicit assembly episode resume."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


CHECKPOINT_FORMAT = "assembly_harness_checkpoint"
CHECKPOINT_VERSION = 2
CHECKPOINT_STATUSES = frozenset({"completed", "succeeded", "grasp_contact_verified"})
PANDA_FINGER_MAX_JOINT_M = 0.04
QUATERNION_NORM_TOLERANCE = 1e-3
SEMANTIC_PROGRESSION_FIELDS = frozenset({
    "initial_sample_seen", "initially_unheld", "initial_root_z_m",
    "pickup_lift_observed", "pickup_held_streak", "pickup_proven",
    "pickup_sequence_valid", "held_before_selected_passage",
    "held_inserted_sample_count", "ever_cap_grasped", "held_seat_certified",
})


def _finite_vector(value: Any, size: int, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != size:
        raise ValueError(f"{name} must contain {size} values")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise ValueError(f"{name} must contain finite numeric values")
        result.append(float(item))
    return result


def _finite_nonnegative_scalar(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return numeric


def _validate_wxyz_quaternion(value: Sequence[float], name: str) -> None:
    norm = math.sqrt(sum(component * component for component in value))
    if norm == 0.0 or abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE:
        raise ValueError(f"{name} must be non-zero and near-unit")


def _nonnegative_integral(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative integer")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{name} must be a finite non-negative integer")
        return value
    if not isinstance(value, float):
        raise ValueError(f"{name} must be a finite non-negative integer")
    numeric = value
    if not math.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
        raise ValueError(f"{name} must be a finite non-negative integer")
    return int(numeric)


def _semantic_progression(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != SEMANTIC_PROGRESSION_FIELDS:
        raise ValueError("checkpoint.semantic_progression has an invalid shape")
    boolean_fields = (
        "initial_sample_seen", "initially_unheld", "pickup_lift_observed",
        "pickup_proven", "pickup_sequence_valid", "held_before_selected_passage",
        "ever_cap_grasped", "held_seat_certified",
    )
    for name in boolean_fields:
        if type(value[name]) is not bool:
            raise ValueError(f"checkpoint.semantic_progression.{name} must be boolean")
    integer_fields = ("pickup_held_streak", "held_inserted_sample_count")
    for name in integer_fields:
        if isinstance(value[name], bool) or not isinstance(value[name], int) or value[name] < 0:
            raise ValueError(f"checkpoint.semantic_progression.{name} must be non-negative")
    root_z = value["initial_root_z_m"]
    if root_z is not None and (
        isinstance(root_z, bool) or not isinstance(root_z, (int, float)) or not math.isfinite(float(root_z))
    ):
        raise ValueError("checkpoint.semantic_progression.initial_root_z_m must be finite or null")
    if value["initial_sample_seen"] and root_z is None:
        raise ValueError("checkpoint.semantic_progression needs initial_root_z_m after its first sample")
    return {
        **{name: value[name] for name in boolean_fields},
        "initial_root_z_m": None if root_z is None else float(root_z),
        **{name: value[name] for name in integer_fields},
    }


def validate_checkpoint(value: Any) -> dict[str, Any]:
    """Validate and normalize private checkpoint state."""
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint must be a JSON object")
    if set(value) != {"format", "version", "robot", "bolt", "provenance", "semantic_progression"}:
        raise ValueError("checkpoint has an invalid top-level shape")
    if value["format"] != CHECKPOINT_FORMAT or type(value["version"]) is not int or value["version"] != CHECKPOINT_VERSION:
        raise ValueError("checkpoint format or version is unsupported")

    robot = value["robot"]
    if not isinstance(robot, Mapping) or set(robot) != {
        "joint_positions", "joint_velocities", "gripper_target_joint_m",
    }:
        raise ValueError("checkpoint.robot has an invalid shape")
    bolt = value["bolt"]
    if not isinstance(bolt, Mapping) or set(bolt) != {"root_pose_w", "root_velocity_w"}:
        raise ValueError("checkpoint.bolt has an invalid shape")
    provenance = value["provenance"]
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "tool", "status", "terminal", "driver_step_count",
    }:
        raise ValueError("checkpoint.provenance has an invalid shape")
    if not isinstance(provenance["tool"], str) or not provenance["tool"] or len(provenance["tool"]) > 64:
        raise ValueError("checkpoint.provenance.tool is invalid")
    if not isinstance(provenance["status"], str) or provenance["status"] not in CHECKPOINT_STATUSES:
        raise ValueError("checkpoint.provenance.status is not resumable")
    if type(provenance["terminal"]) is not bool:
        raise ValueError("checkpoint.provenance.terminal must be boolean")
    if provenance["terminal"]:
        raise ValueError("checkpoint.provenance.terminal must be false for resume")
    driver_step_count = _nonnegative_integral(
        provenance["driver_step_count"], "checkpoint.provenance.driver_step_count",
    )
    joint_target = _finite_nonnegative_scalar(
        robot["gripper_target_joint_m"], "checkpoint.robot.gripper_target_joint_m",
    )
    if joint_target > PANDA_FINGER_MAX_JOINT_M:
        raise ValueError(
            "checkpoint.robot.gripper_target_joint_m exceeds the Panda finger range"
        )
    root_pose_w = _finite_vector(bolt["root_pose_w"], 7, "checkpoint.bolt.root_pose_w")
    _validate_wxyz_quaternion(root_pose_w[3:], "checkpoint.bolt.root_pose_w quaternion")

    return {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "robot": {
            "joint_positions": _finite_vector(robot["joint_positions"], 9, "checkpoint.robot.joint_positions"),
            "joint_velocities": _finite_vector(robot["joint_velocities"], 9, "checkpoint.robot.joint_velocities"),
            "gripper_target_joint_m": joint_target,
        },
        "bolt": {
            "root_pose_w": root_pose_w,
            "root_velocity_w": _finite_vector(bolt["root_velocity_w"], 6, "checkpoint.bolt.root_velocity_w"),
        },
        "provenance": {
            "tool": provenance["tool"],
            "status": provenance["status"],
            "terminal": provenance["terminal"],
            "driver_step_count": driver_step_count,
        },
        "semantic_progression": _semantic_progression(value["semantic_progression"]),
    }


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise ValueError(f"resume checkpoint does not exist: {checkpoint_path}")
    try:
        value = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"resume checkpoint could not be read: {checkpoint_path}") from exc
    return validate_checkpoint(value)


def write_checkpoint(path: str | Path, value: Mapping[str, Any]) -> Path:
    checkpoint = validate_checkpoint(value)
    checkpoint_path = Path(path).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(checkpoint, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(checkpoint_path)
    return checkpoint_path


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_STATUSES",
    "CHECKPOINT_VERSION",
    "PANDA_FINGER_MAX_JOINT_M",
    "QUATERNION_NORM_TOLERANCE",
    "SEMANTIC_PROGRESSION_FIELDS",
    "load_checkpoint",
    "validate_checkpoint",
    "write_checkpoint",
]
