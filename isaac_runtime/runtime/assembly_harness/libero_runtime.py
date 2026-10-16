"""Single-task LIBERO bridge for the shared RGB-D VLM assembly harness."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from .executor import (
    AssemblyExecutor,
    ExecutorLimits,
    _compose_pose,
    _contact_surface_point_mask,
    _moving_feature_at_contact,
    _validated_contact_pairs,
)
from .perception import (
    PublicRGBDFrame,
    build_public_observation,
    estimate_camera_features,
    point_cloud_world,
)


TASK_GOAL = "Pick up the alphabet soup and place it in the basket."
TASK_BDDL = "pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl"
TASK_INIT = "pick_up_the_alphabet_soup_and_place_it_in_the_basket.pruned_init"
CONTROL_HZ = 20
CAMERA_SIDE = 512
FRAME_HISTORY = 8
LIBERO_RGBD_OBSTACLE_STRIDE = 3
ROBOT_INSTANCE_PREFIXES = (
    "Panda",
    "MountedPanda",
    "RethinkMount",
    "OnTheGroundPanda",
    "NullMount",
)
SWEEP_COLLISION_RECORD_LIMIT = 256


@dataclass(frozen=True)
class LiberoTaskConfig:
    task_suite: str
    task_name: str
    task_id: int
    task_order_index: int
    task_instruction: str
    bddl_file: str
    init_state_file: str
    init_state_id: int
    goal_expression: str
    upstream_commit: str | None = None


def load_libero_task_config(path: str | Path) -> LiberoTaskConfig:
    config_path = Path(path).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("LIBERO task config must be a JSON object")
    required_strings = (
        "task_suite", "task_name", "task_instruction", "bddl_file",
        "init_state_file", "goal_expression",
    )
    for key in required_strings:
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise ValueError(f"LIBERO task config field {key!r} must be non-empty text")
    for key in ("bddl_file", "init_state_file"):
        if Path(raw[key]).name != raw[key]:
            raise ValueError(f"LIBERO task config field {key!r} must be a filename")
    integer_fields = ("task_id", "task_order_index", "init_state_id")
    for key in integer_fields:
        if type(raw.get(key)) is not int or raw[key] < 0:
            raise ValueError(f"LIBERO task config field {key!r} must be a non-negative integer")
    commit = raw.get("upstream_commit")
    if commit is not None and (not isinstance(commit, str) or not commit.strip()):
        raise ValueError("LIBERO task config upstream_commit must be non-empty text when provided")
    return LiberoTaskConfig(
        task_suite=raw["task_suite"],
        task_name=raw["task_name"],
        task_id=raw["task_id"],
        task_order_index=raw["task_order_index"],
        task_instruction=raw["task_instruction"],
        bddl_file=raw["bddl_file"],
        init_state_file=raw["init_state_file"],
        init_state_id=raw["init_state_id"],
        goal_expression=raw["goal_expression"],
        upstream_commit=commit,
    )


def _task_config_from_args(args: Any) -> LiberoTaskConfig:
    config_path = getattr(args, "task_config", None)
    if config_path:
        path = Path(config_path).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        return load_libero_task_config(path)
    return LiberoTaskConfig(
        task_suite="libero_object",
        task_name=Path(TASK_BDDL).stem,
        task_id=0,
        task_order_index=0,
        task_instruction=TASK_GOAL,
        bddl_file=TASK_BDDL,
        init_state_file=TASK_INIT,
        init_state_id=0,
        goal_expression="",
    )


def _normalized_instruction(value: str) -> str:
    return " ".join(value.strip().rstrip(".").lower().split())


def _instance_labels(instances_to_ids: Mapping[str, Any]) -> dict[str, int]:
    return {str(name): index + 1 for index, name in enumerate(instances_to_ids)}


def _robot_instance_labels(instances_to_ids: Mapping[str, Any]) -> set[int]:
    return {
        label for name, label in _instance_labels(instances_to_ids).items()
        if name.startswith(ROBOT_INSTANCE_PREFIXES)
    }


def _moving_robot_collision_geoms(
    robot_geom_ids: set[int],
    geom_body_ids: Any,
    geom_contype: Any,
    geom_conaffinity: Any,
    static_body_ids: set[int],
) -> set[int]:
    body_ids = np.asarray(geom_body_ids, dtype=np.int64)
    contype = np.asarray(geom_contype, dtype=np.int64)
    conaffinity = np.asarray(geom_conaffinity, dtype=np.int64)
    if body_ids.ndim != 1 or contype.shape != body_ids.shape or conaffinity.shape != body_ids.shape:
        raise ValueError("MuJoCo collision geometry metadata must have matching one-dimensional arrays")
    if any(geom_id < 0 or geom_id >= len(body_ids) for geom_id in robot_geom_ids):
        raise ValueError("robot collision geometry ID is outside the MuJoCo model")
    return {
        geom_id for geom_id in robot_geom_ids
        if body_ids[geom_id] not in static_body_ids and (contype[geom_id] or conaffinity[geom_id])
    }


def _mesh_geom_local_bounds(model: Any, geom_id: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a conservative local AABB from MuJoCo's compiled collision-mesh vertices."""
    try:
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        vertices = np.asarray(model.mesh_vert[start:start + count], dtype=np.float64)
    except (AttributeError, IndexError, TypeError, ValueError):
        return None
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        return None
    return vertices.min(axis=0), vertices.max(axis=0)


def _mask_robot_depth(
    depth_m: Any,
    segmentation: Any,
    robot_labels: set[int],
) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32).squeeze().copy()
    labels = np.asarray(segmentation).squeeze()
    if depth.ndim != 2 or labels.shape != depth.shape:
        raise ValueError("LIBERO depth and instance segmentation must be matching HxW images")
    depth[(labels == 0) | np.isin(labels, tuple(robot_labels))] = np.nan
    return depth


def _feature_instance_labels(
    features: Sequence[Mapping[str, Any]],
    segmentation: np.ndarray,
    robot_labels: set[int],
) -> dict[str, int]:
    result: dict[str, int] = {}
    height, width = segmentation.shape
    for feature in features:
        geometry = feature.get("geometry")
        pixel = geometry.get("pixel_centroid_xy") if isinstance(geometry, Mapping) else None
        ref = feature.get("feature_ref")
        if not isinstance(ref, str) or not isinstance(pixel, Sequence) or len(pixel) != 2:
            continue
        x, y = (int(round(float(value))) for value in pixel)
        if not (0 <= x < width and 0 <= y < height):
            continue
        label = int(segmentation[y, x])
        if label > 0 and label not in robot_labels:
            result[ref] = label
    return result


def _project_world_point_to_frame(
    point_world_m: Any,
    frame: PublicRGBDFrame | None,
    segmentation: Any,
    robot_labels: set[int],
) -> dict[str, Any] | None:
    """Privately map a collision point back to its source camera pixel and label."""
    if not isinstance(frame, PublicRGBDFrame) or segmentation is None:
        return None
    try:
        point = np.asarray(point_world_m, dtype=np.float64)
        labels = np.asarray(segmentation).squeeze()
    except (TypeError, ValueError):
        return None
    if point.shape != (3,) or not np.isfinite(point).all() or labels.shape != frame.depth_m.shape:
        return None
    point_camera = frame.T_world_camera[:3, :3].T @ (
        point - frame.T_world_camera[:3, 3]
    )
    if not np.isfinite(point_camera).all() or point_camera[2] <= 1e-9:
        return None
    pixel_h = frame.K @ (point_camera / point_camera[2])
    if not np.isfinite(pixel_h).all() or abs(float(pixel_h[2])) <= 1e-12:
        return None
    u, v = (int(round(float(value))) for value in pixel_h[:2] / pixel_h[2])
    height, width = labels.shape
    if not (0 <= u < width and 0 <= v < height):
        return {
            "camera_frame_id": frame.frame_id,
            "camera_timestamp_s": frame.timestamp_s,
            "pixel_xy": [u, v],
            "inside_image": False,
        }
    label = int(labels[v, u])
    depth = float(frame.depth_m[v, u])
    return {
        "camera_frame_id": frame.frame_id,
        "camera_timestamp_s": frame.timestamp_s,
        "pixel_xy": [u, v],
        "inside_image": True,
        "instance_label": label,
        "is_robot_label": label in robot_labels,
        "masked_depth_m": round(depth, 6) if math.isfinite(depth) else None,
        "projected_camera_depth_m": float(point_camera[2]),
    }


def _exclude_instance_points(
    points_world_m: Any,
    pixel_xy: Any,
    segmentation: Any,
    excluded_labels: set[int],
) -> np.ndarray:
    points = np.asarray(points_world_m, dtype=np.float64)
    pixels = np.asarray(pixel_xy)
    labels = np.asarray(segmentation).squeeze()
    if points.ndim != 2 or points.shape[1] != 3 or pixels.shape != (len(points), 2):
        raise ValueError("point cloud and source pixels must have matching Nx3 and Nx2 shapes")
    if labels.ndim != 2 or not np.isfinite(pixels).all():
        raise ValueError("segmentation and point-cloud pixels must be finite images")
    if not excluded_labels:
        return points
    xy = np.rint(pixels).astype(np.int64)
    if (np.any(xy[:, 0] < 0) or np.any(xy[:, 0] >= labels.shape[1])
            or np.any(xy[:, 1] < 0) or np.any(xy[:, 1] >= labels.shape[0])):
        raise ValueError("point-cloud source pixel falls outside the segmentation image")
    return points[~np.isin(labels[xy[:, 1], xy[:, 0]], tuple(excluded_labels))]


def _supported_rgbd_grasp_width(
    feature: Mapping[str, Any],
    opening_axis_world: Any,
    obstacle_points_world_m: Any,
    max_gripper_width_m: float,
) -> float | None:
    """Estimate the widest symmetric pinch span supported by current RGB-D on both sides."""
    geometry = feature.get("geometry")
    uncertainty = feature.get("uncertainty")
    if (
        feature.get("kind") != "depth_surface_patch"
        or feature.get("frame") not in {"world", "world_m"}
        or not isinstance(geometry, Mapping)
        or not isinstance(uncertainty, Mapping)
        or geometry.get("grasp_width_m") is not None
    ):
        return None
    try:
        axis = np.asarray(opening_axis_world, dtype=np.float64)
        center = np.asarray(geometry.get("position_m"), dtype=np.float64)
        extent = np.asarray(geometry.get("extent_m"), dtype=np.float64)
        surface = np.asarray(geometry.get("contact_surface_points_world_m"), dtype=np.float64)
        cloud = np.asarray(obstacle_points_world_m, dtype=np.float64)
        position_std = float(uncertainty.get("position_std_m"))
        max_width = float(max_gripper_width_m)
    except (TypeError, ValueError):
        return None
    if (
        axis.shape != (3,) or center.shape != (3,) or extent.shape != (3,)
        or surface.ndim != 2 or surface.shape[1:] != (3,) or len(surface) < 2
        or cloud.ndim != 2 or cloud.shape[1:] != (3,) or len(cloud) == 0
        or not np.isfinite(axis).all() or not np.isfinite(center).all()
        or not np.isfinite(extent).all() or not np.isfinite(surface).all()
        or not np.isfinite(cloud).all() or not math.isfinite(position_std) or position_std < 0.0
        or not math.isfinite(max_width) or max_width <= 0.0
        or float(np.max(extent)) > 2.0 * max_width
    ):
        return None
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-9:
        return None
    axis /= axis_norm
    tolerance = min(0.008, max(0.003, 3.0 * position_std))
    width_limit = min(max_width, float(np.ptp(surface @ axis)))
    min_width = 2.0 * tolerance
    if width_limit < min_width:
        return None

    widths = np.arange(min_width, width_limit + 0.0005, 0.001)
    if len(widths) == 0 or widths[-1] < width_limit - 1e-9:
        widths = np.append(widths, width_limit)
    for width in widths[::-1]:
        sites = (center - axis * (width * 0.5), center + axis * (width * 0.5))
        if not all(
            float(np.min(np.linalg.norm(surface - site, axis=1))) <= tolerance
            for site in sites
        ):
            continue
        if all(float(np.min(np.linalg.norm(cloud - site, axis=1))) <= tolerance for site in sites):
            return float(width)
    return None


def _as_numpy(value: Any) -> np.ndarray:
    if callable(getattr(value, "detach", None)):
        value = value.detach()
    if callable(getattr(value, "cpu", None)):
        value = value.cpu()
    if callable(getattr(value, "numpy", None)):
        value = value.numpy()
    return np.asarray(value)


def _quat_multiply(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    aw, ax, ay, az = np.asarray(left, dtype=np.float64)
    bw, bx, by, bz = np.asarray(right, dtype=np.float64)
    return np.asarray((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ))


def _quat_inverse(value: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    return quaternion * np.asarray((1.0, -1.0, -1.0, -1.0)) / float(np.dot(quaternion, quaternion))


def _rotation_vector(target: Sequence[float], current: Sequence[float]) -> np.ndarray:
    delta = _quat_multiply(target, _quat_inverse(current))
    if delta[0] < 0.0:
        delta = -delta
    vector_norm = float(np.linalg.norm(delta[1:]))
    if vector_norm <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    return delta[1:] * (2.0 * math.atan2(vector_norm, float(delta[0])) / vector_norm)


def _emit_sweep_collision_diagnostic(
    callback: Callable[[Mapping[str, Any]], None] | None,
    evidence: Mapping[str, Any],
) -> None:
    """Keep private diagnostics best-effort so they cannot affect control."""
    if callback is None:
        return
    try:
        callback(evidence)
    except Exception:
        pass


def _held_object_sweep_collision(
    obstacle_points_world_m: np.ndarray,
    start_tcp_pose: Sequence[float],
    end_tcp_pose: Sequence[float],
    moving_geometry: Mapping[str, Any],
    contact_pairs: Sequence[Mapping[str, Any]],
    feature_labels: Mapping[str, int],
    margin_m: float,
    *,
    diagnostic_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> bool:
    radius = moving_geometry.get("collision_radius_m", moving_geometry.get("bounding_radius_m"))
    local_pose = moving_geometry.get("_tcp_to_feature_pose")
    if (
        isinstance(radius, bool) or not isinstance(radius, (int, float))
        or not math.isfinite(float(radius)) or float(radius) <= 0.0
        or not isinstance(local_pose, Sequence) or len(local_pose) != 7
    ):
        return True
    start = np.asarray(start_tcp_pose, dtype=np.float64)
    end = np.asarray(end_tcp_pose, dtype=np.float64)
    local = np.asarray(local_pose, dtype=np.float64)
    if (start.shape != (7,) or end.shape != (7,) or local.shape != (7,)
            or not np.isfinite(np.concatenate((start, end, local))).all()):
        return True
    radius = float(radius)
    start_position = _compose_pose(start, local)[0]
    initial_distances = np.linalg.norm(obstacle_points_world_m - start_position, axis=1)
    start_margin_only = (initial_distances > radius) & (initial_distances <= radius + margin_m)
    travel_m = float(np.linalg.norm(end[:3] - start[:3]))
    rotation_rad = float(np.linalg.norm(_rotation_vector(end[3:], start[3:])))
    sample_count = int(math.ceil(max(travel_m, radius * rotation_rad) / max(margin_m * 0.5, 1e-4)))
    if sample_count > 128:
        return True
    sample_count = max(1, sample_count)
    for index in range(sample_count + 1):
        fraction = index / sample_count
        tcp_pose = _interpolate_pose(start, end, fraction)
        moving_position = _compose_pose(tcp_pose, local)[0]
        distances = np.linalg.norm(obstacle_points_world_m - moving_position, axis=1)
        collisions = distances <= radius + margin_m
        if fraction > 0.0:
            collisions &= ~(start_margin_only & (distances >= initial_distances - 1e-6))
        for pair in contact_pairs:
            if (pair["moving_role"] != "held_object"
                    or feature_labels.get(str(pair["reference_feature_ref"])) is None
                    or not _moving_feature_at_contact(moving_position, pair)
                    or not np.any(collisions)):
                continue
            candidates = np.flatnonzero(collisions)
            collisions[candidates] &= ~_contact_surface_point_mask(
                obstacle_points_world_m[candidates], pair,
            )
        if np.any(collisions):
            candidates = np.flatnonzero(collisions)
            nearest = int(candidates[np.argmin(distances[candidates])])
            _emit_sweep_collision_diagnostic(
                diagnostic_callback,
                {
                    "sample_fraction": float(fraction),
                    "nearest_obstacle_point_m": obstacle_points_world_m[nearest].tolist(),
                    "nearest_obstacle_distance_m": float(distances[nearest]),
                },
            )
            return True
    return False


def _validate_stationary_hold(
    *,
    previous_held_label: int | None,
    current_held_label: int | None,
    previous_width_m: float,
    current_width_m: float,
    wrench: Sequence[float],
    finger_forces_n: Sequence[float],
    max_force_n: float,
    width_tolerance_m: float,
) -> None:
    wrench_values = np.asarray(wrench, dtype=np.float64)
    finger_values = np.asarray(finger_forces_n, dtype=np.float64)
    if (wrench_values.shape != (6,) or finger_values.shape != (2,)
            or not np.isfinite(wrench_values).all() or not np.isfinite(finger_values).all()):
        raise RuntimeError("stationary camera hold requires finite wrench and finger-force readings")
    if (np.linalg.norm(wrench_values[:3]) > max_force_n
            or np.max(finger_values) > max_force_n):
        raise RuntimeError("stationary camera hold stopped at the existing force limit")
    if previous_held_label is not None and current_held_label != previous_held_label:
        raise RuntimeError("stationary camera hold stopped because the held object contact was lost")
    if abs(float(current_width_m) - float(previous_width_m)) > width_tolerance_m:
        raise RuntimeError("stationary camera hold stopped because gripper width was not stable")


def _pose_from_site(mujoco: Any, model: Any, data: Any, site_id: int) -> tuple[float, ...]:
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, data.site_xmat[site_id])
    return (*data.site_xpos[site_id].astype(float).tolist(), *quaternion.tolist())


def _raw_wrench(observation: Mapping[str, Any], robot: Any = None) -> np.ndarray:
    force_value = observation.get("robot0_eef_force")
    torque_value = observation.get("robot0_eef_torque")
    if force_value is None and robot is not None:
        force_value = robot.ee_force
    if torque_value is None and robot is not None:
        torque_value = robot.ee_torque
    force = _as_numpy(force_value if force_value is not None else ()).astype(np.float64).reshape(-1)
    torque = _as_numpy(torque_value if torque_value is not None else ()).astype(np.float64).reshape(-1)
    if force.size != 3 or torque.size != 3 or not np.isfinite(force).all() or not np.isfinite(torque).all():
        raise RuntimeError("LIBERO observation is missing the Panda end-effector force/torque sensors")
    return np.concatenate((force, torque))


class LiberoMujocoDriver:
    """Panda OSC driver with MuJoCo contact checks and public RGB-D sweep checks."""

    def __init__(self, runtime: "LiberoAssemblyRuntime", wrench_baseline: Sequence[float]) -> None:
        import mujoco

        self.runtime = runtime
        self.env = runtime.env
        self.sim = self.env.sim
        self.mujoco = mujoco
        self.model = getattr(self.sim.model, "_model", self.sim.model)
        self.data = getattr(self.sim.data, "_data", self.sim.data)
        self.robot = self.env.robots[0]
        self._wrench_baseline = np.asarray(wrench_baseline, dtype=np.float64).reshape(6)
        self.sweep_margin_m = 0.005
        self.max_gripper_width_m = self._read_max_gripper_width()
        self._gripper_command = 0.0
        self._gripper_force_limit_n = 3.0
        self._held_label: int | None = None
        self._support_count = 0
        self._support_timestamp_s: float | None = None
        self.last_sweep_collision_diagnostic: dict[str, Any] | None = None
        self._configure_model_ids()
        self.measured_gripper_width_max_m = self._calibrate_measured_gripper_width()
        self.joint_limits = self._read_joint_limits()
        self._commanded_tcp_pose = self._tcp_pose()

    def _configure_model_ids(self) -> None:
        mujoco = self.mujoco
        model = self.model
        self.arm_joint_ids = []
        for index in range(1, 8):
            name = f"robot0_joint{index}"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise RuntimeError(f"LIBERO Panda arm is missing expected joint {name}")
            self.arm_joint_ids.append(joint_id)
        self.arm_qpos_ids = np.asarray(model.jnt_qposadr[self.arm_joint_ids], dtype=int)
        self.arm_dof_ids = np.asarray(model.jnt_dofadr[self.arm_joint_ids], dtype=int)

        robot_model = getattr(self.robot, "robot_model", None)
        root_name = getattr(robot_model, "root_body", "robot0_base")
        self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(root_name))
        if self.base_body_id < 0:
            raise RuntimeError("LIBERO Panda root body is unavailable for OSC frame conversion")

        site_names = []
        site_id = getattr(self.robot, "eef_site_id", None)
        if isinstance(site_id, (int, np.integer)) and 0 <= int(site_id) < model.nsite:
            self.tcp_site_id = int(site_id)
        else:
            important_sites = getattr(self.robot.gripper, "important_sites", {})
            if isinstance(important_sites, Mapping):
                value = important_sites.get("grip_site")
                if isinstance(value, str):
                    site_names.append(value)
            site_names.append("robot0_grip_site")
            self.tcp_site_id = -1
            for name in site_names:
                if not name.startswith("robot0_"):
                    name = f"robot0_{name}"
                candidate = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                if candidate >= 0:
                    self.tcp_site_id = int(candidate)
                    break
            if self.tcp_site_id < 0:
                raise RuntimeError("LIBERO Panda gripper TCP site could not be resolved")

        model_info = self.env.env.model
        instances_to_ids = getattr(model_info, "instances_to_ids", None)
        geom_ids_to_instances = getattr(model_info, "geom_ids_to_instances", None)
        if not isinstance(instances_to_ids, Mapping) or not isinstance(geom_ids_to_instances, Mapping):
            raise RuntimeError("LIBERO instance segmentation metadata is unavailable")
        self._instance_ids = _instance_labels(instances_to_ids)
        self._robot_labels = _robot_instance_labels(instances_to_ids)
        if not self._robot_labels:
            raise RuntimeError("LIBERO segmentation has no Panda robot instance IDs")
        self._geom_labels: dict[int, int] = {}
        for raw_geom_id, instance_name in geom_ids_to_instances.items():
            label = self._instance_ids.get(str(instance_name))
            if label is not None:
                self._geom_labels[int(raw_geom_id)] = label
        self._robot_geom_ids = {
            geom_id for geom_id, label in self._geom_labels.items()
            if label in self._robot_labels
            and (int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id]))
        }
        if not self._robot_geom_ids:
            raise RuntimeError("LIBERO robot geometry cannot be mapped to instance segmentation")
        static_body_ids = set()
        body_id = self.base_body_id
        while body_id > 0:
            static_body_ids.add(body_id)
            body_id = int(model.body_parentid[body_id])
        self._static_robot_geom_ids = {
            geom_id for geom_id in self._robot_geom_ids
            if int(model.geom_bodyid[geom_id]) in static_body_ids
        }
        self._moving_robot_geom_ids = _moving_robot_collision_geoms(
            self._robot_geom_ids,
            model.geom_bodyid,
            model.geom_contype,
            model.geom_conaffinity,
            static_body_ids,
        )
        if not self._moving_robot_geom_ids:
            raise RuntimeError("LIBERO Panda collision model has no moving collision geometry")
        mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
        self._mesh_local_bounds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for geom_id in self._moving_robot_geom_ids:
            if int(model.geom_type[geom_id]) != mesh_type:
                continue
            bounds = _mesh_geom_local_bounds(model, geom_id)
            if bounds is None:
                raise RuntimeError(
                    f"LIBERO Panda collision mesh {geom_id} has no valid compiled vertices"
                )
            self._mesh_local_bounds[int(geom_id)] = bounds

        important_geoms = getattr(self.robot.gripper, "important_geoms", {})
        if not isinstance(important_geoms, Mapping):
            raise RuntimeError("Panda gripper contact geometry is unavailable")
        self._finger_geom_ids: dict[str, set[int]] = {"left": set(), "right": set()}
        self._left_pad_geom_ids: set[int] = set()
        self._right_pad_geom_ids: set[int] = set()
        for key, names in important_geoms.items():
            side = "left" if "left" in str(key).lower() else "right" if "right" in str(key).lower() else None
            if side is None:
                continue
            if isinstance(names, str):
                names = (names,)
            if not isinstance(names, Sequence):
                continue
            for raw_name in names:
                name = str(raw_name)
                candidates = (name, name if name.startswith("robot0_") else f"robot0_{name}")
                geom_id = next((
                    candidate
                    for candidate in (
                        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, item)
                        for item in candidates
                    )
                    if candidate >= 0
                ), -1)
                if geom_id < 0:
                    continue
                self._finger_geom_ids[side].add(int(geom_id))
                if "pad" in str(key).lower():
                    (self._left_pad_geom_ids if side == "left" else self._right_pad_geom_ids).add(int(geom_id))
        if not self._finger_geom_ids["left"] or not self._finger_geom_ids["right"]:
            raise RuntimeError("Panda left/right gripper collision geoms could not be resolved")
        if not self._left_pad_geom_ids:
            self._left_pad_geom_ids = set(self._finger_geom_ids["left"])
        if not self._right_pad_geom_ids:
            self._right_pad_geom_ids = set(self._finger_geom_ids["right"])

        self._movable_labels = set()
        free_joint_type = int(mujoco.mjtJoint.mjJNT_FREE)
        for geom_id, label in self._geom_labels.items():
            body_id = int(model.geom_bodyid[geom_id])
            while body_id > 0:
                joint_id = int(model.body_jntadr[body_id])
                if joint_id >= 0 and int(model.jnt_type[joint_id]) == free_joint_type:
                    self._movable_labels.add(label)
                    break
                body_id = int(model.body_parentid[body_id])
        if not self._movable_labels:
            raise RuntimeError("LIBERO scene has no free-body object instances for contact sensing")

    def _read_joint_limits(self) -> tuple[tuple[float, float], ...]:
        ranges = np.asarray(self.model.jnt_range[self.arm_joint_ids], dtype=np.float64)
        limited = np.asarray(self.model.jnt_limited[self.arm_joint_ids], dtype=bool)
        if ranges.shape != (7, 2) or not limited.all() or not np.isfinite(ranges).all():
            raise RuntimeError("LIBERO Panda joint limits are not seven finite bounded intervals")
        if np.any(ranges[:, 0] >= ranges[:, 1]):
            raise RuntimeError("LIBERO Panda joint limits are invalid")
        return tuple((float(low), float(high)) for low, high in ranges)

    def _tcp_pose(self) -> tuple[float, ...]:
        return _pose_from_site(self.mujoco, self.model, self.data, self.tcp_site_id)

    def _tcp_velocity(self) -> np.ndarray:
        jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
        self.mujoco.mj_jacSite(
            self.model, self.data, jacobian_position, jacobian_rotation, self.tcp_site_id,
        )
        return np.concatenate((
            jacobian_position @ self.data.qvel,
            jacobian_rotation @ self.data.qvel,
        ))

    def _gripper_geometry(self) -> tuple[float, np.ndarray]:
        left_ids = sorted(self._left_pad_geom_ids)
        right_ids = sorted(self._right_pad_geom_ids)
        left = np.mean(self.data.geom_xpos[left_ids], axis=0)
        right = np.mean(self.data.geom_xpos[right_ids], axis=0)
        axis = left - right
        center_distance = float(np.linalg.norm(axis))
        if not math.isfinite(center_distance) or center_distance <= 1e-6:
            raise RuntimeError("Panda gripper width or opening axis is unavailable")
        axis /= center_distance

        def projected_half_extent(geom_ids: Sequence[int]) -> float:
            extents = []
            box_type = int(self.mujoco.mjtGeom.mjGEOM_BOX)
            for geom_id in geom_ids:
                if int(self.model.geom_type[geom_id]) != box_type:
                    raise RuntimeError("Panda gripper pad geometry must be a box")
                rotation = np.asarray(self.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
                half_sizes = np.asarray(self.model.geom_size[geom_id], dtype=np.float64).reshape(-1)
                if half_sizes.size < 3 or not np.isfinite(rotation).all() or not np.isfinite(half_sizes[:3]).all():
                    raise RuntimeError("Panda gripper pad box geometry is unavailable")
                # MuJoCo geom_xmat maps local box axes into world axes.
                extents.append(float(np.sum(np.abs(rotation.T @ axis) * half_sizes[:3])))
            if not extents or not np.isfinite(extents).all():
                raise RuntimeError("Panda gripper pad box extent is unavailable")
            return float(max(extents))

        inner_face_gap = center_distance - projected_half_extent(left_ids) - projected_half_extent(right_ids)
        if not math.isfinite(inner_face_gap):
            raise RuntimeError("Panda gripper width is unavailable")
        return max(0.0, float(inner_face_gap)), axis

    def _contact_state(self) -> tuple[tuple[bool, bool], tuple[float, float], int | None, bool]:
        side_labels: dict[str, set[int]] = {"left": set(), "right": set()}
        side_forces: dict[str, dict[int, float]] = {"left": {}, "right": {}}
        support_contact = False
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            label1, label2 = self._geom_labels.get(geom1), self._geom_labels.get(geom2)
            force = np.zeros(6, dtype=np.float64)
            self.mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            force_n = abs(float(force[0]))
            for side in ("left", "right"):
                finger_ids = self._finger_geom_ids[side]
                if geom1 in finger_ids and label2 in self._movable_labels and force_n >= 0.02:
                    side_labels[side].add(label2)
                    side_forces[side][label2] = max(side_forces[side].get(label2, 0.0), force_n)
                if geom2 in finger_ids and label1 in self._movable_labels and force_n >= 0.02:
                    side_labels[side].add(label1)
                    side_forces[side][label1] = max(side_forces[side].get(label1, 0.0), force_n)
            held_label = self._held_label
            if held_label is not None:
                support_contact |= (
                    force_n >= 0.05
                    and ((label1 == held_label and label2 not in self._robot_labels and label2 != held_label)
                         or (label2 == held_label and label1 not in self._robot_labels and label1 != held_label))
                )

        common = side_labels["left"] & side_labels["right"]
        held_label = min(common) if common else None
        contacts = (held_label is not None, held_label is not None)
        forces = (
            side_forces["left"].get(held_label, 0.0) if held_label is not None else 0.0,
            side_forces["right"].get(held_label, 0.0) if held_label is not None else 0.0,
        )
        return contacts, forces, held_label, support_contact

    def _wrench(self) -> np.ndarray:
        return _raw_wrench(self.runtime._last_obs, self.robot) - self._wrench_baseline

    def _record_sweep_collision_evidence(self, evidence: Mapping[str, Any]) -> None:
        recorder = getattr(self.runtime, "_record_sweep_collision", None)
        if not callable(recorder):
            return
        try:
            recorder(evidence)
        except Exception:
            pass

    def _geom_name(self, geom_id: int) -> str | None:
        try:
            name = self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_GEOM, int(geom_id),
            )
        except Exception:
            return None
        return str(name) if name is not None else None

    def observe_public(self) -> Mapping[str, Any]:
        return self.runtime.observe_public()

    def solve_ik(self, tcp_pose: Sequence[float], seed_joint_positions: Sequence[float]) -> Sequence[float] | None:
        target = np.asarray(tcp_pose, dtype=np.float64)
        seed = np.asarray(seed_joint_positions, dtype=np.float64)
        if target.shape != (7,) or seed.shape != (7,) or not np.isfinite(target).all() or not np.isfinite(seed).all():
            return None
        qpos_before = self.data.qpos.copy()
        qvel_before = self.data.qvel.copy()
        limits = np.asarray(self.joint_limits, dtype=np.float64)
        joints = np.clip(seed, limits[:, 0], limits[:, 1])
        target_position, target_quaternion = target[:3], target[3:]
        try:
            for _ in range(60):
                self.data.qpos[self.arm_qpos_ids] = joints
                self.mujoco.mj_forward(self.model, self.data)
                current = self._tcp_pose()
                position_error = target_position - np.asarray(current[:3])
                rotation_error = _rotation_vector(target_quaternion, current[3:])
                error = np.concatenate((position_error, rotation_error))
                if np.linalg.norm(position_error) <= 0.002 and np.linalg.norm(rotation_error) <= 0.025:
                    return joints.tolist()
                jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
                jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
                self.mujoco.mj_jacSite(
                    self.model, self.data, jacobian_position, jacobian_rotation, self.tcp_site_id,
                )
                jacobian = np.vstack((
                    jacobian_position[:, self.arm_dof_ids],
                    jacobian_rotation[:, self.arm_dof_ids],
                ))
                damping = 0.025
                try:
                    step = jacobian.T @ np.linalg.solve(
                        jacobian @ jacobian.T + damping * damping * np.eye(6), error,
                    )
                except np.linalg.LinAlgError:
                    return None
                step = np.clip(step, -0.12, 0.12)
                joints = np.clip(joints + step, limits[:, 0], limits[:, 1])
        finally:
            self.data.qpos[:] = qpos_before
            self.data.qvel[:] = qvel_before
            self.mujoco.mj_forward(self.model, self.data)
        return None

    def check_robot_sweep_collision(
        self,
        start_tcp_pose: Sequence[float],
        end_tcp_pose: Sequence[float],
        start_joint_positions: Sequence[float],
        end_joint_positions: Sequence[float],
        obstacle_points_world_m: Sequence[Sequence[float]],
        moving_geometry: Mapping[str, Any],
    ) -> bool:
        self.last_sweep_collision_diagnostic = None
        cloud = np.asarray(obstacle_points_world_m, dtype=np.float64)
        start = np.asarray(start_joint_positions, dtype=np.float64)
        end = np.asarray(end_joint_positions, dtype=np.float64)
        if (
            cloud.ndim != 2 or cloud.shape[1] != 3 or not np.isfinite(cloud).all()
            or start.shape != (7,) or end.shape != (7,)
            or not np.isfinite(start).all() or not np.isfinite(end).all()
        ):
            return False
        current = self.data.qpos[self.arm_qpos_ids].copy()
        if np.max(np.abs(current - start)) > 1e-3:
            return False
        delta = end - start
        sample_count = max(1, int(math.ceil(float(np.max(np.abs(delta))) / 0.02)))
        pairs = _validated_contact_pairs(moving_geometry.get("_allowed_contact_pairs"))
        if pairs is None:
            return False
        feature_labels = self.runtime._feature_instance_labels
        allowed_targets = {
            (str(pair["moving_role"]), feature_labels.get(str(pair["reference_feature_ref"])))
            for pair in pairs
        }
        allowed_targets.discard(("gripper", None))
        held_label = self._held_label
        qpos_before = self.data.qpos.copy()
        qvel_before = self.data.qvel.copy()
        escape_context: dict[str, Any] = {}
        try:
            for step_index in range(sample_count + 1):
                fraction = step_index / sample_count
                self.data.qpos[self.arm_qpos_ids] = start + fraction * delta
                self.mujoco.mj_forward(self.model, self.data)
                contact_evidence: dict[str, Any] = {}
                if self._has_unallowed_robot_contact(
                    allowed_targets, held_label, diagnostic=contact_evidence,
                ):
                    self.last_sweep_collision_diagnostic = {"kind": "mujoco_contact", "fraction": fraction}
                    self._record_sweep_collision_evidence({
                        "kind": "mujoco_contact",
                        "sample_fraction": float(fraction),
                        "held_label": held_label,
                        **contact_evidence,
                    })
                    return False
                tcp = _interpolate_pose(start_tcp_pose, end_tcp_pose, fraction)
                moving_pose = moving_geometry.get("_tcp_to_feature_pose")
                moving_position = (
                    _compose_pose(tcp, moving_pose)[0]
                    if isinstance(moving_pose, Sequence) and len(moving_pose) == 7
                    else None
                )
                def record_rgbd_robot(evidence: Mapping[str, Any]) -> None:
                    event = dict(evidence)
                    geom_id = event.get("geom_id")
                    if isinstance(geom_id, (int, np.integer)) and not isinstance(geom_id, bool):
                        event["geom_name"] = self._geom_name(int(geom_id))
                    obstacle_point = event.get("nearest_obstacle_point_m")
                    self.last_sweep_collision_diagnostic = {
                        "kind": "rgbd_observed_surface",
                        "sample_fraction": float(fraction),
                        "obstacle_point_world_m": obstacle_point,
                        "preexisting_overlap": bool(event.get("preexisting_overlap", False)),
                        "clearance_change_m": event.get("clearance_change_m"),
                    }
                    if event.get("separation_direction_world") is not None:
                        self.last_sweep_collision_diagnostic["separation_direction_world"] = event[
                            "separation_direction_world"
                        ]
                    self._record_sweep_collision_evidence({
                        "kind": "rgbd_robot_geometry",
                        "sample_fraction": float(fraction),
                        "held_label": held_label,
                        **event,
                    })

                if self._rgbd_sweep_collision(
                    cloud, pairs, moving_position, feature_labels,
                    diagnostic_callback=record_rgbd_robot,
                    sample_index=step_index,
                    sample_count=sample_count,
                    escape_context=escape_context,
                ):
                    if self.last_sweep_collision_diagnostic is None:
                        self.last_sweep_collision_diagnostic = {
                            "kind": "rgbd_robot_geometry", "fraction": fraction,
                        }
                    return False
        finally:
            self.data.qpos[:] = qpos_before
            self.data.qvel[:] = qvel_before
            self.mujoco.mj_forward(self.model, self.data)
        def record_rgbd_held(evidence: Mapping[str, Any]) -> None:
            self._record_sweep_collision_evidence({
                "kind": "rgbd_held_object_geometry",
                "held_label": held_label,
                "geom_id": None,
                "geom_name": None,
                **dict(evidence),
            })

        if moving_geometry.get("_held_object") is True and _held_object_sweep_collision(
            cloud, start_tcp_pose, end_tcp_pose, moving_geometry,
            pairs, feature_labels, self.sweep_margin_m,
            diagnostic_callback=record_rgbd_held,
        ):
            self.last_sweep_collision_diagnostic = {"kind": "rgbd_held_object_geometry"}
            return False
        return True

    def _has_unallowed_robot_contact(
        self,
        allowed_targets: set[tuple[str, int | None]],
        held_label: int | None,
        *,
        diagnostic: dict[str, Any] | None = None,
    ) -> bool:
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            robot1, robot2 = geom1 in self._robot_geom_ids, geom2 in self._robot_geom_ids
            if not robot1 and not robot2:
                continue
            label1, label2 = self._geom_labels.get(geom1), self._geom_labels.get(geom2)
            if robot1 and robot2:
                if diagnostic is not None:
                    diagnostic.update({
                        "geom1_id": geom1,
                        "geom1_name": self._geom_name(geom1),
                        "geom2_id": geom2,
                        "geom2_name": self._geom_name(geom2),
                        "contact_position_m": self._contact_position(contact),
                    })
                return True
            robot_geom, object_label = (geom1, label2) if robot1 else (geom2, label1)
            if robot_geom in self._static_robot_geom_ids:
                continue
            if object_label in self._robot_labels:
                continue
            if robot_geom in self._finger_geom_ids["left"] | self._finger_geom_ids["right"]:
                if held_label is not None and object_label == held_label:
                    continue
                if ("gripper", object_label) in allowed_targets:
                    continue
            if diagnostic is not None:
                diagnostic.update({
                    "geom1_id": geom1,
                    "geom1_name": self._geom_name(geom1),
                    "geom2_id": geom2,
                    "geom2_name": self._geom_name(geom2),
                    "contact_position_m": self._contact_position(contact),
                })
            return True
        return False

    @staticmethod
    def _contact_position(contact: Any) -> list[float] | None:
        try:
            position = np.asarray(contact.pos, dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if position.shape != (3,) or not np.isfinite(position).all():
            return None
        return position.tolist()

    def _rgbd_geometry_clearance_direction_world(
        self, geom_id: int, obstacle_point_world_m: Sequence[float],
    ) -> list[float] | None:
        """Return a translation direction that increases this geom's clearance."""
        point = np.asarray(obstacle_point_world_m, dtype=np.float64)
        position = np.asarray(self.data.geom_xpos[geom_id], dtype=np.float64)
        rotation = np.asarray(self.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        geom_type = int(self.model.geom_type[geom_id])
        size = np.asarray(self.model.geom_size[geom_id], dtype=np.float64)

        if geom_type == int(self.mujoco.mjtGeom.mjGEOM_BOX):
            local = (point - position) @ rotation
            axis = int(np.argmax(np.abs(local) - size))
            local_direction = np.zeros(3, dtype=np.float64)
            local_direction[axis] = -1.0 if local[axis] >= 0.0 else 1.0
            direction = rotation @ local_direction
        elif geom_type == int(self.mujoco.mjtGeom.mjGEOM_MESH):
            bounds = self._mesh_local_bounds.get(int(geom_id))
            if bounds is None:
                return None
            local = (point - position) @ rotation
            # Match the conservative AABB clearance used by the sweep, including points
            # already inside the bounds; clipping those points would yield a zero vector.
            outside = np.maximum(bounds[0] - local, local - bounds[1])
            axis = int(np.argmax(outside))
            midpoint = (bounds[0][axis] + bounds[1][axis]) * 0.5
            local_direction = np.zeros(3, dtype=np.float64)
            local_direction[axis] = -1.0 if local[axis] >= midpoint else 1.0
            direction = rotation @ local_direction
        elif geom_type in {
            int(self.mujoco.mjtGeom.mjGEOM_CAPSULE),
            int(self.mujoco.mjtGeom.mjGEOM_CYLINDER),
        }:
            axis = rotation[:, 2]
            half_length = float(size[1])
            segment_start = position - axis * half_length
            segment = axis * (2.0 * half_length)
            fraction = np.clip(
                float(np.dot(point - segment_start, segment))
                / max(float(np.dot(segment, segment)), 1e-12),
                0.0,
                1.0,
            )
            closest = segment_start + fraction * segment
            direction = closest - point
        else:
            direction = position - point

        norm = float(np.linalg.norm(direction))
        if not np.isfinite(direction).all() or not math.isfinite(norm) or norm <= 1e-9:
            return None
        return (direction / norm).tolist()

    def _rgbd_sweep_collision(
        self,
        cloud: np.ndarray,
        pairs: Sequence[Mapping[str, Any]],
        moving_position: np.ndarray | None,
        feature_labels: Mapping[str, int],
        *,
        diagnostic_callback: Callable[[Mapping[str, Any]], None] | None = None,
        sample_index: int = 0,
        sample_count: int = 0,
        escape_context: dict[str, Any] | None = None,
    ) -> bool:
        if not len(cloud):
            return True
        margin = self.sweep_margin_m
        points = cloud
        context = escape_context if escape_context is not None else {}
        initial_overlaps = context.setdefault("initial_overlaps", {})
        previous_clearance = context.setdefault("previous_clearance", {})
        context.setdefault("escape_improved", False)
        for geom_id in self._moving_robot_geom_ids:
            geom_type = int(self.model.geom_type[geom_id])
            position = self.data.geom_xpos[geom_id]
            rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
            size = self.model.geom_size[geom_id]
            if geom_type == int(self.mujoco.mjtGeom.mjGEOM_PLANE):
                continue
            if geom_type == int(self.mujoco.mjtGeom.mjGEOM_BOX):
                local = (points - position) @ rotation
                clearance = np.max(np.abs(local) - size, axis=1) - margin
            elif geom_type == int(self.mujoco.mjtGeom.mjGEOM_MESH):
                bounds = self._mesh_local_bounds.get(int(geom_id))
                if bounds is None:
                    return True
                local = (points - position) @ rotation
                outside = np.maximum(bounds[0] - local, local - bounds[1])
                clearance = np.max(outside, axis=1) - margin
            elif geom_type in {
                int(self.mujoco.mjtGeom.mjGEOM_CAPSULE),
                int(self.mujoco.mjtGeom.mjGEOM_CYLINDER),
            }:
                axis = rotation[:, 2]
                half_length = float(size[1])
                start = position - axis * half_length
                segment = axis * (2.0 * half_length)
                relative = points - start
                fraction = np.clip((relative @ segment) / max(float(segment @ segment), 1e-12), 0.0, 1.0)
                closest = start + fraction[:, None] * segment
                clearance = np.linalg.norm(points - closest, axis=1) - float(size[0]) - margin
            else:
                radius = float(self.model.geom_rbound[geom_id])
                clearance = np.linalg.norm(points - position, axis=1) - radius - margin
            collision = clearance <= 0.0
            candidates = np.flatnonzero(collision)
            if not len(candidates):
                candidates = np.asarray([], dtype=np.int64)
            if len(candidates) and moving_position is not None:
                for pair in pairs:
                    if (
                        pair["moving_role"] == "gripper"
                        and feature_labels.get(str(pair["reference_feature_ref"])) is not None
                        and _moving_feature_at_contact(moving_position, pair)
                    ):
                        collision[candidates] &= ~_contact_surface_point_mask(points[candidates], pair)

            if sample_index == 0:
                initial_indices = np.flatnonzero(collision)
                if len(initial_indices):
                    initial_overlaps[int(geom_id)] = (initial_indices, clearance[initial_indices].copy())
                    previous_clearance[int(geom_id)] = clearance[initial_indices].copy()
                continue

            overlap = initial_overlaps.get(int(geom_id))
            if overlap is not None:
                initial_indices, initial_clearance = overlap
                previous = previous_clearance[int(geom_id)]
                changes = clearance[initial_indices] - previous
                worsening = changes < -1e-9
                if np.any(worsening):
                    offset = int(np.flatnonzero(worsening)[0])
                    index = int(initial_indices[offset])
                    _emit_sweep_collision_diagnostic(
                        diagnostic_callback,
                        {
                            "geom_id": int(geom_id),
                            "nearest_obstacle_point_m": points[index].tolist(),
                            "nearest_obstacle_distance_m": float(np.linalg.norm(points[index] - position)),
                            "separation_direction_world": self._rgbd_geometry_clearance_direction_world(
                                int(geom_id), points[index],
                            ),
                            "preexisting_overlap": True,
                            "clearance_change_m": float(changes[offset]),
                        },
                    )
                    return True
                collision[initial_indices] = False
                if np.any(clearance[initial_indices] > initial_clearance + 1e-9):
                    context["escape_improved"] = True
                previous_clearance[int(geom_id)] = clearance[initial_indices].copy()
            if np.any(collision):
                blocking_indices = np.flatnonzero(collision)
                nearest = int(blocking_indices[np.argmin(
                    np.linalg.norm(points[blocking_indices] - position, axis=1),
                )])
                _emit_sweep_collision_diagnostic(
                    diagnostic_callback,
                    {
                        "geom_id": int(geom_id),
                        "nearest_obstacle_point_m": points[nearest].tolist(),
                        "nearest_obstacle_distance_m": float(
                            np.linalg.norm(points[nearest] - position),
                        ),
                        "separation_direction_world": self._rgbd_geometry_clearance_direction_world(
                            int(geom_id), points[nearest],
                        ),
                        "preexisting_overlap": False,
                        "clearance_change_m": None,
                    },
                )
                return True
        if sample_index == sample_count and initial_overlaps and not context["escape_improved"]:
            geom_id = next(iter(initial_overlaps))
            initial_indices, initial_values = initial_overlaps[geom_id]
            point_index = int(initial_indices[0])
            current_values = previous_clearance[geom_id]
            clearance_change = float(current_values[0] - initial_values[0])
            _emit_sweep_collision_diagnostic(
                diagnostic_callback,
                {
                    "geom_id": geom_id,
                    "nearest_obstacle_point_m": points[point_index].tolist(),
                    "nearest_obstacle_distance_m": float(
                        np.linalg.norm(points[point_index] - self.data.geom_xpos[geom_id]),
                    ),
                    "separation_direction_world": self._rgbd_geometry_clearance_direction_world(
                        int(geom_id), points[point_index],
                    ),
                    "preexisting_overlap": True,
                    "clearance_change_m": float(clearance_change),
                },
            )
            return True
        return False

    def step(self, normalized_tcp_delta: Sequence[float]) -> Mapping[str, Any]:
        action = np.asarray(normalized_tcp_delta, dtype=np.float64)
        if action.shape != (6,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.0 + 1e-8):
            raise ValueError("normalized TCP action must be six finite values in [-1, 1]")
        if self.runtime.official_physics_steps >= self.runtime.max_episode_steps:
            raise RuntimeError(
                f"official episode step limit ({self.runtime.max_episode_steps}) reached"
            )
        position_delta_world = action[:3] * 0.05
        rotation_delta_world = action[3:] * 0.5
        base_rotation = self.data.xmat[self.base_body_id].reshape(3, 3)
        action_in_base = np.concatenate((
            base_rotation.T @ position_delta_world / 0.05,
            base_rotation.T @ rotation_delta_world / 0.5,
            (self._gripper_command,),
        ))
        self.runtime._last_obs, _, done, _ = self.env.step(action_in_base)
        self.runtime._record_official_evaluator_step(done=bool(done))
        self._commanded_tcp_pose = self._tcp_pose()
        if max(self._contact_state()[1], default=0.0) > self._gripper_force_limit_n:
            self._gripper_command = 0.0
        return {"observation": self.runtime.observe_public()}

    def set_gripper_target_width(self, width_m: float, max_force_n: float) -> None:
        if not math.isfinite(float(width_m)) or not 0.0 <= float(width_m) <= self.max_gripper_width_m:
            raise ValueError("requested gripper width is outside calibrated Panda limits")
        self._gripper_force_limit_n = min(3.0, float(max_force_n))
        self._gripper_command = -1.0 if width_m >= self.max_gripper_width_m - 0.002 else 1.0

    def hold(self) -> Mapping[str, Any]:
        self._commanded_tcp_pose = self._tcp_pose()
        return self.step((0.0,) * 6)

    def _resolve_gripper_joint_ids(self) -> tuple[int, ...]:
        joint_names = getattr(self.robot.gripper, "joints", ())
        if isinstance(joint_names, str):
            joint_names = (joint_names,)
        joint_ids = []
        for raw_name in joint_names:
            name = str(raw_name)
            candidates = (name,) if name.startswith("robot0_") else (name, f"robot0_{name}")
            joint_id = next((
                candidate
                for candidate in (
                    self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, item)
                    for item in candidates
                )
                if candidate >= 0
            ), -1)
            if joint_id >= 0:
                joint_ids.append(int(joint_id))
        if not joint_ids or len(set(joint_ids)) != len(joint_ids):
            raise RuntimeError("Panda gripper finger joints cannot be resolved")
        return tuple(joint_ids)

    def _read_max_gripper_width(self) -> float:
        joint_ids = self._resolve_gripper_joint_ids()
        widths = [float(np.max(np.abs(self.model.jnt_range[joint_id]))) for joint_id in joint_ids]
        result = float(sum(widths))
        if not widths or not math.isfinite(result) or result <= 0.0:
            raise RuntimeError("Panda gripper opening width cannot be calibrated from its joint limits")
        return result

    def _calibrate_measured_gripper_width(self) -> float:
        """Measure the inner pad-face aperture at the legal maximum finger endpoints."""
        joint_ids = self._resolve_gripper_joint_ids()
        ranges = np.asarray(self.model.jnt_range[list(joint_ids)], dtype=np.float64)
        qpos_ids = np.asarray(self.model.jnt_qposadr[list(joint_ids)], dtype=int)
        qvel_ids = np.asarray(self.model.jnt_dofadr[list(joint_ids)], dtype=int)
        if (
            ranges.shape != (len(joint_ids), 2)
            or qpos_ids.shape != (len(joint_ids),)
            or qvel_ids.shape != (len(joint_ids),)
            or not np.isfinite(ranges).all()
            or np.any(ranges[:, 0] > ranges[:, 1])
            or np.any(qpos_ids < 0)
            or np.any(qpos_ids >= self.data.qpos.size)
            or np.any(qvel_ids < 0)
            or np.any(qvel_ids >= self.data.qvel.size)
        ):
            raise RuntimeError("Panda gripper finger joint limits or state addresses are invalid")

        qpos_before = self.data.qpos.copy()
        qvel_before = self.data.qvel.copy()
        try:
            endpoints = np.where(
                np.abs(ranges[:, 0]) >= np.abs(ranges[:, 1]),
                ranges[:, 0],
                ranges[:, 1],
            )
            self.data.qpos[qpos_ids] = endpoints
            self.data.qvel[qvel_ids] = 0.0
            self.mujoco.mj_forward(self.model, self.data)
            width, _ = self._gripper_geometry()
            if not math.isfinite(width) or width <= 0.0:
                raise RuntimeError("Panda measured gripper opening width is invalid")
            return float(width)
        finally:
            self.data.qpos[:] = qpos_before
            self.data.qvel[:] = qvel_before
            self.mujoco.mj_forward(self.model, self.data)


class LiberoAssemblyRuntime:
    video_fps = float(CONTROL_HZ)

    def __init__(
        self,
        env: Any,
        initial_observation: Mapping[str, Any],
        wrench_baseline: Sequence[float],
        output_dir: str | Path,
        *,
        max_episode_steps: int = 1000,
    ) -> None:
        if isinstance(max_episode_steps, bool) or not isinstance(max_episode_steps, int):
            raise ValueError("max_episode_steps must be an integer")
        if not 1 <= max_episode_steps <= 1000:
            raise ValueError("max_episode_steps must be in [1, official LIBERO horizon 1000]")
        self.env = env
        self.max_episode_steps = max_episode_steps
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.frame_dir = self.output_dir / "runtime_frames"
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.private_dir = self.output_dir / "private"
        self.private_dir.mkdir(parents=True, exist_ok=True)
        self._sweep_collision_log_path = self.private_dir / "libero_sweep_collisions.jsonl"
        self._sweep_collision_record_count = 0
        self._last_public_observation_id: str | None = None
        self._last_obs = initial_observation
        self._listeners: list[Callable[[np.ndarray, float], None]] = []
        self._history: deque[dict[str, Any]] = deque(maxlen=FRAME_HISTORY)
        self._frame_id = 0
        self._observation_id = 0
        self._last_frame: PublicRGBDFrame | None = None
        self._last_segmentation: np.ndarray | None = None
        self._feature_instance_labels: dict[str, int] = {}
        self._last_model_timestamp_s: float | None = None
        self._step_context = "runtime"
        self.official_physics_steps = 0
        self.official_harness_action_steps = 0
        self._official_success_ever = False
        self._official_success_after_harness_action = False
        self._official_success_final = False
        self._official_evaluator_trace_path = self.private_dir / "official_evaluator_trace.jsonl"
        self.driver = LiberoMujocoDriver(self, wrench_baseline)
        self.executor = AssemblyExecutor(
            self.driver,
            limits=ExecutorLimits(
                control_dt_s=1.0 / CONTROL_HZ,
                position_action_scale_m=(0.05, 0.05, 0.05),
                rotation_action_scale_rad=(0.5, 0.5, 0.5),
                max_contact_force_n=3.0,
                max_gripper_force_n=3.0,
                max_goal_uncertainty_m=0.02,
                max_goal_axis_uncertainty_rad=0.35,
                max_rgbd_age_s=0.2,
                max_steps=200,
                max_linear_speed_mps=0.03,
                max_contact_speed_mps=0.004,
                max_angular_speed_radps=0.35,
                max_joint_step_rad=0.05,
            ),
            observation_provider=self.observe_public,
        )
        self._record_official_evaluator_step(event="initial")

    @property
    def official_success_ever(self) -> bool:
        return self._official_success_ever

    @property
    def official_success_after_harness_action(self) -> bool:
        return self._official_success_after_harness_action

    def _record_official_evaluator_step(
        self,
        done: bool = False,
        *,
        event: str = "physics_step",
    ) -> None:
        if event == "physics_step":
            self.official_physics_steps += 1
            if self._step_context == "harness_action":
                self.official_harness_action_steps += 1
        success = bool(self.env.check_success())
        became_success = success and not self._official_success_final
        self._official_success_ever = self._official_success_ever or success
        if became_success and self._step_context == "harness_action":
            self._official_success_after_harness_action = True
        self._official_success_final = success
        record = {
            "event": event,
            "physics_step": self.official_physics_steps,
            "step_context": self._step_context,
            "sim_time_s": float(self.env.sim.data.time),
            "env_done": bool(done),
            "official_success": success,
        }
        with self._official_evaluator_trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")

    def official_evaluator_summary(self) -> dict[str, Any]:
        self._official_success_final = bool(self.env.check_success())
        return {
            "checker": "LIBERO env.check_success()",
            "official_success_ever": self._official_success_ever,
            "official_success_after_harness_action": self._official_success_after_harness_action,
            "official_success_final": self._official_success_final,
            "num_physics_steps": self.official_physics_steps,
            "num_harness_action_steps": self.official_harness_action_steps,
            "max_steps": self.max_episode_steps,
            "trace_file": str(self._official_evaluator_trace_path.name),
            "actor_input": False,
        }

    def _record_sweep_collision(self, evidence: Mapping[str, Any]) -> None:
        """Append bounded, private-only sweep evidence without changing control flow."""
        if self._sweep_collision_record_count >= SWEEP_COLLISION_RECORD_LIMIT:
            return
        if not isinstance(evidence, Mapping):
            return

        def finite_float(value: Any) -> float | None:
            try:
                result = float(value)
            except (TypeError, ValueError):
                return None
            return result if math.isfinite(result) else None

        def optional_int(value: Any) -> int | None:
            if isinstance(value, bool) or value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        def finite_vector(value: Any) -> list[float] | None:
            try:
                vector = np.asarray(value, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                return None
            if vector.shape != (3,) or not np.isfinite(vector).all():
                return None
            return vector.tolist()

        observation_id = self._last_public_observation_id
        if observation_id is not None and not isinstance(observation_id, (str, int, float, bool)):
            observation_id = str(observation_id)
        driver = getattr(self, "driver", None)
        camera_evidence = _project_world_point_to_frame(
            evidence.get("nearest_obstacle_point_m"),
            getattr(self, "_last_frame", None),
            getattr(self, "_last_segmentation", None),
            set(getattr(driver, "_robot_labels", set())),
        )
        record = {
            "schema": "libero_sweep_collision_v1",
            "record_index": self._sweep_collision_record_count,
            "timestamp_s": finite_float(getattr(self.env.sim.data, "time", None)),
            "observation_id": observation_id,
            "kind": str(evidence.get("kind", "unknown")),
            "sample_fraction": finite_float(evidence.get("sample_fraction")),
            "held_label": optional_int(evidence.get("held_label")),
            "geom_id": optional_int(evidence.get("geom_id")),
            "geom_name": evidence.get("geom_name"),
            "geom1_id": optional_int(evidence.get("geom1_id")),
            "geom1_name": evidence.get("geom1_name"),
            "geom2_id": optional_int(evidence.get("geom2_id")),
            "geom2_name": evidence.get("geom2_name"),
            "contact_position_m": finite_vector(evidence.get("contact_position_m")),
            "nearest_obstacle_point_m": finite_vector(evidence.get("nearest_obstacle_point_m")),
            "nearest_obstacle_distance_m": finite_float(
                evidence.get("nearest_obstacle_distance_m"),
            ),
            "nearest_obstacle_camera": camera_evidence,
        }
        for key in ("geom_name", "geom1_name", "geom2_name"):
            if record[key] is not None and not isinstance(record[key], str):
                record[key] = str(record[key])
        try:
            line = json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
            with self._sweep_collision_log_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        except Exception:
            return
        self._sweep_collision_record_count += 1

    def _capture_frame(self) -> tuple[PublicRGBDFrame, np.ndarray]:
        observation = self._last_obs
        timestamp = float(self.env.sim.data.time)
        if self._last_frame is not None and timestamp == self._last_frame.timestamp_s:
            return self._last_frame, self._last_segmentation
        from PIL import Image
        from robosuite.utils.camera_utils import (
            get_camera_extrinsic_matrix,
            get_camera_intrinsic_matrix,
            get_real_depth_map,
        )

        rgb = np.asarray(observation["agentview_image"])
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise RuntimeError("LIBERO agentview RGB must be an HxWx3 image")
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        depth_normalized = np.asarray(observation["agentview_depth"], dtype=np.float32).squeeze()
        segmentation = np.asarray(observation["agentview_segmentation_instance"]).squeeze()
        robot_labels = self.driver._robot_labels
        if not robot_labels:
            raise RuntimeError("LIBERO Panda instance segmentation labels are unavailable")
        depth = _mask_robot_depth(get_real_depth_map(self.env.sim, depth_normalized), segmentation, robot_labels)
        intrinsics = get_camera_intrinsic_matrix(
            self.env.sim, "agentview", rgb.shape[0], rgb.shape[1],
        )
        extrinsics = get_camera_extrinsic_matrix(self.env.sim, "agentview")
        frame = PublicRGBDFrame(
            self._frame_id, timestamp, rgb, depth, intrinsics, extrinsics,
            pixel_convention="top_left",
        )
        rgb_path = self.frame_dir / f"rgb_{frame.frame_id:08d}.png"
        depth_path = self.frame_dir / f"depth_{frame.frame_id:08d}.npy"
        Image.fromarray(frame.rgb).save(rgb_path)
        np.save(depth_path, frame.depth_m, allow_pickle=False)
        self._history.append({
            "frame_id": frame.frame_id,
            "timestamp_s": frame.timestamp_s,
            "rgb_path": str(rgb_path.resolve()),
            "depth_path": str(depth_path.resolve()),
        })
        self._frame_id += 1
        self._last_frame = frame
        self._last_segmentation = segmentation.astype(np.int32, copy=True)
        for listener in tuple(self._listeners):
            listener(frame.rgb, frame.timestamp_s)
        return frame, self._last_segmentation

    def observe_public(self) -> Mapping[str, Any]:
        frame, segmentation = self._capture_frame()
        if frame is None or segmentation is None:
            raise RuntimeError("LIBERO did not produce a synchronized RGB-D segmentation frame")
        observation_id = f"libero-observation-{self._observation_id:08d}"
        self._observation_id += 1
        self._last_public_observation_id = observation_id
        features = estimate_camera_features(
            frame, observation_id=observation_id,
            include_planes=False, include_contours=True,
        )
        self._feature_instance_labels = _feature_instance_labels(
            features, segmentation, self.driver._robot_labels,
        )
        contacts, forces, held_label, support_contact = self.driver._contact_state()
        if held_label is not None:
            self.driver._held_label = held_label
        elif not any(contacts):
            self.driver._held_label = None
        # Match the feature extractor's pixel lattice so contact evidence survives cloud sampling.
        cloud_data = point_cloud_world(
            frame, stride=LIBERO_RGBD_OBSTACLE_STRIDE, max_points=16_384,
        )
        cloud = _exclude_instance_points(
            cloud_data["points_world_m"], cloud_data["pixel_xy"], segmentation,
            {self.driver._held_label} if self.driver._held_label is not None else set(),
        )
        if len(cloud) < 16:
            raise RuntimeError("LIBERO public RGB-D produced too few robot-filtered obstacle samples")
        timestamp = float(self.env.sim.data.time)
        if timestamp > (self.driver._support_timestamp_s if self.driver._support_timestamp_s is not None else -math.inf):
            self.driver._support_count = self.driver._support_count + 1 if support_contact else 0
            self.driver._support_timestamp_s = timestamp
        tcp_pose = self.driver._tcp_pose()
        gripper_width, gripper_axis = self.driver._gripper_geometry()
        wrench = self.driver._wrench()
        obstacle_map = {
            "source_observation_id": observation_id,
            "frame": "world",
            "valid": True,
            "robot_self_filtered": True,
            "timestamp_s": timestamp,
            "points_m": cloud,
        }
        sensors = {
            "tcp_pose": list(tcp_pose),
            "commanded_tcp_pose": list(self.driver._commanded_tcp_pose),
            "tcp_velocity": self.driver._tcp_velocity().tolist(),
            "joint_positions": self.env.sim.data.qpos[self.driver.arm_qpos_ids].astype(float).tolist(),
            "joint_velocities": self.env.sim.data.qvel[self.driver.arm_dof_ids].astype(float).tolist(),
            "gripper_width_m": gripper_width,
            "finger_contacts": list(contacts),
            "finger_contact_force_n": list(forces),
            "grasped": bool(all(contacts)),
            "wrench": wrench.tolist(),
            "release_authorized": bool(all(contacts) and self.driver._support_count >= 2),
            "rgbd_obstacle_map": obstacle_map,
            "robot_timestamp_s": timestamp,
            "rgbd_source_timestamp_s": frame.timestamp_s,
            "camera_frame_id": frame.frame_id,
            "camera_observation_id": observation_id,
            "rgbd_source_observation_id": observation_id,
        }
        state = {**sensors, "sensors": sensors, "gripper_axis_world": gripper_axis.tolist()}
        public = build_public_observation(
            frame,
            observation_id=observation_id,
            rgb_path=self._history[-1]["rgb_path"],
            depth_path=self._history[-1]["depth_path"],
            robot_state=state,
            include_planes=False,
            camera_features=features,
            include_feature_overlay=True,
        )
        public["frames"], public["frame_sequence"] = self._model_frame_history(public)
        self._enrich_derived_grasp_widths(public)
        return public

    def _enrich_derived_grasp_widths(self, observation: Mapping[str, Any]) -> None:
        """Add only current-observation RGB-D-supported grasp apertures."""
        features = observation.get("features")
        if not isinstance(features, list):
            return

        feature_geometries: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
        for feature in features:
            if not isinstance(feature, Mapping):
                continue
            geometry = feature.get("geometry")
            if not isinstance(geometry, dict):
                continue
            feature_geometries.append((feature, geometry))

        robot_state = observation.get("robot_state")
        sensors = robot_state.get("sensors") if isinstance(robot_state, Mapping) else None
        obstacle_map = sensors.get("rgbd_obstacle_map") if isinstance(sensors, Mapping) else None
        opening_axis = robot_state.get("gripper_axis_world") if isinstance(robot_state, Mapping) else None
        if not (
            isinstance(robot_state, Mapping)
            and isinstance(sensors, Mapping)
            and isinstance(obstacle_map, Mapping)
            and obstacle_map.get("valid") is True
            and obstacle_map.get("robot_self_filtered") is True
            and obstacle_map.get("source_observation_id") == observation.get("observation_id")
        ):
            return

        for feature, geometry in feature_geometries:
            width = _supported_rgbd_grasp_width(
                feature,
                opening_axis,
                obstacle_map.get("points_m"),
                self.driver.measured_gripper_width_max_m,
            )
            if width is not None:
                geometry["grasp_width_m"] = width

    def _model_frame_history(self, observation: Mapping[str, Any]) -> tuple[dict[str, str], list[dict[str, Any]]]:
        current_overlay = observation.get("frames", {}).get("task_camera_feature_overlay")
        frames: dict[str, str] = {}
        sequence: list[dict[str, Any]] = []
        history = tuple(self._history)
        for index, entry in enumerate(history[:-1]):
            key = f"task_camera_history_{index:02d}"
            frames[key] = entry["rgb_path"]
            sequence.append({"frame_key": key, "frame_id": entry["frame_id"], "timestamp_s": entry["timestamp_s"]})
        latest = history[-1]
        key = "task_camera_current_feature_overlay" if isinstance(current_overlay, str) else "task_camera_current"
        frames[key] = current_overlay if isinstance(current_overlay, str) else latest["rgb_path"]
        sequence.append({"frame_key": key, "frame_id": latest["frame_id"], "timestamp_s": latest["timestamp_s"]})
        return frames, sequence

    def observe_for_model(self) -> Mapping[str, Any]:
        observation = self.observe_public()
        while (
            len(self._history) < FRAME_HISTORY
            or (
                self._last_model_timestamp_s is not None
                and self._history[-1]["timestamp_s"] <= self._last_model_timestamp_s
            )
        ):
            previous_held_label = self.driver._held_label
            previous_width_m = self.driver._gripper_geometry()[0]
            result = self.driver.hold()
            contacts, finger_forces, current_held_label, _ = self.driver._contact_state()
            current_width_m = self.driver._gripper_geometry()[0]
            _validate_stationary_hold(
                previous_held_label=previous_held_label,
                current_held_label=current_held_label if all(contacts) else None,
                previous_width_m=previous_width_m,
                current_width_m=current_width_m,
                wrench=self.driver._wrench(),
                finger_forces_n=finger_forces,
                max_force_n=self.executor.limits.max_contact_force_n,
                width_tolerance_m=self.executor.limits.gripper_stability_tolerance_m,
            )
            if not isinstance(result, Mapping) or not isinstance(result.get("observation"), Mapping):
                raise RuntimeError("stationary camera hold did not return its current observation")
        observation = self.observe_public()
        if len(observation.get("frame_sequence", ())) != FRAME_HISTORY:
            raise RuntimeError("LIBERO observation did not retain eight distinct timestamped frames")
        current_timestamp = float(observation["frame_sequence"][-1]["timestamp_s"])
        if self._last_model_timestamp_s is not None and current_timestamp <= self._last_model_timestamp_s:
            raise RuntimeError("LIBERO model observation is not newer than its preceding request")
        self._last_model_timestamp_s = current_timestamp
        return observation

    def execute_goal(self, goal: Mapping[str, Any], observation: Mapping[str, Any]) -> Mapping[str, Any]:
        self._step_context = "harness_action"
        try:
            return self.executor.execute_goal(goal, observation)
        finally:
            self._step_context = "runtime"

    def execute_tcp_delta(self, call: Mapping[str, Any], observation: Mapping[str, Any]) -> Mapping[str, Any]:
        self._step_context = "harness_action"
        try:
            return self.executor.execute_tcp_delta(call, observation)
        finally:
            self._step_context = "runtime"

    def check_task(self) -> bool:
        return bool(self.env.check_success())

    def subscribe_rgb_frames(self, listener: Callable[[np.ndarray, float], None]) -> Callable[[], None]:
        if not callable(listener):
            raise TypeError("RGB frame listener must be callable")
        self._listeners.append(listener)
        self._capture_frame()

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def close(self) -> None:
        self.env.close()


def _interpolate_pose(start: Sequence[float], end: Sequence[float], fraction: float) -> tuple[float, ...]:
    position = (1.0 - fraction) * np.asarray(start[:3]) + fraction * np.asarray(end[:3])
    rotation = _rotation_vector(end[3:], start[3:]) * fraction
    angle = float(np.linalg.norm(rotation))
    delta = (
        np.asarray((1.0, 0.0, 0.0, 0.0)) if angle <= 1e-12 else
        np.concatenate(((math.cos(angle / 2.0),), rotation * math.sin(angle / 2.0) / angle))
    )
    quaternion = _quat_multiply(delta, start[3:])
    return (*position.tolist(), *quaternion.tolist())


def _load_single_task(
    args: Any,
) -> tuple[Any, Mapping[str, Any], np.ndarray, dict[str, Any]]:
    task_config = _task_config_from_args(args)
    if getattr(args, "task_config", None):
        task_goal = getattr(args, "task_goal", "")
        if _normalized_instruction(task_goal) != _normalized_instruction(task_config.task_instruction):
            raise ValueError(
                "--task-goal must match the instruction in --task-config for this LIBERO task"
            )
    source_root = Path(os.environ.get(
        "HAC_LIBERO_REPO_ROOT", Path.home() / ".cache" / "hac-libero-upstream",
    )).expanduser().resolve()
    if task_config.upstream_commit:
        try:
            source_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=source_root,
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError("could not verify the pinned official LIBERO checkout") from exc
        if source_commit != task_config.upstream_commit:
            raise RuntimeError(
                f"official LIBERO checkout is {source_commit}, expected {task_config.upstream_commit}"
            )
    benchmark_root = source_root / "libero" / "libero"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    config_dir = Path(args.output_dir).expanduser().resolve() / "private" / "libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = config_dir / "unused_datasets"
    dataset_path.mkdir(exist_ok=True)
    config = config_dir / "config.yaml"
    config.write_text(
        "\n".join((
            f"benchmark_root: {benchmark_root}",
            f"bddl_files: {benchmark_root / 'bddl_files'}",
            f"init_states: {benchmark_root / 'init_files'}",
            f"datasets: {dataset_path}",
            f"assets: {benchmark_root / 'assets'}",
            "",
        )),
        encoding="utf-8",
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import torch
    import robosuite.macros

    robosuite.macros.IMAGE_CONVENTION = "opencv"
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import SegmentationRenderEnv

    benchmark_type = benchmark.get_benchmark_dict().get(task_config.task_suite.lower())
    if benchmark_type is None:
        raise ValueError(f"unknown official LIBERO suite {task_config.task_suite!r}")
    task_suite = benchmark_type(task_order_index=task_config.task_order_index)
    if not 0 <= task_config.task_id < task_suite.get_num_tasks():
        raise ValueError(
            f"task_id {task_config.task_id} is outside suite {task_config.task_suite!r}"
        )
    task = task_suite.get_task(task_config.task_id)
    if task.name != task_config.task_name:
        raise ValueError(
            f"official task order mismatch: task_id {task_config.task_id} resolved to "
            f"{task.name!r}, expected {task_config.task_name!r}"
        )
    if _normalized_instruction(task.language) != _normalized_instruction(task_config.task_instruction):
        raise ValueError(
            f"official task language {task.language!r} does not match configured instruction"
        )
    if task.bddl_file != task_config.bddl_file or task.init_states_file != task_config.init_state_file:
        raise ValueError("official task BDDL/init filenames do not match --task-config")
    bddl_path = Path(task_suite.get_task_bddl_file_path(task_config.task_id)).resolve()
    init_path = (
        Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    ).resolve()
    if not bddl_path.is_file() or not init_path.is_file():
        raise FileNotFoundError(
            f"official LIBERO task files are missing: BDDL={bddl_path}, init={init_path}"
        )
    bddl_text = bddl_path.read_text(encoding="utf-8")
    if task_config.goal_expression and task_config.goal_expression not in bddl_text:
        raise ValueError("configured goal expression is absent from the official task BDDL")
    init_states = task_suite.get_task_init_states(task_config.task_id)
    if not 0 <= task_config.init_state_id < len(init_states):
        raise ValueError(
            f"init_state_id {task_config.init_state_id} is outside the official state set "
            f"of length {len(init_states)}"
        )
    device = str(getattr(args, "device", "cuda:0"))
    try:
        render_device_id = int(device.rsplit(":", 1)[-1]) if device.startswith("cuda:") else -1
    except ValueError:
        render_device_id = -1
    max_episode_steps = getattr(args, "max_steps", 1000)
    if isinstance(max_episode_steps, bool) or not isinstance(max_episode_steps, int):
        raise ValueError("--max-steps must be an integer")
    if not 1 <= max_episode_steps <= 1000:
        raise ValueError("--max-steps must be in [1, official LIBERO horizon 1000]")
    env = SegmentationRenderEnv(
        bddl_file_name=str(bddl_path),
        robots=["Panda"],
        controller="OSC_POSE",
        gripper_types="default",
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names=["agentview"],
        camera_heights=CAMERA_SIDE,
        camera_widths=CAMERA_SIDE,
        camera_depths=True,
        camera_segmentations="instance",
        render_gpu_device_id=render_device_id,
        control_freq=CONTROL_HZ,
        horizon=max_episode_steps,
        ignore_done=True,
        hard_reset=False,
    )
    try:
        env.seed(int(getattr(args, "seed", 0)))
        observation = env.reset()
        observation = env.set_init_state(_as_numpy(init_states[task_config.init_state_id]))
        if bool(env.check_success()):
            raise RuntimeError("selected official initial state already satisfies the task goal")
        robot = env.robots[0]
        baseline = _raw_wrench(observation, robot)
        task_metadata = {
            **asdict(task_config),
            "resolved_task_name": task.name,
            "resolved_task_language": task.language,
            "bddl_path": str(bddl_path),
            "init_state_path": str(init_path),
            "controller": "OSC_POSE",
            "robot": "Panda",
            "official_horizon_steps": max_episode_steps,
            "resolved_upstream_commit": source_commit if task_config.upstream_commit else None,
            "bddl_sha256": hashlib.sha256(bddl_path.read_bytes()).hexdigest(),
            "init_state_sha256": hashlib.sha256(init_path.read_bytes()).hexdigest(),
        }
        return env, observation, baseline, task_metadata
    except BaseException:
        env.close()
        raise


def create_runtime(args: Any) -> LiberoAssemblyRuntime:
    """Factory consumed by scripts/run_runtime_assembly_harness.py."""
    env, observation, baseline, task_metadata = _load_single_task(args)
    runtime = LiberoAssemblyRuntime(
        env,
        observation,
        baseline,
        args.output_dir,
        max_episode_steps=getattr(args, "max_steps", 1000),
    )
    runtime.task_metadata = task_metadata
    return runtime


__all__ = [
    "CONTROL_HZ",
    "LIBERO_RGBD_OBSTACLE_STRIDE",
    "LiberoTaskConfig",
    "TASK_BDDL",
    "TASK_GOAL",
    "TASK_INIT",
    "LiberoAssemblyRuntime",
    "LiberoMujocoDriver",
    "create_runtime",
    "load_libero_task_config",
]
