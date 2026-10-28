"""Task-agnostic public RGB-D projection and geometric feature estimates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PublicRGBDFrame:
    """One camera observation; depth is axial camera-z in meters."""

    frame_id: int
    timestamp_s: float
    rgb: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    T_world_camera: np.ndarray
    pixel_convention: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.frame_id, bool) or not isinstance(self.frame_id, (int, np.integer)) or self.frame_id < 0:
            raise ValueError("frame_id must be a non-negative integer")
        if not math.isfinite(float(self.timestamp_s)):
            raise ValueError("timestamp_s must be finite")
        if self.pixel_convention is not None and (
            not isinstance(self.pixel_convention, str) or not self.pixel_convention.strip()
        ):
            raise ValueError("pixel_convention must be a non-empty string or None")

        rgb = np.asarray(self.rgb)
        depth = np.asarray(self.depth_m, dtype=np.float32)
        intrinsics = np.asarray(self.K, dtype=np.float64)
        extrinsics = np.asarray(self.T_world_camera, dtype=np.float64)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("rgb must be uint8 with shape (height, width, 3)")
        if depth.shape != rgb.shape[:2]:
            raise ValueError("depth_m must have shape (height, width) matching rgb")
        if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
            raise ValueError("K must be a finite 3x3 camera intrinsic matrix")
        if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
            raise ValueError("K focal lengths must be positive")
        if extrinsics.shape != (4, 4) or not np.isfinite(extrinsics).all():
            raise ValueError("T_world_camera must be a finite 4x4 matrix")
        rotation = extrinsics[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
            raise ValueError("T_world_camera rotation must be orthonormal")
        if not np.allclose(extrinsics[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
            raise ValueError("T_world_camera must be a homogeneous transform")

        depth = depth.copy()
        depth[~np.isfinite(depth) | (depth <= 0.0)] = np.nan
        object.__setattr__(self, "rgb", rgb.copy())
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "K", intrinsics.copy())
        object.__setattr__(self, "T_world_camera", extrinsics.copy())
        object.__setattr__(self, "timestamp_s", float(self.timestamp_s))
        object.__setattr__(self, "frame_id", int(self.frame_id))


def _numpy(value: Any) -> np.ndarray:
    if callable(getattr(value, "detach", None)):
        value = value.detach()
    if callable(getattr(value, "cpu", None)):
        value = value.cpu()
    if callable(getattr(value, "numpy", None)):
        value = value.numpy()
    return np.asarray(value)


def _one_camera(value: Any, *, ndim: int, name: str) -> np.ndarray:
    array = _numpy(value)
    if array.ndim == ndim + 1:
        if array.shape[0] != 1:
            raise ValueError(f"{name} requires exactly one camera environment, got {array.shape[0]}")
        array = array[0]
    if array.ndim != ndim:
        raise ValueError(f"{name} has unexpected shape {array.shape}")
    return array


def _quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("quaternion must be a finite wxyz 4-vector")
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        raise ValueError("quaternion has zero norm")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert a right-handed rotation matrix to a normalized wxyz quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-4) or np.linalg.det(matrix) < 0.0:
        raise ValueError("rotation must be right-handed and orthonormal")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
             (matrix[0, 2] - matrix[2, 0]) / scale,
             (matrix[1, 0] - matrix[0, 1]) / scale]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                [(matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                 (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                [(matrix[0, 2] - matrix[2, 0]) / scale, (matrix[0, 1] + matrix[1, 0]) / scale,
                 0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.asarray(
                [(matrix[1, 0] - matrix[0, 1]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion


def public_rgbd_frame_from_camera(camera: Any) -> PublicRGBDFrame:
    """Copy RGB, axial-z depth, K, and camera pose from an Isaac camera sensor."""
    data, frame_index, timestamp_s = camera_capture_metadata(camera)
    output = data.output
    rgb_raw = output.get("rgb")
    depth_raw = output.get("distance_to_image_plane")
    if rgb_raw is None or depth_raw is None:
        raise RuntimeError("camera must publish rgb and distance_to_image_plane outputs")

    rgb = _one_camera(rgb_raw, ndim=3, name="rgb")
    if rgb.shape[-1] not in (3, 4):
        raise ValueError(f"rgb must have 3 or 4 channels, got {rgb.shape}")
    rgb = rgb[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        if not np.isfinite(rgb).all():
            raise ValueError("camera RGB contains non-finite values")
        if rgb.size and float(rgb.max()) <= 1.0:
            rgb = np.rint(rgb * 255.0)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8, copy=False)

    depth = _numpy(depth_raw)
    if depth.ndim == 4:
        if depth.shape[0] != 1:
            raise ValueError("depth requires exactly one camera environment")
        depth = depth[0]
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"distance_to_image_plane has unexpected shape {depth.shape}")

    native_intrinsics = _one_camera(data.intrinsic_matrices, ndim=2, name="intrinsic_matrices")
    if native_intrinsics.shape != (3, 3):
        raise ValueError("intrinsic_matrices must have shape (3, 3)")
    # Isaac's native projection uses pixel-edge coordinates; the public RGB-D
    # geometry APIs use integer array indices as pixel centers.
    intrinsics = native_intrinsics.astype(np.float64, copy=True)
    intrinsics[0, 2] -= 0.5
    intrinsics[1, 2] -= 0.5
    position = _one_camera(data.pos_w, ndim=1, name="pos_w").astype(np.float64, copy=False)
    quaternion = _one_camera(data.quat_w_ros, ndim=1, name="quat_w_ros")
    if position.shape != (3,):
        raise ValueError(f"camera position has unexpected shape {position.shape}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quat_wxyz_to_matrix(quaternion)
    transform[:3, 3] = position
    return PublicRGBDFrame(
        frame_id=frame_index,
        timestamp_s=timestamp_s,
        rgb=rgb,
        depth_m=depth,
        K=intrinsics,
        T_world_camera=transform,
        pixel_convention="integer_index_centers",
    )


def camera_capture_timestamp_s(camera: Any) -> float:
    """Read the camera sensor clock at its latest buffer update for one environment."""
    raw_timestamp = getattr(camera, "_timestamp_last_update", None)
    if raw_timestamp is None:
        raise RuntimeError("camera does not expose its last buffer-update timestamp")
    timestamps = _numpy(raw_timestamp).astype(np.float64, copy=False).reshape(-1)
    if timestamps.size != 1:
        raise RuntimeError("public RGB-D timestamp requires exactly one camera instance")
    timestamp_s = float(timestamps[0])
    if not math.isfinite(timestamp_s) or timestamp_s < 0.0:
        raise RuntimeError("camera capture timestamp must be finite and non-negative")
    return timestamp_s


def camera_capture_metadata(camera: Any) -> tuple[Any, int, float]:
    """Materialize lazy sensor data before reading the frame ID and source clock."""
    data = camera.data
    frame_ids = _numpy(camera.frame).reshape(-1)
    if frame_ids.size != 1:
        raise RuntimeError("public RGB-D metadata requires exactly one camera instance")
    return data, int(frame_ids[0]), camera_capture_timestamp_s(camera)


def point_cloud_world(
    frame: PublicRGBDFrame,
    *,
    stride: int = 4,
    max_points: int = 4096,
) -> dict[str, np.ndarray]:
    """Back-project axial-z depth into a bounded, deterministically sampled cloud."""
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    if isinstance(max_points, bool) or not isinstance(max_points, int) or max_points < 1:
        raise ValueError("max_points must be a positive integer")
    ys = np.arange(0, frame.depth_m.shape[0], stride, dtype=np.int32)
    xs = np.arange(0, frame.depth_m.shape[1], stride, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    z = frame.depth_m[grid_y, grid_x]
    valid = np.isfinite(z)
    u = grid_x[valid].astype(np.float64)
    v = grid_y[valid].astype(np.float64)
    z = z[valid].astype(np.float64)
    if z.size == 0:
        return {
            "points_world_m": np.empty((0, 3), dtype=np.float64),
            "colors_rgb": np.empty((0, 3), dtype=np.uint8),
            "pixel_xy": np.empty((0, 2), dtype=np.int32),
        }
    fx, fy = frame.K[0, 0], frame.K[1, 1]
    cx, cy = frame.K[0, 2], frame.K[1, 2]
    points_camera = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
    points_world = points_camera @ frame.T_world_camera[:3, :3].T + frame.T_world_camera[:3, 3]
    pixels = np.column_stack((u, v)).astype(np.int32)
    colors = frame.rgb[pixels[:, 1], pixels[:, 0]]
    if len(points_world) > max_points:
        selected = np.linspace(0, len(points_world) - 1, max_points, dtype=np.int64)
        points_world, colors, pixels = points_world[selected], colors[selected], pixels[selected]
    return {"points_world_m": points_world, "colors_rgb": colors.copy(), "pixel_xy": pixels}


def rgbd_pointcloud_from_frame(
    frame: PublicRGBDFrame,
    *,
    observation_id: str,
    stride: int = 6,
    max_points: int = 30_000,
) -> dict[str, Any]:
    """Expose raw camera-derived world points; no robot self-filter is applied here."""
    if not isinstance(observation_id, str) or not observation_id.strip():
        raise ValueError("observation_id must be a non-empty string")
    cloud = point_cloud_world(frame, stride=stride, max_points=max_points)
    camera_observation_id = f"{observation_id}:camera:{frame.frame_id}"
    points = cloud["points_world_m"]
    return {
        "source_observation_id": camera_observation_id,
        "timestamp_s": frame.timestamp_s,
        "camera_frame_id": frame.frame_id,
        "frame": "world",
        "valid": bool(len(points)),
        "points_m": points.tolist(),
        "robot_self_filtered": False,
        "source": "task camera RGB-D axial-z backprojection",
        "calibration": {
            "K": frame.K.tolist(),
            "T_world_camera": frame.T_world_camera.tolist(),
            "depth_convention": "axial_z_m",
        },
    }


def _unproject_pixels(frame: PublicRGBDFrame, pixels_xy: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixels_xy, dtype=np.float64).reshape(-1, 2)
    z = np.asarray(depth_m, dtype=np.float64).reshape(-1)
    if len(pixels) != len(z):
        raise ValueError("pixel and depth arrays must have matching lengths")
    camera_points = np.column_stack(
        (
            (pixels[:, 0] - frame.K[0, 2]) * z / frame.K[0, 0],
            (pixels[:, 1] - frame.K[1, 2]) * z / frame.K[1, 1],
            z,
        )
    )
    return camera_points @ frame.T_world_camera[:3, :3].T + frame.T_world_camera[:3, 3]


def _fit_world_surface(
    points_world_m: np.ndarray,
    *,
    camera_position_world_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    points = np.asarray(points_world_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("at least three world points are required to fit a surface")
    center = points.mean(axis=0)
    _, singular_values, vh = np.linalg.svd(points - center, full_matrices=False)
    if singular_values[1] <= 1e-12:
        raise ValueError("surface points must have two-dimensional support")
    normal = vh[-1]
    if np.dot(normal, camera_position_world_m - center) < 0.0:
        normal = -normal
    axis = vh[0] - normal * np.dot(vh[0], normal)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    tangent = np.cross(normal, axis)
    tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
    axis = np.cross(tangent, normal)
    rotation = np.column_stack((axis, tangent, normal))
    projected = (points - center) @ rotation
    extent = projected.max(axis=0) - projected.min(axis=0)
    rms = float(np.sqrt(np.mean(np.square((points - center) @ normal))))
    return center, normal, rotation, extent, rms


def _fresh_feature_ref(
    observation_id: str,
    frame_id: int,
    kind: str,
    pixel_xy: np.ndarray | list[float],
) -> str:
    x, y = np.asarray(pixel_xy, dtype=np.float64).reshape(2)
    return f"{observation_id}:cam{frame_id}:{kind}:{x:.1f},{y:.1f}"


def _pixel_quantization_uncertainty(
    frame: PublicRGBDFrame,
    *,
    depth_m: float,
    characteristic_length_m: float,
    fit_rms_m: float = 0.0,
) -> dict[str, Any]:
    pixel_pitch = depth_m * math.sqrt(frame.K[0, 0] ** -2 + frame.K[1, 1] ** -2)
    return {
        "position_std_m": float(math.hypot(pixel_pitch / math.sqrt(12.0), fit_rms_m)),
        "axis_std_rad": float(
            math.atan2(max(pixel_pitch, fit_rms_m), max(characteristic_length_m, pixel_pitch))
        ),
        "uncertainty_source": "pixel quantization and geometric fit residual; depth/calibration noise unmodeled",
    }


def _add_observed_shape_bounds(feature: dict[str, Any]) -> dict[str, Any]:
    """Expose bounds only when current 3-D samples support a finite patch estimate."""
    geometry = feature.get("geometry")
    uncertainty = feature.get("uncertainty")
    if not isinstance(geometry, dict) or not isinstance(uncertainty, Mapping):
        return feature
    extent = np.asarray(geometry.get("extent_m", ()), dtype=np.float64)
    points = np.asarray(geometry.get("pointcloud_world_m", ()), dtype=np.float64)
    sample_count = uncertainty.get("sample_count")
    position_std = uncertainty.get("position_std_m")
    if (
        extent.shape != (3,)
        or not np.isfinite(extent).all()
        or np.any(extent < 0.0)
        or not isinstance(sample_count, int)
        or sample_count < 12
        or points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 6
        or not np.isfinite(points).all()
        or isinstance(position_std, bool)
        or not isinstance(position_std, (int, float))
        or not math.isfinite(float(position_std))
        or position_std < 0.0
    ):
        return feature

    kind = feature.get("kind")
    if kind == "plane_candidate":
        fit_rms = geometry.get("plane_fit_rms_m")
        if not isinstance(fit_rms, (int, float)) or fit_rms > max(0.002, 0.1 * float(np.max(extent))):
            return feature
    elif kind not in {"depth_surface_patch", "pixel_selected_depth_surface"}:
        fit_rms = uncertainty.get("contour_fit_rms_normalized")
        plane_rms = geometry.get("rim_plane_fit_rms_m")
        if (
            not isinstance(fit_rms, (int, float))
            or fit_rms > 0.16
            or not isinstance(plane_rms, (int, float))
            or plane_rms > max(0.002, 0.25 * float(min(extent[:2])))
        ):
            return feature

    radius = float(np.linalg.norm(extent) * 0.5)
    if radius <= 0.0:
        return feature
    geometry["bounding_radius_m"] = radius

    circle = geometry.get("image_circle_candidate_px")
    if kind == "depth_surface_patch" and isinstance(circle, Mapping):
        circle_radius = circle.get("radius_px")
        circle_fit = circle.get("fit_rms_px")
        in_plane = np.sort(extent[:2])
        if (
            isinstance(circle_radius, (int, float))
            and circle_radius > 0.0
            and isinstance(circle_fit, (int, float))
            and circle_fit / circle_radius <= 0.08
            and in_plane[0] > 4.0 * position_std
            and in_plane[1] / in_plane[0] <= 1.5
        ):
            geometry["grasp_width_m"] = float(in_plane[1] + 2.0 * position_std)
    return feature


def _contact_surface_geometry(frame: PublicRGBDFrame, points_world_m: np.ndarray) -> dict[str, Any]:
    # Runtime contact evidence must not inherit the 32-point display budget.
    selected = np.linspace(0, len(points_world_m) - 1, min(512, len(points_world_m)), dtype=np.int64)
    return {
        "contact_surface_points_world_m": points_world_m[selected].tolist(),
        "contact_surface_frame_id": int(frame.frame_id),
        "contact_surface_timestamp_s": float(frame.timestamp_s),
    }


def _surface_feature(
    frame: PublicRGBDFrame,
    *,
    observation_id: str,
    kind: str,
    pixels_xy: np.ndarray,
    points_world_m: np.ndarray,
    description: str,
    extra_geometry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pixels = np.asarray(pixels_xy, dtype=np.float64).reshape(-1, 2)
    points = np.asarray(points_world_m, dtype=np.float64).reshape(-1, 3)
    center, normal, rotation, extent, fit_rms = _fit_world_surface(
        points, camera_position_world_m=frame.T_world_camera[:3, 3]
    )
    axis = rotation[:, 0]
    selected = np.linspace(0, len(points) - 1, min(32, len(points)), dtype=np.int64)
    bbox = [
        int(np.floor(pixels[:, 0].min())), int(np.floor(pixels[:, 1].min())),
        int(np.ceil(pixels[:, 0].max())), int(np.ceil(pixels[:, 1].max())),
    ]
    depth_m = float(np.median(frame.depth_m[
        np.clip(pixels[:, 1].astype(np.int32), 0, frame.depth_m.shape[0] - 1),
        np.clip(pixels[:, 0].astype(np.int32), 0, frame.depth_m.shape[1] - 1),
    ]))
    geometry: dict[str, Any] = {
        "position_m": center.tolist(),
        "axis": axis.tolist(),
        "axis_kind": "surface_patch_principal_extent",
        "normal": normal.tolist(),
        "normal_kind": "local_depth_surface_normal",
        "quaternion_wxyz": _rotation_matrix_to_quat_wxyz(rotation).tolist(),
        "extent_m": extent.tolist(),
        "pointcloud_world_m": points[selected].tolist(),
        **_contact_surface_geometry(frame, points),
        "pixel_bbox_xyxy": bbox,
        "pixel_centroid_xy": pixels.mean(axis=0).tolist(),
        "surface_fit_rms_m": fit_rms,
    }
    if extra_geometry:
        geometry.update(dict(extra_geometry))
    uncertainty = _pixel_quantization_uncertainty(
        frame, depth_m=depth_m, characteristic_length_m=float(max(extent)), fit_rms_m=fit_rms
    )
    uncertainty.update(
        {
            "empirical_point_spread_world_m": np.std(points, axis=0).tolist(),
            "sample_count": int(len(points)),
            "semantic_identity": "unresolved",
        }
    )
    return _add_observed_shape_bounds({
        "feature_ref": _fresh_feature_ref(observation_id, frame.frame_id, kind, pixels.mean(axis=0)),
        "kind": kind,
        "description": description,
        "frame": "world",
        "geometry": geometry,
        "uncertainty": uncertainty,
        "source_observation_id": observation_id,
        "source_camera_frame_id": frame.frame_id,
    })


def estimate_feature_from_pixel_roi(
    frame: PublicRGBDFrame,
    *,
    observation_id: str,
    pixel_xy: tuple[int, int],
    radius_px: int = 24,
    min_points: int = 8,
) -> dict[str, Any]:
    """Refine a selected pixel into an RGB-D circular feature or local surface."""
    if len(pixel_xy) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in pixel_xy):
        raise ValueError("pixel_xy must be an integer (x, y) pair")
    if isinstance(radius_px, bool) or not isinstance(radius_px, int) or radius_px < 2:
        raise ValueError("radius_px must be an integer >= 2")
    if isinstance(min_points, bool) or not isinstance(min_points, int) or min_points < 3:
        raise ValueError("min_points must be an integer >= 3")
    clicked_x, clicked_y = pixel_xy
    height, width = frame.depth_m.shape
    if not (0 <= clicked_x < width and 0 <= clicked_y < height):
        raise ValueError("pixel_xy is outside the current camera frame")
    x0, x1 = max(0, clicked_x - radius_px), min(width, clicked_x + radius_px + 1)
    y0, y1 = max(0, clicked_y - radius_px), min(height, clicked_y + radius_px + 1)
    roi_intrinsics = frame.K.copy()
    roi_intrinsics[0, 2] -= x0
    roi_intrinsics[1, 2] -= y0
    roi_frame = PublicRGBDFrame(
        frame_id=frame.frame_id,
        timestamp_s=frame.timestamp_s,
        rgb=frame.rgb[y0:y1, x0:x1],
        depth_m=frame.depth_m[y0:y1, x0:x1],
        K=roi_intrinsics,
        T_world_camera=frame.T_world_camera,
    )
    try:
        import cv2  # noqa: F401
    except ImportError:
        roi_candidates = []
    else:
        roi_candidates = estimate_rgbd_contour_features(
            roi_frame,
            observation_id=observation_id,
            min_diameter_px=4.0,
            max_diameter_px=float(2 * radius_px),
            max_features=128,
            require_aperture_or_relief=False,
        )
    selected_candidates = []
    for candidate in roi_candidates:
        ellipse = candidate["geometry"]["image_ellipse_candidate_px"]
        major, minor = (float(value) / 2.0 for value in ellipse["axes_px"])
        dx, dy = clicked_x - x0 - ellipse["center_xy"][0], clicked_y - y0 - ellipse["center_xy"][1]
        angle = float(ellipse["angle_rad"])
        local_x = math.cos(angle) * dx + math.sin(angle) * dy
        local_y = -math.sin(angle) * dx + math.cos(angle) * dy
        radial_distance = math.sqrt((local_x / major) ** 2 + (local_y / minor) ** 2)
        if radial_distance <= 1.25:
            evidence = candidate["geometry"]["rgbd_shape_evidence"]
            score = (
                not evidence["aperture_or_relief_supported"],
                radial_distance,
                float(ellipse["fit_rms_normalized"]),
            )
            selected_candidates.append((score, candidate, radial_distance))
    if selected_candidates:
        _, selected, radial_distance = min(selected_candidates, key=lambda item: item[0])
        selected = dict(selected)
        geometry = dict(selected["geometry"])
        ellipse = dict(geometry["image_ellipse_candidate_px"])
        ellipse["center_xy"] = [ellipse["center_xy"][0] + x0, ellipse["center_xy"][1] + y0]
        geometry["image_ellipse_candidate_px"] = ellipse
        geometry["pixel_centroid_xy"] = [
            geometry["pixel_centroid_xy"][0] + x0,
            geometry["pixel_centroid_xy"][1] + y0,
        ]
        bbox = geometry["pixel_bbox_xyxy"]
        geometry["pixel_bbox_xyxy"] = [bbox[0] + x0, bbox[1] + y0, bbox[2] + x0, bbox[3] + y0]
        geometry.update(
            {
                "selected_pixel_xy": [clicked_x, clicked_y],
                "selected_pixel_ellipse_radius": float(radial_distance),
                "pixel_roi_xyxy": [
                    max(0, clicked_x - radius_px),
                    max(0, clicked_y - radius_px),
                    min(width - 1, clicked_x + radius_px),
                    min(height - 1, clicked_y + radius_px),
                ],
            }
        )
        selected["geometry"] = geometry
        selected["feature_ref"] = _fresh_feature_ref(
            observation_id, frame.frame_id, selected["kind"], ellipse["center_xy"]
        )
        selected["source_camera_observation_id"] = f"{observation_id}:camera:{frame.frame_id}"
        return selected
    depth = frame.depth_m[y0:y1, x0:x1]
    valid_y, valid_x = np.where(np.isfinite(depth))
    if len(valid_x) == 0:
        raise ValueError("selected image region has no valid depth")
    distances2 = (valid_x + x0 - clicked_x) ** 2 + (valid_y + y0 - clicked_y) ** 2
    seed_index = int(np.argmin(distances2))
    seed_row, seed_col = int(valid_y[seed_index]), int(valid_x[seed_index])
    seed_z = float(depth[seed_row, seed_col])

    grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
    z_grid = frame.depth_m[y0:y1, x0:x1]
    points_camera = np.stack(
        ((grid_x - frame.K[0, 2]) * z_grid / frame.K[0, 0],
         (grid_y - frame.K[1, 2]) * z_grid / frame.K[1, 1], z_grid),
        axis=-1,
    )
    valid = np.isfinite(z_grid)
    visited = np.zeros_like(valid)
    stack = [(seed_row, seed_col)]
    visited[seed_row, seed_col] = True
    cells = []
    while stack:
        row, col = stack.pop()
        cells.append((row, col))
        point = points_camera[row, col]
        threshold = max(0.0025, 1.6 * float(point[2]) / min(frame.K[0, 0], frame.K[1, 1]))
        for next_row, next_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if (
                next_row < 0 or next_row >= valid.shape[0] or next_col < 0 or next_col >= valid.shape[1]
                or visited[next_row, next_col] or not valid[next_row, next_col]
            ):
                continue
            if np.linalg.norm(points_camera[next_row, next_col] - point) <= threshold:
                visited[next_row, next_col] = True
                stack.append((next_row, next_col))
    if len(cells) < min_points:
        raise ValueError(f"selected depth surface has only {len(cells)} connected samples")
    local_y, local_x = np.asarray(cells, dtype=np.int32).T
    pixels = np.column_stack((local_x + x0, local_y + y0))
    depths = frame.depth_m[pixels[:, 1], pixels[:, 0]]
    points_world = _unproject_pixels(frame, pixels, depths)
    feature = _surface_feature(
        frame,
        observation_id=observation_id,
        kind="pixel_selected_depth_surface",
        pixels_xy=pixels,
        points_world_m=points_world,
        description="Depth-connected surface around a selected pixel; semantic identity is unresolved.",
        extra_geometry={"selected_pixel_xy": [clicked_x, clicked_y]},
    )
    feature["source_camera_observation_id"] = f"{observation_id}:camera:{frame.frame_id}"
    return feature


def estimate_plane_features(
    frame: PublicRGBDFrame,
    *,
    observation_id: str,
    stride: int = 5,
    max_planes: int = 3,
    min_inliers: int = 120,
    threshold_m: float = 0.005,
    max_iterations: int = 96,
) -> list[dict[str, Any]]:
    """Find dominant unlabeled 3-D planes by bounded RGB-D RANSAC."""
    if not observation_id.strip():
        raise ValueError("observation_id must be non-empty")
    if min(stride, max_planes, min_inliers, max_iterations) < 1 or threshold_m <= 0.0:
        raise ValueError("plane detector limits must be positive")
    cloud = point_cloud_world(frame, stride=stride, max_points=16000)
    points = cloud["points_world_m"]
    pixels = cloud["pixel_xy"]
    remaining = np.arange(len(points), dtype=np.int64)
    rng = np.random.default_rng(991 + frame.frame_id)
    features = []
    for _plane_index in range(max_planes):
        if len(remaining) < min_inliers:
            break
        candidates = points[remaining]
        best_local: np.ndarray | None = None
        for _ in range(max_iterations):
            sample = rng.choice(len(candidates), size=3, replace=False)
            p0, p1, p2 = candidates[sample]
            normal = np.cross(p1 - p0, p2 - p0)
            norm = float(np.linalg.norm(normal))
            if norm < 1e-8:
                continue
            normal /= norm
            inliers = np.flatnonzero(np.abs((candidates - p0) @ normal) <= threshold_m)
            if best_local is None or len(inliers) > len(best_local):
                best_local = inliers
        if best_local is None or len(best_local) < min_inliers:
            break
        selected_indices = remaining[best_local]
        plane_points = points[selected_indices]
        plane_pixels = pixels[selected_indices]
        center, normal, rotation, extent, fit_rms = _fit_world_surface(
            plane_points, camera_position_world_m=frame.T_world_camera[:3, 3]
        )
        extra = {
            "axis": rotation[:, 0].tolist(),
            "axis_kind": "dominant_plane_in_extent_direction",
            "normal": normal.tolist(),
            "normal_kind": "rgbd_plane_normal",
            "quaternion_wxyz": _rotation_matrix_to_quat_wxyz(rotation).tolist(),
            "extent_m": extent.tolist(),
            "plane_fit_rms_m": fit_rms,
            "pointcloud_world_m": plane_points[
                np.linspace(0, len(plane_points) - 1, min(32, len(plane_points)), dtype=np.int64)
            ].tolist(),
            "pixel_bbox_xyxy": [
                int(plane_pixels[:, 0].min()), int(plane_pixels[:, 1].min()),
                int(plane_pixels[:, 0].max()), int(plane_pixels[:, 1].max()),
            ],
            "pixel_centroid_xy": plane_pixels.mean(axis=0).tolist(),
        }
        z_values = frame.depth_m[plane_pixels[:, 1], plane_pixels[:, 0]]
        uncertainty = _pixel_quantization_uncertainty(
            frame, depth_m=float(np.median(z_values)),
            characteristic_length_m=float(max(extent)), fit_rms_m=fit_rms,
        )
        uncertainty.update({"sample_count": int(len(plane_points)), "semantic_identity": "unresolved"})
        feature = _add_observed_shape_bounds({
            "feature_ref": _fresh_feature_ref(observation_id, frame.frame_id, "plane_candidate", plane_pixels.mean(axis=0)),
            "kind": "plane_candidate",
            "description": "Unlabeled dominant plane fitted from current RGB-D points.",
            "frame": "world",
            "geometry": {"position_m": center.tolist(), **extra},
            "uncertainty": uncertainty,
            "source_observation_id": observation_id,
            "source_camera_frame_id": frame.frame_id,
        })
        features.append(feature)
        remaining = np.delete(remaining, best_local)
    return features


def estimate_rgbd_contour_features(
    frame: PublicRGBDFrame,
    *,
    observation_id: str,
    min_diameter_px: float = 4.0,
    max_diameter_px: float = 240.0,
    max_features: int = 32,
    require_aperture_or_relief: bool = True,
) -> list[dict[str, Any]]:
    """Fit unlabeled RGB ellipses and validate them against local RGB-D evidence."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("RGB contour fitting requires OpenCV in the runtime image") from exc
    if min_diameter_px < 3.0 or max_diameter_px <= min_diameter_px or max_features < 1:
        raise ValueError("invalid RGB contour detector limits")

    gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0.8)
    edges = cv2.Canny(blurred, 35, 110, L2gradient=True)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    fitted = []
    for contour in contours:
        if len(contour) < 8:
            continue
        (center_x, center_y), (diameter_a, diameter_b), angle_deg = cv2.fitEllipse(contour)
        major, minor = max(diameter_a, diameter_b) / 2.0, min(diameter_a, diameter_b) / 2.0
        if minor * 2.0 < min_diameter_px or major * 2.0 > max_diameter_px or major / minor > 3.0:
            continue
        contour_xy = contour[:, 0, :].astype(np.float64)
        delta = contour_xy - np.asarray([center_x, center_y])
        # OpenCV may report the short-axis angle when the fitted rectangle swaps axes.
        possible_angles = (math.radians(float(angle_deg)), math.radians(float(angle_deg) + 90.0))
        fits = []
        for angle in possible_angles:
            rotated_x = math.cos(angle) * delta[:, 0] + math.sin(angle) * delta[:, 1]
            rotated_y = -math.sin(angle) * delta[:, 0] + math.cos(angle) * delta[:, 1]
            radial = np.sqrt((rotated_x / major) ** 2 + (rotated_y / minor) ** 2)
            residual = float(np.sqrt(np.mean(np.square(radial - 1.0))))
            fits.append((residual, angle, radial))
        fit_rms, angle, radial = min(fits, key=lambda item: item[0])
        if fit_rms > 0.16:
            continue
        rotated_x = math.cos(angle) * delta[:, 0] + math.sin(angle) * delta[:, 1]
        rotated_y = -math.sin(angle) * delta[:, 0] + math.cos(angle) * delta[:, 1]
        ellipse_angles = np.arctan2(rotated_y / minor, rotated_x / major)
        angular_bins = np.floor((ellipse_angles + math.pi) * (32.0 / (2.0 * math.pi))).astype(np.int32)
        arc_coverage = float(len(np.unique(angular_bins)) / 32.0)
        if arc_coverage < 0.58:
            continue
        fitted.append((fit_rms, center_x, center_y, major, minor, angle, "ellipse_fit", arc_coverage))

    # Hough candidates supplement closed contours when low-contrast recess rims
    # are broken by shading; edge support keeps isolated text arcs from winning.
    inverse_edges = np.where(edges > 0, 0, 255).astype(np.uint8)
    edge_distance = cv2.distanceTransform(inverse_edges, cv2.DIST_L2, 3)
    hough = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.25,
        minDist=max(6.0, 1.5 * min_diameter_px),
        param1=90.0,
        param2=14.0,
        minRadius=max(2, int(math.floor(min_diameter_px / 2.0))),
        maxRadius=max(4, int(math.ceil(max_diameter_px / 2.0))),
    )
    if hough is not None:
        for center_x, center_y, radius in hough[0]:
            theta = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
            sample_x = np.clip(np.rint(center_x + radius * np.cos(theta)).astype(int), 0, edges.shape[1] - 1)
            sample_y = np.clip(np.rint(center_y + radius * np.sin(theta)).astype(int), 0, edges.shape[0] - 1)
            support = float(np.mean(edge_distance[sample_y, sample_x] <= 2.5))
            if support < 0.72:
                continue
            normalized_fit = float(1.0 - support)
            fitted.append(
                (
                    normalized_fit,
                    float(center_x),
                    float(center_y),
                    float(radius),
                    float(radius),
                    0.0,
                    "hough_edge_support",
                    support,
                )
            )

    fitted.sort(key=lambda item: (item[0], item[3] * item[4]))
    features = []
    gray_frame = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
    for fit_rms, center_x, center_y, major, minor, angle, fit_method, arc_coverage in fitted:
        candidate_axes = np.asarray([2.0 * major, 2.0 * minor], dtype=np.float64)
        if any(
            np.linalg.norm(np.asarray([center_x, center_y]) - np.asarray(item["geometry"]["pixel_centroid_xy"]))
            < 0.6 * min(major, minor)
            and np.max(
                np.abs(
                    np.log(
                        candidate_axes
                        / np.maximum(
                            np.asarray(item["geometry"]["image_ellipse_candidate_px"]["axes_px"]),
                            1e-6,
                        )
                    )
                )
            ) < 0.18
            for item in features
        ):
            continue
        pad_x, pad_y = int(math.ceil(major + 2)), int(math.ceil(major + 2))
        x0, x1 = max(0, int(center_x) - pad_x), min(frame.rgb.shape[1], int(center_x) + pad_x + 1)
        y0, y1 = max(0, int(center_y) - pad_y), min(frame.rgb.shape[0], int(center_y) + pad_y + 1)
        grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
        dx, dy = grid_x - center_x, grid_y - center_y
        rotated_x = math.cos(angle) * dx + math.sin(angle) * dy
        rotated_y = -math.sin(angle) * dx + math.cos(angle) * dy
        radial = np.sqrt((rotated_x / major) ** 2 + (rotated_y / minor) ** 2)
        depth_crop = frame.depth_m[y0:y1, x0:x1]
        center_valid = (radial <= 0.48) & np.isfinite(depth_crop)
        rim_valid = (radial >= 0.72) & (radial <= 1.15) & np.isfinite(depth_crop)
        gray_crop = gray_frame[y0:y1, x0:x1]
        aperture_gray = gray_crop[radial <= 0.55]
        rim_gray = gray_crop[(radial >= 0.72) & (radial <= 1.15)]
        dark_contrast = None
        dark_aperture = False
        if len(aperture_gray) >= 3 and len(rim_gray) >= 6:
            aperture_median = float(np.median(aperture_gray))
            rim_gray_median = float(np.median(rim_gray))
            dark_contrast = rim_gray_median - aperture_median
            dark_aperture = dark_contrast >= max(12.0, 0.08 * rim_gray_median)
        rim_y, rim_x = np.where(rim_valid)
        if len(rim_x) < 6:
            continue
        rim_pixels = np.column_stack((rim_x + x0, rim_y + y0))
        rim_depth = depth_crop[rim_y, rim_x]
        rim_points_world = _unproject_pixels(frame, rim_pixels, rim_depth)
        center_depths = depth_crop[center_valid]
        center_depth = float(np.median(center_depths)) if len(center_depths) >= 3 else None
        rim_depth_m = float(np.median(rim_depth))
        try:
            rim_center, normal, rotation, extent, plane_rms = _fit_world_surface(
                rim_points_world, camera_position_world_m=frame.T_world_camera[:3, 3]
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        camera_origin = frame.T_world_camera[:3, 3]
        camera_rotation = frame.T_world_camera[:3, :3]
        plane_offset = float(np.dot(normal, rim_center - camera_origin))
        if abs(plane_offset) <= 1e-12:
            continue

        def intersect_rim_plane(pixel: np.ndarray) -> np.ndarray | None:
            ray_camera = np.asarray(
                [(pixel[0] - frame.K[0, 2]) / frame.K[0, 0],
                 (pixel[1] - frame.K[1, 2]) / frame.K[1, 1], 1.0]
            )
            ray_world = camera_rotation @ ray_camera
            denominator = float(np.dot(normal, ray_world))
            if abs(denominator) < 1e-8:
                return None
            scale = plane_offset / denominator
            if scale <= 0.0:
                return None
            return camera_origin + scale * ray_world

        center_world = intersect_rim_plane(np.asarray([center_x, center_y]))
        if center_world is None:
            continue
        expected_center_depth = float((camera_rotation.T @ (center_world - camera_origin))[2])
        depth_delta = None if center_depth is None else center_depth - expected_center_depth
        relief_threshold = max(0.002, 3.0 * plane_rms)
        depth_recess = depth_delta is not None and depth_delta > relief_threshold
        depth_raise = depth_delta is not None and depth_delta < -relief_threshold
        aperture_or_relief_supported = dark_aperture or depth_recess or depth_raise
        if require_aperture_or_relief and not aperture_or_relief_supported:
            continue
        kind = (
            "recessed_circular_candidate" if depth_recess else
            "raised_circular_candidate" if depth_raise else
            "dark_aperture_candidate" if dark_aperture else
            "circular_surface_candidate"
        )
        boundary_world = []
        for theta in np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False):
            px = center_x + math.cos(angle) * major * math.cos(theta) - math.sin(angle) * minor * math.sin(theta)
            py = center_y + math.sin(angle) * major * math.cos(theta) + math.cos(angle) * minor * math.sin(theta)
            point = intersect_rim_plane(np.asarray([px, py]))
            if point is not None:
                boundary_world.append(point)
        if len(boundary_world) < 12:
            continue
        boundary_world = np.asarray(boundary_world)
        tangent_points = boundary_world - center_world
        major_tangent = tangent_points[0] - normal * float(np.dot(tangent_points[0], normal))
        tangent_norm = float(np.linalg.norm(major_tangent))
        if not math.isfinite(tangent_norm) or tangent_norm <= 1e-12:
            continue
        major_tangent /= tangent_norm
        bitangent = np.cross(normal, major_tangent)
        bitangent_norm = float(np.linalg.norm(bitangent))
        if not math.isfinite(bitangent_norm) or bitangent_norm <= 1e-12:
            continue
        bitangent /= bitangent_norm
        major_tangent = np.cross(bitangent, normal)
        feature_rotation = np.column_stack((major_tangent, bitangent, normal))
        projected = tangent_points @ feature_rotation
        ellipse_extent = projected.max(axis=0) - projected.min(axis=0)
        bbox = [
            max(0, int(math.floor(center_x - pad_x))), max(0, int(math.floor(center_y - pad_y))),
            min(frame.rgb.shape[1] - 1, int(math.ceil(center_x + pad_x))),
            min(frame.rgb.shape[0] - 1, int(math.ceil(center_y + pad_y))),
        ]
        center_pixel = [float(center_x), float(center_y)]
        geometry = {
            "position_m": center_world.tolist(),
            "axis": normal.tolist(),
            "axis_kind": "fitted_circular_rim_plane_normal_not_semantic_part_axis",
            "normal": normal.tolist(),
            "normal_kind": "fitted_rim_surface_normal",
            "quaternion_wxyz": _rotation_matrix_to_quat_wxyz(feature_rotation).tolist(),
            "extent_m": ellipse_extent.tolist(),
            "pointcloud_world_m": rim_points_world[
                np.linspace(0, len(rim_points_world) - 1, min(32, len(rim_points_world)), dtype=np.int64)
            ].tolist(),
            "pixel_bbox_xyxy": bbox,
            "pixel_centroid_xy": center_pixel,
            "position_kind": "image_ellipse_center_intersected_with_rgbd_rim_plane",
            "image_ellipse_candidate_px": {
                "center_xy": center_pixel,
                "axes_px": [float(major * 2.0), float(minor * 2.0)],
                "angle_rad": float(angle),
                "fit_rms_normalized": fit_rms,
                "fit_method": fit_method,
                "angular_coverage_fraction": float(arc_coverage),
            },
            "center_minus_rim_depth_m": depth_delta,
            "rim_depth_m": rim_depth_m,
            "center_depth_m": center_depth,
            "center_expected_rim_depth_m": expected_center_depth,
            "rim_plane_fit_rms_m": plane_rms,
            "rgbd_shape_evidence": {
                "dark_aperture_contrast_gray": dark_contrast,
                "dark_aperture_threshold_gray": max(12.0, 0.08 * float(np.median(rim_gray))) if len(rim_gray) else None,
                "dark_aperture_supported": bool(dark_aperture),
                "depth_recess_supported": bool(depth_recess),
                "depth_raise_supported": bool(depth_raise),
                "depth_relief_threshold_m": float(relief_threshold),
                "aperture_or_relief_supported": bool(aperture_or_relief_supported),
                "candidate_identity": "unresolved_circular_surface_or_aperture",
            },
        }
        uncertainty = _pixel_quantization_uncertainty(
            frame,
            depth_m=rim_depth_m,
            characteristic_length_m=float(max(ellipse_extent)),
            fit_rms_m=plane_rms,
        )
        uncertainty.update(
            {
                "axis_std_rad": float(
                    min(
                        math.pi / 2.0,
                        max(
                            uncertainty["axis_std_rad"],
                            fit_rms,
                            plane_rms / max(float(min(ellipse_extent[:2])), 1e-9),
                        ),
                    )
                ),
                "contour_fit_rms_normalized": fit_rms,
                "sample_count": int(len(rim_points_world)),
                "semantic_identity": "unresolved",
                "rgbd_shape_evidence": geometry["rgbd_shape_evidence"],
            }
        )
        features.append(
            _add_observed_shape_bounds({
                "feature_ref": _fresh_feature_ref(observation_id, frame.frame_id, kind, center_pixel),
                "kind": kind,
                "description": "Unlabeled closed RGB ellipse with fitted rim plane and explicit local aperture/relief evidence.",
                "frame": "world",
                "geometry": geometry,
                "uncertainty": uncertainty,
                "source_observation_id": observation_id,
                "source_camera_frame_id": frame.frame_id,
            })
        )
        if len(features) >= max_features:
            break
    return features


def filter_robot_overlapping_contour_features(
    features: Sequence[dict[str, Any]],
    robot_matched_pixels: np.ndarray,
) -> list[dict[str, Any]]:
    """Drop fitted circular candidates whose visible rim is mostly the robot."""
    robot_mask = np.asarray(robot_matched_pixels, dtype=bool)
    if robot_mask.ndim != 2:
        raise ValueError("robot_matched_pixels must be a 2D mask")

    circular_kinds = {
        "dark_aperture_candidate",
        "recessed_circular_candidate",
        "raised_circular_candidate",
        "circular_surface_candidate",
    }
    kept = []
    for feature in features:
        geometry = feature.get("geometry", {})
        ellipse = geometry.get("image_ellipse_candidate_px")
        if feature.get("kind") not in circular_kinds or not isinstance(ellipse, Mapping):
            kept.append(feature)
            continue
        center = np.asarray(ellipse.get("center_xy", ()), dtype=float)
        axes = np.asarray(ellipse.get("axes_px", ()), dtype=float)
        angle = ellipse.get("angle_rad")
        if (
            center.shape != (2,) or axes.shape != (2,) or not np.isfinite(center).all()
            or not np.isfinite(axes).all() or np.any(axes <= 0.0)
            or not isinstance(angle, (int, float)) or not math.isfinite(float(angle))
        ):
            kept.append(feature)
            continue

        major, minor = axes / 2.0
        pad = int(math.ceil(major * 1.15))
        x0, x1 = max(0, int(center[0]) - pad), min(robot_mask.shape[1], int(center[0]) + pad + 1)
        y0, y1 = max(0, int(center[1]) - pad), min(robot_mask.shape[0], int(center[1]) + pad + 1)
        if x0 >= x1 or y0 >= y1:
            kept.append(feature)
            continue
        grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
        dx, dy = grid_x - center[0], grid_y - center[1]
        cosine, sine = math.cos(float(angle)), math.sin(float(angle))
        rotated_x = cosine * dx + sine * dy
        rotated_y = -sine * dx + cosine * dy
        radius = np.sqrt((rotated_x / major) ** 2 + (rotated_y / minor) ** 2)
        rim = (radius >= 0.72) & (radius <= 1.15)
        # shortcut: only suppress majority robot-overlapped rims, revisit if target holes are often occluded by the arm.
        if np.count_nonzero(rim) >= 6 and float(np.mean(robot_mask[y0:y1, x0:x1][rim])) > 0.5:
            continue
        kept.append(feature)
    return kept


def robot_tcp_feature(
    *,
    observation_id: str,
    camera_frame_id: int,
    tcp_pose_wxyz: Any,
    finger_opening_axis_world: Any,
    gripper_width_m: float,
    robot_timestamp_s: float,
) -> dict[str, Any]:
    """Expose canonical TCP-local geometry and measured gripper state."""
    pose = _numpy(tcp_pose_wxyz).astype(np.float64).reshape(-1)
    finger_axis = _numpy(finger_opening_axis_world).astype(np.float64).reshape(-1)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("tcp_pose_wxyz must be a finite position+quaternion 7-vector")
    if finger_axis.shape != (3,) or not np.isfinite(finger_axis).all():
        raise ValueError("finger_opening_axis_world must be a finite 3-vector")
    if not math.isfinite(gripper_width_m) or gripper_width_m < 0.0:
        raise ValueError("gripper_width_m must be finite and non-negative")
    finger_norm = float(np.linalg.norm(finger_axis))
    if finger_norm < 1e-8:
        raise ValueError("finger opening axis must have nonzero length")
    rotation = _quat_wxyz_to_matrix(pose[3:])
    finger_axis_local = rotation.T @ (finger_axis / finger_norm)
    if not math.isfinite(robot_timestamp_s):
        raise ValueError("robot_timestamp_s must be finite")
    return {
        "feature_ref": f"{observation_id}:cam{camera_frame_id}:robot_tcp",
        "kind": "robot_tcp",
        "description": "Current TCP and finger opening from measured joints plus robot-model FK; not part ground truth.",
        "frame": "tcp",
        "geometry": {
            "position_m": [0.0, 0.0, 0.0],
            "axis": [0.0, 0.0, 1.0],
            "axis_kind": "tcp_local_z_approach_direction",
            "normal": finger_axis_local.tolist(),
            "normal_kind": "measured_finger_separation_direction_transformed_to_tcp",
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "gripper_width_m": float(gripper_width_m),
            "robot_timestamp_s": float(robot_timestamp_s),
        },
        "uncertainty": {
            "position_std_m": 0.0,
            "axis_std_rad": 0.0,
            "uncertainty_source": "deterministic simulator robot-model FK; physical calibration error unmodeled",
            "semantic_identity": "robot_tcp",
        },
        "source_observation_id": observation_id,
        "source_camera_frame_id": int(camera_frame_id),
    }


class FeatureTracker:
    """Associate consecutive estimates by measured geometry, never by list rank."""

    def __init__(self, *, max_position_distance_m: float = 0.025, ambiguity_margin_m: float = 0.003):
        if not math.isfinite(max_position_distance_m) or max_position_distance_m <= 0.0:
            raise ValueError("max_position_distance_m must be positive and finite")
        if not math.isfinite(ambiguity_margin_m) or ambiguity_margin_m < 0.0:
            raise ValueError("ambiguity_margin_m must be finite and non-negative")
        self.max_position_distance_m = float(max_position_distance_m)
        self.ambiguity_margin_m = float(ambiguity_margin_m)
        self._previous: list[dict[str, Any]] = []

    @staticmethod
    def _match_cost(previous: Mapping[str, Any], current: Mapping[str, Any]) -> float | None:
        if previous.get("kind") != current.get("kind"):
            return None
        old_geometry, new_geometry = previous.get("geometry", {}), current.get("geometry", {})
        try:
            old_position = np.asarray(old_geometry["position_m"], dtype=np.float64)
            new_position = np.asarray(new_geometry["position_m"], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            return None
        if old_position.shape != (3,) or new_position.shape != (3,):
            return None
        distance = float(np.linalg.norm(old_position - new_position))
        old_extent = np.asarray(old_geometry.get("extent_m", []), dtype=np.float64).reshape(-1)
        new_extent = np.asarray(new_geometry.get("extent_m", []), dtype=np.float64).reshape(-1)
        if old_extent.size and new_extent.size:
            count = min(old_extent.size, new_extent.size, 3)
            old_shape = np.sort(np.maximum(old_extent[:count], 1e-4))
            new_shape = np.sort(np.maximum(new_extent[:count], 1e-4))
            shape_delta = float(np.mean(np.abs(np.log(old_shape / new_shape))))
        else:
            shape_delta = 0.0
        old_axis = old_geometry.get("axis")
        new_axis = new_geometry.get("axis")
        axis_delta = 0.0
        if old_axis is not None and new_axis is not None:
            a = np.asarray(old_axis, dtype=np.float64)
            b = np.asarray(new_axis, dtype=np.float64)
            if a.shape == (3,) and b.shape == (3,):
                axis_delta = 1.0 - min(1.0, abs(float(np.dot(a, b))))
        return distance + 0.01 * shape_delta + 0.01 * axis_delta

    def update(self, features: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return fresh observation refs plus geometry-associated track refs.

        A near-tie is left unmatched rather than inheriting an identity across
        visually/geometrically ambiguous repeated features.
        """
        candidates: list[tuple[float, int, int]] = []
        costs_by_current: dict[int, list[tuple[float, int]]] = {}
        for current_index, current in enumerate(features):
            costs = []
            for previous_index, previous in enumerate(self._previous):
                cost = self._match_cost(previous, current)
                if cost is not None and cost <= self.max_position_distance_m:
                    costs.append((cost, previous_index))
                    candidates.append((cost, current_index, previous_index))
            costs.sort()
            costs_by_current[current_index] = costs

        ambiguous = {
            index
            for index, costs in costs_by_current.items()
            if len(costs) > 1 and costs[1][0] - costs[0][0] < self.ambiguity_margin_m
        }
        assigned_current: set[int] = set()
        assigned_previous: set[int] = set()
        matches: dict[int, tuple[int, float]] = {}
        for cost, current_index, previous_index in sorted(candidates):
            if current_index in ambiguous or current_index in assigned_current or previous_index in assigned_previous:
                continue
            matches[current_index] = (previous_index, cost)
            assigned_current.add(current_index)
            assigned_previous.add(previous_index)

        import uuid

        updated = []
        for index, feature in enumerate(features):
            current = dict(feature)
            if index in matches:
                previous_index, cost = matches[index]
                previous = self._previous[previous_index]
                current["track_ref"] = previous["track_ref"]
                current["tracking"] = {
                    "status": "geometry_matched",
                    "previous_feature_ref": previous.get("feature_ref"),
                    "association_cost": float(cost),
                }
            else:
                current["track_ref"] = f"track-{uuid.uuid4().hex}"
                current["tracking"] = {
                    "status": "ambiguous" if index in ambiguous else "new_or_unmatched",
                    "previous_feature_ref": None,
                    "association_cost": None,
                }
            updated.append(current)
        self._previous = updated
        return [dict(item) for item in updated]


def _depth_components(
    frame: PublicRGBDFrame,
    *,
    stride: int,
    max_neighbor_distance_m: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    height, width = frame.depth_m.shape
    ys = np.arange(0, height, stride, dtype=np.int32)
    xs = np.arange(0, width, stride, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    z = frame.depth_m[grid_y, grid_x].astype(np.float64)
    valid = np.isfinite(z)
    fx, fy = frame.K[0, 0], frame.K[1, 1]
    points = np.stack(
        ((grid_x - frame.K[0, 2]) * z / fx, (grid_y - frame.K[1, 2]) * z / fy, z),
        axis=-1,
    )
    import os
    if os.environ.get("ASSEMBLY_FAST_COMPONENTS") == "1":
        from .fast_components import connected_cells
        return connected_cells(points,valid,max_neighbor_distance_m)
    grid_h, grid_w = z.shape
    visited = np.zeros((grid_h, grid_w), dtype=bool)
    components = []
    for row in range(grid_h):
        for col in range(grid_w):
            if not valid[row, col] or visited[row, col]:
                continue
            visited[row, col] = True
            stack = [(row, col)]
            cells = []
            while stack:
                current_row, current_col = stack.pop()
                cells.append((current_row, current_col))
                current = points[current_row, current_col]
                for next_row, next_col in (
                    (current_row - 1, current_col),
                    (current_row + 1, current_col),
                    (current_row, current_col - 1),
                    (current_row, current_col + 1),
                ):
                    if (
                        next_row < 0 or next_row >= grid_h or next_col < 0 or next_col >= grid_w
                        or visited[next_row, next_col] or not valid[next_row, next_col]
                    ):
                        continue
                    neighbor = points[next_row, next_col]
                    threshold = max(max_neighbor_distance_m, 0.01 * float(current[2]))
                    if float(np.linalg.norm(neighbor - current)) <= threshold:
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))
            if cells:
                rows, cols = np.asarray(cells, dtype=np.int32).T
                components.append((rows, cols))
    return components


def _fit_image_circle(rows: np.ndarray, cols: np.ndarray, *, stride: int) -> dict[str, Any] | None:
    """Return a generic 2-D circular contour fit for a sufficiently round patch."""
    cells = {(int(row), int(col)) for row, col in zip(rows, cols)}
    boundary = [
        (col * stride, row * stride)
        for row, col in cells
        if any((row + dr, col + dc) not in cells for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))
    ]
    if len(boundary) < 8:
        return None
    points = np.asarray(boundary, dtype=np.float64)
    x, y = points.T
    matrix = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    try:
        center_x, center_y, constant = np.linalg.lstsq(matrix, x * x + y * y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    radius_squared = constant + center_x * center_x + center_y * center_y
    if not math.isfinite(float(radius_squared)) or radius_squared <= 0.0:
        return None
    radius = math.sqrt(float(radius_squared))
    residual = np.linalg.norm(points - np.asarray([center_x, center_y]), axis=1) - radius
    rms = float(np.sqrt(np.mean(residual * residual)))
    bbox_width = float(np.ptp(x) + stride)
    bbox_height = float(np.ptp(y) + stride)
    aspect = min(bbox_width, bbox_height) / max(bbox_width, bbox_height)
    if radius < 2.0 * stride or aspect < 0.65 or rms / radius > 0.12:
        return None
    return {
        "center_xy": [float(center_x), float(center_y)],
        "radius_px": radius,
        "fit_rms_px": rms,
        "candidate_type": "unlabeled_image_circle",
    }


def estimate_depth_features(
    frame: PublicRGBDFrame,
    *,
    observation_id: str,
    stride: int = 3,
    min_component_points: int = 12,
    max_features: int = 36,
    max_points_per_feature: int = 32,
    max_neighbor_distance_m: float = 0.004,
) -> list[dict[str, Any]]:
    """Estimate unlabeled spatial surface patches from current RGB-D only.

    Components are image-connected depth surfaces, not object identities. PCA axes
    describe patch extents, a plane normal describes the surface, and neither is a
    bolt/hole axis. Point spread is descriptive, not calibrated localization error.
    """
    if not isinstance(observation_id, str) or not observation_id.strip():
        raise ValueError("observation_id must be a non-empty string")
    for value, name in (
        (stride, "stride"),
        (min_component_points, "min_component_points"),
        (max_features, "max_features"),
        (max_points_per_feature, "max_points_per_feature"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(max_neighbor_distance_m) or max_neighbor_distance_m <= 0.0:
        raise ValueError("max_neighbor_distance_m must be positive and finite")

    components = _depth_components(
        frame, stride=stride, max_neighbor_distance_m=max_neighbor_distance_m
    )
    components = [item for item in components if len(item[0]) >= min_component_points]
    components.sort(key=lambda item: len(item[0]), reverse=True)
    features = []
    for rank, (rows, cols) in enumerate(components[:max_features]):
        pixel_xy = np.column_stack((cols * stride, rows * stride)).astype(np.int32)
        z = frame.depth_m[pixel_xy[:, 1], pixel_xy[:, 0]].astype(np.float64)
        camera_points = np.column_stack(
            (
                (pixel_xy[:, 0] - frame.K[0, 2]) * z / frame.K[0, 0],
                (pixel_xy[:, 1] - frame.K[1, 2]) * z / frame.K[1, 1],
                z,
            )
        )
        world_points = camera_points @ frame.T_world_camera[:3, :3].T + frame.T_world_camera[:3, 3]
        center = world_points.mean(axis=0)
        centered = world_points - center
        covariance = centered.T @ centered / max(1, len(centered) - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        axes = eigenvectors[:, order]
        principal_axis = axes[:, 0].copy()
        sign_index = int(np.argmax(np.abs(principal_axis)))
        if principal_axis[sign_index] < 0.0:
            axes[:, 0] *= -1.0
        if np.linalg.det(axes) < 0.0:
            axes[:, 2] *= -1.0
        principal_axis = axes[:, 0]
        surface_normal = axes[:, 2]
        projected = centered @ axes
        extent = projected.max(axis=0) - projected.min(axis=0)
        bounds_center_offset = axes @ ((projected.min(axis=0) + projected.max(axis=0)) * 0.5)
        selected = np.linspace(0, len(world_points) - 1, min(max_points_per_feature, len(world_points)), dtype=np.int64)
        bbox = [
            int(pixel_xy[:, 0].min()),
            int(pixel_xy[:, 1].min()),
            int(min(frame.rgb.shape[1] - 1, pixel_xy[:, 0].max() + stride - 1)),
            int(min(frame.rgb.shape[0] - 1, pixel_xy[:, 1].max() + stride - 1)),
        ]
        plane_fit_rms = math.sqrt(max(0.0, float(eigenvalues[0])))
        image_circle_fit = _fit_image_circle(rows, cols, stride=stride)
        pixel_resolution_m = float(np.median(z) * math.sqrt(frame.K[0, 0] ** -2 + frame.K[1, 1] ** -2))
        axis_std_rad = (
            math.pi / 2.0
            if eigenvalues[-1] <= 1.15 * max(eigenvalues[-2], 1e-12)
            else math.atan2(pixel_resolution_m, max(float(extent[0]), 1e-9))
        )
        geometry: dict[str, Any] = {
            "position_m": center.tolist(),
            "position_kind": "sample_centroid_of_depth_patch",
            "axis": principal_axis.tolist(),
            "axis_kind": "largest_pca_extent_not_object_or_hole_axis",
            "quaternion_wxyz": _rotation_matrix_to_quat_wxyz(axes).tolist(),
            "extent_m": extent.tolist(),
            "pointcloud_world_m": world_points[selected].tolist(),
            **_contact_surface_geometry(frame, world_points),
            "pixel_bbox_xyxy": bbox,
            "pixel_centroid_xy": pixel_xy.mean(axis=0).tolist(),
        }
        if plane_fit_rms <= max(0.001, 0.02 * float(np.sqrt(max(eigenvalues[-1], 0.0)))):
            geometry["normal"] = surface_normal.tolist()
            geometry["surface_plane_fit_rms_m"] = plane_fit_rms
            geometry["normal_kind"] = "surface_patch_normal_not_hole_axis"
        if image_circle_fit is not None:
            geometry["image_circle_candidate_px"] = image_circle_fit
            geometry["patch_in_plane_min_extent_m"] = float(min(extent[:2]))
            geometry["shape_bounds_center_offset_world_m"] = bounds_center_offset.tolist()
        features.append(
            _add_observed_shape_bounds({
                "feature_ref": _fresh_feature_ref(
                    observation_id, frame.frame_id, "depth_patch", pixel_xy.mean(axis=0)
                ),
                "kind": "depth_surface_patch",
                "description": "Unlabeled RGB-D connected surface patch; semantic identity is unresolved.",
                "frame": "world",
                "geometry": geometry,
                "uncertainty": {
                    "position_std_m": pixel_resolution_m / math.sqrt(12.0),
                    "axis_std_rad": axis_std_rad,
                    "empirical_point_spread_world_m": np.sqrt(np.maximum(np.diag(covariance), 0.0)).tolist(),
                    "empirical_depth_spread_m": float(np.std(z)),
                    "uncertainty_source": "pixel_quantization_lower_bound_only; depth and calibration errors unmodeled",
                    "error_note": "empirical point spread describes patch geometry, not localization accuracy",
                    "patch_in_plane_min_extent_std_m": (
                        2.0 * pixel_resolution_m / math.sqrt(12.0) if image_circle_fit else None
                    ),
                    "patch_in_plane_min_extent_uncertainty_source": (
                        "pixel quantization lower bound only; patch boundaries, depth and calibration errors unmodeled"
                        if image_circle_fit else None
                    ),
                    "image_circle_fit_rms_px": image_circle_fit["fit_rms_px"] if image_circle_fit else None,
                    "sample_count": int(len(world_points)),
                    "semantic_identity": "unresolved",
                },
                "source_observation_id": observation_id,
                "source_camera_frame_id": frame.frame_id,
            })
        )
    return features


def _feature_pixel_center(
    frame: PublicRGBDFrame,
    feature: Mapping[str, Any],
    robot_state: Mapping[str, Any],
) -> tuple[int, int] | None:
    geometry = feature.get("geometry")
    if not isinstance(geometry, Mapping):
        return None
    for key in ("selected_pixel_xy", "pixel_centroid_xy"):
        value = geometry.get(key)
        if isinstance(value, (list, tuple, np.ndarray)):
            point = np.asarray(value, dtype=np.float64).reshape(-1)
            if point.shape == (2,) and np.isfinite(point).all():
                return tuple(int(round(float(axis))) for axis in point)
    for key in ("image_ellipse_candidate_px", "image_circle_candidate_px"):
        candidate = geometry.get(key)
        center = candidate.get("center_xy") if isinstance(candidate, Mapping) else None
        if isinstance(center, (list, tuple, np.ndarray)):
            point = np.asarray(center, dtype=np.float64).reshape(-1)
            if point.shape == (2,) and np.isfinite(point).all():
                return tuple(int(round(float(axis))) for axis in point)
    bbox = geometry.get("pixel_bbox_xyxy")
    if isinstance(bbox, (list, tuple, np.ndarray)):
        bounds = np.asarray(bbox, dtype=np.float64).reshape(-1)
        if bounds.shape == (4,) and np.isfinite(bounds).all():
            return tuple(int(round(float(axis))) for axis in ((bounds[:2] + bounds[2:]) / 2.0))

    position = np.asarray(geometry.get("position_m", ()), dtype=np.float64).reshape(-1)
    if position.shape != (3,) or not np.isfinite(position).all():
        return None
    feature_frame = feature.get("frame")
    if feature_frame == "tcp":
        tcp_pose = np.asarray(robot_state.get("tcp_pose", ()), dtype=np.float64).reshape(-1)
        if tcp_pose.shape != (7,) or not np.isfinite(tcp_pose).all():
            return None
        position = tcp_pose[:3] + _quat_wxyz_to_matrix(tcp_pose[3:]) @ position
    elif feature_frame != "world":
        return None
    camera_point = np.linalg.inv(frame.T_world_camera) @ np.append(position, 1.0)
    camera_point = camera_point[:3]
    if not np.isfinite(camera_point).all() or camera_point[2] <= 1e-9:
        return None
    homogeneous_pixel = frame.K @ camera_point
    if homogeneous_pixel[2] <= 1e-9:
        return None
    return tuple(int(round(float(axis))) for axis in homogeneous_pixel[:2] / homogeneous_pixel[2])


def _draw_public_feature_overlay(
    frame: PublicRGBDFrame,
    features: list[dict[str, Any]],
    path: Path,
    robot_state: Mapping[str, Any],
) -> None:
    import cv2

    overlay = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    height, width = overlay.shape[:2]
    palette = (
        (70, 220, 255), (255, 180, 60), (110, 245, 110), (245, 120, 210),
        (255, 235, 80), (180, 140, 255), (90, 230, 220), (220, 220, 220),
    )
    unprojected: list[str] = []
    for index, feature in enumerate(features, 1):
        geometry = feature.get("geometry", {})
        if not isinstance(geometry, Mapping):
            geometry = {}
        label = f"F{index:02d}"
        color = palette[(index - 1) % len(palette)]
        bbox = geometry.get("pixel_bbox_xyxy")
        if isinstance(bbox, (list, tuple, np.ndarray)):
            bounds = np.asarray(bbox, dtype=np.float64).reshape(-1)
            if bounds.shape == (4,) and np.isfinite(bounds).all():
                corners = np.rint(bounds).astype(int)
                cv2.rectangle(
                    overlay,
                    (int(corners[0]), int(corners[1])),
                    (int(corners[2]), int(corners[3])),
                    color,
                    1,
                    cv2.LINE_AA,
                )
        ellipse = geometry.get("image_ellipse_candidate_px")
        if isinstance(ellipse, Mapping):
            center = ellipse.get("center_xy")
            axes = ellipse.get("axes_px")
            angle = ellipse.get("angle_rad")
            if center is not None and axes is not None and isinstance(angle, (int, float)):
                center_xy = tuple(int(round(float(v))) for v in center)
                axes_xy = tuple(max(1, int(round(float(v) / 2.0))) for v in axes)
                cv2.ellipse(
                    overlay, center_xy, axes_xy, float(np.degrees(angle)), 0.0, 360.0,
                    color, 1, cv2.LINE_AA,
                )
        circle = geometry.get("image_circle_candidate_px")
        if isinstance(circle, Mapping):
            center = circle.get("center_xy")
            radius = circle.get("radius_px")
            if center is not None and isinstance(radius, (int, float)):
                center_xy = tuple(int(round(float(v))) for v in center)
                cv2.circle(overlay, center_xy, max(1, int(round(radius))), color, 1, cv2.LINE_AA)

        center = _feature_pixel_center(frame, feature, robot_state)
        if center is None:
            unprojected.append(label)
            continue
        x, y = center
        if not (0 <= x < width and 0 <= y < height):
            unprojected.append(label)
            continue
        cv2.drawMarker(
            overlay, center, color, markerType=cv2.MARKER_CROSS, markerSize=9,
            thickness=1, line_type=cv2.LINE_AA,
        )
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        label_offsets = ((True, -5), (True, 13), (False, -5), (False, 13), (True, -23), (False, -23))
        right_side, offset_y = label_offsets[(index - 1) % len(label_offsets)]
        offset_x = 6 if right_side else -text_size[0] - 8
        text_x = min(max(0, x + offset_x), max(0, width - text_size[0] - 2))
        text_y = min(
            max(text_size[1] + 2, y + offset_y),
            max(text_size[1] + 2, height - baseline - 1),
        )
        cv2.rectangle(
            overlay,
            (text_x, text_y - text_size[1] - 2),
            (text_x + text_size[0] + 2, text_y + baseline),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            overlay, label, (text_x + 1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
            0.42, color, 1, cv2.LINE_AA,
        )

    for row, label in enumerate(unprojected):
        text = f"{label}: no pixel projection"
        y = min(height - 3, 16 + 16 * row)
        cv2.putText(
            overlay, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (0, 180, 255), 1, cv2.LINE_AA,
        )
    if not cv2.imwrite(str(path), overlay):
        raise OSError(f"could not write public feature overlay: {path}")


def estimate_camera_features(
    frame: PublicRGBDFrame,
    *,
    observation_id: str,
    include_planes: bool = True,
    include_contours: bool = True,
) -> list[dict[str, Any]]:
    """Extract only image-derived features, excluding current robot-state features."""
    features = estimate_depth_features(frame, observation_id=observation_id, max_features=28)
    if include_planes:
        features.extend(estimate_plane_features(frame, observation_id=observation_id))
    if include_contours:
        features.extend(
            estimate_rgbd_contour_features(frame, observation_id=observation_id, max_features=32)
        )
    return features


def _rebind_camera_feature(
    raw_feature: Mapping[str, Any],
    *,
    frame: PublicRGBDFrame,
    observation_id: str,
    camera_observation_id: str,
) -> dict[str, Any]:
    feature = copy.deepcopy(dict(raw_feature))
    source_frame_id = feature.get("source_camera_frame_id")
    if source_frame_id is not None and source_frame_id != frame.frame_id:
        raise ValueError("cached camera feature belongs to a different camera frame")
    geometry = feature.get("geometry")
    if isinstance(geometry, Mapping) and "contact_surface_points_world_m" in geometry:
        contact_frame_id = geometry.get("contact_surface_frame_id")
        contact_timestamp_s = geometry.get("contact_surface_timestamp_s")
        if (
            isinstance(contact_frame_id, bool)
            or not isinstance(contact_frame_id, (int, np.integer))
            or contact_frame_id != frame.frame_id
            or isinstance(contact_timestamp_s, bool)
            or not isinstance(contact_timestamp_s, (int, float))
            or not math.isfinite(float(contact_timestamp_s))
            or contact_timestamp_s != frame.timestamp_s
        ):
            raise ValueError("cached contact surface does not match the camera frame and capture timestamp")
    old_source_id = feature.get("source_observation_id")
    old_ref = feature.get("feature_ref")
    if isinstance(old_ref, str):
        if (
            isinstance(old_source_id, str)
            and (old_ref == old_source_id or old_ref.startswith(old_source_id + ":"))
        ):
            feature["feature_ref"] = observation_id + old_ref[len(old_source_id):]
        else:
            geometry = feature.get("geometry")
            pixel_xy = geometry.get("pixel_centroid_xy") if isinstance(geometry, Mapping) else None
            if pixel_xy is None and isinstance(geometry, Mapping):
                circle = geometry.get("image_circle_candidate_px")
                if isinstance(circle, Mapping):
                    pixel_xy = circle.get("center_xy")
            if pixel_xy is None:
                raise ValueError("camera feature ref cannot be rebound to the current observation")
            feature["feature_ref"] = _fresh_feature_ref(
                observation_id, frame.frame_id, str(feature.get("kind", "camera_feature")), pixel_xy,
            )
    feature["source_observation_id"] = observation_id
    feature["source_camera_observation_id"] = camera_observation_id
    feature["source_camera_frame_id"] = frame.frame_id
    return feature


def build_public_observation(
    frame: PublicRGBDFrame,
    *,
    observation_id: str,
    rgb_path: str,
    depth_path: str,
    robot_state: Mapping[str, Any] | None = None,
    include_planes: bool = True,
    include_contours: bool = True,
    camera_features: Sequence[Mapping[str, Any]] | None = None,
    include_feature_overlay: bool = True,
) -> dict[str, Any]:
    """Package public camera paths/calibration and image-derived features.

    ``rgb_path`` is the model-visible image. ``depth_path`` points to the raw
    axial-z meter array for runtime planning; depth is not sent as an image.
    """
    if not observation_id:
        raise ValueError("observation_id must be non-empty")
    state = dict(robot_state or {})
    camera_observation_id = f"{observation_id}:camera:{frame.frame_id}"
    robot_timestamp_s = float(
        state.get("robot_timestamp_s", state.get("robot_state_timestamp_s", frame.timestamp_s))
    )
    state.update(
        {
            "camera_intrinsics": frame.K.tolist(),
            "camera_extrinsics": frame.T_world_camera.tolist(),
            "camera_depth_path": str(depth_path),
            "camera_depth_convention": "axial_z_m",
            "camera_frame_id": frame.frame_id,
            "camera_timestamp_s": frame.timestamp_s,
            "camera_observation_id": camera_observation_id,
            "rgbd_source_observation_id": camera_observation_id,
            "rgbd_source_timestamp_s": frame.timestamp_s,
            "robot_timestamp_s": robot_timestamp_s,
        }
    )
    if frame.pixel_convention is not None:
        state["camera_pixel_convention"] = frame.pixel_convention
    if camera_features is None:
        camera_features = estimate_camera_features(
            frame,
            observation_id=observation_id,
            include_planes=include_planes,
            include_contours=include_contours,
        )
    elif not isinstance(camera_features, Sequence) or isinstance(camera_features, (str, bytes)):
        raise TypeError("camera_features must be a sequence of camera-derived feature mappings")
    features = [
        _rebind_camera_feature(
            feature,
            frame=frame,
            observation_id=observation_id,
            camera_observation_id=camera_observation_id,
        )
        for feature in camera_features
    ]
    timestamp_key = "robot_timestamp_s" if "robot_timestamp_s" in state else "robot_state_timestamp_s"
    if {"tcp_pose", "gripper_axis_world", "gripper_width_m", timestamp_key} <= state.keys():
        features.append(
            robot_tcp_feature(
                observation_id=observation_id,
                camera_frame_id=frame.frame_id,
                tcp_pose_wxyz=state["tcp_pose"],
                finger_opening_axis_world=state["gripper_axis_world"],
                gripper_width_m=float(state["gripper_width_m"]),
                robot_timestamp_s=float(state[timestamp_key]),
            )
        )
    frames = {"task_camera": str(rgb_path)}
    rgb_file = Path(rgb_path) if isinstance(rgb_path, str) and not rgb_path.startswith("data:") else None
    if include_feature_overlay and rgb_file is not None and rgb_file.is_file():
        overlay_path = rgb_file.with_name(f"{rgb_file.stem}_feature_overlay.png")
        try:
            _draw_public_feature_overlay(frame, features, overlay_path, state)
        except ImportError:
            pass
        else:
            frames["task_camera_feature_overlay"] = str(overlay_path.resolve())
    return {
        "observation_id": observation_id,
        "timestamp": frame.timestamp_s,
        "frames": frames,
        "robot_state": state,
        "features": features,
    }


__all__ = [
    "FeatureTracker",
    "PublicRGBDFrame",
    "build_public_observation",
    "camera_capture_metadata",
    "camera_capture_timestamp_s",
    "estimate_feature_from_pixel_roi",
    "estimate_depth_features",
    "estimate_camera_features",
    "filter_robot_overlapping_contour_features",
    "estimate_plane_features",
    "estimate_rgbd_contour_features",
    "point_cloud_world",
    "public_rgbd_frame_from_camera",
    "rgbd_pointcloud_from_frame",
    "robot_tcp_feature",
]
