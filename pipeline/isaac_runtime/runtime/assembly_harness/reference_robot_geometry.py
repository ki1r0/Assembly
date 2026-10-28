"""Depth-consistent robot self-filtering from robot meshes and public camera data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .perception import PublicRGBDFrame


MAX_DEPTH_MATCH_TOLERANCE_M = 0.005
# shortcut: 1 mm is uncalibrated; widen only after live depth-residual calibration.
DEFAULT_DEPTH_MATCH_TOLERANCE_M = 0.001
MAX_ROBOT_CAMERA_SKEW_S = 1e-6


@dataclass(frozen=True)
class RobotLinkMesh:
    """A visible triangle mesh expressed in its rigid body's local frame, in meters."""

    link_name: str
    vertices_m: np.ndarray
    triangles: np.ndarray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices_m, dtype=np.float64)
        raw_triangles = np.asarray(self.triangles)
        if not np.issubdtype(raw_triangles.dtype, np.integer):
            raise ValueError("triangles must contain integer vertex indices")
        triangles = raw_triangles.astype(np.int64, copy=False)
        if not isinstance(self.link_name, str) or not self.link_name.strip():
            raise ValueError("link_name must be a non-empty string")
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
            raise ValueError("vertices_m must have shape (N, 3) with N >= 3")
        if not np.isfinite(vertices).all():
            raise ValueError("vertices_m must be finite")
        if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) < 1:
            raise ValueError("triangles must have shape (M, 3) with M >= 1")
        if triangles.min() < 0 or triangles.max() >= len(vertices):
            raise ValueError("triangle indices are outside vertices_m")
        object.__setattr__(self, "vertices_m", vertices.copy())
        object.__setattr__(self, "triangles", triangles.copy())


@dataclass(frozen=True)
class RobotSelfFilterResult:
    """RGB-D samples retained as obstacles after exact robot-surface matching."""

    points_world_m: np.ndarray
    pixel_xy: np.ndarray
    removed_sample_count: int
    matched_pixel_count: int
    robot_self_filtered: bool = True


def link_transforms_from_body_state(
    body_names: Sequence[str],
    body_positions_world_m: Any,
    body_quaternions_world_wxyz: Any,
) -> dict[str, np.ndarray]:
    """Build world-from-link transforms from one live articulation body's FK state."""
    names = tuple(str(name) for name in body_names)
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("body_names must be non-empty and unique")
    positions = _single_articulation(body_positions_world_m, (len(names), 3), "body positions")
    quaternions = _single_articulation(
        body_quaternions_world_wxyz, (len(names), 4), "body quaternions"
    )
    if not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
        raise ValueError("articulation body state must be finite")

    transforms: dict[str, np.ndarray] = {}
    for name, position, quaternion in zip(names, positions, quaternions, strict=True):
        rotation = _quat_wxyz_to_matrix(quaternion)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = position
        transforms[name] = transform
    return transforms


def visible_robot_meshes_from_usd(
    stage: Any,
    *,
    robot_root_path: str,
    body_names: Sequence[str],
) -> tuple[RobotLinkMesh, ...]:
    """Extract rendered USD triangles in each robot rigid body's local metric frame.

    The caller supplies only the robot root and articulation body names. Scene/task
    geometry is deliberately not traversed as a filtering input.
    """
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError as exc:
        raise RuntimeError("USD triangle extraction requires the Isaac/USD runtime") from exc

    root = stage.GetPrimAtPath(robot_root_path)
    if not root or not root.IsValid():
        raise RuntimeError("composed robot USD root is unavailable")
    names = tuple(str(name) for name in body_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("body_names must be non-empty and unique")

    bodies: dict[str, Any] = {}
    robot_prims = tuple(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
    for prim in robot_prims:
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName() in names:
            if prim.GetName() in bodies:
                raise RuntimeError(f"duplicate robot rigid body {prim.GetName()!r}")
            bodies[prim.GetName()] = prim
    if set(bodies) != set(names):
        raise RuntimeError("USD robot rigid bodies do not match the articulation bodies")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0.0:
        raise RuntimeError("robot USD stage has invalid metersPerUnit metadata")
    inverse_body = {
        name: xform_cache.GetLocalToWorldTransform(prim).GetInverse()
        for name, prim in bodies.items()
    }
    body_paths = {str(prim.GetPath()): name for name, prim in bodies.items()}
    meshes: list[RobotLinkMesh] = []
    triangles_by_link: set[str] = set()

    for prim in robot_prims:
        if not prim.IsA(UsdGeom.Mesh):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        if imageable.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
            continue

        owner = prim.GetParent()
        while owner and owner.IsValid() and str(owner.GetPath()) not in body_paths:
            owner = owner.GetParent()
        link_name = body_paths.get(str(owner.GetPath())) if owner and owner.IsValid() else None
        if link_name is None:
            continue

        usd_mesh = UsdGeom.Mesh(prim)
        points = usd_mesh.GetPointsAttr().Get(Usd.TimeCode.Default())
        face_counts = usd_mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default())
        face_indices = usd_mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.Default())
        if not points or not face_counts or not face_indices:
            continue
        mesh_to_world = xform_cache.GetLocalToWorldTransform(prim)
        body_from_world = inverse_body[link_name]
        vertices = np.empty((len(points), 3), dtype=np.float64)
        for index, point in enumerate(points):
            world_point = mesh_to_world.Transform(point)
            body_point = body_from_world.Transform(world_point)
            vertices[index] = meters_per_unit * np.asarray(
                (body_point[0], body_point[1], body_point[2]), dtype=np.float64
            )

        counts = np.asarray(face_counts, dtype=np.int64)
        indices = np.asarray(face_indices, dtype=np.int64)
        if np.any(counts < 3) or int(counts.sum()) != len(indices):
            raise RuntimeError(f"USD mesh {prim.GetPath()} has invalid polygon topology")
        triangles: list[tuple[int, int, int]] = []
        offset = 0
        for count_value in counts:
            count = int(count_value)
            polygon = indices[offset:offset + count]
            for index in range(1, count - 1):
                triangles.append((int(polygon[0]), int(polygon[index]), int(polygon[index + 1])))
            offset += count
        if triangles:
            meshes.append(RobotLinkMesh(link_name, vertices, np.asarray(triangles, dtype=np.int64)))
            triangles_by_link.add(link_name)

    missing = set(names) - triangles_by_link
    if missing:
        raise RuntimeError(f"visible robot USD triangle mesh missing for links: {sorted(missing)}")
    return tuple(meshes)


def render_robot_depth(
    meshes: Sequence[RobotLinkMesh],
    link_transforms_world: Mapping[str, Any],
    frame: PublicRGBDFrame,
    *,
    body_names: Sequence[str],
    near_m: float,
    far_m: float,
) -> np.ndarray:
    """Rasterize the robot's nearest visible axial-z mesh surface into camera pixels."""
    if not isinstance(frame, PublicRGBDFrame):
        raise TypeError("frame must be a PublicRGBDFrame")
    if not meshes:
        raise ValueError("at least one robot link mesh is required")
    near = _positive_finite(near_m, "near_m")
    far = _positive_finite(far_m, "far_m")
    if near >= far:
        raise ValueError("near_m must be less than far_m")
    names = _validated_body_names(body_names)
    transforms = _validated_link_transforms(meshes, link_transforms_world, names)

    height, width = frame.depth_m.shape
    rendered = np.full((height, width), np.nan, dtype=np.float32)
    camera_from_world = np.linalg.inv(frame.T_world_camera)
    intrinsics = frame.K
    if not np.allclose(intrinsics[2], (0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError("camera intrinsics must use the pinhole [0, 0, 1] bottom row")
    frustum_planes = np.asarray((
        (intrinsics[0, 0], intrinsics[0, 1], intrinsics[0, 2]),
        (intrinsics[0, 0], intrinsics[0, 1], intrinsics[0, 2] - (width - 1)),
        (intrinsics[1, 0], intrinsics[1, 1], intrinsics[1, 2]),
        (intrinsics[1, 0], intrinsics[1, 1], intrinsics[1, 2] - (height - 1)),
    ), dtype=np.float64)

    for mesh in meshes:
        camera_from_link = camera_from_world @ transforms[mesh.link_name]
        vertices_h = np.column_stack((mesh.vertices_m, np.ones(len(mesh.vertices_m))))
        vertices_camera = (camera_from_link @ vertices_h.T).T[:, :3]
        triangles_camera = vertices_camera[mesh.triangles]
        triangle_z = triangles_camera[:, :, 2]
        outside_depth = (triangle_z.max(axis=1) < near) | (triangle_z.min(axis=1) > far)
        plane_values = triangles_camera @ frustum_planes.T
        plane_tolerance = 16.0 * np.finfo(np.float64).eps * (
            np.abs(triangles_camera) @ np.abs(frustum_planes.T)
        )
        outside_image = (
            np.all(plane_values[:, :, 0] < -plane_tolerance[:, :, 0], axis=1)
            | np.all(plane_values[:, :, 1] > plane_tolerance[:, :, 1], axis=1)
            | np.all(plane_values[:, :, 2] < -plane_tolerance[:, :, 2], axis=1)
            | np.all(plane_values[:, :, 3] > plane_tolerance[:, :, 3], axis=1)
        )
        for triangle in triangles_camera[~(outside_depth | outside_image)]:
            polygon = _clip_z_polygon(triangle, near, far)
            if len(polygon) < 3:
                continue
            for index in range(1, len(polygon) - 1):
                _rasterize_triangle(rendered, np.asarray((polygon[0], polygon[index], polygon[index + 1])), intrinsics)
    return rendered


def filter_robot_surface_samples(
    frame: PublicRGBDFrame,
    cloud: Mapping[str, Any],
    meshes: Sequence[RobotLinkMesh],
    link_transforms_world: Mapping[str, Any],
    *,
    body_names: Sequence[str],
    camera_source_timestamp_s: float,
    robot_state_timestamp_s: float,
    camera_render_step_counter: int | None = None,
    robot_fk_step_counter: int | None = None,
    render_event_id: int | None = None,
    depth_tolerance_m: float,
    near_m: float,
    far_m: float,
    predicted_robot_depth_m: np.ndarray | None = None,
) -> RobotSelfFilterResult:
    """Remove only cloud samples whose measured depth matches predicted robot surface."""
    tolerance = _positive_finite(depth_tolerance_m, "depth_tolerance_m")
    if tolerance > MAX_DEPTH_MATCH_TOLERANCE_M:
        raise ValueError(
            f"depth_tolerance_m cannot exceed {MAX_DEPTH_MATCH_TOLERANCE_M:.3f} m"
        )
    source_timestamp = _finite_number(camera_source_timestamp_s, "camera_source_timestamp_s")
    state_timestamp = _finite_number(robot_state_timestamp_s, "robot_state_timestamp_s")
    if abs(source_timestamp - frame.timestamp_s) > MAX_ROBOT_CAMERA_SKEW_S:
        raise ValueError("frame timestamp must match its camera source timestamp")
    render_identity = (camera_render_step_counter, robot_fk_step_counter, render_event_id)
    if all(value is None for value in render_identity):
        if abs(state_timestamp - source_timestamp) > MAX_ROBOT_CAMERA_SKEW_S:
            raise ValueError("robot FK timestamp must match the RGB-D source timestamp")
    else:
        if any(value is None for value in render_identity):
            raise ValueError("camera/FK render identity must include both step counters and render event ID")
        identity_values = (camera_render_step_counter, robot_fk_step_counter, render_event_id)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0
            for value in identity_values
        ):
            raise ValueError("camera/FK render identity values must be non-negative integers")
        if camera_render_step_counter != robot_fk_step_counter:
            raise ValueError("robot FK native step must match the camera render step")
        if render_event_id == 0:
            raise ValueError("camera render event ID must be positive")
    points = np.asarray(cloud.get("points_world_m"), dtype=np.float64)
    pixels = np.asarray(cloud.get("pixel_xy"))
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("cloud points_world_m must be finite with shape (N, 3)")
    if pixels.shape != (len(points), 2) or not np.issubdtype(pixels.dtype, np.integer):
        raise ValueError("cloud pixel_xy must be integer pixels with shape (N, 2)")
    pixels = pixels.astype(np.int64, copy=False)
    height, width = frame.depth_m.shape
    if np.any(pixels[:, 0] < 0) or np.any(pixels[:, 0] >= width) or np.any(pixels[:, 1] < 0) or np.any(pixels[:, 1] >= height):
        raise ValueError("cloud pixel_xy contains out-of-frame coordinates")

    predicted = (
        render_robot_depth(
            meshes,
            link_transforms_world,
            frame,
            body_names=body_names,
            near_m=near_m,
            far_m=far_m,
        )
        if predicted_robot_depth_m is None
        else _validated_robot_depth(predicted_robot_depth_m, frame)
    )
    # Keep neighbor recovery in image space; advanced indexing would flatten the mask.
    matches = _matches_predicted_robot_depth(frame.depth_m, predicted, tolerance)
    sampled_matches = matches[pixels[:, 1], pixels[:, 0]]
    return RobotSelfFilterResult(
        points_world_m=points[~sampled_matches].copy(),
        pixel_xy=pixels[~sampled_matches].copy(),
        removed_sample_count=int(np.count_nonzero(sampled_matches)),
        matched_pixel_count=int(np.count_nonzero(sampled_matches)),
    )


def invalidate_robot_depth_matches(
    frame: PublicRGBDFrame,
    predicted_robot_depth_m: np.ndarray,
    *,
    depth_tolerance_m: float,
) -> np.ndarray:
    """Copy camera depth and invalidate only pixels matching the rendered robot surface."""
    tolerance = _positive_finite(depth_tolerance_m, "depth_tolerance_m")
    if tolerance > MAX_DEPTH_MATCH_TOLERANCE_M:
        raise ValueError(
            f"depth_tolerance_m cannot exceed {MAX_DEPTH_MATCH_TOLERANCE_M:.3f} m"
        )
    predicted = _validated_robot_depth(predicted_robot_depth_m, frame)
    observed = frame.depth_m
    matches = _matches_predicted_robot_depth(observed, predicted, tolerance)
    filtered = observed.copy()
    filtered[matches] = np.nan
    return filtered


def _matches_predicted_robot_depth(
    observed_depth_m: Any,
    predicted_depth_m: Any,
    tolerance_m: float,
) -> np.ndarray:
    observed = np.asarray(observed_depth_m, dtype=np.float32)
    predicted = np.asarray(predicted_depth_m, dtype=np.float32)
    if observed.shape != predicted.shape:
        raise ValueError("observed and predicted robot depth shapes must match")
    base_matches = (
        np.isfinite(observed)
        & np.isfinite(predicted)
        & (np.abs(observed - predicted) <= tolerance_m)
    )
    if observed.ndim != 2:
        return base_matches

    residual = np.abs(observed - predicted)
    max_recovery_residual = min(2.0 * float(tolerance_m), 0.002)
    # shortcut: isolated residuals within 2 mm use neighbor support, revisit after multi-view calibration.
    supported_samples = (
        np.isfinite(residual)
        & (residual > tolerance_m)
        & (residual <= max_recovery_residual)
    )

    # Recover only isolated samples surrounded by matching robot pixels.
    padded = np.pad(base_matches, 1, mode="constant", constant_values=False)
    neighbor_match_count = np.zeros(base_matches.shape, dtype=np.uint8)
    height, width = base_matches.shape
    for row_offset in (-1, 0, 1):
        for column_offset in (-1, 0, 1):
            if row_offset == 0 and column_offset == 0:
                continue
            row_start = 1 + row_offset
            column_start = 1 + column_offset
            neighbor_match_count += padded[
                row_start:row_start + height,
                column_start:column_start + width,
            ]
    return base_matches | (supported_samples & (neighbor_match_count >= 6))


def _validated_robot_depth(value: Any, frame: PublicRGBDFrame) -> np.ndarray:
    predicted = np.asarray(value, dtype=np.float32)
    if predicted.shape != frame.depth_m.shape:
        raise ValueError("predicted robot depth must match the RGB-D frame dimensions")
    if np.isinf(predicted).any() or np.any(np.isfinite(predicted) & (predicted <= 0.0)):
        raise ValueError("predicted robot depth must contain only positive depths or NaN")
    return predicted


def _single_articulation(value: Any, expected_shape: tuple[int, int], name: str) -> np.ndarray:
    array = _numpy(value)
    if array.shape == (1, *expected_shape):
        array = array[0]
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape} or (1, {expected_shape[0]}, {expected_shape[1]})")
    return np.asarray(array, dtype=np.float64)


def _validated_link_transforms(
    meshes: Sequence[RobotLinkMesh],
    transforms: Mapping[str, Any],
    body_names: set[str],
) -> dict[str, np.ndarray]:
    if any(not isinstance(mesh, RobotLinkMesh) for mesh in meshes):
        raise TypeError("meshes must contain only RobotLinkMesh values")
    mesh_links = {mesh.link_name for mesh in meshes}
    if mesh_links != body_names:
        raise ValueError("visible robot meshes must cover every articulation body")
    if not isinstance(transforms, Mapping) or set(transforms) != body_names:
        raise ValueError("link transforms must exactly cover articulation body names")
    result: dict[str, np.ndarray] = {}
    for name, raw in transforms.items():
        transform = np.asarray(raw, dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(f"link transform for {name!r} must be finite 4x4")
        if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
            raise ValueError(f"link transform for {name!r} must be homogeneous")
        rotation = transform[:3, :3]
        if (
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4)
        ):
            raise ValueError(f"link transform for {name!r} must have a right-handed rigid rotation")
        result[name] = transform
    return result


def _validated_body_names(body_names: Sequence[str]) -> set[str]:
    names = tuple(str(name) for name in body_names)
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("body_names must be non-empty and unique")
    return set(names)


def _quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        raise ValueError("body quaternion has zero norm")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _clip_z_polygon(vertices: np.ndarray, near: float, far: float) -> list[np.ndarray]:
    polygon = [vertex for vertex in vertices]
    for bound, keep_greater in ((near, True), (far, False)):
        if not polygon:
            break
        clipped: list[np.ndarray] = []
        previous = polygon[-1]
        previous_inside = previous[2] >= bound if keep_greater else previous[2] <= bound
        for current in polygon:
            current_inside = current[2] >= bound if keep_greater else current[2] <= bound
            if current_inside != previous_inside:
                denominator = current[2] - previous[2]
                if abs(denominator) > 1e-15:
                    fraction = (bound - previous[2]) / denominator
                    clipped.append(previous + fraction * (current - previous))
            if current_inside:
                clipped.append(current)
            previous, previous_inside = current, current_inside
        polygon = clipped
    return polygon


def _rasterize_triangle(depth: np.ndarray, triangle: np.ndarray, intrinsics: np.ndarray) -> None:
    z = triangle[:, 2]
    if np.any(z <= 0.0):
        return
    projected = np.column_stack((
        (intrinsics[0, 0] * triangle[:, 0] + intrinsics[0, 1] * triangle[:, 1]) / z
        + intrinsics[0, 2],
        (intrinsics[1, 0] * triangle[:, 0] + intrinsics[1, 1] * triangle[:, 1]) / z
        + intrinsics[1, 2],
    ))
    area = _edge(projected[0], projected[1], projected[2])
    if abs(area) < 1e-12:
        return

    height, width = depth.shape
    min_x = max(0, int(math.ceil(float(projected[:, 0].min()))))
    max_x = min(width - 1, int(math.floor(float(projected[:, 0].max()))))
    min_y = max(0, int(math.ceil(float(projected[:, 1].min()))))
    max_y = min(height - 1, int(math.floor(float(projected[:, 1].max()))))
    if min_x > max_x or min_y > max_y:
        return

    xs = np.arange(min_x, max_x + 1, dtype=np.float64)
    ys = np.arange(min_y, max_y + 1, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    point = np.stack((grid_x, grid_y), axis=-1)
    weight0 = _edge(projected[1], projected[2], point) / area
    weight1 = _edge(projected[2], projected[0], point) / area
    weight2 = 1.0 - weight0 - weight1
    inside = (weight0 >= -1e-9) & (weight1 >= -1e-9) & (weight2 >= -1e-9)
    reciprocal_z = weight0 / z[0] + weight1 / z[1] + weight2 / z[2]
    inside &= reciprocal_z > 0.0
    if not np.any(inside):
        return

    patch = depth[min_y:max_y + 1, min_x:max_x + 1]
    candidate = np.full(patch.shape, np.inf, dtype=np.float64)
    candidate[inside] = 1.0 / reciprocal_z[inside]
    existing = np.where(np.isfinite(patch), patch, np.inf)
    update = candidate < existing
    patch[update] = candidate[update].astype(np.float32)


def _edge(first: np.ndarray, second: np.ndarray, point: np.ndarray) -> np.ndarray:
    return (second[0] - first[0]) * (point[..., 1] - first[1]) - (second[1] - first[1]) * (point[..., 0] - first[0])


def _positive_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _finite_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _numpy(value: Any) -> np.ndarray:
    if callable(getattr(value, "detach", None)):
        value = value.detach()
    if callable(getattr(value, "cpu", None)):
        value = value.cpu()
    if callable(getattr(value, "numpy", None)):
        value = value.numpy()
    return np.asarray(value)


__all__ = [
    "MAX_DEPTH_MATCH_TOLERANCE_M",
    "MAX_ROBOT_CAMERA_SKEW_S",
    "RobotLinkMesh",
    "RobotSelfFilterResult",
    "filter_robot_surface_samples",
    "invalidate_robot_depth_matches",
    "link_transforms_from_body_state",
    "render_robot_depth",
    "visible_robot_meshes_from_usd",
]
