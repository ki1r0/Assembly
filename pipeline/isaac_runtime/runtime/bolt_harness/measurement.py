"""Private bolt pose and filtered contact measurements for the bolt evaluator.

The pose path computes a conservative analytic interference bound from audited
CAD envelopes. It is not an exact mesh/SDF distance or proof of contact support.
Raw PhysX contact separation remains a separate diagnostic measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, hypot, isfinite, sqrt
from typing import Mapping, Sequence

from .contacts import BOLT_ACTOR_PATH, CAP_COLLIDER_PATH, FINGER_ACTOR_PATHS, BoltFingerContact
from .evaluator import (
    BoltNativePoseSample,
    BoltPoseWindowTolerances,
    BoltSeatMeasurement,
    BoltSeatStatus,
    BoltSeatTolerances,
    IndependentBoltSeatingEvaluator,
)
from settings import CAD_METERS_PER_SOURCE_UNIT


Vector3 = tuple[float, float, float]
QuaternionWxyz = tuple[float, float, float, float]

CAD_STAGE_SCALE_M = CAD_METERS_PER_SOURCE_UNIT
ASSEMBLY_FRAME_POS_M = (0.50, -0.15, 0.68 + 27.948326 * CAD_STAGE_SCALE_M)
# Casing_Top/socket_bolt_hub_4; this is the ring's lowest small hole in the
# calibrated task-camera view, not a target exposed to the public actor.
SELECTED_HOLE_XY_M = (
    ASSEMBLY_FRAME_POS_M[0] - 39.9973999954978 * CAD_STAGE_SCALE_M,
    ASSEMBLY_FRAME_POS_M[1] + 80.1321976578251 * CAD_STAGE_SCALE_M,
)
CAP_SUPPORT_PLANE_Z_M = ASSEMBLY_FRAME_POS_M[2] + 37.94832611083985 * CAD_STAGE_SCALE_M
CASING_ENTRY_SOURCE_Y = 27.948326
CASING_ENTRY_PLANE_Z_M = ASSEMBLY_FRAME_POS_M[2] + CASING_ENTRY_SOURCE_Y * CAD_STAGE_SCALE_M
CASING_INTERIOR_FLOOR_SOURCE_Z = 6.94815826416015
CASING_INTERIOR_FLOOR_Z_M = (
    ASSEMBLY_FRAME_POS_M[2] + CASING_INTERIOR_FLOOR_SOURCE_Z * CAD_STAGE_SCALE_M
)
BOLT_TIP_LOCAL_Y_SOURCE = -13.07499885559082
CAP_UNDERSIDE_LOCAL_Y_SOURCE = 8.92500114440918
BOLT_TOP_LOCAL_Y_SOURCE = 13.07499885559082
BOLT_TIP_LOCAL_Y_M = BOLT_TIP_LOCAL_Y_SOURCE * CAD_STAGE_SCALE_M
CAP_UNDERSIDE_LOCAL_Y_M = CAP_UNDERSIDE_LOCAL_Y_SOURCE * CAD_STAGE_SCALE_M
BOLT_TOP_LOCAL_Y_M = BOLT_TOP_LOCAL_Y_SOURCE * CAD_STAGE_SCALE_M
CAP_BEARING_RADIUS_SOURCE_UNITS = 5.000001
CAP_BEARING_RADIUS_M = CAP_BEARING_RADIUS_SOURCE_UNITS * CAD_STAGE_SCALE_M
# Source-audited radii in source units. The shaft value is the measured maximum
# radial extent, conservatively above the profile p95 used by the feasibility audit.
SHANK_RADIUS_SOURCE_UNITS = 2.9420009968738574
CASING_MIN_RADIUS_SOURCE_UNITS = 3.1109647137472605
COVER_MIN_RADIUS_SOURCE_UNITS = 3.6479270507992583
SHANK_RADIUS_M = SHANK_RADIUS_SOURCE_UNITS * CAD_STAGE_SCALE_M
CAP_OUTER_RADIUS_M = 5.773503303527832 * CAD_STAGE_SCALE_M
COVER_MIN_RADIUS_M = COVER_MIN_RADIUS_SOURCE_UNITS * CAD_STAGE_SCALE_M
CASING_MIN_RADIUS_M = CASING_MIN_RADIUS_SOURCE_UNITS * CAD_STAGE_SCALE_M
COVER_RADIAL_CLEARANCE_M = COVER_MIN_RADIUS_M - SHANK_RADIUS_M
CASING_RADIAL_CLEARANCE_M = CASING_MIN_RADIUS_M - SHANK_RADIUS_M


def _finite_tuple(values: Sequence[float], size: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size or any(not isfinite(value) for value in result):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


@dataclass(frozen=True)
class BoltSeatGeometryConfig:
    """Task-local geometry values used by the bolt-seat evaluator."""

    selected_hole_xy_m: tuple[float, float] = SELECTED_HOLE_XY_M
    cap_support_plane_z_m: float = CAP_SUPPORT_PLANE_Z_M
    casing_entry_plane_z_m: float = CASING_ENTRY_PLANE_Z_M
    casing_interior_floor_z_m: float = CASING_INTERIOR_FLOOR_Z_M
    assembly_frame_pos_m: tuple[float, float, float] = ASSEMBLY_FRAME_POS_M
    cover_aperture_radius_m: float = COVER_MIN_RADIUS_M
    casing_aperture_radius_m: float = CASING_MIN_RADIUS_M

    def __post_init__(self) -> None:
        _finite_tuple(self.selected_hole_xy_m, 2, "selected_hole_xy_m")
        _finite_tuple(self.assembly_frame_pos_m, 3, "assembly_frame_pos_m")
        scalars = (
            self.cap_support_plane_z_m,
            self.casing_entry_plane_z_m,
            self.casing_interior_floor_z_m,
            self.cover_aperture_radius_m,
            self.casing_aperture_radius_m,
        )
        if any(not isfinite(float(value)) for value in scalars):
            raise ValueError("bolt-seat task geometry values must be finite")
        if self.cover_aperture_radius_m <= 0.0 or self.casing_aperture_radius_m <= 0.0:
            raise ValueError("bolt-seat aperture radii must be positive")


def _unit_quaternion(values: Sequence[float]) -> QuaternionWxyz:
    quaternion = _finite_tuple(values, 4, "bolt_root_quat_wxyz")
    norm = sqrt(sum(value * value for value in quaternion))
    if norm == 0.0:
        raise ValueError("bolt_root_quat_wxyz must be non-zero")
    return tuple(value / norm for value in quaternion)  # type: ignore[return-value]


def _quaternion_angular_distance_rad(
    first: Sequence[float], second: Sequence[float]
) -> float:
    a = _unit_quaternion(first)
    b = _unit_quaternion(second)
    if sum(a[i] * b[i] for i in range(4)) < 0.0:
        b = tuple(-value for value in b)  # type: ignore[assignment]
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    relative = (
        aw * bw + ax * bx + ay * by + az * bz,
        aw * bx - ax * bw - ay * bz + az * by,
        aw * by + ax * bz - ay * bw - az * bx,
        aw * bz - ax * by + ay * bx - az * bw,
    )
    return 2.0 * atan2(sqrt(sum(value * value for value in relative[1:])), abs(relative[0]))


def _rotate(q: QuaternionWxyz, v: Vector3) -> Vector3:
    w, x, y, z = q
    qv = (x, y, z)
    cross = (
        qv[1] * v[2] - qv[2] * v[1],
        qv[2] * v[0] - qv[0] * v[2],
        qv[0] * v[1] - qv[1] * v[0],
    )
    twice_cross = tuple(2.0 * value for value in cross)
    second_cross = (
        qv[1] * twice_cross[2] - qv[2] * twice_cross[1],
        qv[2] * twice_cross[0] - qv[0] * twice_cross[2],
        qv[0] * twice_cross[1] - qv[1] * twice_cross[0],
    )
    return tuple(v[i] + w * twice_cross[i] + second_cross[i] for i in range(3))  # type: ignore[return-value]


def _add_scaled(point: Vector3, direction: Vector3, scale: float) -> Vector3:
    return tuple(point[i] + direction[i] * scale for i in range(3))  # type: ignore[return-value]


def _distance_xy(point: Vector3, center_xy: tuple[float, float]) -> float:
    return hypot(point[0] - center_xy[0], point[1] - center_xy[1])


def _shaft_aperture_interference_bound(
    root_pos: Vector3,
    axis: Vector3,
    plane_z: float,
    aperture_center_xy: tuple[float, float],
    aperture_radius_m: float,
) -> tuple[float | None, float, bool]:
    """Bound local shaft interference at an audited aperture plane.

    A tilted circular cylinder cuts a horizontal plane as an ellipse whose major
    radius is ``shaft_radius / abs(axis_z)``. Its enclosing circle plus axis
    offset gives a conservative radial-overlap bound. The result is applicable
    only where the shaft section overlaps the selected aperture neighborhood;
    distant scene geometry is outside this selected-socket metric. This checks
    critical entry planes only, not an exact mesh sweep through plate thickness.
    """
    if abs(axis[2]) <= 1e-12:
        return None, 0.0, False
    axial_coordinate = (plane_z - root_pos[2]) / axis[2]
    crossing = _add_scaled(root_pos, axis, axial_coordinate)
    radial_error = _distance_xy(crossing, aperture_center_xy)
    if not BOLT_TIP_LOCAL_Y_M <= axial_coordinate <= CAP_UNDERSIDE_LOCAL_Y_M:
        return radial_error, 0.0, False
    projected_radius = SHANK_RADIUS_M / abs(axis[2])
    if radial_error > aperture_radius_m + projected_radius:
        return radial_error, 0.0, False
    interference = max(0.0, radial_error + projected_radius - aperture_radius_m)
    return radial_error, interference, True


@dataclass(frozen=True)
class BoltPoseGeometry:
    """Private pose facts and conservative analytic CAD-envelope bounds.

    Penetration fields are upper bounds local to the selected aperture/bearing
    patch at audited critical planes, not exact mesh/SDF distances. Each
    applicability flag distinguishes an unmeasured distant region from zero
    interference. Contact evidence is recorded separately.
    """

    time_s: float
    bolt_root_pos_w_m: Vector3
    bolt_root_quat_wxyz: QuaternionWxyz
    bolt_axis_w: Vector3
    bolt_tip_pos_w_m: Vector3
    cap_underside_center_w_m: Vector3
    cover_plane_radial_error_m: float | None
    casing_entry_radial_error_m: float | None
    cover_shaft_penetration_applicable: bool
    casing_shaft_penetration_applicable: bool
    cap_bearing_patch_applicable: bool
    cover_shaft_penetration_bound_m: float
    casing_shaft_penetration_bound_m: float
    cap_head_penetration_bound_m: float
    geometric_penetration_bound_m: float
    shaft_radial_error_m: float
    through_selected_cover_hole: bool
    entered_selected_casing_opening: bool
    casing_entry_depth_m: float
    cap_support_gap_m: float
    tip_to_interior_floor_clearance_m: float
    premature_bottoming: bool
    axial_position_m: float
    relative_linear_speed_mps: float
    relative_angular_speed_radps: float


@dataclass(frozen=True)
class BoltNativePoseWindowDiagnostic:
    """Private comparison of native velocities with measured 120 Hz pose changes.

    This is diagnostic evidence only. It does not replace calibrated evaluator
    speed limits or project any private metric to the executor.
    """

    sample_count: int
    duration_s: float
    position_bbox_diagonal_span_m: float
    max_orientation_deviation_from_first_rad: float
    max_pose_fd_linear_speed_mps: float
    max_pose_fd_angular_speed_radps: float
    max_native_linear_speed_mps: float
    max_native_angular_speed_radps: float


def diagnose_native_pose_window(
    samples: Sequence[Mapping[str, object]],
) -> BoltNativePoseWindowDiagnostic:
    """Compare finite-difference motion against native velocity in a private trace.

    Each sample must use ``RigidBodyView.get_transforms`` layout ``x,y,z,qx,qy,qz,qw``
    and ``get_velocities`` layout ``vx,vy,vz,wx,wy,wz``. Position span is the
    diagonal of the sample-window axis-aligned bounds; orientation deviation is
    measured from the first sample. No stability threshold is implied.
    """
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise TypeError("native pose samples must be a sequence")
    if len(samples) < 2:
        raise ValueError("native pose diagnosis requires at least two physics samples")

    times: list[float] = []
    positions: list[Vector3] = []
    quaternions: list[QuaternionWxyz] = []
    native_linear_speeds: list[float] = []
    native_angular_speeds: list[float] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise TypeError(f"native pose sample {index} must be a mapping")
        time_s = _scalar(sample["sim_timestamp_s"])
        transform = _finite_tuple(sample["native_root_transform_raw"], 7, "native transform")
        velocity = _finite_tuple(sample["native_root_velocity_raw"], 6, "native velocity")
        if not isfinite(time_s):
            raise ValueError("native pose sample timestamps must be finite")
        times.append(time_s)
        positions.append((transform[0], transform[1], transform[2]))
        quaternions.append(_unit_quaternion((transform[6], transform[3], transform[4], transform[5])))
        native_linear_speeds.append(sqrt(sum(value * value for value in velocity[:3])))
        native_angular_speeds.append(sqrt(sum(value * value for value in velocity[3:])))

    position_min = tuple(min(point[axis] for point in positions) for axis in range(3))
    position_max = tuple(max(point[axis] for point in positions) for axis in range(3))
    position_span = sqrt(sum((position_max[i] - position_min[i]) ** 2 for i in range(3)))
    reference_quaternion = quaternions[0]
    orientation_deviation = max(
        _quaternion_angular_distance_rad(reference_quaternion, quat)
        for quat in quaternions
    )

    max_pose_linear_speed = 0.0
    max_pose_angular_speed = 0.0
    for index in range(1, len(samples)):
        delta_s = times[index] - times[index - 1]
        if delta_s <= 0.0:
            raise ValueError("native pose sample times must increase strictly")
        position_delta = sqrt(
            sum((positions[index][axis] - positions[index - 1][axis]) ** 2 for axis in range(3))
        )
        orientation_delta = _quaternion_angular_distance_rad(
            quaternions[index - 1], quaternions[index]
        )
        max_pose_linear_speed = max(max_pose_linear_speed, position_delta / delta_s)
        max_pose_angular_speed = max(max_pose_angular_speed, orientation_delta / delta_s)

    return BoltNativePoseWindowDiagnostic(
        sample_count=len(samples),
        duration_s=times[-1] - times[0],
        position_bbox_diagonal_span_m=position_span,
        max_orientation_deviation_from_first_rad=orientation_deviation,
        max_pose_fd_linear_speed_mps=max_pose_linear_speed,
        max_pose_fd_angular_speed_radps=max_pose_angular_speed,
        max_native_linear_speed_mps=max(native_linear_speeds),
        max_native_angular_speed_radps=max(native_angular_speeds),
    )


@dataclass(frozen=True)
class ContactPointReport:
    """One raw signed scalar and normal from a filtered RigidContactView contact."""

    force_n: float
    point_w_m: Vector3
    normal_w: Vector3
    separation_m: float

    def __post_init__(self) -> None:
        if len(self.point_w_m) != 3 or len(self.normal_w) != 3:
            raise ValueError("contact point and normal must be 3D vectors")
        if not isfinite(self.force_n):
            raise ValueError(f"raw scalar contact force must be finite; got {self.force_n!r}")
        if any(not isfinite(value) for value in (*self.point_w_m, *self.normal_w, self.separation_m)):
            raise ValueError("contact point, normal, and separation must be finite")
        if sqrt(sum(value * value for value in self.normal_w)) == 0.0:
            raise ValueError("contact normal must be non-zero")

    @property
    def force_vector_w_n(self) -> Vector3:
        """Return the signed world vector specified by the tensor API: scalar * normal."""
        return tuple(self.force_n * value for value in self.normal_w)  # type: ignore[return-value]

    @property
    def force_magnitude_n(self) -> float:
        return sqrt(sum(value * value for value in self.force_vector_w_n))


@dataclass(frozen=True)
class CapSupportContactCriteria:
    """Calibrated cap patch and directed scalar-times-normal support limits.

    ``normal_axis_sign`` selects the expected sign of the resulting force vector
    along the bolt axis; it does not constrain the raw scalar's sign.
    """

    max_cap_patch_error_m: float
    max_contact_separation_m: float
    min_normal_alignment_cosine: float
    normal_axis_sign: int
    min_contact_force_n: float

    def __post_init__(self) -> None:
        if (
            not isfinite(self.max_cap_patch_error_m)
            or self.max_cap_patch_error_m < 0.0
            or not isfinite(self.max_contact_separation_m)
            or not isfinite(self.min_normal_alignment_cosine)
            or not 0.0 < self.min_normal_alignment_cosine <= 1.0
            or self.normal_axis_sign not in (-1, 1)
            or not isfinite(self.min_contact_force_n)
            or self.min_contact_force_n <= 0.0
        ):
            raise ValueError("cap contact criteria must be finite calibrated limits")


@dataclass(frozen=True)
class BoltSeatSamplingCriteria:
    """Required adapter calibration for grasp, robot support, and motion state."""

    cap_support: CapSupportContactCriteria
    min_cap_grasp_local_y_m: float
    max_cap_grasp_patch_error_m: float
    max_cap_grasp_contact_separation_m: float
    min_cap_grasp_force_n: float
    max_cap_grasp_normal_alignment_cosine: float
    max_robot_contact_separation_m: float
    min_robot_contact_force_n: float
    max_controller_tcp_linear_speed_mps: float
    max_controller_tcp_angular_speed_radps: float
    max_controller_arm_joint_speed_radps: float
    max_tcp_tracking_error_m: float
    max_tcp_orientation_tracking_error_rad: float
    release_gripper_width_m: float
    min_pickup_root_lift_m: float

    def __post_init__(self) -> None:
        non_negative = (
            self.max_cap_grasp_patch_error_m,
            self.max_robot_contact_separation_m,
            self.max_controller_tcp_linear_speed_mps,
            self.max_controller_tcp_angular_speed_radps,
            self.max_controller_arm_joint_speed_radps,
            self.max_tcp_tracking_error_m,
            self.max_tcp_orientation_tracking_error_rad,
        )
        if any(not isfinite(value) or value < 0.0 for value in non_negative):
            raise ValueError("adapter tolerances must be finite and non-negative")
        if not isfinite(self.min_cap_grasp_local_y_m) or not (
            CAP_UNDERSIDE_LOCAL_Y_M <= self.min_cap_grasp_local_y_m <= BOLT_TOP_LOCAL_Y_M
        ):
            raise ValueError("cap grasp y limit must lie within the CAD cap bounds")
        if not isfinite(self.max_cap_grasp_contact_separation_m):
            raise ValueError("cap grasp separation limit must be finite")
        if not isfinite(self.min_cap_grasp_force_n) or self.min_cap_grasp_force_n <= 0.0:
            raise ValueError("minimum cap grasp force must be calibrated and positive")
        if (
            not isfinite(self.max_cap_grasp_normal_alignment_cosine)
            or not 0.0 <= self.max_cap_grasp_normal_alignment_cosine < 1.0
        ):
            raise ValueError("cap grasp normals must have a calibrated non-axial alignment limit")
        if not isfinite(self.min_robot_contact_force_n) or self.min_robot_contact_force_n <= 0.0:
            raise ValueError("minimum robot contact force must be calibrated and positive")
        if not isfinite(self.release_gripper_width_m) or self.release_gripper_width_m <= 0.0:
            raise ValueError("release gripper width must be calibrated and positive")
        if not isfinite(self.min_pickup_root_lift_m) or self.min_pickup_root_lift_m <= 0.0:
            raise ValueError("minimum pickup root lift must be calibrated and positive")


def measure_bolt_pose_geometry(
    *,
    time_s: float,
    bolt_root_pos_w_m: Sequence[float],
    bolt_root_quat_wxyz: Sequence[float],
    bolt_root_lin_vel_w_mps: Sequence[float],
    bolt_root_ang_vel_w_radps: Sequence[float],
    env_origin_w_m: Sequence[float],
    geometry_config: BoltSeatGeometryConfig | None = None,
) -> BoltPoseGeometry:
    """Measure the selected bolt path from its source-local +Y axis and root pose.

    World-space CAD targets are offset by ``env_origin_w_m`` for cloned scenes.
    ``cap_support_gap_m`` is a signed reference-plane gap, not contact evidence.
    """
    root = _finite_tuple(bolt_root_pos_w_m, 3, "bolt_root_pos_w_m")
    env_origin = _finite_tuple(env_origin_w_m, 3, "env_origin_w_m")
    quat = _unit_quaternion(bolt_root_quat_wxyz)
    linear_velocity = _finite_tuple(bolt_root_lin_vel_w_mps, 3, "bolt_root_lin_vel_w_mps")
    angular_velocity = _finite_tuple(bolt_root_ang_vel_w_radps, 3, "bolt_root_ang_vel_w_radps")
    if not isfinite(time_s):
        raise ValueError("time_s must be finite")
    task_geometry = geometry_config or BoltSeatGeometryConfig()
    if not isinstance(task_geometry, BoltSeatGeometryConfig):
        raise TypeError("geometry_config must be BoltSeatGeometryConfig")

    axis = _rotate(quat, (0.0, 1.0, 0.0))
    tip = _add_scaled(root, axis, BOLT_TIP_LOCAL_Y_M)
    cap_center = _add_scaled(root, axis, CAP_UNDERSIDE_LOCAL_Y_M)
    hole_xy = (
        env_origin[0] + task_geometry.selected_hole_xy_m[0],
        env_origin[1] + task_geometry.selected_hole_xy_m[1],
    )
    support_z = env_origin[2] + task_geometry.cap_support_plane_z_m
    entry_z = env_origin[2] + task_geometry.casing_entry_plane_z_m
    floor_z = env_origin[2] + task_geometry.casing_interior_floor_z_m
    cover_clearance = max(0.0, task_geometry.cover_aperture_radius_m - SHANK_RADIUS_M)
    casing_clearance = max(0.0, task_geometry.casing_aperture_radius_m - SHANK_RADIUS_M)
    cover_error, cover_interference, cover_interference_applicable = _shaft_aperture_interference_bound(
        root, axis, support_z, hole_xy, task_geometry.cover_aperture_radius_m
    )
    entry_error, casing_interference, casing_interference_applicable = _shaft_aperture_interference_bound(
        root, axis, entry_z, hole_xy, task_geometry.casing_aperture_radius_m
    )
    tip_error = _distance_xy(tip, hole_xy)
    cap_error = _distance_xy(cap_center, hole_xy)
    shaft_error = max(
        tip_error,
        cap_error,
        *(error for error in (cover_error, entry_error) if error is not None),
    )
    entry_depth = entry_z - tip[2]
    aligned_for_insertion = axis[2] > 0.0
    # The overall head radius conservatively bounds the underside bevel while
    # the support-contact classifier uses the smaller audited flat-face radius.
    # This finite-radius neighborhood is only a local selected-opening bound;
    # it does not claim collision distance against the rest of the cover mesh.
    cap_patch_radius_m = CAP_OUTER_RADIUS_M + task_geometry.cover_aperture_radius_m
    cap_bearing_patch_applicable = _distance_xy(cap_center, hole_xy) <= cap_patch_radius_m
    lowest_cap_envelope_z = cap_center[2] - CAP_OUTER_RADIUS_M * sqrt(
        max(0.0, 1.0 - axis[2] * axis[2])
    )
    cap_interference = (
        max(0.0, support_z - lowest_cap_envelope_z)
        if cap_bearing_patch_applicable
        else 0.0
    )
    penetration_bound = max(cover_interference, casing_interference, cap_interference)
    through_cover = (
        aligned_for_insertion
        and tip[2] < support_z
        and cover_error is not None
        and cover_error <= cover_clearance
        and cover_interference == 0.0
    )
    entered_casing = (
        aligned_for_insertion
        and entry_depth > 0.0
        and entry_error is not None
        and entry_error <= casing_clearance
        and casing_interference == 0.0
    )
    tip_floor_clearance = tip[2] - floor_z

    return BoltPoseGeometry(
        time_s=float(time_s),
        bolt_root_pos_w_m=root,  # type: ignore[arg-type]
        bolt_root_quat_wxyz=quat,
        bolt_axis_w=axis,
        bolt_tip_pos_w_m=tip,
        cap_underside_center_w_m=cap_center,
        cover_plane_radial_error_m=cover_error,
        casing_entry_radial_error_m=entry_error,
        cover_shaft_penetration_applicable=cover_interference_applicable,
        casing_shaft_penetration_applicable=casing_interference_applicable,
        cap_bearing_patch_applicable=cap_bearing_patch_applicable,
        cover_shaft_penetration_bound_m=cover_interference,
        casing_shaft_penetration_bound_m=casing_interference,
        cap_head_penetration_bound_m=cap_interference,
        geometric_penetration_bound_m=penetration_bound,
        shaft_radial_error_m=shaft_error,
        through_selected_cover_hole=through_cover,
        entered_selected_casing_opening=entered_casing,
        casing_entry_depth_m=entry_depth,
        cap_support_gap_m=cap_center[2] - support_z,
        tip_to_interior_floor_clearance_m=tip_floor_clearance,
        premature_bottoming=entered_casing and tip_floor_clearance <= 0.0,
        axial_position_m=root[2] - (env_origin[2] + task_geometry.assembly_frame_pos_m[2]),
        relative_linear_speed_mps=sqrt(sum(value * value for value in linear_velocity)),
        relative_angular_speed_radps=sqrt(sum(value * value for value in angular_velocity)),
    )


def cap_support_contact(
    geometry: BoltPoseGeometry,
    reports: Sequence[ContactPointReport],
    criteria: CapSupportContactCriteria,
) -> bool:
    """Confirm directed bolt-cover support on the cap underside.

    ``reports`` must come from a contact view filtered specifically to the Cover,
    not from aggregate wrist force or the unfiltered sensor net-force tensor.
    """
    inverse_quat = (
        geometry.bolt_root_quat_wxyz[0],
        -geometry.bolt_root_quat_wxyz[1],
        -geometry.bolt_root_quat_wxyz[2],
        -geometry.bolt_root_quat_wxyz[3],
    )
    axial_support_force_n = 0.0
    for report in reports:
        if report.separation_m > criteria.max_contact_separation_m:
            continue
        local_point = _rotate(
            inverse_quat,
            tuple(report.point_w_m[i] - geometry.bolt_root_pos_w_m[i] for i in range(3)),
        )
        radius = hypot(local_point[0], local_point[2])
        radial_error = max(SHANK_RADIUS_M - radius, 0.0, radius - CAP_BEARING_RADIUS_M)
        patch_error = hypot(local_point[1] - CAP_UNDERSIDE_LOCAL_Y_M, radial_error)
        if patch_error > criteria.max_cap_patch_error_m:
            continue
        force_vector = report.force_vector_w_n
        force_norm = report.force_magnitude_n
        if force_norm == 0.0:
            continue
        support_component = criteria.normal_axis_sign * sum(
            force_vector[i] * geometry.bolt_axis_w[i] for i in range(3)
        )
        alignment = support_component / force_norm
        if alignment >= criteria.min_normal_alignment_cosine:
            axial_support_force_n += support_component
    return axial_support_force_n >= criteria.min_contact_force_n


def read_all_filtered_contact_reports(contact_physx_view, *, dt_s: float, sensor_index: int):
    """Aggregate all explicit fixture collider filters for one measured body."""
    count = int(contact_physx_view.filter_count)
    if count < 1:
        raise ValueError('private fixture contact view has no resolved filters')
    return tuple(report for index in range(count)
                 for report in read_filtered_contact_reports(
                     contact_physx_view, dt_s=dt_s, sensor_index=sensor_index, filter_index=index))


def read_filtered_contact_reports(
    contact_physx_view: object,
    *,
    dt_s: float,
    sensor_index: int,
    filter_index: int,
) -> tuple[ContactPointReport, ...]:
    """Read raw per-contact data from IsaacLab's filtered RigidContactView API."""
    if not isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    if sensor_index < 0 or filter_index < 0:
        raise ValueError("sensor and filter indices must be non-negative")
    buffers = contact_physx_view.get_contact_data(dt_s)
    return _reports_from_contact_buffers(buffers, sensor_index, filter_index)


def _reports_from_contact_buffers(
    buffers: tuple[object, object, object, object, object, object],
    sensor_index: int,
    filter_index: int,
) -> tuple[ContactPointReport, ...]:
    forces, points, normals, separations, counts, starts = buffers
    count_value = _scalar(_pair_item(counts, sensor_index, filter_index))
    start_value = _scalar(_pair_item(starts, sensor_index, filter_index))
    if (
        not isfinite(count_value)
        or not isfinite(start_value)
        or count_value != int(count_value)
        or start_value != int(start_value)
    ):
        raise ValueError("contact buffer count and start must be finite integers")
    count = int(count_value)
    start = int(start_value)
    if count < 0 or start < 0:
        raise ValueError("contact buffer count and start must be non-negative")
    if count == 0:
        return ()

    lengths = (len(forces), len(points), len(normals), len(separations))
    if any(start + count > length for length in lengths):
        raise ValueError(
            "contact buffer count/start exceed reported data arrays: "
            f"sensor_index={sensor_index}, filter_index={filter_index}, "
            f"count={count}, start={start}, lengths={lengths}"
        )

    reports = []
    for index in range(start, start + count):
        force_row = _row(forces, index)
        if len(force_row) != 1:
            raise ValueError(
                "expected one raw scalar force per contact: "
                f"sensor_index={sensor_index}, filter_index={filter_index}, "
                f"point_index={index}, force_raw={force_row!r}"
            )
        force_raw = force_row[0]
        point_raw = _row(points, index)
        normal_raw = _row(normals, index)
        separation_row = _row(separations, index)
        if len(separation_row) != 1:
            raise ValueError(
                "expected one raw scalar separation per contact: "
                f"sensor_index={sensor_index}, filter_index={filter_index}, "
                f"point_index={index}, separation_raw={separation_row!r}"
            )
        try:
            reports.append(
                ContactPointReport(
                    force_n=force_raw,
                    point_w_m=point_raw,
                    normal_w=normal_raw,
                    separation_m=separation_row[0],
                )
            )
        except ValueError as exc:
            raise ValueError(
                "invalid raw private contact record: "
                f"sensor_index={sensor_index}, filter_index={filter_index}, "
                f"point_index={index}, force_n_raw={force_raw!r}, "
                f"point_w_raw={point_raw!r}, normal_w_raw={normal_raw!r}, "
                f"separation_raw={separation_row[0]!r}"
            ) from exc
    return tuple(reports)


def _scalar(value: object) -> float:
    if callable(getattr(value, "detach", None)):
        value = value.detach()
    if callable(getattr(value, "cpu", None)):
        value = value.cpu()
    if callable(getattr(value, "numpy", None)):
        value = value.numpy()
    if callable(getattr(value, "item", None)):
        value = value.item()
    return float(value)


def _pair_item(values: object, first: int, second: int) -> object:
    try:
        return values[first, second]
    except (TypeError, IndexError):
        return values[first][second]


def _row(values: object, index: int) -> tuple[float, ...]:
    row = values[index]
    if callable(getattr(row, "reshape", None)):
        row = row.reshape(-1)
    if callable(getattr(row, "detach", None)):
        row = row.detach()
    if callable(getattr(row, "cpu", None)):
        row = row.cpu()
    if callable(getattr(row, "numpy", None)):
        row = row.numpy()
    if callable(getattr(row, "tolist", None)):
        row = row.tolist()
    if not isinstance(row, (tuple, list)):
        row = (row,)
    return tuple(float(value) for value in row)


@dataclass(frozen=True)
class _BoltSeatTraceSample:
    time_s: float
    contact_point_count: int
    contact_data_capacity: int
    geometry: BoltPoseGeometry
    measurement: BoltSeatMeasurement
    status: BoltSeatStatus
    cover_contacts: tuple[ContactPointReport, ...]
    casing_contacts: tuple[ContactPointReport, ...]
    robot_contacts: tuple[ContactPointReport, ...]
    robot_contacts_by_source: tuple[tuple[str, tuple[ContactPointReport, ...]], ...]
    left_grasp_contacts: tuple[ContactPointReport, ...]
    right_grasp_contacts: tuple[ContactPointReport, ...]
    exact_left_cap_grasp_contacts: tuple[BoltFingerContact, ...]
    exact_right_cap_grasp_contacts: tuple[BoltFingerContact, ...]
    exact_left_cap_grasp_contact_batches: tuple[tuple[BoltFingerContact, ...], ...]
    exact_right_cap_grasp_contact_batches: tuple[tuple[BoltFingerContact, ...], ...]
    left_finger_force_n: float
    right_finger_force_n: float
    bilateral_contact: bool
    held_seat_certified: bool
    left_cap_grasp: bool
    right_cap_grasp: bool
    release_started: bool
    pickup_root_lift_m: float
    pickup_proven: bool
    pickup_sequence_valid: bool
    held_inserted_sample_count: int


class BoltSeatSamplingAdapter:
    """Read-only bridge from the private IsaacLab scene to the temporal evaluator.

    ``contact_source`` is the native mapping returned by
        ``env.get_private_bolt_contact_source()``. Its filter map is authoritative;
        left/right cap-grasp evidence is taken from Robot filters whose source prims
        identify the corresponding fingers. An exact provider returns per-side
        sequences of per-physics-tick contact batches so manifold points are summed
        within a tick, never across ticks. The private trace retains measurements
        and raw contacts; ``observe`` returns only the two boolean status fields.
    Pose-window mode additionally requires ``native_pose_sample_provider``,
    ``native_pose_dt_s``, and explicit ``pose_window_tolerances`` in the private
    contact source. The provider consumes completed native root-pose samples since
    its previous call; it must not synthesize or substitute commanded poses.
    """

    _CATEGORIES = ("cover", "casing", "robot")

    def __init__(
        self,
        env: object,
        tolerances: BoltSeatTolerances,
        criteria: BoltSeatSamplingCriteria,
        *,
        contact_source: Mapping[str, object],
        geometry_config: BoltSeatGeometryConfig | None = None,
    ) -> None:
        if not isinstance(tolerances, BoltSeatTolerances):
            raise TypeError("tolerances must be BoltSeatTolerances")
        if not isinstance(criteria, BoltSeatSamplingCriteria):
            raise TypeError("criteria must be BoltSeatSamplingCriteria")
        if getattr(env, "num_envs", 1) != 1:
            raise ValueError("BoltSeatSamplingAdapter currently supports the single-env task only")
        self._env = env
        self._tolerances = tolerances
        self._criteria = criteria
        self._geometry_config = geometry_config or BoltSeatGeometryConfig()
        if not isinstance(self._geometry_config, BoltSeatGeometryConfig):
            raise TypeError("geometry_config must be BoltSeatGeometryConfig")
        native_pose_provider = contact_source.get("native_pose_sample_provider")
        if native_pose_provider is not None and not callable(native_pose_provider):
            raise TypeError("native_pose_sample_provider must be callable")
        pose_tolerance_source = contact_source.get("pose_window_tolerances")
        if (pose_tolerance_source is not None) != (native_pose_provider is not None):
            raise ValueError(
                "pose-window tolerances and the private native pose provider must be configured together"
            )
        self._pose_window_tolerances: BoltPoseWindowTolerances | None = None
        if pose_tolerance_source is not None:
            if not isinstance(pose_tolerance_source, Mapping):
                raise TypeError("pose_window_tolerances must be a private calibrated mapping")
            expected_pose_fields = {
                "max_position_excursion_m",
                "max_orientation_excursion_rad",
            }
            if set(pose_tolerance_source) != expected_pose_fields:
                raise ValueError("pose_window_tolerances must provide exactly its two calibrated limits")
            self._pose_window_tolerances = BoltPoseWindowTolerances(
                **dict(pose_tolerance_source)
            )
        self._native_pose_sample_provider = native_pose_provider
        self._native_pose_dt_s: float | None = None
        if native_pose_provider is not None:
            if contact_source.get("native_pose_dt_s") is None:
                raise ValueError("pose-window mode requires calibrated native_pose_dt_s")
            self._native_pose_dt_s = _scalar(contact_source["native_pose_dt_s"])
            if not isfinite(self._native_pose_dt_s) or self._native_pose_dt_s <= 0.0:
                raise ValueError("native_pose_dt_s must be finite and positive")
        exact_provider = contact_source.get("exact_cap_finger_contact_provider")
        if exact_provider is not None and not callable(exact_provider):
            raise TypeError("exact_cap_finger_contact_provider must be callable")
        self._exact_cap_finger_contact_provider = exact_provider
        (
            self._contact_view,
            self._contact_dt_s,
            self._contact_data_capacity,
            self._filter_indices,
            self._filter_map,
        ) = self._parse_contact_source(contact_source)
        self._evaluator = IndependentBoltSeatingEvaluator(
            tolerances, pose_window_tolerances=self._pose_window_tolerances
        )
        self._trace: list[_BoltSeatTraceSample] = []
        self._initial_sample_seen = False
        self._initially_unheld = False
        self._initial_root_z_m: float | None = None
        self._pickup_lift_observed = False
        self._pickup_held_streak = 0
        self._pickup_proven = False
        self._pickup_sequence_valid = True
        self._held_before_selected_passage = False
        self._held_inserted_sample_count = 0
        self._ever_cap_grasped = False
        self._held_seat_certified = False
        self._release_started = False

    def checkpoint_progression(self) -> dict[str, object]:
        """Return semantic insertion progress, excluding temporal/native windows."""
        return {
            "initial_sample_seen": self._initial_sample_seen,
            "initially_unheld": self._initially_unheld,
            "initial_root_z_m": self._initial_root_z_m,
            "pickup_lift_observed": self._pickup_lift_observed,
            "pickup_held_streak": self._pickup_held_streak,
            "pickup_proven": self._pickup_proven,
            "pickup_sequence_valid": self._pickup_sequence_valid,
            "held_before_selected_passage": self._held_before_selected_passage,
            "held_inserted_sample_count": self._held_inserted_sample_count,
            "ever_cap_grasped": self._ever_cap_grasped,
            "held_seat_certified": self._held_seat_certified,
        }

    def restore_checkpoint_progression(self, state: Mapping[str, object]) -> None:
        """Restore only semantic progress; fresh observations rebuild all windows."""
        expected = {
            "initial_sample_seen", "initially_unheld", "initial_root_z_m",
            "pickup_lift_observed", "pickup_held_streak", "pickup_proven",
            "pickup_sequence_valid", "held_before_selected_passage",
            "held_inserted_sample_count", "ever_cap_grasped", "held_seat_certified",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("checkpoint semantic progression has an invalid shape")
        for name in (
            "initial_sample_seen", "initially_unheld", "pickup_lift_observed",
            "pickup_proven", "pickup_sequence_valid", "held_before_selected_passage",
            "ever_cap_grasped", "held_seat_certified",
        ):
            if type(state[name]) is not bool:
                raise ValueError(f"checkpoint semantic progression {name} must be boolean")
        for name in ("pickup_held_streak", "held_inserted_sample_count"):
            if isinstance(state[name], bool) or not isinstance(state[name], int) or state[name] < 0:
                raise ValueError(f"checkpoint semantic progression {name} must be non-negative")
        root_z = state["initial_root_z_m"]
        if root_z is not None and (
            isinstance(root_z, bool) or not isinstance(root_z, (int, float)) or not isfinite(float(root_z))
        ):
            raise ValueError("checkpoint semantic progression initial_root_z_m must be finite or null")
        if state["initial_sample_seen"] and root_z is None:
            raise ValueError("checkpoint semantic progression needs initial_root_z_m after its first sample")
        self._initial_sample_seen = state["initial_sample_seen"]
        self._initially_unheld = state["initially_unheld"]
        self._initial_root_z_m = None if root_z is None else float(root_z)
        self._pickup_lift_observed = state["pickup_lift_observed"]
        self._pickup_held_streak = state["pickup_held_streak"]
        self._pickup_proven = state["pickup_proven"]
        self._pickup_sequence_valid = state["pickup_sequence_valid"]
        self._held_before_selected_passage = state["held_before_selected_passage"]
        self._held_inserted_sample_count = state["held_inserted_sample_count"]
        self._ever_cap_grasped = state["ever_cap_grasped"]
        self._held_seat_certified = state["held_seat_certified"]

    def _read_native_pose_samples(
        self, sample_time_s: float
    ) -> tuple[BoltNativePoseSample, ...] | None:
        provider = self._native_pose_sample_provider
        if provider is None:
            return None
        raw_batch = provider()
        if not isinstance(raw_batch, Sequence) or isinstance(raw_batch, (str, bytes)):
            raise TypeError("native_pose_sample_provider must return per-tick sample mappings")
        batch: list[BoltNativePoseSample] = []
        for raw in raw_batch:
            if isinstance(raw, BoltNativePoseSample):
                pose_sample = raw
            elif isinstance(raw, Mapping):
                transform = _finite_tuple(
                    raw["native_root_transform_raw"], 7, "native_root_transform_raw"
                )
                velocity = _finite_tuple(
                    raw["native_root_velocity_raw"], 6, "native_root_velocity_raw"
                )
                pose_sample = BoltNativePoseSample(
                    physics_step_index=raw["physics_step_index"],
                    time_s=_scalar(raw["sim_timestamp_s"]),
                    root_pos_w_m=transform[:3],
                    root_quat_wxyz=(transform[6], transform[3], transform[4], transform[5]),
                    native_linear_velocity_w_mps=velocity[:3],
                    native_angular_velocity_w_radps=velocity[3:],
                )
            else:
                raise TypeError(
                    "native pose provider entries must be raw mappings or BoltNativePoseSample"
                )
            if pose_sample.time_s > sample_time_s + max(1e-6, self._native_pose_dt_s * 1e-3):
                raise ValueError("native pose provider returned a future physics sample")
            batch.append(pose_sample)
        return tuple(batch)

    @staticmethod
    def _parse_contact_source(
        source: Mapping[str, object],
    ) -> tuple[
        object,
        float,
        int,
        dict[str, tuple[int, ...]],
        dict[int, dict[str, str]],
    ]:
        if not isinstance(source, Mapping):
            raise TypeError("contact_source must be the mapping returned by the environment")
        view = source.get("contact_physx_view")
        if not callable(getattr(view, "get_contact_data", None)):
            raise TypeError("contact_source must contain contact_physx_view.get_contact_data(dt)")
        if source.get("dt_s") is None:
            raise ValueError("contact_source must declare dt_s")
        dt_s = _scalar(source["dt_s"])
        if not isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("contact_source dt_s must be finite and positive")
        capacity = source.get("contact_data_capacity", source.get("capacity"))
        if capacity is None:
            raise ValueError("contact_source must declare positive contact-data capacity")
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError("contact_source must declare positive contact-data capacity")
        filter_map = source.get("filter_map")
        if not isinstance(filter_map, Sequence) or isinstance(filter_map, (str, bytes)):
            raise TypeError("contact_source filter_map must be a sequence of filter records")

        records: dict[int, dict[str, str]] = {}
        category_indices = {category: [] for category in BoltSeatSamplingAdapter._CATEGORIES}
        left_finger_indices, right_finger_indices = [], []
        for item in filter_map:
            if not isinstance(item, Mapping):
                raise TypeError("filter_map entries must be mappings")
            index = int(item["filter_index"])
            category = str(item["category"]).casefold()
            filter_path = str(item.get("filter_prim_path", ""))
            source_path = str(item["source_prim_path"])
            if index < 0 or index in records:
                raise ValueError("filter_map indices must be unique and non-negative")
            if category not in category_indices:
                raise ValueError(f"unsupported contact filter category: {category!r}")
            if not filter_path or not source_path:
                raise ValueError("each contact filter must retain filter and source prim paths")
            path_folded = source_path.casefold()
            if category == "cover" and "/cover/" not in path_folded:
                raise ValueError(f"Cover category does not resolve below Cover: {source_path!r}")
            if category == "casing" and "/casing/" not in path_folded:
                raise ValueError(f"Casing category does not resolve below Casing: {source_path!r}")
            if category == "robot" and "/robot/" not in path_folded:
                raise ValueError(f"Robot category does not resolve below Robot: {source_path!r}")
            records[index] = {
                "category": category,
                "filter_prim_path": filter_path,
                "source_prim_path": source_path,
            }
            category_indices[category].append(index)
            if category == "robot":
                if "leftfinger" in path_folded:
                    left_finger_indices.append(index)
                if "rightfinger" in path_folded:
                    right_finger_indices.append(index)

        view_count = getattr(view, "filter_count", None)
        expected_count = int(view_count) if view_count is not None else len(records)
        if not records or set(records) != set(range(expected_count)):
            raise ValueError("filter_map must describe every resolved PhysX filter index exactly once")
        if any(not category_indices[category] for category in BoltSeatSamplingAdapter._CATEGORIES):
            raise ValueError("contact_source requires Cover, Casing, and Robot filter categories")
        if not left_finger_indices or not right_finger_indices:
            raise ValueError("Robot filter_map must identify both leftfinger and rightfinger source prims")
        indices = {
            **{category: tuple(values) for category, values in category_indices.items()},
            "left_grasp": tuple(left_finger_indices),
            "right_grasp": tuple(right_finger_indices),
        }
        return view, dt_s, capacity, indices, records

    def observe(self) -> BoltSeatStatus:
        """Sample the current scene, update the evaluator, and return booleans only."""
        env = self._env
        env_index = 0

        time_s = _scalar(env._robot._data._sim_timestamp)
        native_pose_samples = self._read_native_pose_samples(time_s)
        scene_origin = _row(env.scene.env_origins, env_index)
        bolt_data = env._bolt.data
        geometry = measure_bolt_pose_geometry(
            time_s=time_s,
            bolt_root_pos_w_m=_row(bolt_data.root_pos_w, env_index),
            bolt_root_quat_wxyz=_row(bolt_data.root_quat_w, env_index),
            bolt_root_lin_vel_w_mps=_row(bolt_data.root_lin_vel_w, env_index),
            bolt_root_ang_vel_w_radps=_row(bolt_data.root_ang_vel_w, env_index),
            env_origin_w_m=scene_origin,
            geometry_config=self._geometry_config,
        )

        bolt_buffers = self._contact_view.get_contact_data(self._contact_dt_s)
        contact_point_count = _contact_point_count(
            bolt_buffers, env_index, len(self._filter_map)
        )
        if contact_point_count >= self._contact_data_capacity:
            raise RuntimeError(
                "private bolt contact buffer reached configured capacity; "
                "measurement may be truncated and cannot be evaluated"
            )
        cover_contacts = _reports_for_filters(
            bolt_buffers, env_index, self._filter_indices["cover"]
        )
        casing_contacts = _reports_for_filters(
            bolt_buffers, env_index, self._filter_indices["casing"]
        )
        robot_contacts_by_source = tuple(
            (
                self._filter_map[index]["source_prim_path"],
                _reports_for_filters(bolt_buffers, env_index, (index,)),
            )
            for index in self._filter_indices["robot"]
        )
        robot_contacts = tuple(
            report for _, reports in robot_contacts_by_source for report in reports
        )
        left_contacts = _reports_for_filters(
            bolt_buffers, env_index, self._filter_indices["left_grasp"]
        )
        right_contacts = _reports_for_filters(
            bolt_buffers, env_index, self._filter_indices["right_grasp"]
        )

        exact_left_batches: tuple[tuple[BoltFingerContact, ...], ...] = ()
        exact_right_batches: tuple[tuple[BoltFingerContact, ...], ...] = ()
        exact_left_contacts: tuple[BoltFingerContact, ...] = ()
        exact_right_contacts: tuple[BoltFingerContact, ...] = ()
        if self._exact_cap_finger_contact_provider is None:
            left_cap_grasp = _cap_grasp_contact(geometry, left_contacts, self._criteria)
            right_cap_grasp = _cap_grasp_contact(geometry, right_contacts, self._criteria)
        else:
            exact_reports = self._exact_cap_finger_contact_provider()
            if not isinstance(exact_reports, Mapping):
                raise TypeError("exact cap contact provider must return a left/right mapping")
            exact_left_batches = _exact_cap_finger_contact_batches(
                exact_reports.get("left"), "left"
            )
            exact_right_batches = _exact_cap_finger_contact_batches(
                exact_reports.get("right"), "right"
            )
            exact_left_contacts = tuple(
                contact for batch in exact_left_batches for contact in batch
            )
            exact_right_contacts = tuple(
                contact for batch in exact_right_batches for contact in batch
            )
            left_cap_grasp = _exact_cap_grasp_contact_batches(
                geometry, exact_left_batches, self._criteria, expected_finger="left"
            )
            right_cap_grasp = _exact_cap_grasp_contact_batches(
                geometry, exact_right_batches, self._criteria, expected_finger="right"
            )

        grasp_state = env.get_bilateral_grasp_contact()
        left_bilateral = _bool_at(grasp_state["left_contact"], env_index)
        right_bilateral = _bool_at(grasp_state["right_contact"], env_index)
        bilateral = left_bilateral and right_bilateral and _bool_at(
            grasp_state["bilateral_contact"], env_index
        )
        left_force = _scalar_at(grasp_state["left_force_n"], env_index)
        right_force = _scalar_at(grasp_state["right_force_n"], env_index)
        public_state = env.get_public_executor_state()
        gripper_width = _scalar(public_state["gripper_width_m"])
        if bilateral and left_cap_grasp and right_cap_grasp:
            self._ever_cap_grasped = True
        raw_physically_held = bilateral and left_cap_grasp and right_cap_grasp
        if self._ever_cap_grasped and self._held_seat_certified and (
            not raw_physically_held
            or gripper_width >= self._criteria.release_gripper_width_m
        ):
            self._release_started = True
        physically_held = (
            raw_physically_held and not self._release_started
        )
        if not self._initial_sample_seen:
            self._initial_sample_seen = True
            self._initially_unheld = not physically_held
            self._initial_root_z_m = geometry.bolt_root_pos_w_m[2]
        pickup_root_lift_m = geometry.bolt_root_pos_w_m[2] - self._initial_root_z_m
        if physically_held and self._initially_unheld and self._pickup_sequence_valid:
            if pickup_root_lift_m >= self._criteria.min_pickup_root_lift_m:
                self._pickup_lift_observed = True
            if self._pickup_lift_observed:
                self._pickup_held_streak += 1
                if self._pickup_held_streak >= 2:
                    self._pickup_proven = True
            else:
                self._pickup_held_streak = 0
        elif self._pickup_proven and not physically_held and not self._release_started:
            self._pickup_sequence_valid = False
            self._pickup_held_streak = 0
        else:
            self._pickup_held_streak = 0
        if physically_held:
            if not (
                geometry.through_selected_cover_hole
                and geometry.entered_selected_casing_opening
            ):
                self._held_before_selected_passage = True
            elif self._pickup_proven and self._pickup_sequence_valid and self._held_before_selected_passage:
                self._held_inserted_sample_count += 1
        insertion_provenance_valid = (
            self._initially_unheld
            and self._pickup_proven
            and self._pickup_sequence_valid
            and self._held_inserted_sample_count > 0
        )

        controller_stopped = _controller_stopped(public_state, self._criteria)
        robot_contact_or_support = _has_robot_contact(robot_contacts, self._criteria)
        measurement = BoltSeatMeasurement(
            time_s=time_s,
            through_selected_cover_hole=geometry.through_selected_cover_hole,
            entered_selected_casing_opening=geometry.entered_selected_casing_opening,
            casing_entry_depth_m=geometry.casing_entry_depth_m,
            shaft_radial_error_m=geometry.shaft_radial_error_m,
            cap_support_gap_m=geometry.cap_support_gap_m,
            cap_support_contact=cap_support_contact(
                geometry, cover_contacts, self._criteria.cap_support
            ),
            penetration_m=geometry.geometric_penetration_bound_m,
            premature_bottoming=geometry.premature_bottoming,
            physically_held=physically_held,
            insertion_provenance_valid=insertion_provenance_valid,
            controller_stopped=controller_stopped,
            robot_contact_or_support=robot_contact_or_support,
            axial_position_m=geometry.axial_position_m,
            relative_linear_speed_mps=geometry.relative_linear_speed_mps,
            relative_angular_speed_radps=geometry.relative_angular_speed_radps,
            native_pose_samples=native_pose_samples,
            native_pose_dt_s=self._native_pose_dt_s if native_pose_samples is not None else None,
        )
        status = self._evaluator.observe(measurement)
        if status.seat_ready:
            self._held_seat_certified = True
        self._trace.append(
            _BoltSeatTraceSample(
                time_s=time_s,
                contact_point_count=contact_point_count,
                contact_data_capacity=self._contact_data_capacity,
                geometry=geometry,
                measurement=measurement,
                status=status,
                cover_contacts=cover_contacts,
                casing_contacts=casing_contacts,
                robot_contacts=robot_contacts,
                robot_contacts_by_source=robot_contacts_by_source,
                left_grasp_contacts=left_contacts,
                right_grasp_contacts=right_contacts,
                exact_left_cap_grasp_contacts=exact_left_contacts,
                exact_right_cap_grasp_contacts=exact_right_contacts,
                exact_left_cap_grasp_contact_batches=exact_left_batches,
                exact_right_cap_grasp_contact_batches=exact_right_batches,
                left_finger_force_n=left_force,
                right_finger_force_n=right_force,
                bilateral_contact=bilateral,
                held_seat_certified=self._held_seat_certified,
                left_cap_grasp=left_cap_grasp,
                right_cap_grasp=right_cap_grasp,
                release_started=self._release_started,
                pickup_root_lift_m=pickup_root_lift_m,
                pickup_proven=self._pickup_proven,
                pickup_sequence_valid=self._pickup_sequence_valid,
                held_inserted_sample_count=self._held_inserted_sample_count,
            )
        )
        return status


def _reports_for_filters(
    buffers: tuple[object, object, object, object, object, object],
    sensor_index: int,
    filter_indices: Sequence[int],
) -> tuple[ContactPointReport, ...]:
    return tuple(
        report
        for filter_index in filter_indices
        for report in _reports_from_contact_buffers(buffers, sensor_index, filter_index)
    )


def _contact_point_count(
    buffers: tuple[object, object, object, object, object, object],
    sensor_index: int,
    filter_count: int,
) -> int:
    counts = buffers[4]
    return sum(
        int(_scalar(_pair_item(counts, sensor_index, index)))
        for index in range(filter_count)
    )


def _scalar_at(values: object, index: int) -> float:
    row = _row(values, index)
    if len(row) != 1:
        raise ValueError("expected one scalar per environment")
    return row[0]


def _bool_at(values: object, index: int) -> bool:
    value = values[index]
    if callable(getattr(value, "item", None)):
        value = value.item()
    return bool(value)


def _cap_grasp_contact(
    geometry: BoltPoseGeometry,
    reports: Sequence[ContactPointReport],
    criteria: BoltSeatSamplingCriteria,
) -> bool:
    inverse_quat = (
        geometry.bolt_root_quat_wxyz[0],
        -geometry.bolt_root_quat_wxyz[1],
        -geometry.bolt_root_quat_wxyz[2],
        -geometry.bolt_root_quat_wxyz[3],
    )
    inward_contact_force_n = 0.0
    for report in reports:
        if report.separation_m > criteria.max_cap_grasp_contact_separation_m:
            continue
        normal_norm = sqrt(sum(value * value for value in report.normal_w))
        axial_normal_alignment = abs(
            sum(report.normal_w[i] * geometry.bolt_axis_w[i] for i in range(3))
            / normal_norm
        )
        if axial_normal_alignment > criteria.max_cap_grasp_normal_alignment_cosine:
            continue
        local = _rotate(
            inverse_quat,
            tuple(report.point_w_m[i] - geometry.bolt_root_pos_w_m[i] for i in range(3)),
        )
        radius = hypot(local[0], local[2])
        axial_error = max(
            criteria.min_cap_grasp_local_y_m - local[1],
            0.0,
            local[1] - BOLT_TOP_LOCAL_Y_M,
        )
        radial_error = max(radius - CAP_OUTER_RADIUS_M, 0.0)
        if hypot(axial_error, radial_error) <= criteria.max_cap_grasp_patch_error_m:
            point_offset = tuple(
                report.point_w_m[i] - geometry.bolt_root_pos_w_m[i] for i in range(3)
            )
            axial_offset = sum(point_offset[i] * geometry.bolt_axis_w[i] for i in range(3))
            radial = tuple(
                point_offset[i] - axial_offset * geometry.bolt_axis_w[i] for i in range(3)
            )
            radial_norm = sqrt(sum(value * value for value in radial))
            if radial_norm == 0.0:
                continue
            inward_force = -sum(
                report.force_vector_w_n[i] * radial[i] / radial_norm for i in range(3)
            )
            if inward_force > 0.0:
                inward_contact_force_n += inward_force
    return inward_contact_force_n >= criteria.min_cap_grasp_force_n


def _exact_cap_finger_contact_batches(
    value: object, side: str
) -> tuple[tuple[BoltFingerContact, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"exact cap contact provider must return per-tick batches for {side}")
    batches = tuple(value)
    if any(isinstance(item, BoltFingerContact) for item in batches):
        raise TypeError(
            f"exact cap contact provider must preserve per-physics-tick batches for {side}"
        )
    normalized = []
    for batch in batches:
        if not isinstance(batch, Sequence) or isinstance(batch, (str, bytes)):
            raise TypeError(f"exact cap contact provider returned a malformed batch for {side}")
        contacts = tuple(batch)
        if any(not isinstance(contact, BoltFingerContact) for contact in contacts):
            raise TypeError(
                f"exact cap contact provider returned a non-BoltFingerContact for {side}"
            )
        normalized.append(contacts)
    return tuple(normalized)


def _exact_cap_grasp_contact_batches(
    geometry: BoltPoseGeometry,
    batches: Sequence[Sequence[BoltFingerContact]],
    criteria: BoltSeatSamplingCriteria,
    *,
    expected_finger: str,
) -> bool:
    return any(
        _exact_cap_grasp_contact(
            geometry, batch, criteria, expected_finger=expected_finger
        )
        for batch in batches
    )


def _exact_cap_grasp_contact(
    geometry: BoltPoseGeometry,
    reports: Sequence[BoltFingerContact],
    criteria: BoltSeatSamplingCriteria,
    *,
    expected_finger: str,
) -> bool:
    """Classify cap grasp from exact collider pairs, without root-pose patch projection."""
    if expected_finger not in FINGER_ACTOR_PATHS:
        raise ValueError(f"unsupported finger side: {expected_finger!r}")
    finger_actor_path = FINGER_ACTOR_PATHS[expected_finger]
    compression_force_n = 0.0
    for report in reports:
        if report.finger != expected_finger:
            continue
        if report.actor0_path == BOLT_ACTOR_PATH:
            exact_pair = (
                report.actor1_path == finger_actor_path
                and report.collider0_path == CAP_COLLIDER_PATH
            )
        elif report.actor1_path == BOLT_ACTOR_PATH:
            exact_pair = (
                report.actor0_path == finger_actor_path
                and report.collider1_path == CAP_COLLIDER_PATH
            )
        else:
            exact_pair = False
        if not exact_pair:
            continue
        if report.raw_separation_m > criteria.max_cap_grasp_contact_separation_m:
            continue
        normal = report.normal_on_bolt_w
        normal_norm = sqrt(sum(value * value for value in normal))
        if normal_norm == 0.0:
            continue
        axial_normal_alignment = abs(
            sum(normal[i] * geometry.bolt_axis_w[i] for i in range(3)) / normal_norm
        )
        if axial_normal_alignment > criteria.max_cap_grasp_normal_alignment_cosine:
            continue
        compression_n = sum(
            report.force_on_bolt_w_n[i] * normal[i] / normal_norm for i in range(3)
        )
        if not isfinite(compression_n):
            raise ValueError("exact cap contact compression must be finite")
        if compression_n > 0.0:
            compression_force_n += compression_n
    return compression_force_n >= criteria.min_cap_grasp_force_n


def _has_robot_contact(
    reports: Sequence[ContactPointReport], criteria: BoltSeatSamplingCriteria
) -> bool:
    return any(
        report.force_magnitude_n >= criteria.min_robot_contact_force_n
        and report.separation_m <= criteria.max_robot_contact_separation_m
        for report in reports
    )


def _controller_stopped(public_state: Mapping[str, object], criteria: BoltSeatSamplingCriteria) -> bool:
    tcp_velocity = _row(public_state["tcp_velocity"], 0)
    joint_velocity = _row(public_state["joint_velocities"], 0)
    if len(tcp_velocity) != 6 or len(joint_velocity) < 7:
        raise ValueError("public executor state has malformed TCP/joint velocity vectors")
    tcp_linear_speed = sqrt(sum(value * value for value in tcp_velocity[:3]))
    tcp_angular_speed = sqrt(sum(value * value for value in tcp_velocity[3:]))
    arm_joint_speed = max(abs(value) for value in joint_velocity[:7])
    return (
        bool(public_state["motion_safety_armed"])
        and tcp_linear_speed <= criteria.max_controller_tcp_linear_speed_mps
        and tcp_angular_speed <= criteria.max_controller_tcp_angular_speed_radps
        and arm_joint_speed <= criteria.max_controller_arm_joint_speed_radps
        and _scalar_at(public_state["tcp_tracking_error_m"], 0)
        <= criteria.max_tcp_tracking_error_m
        and _scalar_at(public_state["tcp_orientation_tracking_error_rad"], 0)
        <= criteria.max_tcp_orientation_tracking_error_rad
    )


__all__ = [
    "BoltSeatGeometryConfig",
    "BoltPoseGeometry",
    "BoltNativePoseSample",
    "BoltNativePoseWindowDiagnostic",
    "CapSupportContactCriteria",
    "BoltSeatSamplingAdapter",
    "BoltSeatSamplingCriteria",
    "ContactPointReport",
    "cap_support_contact",
    "diagnose_native_pose_window",
    "measure_bolt_pose_geometry",
    "read_filtered_contact_reports",
]
