"""Temporal bolt seating evaluation from simulator-side measurements only."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, isfinite, sqrt


@dataclass(frozen=True)
class BoltSeatTolerances:
    """Required calibration limits; axial motion is measured from the last held seat."""

    max_cap_support_gap_m: float
    min_casing_entry_depth_m: float
    max_shaft_radial_error_m: float
    max_penetration_m: float
    max_axial_motion_since_release_m: float
    max_relative_linear_speed_mps: float
    max_relative_angular_speed_radps: float
    held_seat_dwell_s: float
    stable_dwell_s: float
    max_sample_gap_s: float

    def __post_init__(self) -> None:
        values = (
            self.max_cap_support_gap_m,
            self.min_casing_entry_depth_m,
            self.max_shaft_radial_error_m,
            self.max_penetration_m,
            self.max_axial_motion_since_release_m,
            self.max_relative_linear_speed_mps,
            self.max_relative_angular_speed_radps,
            self.held_seat_dwell_s,
            self.stable_dwell_s,
            self.max_sample_gap_s,
        )
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError("bolt seating tolerances must be finite and non-negative")
        if self.held_seat_dwell_s == 0 or self.stable_dwell_s == 0 or self.max_sample_gap_s == 0:
            raise ValueError("held-seat dwell, release dwell, and maximum sample gap must be positive")


@dataclass(frozen=True)
class BoltPoseWindowTolerances:
    """Private, explicitly calibrated pose envelope for one seating dwell."""

    max_position_excursion_m: float
    max_orientation_excursion_rad: float

    def __post_init__(self) -> None:
        if any(
            not isfinite(value) or value < 0.0
            for value in (self.max_position_excursion_m, self.max_orientation_excursion_rad)
        ):
            raise ValueError("pose-window excursion limits must be finite and non-negative")


@dataclass(frozen=True)
class BoltNativePoseSample:
    """One completed private physics-tick pose plus its unmodified velocity diagnostic."""

    physics_step_index: int
    time_s: float
    root_pos_w_m: tuple[float, float, float]
    root_quat_wxyz: tuple[float, float, float, float]
    native_linear_velocity_w_mps: tuple[float, float, float]
    native_angular_velocity_w_radps: tuple[float, float, float]

    def __post_init__(self) -> None:
        if type(self.physics_step_index) is not int or self.physics_step_index < 0:
            raise ValueError("native physics step index must be a non-negative integer")
        if not isfinite(self.time_s):
            raise ValueError("native pose timestamp must be finite")
        for name, size in (
            ("root_pos_w_m", 3),
            ("root_quat_wxyz", 4),
            ("native_linear_velocity_w_mps", 3),
            ("native_angular_velocity_w_radps", 3),
        ):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != size or any(not isfinite(value) for value in values):
                raise ValueError(f"{name} must contain {size} finite values")
            object.__setattr__(self, name, values)
        if sum(value * value for value in self.root_quat_wxyz) == 0.0:
            raise ValueError("native root orientation must be non-zero")


@dataclass(frozen=True)
class BoltSeatMeasurement:
    """One timestamped adapter sample in the selected assembly frame.

    The adapter measures shaft error over the selected path and supplies a
    conservative analytic CAD-envelope interference bound local to the selected
    socket's critical cover/casing planes and cap-bearing neighborhood. A zero
    bound can mean those local regions do not apply; the independent path flags
    and radial error still reject off-socket poses. This is not global collision
    depth or exact mesh/SDF distance; raw contact separation remains a separate
    diagnostic. Cap support contact is reported only for the underside patch and
    directed contact force. Speeds are bolt-relative to the casing;
    ``axial_position_m`` uses the selected axis.
    """

    time_s: float
    through_selected_cover_hole: bool
    entered_selected_casing_opening: bool
    casing_entry_depth_m: float
    shaft_radial_error_m: float
    cap_support_gap_m: float
    cap_support_contact: bool
    penetration_m: float
    premature_bottoming: bool
    physically_held: bool
    insertion_provenance_valid: bool
    controller_stopped: bool
    robot_contact_or_support: bool
    axial_position_m: float
    relative_linear_speed_mps: float
    relative_angular_speed_radps: float
    native_pose_samples: tuple[BoltNativePoseSample, ...] | None = None
    native_pose_dt_s: float | None = None

    def __post_init__(self) -> None:
        values = (
            self.time_s,
            self.casing_entry_depth_m,
            self.shaft_radial_error_m,
            self.cap_support_gap_m,
            self.penetration_m,
            self.axial_position_m,
            self.relative_linear_speed_mps,
            self.relative_angular_speed_radps,
        )
        if any(not isfinite(value) for value in values):
            raise ValueError("bolt seating measurements must be finite")
        if any(
            value < 0
            for value in (
                self.shaft_radial_error_m,
                self.penetration_m,
                self.relative_linear_speed_mps,
                self.relative_angular_speed_radps,
            )
        ):
            raise ValueError("measured errors, penetration, and speeds must be non-negative")
        if self.native_pose_samples is None:
            if self.native_pose_dt_s is not None:
                raise ValueError("native_pose_dt_s requires a native pose sample batch")
        else:
            if self.native_pose_dt_s is None or not isfinite(self.native_pose_dt_s) or self.native_pose_dt_s <= 0:
                raise ValueError("native pose batches require a finite positive physics dt")
            previous: BoltNativePoseSample | None = None
            for pose_sample in self.native_pose_samples:
                if not isinstance(pose_sample, BoltNativePoseSample):
                    raise TypeError("native_pose_samples must contain BoltNativePoseSample records")
                if previous is not None and (
                    pose_sample.physics_step_index != previous.physics_step_index + 1
                    or pose_sample.time_s <= previous.time_s
                ):
                    raise ValueError("native pose samples within a batch must be ordered contiguous ticks")
                previous = pose_sample


@dataclass(frozen=True)
class BoltSeatStatus:
    """The only public projections: held-seat readiness and final task success."""

    seat_ready: bool
    task_success: bool


class _PoseExcursion:
    """Cumulative all-sample pose envelope for one held or unassisted dwell."""

    def __init__(self, anchor: BoltNativePoseSample) -> None:
        self.anchor = anchor
        self.position_min = list(anchor.root_pos_w_m)
        self.position_max = list(anchor.root_pos_w_m)
        self.max_orientation_excursion_rad = 0.0

    def add(self, sample: BoltNativePoseSample) -> None:
        for axis in range(3):
            self.position_min[axis] = min(self.position_min[axis], sample.root_pos_w_m[axis])
            self.position_max[axis] = max(self.position_max[axis], sample.root_pos_w_m[axis])
        self.max_orientation_excursion_rad = max(
            self.max_orientation_excursion_rad,
            _orientation_distance(self.anchor.root_quat_wxyz, sample.root_quat_wxyz),
        )

    @property
    def position_excursion_m(self) -> float:
        return sqrt(sum((self.position_max[i] - self.position_min[i]) ** 2 for i in range(3)))

    def within(self, tolerances: BoltPoseWindowTolerances) -> bool:
        return (
            self.position_excursion_m <= tolerances.max_position_excursion_m + 1e-12
            and self.max_orientation_excursion_rad
            <= tolerances.max_orientation_excursion_rad + 1e-12
        )


def _unit_quaternion(values: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = sqrt(sum(value * value for value in values))
    return tuple(value / norm for value in values)  # type: ignore[return-value]


def _orientation_distance(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """Quaternion geodesic angle, stable near zero and invariant to q/-q."""
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
    vector_norm = sqrt(sum(value * value for value in relative[1:]))
    return 2.0 * atan2(vector_norm, abs(relative[0]))


class IndependentBoltSeatingEvaluator:
    """Evaluate seat readiness while held, then final success after release.

    If private pose-window tolerances are supplied, bounded native pose
    excursion replaces only the relative-speed stability predicate. All native
    velocity values remain in measurements as diagnostics; geometry, contact,
    provenance, and release-motion predicates are unchanged.
    """

    def __init__(
        self,
        tolerances: BoltSeatTolerances,
        pose_window_tolerances: BoltPoseWindowTolerances | None = None,
    ) -> None:
        self._tolerances = tolerances
        self._pose_window_tolerances = pose_window_tolerances
        self._previous_time_s: float | None = None
        self._seen_held = False
        self._held_seat_since_s: float | None = None
        self._seat_ready = False
        self._seated_at_s: float | None = None
        self._seated_axial_position_m: float | None = None
        self._released_at_s: float | None = None
        self._release_axial_position_m: float | None = None
        self._stable_since_s: float | None = None
        self._last_native_pose: BoltNativePoseSample | None = None
        self._native_pose_dt_s: float | None = None
        self._native_pose_batch_valid = False
        self._held_pose_excursion: _PoseExcursion | None = None
        self._stable_pose_excursion: _PoseExcursion | None = None
        self._failed = False
        self._complete = False

    def _geometry_valid(self, sample: BoltSeatMeasurement) -> bool:
        tol = self._tolerances
        return (
            sample.through_selected_cover_hole
            and sample.entered_selected_casing_opening
            and sample.casing_entry_depth_m >= tol.min_casing_entry_depth_m
            and sample.shaft_radial_error_m <= tol.max_shaft_radial_error_m
            and sample.cap_support_gap_m <= tol.max_cap_support_gap_m
            and sample.cap_support_contact
            and sample.penetration_m <= tol.max_penetration_m
            and not sample.premature_bottoming
        )

    def _motion_stable(self, sample: BoltSeatMeasurement) -> bool:
        tol = self._tolerances
        return (
            sample.relative_linear_speed_mps <= tol.max_relative_linear_speed_mps
            and sample.relative_angular_speed_radps <= tol.max_relative_angular_speed_radps
        )

    def _consume_native_pose_batch(self, sample: BoltSeatMeasurement) -> bool:
        if self._pose_window_tolerances is None:
            return True
        batch = sample.native_pose_samples
        dt_s = sample.native_pose_dt_s
        valid = bool(batch) and dt_s is not None
        if valid and self._native_pose_dt_s is not None:
            valid = abs(dt_s - self._native_pose_dt_s) <= max(1e-9, dt_s * 1e-6)
        previous = self._last_native_pose
        if valid:
            for pose_sample in batch:
                if previous is not None:
                    delta_s = pose_sample.time_s - previous.time_s
                    valid = (
                        pose_sample.physics_step_index == previous.physics_step_index + 1
                        and abs(delta_s - dt_s) <= max(1e-6, dt_s * 1e-3)
                    )
                if not valid:
                    break
                previous = pose_sample
            valid = valid and previous is not None and (
                abs(sample.time_s - previous.time_s) <= max(1e-6, dt_s * 1e-3)
            )
        if not valid:
            self._last_native_pose = batch[-1] if batch else None
            self._native_pose_dt_s = dt_s
            self._native_pose_batch_valid = False
            self._held_pose_excursion = None
            self._stable_pose_excursion = None
            return False
        self._last_native_pose = batch[-1]
        self._native_pose_dt_s = dt_s
        self._native_pose_batch_valid = True
        return True

    def _advance_pose_excursion(
        self,
        current: _PoseExcursion | None,
        batch: tuple[BoltNativePoseSample, ...] | None,
    ) -> tuple[_PoseExcursion | None, bool]:
        tolerances = self._pose_window_tolerances
        if tolerances is None or not self._native_pose_batch_valid or not batch:
            return None, False
        if current is None:
            return _PoseExcursion(batch[-1]), False
        for pose_sample in batch:
            current.add(pose_sample)
            if not current.within(tolerances):
                return _PoseExcursion(batch[-1]), False
        return current, True

    def _fail(self) -> BoltSeatStatus:
        self._failed = True
        self._seat_ready = False
        return BoltSeatStatus(False, False)

    def observe(self, sample: BoltSeatMeasurement) -> BoltSeatStatus:
        """Consume one sample and return only the two public boolean projections."""
        if not isinstance(sample, BoltSeatMeasurement):
            raise TypeError("sample must be a BoltSeatMeasurement")
        if self._failed or self._complete:
            return BoltSeatStatus(False, self._complete)

        self._consume_native_pose_batch(sample)

        if self._previous_time_s is not None:
            delta_s = sample.time_s - self._previous_time_s
            if delta_s <= 0:
                raise ValueError("measurement times must increase strictly")
            if self._seen_held and delta_s > self._tolerances.max_sample_gap_s + 1e-12:
                return self._fail()
        self._previous_time_s = sample.time_s

        if self._released_at_s is not None:
            if sample.physically_held or not self._geometry_valid(sample):
                return self._fail()
            axial_motion_m = abs(sample.axial_position_m - self._release_axial_position_m)
            if axial_motion_m > self._tolerances.max_axial_motion_since_release_m + 1e-12:
                return self._fail()
            if sample.robot_contact_or_support:
                self._stable_since_s = None
                self._stable_pose_excursion = None
                return BoltSeatStatus(False, False)
            if self._pose_window_tolerances is not None:
                self._stable_pose_excursion, pose_stable = self._advance_pose_excursion(
                    self._stable_pose_excursion, sample.native_pose_samples
                )
                if pose_stable:
                    if self._stable_since_s is None:
                        self._stable_since_s = sample.time_s
                    if sample.time_s - self._stable_since_s + 1e-12 >= self._tolerances.stable_dwell_s:
                        self._complete = True
                else:
                    self._stable_since_s = (
                        sample.time_s if self._stable_pose_excursion is not None else None
                    )
            elif self._motion_stable(sample):
                if self._stable_since_s is None:
                    self._stable_since_s = sample.time_s
                if sample.time_s - self._stable_since_s + 1e-12 >= self._tolerances.stable_dwell_s:
                    self._complete = True
            else:
                self._stable_since_s = None
                self._stable_pose_excursion = None
            return BoltSeatStatus(False, self._complete)

        if sample.physically_held:
            self._seen_held = True
            held_conditions_valid = (
                self._geometry_valid(sample)
                and sample.insertion_provenance_valid
                and sample.controller_stopped
            )
            if self._pose_window_tolerances is not None:
                self._held_pose_excursion, pose_stable = self._advance_pose_excursion(
                    self._held_pose_excursion, sample.native_pose_samples
                )
            else:
                pose_stable = self._motion_stable(sample)
            if not held_conditions_valid or not pose_stable:
                self._held_seat_since_s = None
                self._seat_ready = False
                self._seated_at_s = None
                self._seated_axial_position_m = None
                if not held_conditions_valid:
                    self._held_pose_excursion = None
                elif self._pose_window_tolerances is not None and self._held_pose_excursion is not None:
                    self._held_seat_since_s = sample.time_s
                return BoltSeatStatus(False, False)
            if self._held_seat_since_s is None:
                self._held_seat_since_s = sample.time_s
            if sample.time_s - self._held_seat_since_s + 1e-12 >= self._tolerances.held_seat_dwell_s:
                self._seat_ready = True
                self._seated_at_s = sample.time_s
                self._seated_axial_position_m = sample.axial_position_m
            return BoltSeatStatus(self._seat_ready, False)

        if not self._seen_held:
            return BoltSeatStatus(False, False)
        if (
            not self._seat_ready
            or self._seated_axial_position_m is None
            or not self._geometry_valid(sample)
        ):
            return self._fail()

        self._released_at_s = sample.time_s
        self._release_axial_position_m = self._seated_axial_position_m
        self._seat_ready = False
        axial_motion_m = abs(sample.axial_position_m - self._release_axial_position_m)
        if axial_motion_m > self._tolerances.max_axial_motion_since_release_m + 1e-12:
            return self._fail()
        if not sample.robot_contact_or_support:
            if self._pose_window_tolerances is not None:
                if self._native_pose_batch_valid and sample.native_pose_samples:
                    self._stable_pose_excursion = _PoseExcursion(sample.native_pose_samples[-1])
                    self._stable_since_s = sample.time_s
            elif self._motion_stable(sample):
                self._stable_since_s = sample.time_s
        return BoltSeatStatus(False, False)


__all__ = [
    "BoltSeatMeasurement",
    "BoltNativePoseSample",
    "BoltPoseWindowTolerances",
    "BoltSeatStatus",
    "BoltSeatTolerances",
    "IndependentBoltSeatingEvaluator",
]
