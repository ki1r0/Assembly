"""Private evaluator for calibrated rigid part-to-fixture seating relations."""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class RelationMeasurement:
    simulation_time_s: float
    moving_position_world_m: list[float]
    moving_quaternion_world_wxyz: list[float]
    fixture_position_world_m: list[float]
    fixture_quaternion_world_wxyz: list[float]
    moving_linear_velocity_world_mps: list[float]
    moving_angular_velocity_world_radps: list[float]
    released: bool
    gripper_contact_forces_n: list[float]
    fixture_contact_force_n: float
    maximum_penetration_m: float
    stable_window_s: float
    stable: bool
    pickup_observed: bool
    fixture_contact_points_world_m: list[list[float]] = field(default_factory=list)
    fixture_contact_normal_forces_world_n: list[list[float]] | None = None


def _vec3(value: Any, field: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field} must be a finite 3-vector")
    vector = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in vector):
        raise ValueError(f"{field} must be a finite 3-vector")
    return vector


def _unit_quaternion(value: Any, field: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{field} must be a finite wxyz quaternion")
    quaternion = tuple(float(item) for item in value)
    norm = math.sqrt(sum(item * item for item in quaternion))
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError(f"{field} must be a finite nonzero wxyz quaternion")
    return tuple(item / norm for item in quaternion)


def _quat_multiply(left, right):
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _quat_rotate(quaternion, vector):
    pure = (0.0, vector[0], vector[1], vector[2])
    conjugate = (quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3])
    rotated = _quat_multiply(_quat_multiply(quaternion, pure), conjugate)
    return rotated[1:]


def _norm(vector):
    return math.sqrt(sum(item * item for item in vector))


def _dot(left, right):
    return sum(left[index] * right[index] for index in range(3))


def _cross(left, right):
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _unit_vector(value: Any, field: str) -> tuple[float, float, float]:
    vector = _vec3(value, field)
    norm = _norm(vector)
    if norm < 1e-8:
        raise ValueError(f"{field} must be nonzero")
    return tuple(item / norm for item in vector)


class PendingRelationEvaluator:
    """Record simulator measurements without treating reset hints as ground truth."""

    def __init__(self, task_id: str, calibration_file: str, pending_reason: str = "calibration_missing"):
        self.task_id = task_id
        self.calibration_file = calibration_file
        self.pending_reason = pending_reason

    def observe(self, measurement: RelationMeasurement) -> dict:
        return {
            "task_success": False,
            "evaluator_status": "pending_calibration",
            "physical_gt_verified": False,
            "task_id": self.task_id,
            "calibration_file": self.calibration_file,
            "pending_reason": self.pending_reason,
            "measurement_only": True,
            "measurement": asdict(measurement),
        }


class CalibratedRelationEvaluator:
    """Score a rigid seat using a private, physically calibrated relative pose."""

    def __init__(self, task_id: str, calibration_file: str, calibration: Mapping[str, Any]):
        self.task_id = task_id
        self.calibration_file = calibration_file
        self.reference = calibration["reference_relative_pose"]
        self.reference_position = _vec3(self.reference["position_m"], "reference position")
        self.reference_quaternion = _unit_quaternion(self.reference["orientation_wxyz"], "reference orientation")
        self.tolerances = calibration["tolerances"]

    def observe(self, measurement: RelationMeasurement) -> dict:
        try:
            moving_position = _vec3(measurement.moving_position_world_m, "moving position")
            fixture_position = _vec3(measurement.fixture_position_world_m, "fixture position")
            moving_quaternion = _unit_quaternion(measurement.moving_quaternion_world_wxyz, "moving orientation")
            fixture_quaternion = _unit_quaternion(measurement.fixture_quaternion_world_wxyz, "fixture orientation")
            inverse_fixture = (fixture_quaternion[0], -fixture_quaternion[1], -fixture_quaternion[2], -fixture_quaternion[3])
            relative_position = _quat_rotate(
                inverse_fixture,
                tuple(moving_position[i] - fixture_position[i] for i in range(3)),
            )
            relative_quaternion = _unit_quaternion(
                _quat_multiply(inverse_fixture, moving_quaternion), "relative orientation"
            )
            position_error = _norm(tuple(relative_position[i] - self.reference_position[i] for i in range(3)))
            qdot = abs(sum(relative_quaternion[i] * self.reference_quaternion[i] for i in range(4)))
            orientation_error_deg = math.degrees(2.0 * math.acos(min(1.0, qdot)))
            if 'rotational_symmetry_axis_local' in self.reference:
                axis = _unit_vector(self.reference['rotational_symmetry_axis_local'], 'symmetry axis')
                actual_axis = _quat_rotate(relative_quaternion, axis)
                reference_axis = _quat_rotate(self.reference_quaternion, axis)
                orientation_error_deg = math.degrees(math.acos(max(-1., min(1., _dot(actual_axis, reference_axis)))))
            linear_speed = _norm(_vec3(measurement.moving_linear_velocity_world_mps, "linear velocity"))
            angular_speed = _norm(_vec3(measurement.moving_angular_velocity_world_radps, "angular velocity"))
            numeric = {
                "relative_position_error_m": position_error,
                "relative_orientation_error_deg": orientation_error_deg,
                "linear_speed_mps": linear_speed,
                "angular_speed_radps": angular_speed,
                "fixture_contact_force_n": float(measurement.fixture_contact_force_n),
                "maximum_penetration_m": float(measurement.maximum_penetration_m),
                "stable_window_s": float(measurement.stable_window_s),
            }
            if 'support_axis_fixture_local_unit' in self.reference:
                if measurement.fixture_contact_normal_forces_world_n is None:
                    raise ValueError('axial support normal-force measurement unavailable')
                support_axis = _quat_rotate(fixture_quaternion, _unit_vector(
                    self.reference['support_axis_fixture_local_unit'], 'support axis'))
                numeric['axial_support_force_n'] = sum(max(0., _dot(
                    _vec3(force, 'contact normal force'), support_axis))
                    for force in measurement.fixture_contact_normal_forces_world_n)
            if not all(math.isfinite(value) for value in numeric.values()):
                raise ValueError("measurement contains a non-finite metric")
        except (TypeError, ValueError, OverflowError) as error:
            return {
                "task_success": False,
                "evaluator_status": "UNKNOWN",
                "physical_gt_verified": True,
                "task_id": self.task_id,
                "calibration_file": self.calibration_file,
                "reasons": ["measurement_unavailable", str(error)],
                "measurement": asdict(measurement),
            }

        limits = self.tolerances
        checks = {
            "relative_position": position_error <= float(limits["translation_m"]),
            "relative_orientation": orientation_error_deg <= float(limits["orientation_deg"]),
            "contact": numeric["fixture_contact_force_n"] >= float(limits["min_fixture_contact_force_n"]),
            "penetration": numeric["maximum_penetration_m"] <= float(limits["max_penetration_m"]),
            "released": bool(measurement.released),
            "stable": bool(measurement.stable),
            "pickup_observed": bool(measurement.pickup_observed),
            "linear_speed": linear_speed <= float(limits["max_linear_speed_mps"]),
            "angular_speed": angular_speed <= float(limits["max_angular_speed_radps"]),
            "stable_window": numeric["stable_window_s"] >= float(limits["min_stable_window_s"]),
        }
        if 'axial_support_force_n' in numeric:
            checks['axial_support'] = numeric['axial_support_force_n'] >= float(limits['min_axial_support_force_n'])
        reasons = [f"{name}_criterion_failed" for name, passed in checks.items() if not passed]
        passed = all(checks.values())
        return {
            "task_success": passed,
            "evaluator_status": "PASS" if passed else "FAIL",
            "physical_criteria_passed": passed,
            "physical_gt_verified": True,
            "task_id": self.task_id,
            "calibration_file": self.calibration_file,
            "checks": checks,
            "reasons": reasons,
            "metrics": numeric | {
                "relative_position_m": list(relative_position),
                "relative_orientation_wxyz": list(relative_quaternion),
            },
            "measurement": asdict(measurement),
        }


class CalibratedMultiSurfaceMateEvaluator(CalibratedRelationEvaluator):
    """Require calibrated fixture-local contact in multiple distinct regions."""

    def __init__(self, task_id: str, calibration_file: str, calibration: Mapping[str, Any]):
        super().__init__(task_id, calibration_file, calibration)
        self.contact_regions = [
            _vec3(point, "contact region")
            for point in calibration["contact_regions_fixture_local_m"]
        ]
        self.contact_region_radius_m = float(calibration["contact_region_radius_m"])
        self.min_contact_points_per_region = int(calibration["min_contact_points_per_region"])

    def observe(self, measurement: RelationMeasurement) -> dict:
        result = super().observe(measurement)
        if result["evaluator_status"] == "UNKNOWN":
            return result

        try:
            fixture_position = _vec3(measurement.fixture_position_world_m, "fixture position")
            fixture_quaternion = _unit_quaternion(
                measurement.fixture_quaternion_world_wxyz, "fixture orientation"
            )
            inverse_fixture = (
                fixture_quaternion[0], -fixture_quaternion[1],
                -fixture_quaternion[2], -fixture_quaternion[3],
            )
            contact_points_local = [
                _quat_rotate(
                    inverse_fixture,
                    tuple(point[i] - fixture_position[i] for i in range(3)),
                )
                for point in (
                    _vec3(item, "fixture contact point")
                    for item in measurement.fixture_contact_points_world_m
                )
            ]
        except (TypeError, ValueError, OverflowError) as error:
            result.update(
                task_success=False,
                evaluator_status="UNKNOWN",
                physical_criteria_passed=False,
            )
            result["reasons"].append(f"contact_geometry_unavailable:{error}")
            return result

        counts = [
            sum(_norm(tuple(point[i] - center[i] for i in range(3))) <= self.contact_region_radius_m
                for point in contact_points_local)
            for center in self.contact_regions
        ]
        regions_supported = all(count >= self.min_contact_points_per_region for count in counts)
        result["checks"]["multi_surface_contact_regions"] = regions_supported
        result["metrics"].update(
            fixture_contact_point_count=len(contact_points_local),
            fixture_contact_region_counts=counts,
        )
        if not regions_supported:
            result["reasons"].append("multi_surface_contact_regions_criterion_failed")
        passed = all(result["checks"].values())
        result.update(
            task_success=passed,
            evaluator_status="PASS" if passed else "FAIL",
            physical_criteria_passed=passed,
        )
        return result


class CalibratedThreadedEngagementEvaluator:
    """Require measured screw-like rotation/advance plus released stable seating."""

    def __init__(self, task_id: str, calibration_file: str, calibration: Mapping[str, Any]):
        self.task_id = task_id
        self.calibration_file = calibration_file
        thread = calibration["thread_model"]
        self.axis = _unit_vector(thread["thread_axis_fixture_local_unit"], "thread axis")
        self.axis_origin = _vec3(thread["thread_axis_origin_fixture_local_m"], "thread axis origin")
        self.moving_axis = _unit_vector(thread["moving_axis_local_unit"], "moving axis")
        self.rotation_reference = _unit_vector(
            thread["rotation_reference_local_unit"], "rotation reference"
        )
        if abs(_dot(self.moving_axis, self.rotation_reference)) > 0.1:
            raise ValueError("rotation reference must be perpendicular to moving axis")
        self.pitch_m = float(thread["pitch_m_per_revolution"])
        self.advance_sign = int(thread["advance_sign_per_revolution"])
        self.tolerances = calibration["tolerances"]
        self.started = False
        self.anchor_axial_m = None
        self.previous_rotation_reference = None
        self.rotation_rad = 0.0
        self.contact_samples = 0

    def observe(self, measurement: RelationMeasurement) -> dict:
        try:
            moving_position = _vec3(measurement.moving_position_world_m, "moving position")
            fixture_position = _vec3(measurement.fixture_position_world_m, "fixture position")
            moving_quaternion = _unit_quaternion(
                measurement.moving_quaternion_world_wxyz, "moving orientation"
            )
            fixture_quaternion = _unit_quaternion(
                measurement.fixture_quaternion_world_wxyz, "fixture orientation"
            )
            fixture_inverse = (
                fixture_quaternion[0], -fixture_quaternion[1],
                -fixture_quaternion[2], -fixture_quaternion[3],
            )
            relative_position = _quat_rotate(
                fixture_inverse,
                tuple(moving_position[i] - fixture_position[i] for i in range(3)),
            )
            relative_quaternion = _unit_quaternion(
                _quat_multiply(fixture_inverse, moving_quaternion), "relative orientation"
            )
            moving_axis = _quat_rotate(relative_quaternion, self.moving_axis)
            axis_dot = max(-1.0, min(1.0, _dot(moving_axis, self.axis)))
            coaxial_error_deg = math.degrees(math.acos(axis_dot))
            offset = tuple(relative_position[i] - self.axis_origin[i] for i in range(3))
            axial_position_m = _dot(offset, self.axis)
            radial_vector = tuple(offset[i] - axial_position_m * self.axis[i] for i in range(3))
            radial_error_m = _norm(radial_vector)
            rotation_reference = _quat_rotate(relative_quaternion, self.rotation_reference)
            projection = tuple(
                rotation_reference[i] - _dot(rotation_reference, self.axis) * self.axis[i]
                for i in range(3)
            )
            projection_norm = _norm(projection)
            if projection_norm < 1e-8:
                raise ValueError("rotation reference became parallel to thread axis")
            rotation_reference = tuple(value / projection_norm for value in projection)
            contact_force_n = float(measurement.fixture_contact_force_n)
            penetration_m = float(measurement.maximum_penetration_m)
            linear_speed = _norm(_vec3(
                measurement.moving_linear_velocity_world_mps, "linear velocity"
            ))
            angular_speed = _norm(_vec3(
                measurement.moving_angular_velocity_world_radps, "angular velocity"
            ))
            numeric = (
                axial_position_m, radial_error_m, coaxial_error_deg,
                contact_force_n, penetration_m, linear_speed, angular_speed,
                float(measurement.stable_window_s),
            )
            if not all(math.isfinite(value) for value in numeric):
                raise ValueError("measurement contains a non-finite metric")
        except (TypeError, ValueError, OverflowError) as error:
            return {
                "task_success": False,
                "evaluator_status": "UNKNOWN",
                "physical_gt_verified": True,
                "task_id": self.task_id,
                "calibration_file": self.calibration_file,
                "reasons": ["measurement_unavailable", str(error)],
                "measurement": asdict(measurement),
            }

        limits = self.tolerances
        aligned = (
            radial_error_m <= float(limits["max_radial_error_m"])
            and coaxial_error_deg <= float(limits["max_coaxial_error_deg"])
        )
        contact = contact_force_n >= float(limits["min_fixture_contact_force_n"])
        if not self.started and measurement.pickup_observed and aligned and contact:
            self.started = True
            self.anchor_axial_m = axial_position_m
            self.previous_rotation_reference = rotation_reference
            self.contact_samples = 1
        elif self.started:
            if aligned and contact:
                cross = _cross(self.previous_rotation_reference, rotation_reference)
                sine = _dot(self.axis, cross)
                cosine = max(-1.0, min(1.0, _dot(
                    self.previous_rotation_reference, rotation_reference
                )))
                self.rotation_rad += math.atan2(sine, cosine)
                self.contact_samples += 1
            self.previous_rotation_reference = rotation_reference

        turns = self.rotation_rad / (2.0 * math.pi)
        advancement_m = (
            axial_position_m - self.anchor_axial_m if self.started else 0.0
        )
        expected_advancement_m = self.advance_sign * self.pitch_m * turns
        pitch_residual_m = advancement_m - expected_advancement_m
        pitch_tolerance_m = float(limits["max_pitch_error_per_turn_m"]) * max(abs(turns), 1.0)
        checks = {
            "pickup_observed": bool(measurement.pickup_observed),
            "engagement_started": self.started,
            "coaxial_alignment": aligned,
            "fixture_contact": contact,
            "minimum_thread_turns": abs(turns) >= float(limits["min_turns"]),
            "advance_in_thread_direction": expected_advancement_m >= float(limits["min_axial_advance_m"]),
            "helical_pitch_consistency": abs(pitch_residual_m) <= pitch_tolerance_m,
            "thread_contact_samples": self.contact_samples >= int(limits["min_thread_contact_samples"]),
            "penetration": penetration_m <= float(limits["max_penetration_m"]),
            "released": bool(measurement.released),
            "stable": bool(measurement.stable),
            "linear_speed": linear_speed <= float(limits["max_linear_speed_mps"]),
            "angular_speed": angular_speed <= float(limits["max_angular_speed_radps"]),
            "stable_window": float(measurement.stable_window_s) >= float(limits["min_stable_window_s"]),
        }
        reasons = [f"{name}_criterion_failed" for name, passed in checks.items() if not passed]
        passed = all(checks.values())
        return {
            "task_success": passed,
            "evaluator_status": "PASS" if passed else "FAIL",
            "physical_criteria_passed": passed,
            "physical_gt_verified": True,
            "task_id": self.task_id,
            "calibration_file": self.calibration_file,
            "checks": checks,
            "reasons": reasons,
            "metrics": {
                "relative_position_m": list(relative_position),
                "thread_axis_fixture_local_unit": list(self.axis),
                "coaxial_error_deg": coaxial_error_deg,
                "radial_error_m": radial_error_m,
                "signed_thread_turns": turns,
                "axial_advancement_m": advancement_m,
                "expected_helical_advancement_m": expected_advancement_m,
                "helical_pitch_residual_m": pitch_residual_m,
                "thread_contact_samples": self.contact_samples,
                "fixture_contact_force_n": contact_force_n,
                "maximum_penetration_m": penetration_m,
                "linear_speed_mps": linear_speed,
                "angular_speed_radps": angular_speed,
                "stable_window_s": float(measurement.stable_window_s),
            },
            "measurement": asdict(measurement),
        }


def _threaded_calibration_problem(task_id, evaluator_kind, calibration, current_scale):
    if evaluator_kind != "threaded_engagement":
        return "specialized_evaluator_required:" + str(evaluator_kind or "unspecified")
    if not calibration:
        return "calibration_file_missing"
    if calibration.get("task_id") != task_id:
        return "calibration_task_mismatch"
    if calibration.get("evaluator_kind") != evaluator_kind:
        return "calibration_evaluator_kind_mismatch"
    if calibration.get("calibration_status") != "calibrated" or calibration.get("physical_gt_verified") is not True:
        return "physical_calibration_not_verified"
    try:
        scale = float(calibration["cad_meters_per_source_unit"])
    except (KeyError, TypeError, ValueError):
        return "calibration_scale_missing"
    if not math.isclose(scale, float(current_scale), rel_tol=0.0, abs_tol=1e-12):
        return "calibration_scale_mismatch"
    trials = calibration.get("validation_trials", {})
    if not all(isinstance(trials.get(kind), list) and trials[kind] for kind in ("positive", "negative")):
        return "positive_and_negative_physical_trials_required"
    if not all(isinstance(item, str) and item.strip() for kind in ("positive", "negative") for item in trials[kind]):
        return "physical_trial_references_invalid"
    try:
        thread = calibration["thread_model"]
        axis = _unit_vector(thread["thread_axis_fixture_local_unit"], "thread axis")
        moving_axis = _unit_vector(thread["moving_axis_local_unit"], "moving axis")
        rotation_reference = _unit_vector(
            thread["rotation_reference_local_unit"], "rotation reference"
        )
        _vec3(thread["thread_axis_origin_fixture_local_m"], "thread axis origin")
        if abs(_dot(moving_axis, rotation_reference)) > 0.1:
            return "thread_rotation_reference_invalid"
        if not math.isfinite(float(thread["pitch_m_per_revolution"])) or float(thread["pitch_m_per_revolution"]) <= 0:
            return "thread_pitch_invalid"
        sign = thread["advance_sign_per_revolution"]
        if isinstance(sign, bool) or sign not in (-1, 1):
            return "thread_advance_sign_invalid"
        limits = calibration["tolerances"]
        positive_limits = (
            "max_radial_error_m", "max_coaxial_error_deg", "min_turns",
            "min_axial_advance_m", "max_pitch_error_per_turn_m",
            "max_linear_speed_mps", "max_angular_speed_radps", "min_stable_window_s",
        )
        nonnegative_limits = (
            "min_fixture_contact_force_n", "max_penetration_m",
        )
        if any(not math.isfinite(float(limits[name])) or float(limits[name]) <= 0 for name in positive_limits):
            return "thread_tolerances_invalid"
        if any(not math.isfinite(float(limits[name])) or float(limits[name]) < 0 for name in nonnegative_limits):
            return "thread_tolerances_invalid"
        if float(limits["max_coaxial_error_deg"]) > 180:
            return "thread_tolerances_invalid"
        minimum_samples = limits["min_thread_contact_samples"]
        if isinstance(minimum_samples, bool) or int(minimum_samples) != minimum_samples or minimum_samples < 1:
            return "thread_tolerances_invalid"
        _unit_vector(axis, "thread axis")
    except (KeyError, TypeError, ValueError, OverflowError):
        return "thread_calibration_schema_invalid"
    return None


def _calibration_problem(task_id, evaluator_kind, calibration, current_scale):
    if evaluator_kind not in {"rigid_pose_seat", "multi_surface_mate"}:
        return f"specialized_evaluator_required:{evaluator_kind or 'unspecified'}"
    if not calibration:
        return "calibration_file_missing"
    if calibration.get("task_id") != task_id:
        return "calibration_task_mismatch"
    if calibration.get("evaluator_kind") != evaluator_kind:
        return "calibration_evaluator_kind_mismatch"
    if calibration.get("calibration_status") != "calibrated" or calibration.get("physical_gt_verified") is not True:
        return "physical_calibration_not_verified"
    try:
        scale = float(calibration["cad_meters_per_source_unit"])
    except (KeyError, TypeError, ValueError):
        return "calibration_scale_missing"
    if not math.isclose(scale, float(current_scale), rel_tol=0.0, abs_tol=1e-12):
        return "calibration_scale_mismatch"
    trials = calibration.get("validation_trials", {})
    if not all(isinstance(trials.get(kind), list) and trials[kind] for kind in ("positive", "negative")):
        return "positive_and_negative_physical_trials_required"
    if not all(isinstance(item, str) and item.strip() for kind in ("positive", "negative") for item in trials[kind]):
        return "physical_trial_references_invalid"
    try:
        reference = calibration["reference_relative_pose"]
        _vec3(reference["position_m"], "reference position")
        _unit_quaternion(reference["orientation_wxyz"], "reference orientation")
        if 'rotational_symmetry_axis_local' in reference:
            _unit_vector(reference['rotational_symmetry_axis_local'], 'symmetry axis')
        limits = calibration["tolerances"]
        if 'support_axis_fixture_local_unit' in reference:
            _unit_vector(reference['support_axis_fixture_local_unit'], 'support axis')
            minimum_support = float(limits['min_axial_support_force_n'])
            if not math.isfinite(minimum_support) or minimum_support <= 0:
                return 'calibration_tolerances_invalid'
        positive_limits = (
            "translation_m", "orientation_deg", "max_linear_speed_mps",
            "max_angular_speed_radps", "min_stable_window_s",
        )
        nonnegative_limits = ("max_penetration_m", "min_fixture_contact_force_n")
        if any(not math.isfinite(float(limits[name])) or float(limits[name]) <= 0 for name in positive_limits):
            return "calibration_tolerances_invalid"
        if any(not math.isfinite(float(limits[name])) or float(limits[name]) < 0 for name in nonnegative_limits):
            return "calibration_tolerances_invalid"
        if evaluator_kind == "multi_surface_mate":
            regions = calibration["contact_regions_fixture_local_m"]
            if not isinstance(regions, list) or len(regions) < 2:
                return "multi_surface_contact_regions_invalid"
            for region in regions:
                _vec3(region, "contact region")
            radius = float(calibration["contact_region_radius_m"])
            minimum_points = calibration["min_contact_points_per_region"]
            if not math.isfinite(radius) or radius <= 0:
                return "multi_surface_contact_regions_invalid"
            centers = [_vec3(region, "contact region") for region in regions]
            for index, center in enumerate(centers):
                for other in centers[index + 1:]:
                    separation = _norm(tuple(center[i] - other[i] for i in range(3)))
                    if separation <= 2.0 * radius:
                        return "multi_surface_contact_regions_overlap"
            if isinstance(minimum_points, bool) or int(minimum_points) != minimum_points or minimum_points < 1:
                return "multi_surface_contact_regions_invalid"
    except (KeyError, TypeError, ValueError, OverflowError):
        return "calibration_schema_invalid"
    return None


def build_relation_evaluator(
    task_id: str,
    evaluator_profile: Mapping[str, Any],
    calibration: Mapping[str, Any] | None,
    current_scale: float,
) -> PendingRelationEvaluator | CalibratedRelationEvaluator | CalibratedThreadedEngagementEvaluator:
    calibration_file = str(evaluator_profile.get("calibration_file", ""))
    kind = evaluator_profile.get("evaluator_kind")
    calibration = calibration or {}
    if kind == "threaded_engagement":
        problem = _threaded_calibration_problem(task_id, kind, calibration, current_scale)
    else:
        problem = _calibration_problem(task_id, kind, calibration, current_scale)
    if problem:
        return PendingRelationEvaluator(task_id, calibration_file, problem)
    try:
        if kind == "multi_surface_mate":
            return CalibratedMultiSurfaceMateEvaluator(task_id, calibration_file, calibration)
        if kind == "threaded_engagement":
            return CalibratedThreadedEngagementEvaluator(task_id, calibration_file, calibration)
        return CalibratedRelationEvaluator(task_id, calibration_file, calibration)
    except (KeyError, TypeError, ValueError) as error:
        return PendingRelationEvaluator(task_id, calibration_file, f"calibration_schema_invalid:{error}")


__all__ = [
    "CalibratedRelationEvaluator",
    "CalibratedMultiSurfaceMateEvaluator",
    "CalibratedThreadedEngagementEvaluator",
    "PendingRelationEvaluator",
    "RelationMeasurement",
    "build_relation_evaluator",
]
