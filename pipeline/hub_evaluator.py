"""Evaluator-side seating criteria, isolated from planner-visible state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def datum_equivalent_root(position,rotation,reference_rotation,datum):
    """Compare a mating datum using existing canonical-root tolerances, not an off-axis origin."""
    import numpy as np
    p,r,ref,d=map(lambda value:np.asarray(value,float),(position,rotation,reference_rotation,datum))
    if p.shape!=(3,) or d.shape!=(3,) or r.shape!=(3,3) or ref.shape!=(3,3) or not all(np.isfinite(v).all() for v in (p,r,ref,d)):
        raise ValueError('Finite positions and 3D rotation matrices required')
    return p+r@d-ref@d


def settle_window_samples(physics_rate,duration_s=.5):
    from math import ceil,isfinite
    if not isfinite(physics_rate) or not isfinite(duration_s) or physics_rate<=0 or duration_s<=0:raise ValueError('Positive finite physics rate and duration required')
    return ceil(physics_rate*duration_s)


def axisymmetric_seat_stable(positions, axes, sample_count=60):
    """Pose retention over the requested physical window; axial spin is not tilt."""
    import numpy as np

    p, a = np.asarray(positions, float), np.asarray(axes, float)
    if p.shape != (sample_count, 3) or a.shape != (sample_count, 3):
        return False
    if not np.isfinite(p).all() or not np.isfinite(a).all():
        return False
    lengths = np.linalg.norm(a, axis=1)
    if np.any(lengths < 1e-9):
        return False
    a = a / lengths[:, None]
    position_excursion = np.max(np.linalg.norm(p - p[0], axis=1))
    axis_excursion = np.max(np.degrees(np.arccos(np.clip(a @ a[0], -1, 1))))
    return bool(position_excursion <= .001 and axis_excursion <= .5)


def settle_window_contact(contacts, penetrations, sample_count=60):
    """Retain real intermittent support, but never discard penetration peaks."""
    import numpy as np

    p = np.asarray(penetrations, float)
    if len(contacts) != sample_count or p.shape != (sample_count,) or not np.isfinite(p).all() or np.any(p < 0):
        return False, None
    return any(contacts), float(p.max())


@dataclass(frozen=True)
class SeatTolerances:
    axial_depth_m: float = 0.062
    axial_tolerance_m: float = 0.008
    radial_tolerance_m: float = 0.004
    tilt_tolerance_deg: float = 2.0
    max_penetration_m: float = 0.001
    settle_speed_mps: float = 0.01
    settle_window_s: float = 0.5


@dataclass(frozen=True)
class SeatMeasurement:
    axial_depth_m: float | None = None
    radial_error_m: float | None = None
    tilt_deg: float | None = None
    penetration_m: float | None = None
    released: bool = False
    stable: bool = False
    contact_valid: bool = False
    settle_speed_mps: float | None = None
    settle_window_s: float | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SeatMeasurement":
        allowed = {
            "axial_depth_m", "radial_error_m", "tilt_deg", "penetration_m",
            "released", "stable", "contact_valid", "settle_speed_mps", "settle_window_s",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown evaluator measurement fields: {sorted(unknown)}")
        kwargs: dict[str, Any] = {}
        for name in allowed:
            if name in value:
                kwargs[name] = value[name]
        for name in ("axial_depth_m", "radial_error_m", "tilt_deg", "penetration_m", "settle_speed_mps", "settle_window_s"):
            if name in kwargs and kwargs[name] is not None:
                kwargs[name] = float(kwargs[name])
        for name in ("released", "stable", "contact_valid"):
            if name in kwargs:
                kwargs[name] = bool(kwargs[name])
        return cls(**kwargs)


@dataclass(frozen=True)
class EvaluationResult:
    task_success: bool
    physical_criteria_passed: bool
    status: str
    calibration_status: str
    reasons: tuple[str, ...] = field(default_factory=tuple)
    metrics: dict[str, float | bool | None] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "task_success": self.task_success,
            "physical_criteria_passed": self.physical_criteria_passed,
            "status": self.status,
            "calibration_status": self.calibration_status,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
        }


class IndependentSeatEvaluator:
    """Evaluate physical measurements supplied by the simulator-side evaluator.

    A pending calibration can produce a provisional physical result but never a
    public M1 task success.  The planner only receives a tri-state verifier
    result, not these measurements or reasons.
    """

    def __init__(self, tolerances: SeatTolerances | None = None, *, calibration_status: str = "pending") -> None:
        self.tolerances = tolerances or SeatTolerances()
        if calibration_status not in {"pending", "calibrated", "provisional"}:
            raise ValueError("calibration_status must be pending, provisional or calibrated")
        self.calibration_status = calibration_status

    def evaluate(self, measurement: SeatMeasurement | Mapping[str, Any]) -> EvaluationResult:
        if not isinstance(measurement, SeatMeasurement):
            measurement = SeatMeasurement.from_mapping(measurement)
        tol = self.tolerances
        reasons: list[str] = []
        checks: dict[str, bool] = {}

        if measurement.axial_depth_m is None:
            checks["axial"] = False
            reasons.append("axial_depth_unavailable")
        else:
            checks["axial"] = abs(measurement.axial_depth_m - tol.axial_depth_m) <= tol.axial_tolerance_m
            if not checks["axial"]:
                reasons.append("axial_depth_out_of_tolerance")
        if measurement.radial_error_m is None:
            checks["radial"] = False
            reasons.append("radial_error_unavailable")
        else:
            checks["radial"] = abs(measurement.radial_error_m) <= tol.radial_tolerance_m
            if not checks["radial"]:
                reasons.append("radial_error_out_of_tolerance")
        if measurement.tilt_deg is None:
            checks["tilt"] = False
            reasons.append("tilt_unavailable")
        else:
            checks["tilt"] = abs(measurement.tilt_deg) <= tol.tilt_tolerance_deg
            if not checks["tilt"]:
                reasons.append("tilt_out_of_tolerance")
        if measurement.penetration_m is None:
            checks["penetration"] = False
            reasons.append("penetration_unavailable")
        else:
            checks["penetration"] = measurement.penetration_m <= tol.max_penetration_m
            if not checks["penetration"]:
                reasons.append("penetration_exceeded")
        checks["contact"] = bool(measurement.contact_valid)
        checks["released"] = bool(measurement.released)
        checks["stable"] = bool(measurement.stable)
        if not checks["contact"]:
            reasons.append("valid_contact_not_observed")
        if not checks["released"]:
            reasons.append("not_released")
        if not checks["stable"]:
            reasons.append("not_stable")
        if measurement.settle_speed_mps is not None and measurement.settle_speed_mps > tol.settle_speed_mps:
            reasons.append("settle_speed_exceeded")
            checks["settle_speed"] = False
        else:
            checks["settle_speed"] = measurement.settle_speed_mps is not None
            if measurement.settle_speed_mps is None:
                reasons.append("settle_speed_unavailable")
        # The stability accumulator advances in fixed simulation quanta.  Do
        # not turn an exact boundary hit (e.g. 0.49999999999999994 for a
        # requested 0.5 s) into a false failure solely because of float
        # representation.
        if measurement.settle_window_s is not None and measurement.settle_window_s + 1.0e-9 < tol.settle_window_s:
            reasons.append("settle_window_too_short")
            checks["settle_window"] = False
        else:
            checks["settle_window"] = measurement.settle_window_s is not None
            if measurement.settle_window_s is None:
                reasons.append("settle_window_unavailable")

        physical = all(checks.values())
        unavailable = any(reason.endswith("_unavailable") for reason in reasons)
        if unavailable:
            status = "UNKNOWN"
            task_success = False
            physical = False
        elif physical and self.calibration_status == "calibrated":
            status = "PASS"
            task_success = True
        elif physical:
            status = "PROVISIONAL_PASS"
            task_success = False
            reasons.append("calibration_not_final")
        else:
            status = "FAIL"
            task_success = False
        metrics = {
            "axial_depth_m": measurement.axial_depth_m,
            "radial_error_m": measurement.radial_error_m,
            "tilt_deg": measurement.tilt_deg,
            "penetration_m": measurement.penetration_m,
            "released": measurement.released,
            "stable": measurement.stable,
            "contact_valid": measurement.contact_valid,
            "settle_speed_mps": measurement.settle_speed_mps,
            "settle_window_s": measurement.settle_window_s,
        }
        return EvaluationResult(task_success, physical, status, self.calibration_status, tuple(reasons), metrics)


__all__ = ["EvaluationResult", "IndependentSeatEvaluator", "SeatMeasurement", "SeatTolerances"]

def robot_release_verified(widths,forces,withdrawn=None):
    """Geometric withdrawal plus no support, or the legacy full-open fallback."""
    return bool(widths) and (bool(withdrawn) if withdrawn is not None else all(w>=.079 for w in widths)) and all(abs(f)<=.02 for arm in forces for f in arm)


def aabb_separation(first,second):
    import numpy as np
    a,b=np.asarray(first,float),np.asarray(second,float)
    if a.shape!=(2,3) or b.shape!=(2,3) or not np.isfinite(a).all() or not np.isfinite(b).all() or (a[0]>a[1]).any() or (b[0]>b[1]).any():raise ValueError('Finite ordered 3D bounds required')
    return float(np.linalg.norm(np.maximum(0,np.maximum(a[0]-b[1],b[0]-a[1]))))
