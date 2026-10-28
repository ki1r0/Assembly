import math
import pytest
from dataclasses import replace

from relation_evaluator import (
    CalibratedMultiSurfaceMateEvaluator,
    CalibratedRelationEvaluator,
    CalibratedThreadedEngagementEvaluator,
    PendingRelationEvaluator,
    RelationMeasurement,
    build_relation_evaluator,
)


def _measurement(
    *, released=True, stable=True, pickup=True, position_world=(1.0, 2.01, 3.0),
    force=2.0, contact_points_world=None,
):
    half = math.sqrt(0.5)
    return RelationMeasurement(
        simulation_time_s=4.0,
        moving_position_world_m=list(position_world),
        moving_quaternion_world_wxyz=[half, 0.0, 0.0, half],
        fixture_position_world_m=[1.0, 2.0, 3.0],
        fixture_quaternion_world_wxyz=[half, 0.0, 0.0, half],
        moving_linear_velocity_world_mps=[0.0, 0.0, 0.0],
        moving_angular_velocity_world_radps=[0.0, 0.0, 0.0],
        released=released,
        gripper_contact_forces_n=[0.0, 0.0],
        fixture_contact_force_n=force,
        maximum_penetration_m=0.0001,
        stable_window_s=0.6,
        stable=stable,
        pickup_observed=pickup,
        fixture_contact_points_world_m=contact_points_world or [],
    )


def _calibration():
    return {
        "task_id": "bearing-fit",
        "evaluator_kind": "rigid_pose_seat",
        "cad_meters_per_source_unit": 0.001,
        "calibration_status": "calibrated",
        "physical_gt_verified": True,
        "validation_trials": {
            "positive": ["runs/positive-seat"],
            "negative": ["runs/underseated", "runs/unreleased"],
        },
        "reference_relative_pose": {
            "position_m": [0.01, 0.0, 0.0],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "tolerances": {
            "translation_m": 0.001,
            "orientation_deg": 1.0,
            "min_fixture_contact_force_n": 0.5,
            "max_penetration_m": 0.001,
            "max_linear_speed_mps": 0.01,
            "max_angular_speed_radps": 0.02,
            "min_stable_window_s": 0.5,
        },
    }


def test_axisymmetric_seat_ignores_spin_but_not_tilt_or_flipped_face():
    from dataclasses import replace
    calibration = _calibration()
    calibration['reference_relative_pose']['rotational_symmetry_axis_local'] = [0., 0., 1.]
    evaluator = CalibratedRelationEvaluator('bearing-fit', 'test', calibration)
    spun = replace(_measurement(), moving_quaternion_world_wxyz=[0., 0., 0., 1.])
    assert evaluator.observe(spun)['checks']['relative_orientation'] is True
    tilted = replace(_measurement(), moving_quaternion_world_wxyz=[.5, .5, .5, .5])
    assert evaluator.observe(tilted)['checks']['relative_orientation'] is False
    flipped = replace(_measurement(), moving_quaternion_world_wxyz=[0., math.sqrt(.5), math.sqrt(.5), 0.])
    assert evaluator.observe(flipped)['checks']['relative_orientation'] is False
    ordinary = CalibratedRelationEvaluator('bearing-fit', 'test', _calibration())
    assert ordinary.observe(spun)['checks']['relative_orientation'] is False


def _thread_calibration():
    return {
        "task_id": "m10-casing-nut",
        "evaluator_kind": "threaded_engagement",
        "cad_meters_per_source_unit": 0.001,
        "calibration_status": "calibrated",
        "physical_gt_verified": True,
        "validation_trials": {
            "positive": ["runs/thread-positive"],
            "negative": ["runs/push-fit-negative", "runs/reverse-thread-negative"],
        },
        "thread_model": {
            "thread_axis_fixture_local_unit": [0.0, 0.0, 1.0],
            "thread_axis_origin_fixture_local_m": [0.0, 0.0, 0.0],
            "moving_axis_local_unit": [0.0, 0.0, 1.0],
            "rotation_reference_local_unit": [1.0, 0.0, 0.0],
            "pitch_m_per_revolution": 0.0015,
            "advance_sign_per_revolution": 1,
        },
        "tolerances": {
            "max_radial_error_m": 0.001,
            "max_coaxial_error_deg": 2.0,
            "min_turns": 1.5,
            "min_axial_advance_m": 0.002,
            "max_pitch_error_per_turn_m": 0.0001,
            "min_thread_contact_samples": 8,
            "min_fixture_contact_force_n": 0.5,
            "max_penetration_m": 0.001,
            "max_linear_speed_mps": 0.01,
            "max_angular_speed_radps": 0.02,
            "min_stable_window_s": 0.5,
        },
    }


def _thread_measurement(
    angle_rad, axial_m, *, index, released=False, stable=False,
    contact_force_n=2.0, radial_offset_m=(0.0, 0.0),
):
    half_angle = angle_rad / 2.0
    return RelationMeasurement(
        simulation_time_s=index * 0.1,
        moving_position_world_m=[radial_offset_m[0], radial_offset_m[1], axial_m],
        moving_quaternion_world_wxyz=[math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)],
        fixture_position_world_m=[0.0, 0.0, 0.0],
        fixture_quaternion_world_wxyz=[1.0, 0.0, 0.0, 0.0],
        moving_linear_velocity_world_mps=[0.0, 0.0, 0.0],
        moving_angular_velocity_world_radps=[0.0, 0.0, 0.0],
        released=released,
        gripper_contact_forces_n=[0.0, 0.0],
        fixture_contact_force_n=contact_force_n,
        maximum_penetration_m=0.0001,
        stable_window_s=0.6 if stable else 0.0,
        stable=stable,
        pickup_observed=True,
    )


def test_calibrated_evaluator_scores_fixture_relative_pose_and_physics():
    evaluator = CalibratedRelationEvaluator("bearing-fit", "bearing-fit.json", _calibration())

    result = evaluator.observe(_measurement())

    assert result["task_success"] is True
    assert result["evaluator_status"] == "PASS"
    assert result["physical_gt_verified"] is True
    assert result["metrics"]["relative_position_error_m"] < 1e-9
    assert result["metrics"]["relative_orientation_error_deg"] < 1e-6


def test_calibrated_evaluator_rejects_wrong_pose_or_unreleased_part():
    evaluator = CalibratedRelationEvaluator("bearing-fit", "bearing-fit.json", _calibration())

    wrong_pose = evaluator.observe(_measurement(position_world=(1.0, 2.04, 3.0)))
    held_part = evaluator.observe(_measurement(released=False))

    assert wrong_pose["task_success"] is False
    assert "relative_position_criterion_failed" in wrong_pose["reasons"]
    assert held_part["task_success"] is False
    assert "released_criterion_failed" in held_part["reasons"]


def test_unstable_missing_pickup_and_unsupported_contact_fail():
    evaluator = CalibratedRelationEvaluator("bearing-fit", "bearing-fit.json", _calibration())
    result = evaluator.observe(_measurement(stable=False, pickup=False, force=0.0))

    assert result["task_success"] is False
    assert "stable_criterion_failed" in result["reasons"]
    assert "pickup_observed_criterion_failed" in result["reasons"]
    assert "contact_criterion_failed" in result["reasons"]


def test_uncalibrated_or_specialized_relations_never_activate_generic_scorer():
    profile = {
        "calibration_file": "nut.json",
        "evaluator_kind": "threaded_engagement",
        "candidate_reset_hint": {"position": [0.0, 0.0, 0.0]},
    }
    wrong_kind_calibration = _calibration() | {"task_id": "m10-casing-nut"}
    result = build_relation_evaluator("m10-casing-nut", profile, wrong_kind_calibration, 0.001)

    assert isinstance(result, PendingRelationEvaluator)
    assert result.pending_reason == "calibration_evaluator_kind_mismatch"
    assert result.observe(_measurement())["task_success"] is False


def test_threaded_evaluator_requires_calibration_and_scores_helical_progress():
    profile = {
        "calibration_file": "m10_casing_nut.json",
        "evaluator_kind": "threaded_engagement",
    }
    pending = build_relation_evaluator("m10-casing-nut", profile, None, 0.001)
    assert isinstance(pending, PendingRelationEvaluator)
    assert pending.pending_reason == "calibration_file_missing"
    assert pending.observe(_measurement())["task_success"] is False

    calibration = _thread_calibration()
    evaluator = build_relation_evaluator("m10-casing-nut", profile, calibration, 0.001)
    assert isinstance(evaluator, CalibratedThreadedEngagementEvaluator)
    for index in range(17):
        turns = index / 8.0
        result = evaluator.observe(
            _thread_measurement(
                turns * 2.0 * math.pi,
                turns * calibration["thread_model"]["pitch_m_per_revolution"],
                index=index,
                released=index == 16,
                stable=index == 16,
            )
        )

    assert result["task_success"] is True
    assert result["evaluator_status"] == "PASS"
    assert result["metrics"]["signed_thread_turns"] == pytest.approx(2.0)
    assert result["metrics"]["helical_pitch_residual_m"] == pytest.approx(0.0, abs=1e-9)


def test_threaded_calibration_rejects_scale_trial_and_thread_model_errors():
    profile = {
        "calibration_file": "m10_casing_nut.json",
        "evaluator_kind": "threaded_engagement",
    }
    calibration = _thread_calibration() | {"cad_meters_per_source_unit": 0.002}
    evaluator = build_relation_evaluator("m10-casing-nut", profile, calibration, 0.001)
    assert isinstance(evaluator, PendingRelationEvaluator)
    assert evaluator.pending_reason == "calibration_scale_mismatch"

    calibration = _thread_calibration()
    calibration["validation_trials"]["negative"] = []
    evaluator = build_relation_evaluator("m10-casing-nut", profile, calibration, 0.001)
    assert isinstance(evaluator, PendingRelationEvaluator)
    assert evaluator.pending_reason == "positive_and_negative_physical_trials_required"

    calibration = _thread_calibration()
    calibration["thread_model"]["advance_sign_per_revolution"] = 0
    evaluator = build_relation_evaluator("m10-casing-nut", profile, calibration, 0.001)
    assert isinstance(evaluator, PendingRelationEvaluator)
    assert evaluator.pending_reason == "thread_advance_sign_invalid"


@pytest.mark.parametrize(
    "angle_fn,advance_fn,expected_reason",
    [
        (lambda turns: 0.0, lambda turns: 0.0015 * turns, "minimum_thread_turns_criterion_failed"),
        (lambda turns: -2.0 * math.pi * turns, lambda turns: 0.0015 * turns, "advance_in_thread_direction_criterion_failed"),
    ],
)
def test_threaded_evaluator_rejects_push_fit_and_wrong_handed_rotation(
    angle_fn, advance_fn, expected_reason
):
    evaluator = CalibratedThreadedEngagementEvaluator(
        "m10-casing-nut", "m10.json", _thread_calibration()
    )
    result = None
    for index in range(17):
        turns = index / 8.0
        result = evaluator.observe(
            _thread_measurement(
                angle_fn(turns),
                advance_fn(turns),
                index=index,
                released=index == 16,
                stable=index == 16,
            )
        )

    assert result["task_success"] is False
    assert expected_reason in result["reasons"]


@pytest.mark.parametrize(
    "final_measurement,expected_reason",
    [
        ({"released": False, "stable": True}, "released_criterion_failed"),
        ({"released": True, "stable": False}, "stable_criterion_failed"),
        ({"released": True, "stable": True, "contact_force_n": 0.0}, "fixture_contact_criterion_failed"),
        ({"released": True, "stable": True, "radial_offset_m": (0.01, 0.0)}, "coaxial_alignment_criterion_failed"),
    ],
)
def test_threaded_evaluator_requires_release_stability_contact_and_alignment(
    final_measurement, expected_reason
):
    calibration = _thread_calibration()
    evaluator = CalibratedThreadedEngagementEvaluator(
        "m10-casing-nut", "m10.json", calibration
    )
    result = None
    for index in range(17):
        turns = index / 8.0
        options = final_measurement if index == 16 else {}
        result = evaluator.observe(
            _thread_measurement(
                turns * 2.0 * math.pi,
                turns * calibration["thread_model"]["pitch_m_per_revolution"],
                index=index,
                **options,
            )
        )

    assert result["task_success"] is False
    assert expected_reason in result["reasons"]


def test_scale_mismatch_and_unverified_trials_remain_pending():
    profile = {"calibration_file": "bearing-fit.json", "evaluator_kind": "rigid_pose_seat"}
    calibration = _calibration()
    calibration["cad_meters_per_source_unit"] = 0.002
    evaluator = build_relation_evaluator("bearing-fit", profile, calibration, 0.001)
    assert isinstance(evaluator, PendingRelationEvaluator)
    assert evaluator.pending_reason == "calibration_scale_mismatch"

    calibration = _calibration()
    calibration["validation_trials"]["negative"] = []
    evaluator = build_relation_evaluator("bearing-fit", profile, calibration, 0.001)
    assert isinstance(evaluator, PendingRelationEvaluator)
    assert evaluator.pending_reason == "positive_and_negative_physical_trials_required"


def test_multi_surface_evaluator_requires_contact_at_each_calibrated_region():
    calibration = _calibration() | {
        "evaluator_kind": "multi_surface_mate",
        "contact_regions_fixture_local_m": [[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]],
        "contact_region_radius_m": 0.002,
        "min_contact_points_per_region": 1,
    }
    profile = {"calibration_file": "casing-top-base.json", "evaluator_kind": "multi_surface_mate"}
    evaluator = build_relation_evaluator("bearing-fit", profile, calibration, 0.001)
    assert isinstance(evaluator, CalibratedMultiSurfaceMateEvaluator)

    both_regions = evaluator.observe(_measurement(contact_points_world=[[1.0, 2.01, 3.0], [1.0, 2.02, 3.0]]))
    one_region = evaluator.observe(_measurement(contact_points_world=[[1.0, 2.01, 3.0]]))

    assert both_regions["task_success"] is True
    assert both_regions["metrics"]["fixture_contact_region_counts"] == [1, 1]
    assert one_region["task_success"] is False
    assert "multi_surface_contact_regions_criterion_failed" in one_region["reasons"]


def test_multi_surface_evaluator_stays_pending_without_multiple_calibrated_regions():
    calibration = _calibration() | {
        "evaluator_kind": "multi_surface_mate",
        "contact_regions_fixture_local_m": [[0.0, 0.0, 0.0]],
        "contact_region_radius_m": 0.002,
        "min_contact_points_per_region": 1,
    }
    profile = {"calibration_file": "casing-top-base.json", "evaluator_kind": "multi_surface_mate"}
    evaluator = build_relation_evaluator("bearing-fit", profile, calibration, 0.001)
    assert isinstance(evaluator, PendingRelationEvaluator)
    assert evaluator.pending_reason == "multi_surface_contact_regions_invalid"


def test_multi_surface_calibration_rejects_overlapping_regions():
    calibration = _calibration() | {
        "evaluator_kind": "multi_surface_mate",
        "contact_regions_fixture_local_m": [[0.0, 0.0, 0.0], [0.003, 0.0, 0.0]],
        "contact_region_radius_m": 0.002,
        "min_contact_points_per_region": 1,
    }
    profile = {"calibration_file": "casing-top-base.json", "evaluator_kind": "multi_surface_mate"}
    evaluator = build_relation_evaluator("bearing-fit", profile, calibration, 0.001)
    assert isinstance(evaluator, PendingRelationEvaluator)
    assert evaluator.pending_reason == "multi_surface_contact_regions_overlap"


def test_calibrated_axial_support_distinguishes_sidewall_jam_from_seat():
    calibration = _calibration()
    calibration['reference_relative_pose']['support_axis_fixture_local_unit'] = [0, 0, 1]
    calibration['tolerances']['min_axial_support_force_n'] = .1
    evaluator = CalibratedRelationEvaluator('bearing-fit', 'test', calibration)
    sidewall = replace(_measurement(), fixture_contact_normal_forces_world_n=[[2, 0, 0], [-2, 0, 0]])
    result = evaluator.observe(sidewall)
    assert not result['checks']['axial_support']
    assert 'axial_support_criterion_failed' in result['reasons']
    seated = replace(_measurement(), fixture_contact_normal_forces_world_n=[[0, 0, 2]])
    assert evaluator.observe(seated)['task_success']
    assert evaluator.observe(_measurement())['evaluator_status'] == 'UNKNOWN'
    empty = replace(_measurement(), fixture_contact_normal_forces_world_n=[])
    assert not evaluator.observe(empty)['checks']['axial_support']


def test_support_axis_is_transformed_from_fixture_not_assumed_world_z():
    calibration = _calibration()
    calibration['reference_relative_pose']['support_axis_fixture_local_unit'] = [1, 0, 0]
    calibration['tolerances']['min_axial_support_force_n'] = .1
    evaluator = CalibratedRelationEvaluator('bearing-fit', 'test', calibration)
    # Fixture Rz(+90) maps its +X support normal to world +Y.
    measurement = replace(_measurement(), fixture_contact_normal_forces_world_n=[[0, 2, 0]])
    assert evaluator.observe(measurement)['checks']['axial_support']
