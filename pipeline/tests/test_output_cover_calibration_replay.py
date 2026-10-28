import json
import pytest
import settings
from hub_evaluator import IndependentSeatEvaluator,SeatTolerances,SeatMeasurement


def test_user_accepted_output_cover_tilt_keeps_other_physical_requirements():
    evaluator=IndependentSeatEvaluator(SeatTolerances(axial_depth_m=.031,tilt_tolerance_deg=2.2),calibration_status='calibrated')
    good=dict(axial_depth_m=.03328,radial_error_m=.0004,tilt_deg=2.12,penetration_m=.000005,released=True,stable=True,contact_valid=True,settle_speed_mps=0.,settle_window_s=.5)
    assert evaluator.evaluate(SeatMeasurement(**good)).task_success
    for changed in [dict(tilt_deg=2.21),dict(released=False),dict(stable=False),dict(contact_valid=False),dict(radial_error_m=.005),dict(penetration_m=.002)]:
        assert not evaluator.evaluate(SeatMeasurement(**(good|changed))).task_success


@pytest.mark.parametrize('tilt_tolerance_deg',[2.0,2.2])
def test_output_cover_real_endpoint_replay_positive_and_negatives(tilt_tolerance_deg):
    calibration=json.loads((settings.ROOT/'config/hub_output_scale001_evaluator_calibration.json').read_text())
    evaluator=IndependentSeatEvaluator(SeatTolerances(axial_depth_m=.031,tilt_tolerance_deg=tilt_tolerance_deg),calibration_status='calibrated')
    for case in [calibration['positive'],*calibration['negatives']]:
        source=settings.ROOT/case['run']/'private/evaluator-trace.json'
        if not source.exists():pytest.skip('Preserved physical calibration artifacts not installed')
        actual=json.loads(source.read_text())[-1]
        result=evaluator.evaluate(SeatMeasurement.from_mapping(actual['metrics']))
        assert result.status==case['regraded_status']
        # Tilted support is unknown, so prove it is rejected independently of sleep reports.
        if 'tilt-negative' in case['run']:
            values=actual['metrics'] | {'contact_valid':True}
            assert 'tilt_out_of_tolerance' in evaluator.evaluate(SeatMeasurement.from_mapping(values)).reasons
        if case is calibration['positive']:
            assert actual['contact_valid_samples_in_window']==60
            assert not actual['pickup_observed']
            assert result.physical_criteria_passed
