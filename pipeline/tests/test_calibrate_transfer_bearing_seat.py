from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tools.calibrate_transfer_bearing_seat import (
    DT, Limits, measure, stable_seat, summarize, trial_plan,
)


GEOMETRY = dict(
    bearing_wide_face_x_m=.01925,
    shaft_shoulder_x_m=-.0313130188,
    shoulder_annulus_radius_m=[.0135, .0152],
    shaft_axis_yz_m=[-.0000459, 0.],
    bore_axis_yz_m=[0., 0.],
    candidate_relative_position_m=[-.0505630188, -.0000459, 0.],
)


def test_simulated_probe_bootstraps_kit_before_usd_inspection(monkeypatch, tmp_path):
    import sys
    from types import ModuleType, SimpleNamespace
    from tools import calibrate_transfer_bearing_seat as probe

    events = []
    app_module = ModuleType('isaaclab.app')
    class Launcher:
        def __init__(self, **kwargs):
            events.append('started')
            self.app = SimpleNamespace(close=lambda: events.append('closed'))
    app_module.AppLauncher = Launcher
    monkeypatch.setitem(sys.modules, 'isaaclab', ModuleType('isaaclab'))
    monkeypatch.setitem(sys.modules, 'isaaclab.app', app_module)
    def inspect(_):
        assert events == ['started']
        raise RuntimeError('inspection sentinel')
    monkeypatch.setattr(probe, 'inspect_geometry', inspect)
    with pytest.raises(RuntimeError, match='inspection sentinel'):
        probe.simulate(SimpleNamespace(asset_root=tmp_path), None, Limits())
    assert events == ['started', 'closed']


def sample(*, offset=0., gap=0., tilt=0., released=True, penetration=0.,
           shoulder=True, contacts_available=True, velocity=None):
    rotation = Rotation.from_euler('y', 90, degrees=True)
    fixture = np.r_[[0., 0., .2], rotation.as_quat()[[3, 0, 1, 2]]]
    relative = np.array(GEOMETRY['candidate_relative_position_m'])
    relative[0] -= gap
    relative[1] += offset
    bq = (rotation * Rotation.from_euler('z', tilt, degrees=True)).as_quat()[[3, 0, 1, 2]]
    bearing = np.r_[fixture[:3] + rotation.apply(relative), bq]
    local_contact = np.array([GEOMETRY['shaft_shoulder_x_m'], .014, 0.])
    normal = rotation.apply([-1., 0., 0.] if shoulder else [0., 1., 0.])
    contacts = [dict(force_n=1.2, point_w_m=(fixture[:3] + rotation.apply(local_contact)).tolist(),
                     normal_w=normal.tolist(), separation_m=-penetration)]
    return measure(bearing, fixture, np.zeros(6) if velocity is None else velocity,
                   contacts if contacts_available else None, released, GEOMETRY, Limits())


def test_wide_face_matches_negative_shoulder_with_real_contact_requirement():
    result = sample()
    assert result['status'] == 'measured'
    assert result['instantaneous_seat']
    assert result['axial_gap_m'] == pytest.approx(0, abs=1e-12)
    assert result['radial_error_m'] == pytest.approx(0, abs=1e-12)
    assert result['shoulder_support_force_n'] == pytest.approx(1.2)


@pytest.mark.parametrize('arguments,failed_check', [
    ({'offset': .003}, 'coaxiality'),
    ({'gap': .008}, 'axial_shoulder'),
    ({'tilt': 8.}, 'orientation'),
    ({'released': False}, 'released'),
    ({'penetration': .001}, 'nonpenetrating'),
    ({'shoulder': False}, 'shoulder_contact'),
    ({'velocity': [.01, 0., 0., 0., 0., 0.]}, 'linear_speed'),
    ({'velocity': [0., 0., 0., .1, 0., 0.]}, 'angular_speed'),
])
def test_each_negative_rejects_a_seated_pose(arguments, failed_check):
    result = sample(**arguments)
    assert not result['instantaneous_seat']
    assert result['checks'][failed_check] is False


@pytest.mark.parametrize('arguments', [
    {'contacts_available': False},
    {'penetration': float('nan')},
    {'released': None},
    {'velocity': [float('inf'), 0., 0., 0., 0., 0.]},
])
def test_missing_or_nonfinite_measurement_is_unknown_not_clear(arguments):
    result = sample(**arguments)
    assert result['status'] == 'UNKNOWN'
    assert result['instantaneous_seat'] is False


def test_transient_contact_or_bad_sample_does_not_satisfy_stability():
    limits = Limits()
    count = int(np.ceil(limits.dwell_s / DT))
    good = sample()
    assert not stable_seat([good] * (count - 1), limits)
    assert stable_seat([good] * count, limits)
    assert not stable_seat([good] * (count - 1) + [sample(contacts_available=False)], limits)


def test_end_for_end_flip_is_not_accepted_as_coaxial_orientation():
    assert sample(tilt=180.)['checks']['orientation'] is False


def passing_trials():
    good = sample()
    count = int(np.ceil(Limits().dwell_s / DT))
    return dict(drop_1mm=[sample(gap=.001)] + [good] * count,
                drop_3mm=[sample(gap=.003)] + [good] * count,
                lateral=[sample(offset=.003)], height=[sample(gap=.008)],
                tilt=[sample(tilt=8.)], unreleased=[sample(released=False)])


def test_source_initialized_pose_without_measured_drop_cannot_be_positive():
    trials = passing_trials()
    trials['drop_1mm'] = [sample()] * 100
    result = summarize(trials, GEOMETRY, Limits())
    assert not result['physical_probe_passed']


def test_negative_that_self_corrects_is_not_mislabelled_as_final_failure():
    trials = passing_trials()
    trials['height'].append(sample())
    assert summarize(trials, GEOMETRY, Limits())['physical_probe_passed']
    trials['height'] = [sample()]
    assert not summarize(trials, GEOMETRY, Limits())['physical_probe_passed']


def test_even_passing_physical_suite_cannot_publish_native_success_or_calibration():
    result = summarize(passing_trials(), GEOMETRY, Limits())
    assert result['physical_probe_passed']
    assert result['native_success'] is False
    assert result['calibration_status'] == 'pending'
    assert result['physical_gt_verified'] is False
    assert len(result['trials']) == 6
    json.dumps(result, allow_nan=False)


def test_suite_requires_every_real_case_not_only_positives():
    trials = passing_trials()
    del trials['tilt']
    assert not summarize(trials, GEOMETRY, Limits())['physical_probe_passed']
    assert {x['name'] for x in trial_plan()} == set(passing_trials())


def test_unreleased_negative_must_be_at_the_seat_not_just_any_held_pose():
    trials = passing_trials()
    trials['unreleased'] = [sample(released=False, gap=.008)]
    assert not summarize(trials, GEOMETRY, Limits())['physical_probe_passed']


def test_free_perturbation_negatives_cannot_all_remain_supported():
    trials = passing_trials()
    trials['lateral'] = [sample(offset=.003, released=False)]
    assert not summarize(trials, GEOMETRY, Limits())['physical_probe_passed']


def test_tighter_penetration_limit_is_not_implicitly_relaxed():
    limits = replace(Limits(), dwell_s=1.)
    assert not stable_seat([sample()] * 60, limits)
    assert not sample(penetration=.00011)['checks']['nonpenetrating']
