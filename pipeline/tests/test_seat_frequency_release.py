import numpy as np
import pytest
from hub_evaluator import settle_window_samples,axisymmetric_seat_stable,settle_window_contact,robot_release_verified,aabb_separation


@pytest.mark.parametrize('rate',[120,240,480])
def test_half_second_windows_preserve_physical_duration_and_penetration_peaks(rate):
    count=settle_window_samples(rate)
    assert count/rate==.5
    positions=np.zeros((count,3));axes=np.tile([0.,0.,1.],(count,1))
    assert axisymmetric_seat_stable(positions,axes,count)
    assert not axisymmetric_seat_stable(positions[:-1],axes[:-1],count)
    penetration=np.zeros(count);penetration[0]=.002
    assert settle_window_contact([True]+[False]*(count-1),penetration,count)==(True,.002)
    positions[-1,0]=.002
    assert not axisymmetric_seat_stable(positions,axes,count)


def test_release_requires_measured_withdrawal_and_no_remaining_support():
    assert robot_release_verified([.075],[[0.,0.]],withdrawn=True)
    assert not robot_release_verified([.08],[[0.,0.]],withdrawn=False)
    assert not robot_release_verified([.075],[[0.,.1]],withdrawn=True)
    assert not robot_release_verified([.075],[[0.,0.]])
    assert robot_release_verified([.01],[[0.,0.]],withdrawn=True)


def test_conservative_hand_bounds_must_be_separated_from_the_object():
    obj=np.array([[0.,0.,0.],[.1,.1,.02]])
    assert aabb_separation(obj,[[.02,.02,.01],[.04,.04,.04]])==0.
    assert aabb_separation(obj,[[.02,.02,.03],[.04,.04,.04]])==pytest.approx(.01)
    with pytest.raises(ValueError):aabb_separation(obj,[[1,1,1],[0,0,0]])
