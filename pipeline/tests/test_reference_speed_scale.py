import numpy as np
import pytest
from robot_kinematics import CartesianReference,reset_motion_references


def test_faster_reference_keeps_tracking_pause_and_reset_behavior():
    initial=np.array([.4,.2,.9,1.,0,0,0]);goal=initial.copy();goal[0]+=.1
    normal=CartesianReference(initial);fast=CartesianReference(initial,2.)
    assert fast.advance(goal,initial,.05)[0]-initial[0]==pytest.approx(2*(normal.advance(goal,initial,.05)[0]-initial[0]))
    behind=initial.copy();behind[0]-=.01
    before=fast.pose.copy();np.testing.assert_array_equal(fast.advance(goal,behind,.05),before)
    _,reset=reset_motion_references({'left':initial},2.);assert reset['left'].speed_scale==2.


@pytest.mark.parametrize('scale',[0,2.1,float('nan')])
def test_reference_scale_rejects_invalid_values(scale):
    with pytest.raises(ValueError):CartesianReference([0,0,0,1,0,0,0],scale)
