import numpy as np
import pytest
from xarm_robot_cfg import friction_tracking_bias
from xarm_robot_cfg import tracking_integral_increment
from xarm_robot_cfg import tracking_integral_active


def test_paused_nearby_reference_can_clear_tracking_deadlock_far_from_final_goal():
    from scipy.spatial.transform import Rotation
    current=np.array([.3,.26,.724,1.,0.,0.,0.])
    target=current.copy();target[3:]=Rotation.from_rotvec([1.3,0,0]).as_quat()[[3,0,1,2]]
    reference=current.copy();reference[0]+=.0025
    reference[3:]=Rotation.from_rotvec([.015,0,0]).as_quat()[[3,0,1,2]]
    assert tracking_integral_active(target,reference,current,True)
    assert not tracking_integral_active(target,reference,current,False)
    reference[0]+=.01
    assert not tracking_integral_active(target,reference,current,True)
    reference=current.copy();reference[3:]=Rotation.from_rotvec([.1,0,0]).as_quat()[[3,0,1,2]]
    assert not tracking_integral_active(target,reference,current,True)
    assert tracking_integral_active(current,current,current,False)


def test_bias_is_continuous_signed_and_bounded_by_declared_friction():
    error=np.array([-.1,-.001,0.,.001,.1])
    bias=friction_tracking_bias(error,np.ones(5),np.full(5,400.))
    assert bias[2]==0.
    np.testing.assert_allclose(bias,-bias[::-1])
    assert np.max(np.abs(bias))<=.0025
    assert np.all(bias*error>=0.)
    np.testing.assert_array_equal(friction_tracking_bias(error,np.zeros(5),np.full(5,400.)),np.zeros(5))


@pytest.mark.parametrize('error,friction,stiffness', [([0.],[1.],[0.]),([0.],[-1.],[400.]),([float('nan')],[1.],[400.]),([0.,0.],[1.],[400.])])
def test_invalid_joint_models_are_rejected(error,friction,stiffness):
    with pytest.raises(ValueError):friction_tracking_bias(error,friction,stiffness)


def test_travel_cannot_wind_up_the_fine_tracking_integral():
    previous=np.array([.015,-.006,0.])
    delta=tracking_integral_increment([.01,-.02,.03],previous,.05,False)
    assert np.all(np.abs(previous+delta)<=np.abs(previous))
    assert (previous+delta)[2]==0.
    np.testing.assert_allclose(tracking_integral_increment([.001,-.002,.003],np.zeros(3),.05,True),[.000025,-.00005,.000075])


def test_saturated_reference_cannot_accumulate_outward_integral():
    bias=np.zeros(2)
    for _ in range(400):bias+=tracking_integral_increment([.02,-.02],bias,.05,True)
    np.testing.assert_array_equal(bias,[0.,0.])
    previous=np.array([.02,-.02])
    delta=tracking_integral_increment([.003,-.003],previous,.05,True)
    assert np.all(np.abs(previous+delta)<np.abs(previous))


def test_reverse_integral_unwinds_even_when_actuator_is_saturated():
    np.testing.assert_allclose(tracking_integral_increment([-.001,.001],[.02,-.02],.05,True),[-.000025,.000025])


def test_friction_feedforward_counts_toward_the_existing_lead_budget():
    np.testing.assert_array_equal(tracking_integral_increment([.006,-.006],[0.,0.],.05,True,[.0025,-.0025]),[0.,0.])
    with pytest.raises(ValueError):tracking_integral_increment([.001],[0.],.05,True,[float('nan')])


def test_integral_decay_does_not_depend_on_control_frequency():
    initial=np.array([.02,-.02])
    end=[]
    for rate in [20,40]:
        bias=initial.copy()
        for _ in range(rate*5):bias+=tracking_integral_increment([1,1],bias,1/rate,False)
        end.append(bias)
    np.testing.assert_allclose(end[0],end[1],atol=1e-15)
