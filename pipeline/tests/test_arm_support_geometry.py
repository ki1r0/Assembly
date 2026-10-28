from pathlib import Path
import numpy as np
import pytest
from arm_support_geometry import ArmSupportGeometry


@pytest.fixture(scope='module')
def arm():
    root=Path(__file__).parents[1]
    return ArmSupportGeometry(root/'assets/xarm7/xarm7_g1_simulation_assumption.urdf',[f'joint{i}' for i in range(1,8)])


def test_public_arm_mesh_detects_table_collision_missed_by_finger_checks(arm):
    # Recorded public robot postures: pregrasp versus failed transverse approach.
    clear=[1.3699875774,-1.6072478667,-2.2354506057,.4962613264,-.5169856576,.5600842621,-1.8593602830]
    contact=[1.5589735508,-1.6453869343,-2.2661602497,.4934478700,-.5078474283,.3598055542,-1.8668386936]
    base=np.eye(4);base[2,3]=.68
    assert arm.plane_clearances(clear,base,.68)['link5']>.004
    assert arm.plane_clearances(contact,base,.68)['link5']<0
    shifted=base.copy();shifted[2,3]+=.2
    original=arm.plane_clearances(contact,base,.68)
    translated=arm.plane_clearances(contact,shifted,.88)
    np.testing.assert_allclose(list(original.values()),list(translated.values()),atol=1e-12)


def test_arm_geometry_rejects_incomplete_robot_state(arm):
    with pytest.raises(ValueError):arm.plane_clearances([0]*6,np.eye(4),.68)
