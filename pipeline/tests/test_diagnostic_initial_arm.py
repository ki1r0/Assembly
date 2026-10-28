from pathlib import Path

import pytest

from xarm_robot_cfg import diagnostic_initial_joints


URDF=Path(__file__).parents[1]/'assets/xarm7/xarm7_g1_simulation_assumption.urdf'


def test_only_finite_inset_arm_joint_values_are_accepted():
    values=[0.,-.5,0.,1.,0.,.5,0.]
    result=diagnostic_initial_joints(values,URDF)
    assert list(result)==[f'joint{i}' for i in range(1,8)]
    assert list(result.values())==values
    for invalid in [values[:-1],[float('nan')]*7,[100.]*7]:
        with pytest.raises(ValueError):diagnostic_initial_joints(invalid,URDF)
