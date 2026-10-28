import numpy as np
from robot_kinematics import RobotKinematics


class RedundantTranslation(RobotKinematics):
    def __init__(self):
        self.limits=np.array([[-2.,2.]]*7)
    def fk(self,q):
        pose=np.eye(4);pose[0,3]=q[0]
        return pose


def test_ik_uses_redundancy_to_clear_support_without_changing_tcp():
    kin=RedundantTranslation();target=np.eye(4);target[0,3]=.2
    plain=kin.solve(target,np.zeros(7))
    clear=kin.solve(target,np.zeros(7),configuration_clearances=lambda q:[q[1]-.5])
    assert plain['reachable'] and clear['reachable']
    assert abs(plain['joints'][1])<1e-5
    assert clear['joints'][1]>=.5
    assert clear['position_error_m']<.001


def test_pose_reachability_does_not_certify_an_impossible_clearance():
    kin=RedundantTranslation()
    result=kin.solve(np.eye(4),np.zeros(7),configuration_clearances=lambda q:[-1.])
    assert result['position_error_m']<.001
    assert not result['reachable']
