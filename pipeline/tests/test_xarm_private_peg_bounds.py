import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation


def test_private_peg_recorder_uses_measured_g1_drive_and_full_hand_bounds():
    tree=ast.parse((Path(__file__).resolve().parents[1]/'generic_backend.py').read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
              and ast.unparse(n.test)=="robot_name == 'xarm7'"
              and any(isinstance(a,ast.Assign) and ast.unparse(a.targets[0])=='hand_points' for a in n.body))
    calls=[]
    def world_points(position,rotation,drive,*,support_only=False):
        calls.append((position,rotation,drive,support_only))
        return np.array([[-.01,-.02,-.03],[.04,.05,.06]])
    namespace=dict(np=np,robot_name='xarm7',robot=SimpleNamespace(data=SimpleNamespace(joint_pos=np.array([[.4]]))),
        info={'left':{'grips':[0]}},tcp=np.array([0,0,1,1,0,0,0]),rotation=Rotation.identity(),
        xarm_gripper=SimpleNamespace(world_points=world_points))
    exec(compile(ast.Module(body=[node],type_ignores=[]),'g1-recorder','exec'),namespace)
    assert calls[0][2]==.4
    assert calls[0][3] is True
    np.testing.assert_array_equal(namespace['bounds'],[[-.01,.04],[-.02,.05],[-.03,.06]])
