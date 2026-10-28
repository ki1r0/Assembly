"""Exercise the backend setter without importing Isaac Sim."""
import ast
from pathlib import Path

import numpy as np


def test_jaw_updates_preserve_arm_targets_for_panda_xarm_and_dual_arm():
    tree = ast.parse((Path(__file__).parents[1] / 'generic_backend.py').read_text())
    block = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == 'torque_controller is None'
                 and 'jaw_reference_step' in ast.unparse(n))
    calls = [n for n in ast.walk(block) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == 'robot.set_joint_position_target']
    assert len(calls) == 1
    code = compile(ast.fix_missing_locations(ast.Module(
        body=[ast.Expr(value=calls[0])], type_ignores=[])), 'jaw_update', 'exec')
    for size, groups in ((9, [[7, 8]]), (13, [[7]]), (18, [[14, 15], [16, 17]])):
        class Robot:
            def __init__(self):
                self.target = np.full((1, size), .25)

            def set_joint_position_target(self, value, joint_ids=None):
                assert joint_ids is not None
                self.target[:, joint_ids] = value

        robot = Robot()
        endpoint = np.ones((1, size))
        for ids in groups:
            endpoint[:, ids] = .012
            exec(code, dict(robot=robot, joint_targets=endpoint, data={'grips': ids}))
        grips = [i for ids in groups for i in ids]
        arms = [i for i in range(size) if i not in grips]
        np.testing.assert_array_equal(robot.target[:, arms], .25)
        np.testing.assert_array_equal(robot.target[:, grips], .012)
