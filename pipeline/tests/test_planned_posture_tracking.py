import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_kinematics import nullspace_posture_increment, posture_reference


def test_supplied_posture_drives_only_nullspace():
    rng=np.random.default_rng(42)
    jacobian=rng.normal(size=(6,7))
    current=np.zeros(7)
    home=np.ones(7)*-.2
    supplied=np.linspace(-.5,.5,7)
    limits=np.tile([-1.,1.],(7,1))
    reference=posture_reference(supplied,home,limits)
    increment=nullspace_posture_increment(jacobian,current,reference,.05)
    null=np.eye(7)-np.linalg.pinv(jacobian)@jacobian
    np.testing.assert_allclose(increment,null@(supplied-current)*.05*.05,atol=1e-15)
    np.testing.assert_allclose(jacobian@increment,0,atol=1e-15)
    assert np.dot(increment,supplied-current)>0
    assert not np.allclose(increment,nullspace_posture_increment(jacobian,current,home,.05))
    supplied[:]=0
    assert np.any(reference!=0)


def test_missing_posture_uses_home_not_previous_reference():
    home=np.arange(7)*.1
    limits=np.tile([-1.,1.],(7,1))
    posture_reference(-home,home,limits)
    reference=posture_reference(None,home,limits)
    np.testing.assert_array_equal(reference,home)
    reference[:]=0
    assert np.any(home!=0)


@pytest.mark.parametrize('supplied',[
    [0.]*6, [[0.]*7], [float('nan')]*7, [float('inf')]*7,
    [1.000001]+[0.]*6, [-1.000001]+[0.]*6,
])
def test_invalid_supplied_posture_is_rejected(supplied):
    with pytest.raises(ValueError):
        posture_reference(supplied,np.zeros(7),np.tile([-1.,1.],(7,1)))


def test_inset_limit_endpoints_are_accepted():
    limits=np.tile([-1.,1.],(7,1))
    for reference in [limits[:,0],limits[:,1]]:
        np.testing.assert_array_equal(posture_reference(reference,np.zeros(7),limits),reference)


def test_move_retains_validates_and_resets_high_pd_reference():
    # Execute only the real move() posture plumbing, without importing Isaac.
    source=Path(__file__).resolve().parents[1]/'generic_backend.py'
    tree=ast.parse(source.read_text())
    move=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='move')
    branch=next(node for node in ast.walk(move) if isinstance(node,ast.If) and ast.unparse(node.test)=="controller_mode == 'reference_ik_high_pd'")
    code=compile(ast.Module(body=[branch],type_ignores=[]),str(source),'exec')

    class Home:
        def __getitem__(self,key):return self
        def cpu(self):return self
        def numpy(self):return np.zeros(7)

    references={}
    namespace=dict(controller_mode='reference_ik_high_pd',robot_name='panda',arm='left',
                   posture_targets={'left':np.ones(7)*.5},posture_references=references,posture_reference_sources={},
                   posture_reference=posture_reference,base_joint_targets=Home(),
                   info={'left':{'joints':list(range(7))}},
                   kin={'left':SimpleNamespace(limits=np.tile([-1.,1.],(7,1)))})
    exec(code,namespace)
    np.testing.assert_array_equal(references['left'],np.ones(7)*.5)
    assert namespace['posture_reference_sources']['left']=='supplied_waypoint'
    namespace['posture_targets']={'left':np.ones(7)*2}
    with pytest.raises(ValueError):exec(code,namespace)
    np.testing.assert_array_equal(references['left'],np.ones(7)*.5)
    for missing in [None,{}, {'right':np.ones(7)}]:
        namespace['posture_targets']=missing
        exec(code,namespace)
        np.testing.assert_array_equal(references['left'],np.zeros(7))
        assert namespace['posture_reference_sources']['left']=='home_fallback'

    from robot_kinematics import transform
    calls=[]
    namespace.update(robot_name='xarm7',value=np.array([.4,0.,.8,1.,0.,0.,0.]),transform=transform,
                     robot=SimpleNamespace(data=SimpleNamespace(joint_pos=Home())))
    def solve(target,seed):
        calls.append(target)
        return {'reachable':True,'joints':np.full(7,.25)}
    namespace['kin']['left'].solve=solve
    exec(code,namespace)
    np.testing.assert_array_equal(references['left'],np.full(7,.25))
    assert namespace['posture_reference_sources']['left']=='target_ik'
    assert len(calls)==1
    namespace['kin']['left'].solve=lambda *args:{'reachable':False}
    exec(code,namespace)
    np.testing.assert_array_equal(references['left'],np.zeros(7))
    assert namespace['posture_reference_sources']['left']=='home_fallback'
