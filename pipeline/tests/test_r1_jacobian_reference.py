"""Independent offline reference-point regressions; no Isaac or USD imports."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_kinematics import bounded_reference_increment, com_to_tcp_jacobian as shift


def skew(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


@pytest.mark.parametrize('angles', [[0.,0.,0.], [.4,-.9,1.2], [1.1,.2,-.7]])
def test_com_shift_matches_independent_tcp_finite_difference(angles):
    rotation = Rotation.from_euler('xyz',angles).as_matrix()
    tcp = np.array([.0079,0.,.09089])
    com = np.array([-1.4611e-7,6.6949e-8,-.031107])
    link = np.random.default_rng(5).normal(size=(6,6))
    raw = np.vstack([link[:3]-skew(rotation@com)@link[3:],link[3:]])
    corrected = shift(raw,rotation,tcp,com)
    eps = 1e-6
    numeric = np.empty((3,6))
    for i in range(6):
        def position(delta):
            moved = Rotation.from_rotvec(link[3:,i]*delta).as_matrix()@rotation
            return link[:3,i]*delta+moved@tcp
        numeric[:,i]=(position(eps)-position(-eps))/(2*eps)
    np.testing.assert_allclose(corrected[:3],numeric,atol=1e-10)
    np.testing.assert_array_equal(corrected[3:],link[3:])
    wrong = shift(raw,rotation,tcp,np.zeros(3))
    assert np.max(np.abs(wrong[:3]-numeric))>.01


def test_zero_com_preserves_original_shift():
    raw=np.random.default_rng(4).normal(size=(6,7))
    rotation=Rotation.from_euler('x',.4).as_matrix()
    tcp=np.array([.01,0.,.1])
    expected=np.vstack([raw[:3]-skew(rotation@tcp)@raw[3:],raw[3:]])
    np.testing.assert_array_equal(shift(raw,rotation,tcp,np.zeros(3)),expected)


def test_omitting_com_can_underlimit_without_relaxing_other_bounds():
    # A local instantaneous twist, not a simulator rollout or recorded command.
    tcp=np.array([.0079,0.,.09089])
    com=np.array([0.,0.,-.031107])
    link=np.array([[0.],[-.3],[0.],[1.],[0.],[0.]])
    raw=np.vstack([link[:3]-skew(com)@link[3:],link[3:]])
    true_tcp=shift(raw,np.eye(3),tcp,com)
    old=bounded_reference_increment(raw,tcp,np.array([.0025]),.05)
    corrected=bounded_reference_increment(raw,tcp-com,np.array([.0025]),.05)
    assert np.linalg.norm(true_tcp[:3]@old)/.05>.015
    assert np.linalg.norm(true_tcp[:3]@corrected)/.05==pytest.approx(.015)
    assert np.linalg.norm(true_tcp[3:]@old)/.05<.15
    assert np.max(np.abs(old))<=.0025
    assert np.max(np.abs(corrected))<=.0025
    # With the previous reference equal to measured joints, neither lead nor
    # position clamping masks this otherwise valid command-bound error.
    assert np.max(np.abs(old))<.008
    assert np.all((-1.<old)&(old<1.))


class Tensor(np.ndarray):
    """Only the CPU tensor operations used in the extracted backend snippets."""
    def cpu(self):return self
    def numpy(self):return np.asarray(self)
    def clone(self):return self.copy()
    @property
    def device(self):return 'cpu'


def tensor(value, dtype=float, device=None):
    return np.asarray(value,dtype=dtype).view(Tensor)


def backend_tree():
    path=Path(__file__).resolve().parents[1]/'generic_backend.py'
    return ast.parse(path.read_text())


def execute(nodes,namespace):
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'generic_backend.py','exec'),namespace)


@pytest.mark.parametrize('fixed_base',[True,False])
@pytest.mark.parametrize('com',[[0.,0.,0.],[0.,0.,-.031107]])
def test_actual_backend_wrapper_calls_production_helper(fixed_base,com):
    function=next(n for n in ast.walk(backend_tree()) if isinstance(n,ast.FunctionDef) and n.name=='sdk_tcp_jacobian')
    rotation=Rotation.from_euler('xyz',[.2,-.4,.9])
    pose=np.r_[np.zeros(3),rotation.as_quat()[[3,0,1,2]]]
    raw=np.random.default_rng(3).normal(size=(6,6))
    sdk=tensor(np.zeros((1,4,6,12)))
    ids=[1,3,5,7,9,11]
    sdk[0,1 if fixed_base else 2][:,ids]=raw
    coms=tensor(np.zeros((1,4,3)));coms[0,2]=com
    calls=[]
    def spy(*args):
        calls.append(args)
        return shift(*args)
    namespace=dict(np=np,Rotation=Rotation,com_to_tcp_jacobian=spy,
                   info={'left':{'body':2,'joints':ids,'offset':np.array([.0079,0.,.09089])}},
                   robot=SimpleNamespace(is_fixed_base=fixed_base,
                       root_physx_view=SimpleNamespace(get_jacobians=lambda:sdk),
                       data=SimpleNamespace(body_com_pos_b=coms)))
    execute([function],namespace)
    result=namespace['sdk_tcp_jacobian']('left',pose)
    assert len(calls)==1
    np.testing.assert_allclose(result,shift(raw,rotation.as_matrix(),[.0079,0.,.09089],com))
    np.testing.assert_array_equal(calls[0][3],com)


def test_all_backend_sdk_consumers_use_the_wrapper():
    tree=backend_tree()
    # No separate hand-coded raw shift can silently bypass the production helper.
    raw_calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='get_jacobians']
    assert len(raw_calls)==1
    for name in ['advance','compensate_gravity']:
        function=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
        assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='sdk_tcp_jacobian' for n in ast.walk(function))


def test_backend_telemetry_records_real_pipeline_and_resets_at_cadence():
    tree=backend_tree()
    advance=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='advance')
    loop=next(n for n in ast.walk(advance) if isinstance(n,ast.For) and ast.unparse(n.target)=='(arm, data)' and any(isinstance(x,ast.Assign) and ast.unparse(x.targets[0])=='current' for x in n.body))
    start=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='measured')
    clamp=lambda value,low,high:np.clip(value,low,high).view(Tensor)
    torch=SimpleNamespace(tensor=tensor,float32=np.float32,max=np.max,abs=np.abs,clamp=clamp)
    q=tensor([.9949,0.,0.]);goal=np.array([2.,2.,2.])
    data={'joints':[0,1,2],'grips':[3,4],'body':0}
    robot=SimpleNamespace(data=SimpleNamespace(
        soft_joint_pos_limits=tensor([[[-1.,1.],[-1.,1.],[-1.,1.]]]),
        joint_pos=tensor([[*q,.04,.04]]),body_com_pos_b=tensor(np.zeros((1,1,3)))))
    pose=np.array([0.,0.,1.,1.,0.,0.,0.])
    namespace=dict(np=np,torch=torch,arm='left',data=data,robot=robot,robot_name='r1',
        current_joints=q,joint_goals={'left':goal},joint_bias={'left':np.ones(3)*.0199},
        control_dt=.05,commanded_joints=tensor([[*q,.04,.04]]),joint_targets=tensor(np.zeros((1,5))),
        widths={'left':.08},jaw_joint_target=lambda width:width/2,tcp_jac_np=np.zeros((6,3)),bounded_reference_increment=bounded_reference_increment,
        controller_diagnostics={},controller_mode='reference_ik_high_pd',posture=np.zeros(3),
        posture_reference_sources={'left':'supplied_waypoint'},task_increment=np.zeros(3),
        posture_increment=np.zeros(3),current=pose,target=pose,reference=pose,
        reference_paused=True,step_index=228,n=6)
    for _ in range(2):execute(loop.body[start:],namespace)
    diagnostic=namespace['controller_diagnostics']['left']
    assert diagnostic['updates']==2
    assert diagnostic['activation_counts']['joint_step']==2
    assert diagnostic['activation_counts']['joint_position_clamp']==2
    assert diagnostic['activation_counts']['bias_clip']==2
    assert diagnostic['activation_counts']['reference_pause']==2
    last=diagnostic['last_control_update']
    assert last['posture_reference_source']=='supplied_waypoint'
    np.testing.assert_allclose(last['bias_after_rad'],.02)
    assert last['actual_posture_joints_rad']==list(q)
    assert last['joint_clamp_delta_rad'][0]<0
    # Run the two-simulation-second branch, including emission and reset.
    cadence=next(n for n in ast.walk(advance) if isinstance(n,ast.If) and ast.unparse(n.test)=='step_index % (2 * physics_rate) == 0')
    logs=[]
    namespace.update(step_index=240,physics_rate=120,info={'left':data},pose=lambda arm:pose,
        sdk_tcp_jacobian=lambda arm,value:np.zeros((6,3)),
        kin={'left':SimpleNamespace(jacobian=lambda value:np.zeros((6,3)))},
        finger_forces_world=lambda arm:[],jaw_pressure=lambda arm:[0.,0.],
        grip_force_targets={},targets={'left':pose},log=lambda name,value:logs.append(value))
    execute([cadence],namespace)
    emitted=logs[0]['controller_tracking']
    assert emitted['activation_ratios']['joint_step']==1.
    assert emitted['activation_ratios']['tcp_linear']==0.
    assert namespace['controller_diagnostics']=={}
    assert emitted['last_control_update']['start_step']==228
    assert emitted['last_control_update']['end_step']==234


@pytest.mark.parametrize('effort_limit',[100.,.1])
def test_actual_payload_gravity_uses_tcp_helper_and_retains_effort_clamp(effort_limit):
    from robot_kinematics import payload_support_wrench
    tree=backend_tree()
    functions=[next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
               for name in ['sdk_tcp_jacobian','compensate_gravity']]
    raw=np.random.default_rng(20).normal(size=(6,2))
    rotation=Rotation.from_euler('xyz',[.1,.3,-.4])
    pose=np.r_[np.zeros(3),rotation.as_quat()[[3,0,1,2]],np.zeros(6)]
    com=np.array([0.,0.,-.031107]);tcp=np.array([.0079,0.,.09089])
    states=tensor(np.zeros((1,2,13)));states[0,1]=pose
    coms=tensor(np.zeros((1,2,3)));coms[0,1]=com
    applied=[]
    robot=SimpleNamespace(is_fixed_base=True,
        root_physx_view=SimpleNamespace(get_jacobians=lambda:tensor(raw[None,None]),
            get_gravity_compensation_forces=lambda:tensor([[0.,0.]])),
        data=SimpleNamespace(body_state_w=states,body_com_pos_b=coms,
            joint_effort_limits=tensor([[effort_limit,effort_limit]])),
        set_joint_effort_target=lambda value:applied.append(value))
    payload={'mass_kg':1.2,'local_com_proxy_m':[.02,.01,.03]}
    namespace=dict(np=np,Rotation=Rotation,com_to_tcp_jacobian=shift,robot=robot,
        torch=SimpleNamespace(tensor=tensor,float32=np.float32,clamp=lambda v,lo,hi:np.clip(v,lo,hi)),
        info={'left':{'body':1,'joints':[0,1],'offset':tcp}},payload_estimates={'left':payload},
        cfg=SimpleNamespace(spawn=SimpleNamespace(rigid_props=SimpleNamespace(disable_gravity=False))))
    execute(functions,namespace)
    namespace['compensate_gravity']()
    wrench=payload_support_wrench(payload['mass_kg'],rotation.apply(payload['local_com_proxy_m']))
    expected=shift(raw,rotation.as_matrix(),tcp,com).T@wrench
    np.testing.assert_allclose(applied[0][0],np.clip(expected,-effort_limit,effort_limit),rtol=1e-6)
    assert np.max(np.abs(applied[0]))<=effort_limit
