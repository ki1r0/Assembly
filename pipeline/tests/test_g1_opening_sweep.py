from pathlib import Path
import ast
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from xarm_gripper import G1Geometry
from assembly_skills import aligned_tcp


@pytest.fixture(scope='module')
def geometry():
    return G1Geometry(Path(__file__).parents[1]/'assets/xarm7/xarm7_g1_simulation_assumption.urdf')


def test_hull_support_preserves_full_mesh_bounds_and_arbitrary_plane_extrema(geometry):
    rng=np.random.default_rng(7)
    for drive in [0.,.17,.425,.67,.85]:
        rotation=Rotation.random(random_state=rng).as_matrix()
        position=rng.uniform(-1,1,3);axis=rng.normal(size=3);axis/=np.linalg.norm(axis)
        plane=rng.uniform(-1,1,3)
        full=geometry.world_points(position,rotation,drive)
        support=geometry.world_points(position,rotation,drive,support_only=True)
        np.testing.assert_allclose([support.min(0),support.max(0)],[full.min(0),full.max(0)],atol=1e-12)
        expected=((full-plane)@axis).min()
        assert geometry.support_clearance(position,rotation,drive,axis,plane)==pytest.approx(expected,abs=1e-12)


@pytest.mark.parametrize('angles,normal',[
    ([0,0,0],[0,0,1]),([90,0,0],[0,0,1]),([43,21,78],[.2,-.3,1.]),
])
def test_analytic_stroke_minimum_includes_all_vertices_and_interior_angles(geometry,angles,normal):
    rotation=Rotation.from_euler('xyz',angles,degrees=True).as_matrix()
    p=np.array([.4,-.1,.8]);plane=np.array([.5,-.15,.75]);start,end=.12,.82
    analytic=geometry.swept_support_clearance(p,rotation,start,end,normal,plane)
    dense=min(geometry.support_clearance(p,rotation,q,normal,plane) for q in np.linspace(start,end,65))
    assert analytic<=dense+1e-12
    assert dense-analytic<.00001
    coefficients=geometry.motion_coefficients
    for q in [.19,.54,.77]:
        prediction=coefficients[0]*np.cos(q)+coefficients[1]*np.sin(q)+coefficients[2]
        np.testing.assert_allclose(prediction,geometry.world_points([0,0,0],np.eye(3),q,support_only=True),atol=1e-12)


def test_closed_clearance_does_not_prove_full_opening_is_safe(geometry):
    rotation=Rotation.from_euler('x',90,degrees=True).as_matrix()
    closed=geometry.drive_for_opening(.0266);opened=geometry.drive_for_opening(.08)
    p=np.zeros(3);normal=[0,0,1];plane=np.zeros(3)
    p[2]=.003-geometry.support_clearance(p,rotation,closed,normal,plane)
    swept=geometry.swept_support_clearance(p,rotation,closed,opened,normal,plane)
    assert swept<-.01
    p[2]+=.003-swept
    assert geometry.swept_support_clearance(p,rotation,closed,opened,normal,plane)==pytest.approx(.003,abs=1e-12)


def test_unsupported_stroke_is_not_silently_clipped(geometry):
    with pytest.raises(ValueError):geometry.swept_support_clearance([0,0,0],np.eye(3),-.1,.4,[0,0,1],[0,0,0])


def test_assembly_release_pose_reserves_the_actual_80mm_opening(geometry):
    tree=ast.parse((Path(__file__).parents[1]/'generic_backend.py').read_text())
    function=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='assembly_targets')
    closed=geometry.drive_for_opening(.0266)
    namespace=dict(np=np,Rotation=Rotation,aligned_tcp=aligned_tcp,robot_name='xarm7',
        active_arm='left',info={'left':{'grips':[0]}},xarm_gripper=geometry,
        robot=SimpleNamespace(data=SimpleNamespace(joint_pos=np.array([[closed]]))))
    exec(compile(ast.Module(body=[function],type_ignores=[]),'targets','exec'),namespace)
    quaternion=Rotation.from_euler('x',90,degrees=True).as_quat()[[3,0,1,2]]
    tcp=np.r_[[0.,0.,.2],quaternion]
    moving={'position_m':[0.,0.,.15],'axis_world':[0,0,1],'axial_protrusion_m':.022}
    target={'position_m':[0.,0.,0.],'axis_world':[0,0,1]}
    _,pre,seat=namespace['assembly_targets'](tcp,moving,target,0.,'nearest')
    rotation=Rotation.from_quat(seat[[4,5,6,3]]).as_matrix()
    assert geometry.swept_support_clearance(seat[:3],rotation,closed,geometry.drive_for_opening(.08),[0,0,1],[0,0,0])>=.003-1e-12
    assert pre[2]>seat[2]
