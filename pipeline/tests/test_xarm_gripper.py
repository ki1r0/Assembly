from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from xarm_gripper import G1Geometry,antipodal_contact_quality


def test_final_envelope_contacts_must_still_support_the_assumed_friction_cones():
    points=np.array([[-.0146,.0177,0],[0,-.0177,0]])
    normals=np.array([[.407,.913,0],[0,-1.,0]])
    quality=antipodal_contact_quality(points,normals)
    assert quality['supported'] is False
    assert quality['required_friction_coefficient']>1.
    assert quality['assumed_friction_coefficient']==.5
    aligned=antipodal_contact_quality([[0,.02,0],[0,-.02,0]],[[0,1,0],[0,-1,0]])
    assert aligned['supported'] is True
    assert aligned['required_friction_coefficient']==0.


def geometry():
    path=Path(__file__).resolve().parents[1]/'assets/xarm7/xarm7_g1_simulation_assumption.urdf'
    if not path.exists():pytest.skip('Official G1 offline model not prepared')
    return G1Geometry(path)


def test_revolute_linkage_closes_monotonically_and_inverts_width():
    model=geometry()
    openings=[model.opening(q) for q in np.linspace(0,.85,20)]
    assert all(a>b for a,b in zip(openings,openings[1:]))
    assert .08<openings[0]<.10
    for width in [.005,.02,.04,.08]:
        assert model.opening(model.drive_for_opening(width))==pytest.approx(width,abs=1e-9)
    with pytest.raises(ValueError):model.drive_for_opening(.2)


def test_mimic_joints_keep_finger_faces_parallel():
    model=geometry()
    for q in [0,.4,.8]:
        left=model.link_transform('left_finger',q)
        right=model.link_transform('right_finger',q)
        assert np.allclose(left[:3,:3],np.eye(3),atol=1e-8)
        assert np.allclose(right[:3,:3],np.eye(3),atol=1e-8)
        assert np.isclose(left[1,3],-right[1,3])


def test_linkage_diagnostic_detects_measured_finger_translation_and_rotation():
    model=geometry();base=np.array([.2,.3,.7,1.,0.,0.,0.]);drive=.5
    link=model.link_transform('left_finger',drive)
    measured=np.r_[base[:3]+link[:3,3],Rotation.from_matrix(link[:3,:3]).as_quat()[[3,0,1,2]]]
    result=model.linkage_errors(base,drive,{'left_finger':measured})['left_finger']
    assert result['position_error_m']<1e-12
    assert result['angular_error_rad']<1e-12
    measured[0]+=.003;measured[3:]=Rotation.from_euler('x',.1).as_quat()[[3,0,1,2]]
    result=model.linkage_errors(base,drive,{'left_finger':measured})['left_finger']
    assert result['position_error_m']==pytest.approx(.003)
    assert result['angular_error_rad']==pytest.approx(.1)
    import json
    json.dumps(model.linkage_errors(base,drive,{'left_finger':measured.astype(np.float32)}))


def test_identical_width_reuses_exact_inverse_without_mesh_recomputation(monkeypatch):
    model=geometry();calls=[]
    original=model.finger_points
    def measured(drive):
        calls.append(drive)
        return original(drive)
    monkeypatch.setattr(model,'finger_points',measured)
    drive=model.drive_for_opening(.037123)
    count=len(calls)
    assert count>0
    for _ in range(100):assert model.drive_for_opening(.037123)==drive
    assert len(calls)==count
    assert model.opening(drive)==pytest.approx(.037123,abs=1e-9)
    assert model.drive_for_opening.cache_info().maxsize==128


def test_whole_gripper_support_clearance_uses_tcp_frame_and_arbitrary_plane():
    model=geometry()
    rotation=np.diag([1.,-1.,-1.])
    position=np.array([.3,.2,1.])
    points=model.world_points(position,rotation,.4)
    assert np.isclose(model.support_clearance(position,rotation,.4,[0,0,1],[0,0,.68]),points[:,2].min()-.68)
    assert np.isclose(model.support_clearance(position,rotation,.4,[1,0,0],[.1,0,0]),points[:,0].min()-.1)
    with pytest.raises(ValueError):model.support_clearance(position,rotation,.4,[0,0,0],[0,0,0])


def test_g1_screen_accepts_opposed_contact_and_rejects_overwidth():
    model=geometry()
    candidate={'position_m':[0,0,1.],'quaternion_wxyz':[1,0,0,0],
               'jaw_width_m':.05,'observed_width_m':.04}
    def surfaces(width):
        centre=model.pad_bounds(model.drive_for_opening(.04))[:,2].mean()
        points=np.array([[x,sign*width/2,z] for sign in [-1,1]
                         for x in [-.005,0,.005] for z in centre+np.array([-.0008,0,.0008])])
        normals=np.array([[0,sign,0] for sign in [-1,1] for _ in range(9)])
        return points+[0,0,1],normals
    points,normals=surfaces(.04)
    accepted,counts=model.screen_candidates([candidate],points,normals,.68,True)
    assert counts['initial_generated']==1
    assert len(accepted)==1
    assert accepted[0]['observed_width_m']==pytest.approx(.04)
    points,normals=surfaces(.09)
    rejected,counts=model.screen_candidates([candidate],points,normals,.68,True)
    assert not rejected
    assert counts['contact_band_rejected']==1


def test_self_filter_uses_actual_g1_linkage_not_panda_palm_coordinates():
    model=geometry()
    q=.4
    name='left_finger'
    frame=model.link_transform(name,q)
    inside=model.hand_points[name].mean(0)@frame[:3,:3].T+frame[:3,3]-[0,0,.172]
    assert model.contains_hand_points(np.array([inside,[10,10,10]]),q).tolist()==[True,False]
    assert not model.contains_hand_points(np.array([[0,0,-.005]]),q)[0]


def test_curved_pad_patch_is_not_replaced_by_its_narrow_central_pair():
    model=geometry()
    centre=model.pad_bounds(model.drive_for_opening(.036))[:,2].mean()
    points=np.array([[x,sign*(.018 if x==0 else .020),z]
                     for sign in [-1,1] for x in [-.005,0,.005]
                     for z in centre+np.array([-.0008,0,.0008])])
    normals=np.array([[0,sign,0] for sign in [-1,1] for _ in range(9)])
    candidate={'position_m':[0,0,1.],'quaternion_wxyz':[1,0,0,0],
               'jaw_width_m':.044,'observed_width_m':.036,
               'contact_pair_world_m':[[0,.018,1+centre],[0,-.018,1+centre]]}
    accepted,counts=model.screen_candidates([candidate],points+[0,0,1],normals,.68,True)
    assert len(accepted)==1
    assert accepted[0]['observed_width_m']==pytest.approx(.040)
    assert 'Finite-pad' in accepted[0]['contact_width_source']
    assert model.contact_offset_samples[1]<model.contact_z_range[1]
    chosen=accepted[0];local=np.array(chosen['contact_pair_world_m'])-chosen['position_m']
    pad=model.pad_bounds(model.drive_for_opening(chosen['observed_width_m']))
    assert (local[:,2]>=pad[0,2]-1e-12).all() and (local[:,2]<=pad[1,2]+1e-12).all()
    repeated,counts=model.screen_candidates([candidate,candidate],points+[0,0,1],normals,.68,True)
    assert len(repeated)==1
    assert counts['duplicate_geometry_rejected']==1


def test_real_innermost_face_is_three_mm_not_the_distal_twenty_mm_band():
    model=geometry()
    narrow=model.pad_bounds(model.drive_for_opening(.0115))
    wide=model.pad_bounds(model.drive_for_opening(.08))
    assert narrow[1,2]-narrow[0,2]==pytest.approx(.003,abs=1e-7)
    assert narrow[:,2].mean()-wide[:,2].mean()>.008
    assert np.all(narrow[0]<=narrow[1])


def test_cached_local_hulls_preserve_transformed_collision_distances():
    from scipy.spatial import ConvexHull
    model=geometry();rng=np.random.default_rng(21)
    for drive in [.1,.5,.8]:
        for name,mesh in model.hand_points.items():
            frame=model.link_transform(name,drive)
            vertices=mesh@frame[:3,:3].T+frame[:3,3]
            query=rng.uniform(vertices.min(0)-.003,vertices.max(0)+.003,(80,3))
            world=ConvexHull(vertices).equations
            local=model.hand_equations[name]
            old=(query@world[:,:3].T+world[:,3]).max(1)
            transformed=(query-frame[:3,3])@frame[:3,:3]
            new=(transformed@local[:,:3].T+local[:,3]).max(1)
            np.testing.assert_allclose(old,new,atol=1e-10)
            np.testing.assert_array_equal(old<-.0005,new<-.0005)
