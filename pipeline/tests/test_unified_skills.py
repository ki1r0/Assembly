import numpy as np
from scipy.spatial.transform import Rotation
from assembly_skills import AssemblySkill,grasp_candidates,aligned_tcp,selected_points
from robot_kinematics import RobotKinematics,transform

def test_alignment_transports_feature_not_tcp_to_target():
    tcp=np.array([.3,.2,.9,1.,0,0,0]);moving={'position_m':[.31,.2,.85],'axis_world':[1.,0,0]};target={'position_m':[.5,-.1,.8],'axis_world':[0.,0,1.]}
    actual=aligned_tcp(tcp,moving,target,.02)
    rotation=Rotation.from_quat(actual[[4,5,6,3]])
    mapped=actual[:3]+rotation.apply(np.array(moving['position_m'])-tcp[:3])
    np.testing.assert_allclose(mapped,[.5,-.1,.82],atol=1e-10)
    np.testing.assert_allclose(rotation.apply(moving['axis_world']),target['axis_world'],atol=1e-10)

def test_coaxial_sign_ambiguity_does_not_flip_chosen_seating_face():
    import pytest
    tcp=np.array([.3,.2,.9,1.,0,0,0])
    moving={'position_m':[.31,.2,.85],'axis_world':[0.,0,-1.]}
    target={'position_m':[.5,-.1,.8],'axis_world':[0.,0,1.]}
    for mode,expected_angle in [('nearest',0.),('parallel',np.pi),('antiparallel',0.)]:
        actual=aligned_tcp(tcp,moving,target,.02,mode)
        rotation=Rotation.from_quat(actual[[4,5,6,3]])
        assert rotation.magnitude()==pytest.approx(expected_angle)
        np.testing.assert_allclose(actual[:3]+rotation.apply(np.array(moving['position_m'])-tcp[:3]),[.5,-.1,.82],atol=1e-10)
    with pytest.raises(ValueError):
        aligned_tcp(tcp,moving,target,0.,'unknown')


def test_release_clearance_uses_aligned_hand_and_outward_target_normal():
    from assembly_skills import assembly_release_clearance,hand_plane_clearance
    tcp=np.array([.3,.2,.9,0.,1.,0.,0.])
    moving={'position_m':[.3,.2,.89],'axis_world':[0.,0,-1.]}
    target={'position_m':[.5,-.1,.8],'axis_world':[0.,0,1.]}
    width=.05
    for mode in ('nearest','parallel','antiparallel'):
        gap=assembly_release_clearance(tcp,moving,target,width,mode)
        actual=aligned_tcp(tcp,moving,target,gap,mode)
        clearance=hand_plane_clearance(actual[:3],Rotation.from_quat(actual[[4,5,6,3]]),width,target['axis_world'],target['position_m'])
        assert clearance>=.003-1e-10

def test_prompt_distinguishes_cad_datum_from_seating_face_and_axis_sign():
    from task_profiles import GENERIC_SYSTEM
    assert 'whose seating face offset is already included' not in GENERIC_SYSTEM
    assert 'A CAD datum is not automatically a physical seating face' in GENERIC_SYSTEM
    assert 'axis_alignment=nearest' in GENERIC_SYSTEM

def test_full_grasp_frames_for_upright_horizontal_and_rim():
    rng=np.random.default_rng(3)
    for size in [[.012,.012,.04],[.04,.012,.012],[.035,.012,.02]]:
        points=(rng.random((300,3))-.5)*size+[.3,.2,.71]
        candidates=grasp_candidates(points,.68)
        assert candidates
        assert any(abs(c['approach_axis_world'][2])<.9 for c in candidates)
        for c in candidates:
            assert .002<c['observed_width_m']<.072
            rotation=Rotation.from_quat(np.array(c['quaternion_wxyz'])[[1,2,3,0]])
            np.testing.assert_allclose(rotation.apply([0,0,1]),c['approach_axis_world'],atol=1e-10)

def test_rgbd_component_comes_from_current_selected_region():
    d=np.ones((50,50));d[15:35,15:35]=.95
    K=np.array([[500.,0,25],[0,500.,25],[0,0,1]])
    T=np.diag([1.,-1,-1,1]);T[:3,3]=[.3,.2,1]
    points=selected_points(d,K,T,[25,25],[10,10,40,40],0.)
    assert len(points)==400
    np.testing.assert_allclose(points[:,2],.05)

def test_ik_respects_robot_joint_limits():
    model=RobotKinematics(np.eye(4),[(np.eye(4),np.eye(4),0,np.array([0.,0,1.]))],['joint'],[[-.4,.4]],[.2,0,0])
    goal=model.fk([.2]);good=model.solve(goal,np.array([0.]));assert good['reachable']
    impossible=model.fk([1.]);bad=model.solve(impossible,np.array([0.]));assert not bad['reachable'];assert -.4<=bad['joints'][0]<=.4

def test_skill_spec_is_relational_and_coordinate_free():
    for moving,mate in [('bolt','cover_hole'),('cover','casing_opening')]:
        spec=AssemblySkill(moving,mate).public();assert spec['relation']=='coaxial_insertion';assert not any('position' in k or 'world' in k for k in spec)

def test_low_side_grasp_rejects_hand_table_collision():
    from assembly_skills import clears_support_plane
    rng=np.random.default_rng(4);pts=(rng.random((300,3))-.5)*[.012,.012,.04]+[.3,.2,.70]
    candidates=grasp_candidates(pts,.68)
    assert any(not clears_support_plane(c,.68) for c in candidates)
    assert any(clears_support_plane(c,.68) for c in candidates if abs(c['approach_axis_world'][2])>.99)

def test_generic_clock_geometry_without_target_identity():
    import cv2
    from assembly_skills import measure_pattern
    rgb=np.full((200,200,3),220,np.uint8)
    cv2.circle(rgb,(152,70),5,(20,20,20),-1);cv2.circle(rgb,(100,40),5,(20,20,20),-1)
    found=measure_pattern(rgb,[100,100],[0,0,200,200])
    assert len(found)==2
    assert abs(found[0]['clock_hour']-0)<.05
    assert abs(found[1]['clock_hour']-2)<.05

def test_public_cad_registration_on_asymmetric_partial_shape():
    from cad_registration import register
    rng=np.random.default_rng(8);cad=np.r_[rng.normal(0,1,(350,3))*[.018,.007,.003],rng.normal(0,1,(100,3))*[.004,.008,.003]+[.022,.003,.008]]
    rotation=Rotation.from_euler('xyz',[.4,-.3,.9]).as_matrix();translation=np.array([.3,.2,.8]);observed=cad@rotation.T+translation
    fit=register(cad,observed)
    assert fit['rms_m']<.0001
    np.testing.assert_allclose(np.array(fit['world_from_cad'])[:3,3],translation,atol=.0002)

def test_joint_limits_alone_do_not_bound_tcp_reference_speed():
    from robot_kinematics import bounded_reference_increment
    jac=np.zeros((6,7));jac[0,:]=.6;jac[3,:]=1
    dq=np.full(7,.0025);dt=.05
    assert np.linalg.norm(jac[:3]@dq)/dt>.08
    bounded=bounded_reference_increment(jac,[0,0,0],dq,dt)
    assert np.linalg.norm(jac[:3]@bounded)/dt<=.015+1e-12
    assert np.linalg.norm(jac[3:]@bounded)/dt<=.15+1e-12

def test_cartesian_reference_stays_on_line_and_waits_for_tracking():
    from robot_kinematics import CartesianReference
    initial=np.array([.4,.2,1.,1.,0,0,0]);goal=initial.copy();goal[:3]+=[.1,.15,-.3]
    reference=CartesianReference(initial);measured=initial.copy()
    first=reference.advance(goal,measured,.05)
    for _ in range(2000):
        measured=reference.pose.copy();actual=reference.advance(goal,measured,.05)
        line=goal[:3]-initial[:3];delta=actual[:3]-initial[:3]
        assert np.linalg.norm(np.cross(line,delta))<1e-10
    np.testing.assert_allclose(reference.pose,goal,atol=1e-9)
    reference=CartesianReference(initial);distant=initial.copy();distant[0]-=.01
    np.testing.assert_array_equal(reference.advance(goal,distant,.05),initial)

def test_invalid_target_recovery_rebuilds_reference_from_measured_pose():
    from robot_kinematics import reset_motion_references
    measured=np.array([.4,.2,.9,1.,0,0,0]);targets,references=reset_motion_references({'left':measured})
    measured[0]=100
    assert targets['left'][0]==.4
    np.testing.assert_array_equal(references['left'].pose,targets['left'])
    goal=targets['left'].copy();goal[0]+=.01
    advanced=references['left'].advance(goal,targets['left'],.05)
    assert .4<advanced[0]<.401

def test_task_oriented_grasp_uses_opposed_generic_cad_faces():
    from assembly_skills import mating_plane_grasp,clears_support_plane
    rng=np.random.default_rng(5);points=[];normals=[]
    for sign in [-1,1]:
        face=np.c_[rng.uniform(-.008,.008,500),np.full(500,sign*.010),rng.uniform(.700,.720,500)];points.extend(face.tolist());normals.extend(np.tile([0,sign,0],(500,1)).tolist())
    points=np.array(points);points[:,:2]+=[.3,.2]
    candidate={'position_m':[.3,.2,.710],'quaternion_wxyz':[0,1,0,0],'jaw_width_m':.032,'approach_axis_world':[0,0,-1]}
    grasp=mating_plane_grasp(candidate,points,np.array(normals),[.3,.2,.700],[0,0,1],.68)
    assert grasp is not None and grasp['canonical_mating_plane_clearance_m']>=.001
    assert clears_support_plane(grasp,.68)

def test_hub_single_finger_support_is_not_release():
    from hub_evaluator import robot_release_verified
    assert not robot_release_verified([.08],[[0.,2.]])
    assert not robot_release_verified([.01],[[0.,0.]])
    assert robot_release_verified([.08],[[0.,0.]])

def test_observed_pattern_catalog_has_no_target_identity():
    import cv2
    from assembly_skills import geometric_patterns
    rgb=np.full((220,220,3),220,np.uint8);cv2.circle(rgb,(110,110),28,(60,60,60),-1)
    for p in [(110,50),(170,110),(110,170),(50,110)]:cv2.circle(rgb,p,5,(40,40,40),-1)
    patterns=geometric_patterns(rgb,np.ones((220,220)),np.array([[250.,0,110],[0,250.,110],[0,0,1]]),np.eye(4))
    assert patterns
    assert len(patterns[0]['surrounding_candidates'])>=4
    assert 'not task-selected' in patterns[0]['identity']

def test_local_antipodal_candidates_can_pinch_a_wide_ring_wall():
    from assembly_skills import antipodal_grasp_candidates,clears_support_plane
    angles=np.linspace(0,2*np.pi,180,endpoint=False);points=[];normals=[]
    for radius,sign in [(.12,1),(.10,-1)]:
        for z in [.696,.704,.712]:
            points.extend(np.c_[.3+radius*np.cos(angles),.2+radius*np.sin(angles),np.full(len(angles),z)])
            normals.extend(np.c_[sign*np.cos(angles),sign*np.sin(angles),np.zeros(len(angles))])
    points=np.array(points);assert np.ptp(points[:,0])>.08
    candidates=antipodal_grasp_candidates(points,np.array(normals),.68)
    assert candidates and all(clears_support_plane(c,.68) for c in candidates)
    assert all(.004<c['antipodal_pair_distance_m']<.071 for c in candidates)

def test_grasp_requiring_early_release_reports_clearance_instead_of_seat_safety():
    from assembly_skills import mating_plane_grasp
    rng=np.random.default_rng(42);points=[];normals=[]
    for sign in [-1,1]:
        points.extend(np.c_[rng.uniform(.292,.308,500),np.full(500,.2+sign*.01),rng.uniform(.700,.718,500)])
        normals.extend(np.tile([0,sign,0],(500,1)))
    candidate={'position_m':[.3,.2,.710],'quaternion_wxyz':[0,1,0,0],'jaw_width_m':.032,'approach_axis_world':[0,0,-1]}
    grasp=mating_plane_grasp(candidate,np.array(points),np.array(normals),[.3,.2,.720],[0,0,1],.68,require_seat_clearance=False)
    assert grasp is not None and grasp['canonical_mating_plane_clearance_m']<0
    assert grasp['required_release_clearance_m']>.003


def test_closed_cad_volume_uses_scale_and_rejects_open_mesh(tmp_path):
    import struct
    from tools.calibrate_part_mass import closed_stl_volume,cover_mass_profile
    from settings import CAD_METERS_PER_SOURCE_UNIT
    vertices=np.array([[0.,0,0],[1.,0,0],[0,1.,0],[0,0,1.]])
    faces=[[0,2,1],[0,1,3],[0,3,2],[1,2,3]]
    def save(path, selected):
        data=bytearray(80)+struct.pack('<I',len(selected))
        for face in selected:data+=struct.pack('<12fH',*([0.]*3),*vertices[face].reshape(-1),0)
        path.write_bytes(data)
    mesh=tmp_path/'tetra.stl';save(mesh,faces)
    assert np.isclose(closed_stl_volume(mesh,2.)['volume_m3'],8./6.)
    parts=tmp_path/'parts';parts.mkdir();save(parts/'Hub Cover Output.stl',faces)
    plastic=cover_mass_profile(tmp_path,'polypropylene-assumption')
    assert np.isclose(plastic['mass_kg'],(CAD_METERS_PER_SOURCE_UNIT**3/6)*900.)
    assert 'User-confirmed plastic' in plastic['material_status']
    save(mesh,faces[:-1])
    import pytest
    with pytest.raises(ValueError,match='closed'):closed_stl_volume(mesh,2.)


def test_force_regulated_jaw_stops_squeezing_and_releases_normally():
    from gripper_control import jaw_reference_step,mass_based_grip_force
    width=.02
    assert jaw_reference_step(width,0.,1.,1.,.01)==width
    assert jaw_reference_step(width,0.,3.,1.,.01)>width
    assert jaw_reference_step(width,0.,.4,1.,.01)<width
    assert jaw_reference_step(width,.08,3.,None,.01)>width
    assert mass_based_grip_force(.045)==2.
    assert mass_based_grip_force(.045,minimum_force_n=4.)==4.
    assert mass_based_grip_force(100.,minimum_force_n=4.)==35.
    assert 8.<mass_based_grip_force(.658)<11.
    assert mass_based_grip_force(100.)==35.
    assert mass_based_grip_force(.658,[.11,0,0],[1,0,0],.06)>mass_based_grip_force(.658)
    reference=.025
    for _ in range(1200):
        force=max(0.,(.02-reference)*1000.)
        reference=jaw_reference_step(reference,0.,force,.7,1/120)
    assert .65<max(0.,(.02-reference)*1000.)<.75
    assert reference>.019


def test_circular_mouth_uses_rim_surface_instead_of_deep_floor():
    import cv2
    from rim_measurement import observed_rim
    rgb=np.full((201,201,3),220,np.uint8);cv2.circle(rgb,(100,100),30,(45,45,45),-1)
    depth=np.full((201,201),.5);yy,xx=np.indices(depth.shape);depth[(xx-100)**2+(yy-100)**2<30**2]=.7
    K=np.array([[300.,0,100],[0,300.,100],[0,0,1.]])
    T=np.diag([1.,-1.,-1.,1.]);T[:3,3]=[.1,-.2,1.]
    feature=observed_rim(rgb,depth,K,T,[100,100],24)
    assert feature is not None
    np.testing.assert_allclose(feature['position_m'],[.1,-.2,.5],atol=.001)
    assert feature['radius_px_used']>24
    assert feature['circle_rms_m']<.001


def test_public_cad_bore_axis_is_not_assumed_to_be_root_origin():
    from tools.measure_public_bore import measure_bore
    theta=np.linspace(0,2*np.pi,400,endpoint=False)
    points=np.c_[.003+.09*np.cos(theta),np.zeros(len(theta)),-.007+.09*np.sin(theta)]
    normals=np.c_[np.cos(theta),np.zeros(len(theta)),np.sin(theta)]
    fit=measure_bore(points,normals)
    np.testing.assert_allclose(fit['centre_local_m'],[.003,-.007],atol=1e-8)
    assert np.isclose(fit['radius_m'],.09)


def test_tactile_centering_unloads_contacting_side_without_drifting_when_balanced():
    from gripper_control import centering_increment
    assert centering_increment([1.,0.],.7,[0,1,0],.05,.004)[1]>0
    assert centering_increment([0.,1.],.7,[0,1,0],.05,.004)[1]<0
    np.testing.assert_array_equal(centering_increment([.7,.7],.7,[0,1,0],.05,.004),[0,0,0])
    np.testing.assert_array_equal(centering_increment([1.,0.],.7,[0,1,0],.05,0.),[0,0,0])


def test_joint_guard_ignores_observed_solver_roundoff_but_not_real_excursions():
    from robot_kinematics import joint_limit_violation
    limits=np.array([[-2.8973,2.8973]])
    assert not joint_limit_violation([-2.8973107338],limits)
    assert joint_limit_violation([-2.8983],limits)
    assert joint_limit_violation([2.8983],limits)
    assert joint_limit_violation([np.nan],limits)


def test_renderer_pixel_centres_backproject_to_known_calibration_plane():
    from assembly_skills import integer_pixel_intrinsics,cloud
    raster=np.array([[300.,0,100.5],[0,300.,100.5],[0,0,1.]])
    T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',[np.pi-.3,0,.2]).as_matrix();T[:3,3]=[.2,.1,1.]
    yy,xx=np.indices((201,201));uv=np.c_[xx.ravel()+.5,yy.ravel()+.5,np.ones(xx.size)]
    rays=T[:3,:3]@np.linalg.solve(raster,uv.T);depth=(-T[2,3]/rays[2]).reshape(xx.shape)
    points,_=cloud(depth,integer_pixel_intrinsics(raster),T)
    assert np.max(abs(points[:,2]))<1e-10
    wrong,_=cloud(depth,raster,T)
    assert np.sqrt(np.mean(wrong[:,2]**2))>1e-4
    assert raster[0,2]==100.5


def test_inhand_registration_uses_current_partial_surface_without_table_constraint():
    from cad_registration import register
    from scipy.spatial.transform import Rotation
    rng=np.random.default_rng(902)
    normals=rng.normal(size=(1200,3));normals/=np.linalg.norm(normals,axis=1)[:,None]
    radii=np.array([.014,.023,.038]);cad=normals*radii
    actual_normals=normals/radii;actual_normals/=np.linalg.norm(actual_normals,axis=1)[:,None]
    rotation=Rotation.from_rotvec([.13,-.19,.06]).as_matrix();translation=np.array([.35,.02,.95])
    visible=normals[:,2]>.1;observed=cad[visible]@rotation.T+translation
    initial=np.eye(4);initial[:3,:3]=rotation;initial[:3,3]=translation+[.001,-.001,.0005]
    fitted=register(cad,observed,normals=actual_normals,initial_pose=initial)
    estimated=np.array(fitted['world_from_cad'])
    assert np.linalg.norm(estimated[:3,3]-translation)<.0002
    assert fitted['hypotheses_tested']==1
    assert fitted['support_penetration_m']==0


def test_new_move_recovers_reference_at_actual_pause_boundary():
    # Regression: a 2.26 mm tracking error paused CartesianReference but the
    # previous 4 mm reset condition never recovered it on a new command.
    from robot_kinematics import motion_reference_stale,CartesianReference
    measured=np.array([.4,.2,.9,1.,0,0,0]);reference=measured.copy();reference[0]+=.00226
    assert motion_reference_stale(reference,measured)
    recovered=CartesianReference(measured);goal=measured.copy();goal[2]+=.01
    assert recovered.advance(goal,measured,.05)[2]>measured[2]
    assert not motion_reference_stale(recovered.pose,measured)


def test_payload_compensation_is_public_mass_and_lever_wrench():
    from robot_kinematics import payload_support_wrench
    np.testing.assert_allclose(payload_support_wrench(.658282,[.1,0,0]),[0,0,.658282*9.81,0,-.1*.658282*9.81,0])
    np.testing.assert_array_equal(payload_support_wrench(0,[.1,.2,.3]),np.zeros(6))


def test_antipodal_pair_is_not_replaced_by_nearby_unrelated_surfaces():
    from assembly_skills import mating_plane_grasp
    rotation=Rotation.from_quat([1.,0,0,0])
    centre=np.array([.3,.2,.8]);points=[];normals=[]
    for y in [-.03,-.015,.015,.03]:
        for x in np.linspace(-.004,.004,8):
            points.append(centre+rotation.apply([x,y,0.]));normals.append(rotation.apply([0,np.sign(y),0]))
    candidate={'position_m':centre.tolist(),'quaternion_wxyz':[0,1,0,0],'jaw_width_m':.042,'observed_width_m':.03,'contact_pair_world_m':[(centre+rotation.apply([0,.015,0])).tolist(),(centre+rotation.apply([0,-.015,0])).tolist()]}
    result=mating_plane_grasp(candidate,np.array(points),np.array(normals),[.3,.2,.75],[0,0,1],.68,require_seat_clearance=False)
    assert result is not None
    assert abs(result['observed_width_m']-.03)<1e-12


def test_public_candidate_projection_matches_raw_camera_geometry():
    from assembly_skills import project_world_points
    K=np.array([[500.,0,100],[0,400.,80],[0,0,1.]])
    T=np.eye(4);T[:3,:3]=Rotation.from_euler('z',.4).as_matrix();T[:3,3]=[.3,.2,.9]
    optical=np.array([[.04,-.03,2.],[0,0,-1.]])
    world=optical@T[:3,:3].T+T[:3,3]
    result=project_world_points(world,K,T)
    np.testing.assert_allclose(result[0],[110.,74.])
    assert np.isnan(result[1]).all()


def test_closed_mesh_uniform_com_preserves_stl_coordinate_frame(tmp_path):
    import struct
    from tools.calibrate_part_mass import closed_stl_volume
    v=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])+[5.,-3.,2.]
    faces=[[0,2,1],[0,1,3],[0,3,2],[1,2,3]];raw=bytearray(80)+struct.pack('<I',4)
    for face in faces:raw+=struct.pack('<12fH',*([0.]*3),*v[face].reshape(-1),0)
    p=tmp_path/'tetra.stl';p.write_bytes(raw)
    result=closed_stl_volume(p,2.)
    np.testing.assert_allclose(result['uniform_density_com_stl_frame_m'],[10.5,-5.5,4.5])


def test_public_uniform_solid_inertia_matches_translated_tetrahedron():
    from tools.calibrate_part_mass import uniform_solid_properties
    v=2*np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])+[5.,-3.,2.]
    triangles=v[[[0,2,1],[0,1,3],[0,3,2],[1,2,3]]]
    props=uniform_solid_properties(triangles)
    np.testing.assert_allclose(props['uniform_density_com_local_m'],[5.5,-2.5,2.5])
    expected=np.full((3,3),.05);np.fill_diagonal(expected,.3)
    np.testing.assert_allclose(props['inertia_per_mass_local_m2'],expected,atol=1e-12)
    reversed_props=uniform_solid_properties(triangles[:,::-1])
    np.testing.assert_allclose(reversed_props['inertia_per_mass_local_m2'],expected,atol=1e-12)


def test_payload_joint_model_matches_rigid_body_kinetic_energy():
    from robot_kinematics import payload_joint_inertia
    rng=np.random.default_rng(48);jac=rng.normal(size=(6,7));dq=rng.normal(size=7)
    lever=np.array([.11,-.02,.03]);inertia=np.diag([.003,.004,.006]);mass=.658
    model=payload_joint_inertia(jac,mass,lever,inertia)
    twist=jac@dq;linear_com=twist[:3]+np.cross(twist[3:],lever)
    expected=mass*(linear_com@linear_com)+twist[3:]@inertia@twist[3:]
    assert abs(dq@model@dq-expected)<1e-10
    assert np.linalg.eigvalsh(model).min()>-1e-10


def test_opening_discards_preload_without_changing_physical_jaw_state():
    from gripper_control import bumpless_open_reference,jaw_reference_step
    measured=.065;compressed=.035
    neutral=bumpless_open_reference(compressed,measured,.08)
    assert neutral==measured
    next_reference=jaw_reference_step(neutral,.08,30.,None,1/120)
    assert next_reference>measured
    assert bumpless_open_reference(compressed,measured,0.)==compressed


def test_public_fk_jacobian_matches_pose_finite_difference():
    model=RobotKinematics(np.eye(4),[(np.eye(4),transform([.2,0,0],[1,0,0,0]),0,np.array([0.,0,1.])),(np.eye(4),transform([.15,0,0],[1,0,0,0]),1,np.array([0.,1,0.]))],['a','b'],[[-1,1],[-1,1]],[.03,.02,.01])
    q=np.array([.3,-.4]);numeric=np.zeros((6,2));eps=1e-6
    for j in range(2):
        plus=q.copy();minus=q.copy();plus[j]+=eps;minus[j]-=eps
        a=model.fk(plus);b=model.fk(minus);numeric[:3,j]=(a[:3,3]-b[:3,3])/(2*eps);numeric[3:,j]=Rotation.from_matrix(a[:3,:3]@b[:3,:3].T).as_rotvec()/(2*eps)
    np.testing.assert_allclose(model.jacobian(q),numeric,atol=1e-9)


def test_jaw_pressure_excludes_tangential_load_without_identity_filtering():
    from gripper_control import projected_jaw_pressure
    forces=np.array([[10.,4.,20.],[-8.,-3.,-25.]])
    np.testing.assert_allclose(projected_jaw_pressure(forces,[1,0,0]),[10,8])
    rotation=Rotation.from_euler('z',.7).as_matrix()
    np.testing.assert_allclose(projected_jaw_pressure(forces@rotation.T,rotation@[1.,0,0]),[10,8])
    np.testing.assert_allclose(projected_jaw_pressure([[0,0,30],[0,0,30]],[1,0,0]),[0,0])


def test_assembly_grasp_cost_uses_geometry_not_controller_or_task_names():
    from assembly_skills import assembly_grasp_metrics
    perpendicular={'quaternion_wxyz':[0.,1.,0,0],'required_release_clearance_m':.003}
    parallel={'quaternion_wxyz':[1.,0,0,0],'required_release_clearance_m':.003}
    good=assembly_grasp_metrics(perpendicular,[0,0,1],.044)
    bad=assembly_grasp_metrics(parallel,[0,1,0],.044)
    assert good['absolute_dot_jaw_vs_mating_axis']==0
    assert bad['absolute_dot_jaw_vs_mating_axis']==1
    assert good['cost']<bad['cost']
    assert abs(good['guided_entry_before_release_m']-.041)<1e-12


def test_normal_grip_pressure_uses_registered_lever_and_declared_margin():
    from gripper_control import mass_based_grip_force
    mass=.658282;span=.065;lever=.095
    expected=1.3*(mass*9.81/2+mass*9.81*lever/span)/.5
    assert abs(mass_based_grip_force(mass,[-lever,0,0],[1,0,0],span)-expected)<1e-10
    assert 30<expected<35


def test_large_mouth_is_not_replaced_by_smaller_deep_circle():
    import cv2
    from rim_measurement import observed_rim
    rgb=np.full((260,260,3),220,np.uint8);cv2.circle(rgb,(120,120),70,(100,100,100),-1);cv2.circle(rgb,(130,145),18,(0,0,0),-1)
    yy,xx=np.indices((260,260));depth=np.ones((260,260));depth[(xx-120)**2+(yy-120)**2<=70**2]=1.3
    K=np.array([[500.,0,120],[0,500.,120],[0,0,1.]])
    T=np.diag([1.,-1.,-1.,1.]);T[2,3]=1.8
    rim=observed_rim(rgb,depth,K,T,[130,145],24)
    assert rim is not None
    assert abs(rim['position_m'][2]-.8)<1e-8
    assert np.linalg.norm(rim['position_m'][:2])<.002
    assert rim['estimated_radius_m']>.13


def test_grasp_guidance_uses_full_opening_clearance_for_each_axis_sign():
    from assembly_skills import assembly_grasp_metrics
    candidate = {'quaternion_wxyz': [1., 0., 0., 0.]}
    result = assembly_grasp_metrics(candidate, [0., 0., 1.], .022,
                                    {'parallel': .0868, 'antiparallel': .0044})
    assert result['guided_entry_before_release_m'] is None
    assert result['guided_entry_before_release_by_axis_alignment_m']['parallel'] == 0.
    assert abs(result['guided_entry_before_release_by_axis_alignment_m']['antiparallel'] - .0176) < 1e-12
