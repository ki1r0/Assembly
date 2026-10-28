"""Robot/task adapter: real Panda or R1, physical parts, public RGB-D; no attachment shortcuts."""
import os,sys,json,time,math,traceback,dataclasses,types,importlib.util
from pathlib import Path
import settings
from camera_readiness import update_ready_camera
from common import R,write,log
from m1_reach.control import snapshot,terminate,begin_step,complete_step

def main():
    from isaaclab.app import AppLauncher
    import argparse
    parser=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
    from task_profiles import GEAR_TASKS,HUB_TASKS,OTHER_TASKS,SHAFT_TASKS,SUPPORTED_TASKS,skill_specification
    task=os.environ.get('ASSEMBLY_TASK','peg')
    if task not in SUPPORTED_TASKS:parser.error(f'Unsupported ASSEMBLY_TASK {task!r}; refusing to run a different task under this label')
    app=AppLauncher(args).app
    if os.environ.get('ASSEMBLY_CPU_THREADS'):
        import carb
        write(R/'checks/cpu-workers.json',{'requested':int(os.environ['ASSEMBLY_CPU_THREADS']),
            'carbonite':carb.settings.get_settings().get('/plugins/carb.tasking.plugin/threadCount'),
            'tbb':carb.settings.get_settings().get('/plugins/omni.tbb.globalcontrol/maxThreadCount'),
            'scope':'Worker scheduling only; no physics precision or safety-limit change'})
    if os.environ.get('ASSEMBLY_PHYSICS_THREADS'):
        import carb
        from omni.physx.bindings import _physx as physx_bindings
        workers=int(os.environ['ASSEMBLY_PHYSICS_THREADS'])
        carb.settings.get_settings().set_int(physx_bindings.SETTING_NUM_THREADS,workers)
        write(R/'checks/physics-workers.json',{'requested':workers,'effective':carb.settings.get_settings().get(physx_bindings.SETTING_NUM_THREADS),'scope':'PhysX scheduling only; no solver iteration, timestep or safety-limit change'})
    profiler=None
    if os.environ.get('ASSEMBLY_PROFILE_BACKEND')=='1':
        import cProfile
        profiler=cProfile.Profile();profiler.enable()
    import numpy as np,torch,cv2
    from scipy.spatial.transform import Rotation
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation,RigidObject,RigidObjectCfg
    from isaaclab.sensors import Camera,CameraCfg,ContactSensor,ContactSensorCfg
    from isaaclab.controllers import DifferentialIKController,DifferentialIKControllerCfg
    from isaaclab.utils.math import subtract_frame_transforms
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG,FRANKA_PANDA_HIGH_PD_CFG
    from pxr import UsdPhysics,PhysxSchema,Usd,UsdGeom,Gf,Sdf
    from runtime.assembly_harness.depth_io import save_depth
    from runtime.bolt_harness.recording import EpisodeRGBVideoRecorder
    controller_mode=os.environ.get('ASSEMBLY_CONTROLLER','factory')
    cover_mass=float(os.environ.get('ASSEMBLY_COVER_MASS_KG','5.7'))
    robot_name=os.environ.get('ASSEMBLY_ROBOT','r1');flat=os.environ.get('ASSEMBLY_CASE')=='bolt-horizontal-pose';device='cpu'
    assets=Path(os.environ['ASSEMBLY_ASSET_ROOT']);skill_data=skill_specification()
    condition_name=os.environ.get('ASSEMBLY_INPUT_CONDITION','clean')
    suite=json.loads((settings.ROOT/'config/xarm7_robustness_suite.json').read_text())
    condition=next(row for row in suite['conditions'] if row['id']==condition_name)
    from input_perturbation import InputPerturbation
    input_noise=InputPerturbation(condition)
    setup_rng=np.random.default_rng(condition['seed'])
    write(R/'checks/input-condition.json',{'condition':condition,'noise_boundary':suite['noise_boundary']})
    is_hub_task=task in HUB_TASKS;is_fixture_task=task in GEAR_TASKS|SHAFT_TASKS|OTHER_TASKS;is_relation_task=is_hub_task or is_fixture_task
    evaluator_profiles=json.loads((settings.ROOT/'config/task_evaluator_profiles.json').read_text())
    evaluator_profile=evaluator_profiles[task] if is_relation_task else None
    compound_initialization=None
    if task in SHAFT_TASKS:
        # Launcher generated these robot-neutral CAD assets in the mounted run
        # directory. Never silently fall back to the old bare shaft.
        compound_initialization=json.loads((R/'checks/initial-shaft-compound.json').read_text())
        if compound_initialization['provenance']['task']!=task:
            raise ValueError('Initial compound does not match the requested shaft task')
        if (compound_initialization['asset_path']!=skill_data['moving_asset_path'] or
                compound_initialization['cache_path']!=skill_data['moving_cad_cache']):
            raise ValueError('Initial compound USD/cache do not match the effective public skill')
        cover_mass=compound_initialization['mass_profile']['mass_kg']
        os.environ['ASSEMBLY_COVER_MASS_KG']=str(cover_mass)
        os.environ['ASSEMBLY_COVER_MASS_STATUS']=compound_initialization['mass_profile']['mass_estimate_status']
        write(R/'checks/moving-part-mass-profile.json',compound_initialization['mass_profile'])
    physics_rate=int(os.environ.get('ASSEMBLY_XARM_PHYSICS_HZ','480')) if robot_name=='xarm7' else 120
    if robot_name=='xarm7' and physics_rate not in {120,240,480}:raise ValueError('Unsupported xArm physics frequency')
    control_decimation=physics_rate//20
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/physics_rate,device=device,render_interval=control_decimation,enable_scene_query_support=os.environ.get('ASSEMBLY_PEG_MVP')=='1'))
    collision_backend=os.environ.get('ASSEMBLY_COLLISION_BACKEND','gpu-sdf')
    if collision_backend=='gpu-sdf':
        # GPU contacts with CPU tensor readback: keep SciPy/control on CPU,
        # while enabling the PhysX collision pipeline required by dynamic SDFs.
        import carb
        carb.settings.get_settings().set_int('/physics/cudaDevice',0)
        sim.get_physics_context().set_broadphase_type('GPU')
        sim.get_physics_context().enable_gpu_dynamics(True)
    dt=1/physics_rate;control_dt=control_decimation*dt;step_index=0;frame_index=0;native_success=False;active_arm='left';widths={'left':.08,'right':.08};targets={};history=[];trace=[]
    sim_utils.GroundPlaneCfg().func('/World/Ground',sim_utils.GroundPlaneCfg())
    # Match the earlier bolt-harness scene rather than the brighter generic dome.
    light=sim_utils.DomeLightCfg(intensity=1800.*condition.get('light_intensity_multiplier',1.),color=(.75,.75,.75));light.func('/World/Light',light)
    table_z=.934 if robot_name=='r1' else .68;offset_z=table_z-.68
    table=sim_utils.CuboidCfg(size=(1.4,1.4,.10),collision_props=sim_utils.CollisionPropertiesCfg(),visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.32,.34,.36)))
    table.func('/World/Table',table,translation=(.75 if robot_name=='r1' else .55,0,table_z-.05))
    if robot_name=='panda':
        cfg=FRANKA_PANDA_CFG.copy();cfg.prim_path='/World/Robot';cfg.init_state.pos=(0,0,table_z);cfg.spawn.usd_path=str(assets/'vendor/remote_mirror/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd')
        # Same contact-solver precision used by installed Isaac Lab Factory.
        cfg.spawn.articulation_props.solver_position_iteration_count=max(192,cfg.spawn.articulation_props.solver_position_iteration_count or 0)
        cfg.spawn.articulation_props.solver_velocity_iteration_count=max(1,cfg.spawn.articulation_props.solver_velocity_iteration_count or 0)
        for actuator in cfg.actuators.values():
            if any('finger' in pattern for pattern in actuator.joint_names_expr):actuator.effort_limit_sim=70.;actuator.effort_limit=70.
        if controller_mode=='reference_ik_high_pd':
            for actuator_name in ['panda_shoulder','panda_forearm']:
                cfg.actuators[actuator_name].stiffness=FRANKA_PANDA_HIGH_PD_CFG.actuators[actuator_name].stiffness
                cfg.actuators[actuator_name].damping=FRANKA_PANDA_HIGH_PD_CFG.actuators[actuator_name].damping
        if controller_mode.startswith('factory'):
            from isaaclab_tasks.direct.factory.factory_env_cfg import CtrlCfg
            cfg.init_state.joint_pos={**{f'panda_joint{i+1}':float(q) for i,q in enumerate(CtrlCfg().reset_joints)},'panda_finger_joint.*':.04}
            cfg.spawn.articulation_props.solver_position_iteration_count=max(192,cfg.spawn.articulation_props.solver_position_iteration_count or 0)
            cfg.spawn.articulation_props.solver_velocity_iteration_count=max(1,cfg.spawn.articulation_props.solver_velocity_iteration_count or 0)
            for actuator_name in ['panda_shoulder','panda_forearm']:
                cfg.actuators[actuator_name].stiffness=0.
                if controller_mode=='factory':cfg.actuators[actuator_name].damping=0.  # preserved V4/V5 diagnostic
                # factory_damped retains the public Panda actuator's implicit damping (4 Nm s/rad).
        if controller_mode=='isaac_osc':
            for actuator_name in ['panda_shoulder','panda_forearm']:
                cfg.actuators[actuator_name].stiffness=0.;cfg.actuators[actuator_name].damping=0.
        arms={'left':('panda_joint[1-7]','panda_hand','panda_finger_joint.*',[0,0,.1034],['panda_leftfinger','panda_rightfinger'])}
    elif robot_name=='xarm7':
        from xarm_robot_cfg import xarm7_config,restore_g1_mimic_joints,apply_official_adjacent_collision_filters,apply_urdf_effort_friction
        from xarm_gripper import G1Geometry
        cfg=xarm7_config(settings.ROOT/'assets/xarm7/usd/xarm7_g1.usd',base_position=(0,0,table_z))
        cfg.actuators['arm'].stiffness=float(os.environ.get('ASSEMBLY_XARM_DIAGNOSTIC_ARM_STIFFNESS','400'))
        cfg.actuators['arm'].damping=float(os.environ.get('ASSEMBLY_XARM_DIAGNOSTIC_ARM_DAMPING','40'))
        if not np.isfinite(cfg.actuators['arm'].stiffness) or not 400<=cfg.actuators['arm'].stiffness<=1000 or not np.isfinite(cfg.actuators['arm'].damping) or not 40<=cfg.actuators['arm'].damping<=100:raise ValueError('Invalid diagnostic arm drive gains')
        if os.environ.get('ASSEMBLY_ADAPTER_START_JOINTS'):
            from xarm_robot_cfg import diagnostic_initial_joints
            values=json.loads(os.environ['ASSEMBLY_ADAPTER_START_JOINTS'])
            cfg.init_state.joint_pos={name:value for name,value in cfg.init_state.joint_pos.items() if not name.startswith('joint')}
            cfg.init_state.joint_pos.update(diagnostic_initial_joints(values,settings.ROOT/'assets/xarm7/xarm7_g1_simulation_assumption.urdf'))
            write(R/'checks/diagnostic-initial-arm.json',{'joints_rad':values,'scope':'Robot-only diagnostic initial state; parts unchanged, not an autonomous rollout'})
        xarm_gripper=G1Geometry(settings.ROOT/'assets/xarm7/xarm7_g1_simulation_assumption.urdf')
        from arm_support_geometry import ArmSupportGeometry
        arm_support=ArmSupportGeometry(settings.ROOT/'assets/xarm7/xarm7_g1_simulation_assumption.urdf',[f'joint{i}' for i in range(1,8)])
        from robot_kinematics import transform
        arm_mount=transform(cfg.init_state.pos,cfg.init_state.rot)
        arms={'left':('joint[1-7]','xarm_gripper_base_link','drive_joint',[0,0,.172],['left_finger','right_finger'])}
    elif robot_name=='r1':
        ext=Path(os.environ['ASSEMBLY_R1_ROOT']);package=types.ModuleType('Galaxea_Lab_External');package.GALAXEA_LAB_ASSETS_DIR=str(ext/'assets');sys.modules['Galaxea_Lab_External']=package
        spec=importlib.util.spec_from_file_location('_r1_robot_cfg',ext/'Galaxea_Lab_External/robots/galaxea_robots.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        cfg=module.GALAXEA_R1_CHALLENGE_CFG.copy();cfg.prim_path='/World/Robot';cfg.spawn.collision_props.contact_offset=.0001;cfg.spawn.collision_props.rest_offset=0.
        arms={side:(side+'_arm_joint.*',side+'_arm_link6',side+'_gripper_axis1',[.0079,0,.09089],[side+'_gripper_link1',side+'_gripper_link2']) for side in ['left','right']}
    else:raise ValueError('Unknown robot embodiment: '+robot_name)
    cfg.spawn.activate_contact_sensors=True;robot=Articulation(cfg)
    if robot_name=='xarm7':
        mimic=restore_g1_mimic_joints(sim.stage)
        filters=apply_official_adjacent_collision_filters(sim.stage,settings.ROOT/'third_party/xarm_ros2/xarm_moveit_config/srdf/_xarm7_macro.srdf.xacro')
        friction=apply_urdf_effort_friction(sim.stage,settings.ROOT/'assets/xarm7/xarm7_g1_simulation_assumption.urdf')
        write(R/'checks/xarm7-linkage.json',{'mimic_joints':mimic,'official_collision_pairs':filters,'urdf_effort_friction':friction,'scope':'simulation assumption, no real robot connection'})
    calibration=json.loads((settings.ROOT/'config/fixture_calibration.json').read_text())
    casing_pos=[.5,-.15,.735896652+offset_z];cover_pos=[.5000051945,-.0623984266,.7977342518+offset_z];casing_asset='parts/Casing Top.usd';fixture_quat=[1,0,0,0]
    if is_hub_task:cover_pos=[.35,.30,table_z+.006]
    if is_fixture_task:
        fixture_rotation=Rotation.from_euler('xyz',evaluator_profile['fixture_rotation_xyz_deg'],degrees=True)
        fixture_quat=fixture_rotation.as_quat()[[3,0,1,2]].tolist()
        fixture_xy=evaluator_profile.get('fixture_world_xy_m',[.55,-.15])
        casing_asset=evaluator_profile['fixture_asset']
        fixture_stage=Usd.Stage.Open(str(assets/evaluator_profile['fixture_asset']))
        fixture_bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default']).ComputeWorldBound(fixture_stage.GetPseudoRoot()).ComputeAlignedRange()
        fixture_low=np.array(fixture_bounds.GetMin());fixture_high=np.array(fixture_bounds.GetMax())
        fixture_corners=np.array([[x,y,z] for x in [fixture_low[0],fixture_high[0]] for y in [fixture_low[1],fixture_high[1]] for z in [fixture_low[2],fixture_high[2]]])*settings.CAD_METERS_PER_SOURCE_UNIT
        fixture_bottom=float(fixture_rotation.apply(fixture_corners)[:,2].min())
        casing_pos=[float(fixture_xy[0]),float(fixture_xy[1]),table_z-fixture_bottom+.002+evaluator_profile.get('fixture_table_clearance_m',0.)]
        reset_hint=evaluator_profile['candidate_reset_hint'];reset_rotation=Rotation.from_euler('xyz',reset_hint['moving_rotation_xyz_deg'],degrees=True)
        local_phase=Rotation.from_euler('y',reset_hint.get('moving_local_phase_y_deg',0.),degrees=True)
        cover_quat=(fixture_rotation*reset_rotation*local_phase).as_quat()[[3,0,1,2]].tolist()
        cover_pos=[.35,.30,table_z+.006]
    else:
        # Fit the static casing to the tabletop using its current scaled CAD bounds.
        fixture_stage=Usd.Stage.Open(str(assets/casing_asset))
        fixture_bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default']).ComputeWorldBound(fixture_stage.GetPseudoRoot()).ComputeAlignedRange()
        fixture_low=np.array(fixture_bounds.GetMin());fixture_high=np.array(fixture_bounds.GetMax())
        fixture_corners=np.array([[x,y,z] for x in [fixture_low[0],fixture_high[0]] for y in [fixture_low[1],fixture_high[1]] for z in [fixture_low[2],fixture_high[2]]])*settings.CAD_METERS_PER_SOURCE_UNIT
        casing_bottom=float(Rotation.from_quat(np.array(fixture_quat)[[1,2,3,0]]).apply(fixture_corners)[:,2].min())
        casing_pos[2]=table_z-casing_bottom
        if task=='peg':
            cover_pos=(np.asarray(casing_pos)+np.asarray(calibration['output_cover_root_offset_from_assembly_frame_source_units'])*settings.CAD_METERS_PER_SOURCE_UNIT).tolist()
    diagnostic=os.environ.get('ASSEMBLY_DIAGNOSTIC')
    peg_probe=diagnostic in {'peg-seat-probe','peg-offset-probe'} and task=='peg'
    m10_probe=diagnostic=='m10-seat-probe' and task=='m10-casing-bolt'
    bearing_probe=diagnostic in {'bearing-middle-seat-probe','bearing-middle-offset-probe','bearing-middle-flipped-probe'} and task=='bearing-input-top-onto-shaft'
    if diagnostic and not is_hub_task and not peg_probe and not m10_probe and not bearing_probe:raise ValueError('Unsupported task for seat diagnostic')
    if is_hub_task:cover_quat=Rotation.from_euler('xyz',evaluator_profile['expected_rotation_xyz_deg'],degrees=True).as_quat()[[3,0,1,2]].tolist()
    elif not is_fixture_task:cover_quat=[0,0,.70710678,.70710678]
    if bearing_probe:
        # User-selected B/source fit; initial-pose probe, not proof of insertion access.
        from fixture_geometry import bearing_middle_probe_pose
        cover_pos,cover_quat=bearing_middle_probe_pose(casing_pos,fixture_rotation,diagnostic)
    elif m10_probe:
        from fixture_geometry import m10_seat_probe_geometry
        probe_geometry=m10_seat_probe_geometry(settings.ROOT/'assets/public_cad_models/casing_top.npz',settings.ROOT/'assets/public_cad_models/m10_bolt.npz')
        write(R/'private/m10-seat-probe-geometry.json',probe_geometry)
        cover_pos=(np.array(casing_pos)+fixture_rotation.apply(probe_geometry['root_offset_m'])+[0.,0.,.002]).tolist()
        cover_quat=(fixture_rotation*Rotation.from_euler('x',90,degrees=True)).as_quat()[[3,0,1,2]].tolist()
    elif diagnostic and not peg_probe:
        cover_pos=(np.array(casing_pos)+np.array(evaluator_profile['expected_root_offset_m'])).tolist()
        if diagnostic in {'hub-positive-cad-datum','hub-yaw-positive','hub-yaw-origin-negative'} and task=='hub-cover':
            # Same CAD wall alignment used by the corrected static M6 fixture.
            cover_pos=(np.array(casing_pos)+np.array(calibration['output_cover_root_offset_from_assembly_frame_source_units'])*settings.CAD_METERS_PER_SOURCE_UNIT).tolist()
            cover_pos[2]+=.002
        if diagnostic in {'hub-yaw-positive','hub-yaw-origin-negative'}:
            original=Rotation.from_quat(np.array(cover_quat)[[1,2,3,0]])
            rotated=original*Rotation.from_rotvec(np.array(skill_data['moving_feature_axis_local'])*np.pi)
            if diagnostic=='hub-yaw-positive':
                datum=np.array(skill_data['moving_feature_local_m'])
                cover_pos=(np.array(cover_pos)+original.apply(datum)-rotated.apply(datum)).tolist()
            cover_quat=rotated.as_quat()[[3,0,1,2]].tolist()
        if diagnostic=='hub-positive-near-seat':cover_pos[2]+=.002
        if diagnostic in ['hub-positive-from-clearance','hub-offset-negative','hub-tilt-negative'] or (diagnostic=='hub-positive-cad-datum' and task!='hub-cover'):cover_pos[2]+=.035
        if diagnostic=='hub-offset-negative':cover_pos[0]+=.025
        if diagnostic=='hub-tilt-negative':cover_quat=(Rotation.from_rotvec([.20,0,0])*Rotation.from_quat(np.array(cover_quat)[[1,2,3,0]])).as_quat()[[3,0,1,2]].tolist()
    def part(name,file,pos,quat,dynamic,mass):
        file_path=Path(file)
        asset_path=file_path if file_path.is_absolute() else assets/(file_path if len(file_path.parts)>1 else Path('parts')/file_path)
        if dynamic:
            source_stage=Usd.Stage.Open(str(asset_path))
            source_has_rigid_body=any(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in source_stage.Traverse())
            if not source_has_rigid_body:
                wrapper_path=R/'private'/'asset_overrides'/f'{name.lower().replace(" ","_")}.usd'
                wrapper_path.parent.mkdir(parents=True,exist_ok=True)
                wrapper=Usd.Stage.CreateNew(str(wrapper_path))
                root=UsdGeom.Xform.Define(wrapper,'/Asset').GetPrim()
                root.GetReferences().AddReference(str(asset_path),Sdf.Path('/World'))
                UsdPhysics.RigidBodyAPI.Apply(root)
                wrapper.SetDefaultPrim(root)
                for prim in wrapper.Traverse():
                    if prim.IsA(UsdGeom.Mesh):
                        UsdPhysics.CollisionAPI.Apply(prim)
                        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('sdf')
                        PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim).CreateSdfResolutionAttr(int(os.environ.get('ASSEMBLY_SDF_RESOLUTION','256')))
                wrapper.GetRootLayer().Save()
                asset_path=wrapper_path
        if name=='Cover' and dynamic and not diagnostic:
            asset_stage=Usd.Stage.Open(str(asset_path));bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default']).ComputeWorldBound(asset_stage.GetPseudoRoot()).ComputeAlignedRange()
            low=np.array(bounds.GetMin());high=np.array(bounds.GetMax());corners=np.array([[x,y,z] for x in [low[0],high[0]] for y in [low[1],high[1]] for z in [low[2],high[2]]])*settings.CAD_METERS_PER_SOURCE_UNIT
            bottom=float(Rotation.from_quat(np.array(quat)[[1,2,3,0]]).apply(corners)[:,2].min());pos[2]=table_z-bottom+.002
        if dynamic and not diagnostic and condition.get('part_initial_xy_jitter_m'):
            pos=list(pos);pos[:2]=(np.asarray(pos[:2])+setup_rng.uniform(-condition['part_initial_xy_jitter_m'],condition['part_initial_xy_jitter_m'],2)).tolist()
        # TGS favors position iterations; use the Factory-style single velocity iteration.
        # Keep dynamic part contacts measurable through the release/stability window.
        spawn=sim_utils.UsdFileCfg(usd_path=str(asset_path),scale=(settings.CAD_METERS_PER_SOURCE_UNIT,)*3,activate_contact_sensors=True,rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=not dynamic,disable_gravity=not dynamic,solver_position_iteration_count=192,solver_velocity_iteration_count=1,sleep_threshold=0.),mass_props=sim_utils.MassPropertiesCfg(mass=mass),collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.00005,rest_offset=0))
        if dynamic and task=='peg' and diagnostic and os.environ.get('ASSEMBLY_PEG_CONTACT_DIAGNOSTIC')=='1':
            spawn.rigid_props.stabilization_threshold=0.
        obj=RigidObject(RigidObjectCfg(prim_path='/World/'+name,spawn=spawn,init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(pos),rot=tuple(quat))))
        object_meshes=[prim for prim in sim.stage.Traverse() if str(prim.GetPath()).startswith('/World/'+name) and prim.GetTypeName()=='Mesh']
        authored_collision_meshes=[prim for prim in object_meshes if prim.HasAPI(UsdPhysics.CollisionAPI)]
        for prim in authored_collision_meshes or object_meshes:
            if prim.GetTypeName()=='Mesh':
                UsdPhysics.CollisionAPI.Apply(prim);mesh=UsdPhysics.MeshCollisionAPI.Apply(prim);mesh.CreateApproximationAttr('sdf' if dynamic else 'none')
                if dynamic:PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim).CreateSdfResolutionAttr(int(os.environ.get('ASSEMBLY_SDF_RESOLUTION','256')))
        if diagnostic:
            source=Usd.Stage.Open(str(asset_path))
            source_bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default']).ComputeWorldBound(source.GetPseudoRoot()).ComputeAlignedRange()
            scene_bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default']).ComputeWorldBound(sim.stage.GetPrimAtPath('/World/'+name)).ComputeAlignedRange()
            write(R/f'private/asset-placement-{name}.json',{'asset':str(asset_path),'dynamic':dynamic,'requested_position_m':pos,'requested_quaternion_wxyz':quat,'source_bounds_scaled_m':[(np.asarray(source_bounds.GetMin())*settings.CAD_METERS_PER_SOURCE_UNIT).tolist(),(np.asarray(source_bounds.GetMax())*settings.CAD_METERS_PER_SOURCE_UNIT).tolist()],'scene_bounds_world_m':[list(scene_bounds.GetMin()),list(scene_bounds.GetMax())]})
        return obj
    casing=part('Casing',casing_asset,casing_pos,fixture_quat,False,10)
    cover=part('Cover',skill_data.get('fixture_cover_asset',skill_data.get('moving_asset_path',skill_data.get('moving_cad_file','Hub Cover Output.usd'))),cover_pos,cover_quat,is_relation_task,cover_mass)
    bolt_pos=[.30,.25,table_z+(.00505 if flat else .013075)]
    bolt_quat=[1,0,0,0] if flat else [.70710678,.70710678,0,0]
    if os.environ.get('ASSEMBLY_PEG_MVP')=='1' and not peg_probe:
        offset=np.asarray(json.loads(os.environ.get('ASSEMBLY_PEG_OFFSET_M','[0,0]')),float)
        if offset.shape!=(2,) or not np.isfinite(offset).all() or np.max(np.abs(offset))>.01:
            raise ValueError('Invalid MVP initial XY offset')
        bolt_pos[:2]=(np.asarray(bolt_pos[:2])+offset).tolist()
    if peg_probe:
        from fixture_geometry import bolt_fixture_geometry
        from runtime.bolt_harness.measurement import CAP_UNDERSIDE_LOCAL_Y_M
        probe_geometry=bolt_fixture_geometry(calibration,casing_pos,settings.CAD_METERS_PER_SOURCE_UNIT)
        bolt_pos=[*probe_geometry['selected_hole_xy_m'],probe_geometry['cap_support_plane_z_m']-CAP_UNDERSIDE_LOCAL_Y_M+.002]
        if diagnostic=='peg-offset-probe':bolt_pos[0]+=.010
        if os.environ.get('ASSEMBLY_PEG_MVP')=='1':
            seed=int(os.environ.get('ASSEMBLY_PEG_PROBE_SEED','0'))
            rng=np.random.default_rng(seed)
            bolt_pos[:2]=(np.asarray(bolt_pos[:2])+rng.uniform(-.00005,.00005,2)).tolist()
            clearance=float(rng.uniform(.001,.003))
            inverted=os.environ.get('ASSEMBLY_PEG_PROBE_INVERTED')=='1'
            bolt_pos[2]=probe_geometry['cap_support_plane_z_m']+(.01307499885559082 if inverted else -CAP_UNDERSIDE_LOCAL_Y_M)+clearance
            q=Rotation.from_quat(np.asarray(bolt_quat)[[1,2,3,0]])*Rotation.from_euler('y',float(rng.uniform(0,360)),degrees=True)
            if inverted:q=q*Rotation.from_euler('x',180,degrees=True)
            bolt_quat=q.as_quat()[[3,0,1,2]].tolist()
            write(R/'checks/peg-probe-initialization.json',{'seed':seed,'inverted':inverted,
                'initial_position_m':bolt_pos,'initial_quaternion_wxyz':bolt_quat,
                'drop_clearance_m':clearance,'scope':'Privileged initial condition only; no runtime pose editing'})
    bolt=part('Bolt','M6 Hub Bolt.usd',bolt_pos,bolt_quat,True,.045) if task=='peg' else None
    if task=='peg':
        from runtime.bolt_harness.collision import author_split_bolt_convex_proxies
        split_info=author_split_bolt_convex_proxies(sim.stage,bolt_root_path='/World/Bolt',source_mesh_path='/World/Bolt/node_/mesh_')
        write(R/'checks/bolt-collision-model.json',split_info|{'source':'Existing bolt_harness CAD-clipped shaft/cap convex decomposition; not a scene target or attachment'})
    elif task=='m10-casing-bolt':
        from fixture_geometry import m10_seat_probe_geometry
        from runtime.bolt_harness.collision import author_split_bolt_convex_proxies
        m10_geometry=m10_seat_probe_geometry(settings.ROOT/'assets/public_cad_models/casing_top.npz',settings.ROOT/'assets/public_cad_models/m10_bolt.npz')
        split_info=author_split_bolt_convex_proxies(sim.stage,bolt_root_path='/World/Cover',source_mesh_path='/World/Cover/node_/mesh_',split_y_source_units=m10_geometry['cap_underside_local_y_m']/settings.CAD_METERS_PER_SOURCE_UNIT)
        write(R/'checks/bolt-collision-model.json',split_info|{'source':'M10 CAD-clipped shaft/cap hulls using the measured head underside, not the M6 split plane; no attachment or pose correction'})
    import carb
    from camera_readiness import configure_full_frame_window
    configure_full_frame_window(carb.settings.get_settings())
    camera=Camera(CameraCfg(prim_path='/World/Camera',update_period=0,height=720,width=960,data_types=['rgb','distance_to_image_plane'],update_latest_camera_pose=True,spawn=sim_utils.PinholeCameraCfg(focal_length=16,horizontal_aperture=20.955,clipping_range=(.02,5))))
    finger_sensors={}
    peg_mvp_enabled=os.environ.get('ASSEMBLY_PEG_MVP')=='1'
    bolt_collider_paths=[str(p.GetPath()) for p in sim.stage.Traverse() if str(p.GetPath()).startswith('/World/Bolt/') and p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()] if peg_mvp_enabled else []
    for arm,(_,_,_,_,fingers) in arms.items():
        finger_sensors[arm]=[ContactSensor(ContactSensorCfg(prim_path='/World/Robot/'+finger,update_period=0,filter_prim_paths_expr=bolt_collider_paths)) for finger in fingers]
    obj=bolt if task=='peg' else cover
    fixture_contact_root='/World/Cover' if task=='peg' else '/World/Casing'
    fixture_contact_paths=[str(prim.GetPath()) for prim in sim.stage.Traverse() if str(prim.GetPath()).startswith(fixture_contact_root+'/') and prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()]
    if not fixture_contact_paths:raise RuntimeError('Fixture has no enabled collision shapes for private contact measurement')
    private_contact=ContactSensor(ContactSensorCfg(prim_path='/World/'+('Bolt' if task=='peg' else 'Cover'),filter_prim_paths_expr=fixture_contact_paths,track_contact_points=True,max_contact_data_count_per_prim=1024))
    def fixture_reports():
        from runtime.bolt_harness.measurement import read_all_filtered_contact_reports
        return read_all_filtered_contact_reports(private_contact._contact_physx_view,dt_s=dt,sensor_index=0)
    robot_contact_paths=[str(p.GetPath()) for p in sim.stage.Traverse() if str(p.GetPath()).startswith('/World/Robot/') and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    diagnostic_filter_paths=[str(p.GetPath()) for p in Usd.PrimRange.Stage(sim.stage,Usd.TraverseInstanceProxies()) if str(p.GetPath()).startswith(('/World/Casing/','/World/Cover/','/World/Bolt/','/World/Table/','/World/Robot/')) and p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
    diagnostic_target_indices=[i for i,path in enumerate(diagnostic_filter_paths) if path.startswith('/World/Bolt/')]
    # Isaac Lab filtered reports support one sensor body to many colliders, not many-to-many.
    diagnostic_contacts={path:ContactSensor(ContactSensorCfg(prim_path=path,filter_prim_paths_expr=diagnostic_filter_paths,track_contact_points=True,max_contact_data_count_per_prim=4096,update_period=0)) for path in robot_contact_paths}
    if diagnostic:
        collision_records=[]
        for prim in sim.stage.Traverse():
            if str(prim.GetPath()).startswith(('/World/Casing','/World/Cover','/World/Bolt')) and prim.IsA(UsdGeom.Mesh):
                collision=UsdPhysics.CollisionAPI(prim)
                collision_records.append({'path':str(prim.GetPath()),'collision_api':bool(collision),'collision_enabled':collision.GetCollisionEnabledAttr().Get() if collision else None,'approximation':UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(),'local_to_world_matrix':np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).tolist(),'visibility':UsdGeom.Imageable(prim).ComputeVisibility()})
        write(R/'private/actual-collision-meshes.json',collision_records)
    sim.reset()
    write(R/'private/robot-contact-source.json',{'sensor_bodies':robot_contact_paths,'filter_paths':diagnostic_filter_paths,'target_indices':diagnostic_target_indices,'resolved_views':{path:{'body_count':sensor.body_physx_view.count,'filter_count':sensor._contact_physx_view.filter_count,'filter_paths':sensor._contact_physx_view.filter_paths} for path,sensor in diagnostic_contacts.items()},'scope':'Private one-body-per-sensor contacts including instance-proxy colliders; no policy input'})
    if diagnostic:
        write(R/'private/fixture-contact-source.json',{'evaluator_only':True,'requested_collider_paths':fixture_contact_paths,'resolved_filter_paths':private_contact._contact_physx_view.filter_paths,'filter_count':int(private_contact._contact_physx_view.filter_count)})
        write(R/'private/moving-physical-properties.json',{'evaluator_only':True,'mass_kg':obj.root_physx_view.get_masses().cpu().tolist(),'inertia_kg_m2':obj.root_physx_view.get_inertias().cpu().tolist(),'com_pose_local':obj.root_physx_view.get_coms().cpu().tolist()})
        body_properties=[]
        for prim in sim.stage.Traverse():
            if str(prim.GetPath()).startswith('/World/'+('Bolt' if task=='peg' else 'Cover')) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                body_properties.append({'path':str(prim.GetPath()),'attributes':{
                    a.GetName():str(a.Get()) for a in prim.GetAttributes()
                    if a.GetName().startswith(('physics:','physxRigidBody:'))}})
        write(R/'private/moving-body-state-source.json',{'bodies':body_properties,
            'disable_gravities':obj.root_physx_view.get_disable_gravities().cpu().tolist(),
            'scope':'Private resolved physics attributes, not actor input'})
    public_camera=json.loads((settings.ROOT/'config/public_camera.json').read_text());camera_pose=np.array(public_camera['world_from_camera']);camera_pose[2,3]+=offset_z;camera_q=Rotation.from_matrix(camera_pose[:3,:3]).as_quat()[[3,0,1,2]]
    camera.set_world_poses(torch.tensor(camera_pose[None,:3,3],dtype=torch.float32),torch.tensor(camera_q[None],dtype=torch.float32),convention='ros')
    info={}
    for arm,(joint_pattern,body_pattern,grip_pattern,offset,_) in arms.items():
        ids,_=robot.find_joints(joint_pattern);bodies,_=robot.find_bodies(body_pattern);grips,_=robot.find_joints(grip_pattern)
        info[arm]={'joints':ids,'body':bodies[0],'grips':grips,'offset':np.array(offset),'ik':DifferentialIKController(DifferentialIKControllerCfg(command_type='pose',use_relative_mode=False,ik_method='dls'),1,device)}
    # SimulationContext does not run DirectRLEnv's joint reset. Reset once before the episode.
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),robot.data.default_joint_vel.clone())
    robot.set_joint_position_target(robot.data.default_joint_pos.clone());robot.write_data_to_sim()
    payload_estimates={}
    from robot_kinematics import com_to_tcp_jacobian
    def sdk_tcp_jacobian(arm,tcp_pose):
        data=info[arm];body=data['body']
        raw=robot.root_physx_view.get_jacobians()[0,body-1 if robot.is_fixed_base else body][:,data['joints']].cpu().numpy()
        com=robot.data.body_com_pos_b[0,body].cpu().numpy()
        rotation=Rotation.from_quat(np.asarray(tcp_pose)[[4,5,6,3]]).as_matrix()
        return com_to_tcp_jacobian(raw,rotation,data['offset'],com)
    physics_context=sim.get_physics_context()
    physics_backend_info={'requested_collision_backend':collision_backend,'gpu_dynamics_enabled':physics_context.is_gpu_dynamics_enabled(),'broadphase':physics_context.get_broadphase_type(),'control_tensor_device':device,'sdf_resolution':int(os.environ.get('ASSEMBLY_SDF_RESOLUTION','256')),'part_solver_position_iterations':192,'part_solver_velocity_iterations':1}
    physics_backend_info['part_sleep_threshold']=0.
    physics_backend_info.update(physics_dt_s=dt,control_dt_s=control_dt,video_period_s=.2)
    if robot_name=='xarm7':physics_backend_info['robot_solver_velocity_iterations']=cfg.spawn.articulation_props.solver_velocity_iteration_count
    write(R/'checks/physics-backend.json',physics_backend_info)
    if collision_backend=='gpu-sdf' and (not physics_backend_info['gpu_dynamics_enabled'] or physics_backend_info['broadphase']!='GPU'):raise RuntimeError('Requested SDF GPU collision pipeline was not enabled')
    def compensate_gravity():
        if not cfg.spawn.rigid_props.disable_gravity:
            gravity = robot.root_physx_view.get_gravity_compensation_forces().clone()
            from robot_kinematics import payload_support_wrench
            for arm,payload in payload_estimates.items():
                data=info[arm];state=robot.data.body_state_w[0,data['body']]
                rotation=Rotation.from_quat(state[[4,5,6,3]].cpu().numpy())
                tcp_jac=torch.tensor(sdk_tcp_jacobian(arm,state.cpu().numpy()),dtype=gravity.dtype,device=gravity.device)
                wrench=payload_support_wrench(payload['mass_kg'],rotation.apply(payload['local_com_proxy_m']))
                gravity[0,data['joints']]+=tcp_jac.T@torch.tensor(wrench,dtype=torch.float32)
            limits = robot.data.joint_effort_limits
            robot.set_joint_effort_target(torch.clamp(gravity, -limits, limits))
    torque_controller=None;wrench_reader=None
    if controller_mode.startswith('factory') or controller_mode=='isaac_osc':
        if robot_name!='panda':raise ValueError('Factory torque adapter is validated only for Panda')
        from factory_cartesian import FactoryCartesian
        if controller_mode=='isaac_osc':
            from osc_cartesian import OfficialOSC as FactoryCartesian
        data=info['left'];torque_controller=FactoryCartesian(robot,data['body'],data['joints'],data['grips'],data['offset'],sim.stage,bool(cfg.spawn.rigid_props.disable_gravity))
        wrench_reader=torque_controller
    elif controller_mode=='reference_ik_high_pd':
        from factory_cartesian import FactoryCartesian
        data=info['left'];wrench_reader=FactoryCartesian(robot,data['body'],data['joints'],data['grips'],data['offset'],sim.stage,bool(cfg.spawn.rigid_props.disable_gravity))
        # Reader construction has no actuator writes; only wrench_world is used.
    if torque_controller is not None:
        compensate_gravity();robot.write_data_to_sim();sim.step(render=False);robot.update(dt)
        from isaaclab.utils.math import quat_apply
        data=info['left'];hand=robot.data.body_state_w[0,data['body']]
        warmup_pose=torch.cat([hand[:3]+quat_apply(hand[3:7][None],torch.tensor(data['offset'][None],dtype=torch.float32))[0],hand[3:7]]).cpu().numpy()
    setup_trace=[]
    for setup_step in range(5*physics_rate):
        if torque_controller is None:compensate_gravity()
        else:torque_controller.apply(warmup_pose,warmup_pose,.08,dt,False)
        robot.write_data_to_sim()
        sim.step(render=False);robot.update(dt)
        for body in [casing,cover]+([bolt] if bolt else []):body.update(dt)
        if diagnostic and setup_step%(physics_rate//5)==0:
            from runtime.bolt_harness.measurement import read_filtered_contact_reports
            private_contact.update(dt,force_recompute=True)
            reports=fixture_reports()
            setup_trace.append({'setup_physics_step':setup_step+1,'moving_pose_world_wxyz':obj.data.root_state_w[0,:7].cpu().tolist(),'moving_velocity_world':obj.data.root_state_w[0,7:13].cpu().tolist(),'fixture_contact_reports':[dataclasses.asdict(x) for x in reports]})
    if diagnostic:write(R/'private/setup-settle-trace.json',{'evaluator_only':True,'scope':'Initial five simulation seconds before actor observations; not a robot assembly trajectory','initial_moving_position_world_m':bolt_pos if task=='peg' else cover_pos,'samples':setup_trace})
    initialization_phases=[]
    def initialization_checkpoint(phase):
        if diagnostic:
            initialization_phases.append({'phase':phase,'evaluator_only':True,'moving_pose_world_xyzw':obj.root_physx_view.get_transforms().cpu().tolist(),'moving_velocity_world':obj.root_physx_view.get_velocities().cpu().tolist()})
            write(R/'private/initialization-phases.json',initialization_phases)
    initialization_checkpoint('after_physics_warmup_before_render')
    for _ in range(8):sim.render()
    initialization_checkpoint('after_render_warmup')
    base_joint_targets=robot.data.default_joint_pos.clone();commanded_joints=base_joint_targets.clone();joint_goals={};joint_goal_poses={};joint_bias={};posture_references={};posture_reference_sources={};controller_diagnostics={}
    def pose(arm):
        p=robot.data.body_pos_w[0,info[arm]['body']].cpu().numpy();q=robot.data.body_quat_w[0,info[arm]['body']].cpu().numpy();return np.r_[p+Rotation.from_quat(q[[1,2,3,0]]).apply(info[arm]['offset']),q]
    from robot_kinematics import RobotKinematics,transform,bounded_reference_increment,CartesianReference,reset_motion_references,motion_reference_stale,PANDA_TCP_ANGULAR_SPEED_RADPS,PANDA_TCP_LINEAR_SPEED_MPS,joint_limit_violation,public_motion_reference_limits
    from robot_kinematics import posture_reference,nullspace_posture_increment
    from assembly_skills import selected_points,grasp_candidates,circular_feature,aligned_tcp,AssemblySkill,cloud,clears_support_plane,part_component,measure_pattern,mating_plane_grasp,geometric_patterns,antipodal_grasp_candidates,closed_fingers_clear_part,integer_pixel_intrinsics,project_world_points,assembly_grasp_metrics
    angular_stop_limit=PANDA_TCP_ANGULAR_SPEED_RADPS if robot_name=='panda' else .8
    linear_stop_limit=PANDA_TCP_LINEAR_SPEED_MPS if robot_name=='panda' else .08
    from gripper_control import jaw_reference_step,mass_based_grip_force,centering_increment,bumpless_open_reference,projected_jaw_pressure
    reference_speed_scale=float(os.environ.get('ASSEMBLY_XARM_REFERENCE_SPEED_SCALE','1')) if robot_name=='xarm7' else 1.
    if not math.isfinite(reference_speed_scale) or not 1<=reference_speed_scale<=2:raise ValueError('Reference speed scale must be within 1..2')
    grip_force_targets={}
    kin={};kin_diagnostics={};planned_grasps={};feature_records={};grasp_geometry=None;cartesian_references={};motion_metrics={}
    for arm,data in info.items():
        joint_names=[robot.joint_names[i] for i in data['joints']]
        limits=robot.data.soft_joint_pos_limits[0,data['joints']].cpu().numpy().copy();limits[:,0]+=.005;limits[:,1]-=.005
        kin[arm]=RobotKinematics.from_stage(sim.stage,'/World/Robot',joint_names,limits,robot.body_names[data['body']],data['offset'])
        fk=kin[arm].fk(robot.data.joint_pos[0,data['joints']].cpu().numpy());measured=pose(arm)
        kin_diagnostics[arm]={'position_error_m':float(np.linalg.norm(fk[:3,3]-measured[:3])),'angular_error_rad':float((Rotation.from_matrix(fk[:3,:3])*Rotation.from_quat(measured[[4,5,6,3]]).inv()).magnitude())}
    write(R/'checks/robot-model-fk.json',kin_diagnostics)
    initialization_checkpoint('after_readonly_robot_model_setup')
    contact_audit=os.environ.get('ASSEMBLY_CONTACT_AUDIT')=='1'
    if contact_audit:
        write(R/'private/physical-material-readback.json',{'scope':'Private measured simulation parameters, not actor observations','robot_masses_kg':robot.root_physx_view.get_masses().cpu().tolist(),'robot_material_properties':robot.root_physx_view.get_material_properties().cpu().tolist(),'moving_masses_kg':obj.root_physx_view.get_masses().cpu().tolist(),'moving_material_properties':obj.root_physx_view.get_material_properties().cpu().tolist(),'contact_filter_paths':diagnostic_filter_paths})
    write(R/'checks/robot-kinematics.json',{a:{'base':model.base.tolist(),'chain':[[left.tolist(),right.tolist(),index,axis.tolist() if axis is not None else None] for left,right,index,axis in model.chain],'joint_names':model.names,'limits':model.limits.tolist(),'offset':model.offset.tolist()} for a,model in kin.items()})
    r1_geometry={};r1_finger_joints={}
    if robot_name=='r1':
        from r1_gripper_geometry import extract_r1_gripper_geometry
        for arm in arms:
            r1_geometry[arm]=extract_r1_gripper_geometry(cfg.spawn.usd_path,arm,info[arm]['offset'])
            names=[finger.joint_path.rsplit('/',1)[-1] for finger in r1_geometry[arm].fingers]
            ids=[robot.find_joints(name)[0] for name in names]
            if all(len(found)==1 for found in ids):r1_finger_joints[arm]=[found[0] for found in ids]
    def measured_width(arm):
        values=robot.data.joint_pos[0,info[arm]['grips']].cpu().numpy()
        if robot_name=='xarm7':return xarm_gripper.opening(float(np.clip(values[0],0,.85)))
        return float(values.sum() if len(values)>1 else 2*values[0])
    def jaw_joint_target(width):
        if robot_name=='xarm7':return xarm_gripper.drive_for_opening(float(np.clip(width,xarm_gripper.opening(.85),xarm_gripper.opening(0.))))
        return width/2
    def touch(arm):return [float(torch.linalg.vector_norm(s.data.net_forces_w).item()) for s in finger_sensors[arm]]
    def finger_forces_world(arm):return [sensor.data.net_forces_w.reshape(-1,3).sum(0).cpu().numpy().tolist() for sensor in finger_sensors[arm]]
    def jaw_pressure(arm):
        if arm in r1_geometry:return r1_geometry[arm].closing_pressures(finger_forces_world(arm),Rotation.from_quat(pose(arm)[[4,5,6,3]]).as_matrix())
        if robot_name not in {'panda','xarm7'}:return touch(arm)
        return projected_jaw_pressure(finger_forces_world(arm),Rotation.from_quat(pose(arm)[[4,5,6,3]]).apply([0,1,0]))
    def record_g1_linkage(label):
        if robot_name!='xarm7':return
        base_id=info['left']['body']
        base=np.r_[robot.data.body_pos_w[0,base_id].cpu().numpy(),robot.data.body_quat_w[0,base_id].cpu().numpy()]
        drive=float(np.clip(robot.data.joint_pos[0,info['left']['grips'][0]].item(),0,.85))
        measured={}
        for name in ['left_finger','right_finger']:
            if name not in robot.body_names:continue
            i=robot.body_names.index(name)
            measured[name]=np.r_[robot.data.body_pos_w[0,i].cpu().numpy(),robot.data.body_quat_w[0,i].cpu().numpy()]
        log('g1-linkage-checks.jsonl',{'step':step_index,'label':label,'drive_rad':drive,
            'joint_positions_rad':{name:float(robot.data.joint_pos[0,i]) for i,name in enumerate(robot.joint_names) if any(k in name for k in ['drive','finger','knuckle'])},
            'jaw_width_from_drive_m':measured_width('left'),'jaw_reference_m':jaw_references['left'],
            'jaw_pressure_n':jaw_pressure('left'),'requested_pressure_n':grip_force_targets.get('left'),
            'link_errors':xarm_gripper.linkage_errors(base,drive,measured),'scope':'Measured robot-only link poses versus official G1 linkage; no object poses or semantic contact identity'})
    for arm in arms:targets[arm]=pose(arm);cartesian_references[arm]=CartesianReference(targets[arm],reference_speed_scale)
    jaw_references={a:measured_width(a) for a in arms}
    recorder=EpisodeRGBVideoRecorder(R/'episode/episode.mp4',fps=5)
    # Private grading retains nominal physical criteria; calibration is not silently certified.
    from runtime.bolt_harness.evaluator import IndependentBoltSeatingEvaluator,BoltSeatTolerances,BoltSeatMeasurement
    from runtime.bolt_harness.measurement import measure_bolt_pose_geometry,BoltSeatGeometryConfig,read_filtered_contact_reports,cap_support_contact,CapSupportContactCriteria
    from fixture_geometry import bolt_fixture_geometry,physical_calibration_matches_scale
    import yaml
    task_cfg=yaml.safe_load((settings.ROOT/'config/bolt_insertion.yaml').read_text());peg_eval=IndependentBoltSeatingEvaluator(BoltSeatTolerances(**task_cfg['tolerances']))
    geometry_cfg=BoltSeatGeometryConfig(**bolt_fixture_geometry(calibration,casing_pos,settings.CAD_METERS_PER_SOURCE_UNIT))
    from hub_evaluator import IndependentSeatEvaluator,SeatMeasurement,SeatTolerances,robot_release_verified,axisymmetric_seat_stable,settle_window_contact,settle_window_samples,aabb_separation
    settle_samples=settle_window_samples(physics_rate)
    hub_calibration_path=settings.ROOT/'config'/evaluator_profile['calibration_file'] if is_relation_task else settings.ROOT/'config/hub_evaluator_calibration.json'
    hub_calibration=json.loads(hub_calibration_path.read_text()) if hub_calibration_path.exists() else {}
    hub_depth=evaluator_profile.get('expected_root_offset_m',[0,0,.062])[2] if is_hub_task else .062
    hub_calibration_valid=physical_calibration_matches_scale(hub_calibration,settings.CAD_METERS_PER_SOURCE_UNIT)
    hub_eval=IndependentSeatEvaluator(SeatTolerances(axial_depth_m=hub_depth,tilt_tolerance_deg=2.2 if task=='hub-cover' else 2.0),calibration_status='calibrated' if hub_calibration_valid else 'pending')
    release_diagnostic={};release_local_corners=None;release_hand_ids=[];release_hand_corners=None
    if robot_name=='xarm7' and is_hub_task:
        def bound_corners(points):
            low,high=points.min(0),points.max(0)
            return np.array([[x,y,z] for x in [low[0],high[0]] for y in [low[1],high[1]] for z in [low[2],high[2]]])
        with np.load(settings.ROOT/'assets/public_cad_models'/skill_data['moving_cad_cache']) as model:release_local_corners=bound_corners(model['points'])
        if all(name in robot.body_names for name in xarm_gripper.hand_points):
            release_hand_ids=[robot.body_names.index(name) for name in xarm_gripper.hand_points]
            release_hand_corners=np.array([bound_corners(mesh) for mesh in xarm_gripper.hand_points.values()])
    def release_verified():
        widths=[measured_width(a) for a in arms];forces=[touch(a) for a in arms]
        if release_local_corners is None:return robot_release_verified(widths,forces)
        if release_hand_corners is None:
            release_diagnostic.update(status='UNKNOWN',reason='Missing measured G1 hand bodies')
            return False
        rotations=Rotation.from_quat(robot.data.body_quat_w[0,release_hand_ids].cpu().numpy()[:,[1,2,3,0]]).as_matrix()
        hand=np.einsum('lij,lkj->lki',rotations,release_hand_corners)+robot.data.body_pos_w[0,release_hand_ids].cpu().numpy()[:,None,:]
        hand=hand.reshape(-1,3)
        rotation=Rotation.from_quat(obj.data.root_quat_w[0].cpu().numpy()[[1,2,3,0]]).as_matrix()
        part=release_local_corners@rotation.T+obj.data.root_pos_w[0].cpu().numpy()
        gap=aabb_separation([hand.min(0),hand.max(0)],[part.min(0),part.max(0)])
        release_diagnostic.update(status='measured',hand_object_clearance_m=gap,required_clearance_m=.005,
            source='Evaluator-only actual robot-link and object-pose conservative bounds; never actor input')
        return robot_release_verified(widths,forces,withdrawn=gap>.005)
    if is_fixture_task:
        from relation_evaluator import RelationMeasurement, build_relation_evaluator
        relation_calibration_path=settings.ROOT/'config'/evaluator_profile['calibration_file']
        relation_calibration=json.loads(relation_calibration_path.read_text()) if relation_calibration_path.exists() else None
        relation_eval=build_relation_evaluator(task,evaluator_profile,relation_calibration,settings.CAD_METERS_PER_SOURCE_UNIT)
    initial_z=float(obj.data.root_pos_w[0,2]);seen_pickup=False;last_position=None;stable_since=None;last_private={};angular_history=[];seat_positions=[];seat_axes=[];hub_contacts=[];hub_penetrations=[]
    peg_process=None;peg_process_stream=None;peg_relative_previous=None;peg_applied_load_n=0.
    if peg_mvp_enabled:
        sys.path.insert(0,'/opt/cape_peg')
        from peg_mvp import PegProcessEvaluator,PegCriteria,scene_query_hit,diagnostic_load_force
        initial_valid=bool(not peg_probe and np.linalg.norm(obj.data.root_pos_w[0,:2].cpu().numpy()-np.asarray(geometry_cfg.selected_hole_xy_m))>.03 and initial_z<geometry_cfg.cap_support_plane_z_m)
        peg_process=PegProcessEvaluator(PegCriteria(version='cape-peg-process-v0.3-collider-dev',support_evidence='collider_geometry',aperture_evidence='collider_sweep'),calibrated=False,privileged=bool(diagnostic) or os.environ.get('ASSEMBLY_PEG_VALIDATION_ONLY')=='1',initial_state_valid=initial_valid)
        write(R/'checks/peg-mvp-criteria.json',dataclasses.asdict(peg_process.criteria))
        peg_process_stream=(R/'private/peg-process-measurements.jsonl').open('x')
    def native_bolt_contact_diagnostic():
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools
        native=get_physx_simulation_interface()
        headers,data=native.get_contact_report()
        return {'sleeping':bool(native.is_sleeping(native.get_attached_stage(),PhysicsSchemaTools.sdfPathToInt(Sdf.Path('/World/Bolt')))),
            'native_contact_header_count':len(headers),'native_contact_point_count':len(data),
            'native_contact_impulses_ns':[list(point.impulse) for point in data]}
    def cap_bearing_surface_query(g):
        from omni.physx import get_physx_scene_query_interface
        query=get_physx_scene_query_interface()
        axis=np.asarray(g.bolt_axis_w);center=np.asarray(g.cap_underside_center_w_m)
        rotation=Rotation.from_quat(np.asarray(g.bolt_root_quat_wxyz)[[1,2,3,0]])
        hits=[];all_colliders=set();callback_errors=[]
        for index,angle in enumerate(np.linspace(0,2*np.pi,8,endpoint=False)):
            radial=rotation.apply([.0044*np.cos(angle),0,.0044*np.sin(angle)])
            def collect(hit):
                try:record=scene_query_hit(hit,axis,index)
                except Exception as error:
                    callback_errors.append(str(error))
                    return False
                collider=record['collider']
                all_colliders.add(collider)
                if collider.startswith(('/World/Cover/','/World/Casing/')):
                    hits.append(record)
                return True
            query.raycast_all(tuple(center+radial+.002*axis),tuple(-axis),.06,collect,True)
        supported={hit['ray'] for hit in hits if hit['collider'].startswith('/World/Cover/') and abs(hit['gap_m'])<=.0001 and hit['normal_alignment']>=.95}
        control=query.raycast_closest((.2,0.,table_z+.2),(0.,0.,-1.),.4,True)
        return {'status':'UNKNOWN' if callback_errors else 'measured','callback_errors':callback_errors,
            'supported_ray_count':None if callback_errors else len(supported),'ray_count':8,'hits':hits,'all_hit_colliders':sorted(all_colliders),
            'table_query_control':{'hit':bool(control.get('hit',False)),'collision':str(control.get('collision',''))},
            'scope':'Actual collider surface query at CAD cap-bearing annulus; geometric evidence, not measured force or certified support'}
    def shaft_bore_query(g):
        from omni.physx import get_physx_scene_query_interface
        from runtime.bolt_harness.measurement import SHANK_RADIUS_M
        query=get_physx_scene_query_interface();axis=np.asarray(g.bolt_axis_w)
        if axis[2]<.9:return {'status':'not_applicable','reason':'shaft is not near the insertion direction'}
        rotation=Rotation.from_quat(np.asarray(g.bolt_root_quat_wxyz)[[1,2,3,0]])
        root=np.asarray(g.bolt_root_pos_w_m);samples=[];errors=[]
        for depth in [.001,.006,.011]:
            z=geometry_cfg.casing_entry_plane_z_m-depth
            if z<g.bolt_tip_pos_w_m[2]:continue
            origin=root+axis*((z-root[2])/axis[2])
            for angle in np.linspace(0,2*np.pi,16,endpoint=False):
                direction=rotation.apply([np.cos(angle),0,np.sin(angle)])
                hits=[]
                def collect(hit):
                    try:
                        if str(hit.collision).startswith('/World/Casing/'):
                            hits.append(float(hit.distance))
                    except Exception as error:errors.append(str(error))
                    return True
                query.raycast_all(tuple(origin),tuple(direction),.008,collect,True)
                samples.append({'depth_below_entry_m':depth,'angle_rad':float(angle),
                    'nearest_wall_distance_m':min(hits) if hits else None})
        distances=[row['nearest_wall_distance_m'] for row in samples if row['nearest_wall_distance_m'] is not None]
        sweep=shaft_casing_sweep(g)
        return {'status':'UNKNOWN' if errors or len(distances)!=len(samples) or not samples else 'measured',
            'errors':errors,'samples':samples,'shaft_envelope_radius_m':SHANK_RADIUS_M,
            'minimum_envelope_clearance_m':min(distances)-SHANK_RADIUS_M if distances else None,
            **sweep,
            'scope':'Private sampled real casing collider versus conservative shaft envelope; diagnostic only, not a replacement scorer'}
    def shaft_casing_sweep(g):
        from omni.physx import get_physx_scene_query_interface
        from runtime.bolt_harness.measurement import SHANK_RADIUS_M
        query=get_physx_scene_query_interface();axis=np.asarray(g.bolt_axis_w);root=np.asarray(g.bolt_root_pos_w_m)
        if axis[2]<.9:return {'sweep_status':'not_applicable','casing_sweep_clear':False}
        errors=[]
        entry=root+axis*((geometry_cfg.casing_entry_plane_z_m-root[2])/axis[2])
        tip=np.asarray(g.bolt_tip_pos_w_m);travel=tip-entry;length=float(np.linalg.norm(travel))
        sweep_hits=[]
        def collect_sweep(hit):
            try:
                if str(hit.collision).startswith('/World/Casing/'):
                    sweep_hits.append({'collider':str(hit.collision),'distance_m':float(hit.distance),
                        'point_world_m':list(hit.position),'normal_world':list(hit.normal)})
            except Exception as error:errors.append(str(error))
            return True
        if length>1e-6:
            query.sweep_sphere_all(SHANK_RADIUS_M-peg_process.criteria.penetration_m,
                tuple(entry),tuple(travel/length),length,collect_sweep,True)
        sweep_control=query.sweep_sphere_closest(.001,(.2,0.,table_z+.2),(0.,0.,-1.),.4,True)
        valid=not errors and length>1e-6 and bool(sweep_control.get('hit',False))
        return {'sweep_status':'measured' if valid else 'UNKNOWN','sweep_errors':errors,
            'casing_sweep_clear':not sweep_hits if valid else None,
            'casing_sweep_hits':sweep_hits,'sweep_length_m':length,
            'shaft_radius_allowance_m':peg_process.criteria.penetration_m,
            'table_sweep_control_hit':bool(sweep_control.get('hit',False)),
            'sweep_scope':'Continuous shaft-envelope sphere sweep against actual casing collider; nominal radius minus unchanged numerical penetration allowance'}
    def grade():
        nonlocal native_success,seen_pickup,last_position,stable_since,last_private,peg_relative_previous
        position=obj.data.root_pos_w[0].cpu().numpy();q=obj.data.root_quat_w[0].cpu().numpy();lin=obj.data.root_lin_vel_w[0].cpu().numpy();ang=obj.data.root_ang_vel_w[0].cpu().numpy();held=any(measured_width(a)<.075 and all(x>.02 for x in touch(a)) for a in arms)
        measurement_position=position
        if is_hub_task:
            from hub_evaluator import datum_equivalent_root
            measurement_position=datum_equivalent_root(position,Rotation.from_quat(q[[1,2,3,0]]).as_matrix(),Rotation.from_euler('xyz',evaluator_profile['expected_rotation_xyz_deg'],degrees=True).as_matrix(),skill_data['moving_feature_local_m'])
        if held and position[2]>initial_z+.005:seen_pickup=True
        now=step_index*dt
        speed=0 if last_position is None else float(np.linalg.norm(measurement_position-last_position)/dt)
        history.append(speed);history[:]=history[-settle_samples:]
        angular_history.append(float(np.linalg.norm(ang)));angular_history[:]=angular_history[-settle_samples:]
        stable=len(history)==settle_samples and max(history)<.01 and max(angular_history)<.02
        if task in {'hub-cover','hub-cover-small','m10-casing-bolt','bearing-input-top-onto-shaft'}:
            seat_positions.append(measurement_position.copy());seat_positions[:]=seat_positions[-settle_samples:]
            seat_axes.append(Rotation.from_quat(q[[1,2,3,0]]).apply(skill_data['moving_feature_axis_local']));seat_axes[:]=seat_axes[-settle_samples:]
            stable=axisymmetric_seat_stable(seat_positions,seat_axes,settle_samples) and max(history)<.01
        if task=='peg':
            g=measure_bolt_pose_geometry(time_s=now,bolt_root_pos_w_m=position,bolt_root_quat_wxyz=q,bolt_root_lin_vel_w_mps=lin,bolt_root_ang_vel_w_radps=ang,env_origin_w_m=(0,0,0),geometry_config=geometry_cfg)
            reports=fixture_reports()
            support=cap_support_contact(g,reports,CapSupportContactCriteria(**task_cfg['sampling_criteria']['cap_support']))
            m=BoltSeatMeasurement(time_s=now,through_selected_cover_hole=g.through_selected_cover_hole,entered_selected_casing_opening=g.entered_selected_casing_opening,casing_entry_depth_m=g.casing_entry_depth_m,shaft_radial_error_m=g.shaft_radial_error_m,cap_support_gap_m=g.cap_support_gap_m,cap_support_contact=support,penetration_m=g.geometric_penetration_bound_m,premature_bottoming=g.premature_bottoming,physically_held=held,insertion_provenance_valid=seen_pickup,controller_stopped=all(np.linalg.norm(pose(a)[:3]-targets[a][:3])<.003 for a in arms),robot_contact_or_support=any(any(v>.02 for v in touch(a)) for a in arms),axial_position_m=g.axial_position_m,relative_linear_speed_mps=g.relative_linear_speed_mps,relative_angular_speed_radps=g.relative_angular_speed_radps)
            status=peg_eval.observe(m)
            bolt_calibration=task_cfg.get('evaluator_calibration',{})
            bolt_calibration_valid=physical_calibration_matches_scale(bolt_calibration,settings.CAD_METERS_PER_SOURCE_UNIT)
            native_success=status.task_success and bolt_calibration_valid
            last_private=dataclasses.asdict(m)|dataclasses.asdict(status)|{
                'task_success':native_success,
                'unscaled_evaluator_candidate_success':status.task_success,
                'evaluator_calibration_status':'calibrated' if bolt_calibration_valid else 'pending_scale_revalidation',
                'evaluator_calibration_scale_m_per_source_unit':bolt_calibration.get('cad_meters_per_source_unit'),
            }
            if peg_process is not None:
                from assembly_skills import hand_plane_clearance
                # Target-filtered finger forces never alter the public force/safety path.
                target_forces=[float(torch.linalg.vector_norm(s.data.force_matrix_w,dim=-1).sum()) for s in finger_sensors['left']]
                robot_target_force=sum(float(torch.linalg.vector_norm(sensor.data.force_matrix_w[:,:,diagnostic_target_indices],dim=-1).sum()) for sensor in diagnostic_contacts.values())
                tcp=pose('left');rotation=Rotation.from_quat(tcp[[4,5,6,3]])
                relative_position=rotation.inv().apply(position-tcp[:3])
                relative_axis=rotation.inv().apply(g.bolt_axis_w)
                relative_speed=None;relative_axis_speed=None
                if peg_relative_previous is not None:
                    relative_speed=float(np.linalg.norm(relative_position-peg_relative_previous[0])/dt)
                    relative_axis_speed=float(np.linalg.norm(relative_axis-peg_relative_previous[1])/dt)
                target_controlled=(False if min(target_forces)<=.02 else None if relative_speed is None else bool(relative_speed<.01 and relative_axis_speed<.1))
                peg_relative_previous=(relative_position.copy(),relative_axis.copy())
                if robot_name=='xarm7':
                    drive=float(np.clip(robot.data.joint_pos[0,info['left']['grips'][0]].item(),0.,.85))
                    hand_points=xarm_gripper.world_points(tcp[:3],rotation.as_matrix(),drive,support_only=True)
                    bounds=np.column_stack([hand_points.min(0),hand_points.max(0)])
                else:
                    bounds=[]
                    for axis in np.eye(3):
                        bounds.append((hand_plane_clearance(tcp[:3],rotation,measured_width('left'),axis,np.zeros(3)),
                                       -hand_plane_clearance(tcp[:3],rotation,measured_width('left'),-axis,np.zeros(3))))
                    bounds=np.array(bounds)
                hand_distance=float(np.linalg.norm(np.maximum(np.maximum(bounds[:,0]-position,position-bounds[:,1]),0)))
                process_sample=dataclasses.asdict(m)|{
                    'axis_tilt_rad':float(np.arccos(np.clip(g.bolt_axis_w[2],-1,1))),
                    'target_controlled':target_controlled,
                    'initial_state_valid':initial_valid,
                    'support_evidence':'collider_geometry',
                    'aperture_evidence':'collider_sweep',
                    'non_casing_penetration_bound_m':max(g.cover_shaft_penetration_bound_m,g.cap_head_penetration_bound_m),
                    'object_tcp_relative_speed_mps':relative_speed,
                    'object_tcp_axis_change_per_s':relative_axis_speed,
                    'lifted_from_support':bool(position[2]>initial_z+.005),
                    'in_target_neighborhood':bool(np.linalg.norm(position[:2]-np.asarray(geometry_cfg.selected_hole_xy_m))<.06),
                    'robot_released':bool(robot_target_force<.02 and max(target_forces)<.02),
                    'robot_withdrawn':bool(hand_distance>.03 and robot_target_force<.02),
                    'target_finger_forces_n':target_forces,
                    'robot_target_force_n':robot_target_force,
                    'conservative_hand_to_bolt_root_distance_m':hand_distance,
                    'fixture_net_force_world_n':private_contact.data.net_forces_w.cpu().tolist(),
                    'fixture_filtered_force_world_n':private_contact.data.force_matrix_w.cpu().tolist(),
                    'fixture_contact_reports':[dataclasses.asdict(report) for report in reports],
                    'diagnostic_applied_downward_force_n':peg_applied_load_n,
                    'evidence_scope':'Private target-filtered contacts and embodiment-specific conservative hand bounds; not actor input',
                }
                process_sample['cap_bearing_geometry_supported']=False
                process_sample['casing_sweep_clear']=False
                if abs(g.cap_support_gap_m)<.003 and g.shaft_radial_error_m<.01:
                    try:
                        query_result=cap_bearing_surface_query(g)
                        process_sample['bearing_surface_query']=query_result
                        query_valid=query_result['status']=='measured' and query_result['table_query_control']['hit']
                        process_sample['cap_bearing_geometry_supported']=bool(query_result['supported_ray_count']>=3) if query_valid else None
                        sweep_result=shaft_casing_sweep(g)
                        process_sample['shaft_casing_sweep']=sweep_result
                        process_sample['casing_sweep_clear']=sweep_result['casing_sweep_clear']
                    except Exception as query_error:
                        process_sample['cap_bearing_geometry_supported']=None
                        process_sample['bearing_surface_query']={'status':'UNKNOWN','error':str(query_error)}
                if diagnostic:
                    try:process_sample['native_contact_diagnostic']=native_bolt_contact_diagnostic()
                    except Exception as contact_error:process_sample['native_contact_diagnostic']={'status':'UNKNOWN','error':str(contact_error)}
                    if abs(g.cap_support_gap_m)<.003 and g.shaft_radial_error_m<.01 and step_index%control_decimation==0:
                        process_sample['shaft_bore_query']=shaft_bore_query(g)
                peg_process_stream.write(json.dumps(process_sample,allow_nan=False)+'\n')
                peg_process_stream.flush()
                peg_process.observe(process_sample)
                # Runtime completion is distinct from calibrated benchmark certification.
                from peg_completion_feedback import peg_process_completed
                process_result=peg_process.result()
                legacy_native_success=native_success
                native_success=peg_process_completed(process_result)
                last_private.update(legacy_native_success=legacy_native_success,
                    task_success=native_success,
                    completion_evaluator=process_result['evaluator_version'],
                    native_success_scope='CAPE process physical completion; benchmark calibration status remains separate',
                    process_evaluation_status=process_result['evaluation_status'],
                    process_assembly_success=process_result['assembly_success'])
        elif is_hub_task:
            # CAD root relation is evaluator-only. No task targets are returned to the actor.
            expected=np.array(casing_pos)+np.array(evaluator_profile['expected_root_offset_m']);radial=float(np.linalg.norm(measurement_position[:2]-expected[:2]));axial=float(measurement_position[2]-casing_pos[2])
            axis_local=np.array(skill_data['moving_feature_axis_local'],dtype=float);axis=Rotation.from_quat(q[[1,2,3,0]]).apply(axis_local);expected_axis=Rotation.from_euler('xyz',evaluator_profile['expected_rotation_xyz_deg'],degrees=True).apply(axis_local);cosine=float(np.dot(axis,expected_axis)/(np.linalg.norm(axis)*np.linalg.norm(expected_axis)));tilt=float(np.degrees(np.arccos(np.clip(cosine,-1,1))));contact=False
            reports=fixture_reports();contact=sum(abs(x.force_n) for x in reports)>.1;penetration=max([0.]+[-x.separation_m for x in reports])
            contact_current=contact;hub_contacts.append(contact);hub_contacts[:]=hub_contacts[-settle_samples:]
            hub_penetrations.append(penetration);hub_penetrations[:]=hub_penetrations[-settle_samples:]
            contact,penetration=settle_window_contact(hub_contacts,hub_penetrations,settle_samples)
            result=hub_eval.evaluate(SeatMeasurement(axial_depth_m=axial,radial_error_m=radial,tilt_deg=tilt,penetration_m=penetration,released=release_verified(),stable=stable,contact_valid=contact,settle_speed_mps=max(history),settle_window_s=len(history)*dt))
            native_success=result.task_success
            last_private=result.to_public_dict()|{'pickup_observed':seen_pickup,'evaluator_only':True,'max_settle_angular_speed_radps':max(angular_history),'contact_valid_current':contact_current,'contact_valid_samples_in_window':sum(hub_contacts),'release_diagnostic':dict(release_diagnostic)}
            last_private.update(datum_equivalent_position_world_m=measurement_position.tolist(),raw_root_radial_error_m=float(np.linalg.norm(position[:2]-expected[:2])),radial_reference='Public CAD mating datum under actual and canonical rotations; unchanged tolerances')
        else:
            reports=fixture_reports()
            measurement=RelationMeasurement(
                simulation_time_s=now,
                moving_position_world_m=position.tolist(),
                moving_quaternion_world_wxyz=q.tolist(),
                fixture_position_world_m=np.asarray(casing_pos,dtype=float).tolist(),
                fixture_quaternion_world_wxyz=np.asarray(fixture_quat,dtype=float).tolist(),
                moving_linear_velocity_world_mps=lin.tolist(),
                moving_angular_velocity_world_radps=ang.tolist(),
                released=release_verified(),
                gripper_contact_forces_n=[float(sum(touch(a))) for a in arms],
                fixture_contact_force_n=float(sum(abs(report.force_n) for report in reports)),
                maximum_penetration_m=float(max([0.]+[-report.separation_m for report in reports])),
                stable_window_s=len(history)*dt,
                stable=stable,
                pickup_observed=seen_pickup,
                fixture_contact_points_world_m=[list(report.point_w_m) for report in reports],
                fixture_contact_normal_forces_world_n=[list(report.force_vector_w_n) for report in reports],
            )
            relation_result=relation_eval.observe(measurement)
            native_success=bool(relation_result['task_success'])
            last_private=relation_result|{'candidate_reset_hint_is_not_ground_truth':True,'evaluator_only':True,'pickup_diagnostic':{'initial_z_reference_m':initial_z,'held_from_current_finger_contacts':held,'height_above_initial_m':float(position[2]-initial_z)}}

        if diagnostic:
            last_private['diagnostic_pose_world_wxyz']=np.r_[position,q].tolist()
            last_private['fixture_contact_diagnostic']={
                'net_force_world_n':private_contact.data.net_forces_w.cpu().tolist(),
                'filtered_force_world_n':private_contact.data.force_matrix_w.cpu().tolist(),
                'reports':[dataclasses.asdict(report) for report in fixture_reports()],
                'scope':'Private sensor diagnostic, not actor input or replacement contact evidence'}
        last_position=measurement_position.copy();trace.append({'step':step_index,**last_private})
    def advance(n=control_decimation):
        nonlocal step_index,peg_applied_load_n
        if snapshot()['terminal']:raise RuntimeError('episode endpoint reached')
        if peg_mvp_enabled and step_index*dt>=float(os.environ.get('ASSEMBLY_PEG_SIM_BUDGET_S','600')):
            terminate('simulation_budget',simulation_time_s=step_index*dt)
            raise RuntimeError('Peg MVP simulation-time budget reached')
        begin_step(step_index);started=time.monotonic()
        joint_targets=base_joint_targets.clone();before={a:pose(a).copy() for a in arms}
        if torque_controller is None:
            for arm,data in info.items():
                current=pose(arm);target=targets[arm];reference_paused=motion_reference_stale(cartesian_references[arm].pose,current);reference=cartesian_references[arm].advance(target,current,control_dt);current_joints=robot.data.joint_pos[0,data['joints']]
                tcp_jac_np=sdk_tcp_jacobian(arm,current);task_increment=None;posture_increment=None
                if controller_mode=='reference_ik_high_pd':
                    tcp_jac=torch.tensor(tcp_jac_np,dtype=current_joints.dtype,device=current_joints.device)
                    data['ik'].set_command(torch.tensor(reference[None],dtype=torch.float32))
                    local=data['ik'].compute(torch.tensor(current[None,:3],dtype=torch.float32),torch.tensor(current[None,3:],dtype=torch.float32),tcp_jac[None],current_joints[None])[0]
                    posture=posture_references.get(arm,base_joint_targets[0,data['joints']].cpu().numpy())
                    task_increment=(local-current_joints).cpu().numpy().copy();posture_increment=nullspace_posture_increment(tcp_jac.cpu().numpy(),current_joints.cpu().numpy(),posture,control_dt)
                    local+=torch.tensor(posture_increment,dtype=local.dtype,device=local.device)
                    joint_goals[arm]=local.cpu().numpy()
                elif arm not in joint_goal_poses or np.linalg.norm(reference-joint_goal_poses[arm])>1e-8:
                    solution=kin[arm].solve(transform(reference[:3],reference[3:]),commanded_joints[0,data['joints']].cpu().numpy())
                    if not solution['reachable']:raise ValueError('Cartesian target has no supported robot-model IK solution')
                    joint_goals[arm]=np.array(solution['joints']);joint_goal_poses[arm]=reference.copy()
                if arm not in joint_bias:joint_bias[arm]=np.zeros(len(data['joints']))
                # Accumulated reference trajectory provides actual torque while limiting
                # reference velocity, tracking lead and integral load compensation.
                measured=current_joints.cpu().numpy();goal=joint_goals[arm]
                bias_before=joint_bias[arm].copy();bias_delta=(goal-measured)*control_dt*.10
                friction_bias=np.zeros_like(measured)
                if robot_name=='xarm7':
                    from xarm_robot_cfg import tracking_integral_increment,tracking_integral_active,friction_tracking_bias
                    static_friction=[friction[robot.joint_names[i]]['static_friction_nm'] for i in data['joints']]
                    friction_bias=friction_tracking_bias(goal-measured,static_friction,robot.data.joint_stiffness[0,data['joints']].cpu().numpy())
                    near_goal=tracking_integral_active(target,reference,current,reference_paused)
                    bias_delta=tracking_integral_increment(goal-measured,bias_before,control_dt,near_goal,friction_bias)
                joint_bias[arm]=np.clip(joint_bias[arm]+bias_delta,-.02,.02)
                desired=torch.tensor(goal+np.clip(joint_bias[arm]+friction_bias,-.02,.02),dtype=torch.float32)
                previous=commanded_joints[0,data['joints']]
                delta_joints=desired-previous
                joint_step_scale=min(1.,.0025/max(float(torch.max(torch.abs(delta_joints))),1e-12));increment=delta_joints*joint_step_scale
                bound_diagnostics={}
                bounded=bounded_reference_increment(tcp_jac_np,np.zeros(3),increment.cpu().numpy(),control_dt,bound_diagnostics)
                new=previous+torch.tensor(bounded,dtype=torch.float32)
                lead=new-current_joints;joint_lead_scale=min(1.,.008/max(float(torch.max(torch.abs(lead))),1e-12));new=current_joints+lead*joint_lead_scale
                limits=robot.data.soft_joint_pos_limits[:,data['joints']].clone();limits[:,:,0]+=.005;limits[:,:,1]-=.005
                joint_targets[:,data['joints']]=torch.clamp(new,limits[0,:,0],limits[0,:,1]);joint_targets[:,data['grips']]=jaw_joint_target(widths[arm])
                clamp_delta=(joint_targets[0,data['joints']]-new).cpu().numpy();bias_clip_mask=np.abs(bias_before+bias_delta)>.02
                scales={'joint_step':float(joint_step_scale),'tcp_linear':bound_diagnostics['tcp_linear_scale'],'tcp_angular':bound_diagnostics['tcp_angular_scale'],'joint_lead':float(joint_lead_scale)}
                active={name:value<1. for name,value in scales.items()};active.update(joint_position_clamp=bool(np.any(clamp_delta!=0)),bias_clip=bool(np.any(bias_clip_mask)),reference_pause=bool(reference_paused))
                diagnostic=controller_diagnostics.setdefault(arm,{'interval_start_step':step_index,'updates':0,'activation_counts':dict.fromkeys(active,0),'minimum_scales':dict.fromkeys(scales,1.)})
                diagnostic['updates']+=1
                for name,value in active.items():diagnostic['activation_counts'][name]+=int(value)
                for name,value in scales.items():diagnostic['minimum_scales'][name]=min(diagnostic['minimum_scales'][name],value)
                diagnostic['last_control_update']={'start_step':step_index,'end_step':step_index+n,'control_dt_s':control_dt,'actual_posture_joints_rad':measured.tolist(),'posture_reference_joints_rad':np.asarray(posture).tolist() if controller_mode=='reference_ik_high_pd' else None,'posture_reference_source':posture_reference_sources.get(arm,'home_fallback') if controller_mode=='reference_ik_high_pd' else 'not_applicable','task_ik_increment_rad':task_increment.tolist() if task_increment is not None else None,'nullspace_increment_rad':posture_increment.tolist() if posture_increment is not None else None,'joint_goal_rad':goal.tolist(),'bias_before_rad':bias_before.tolist(),'bias_delta_unclipped_rad':bias_delta.tolist(),'bias_after_rad':joint_bias[arm].tolist(),'bias_clip_mask':bias_clip_mask.tolist(),'tcp_actual_wxyz':current.tolist(),'tcp_target_wxyz':target.tolist(),'tcp_reference_wxyz':reference.tolist(),'command_before_rad':previous.cpu().numpy().tolist(),'command_after_rad':joint_targets[0,data['joints']].cpu().numpy().tolist(),'proposed_lead_rad':lead.cpu().numpy().tolist(),'joint_clamp_delta_rad':clamp_delta.tolist(),'reference_advance_paused':bool(reference_paused),'scales':scales,**bound_diagnostics}
                diagnostic['last_control_update']['friction_feedforward_bias_rad']=friction_bias.tolist()
            commanded_joints[:]=joint_targets
            robot.set_joint_position_target(joint_targets);compensate_gravity();robot.write_data_to_sim()
        else:
            current=pose('left');reference=cartesian_references['left'].advance(targets['left'],current,control_dt)
        armed=snapshot()['decision_started_monotonic'] is not None
        for _ in range(n):
            expired_payloads=[a for a in payload_estimates if min(jaw_pressure(a))<=.02]
            for arm in expired_payloads:payload_estimates.pop(arm,None)
            if expired_payloads and torque_controller is None:compensate_gravity();robot.write_data_to_sim()
            if torque_controller is None:
                for arm,data in info.items():
                    forces=jaw_pressure(arm);pressure=grip_force_targets.get(arm)
                    if pressure is not None and min(forces)<.2:pressure=min(pressure,2.)
                    jaw_references[arm]=jaw_reference_step(jaw_references[arm],widths[arm],max(forces),pressure,dt)
                    joint_targets[:,data['grips']]=jaw_joint_target(jaw_references[arm])
                robot.set_joint_position_target(joint_targets);robot.write_data_to_sim()
            if torque_controller is not None:
                forces=jaw_pressure('left');pressure=grip_force_targets.get('left')
                if pressure is not None and min(forces)<.2:pressure=min(pressure,2.)
                if controller_mode=='isaac_osc':torque_controller.payload_estimate=payload_estimates.get('left') if min(jaw_pressure('left'))>.02 else None
                torque_controller.apply(reference,targets['left'],widths['left'],dt,armed,pressure,max(forces));robot.write_data_to_sim()
            if peg_mvp_enabled and os.environ.get('ASSEMBLY_PEG_LOAD_PROBE')=='1':
                peg_applied_load_n=diagnostic_load_force(os.environ.get('ASSEMBLY_DIAGNOSTIC'),True,step_index*dt)
                external=torch.zeros((1,1,3),dtype=torch.float32)
                external[0,0,2]=-peg_applied_load_n
                bolt.set_external_force_and_torque(external,torch.zeros_like(external),is_global=True)
                bolt.write_data_to_sim()
            sim.step(render=False);robot.update(dt)
            for sensor in diagnostic_contacts.values():sensor.update(dt)
            if torque_controller is None and controller_mode=='reference_ik_high_pd':
                from isaaclab.utils.math import quat_apply
                data=info['left'];body_index=data['body'];omega=robot.data.body_ang_vel_w[0,body_index];r=quat_apply(robot.data.body_quat_w[0,body_index][None],torch.tensor(data['offset'][None],dtype=torch.float32))[0];velocity=robot.data.body_link_lin_vel_w[0,body_index]+torch.cross(omega,r,dim=-1)
                if armed and (float(torch.linalg.vector_norm(velocity))>linear_stop_limit or float(torch.linalg.vector_norm(omega))>angular_stop_limit):
                    try:
                        contact_pairs=[]
                        for path,sensor in diagnostic_contacts.items():
                            for filter_index,other in enumerate(diagnostic_filter_paths):
                                reports=read_filtered_contact_reports(sensor._contact_physx_view,dt_s=dt,sensor_index=0,filter_index=filter_index)
                                if reports:contact_pairs.append({'robot_body':path,'other':other,'reports':[dataclasses.asdict(report) for report in reports]})
                        write(R/'private/emergency-contact-pairs.json',{'step':step_index,'pairs':contact_pairs,'scope':'Private diagnostic only; not actor input'})
                    except Exception as error:
                        write(R/'private/emergency-contact-pairs.json',{'step':step_index,'status':'UNKNOWN','error':str(error)})
                    # Preserve the triggering tick, which precedes periodic logging.
                    log('emergency-motion.jsonl',{'step':step_index,'arm':'left','tcp_pose':pose('left').tolist(),'tcp_target':targets['left'].tolist(),'tcp_reference':cartesian_references['left'].pose.tolist(),'measured_body_angular_velocity_world_radps':omega.cpu().tolist(),'measured_tcp_linear_velocity_world_mps':velocity.cpu().tolist(),'joint_positions':robot.data.joint_pos[0].cpu().tolist(),'joint_velocities':robot.data.joint_vel[0].cpu().tolist(),'joint_position_targets':joint_targets[0].cpu().tolist(),'finger_forces_world_n':finger_forces_world('left'),'controller_tracking':controller_diagnostics.get('left',{}),'scope':'Triggering-tick robot telemetry; finger forces alone do not identify an arm/fixture contact pair'})
                    terminate('executor_safety_stop',speed_mps=float(torch.linalg.vector_norm(velocity)),angular_speed_radps=float(torch.linalg.vector_norm(omega)),source_measurement='measured robot velocity each physics tick');raise RuntimeError('Measured motion emergency stop')
                q=robot.data.joint_pos[0,data['joints']];limits=robot.data.soft_joint_pos_limits[0,data['joints']]
                if joint_limit_violation(q.cpu().numpy(),limits.cpu().numpy()):terminate('joint_limit_stop');raise RuntimeError('Measured joint limit stop')
                wrench_reader._wrench=.8*wrench_reader._wrench+.2*wrench_reader.wrench_world()
            for item in [casing,cover]+([bolt] if bolt else []):item.update(dt)
            for sensors in finger_sensors.values():
                for sensor in sensors:sensor.update(dt)
            private_contact.update(dt);step_index+=1;grade()
        for arm,metric in motion_metrics.items():
            actual=pose(arm)[:3];start=np.array(metric['start_tcp'])[:3];goal=np.array(metric['target_tcp'])[:3];line=goal-start
            error=actual-start-line*np.dot(actual-start,line)/max(np.dot(line,line),1e-16) if np.linalg.norm(line)>1e-8 else actual-start
            metric['max_cross_track_error_m']=max(metric['max_cross_track_error_m'],float(np.linalg.norm(error)))
            metric['max_reference_tracking_error_m']=max(metric['max_reference_tracking_error_m'],float(np.linalg.norm(actual-cartesian_references[arm].pose[:3])))
        if snapshot()['decision_started_monotonic'] is not None:
            for arm in arms:
                measured=pose(arm);speed=np.linalg.norm(measured[:3]-before[arm][:3])/control_dt;angular=(Rotation.from_quat(measured[[4,5,6,3]])*Rotation.from_quat(before[arm][[4,5,6,3]]).inv()).magnitude()/control_dt
                if not np.isfinite(measured).all() or speed>linear_stop_limit or angular>angular_stop_limit:
                    terminate('executor_safety_stop',arm=arm,speed_mps=float(speed),angular_speed_radps=float(angular));raise RuntimeError('Measured motion emergency stop')
        if contact_audit:
            contact_points=[]
            moving_prefix='/World/Bolt/' if task=='peg' else '/World/Cover/'
            for body_path,sensor in diagnostic_contacts.items():
                if not body_path.endswith(('/left_finger','/right_finger')):
                    continue
                for filter_index,other_path in enumerate(diagnostic_filter_paths):
                    if other_path.startswith(moving_prefix):
                        reports=read_filtered_contact_reports(sensor._contact_physx_view,dt_s=dt,sensor_index=0,filter_index=filter_index)
                        if reports:
                            contact_points.append({'robot_body':body_path,'other':other_path,'reports':[dataclasses.asdict(report) for report in reports]})
            with (R/'private/grasp-contact-audit.jsonl').open('a') as audit_stream:
                audit_stream.write(json.dumps({'step':step_index,'time_s':step_index*dt,'target_root_state_world':obj.data.root_state_w[0].cpu().tolist(),'tcp_wxyz':pose(active_arm).tolist(),'jaw_width_m':measured_width(active_arm),'jaw_pressure_n':jaw_pressure(active_arm),'requested_pressure_n':grip_force_targets.get(active_arm),'private_contact_points':contact_points,'joint_positions':robot.data.joint_pos[0].cpu().tolist(),'applied_joint_torques':robot.data.applied_torque[0].cpu().tolist(),'filtered_robot_contacts':{path:sensor.data.force_matrix_w.cpu().tolist() for path,sensor in diagnostic_contacts.items()},'scope':'Private diagnostic only; no data returned to actor'})+'\n')
        if step_index%(physics_rate//5)==0:
            sim.render();update_ready_camera(sim,camera,.2);recorder.append(camera.data.output['rgb'][0,:,:,:3].cpu().numpy().astype(np.uint8),step_index*dt)
        if step_index%(2*physics_rate)==0:
            for arm,data in info.items():
                actual=pose(arm);measured_jac=sdk_tcp_jacobian(arm,actual)
                expected_jac=kin[arm].jacobian(robot.data.joint_pos[0,data['joints']].cpu().numpy())
                diagnostic=controller_diagnostics.get(arm)
                if diagnostic is not None:diagnostic['activation_ratios']={name:count/diagnostic['updates'] for name,count in diagnostic['activation_counts'].items()}
                log('robot-jacobian-checks.jsonl',{'step':step_index,'arm':arm,'max_linear_error_m':float(np.max(np.abs(measured_jac[:3]-expected_jac[:3]))),'max_angular_error':float(np.max(np.abs(measured_jac[3:]-expected_jac[3:]))),'finger_net_force_vectors_world_n':finger_forces_world(arm),'jaw_closing_axis_pressure_n':jaw_pressure(arm),'requested_grip_pressure_n':grip_force_targets.get(arm),'osc_posture_target':torque_controller.rest[0].cpu().tolist() if controller_mode=='isaac_osc' else None,'sdk_tcp_jacobian':measured_jac.tolist(),'public_fk_jacobian':expected_jac.tolist(),'sdk_jacobian_reference':'COM world axes shifted to link-defined TCP','body_com_position_link_m':robot.data.body_com_pos_b[0,data['body']].cpu().tolist(),'actual_posture_joints_rad':robot.data.joint_pos[0,data['joints']].cpu().tolist(),'tcp_actual_wxyz':actual.tolist(),'tcp_target_wxyz':targets[arm].tolist(),'controller_tracking':diagnostic})
                controller_diagnostics.pop(arm,None)
                if robot_name=='xarm7':record_g1_linkage('periodic')
        log('physics.jsonl',{'step':step_index,'elapsed_s':time.monotonic()-started,'robot':robot_name,**({'tcp_poses':{a:pose(a).tolist() for a in arms},'tcp_targets':{a:targets[a].tolist() for a in arms},'tcp_references':{a:cartesian_references[a].pose.tolist() for a in arms},'joint_positions':robot.data.joint_pos[0].cpu().tolist()} if step_index%physics_rate==0 else {})})
        complete_step(step_index,started)
        if native_success:terminate('native_success')
    def move(requested,posture_targets=None,verify_retention=None):
        if verify_retention is None:
            verify_retention=bool(grasp_geometry) and active_arm in requested and min(jaw_pressure(active_arm))>.02
        for arm,target in requested.items():
            value=np.asarray(target,float)
            if value.shape!=(7,) or not np.isfinite(value).all() or abs(np.linalg.norm(value[3:])-1)>.02:raise ValueError('Finite 7D TCP target and unit wxyz quaternion required')
            posture_solution=None
            if controller_mode=='reference_ik_high_pd':
                supplied=posture_targets.get(arm) if posture_targets else None
                source='supplied_waypoint' if supplied is not None else 'home_fallback'
                if supplied is None and robot_name=='xarm7':
                    posture_solution=kin[arm].solve(transform(value[:3],value[3:]),robot.data.joint_pos[0,info[arm]['joints']].cpu().numpy())
                    if posture_solution['reachable']:supplied=posture_solution['joints'];source='target_ik'
                posture_references[arm]=posture_reference(supplied,base_joint_targets[0,info[arm]['joints']].cpu().numpy(),kin[arm].limits)
                posture_reference_sources[arm]=source
            measured=pose(arm)
            reference=cartesian_references[arm].pose
            if motion_reference_stale(reference,measured):
                cartesian_references[arm]=CartesianReference(measured,reference_speed_scale)
                commanded_joints[0,info[arm]['joints']]=robot.data.joint_pos[0,info[arm]['joints']]
                joint_goal_poses.pop(arm,None);joint_bias.pop(arm,None)
            if controller_mode=='isaac_osc':
                if posture_targets and arm in posture_targets:posture_solution={'reachable':True,'joints':posture_targets[arm],'source':'same validated grasp-waypoint IK'}
                else:posture_solution=kin[arm].solve(transform(value[:3],value[3:]),robot.data.joint_pos[0,info[arm]['joints']].cpu().numpy())
                if posture_solution['reachable']:torque_controller.rest=torch.tensor([posture_solution['joints']],dtype=torch.float32)
            targets[arm]=value
            motion_metrics[arm]={'start_tcp':pose(arm).tolist(),'target_tcp':value.tolist(),'max_cross_track_error_m':0.,'max_reference_tracking_error_m':0.,'posture_ik':posture_solution}
        status='motion_timeout';progress_anchor={a:pose(a).copy() for a in requested};stalled=False;missing_grip_s=0.;expired_features=[]
        for move_step in range(3000):
            advance()
            if verify_retention:
                missing_grip_s=missing_grip_s+control_dt if any(min(jaw_pressure(a))<=.02 for a in requested) else 0.
                if missing_grip_s>=.1-1e-9:
                    held_targets,held_references=reset_motion_references({a:pose(a) for a in requested},reference_speed_scale)
                    targets.update(held_targets);cartesian_references.update(held_references)
                    for a in requested:
                        commanded_joints[0,info[a]['joints']]=robot.data.joint_pos[0,info[a]['joints']]
                        joint_goal_poses.pop(a,None);joint_bias.pop(a,None)
                    expired_features=[key for key,value in feature_records.items() if 'rigid_grasp_reference_tcp' in value]
                    for key in expired_features:feature_records.pop(key)
                    status='grasp_retention_unconfirmed';break
            if all(np.linalg.norm(pose(a)[:3]-targets[a][:3])<.0005 and (Rotation.from_quat(pose(a)[[4,5,6,3]])*Rotation.from_quat(targets[a][[4,5,6,3]]).inv()).magnitude()<.01 for a in requested):status='reached';break
            if move_step%40==39:
                # Two physical seconds without measurable motion should return
                # feedback, not spend 150 simulation seconds in a frozen move.
                stalled=all(np.linalg.norm(pose(a)[:3]-progress_anchor[a][:3])<.00005 and (Rotation.from_quat(pose(a)[[4,5,6,3]])*Rotation.from_quat(progress_anchor[a][[4,5,6,3]]).inv()).magnitude()<.0005 for a in requested)
                progress_anchor={a:pose(a).copy() for a in requested}
                if stalled:break
        result={'status':status,'stalled_without_measurable_motion':stalled,'actual_poses':{a:pose(a).tolist() for a in arms},'errors_m':{a:float(np.linalg.norm(pose(a)[:3]-targets[a][:3])) for a in requested},'angular_errors_rad':{a:float((Rotation.from_quat(pose(a)[[4,5,6,3]])*Rotation.from_quat(targets[a][[4,5,6,3]]).inv()).magnitude()) for a in requested},'path_metrics':{a:dict(motion_metrics[a]) for a in requested}}
        if status=='grasp_retention_unconfirmed':
            result.update(invalidated_feature_ids=expired_features,retention_verified=False,note='Bilateral pressure disappeared during held-part motion; motion held and rigid held-feature estimates invalidated. Errors refer to the original requested target, not the safety hold pose. Reobserve part and fixture; this is not evidence of jam or successful seating.')
            result['errors_m']={a:float(np.linalg.norm(pose(a)[:3]-np.asarray(requested[a])[:3])) for a in requested}
            result['angular_errors_rad']={a:float((Rotation.from_quat(pose(a)[[4,5,6,3]])*Rotation.from_quat(np.asarray(requested[a])[[4,5,6,3]]).inv()).magnitude()) for a in requested}
        return result
    def settle_jaws():
        # Physical jaw target is rate-limited in the robot adapter. Wait for
        # measured convergence/contact rather than assuming a one-second close.
        previous={a:measured_width(a) for a in arms};stable=0;stopped_samples=0;centering={a:np.zeros(3) for a in arms};force_endstop=False
        for _ in range(1200):
            for arm,force_target in grip_force_targets.items():
                centering_target=force_target if robot_name=='xarm7' else min(force_target,2.)
                if robot_name in {'panda','xarm7'} and max(jaw_pressure(arm))>=.5*centering_target:
                    axis=Rotation.from_quat(pose(arm)[[4,5,6,3]]).apply([0,1,0])
                    if robot_name=='xarm7':
                        increment=centering_increment(jaw_pressure(arm),centering_target,axis,control_dt,.004-np.linalg.norm(centering[arm]),minimum_fraction=.9)
                    else:increment=centering_increment(jaw_pressure(arm),centering_target,axis,control_dt,.004-np.linalg.norm(centering[arm]))
                    targets[arm][:3]+=increment;centering[arm]+=increment
            advance();actual={a:measured_width(a) for a in arms}
            stopped=all(abs(actual[a]-previous[a])/control_dt<.001 for a in arms)
            attained=all((min(jaw_pressure(a))>=.9*grip_force_targets[a]) if a in grip_force_targets else (abs(actual[a]-widths[a])<.001 or (widths[a]<actual[a] and all(f>.02 for f in touch(a)))) for a in arms)
            stable=stable+1 if stopped and attained else 0
            stopped_samples=stopped_samples+1 if stopped else 0
            previous=actual
            if stable>=8:break
            if grip_force_targets and robot_name=='xarm7' and stopped_samples>=40:
                minimum=xarm_gripper.opening(.85)
                force_endstop=all(actual[a]<=minimum+.0005 and jaw_references[a]<=minimum+.0005 and max(jaw_pressure(a))<.05 for a in grip_force_targets)
                if force_endstop:break
            if stopped_samples>=40 and not attained and not grip_force_targets:break
        return {'settled':stable>=8,'stalled_before_target':stopped_samples>=40 and not attained and not grip_force_targets,'force_endstop_without_contact':force_endstop,'measured_widths_m':actual,'tactile_centering_m':{a:v.tolist() for a,v in centering.items()},'note':'A stopped closing jaw/contact is not verified object retention; stalled_before_target is an unmet position target, not successful settling. Force mode retains its pressure convergence check; force_endstop_without_contact is a failed closure, never grasp evidence.'}
    from task_profiles import instruction as task_instruction
    instruction=task_instruction()
    def publish(index,result):
        nonlocal frame_index
        sim.render();update_ready_camera(sim,camera,0);rgb=camera.data.output['rgb'][0,:,:,:3].cpu().numpy().astype(np.uint8);depth=camera.data.output['distance_to_image_plane'][0].cpu().numpy().squeeze();folder=R/'episode/public';folder.mkdir(exist_ok=True);path=folder/f'rgb_{frame_index:05d}.png';overlay=folder/f'grid_{frame_index:05d}.png';cv2.imwrite(str(path),rgb[:,:,::-1]);grid=rgb[:,:,::-1].copy()
        for y in range(0,720,120):
            for x in range(0,960,120):cv2.putText(grid,f'{x},{y}',(x+2,y+18),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,180,255),1)
        depth=input_noise.depth(depth,f'generic-{frame_index}')
        save_depth(folder/f'depth_{frame_index:05d}.npy',depth)
        q=camera.data.quat_w_ros[0].cpu().numpy();T=np.eye(4);T[:3,:3]=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();T[:3,3]=camera.data.pos_w[0].cpu().numpy();K=integer_pixel_intrinsics(camera.data.intrinsic_matrices[0].cpu().numpy());s={'tcp_pose':pose(active_arm).tolist(),'joint_positions':robot.data.joint_pos[0].cpu().tolist(),'gripper_width_m':measured_width(active_arm),'finger_contact_force_n':touch(active_arm),'finger_net_force_vectors_world_n':finger_forces_world(active_arm),'jaw_closing_axis_pressure_n':jaw_pressure(active_arm),'finger_contacts':[x>.02 for x in touch(active_arm)],'camera_intrinsics':K.tolist(),'camera_extrinsics':T.tolist(),'camera_pixel_convention':'raw image integer-centred pixel xy; USD raster principal converted by -0.5px; optical depth in metres'}
        if robot_name not in {'panda','xarm7'} and not r1_geometry:s.pop('jaw_closing_axis_pressure_n',None)
        patterns=geometric_patterns(rgb,depth,K,T)
        for pattern in patterns:
            parent=pattern['centre'];cx,cy=map(round,parent['pixel_xy']);cv2.putText(grid,parent['candidate_id'],(cx+4,cy),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,200),1)
            for child in pattern['surrounding_candidates']:
                px,py=map(round,child['pixel_xy']);cv2.circle(grid,(px,py),4,(0,255,200),1);cv2.putText(grid,child['candidate_id'],(px+4,py-2),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,255,200),1)
        cv2.imwrite(str(overlay),grid)
        if wrench_reader is not None:s['wrench']=wrench_reader._wrench.tolist();s['wrench_frame']='world axes, moment about measured robot TCP, unfiltered real hand joint wrench'
        if 'wrench' in s:s['wrench']=input_noise.wrench(s['wrench']).tolist()
        data={'geometric_patterns':patterns,'observation_id':f'generic-{frame_index}','robot':robot_name,'task_instruction':instruction,'active_arm':active_arm,'arms':{a:{'tcp_pose':pose(a).tolist(),'gripper_width_m':measured_width(a),'commanded_gripper_width_m':widths[a],'finger_total_force_n':touch(a)} for a in arms},'features':[],'sensors':s,'rgb_path':str(path),'overlay_path':str(overlay),'result':result}
        for arm,ids in r1_finger_joints.items():
            data['arms'][arm]['finger_joint_positions_m']=robot.data.joint_pos[0,ids].cpu().tolist()
            data['arms'][arm]['finger_joint_names']=[finger.joint_path.rsplit('/',1)[-1] for finger in r1_geometry[arm].fingers]
        write(R/'ipc'/f'reply_{index:04d}.json',data);frame_index+=1;return data,depth,K,T
    def back_project(pixel):
        p=np.asarray(pixel,float)
        if p.shape!=(2,) or not np.isfinite(p).all():raise ValueError('Finite pixel xy required')
        x,y=map(round,p)
        if not 0<=x<960 or not 0<=y<720:raise ValueError('Pixel outside camera')
        z=float(depth[y,x])
        if not math.isfinite(z) or z<=0:raise ValueError('Invalid depth')
        return (T@np.r_[np.linalg.solve(K,[x,y,1])*z,1])[:3]
    def reobserve_held_cad(prior,arm,stage):
        """Refresh a held pose from public RGB-D and robot meshes, never object state."""
        result={'status':'rejected'};current_tcp=pose(arm).copy()
        try:
            from cad_registration import register
            from scipy.spatial import cKDTree
            with np.load(settings.ROOT/'assets/public_cad_models'/skill_specification()['moving_cad_cache']) as model:
                cp=model['points'];cn=model['normals']
            sim.render();update_ready_camera(sim,camera,0)
            fresh_depth=camera.data.output['distance_to_image_plane'][0].cpu().numpy().squeeze()
            observation_id=f'held-{stage}-{step_index}'
            fresh_depth=input_noise.depth(fresh_depth,observation_id)
            folder=R/'episode/public';folder.mkdir(exist_ok=True)
            rgb_path=folder/(observation_id+'.png');depth_path=save_depth(folder/(observation_id+'.npy'),fresh_depth)
            cv2.imwrite(str(rgb_path),camera.data.output['rgb'][0,:,:,:3].cpu().numpy()[:,:,::-1])
            result.update(observation_id=observation_id,source_rgb_path=str(rgb_path),source_depth_path=str(depth_path),source_simulation_time_s=step_index*dt,camera_intrinsics=K.tolist(),camera_extrinsics=T.tolist(),measured_robot_tcp=current_tcp.tolist())
            observed,_=cloud(fresh_depth,K,T)
            expected=cp@prior[:3,:3].T+prior[:3,3]
            observed=observed[cKDTree(expected).query(observed)[0]<.006]
            tcp_rotation=transform(current_tcp[:3],current_tcp[3:])[:3,:3]
            local=(observed-current_tcp[:3])@tcp_rotation
            if robot_name=='xarm7':
                drive=float(np.clip(robot.data.joint_pos[0,info[arm]['grips'][0]].item(),0,.85))
                remove=xarm_gripper.contains_hand_points(local,drive)
            else:
                hand_local=local+np.array(info[arm]['offset'])
                with np.load(settings.ROOT/'assets/public_cad_models/hand.npz') as hand_model:
                    remove=np.linalg.norm(local-np.array([0,0,-.065]),axis=1)<.04
                    for side,sign in [('left',1),('right',-1)]:
                        points_hand=hand_local-np.array([0,sign*measured_width(arm)/2,0])
                        for hull_key in hand_model.files:
                            if side in hull_key:
                                hull=hand_model[hull_key]
                                if hull.ndim==2 and hull.shape[1]==4:remove|=np.all(points_hand@hull[:,:3].T+hull[:,3]<=.001,axis=1)
            observed=observed[~remove]
            fitted=register(cp,observed,normals=cn,initial_pose=prior)
            refined=np.array(fitted['world_from_cad'])
            translation_change=float(np.linalg.norm(refined[:3,3]-prior[:3,3]))
            angle_change=float(Rotation.from_matrix(refined[:3,:3]@prior[:3,:3].T).magnitude())
            result.update(observed_points=len(observed),translation_change_m=translation_change,angular_change_rad=angle_change,registration=fitted)
            if fitted['rms_m']<.0015 and fitted['observed_surface_coverage']>.7 and translation_change<.01 and angle_change<.2:result['status']='visually_refined'
        except (ValueError,KeyError) as error:result['reason']=str(error)
        log('held-registration.jsonl',{'stage':stage,'step':step_index,'result':result,'scope':'Public sensor/CAD estimate; no object GT or task evaluator input'})
        return result,current_tcp
    def refresh_payload_estimate(arm,component,cad_pose,tcp):
        if arm not in payload_estimates:return
        cad_com=(cad_pose@np.r_[component['uniform_density_com_local_m'],1])[:3]
        inverse=Rotation.from_quat(tcp[[4,5,6,3]]).as_matrix().T
        payload_estimates[arm]['local_com_proxy_m']=inverse@(cad_com-tcp[:3])
        relative=inverse@cad_pose[:3,:3]
        payload_estimates[arm]['local_inertia_per_mass_m2']=relative@np.array(component['inertia_per_mass_local_m2'])@relative.T
    def assembly_targets(current,moving,target_feature,offset,axis_alignment):
        from assembly_skills import assembly_release_clearance
        if robot_name=='panda':
            release_clearance=assembly_release_clearance(current,moving,target_feature,measured_width(active_arm),axis_alignment)
        else:
            aligned=aligned_tcp(current,moving,target_feature,0.,axis_alignment)
            if robot_name=='xarm7':
                drive=float(np.clip(robot.data.joint_pos[0,info[active_arm]['grips'][0]].item(),0,.85))
                opened=xarm_gripper.drive_for_opening(.08)
                clearance_at_plane=xarm_gripper.swept_support_clearance(aligned[:3],Rotation.from_quat(aligned[[4,5,6,3]]).as_matrix(),drive,opened,target_feature['axis_world'],target_feature['position_m'])
            else:clearance_at_plane=r1_geometry[active_arm].support_clearance(aligned[:3],Rotation.from_quat(aligned[[4,5,6,3]]).as_matrix(),robot.data.joint_pos[0,r1_finger_joints[active_arm]].cpu().tolist(),target_feature['axis_world'],target_feature['position_m'])
            release_clearance=max(0.,.003-clearance_at_plane)
        clearance=max(.02,float(moving.get('axial_protrusion_m',.06))+.015,release_clearance+.015)
        return release_clearance,aligned_tcp(current,moving,target_feature,clearance+offset,axis_alignment),aligned_tcp(current,moving,target_feature,max(offset,release_clearance),axis_alignment)
    def dispatch(c):
        nonlocal active_arm,grasp_geometry
        action=c['action']
        if action=='observe':return {'status':'observed'}
        if action=='select_arm':
            if c['arm'] not in arms:raise ValueError('Unknown arm')
            active_arm=c['arm'];return {'status':'selected','arm':active_arm}
        if action=='back_project':return {'point_world_m':back_project(c['pixel_xy']).tolist(),'status':'measured'}
        if action in ['set_gripper','set_both_grippers']:
            values={active_arm:c['width_m']} if action=='set_gripper' else {'left':c['left_width_m'],'right':c['right_width_m']}
            if any(a not in arms or not math.isfinite(w) or not 0<=w<=.08 for a,w in values.items()):raise ValueError('Jaw width 0..0.08m and existing arm required')
            for arm in values:
                grip_force_targets.pop(arm,None)
                jaw_references[arm]=bumpless_open_reference(jaw_references[arm],measured_width(arm),values[arm])
                if torque_controller is not None:torque_controller._jaw_reference=bumpless_open_reference(torque_controller._jaw_reference,measured_width(arm),values[arm])
            widths.update(values);jaw_result=settle_jaws()
            record_g1_linkage('position_grip_settled')
            return {'status':'gripper_commanded','widths':widths.copy(),'measured_widths':{a:measured_width(a) for a in arms},'physical_settle':jaw_result}
        if action=='settle':
            duration=float(c.get('duration_s',1))
            if not .05<=duration<=5:raise ValueError('settle 0.05..5 seconds required')
            for _ in range(round(duration/control_dt)):advance()
            return {'status':'settled'}
        if action=='inspect_pose':
            target=np.r_[c['position_m'],c['quaternion_wxyz']]
            width=float(c.get('gripper_width_m',measured_width(active_arm)))
            if target.shape!=(7,) or not np.isfinite(target).all() or abs(np.linalg.norm(target[3:])-1)>.02 or not math.isfinite(width) or not 0<=width<=.08:
                raise ValueError('Finite position, unit wxyz quaternion and opening 0..0.08m required')
            solution=kin[active_arm].solve(transform(target[:3],target[3:]),robot.data.joint_pos[0,info[active_arm]['joints']].cpu().numpy())
            clearance=None;clearance_source='uncalibrated';finger_coordinates=None;hand_bounds=None
            if robot_name=='panda':
                from assembly_skills import hand_plane_clearance
                clearance=float(hand_plane_clearance(target[:3],Rotation.from_quat(target[[4,5,6,3]]),width,[0,0,1],[0,0,table_z]))
                clearance_source='public Panda conservative hand bounds'
            elif active_arm in r1_finger_joints:
                finger_coordinates=([width/2,width/2] if 'gripper_width_m' in c else robot.data.joint_pos[0,r1_finger_joints[active_arm]].cpu().tolist())
                clearance=r1_geometry[active_arm].support_clearance(target[:3],Rotation.from_quat(target[[4,5,6,3]]).as_matrix(),finger_coordinates,[0,0,1],[0,0,table_z])
                hand_bounds=r1_geometry[active_arm].mesh_bounds_world(target[:3],Rotation.from_quat(target[[4,5,6,3]]).as_matrix(),finger_coordinates)
                clearance_source='public R1 whole hand meshes; nominal symmetric mimic coordinates for proposed stroke, measured independent coordinates otherwise'
            elif robot_name=='xarm7':
                drive=xarm_gripper.drive_for_opening(float(np.clip(width,xarm_gripper.opening(.85),xarm_gripper.opening(0.))))
                world=xarm_gripper.world_points(target[:3],Rotation.from_quat(target[[4,5,6,3]]).as_matrix(),drive,support_only=True)
                clearance=float(world[:,2].min()-table_z);hand_bounds=[world.min(0).tolist(),world.max(0).tolist()]
                finger_coordinates=[drive];clearance_source='official G1 collision meshes transformed through coupled revolute linkage; whole gripper versus table only'
            return {'status':'pose_inspected','frame':'world metres, wxyz','target_tcp':target.tolist(),'gripper_width_m':width,'robot_model_ik':solution,'conservative_hand_table_clearance_m':clearance,'hand_clearance_source':clearance_source,'finger_joint_positions_used_m':finger_coordinates,'hand_mesh_bounds_world_m':hand_bounds,'scope':'No motion. Whole hand mesh bounds/support-plane estimate only; unknown geometry returns null. Bounds are not pad contact locations or free-space certification. Proposed R1 stroke uses nominal symmetric mimic tracking, not measured pad gap. Does not check the transit path, other parts, arm collision, or grasp retention. Negative clearance predicts hand/table overlap.'}
        if action=='move_both':return move({a:np.r_[c[a+'_position_m'],c.get(a+'_quaternion_wxyz') or pose(a)[3:]] for a in ['left','right']})
        if action in ['move_to','move_relative','rotate']:
            target=pose(active_arm).copy()
            if action=='move_to':target[:3]=c['position_m'];target[3:]=c.get('quaternion_wxyz',target[3:])
            if action=='move_relative':target[:3]+=np.array(c['translation_m'])
            if c.get('rotation_rad') is not None:target[3:]=(Rotation.from_rotvec(c['rotation_rad'])*Rotation.from_quat(target[[4,5,6,3]])).as_quat()[[3,0,1,2]]
            return move({active_arm:target})
        if action=='measure_pattern':return {'status':'measured','features':measure_pattern(camera.data.output['rgb'][0,:,:,:3].cpu().numpy().astype(np.uint8),c['center_pixel_xy'],c['bbox_xyxy']),'clock_convention':'image plane: 12 up, 3 right, 6 down, 9 left; relative to model-selected centre'}
        if action=='measure_feature':
            result=circular_feature(depth,K,T,c['pixel_xy'],c.get('radius_px',24),camera.data.output['rgb'][0,:,:,:3].cpu().numpy().astype(np.uint8));key=f'feature-{len(feature_records)}';result['measured_robot_tcp']=pose(active_arm).tolist();feature_records[key]=result
            return {'status':'measured','feature_id':key,**result}
        if action=='measure_object':
            from assembly_skills import observed_object_envelope
            points=selected_points(depth,K,T,c['pixel_xy'],c['bbox_xyxy'],table_z)
            return {'status':'object_measured',**observed_object_envelope(points)}
        if action=='plan_grasp':
            arm=active_arm
            if kin_diagnostics[arm]['position_error_m']>.001 or kin_diagnostics[arm]['angular_error_rad']>.01:raise ValueError('Robot FK calibration mismatch; grasp feasibility unavailable')
            points=selected_points(depth,K,T,c['pixel_xy'],c['bbox_xyxy'],table_z)
            skill_data=skill_specification()
            from cad_registration import register
            with np.load(settings.ROOT/'assets/public_cad_models'/skill_data['moving_cad_cache']) as model_file:cad_points=model_file['points'];cad_normals=model_file['normals'];cad_com_local=model_file['uniform_density_com_local_m'];cad_inertia_local=model_file['inertia_per_mass_local_m2']
            component=part_component(depth,K,T,c['pixel_xy'],table_z,radius=float(np.linalg.norm(np.ptp(cad_points,axis=0)))+.02)
            component_points=component.pop('_surface_points')
            registration=register(cad_points,component_points,support_z=table_z,normals=cad_normals)
            component['uniform_density_com_local_m']=cad_com_local.tolist();component['inertia_per_mass_local_m2']=cad_inertia_local.tolist()
            component['cad_registration']=registration
            component['cad_feature_local']=skill_data['moving_feature_local_m'];component['cad_feature_axis_local']=skill_data['moving_feature_axis_local']
            component['feature_provenance']=skill_data.get('feature_provenance','Public CAD datum; physical seating face is not certified')
            component['registered_mating_axis_world']=(np.array(registration['world_from_cad'])[:3,:3]@np.array(component['cad_feature_axis_local'])).tolist()
            component['axis_note']='PCA axis_world is the visible extent axis, not necessarily the mating axis. registered_mating_axis_world follows the directed public CAD datum, not automatically the tool approach or insertion direction. Signed material extents below describe both sides of that datum.'
            axial_coordinates=(cad_points-np.array(component['cad_feature_local']))@np.array(component['cad_feature_axis_local'])
            component['mating_axis_material_extent_relative_to_datum_m']=[float(np.min(axial_coordinates)),float(np.max(axial_coordinates))]
            component['axial_protrusion_m']=max(0.,-float(np.min(axial_coordinates)))
            registration_summary={key:registration[key] for key in ['rms_m','observed_surface_coverage','support_penetration_m']}
            registration_supported=registration['rms_m']<.002 and registration['observed_surface_coverage']>.6 and registration['support_penetration_m']<.001
            registration_summary.update(fit_supported=bool(registration_supported),criteria={'rms_m_below':.002,'observed_surface_coverage_above':.6,'support_penetration_m_below':.001})
            component_uv=project_world_points(component_points,K,T)
            component_diagnostics={'observed_bounds_world_m':[component_points.min(0).tolist(),component_points.max(0).tolist()],'observed_image_bbox_xyxy':np.r_[np.nanmin(component_uv,axis=0),np.nanmax(component_uv,axis=0)].tolist(),'observed_point_count':len(component_points),'requested_public_CAD':skill_data['moving_cad_file'],'CAD_extent_local_m':np.ptp(cad_points,axis=0).tolist(),'registration':registration_summary,'meaning':'Fit of requested public moving-part CAD to your selected visible component; low residual and adequate coverage support geometry consistency, not semantic identity proof. Compare multiple selections if the fit is poor.'}
            if robot_name not in {'panda','xarm7'}:
                return {'status':'unsupported_robot_grasp_geometry','perception_status':'cad_fit_supported' if registration_supported else 'cad_fit_unsupported','candidates':[],'component_diagnostics':component_diagnostics,'feedback':'Robot-independent perception is available, but R1 pad/contact geometry is not calibrated in the automatic planner. fit_supported=false means this selection does not sufficiently support the requested CAD; obtain better evidence or reconsider the selection before relying on that fit. IK reachability is not object identity or grasp confirmation. No grasp feasibility certification or action was produced; direct motion tools remain available.'}
            hand_hulls={}
            if robot_name=='panda':
                with np.load(settings.ROOT/'assets/public_cad_models/hand.npz') as hand_file:hand_hulls={key:hand_file[key] for key in hand_file.files}
            options=grasp_candidates(points,table_z)
            pad_offset=xarm_gripper.contact_offset_m if robot_name=='xarm7' else .0022670751
            clearance_test=xarm_gripper.clears_table if robot_name=='xarm7' else clears_support_plane
            for option in options:
                option['position_m']=(np.array(option['position_m'])-np.array(option['approach_axis_world'])*pad_offset).tolist()
                option['tcp_pad_offset_source']='official G1 innermost planar finger face versus TCP; refined at candidate closing width' if robot_name=='xarm7' else 'authored Franka inner-pad median Z=0.105667m minus TCP Z=0.1034m'
            valid=[];rejected=[]
            joints=robot.data.joint_pos[0,info[arm]['joints']].cpu().numpy()
            scene,_=cloud(depth[::3,::3],np.array([[K[0,0]/3,0,K[0,2]/3],[0,K[1,1]/3,K[1,2]/3],[0,0,1]]),T)
            from scipy.spatial import cKDTree
            distance=cKDTree(component_points).query(scene)[0];obstacles=scene[(distance>.012)&(np.linalg.norm(scene-pose(arm)[:3],axis=1)>.06)]
            registered=np.array(registration['world_from_cad']);cad_world=cad_points@registered[:3,:3].T+registered[:3,3];cad_normals_world=cad_normals@registered[:3,:3].T;datum=(registered@np.r_[component['cad_feature_local'],1])[:3];datum_axis=registered[:3,:3]@np.array(component['cad_feature_axis_local'])
            rgbd_count=len(options)
            if registration_supported:
                cad_options=[]
                for offset in (pad_offset,):
                    if robot_name=='xarm7':
                        # G1 recentres onto its real pad before the mandatory full-stroke support/body checks below.
                        variants=antipodal_grasp_candidates(cad_world,cad_normals_world,table_z,max_candidates=4096,pad_offset_m=offset,clearance_test=lambda candidate,table:True,approach_tilts=(0.,-.45,.45,-.8,.8,-1.05,1.05,-1.2,1.2,-1.3,1.3,-1.4,1.4))
                    else:
                        variants=antipodal_grasp_candidates(cad_world,cad_normals_world,table_z,pad_offset_m=offset,clearance_test=clearance_test)
                    for variant in variants:variant['pad_contact_offset_m']=float(offset)
                    cad_options.extend(variants)
                options=cad_options+options
            from assembly_skills import screen_grasp_geometry
            if robot_name=='xarm7':options,screening=xarm_gripper.screen_candidates(options,cad_world,cad_normals_world,table_z,registration_supported)
            else:options,screening=screen_grasp_geometry(options,cad_world,cad_normals_world,datum,datum_axis,table_z,hand_hulls,registration_supported)
            screening.update(rgbd_generated=rgbd_count,cad_generated=screening['initial_generated']-rgbd_count,registration_supported=bool(registration_supported),support_plane_rejected=0,ik_rejected=0,visible_obstacle_rejected=0)
            current_rotation=Rotation.from_quat(pose(arm)[[4,5,6,3]])
            geometric_options=[]
            for candidate in options:
                candidate['assembly_compatibility']=assembly_grasp_metrics(candidate,datum_axis,component['axial_protrusion_m'])
                if robot_name=='xarm7':
                    release_by_axis=xarm_gripper.mating_release_clearances(candidate,datum,datum_axis)
                    candidate['assembly_compatibility']=assembly_grasp_metrics(candidate,datum_axis,component['axial_protrusion_m'],release_by_axis)
                    candidate['assembly_compatibility'].update(full_opening_release_clearance_by_axis_alignment_m=release_by_axis,release_clearance_scope='Whole G1 opening to 80mm against a public CAD datum plane with 3mm margin; choose axis sign from seating-face geometry, not from the smaller clearance. No fixture GT, route or retention certification.')
                geometric_options.append(candidate)
            options=geometric_options
            options.sort(key=lambda candidate:float(np.linalg.norm(np.array(candidate['position_m'])-pose(arm)[:3]))+.05*float((Rotation.from_quat(np.array(candidate['quaternion_wxyz'])[[1,2,3,0]])*current_rotation.inv()).magnitude())+(candidate.get('contact_quality',{}).get('friction_margin_cost',0.) if robot_name=='xarm7' else candidate['assembly_compatibility']['cost']))
            tested=0
            for candidate in options:
                tested+=1
                p=np.array(candidate['position_m']);axis=np.array(candidate['approach_axis_world']);pre=p-axis*candidate['pregrasp_distance_m'];q=candidate['quaternion_wxyz']
                if min(p[2],pre[2])<table_z+.003 or not clearance_test(candidate,table_z):
                    screening['support_plane_rejected']+=1;continue
                current=pose(arm);safe=max(current[2],pre[2]+.025);waypoints=[np.r_[current[:2],safe,current[3:]],np.r_[pre[:2],safe,current[3:]],np.r_[pre[:2],safe,q],np.r_[pre,q],np.r_[p,q]];seed=joints.copy();solutions=[];reachable=True
                for waypoint in waypoints:
                    solution=kin[arm].solve(transform(waypoint[:3],waypoint[3:]),seed,configuration_clearances=(lambda values:list(arm_support.plane_clearances(values,arm_mount,table_z).values())) if robot_name=='xarm7' else None)
                    if not solution['reachable']:reachable=False;break
                    seed=np.asarray(solution['joints']);solutions.append(solution)
                if not reachable:
                    screening['ik_rejected']+=1;continue
                if robot_name=='xarm7':
                    candidate['arm_table_clearance_m']=min(min(arm_support.plane_clearances(values,arm_mount,table_z).values()) for values in [joints,*[solution['joints'] for solution in solutions]])
                    candidate['arm_table_screening_scope']='Public URDF arm meshes at sampled IK waypoints versus calibrated table plane; not continuous path, tracking, self-collision or hidden-obstacle proof'
                ik_pre,ik_grasp=solutions[-2:];candidate['planned_waypoints_wxyz']=[w.tolist() for w in waypoints];candidate['planned_waypoint_joints']=[solution['joints'] for solution in solutions]
                # Visible, non-target points must clear the swept fingertip centre.
                samples=np.linspace(pre,p,20)
                if len(obstacles) and np.min(cKDTree(obstacles).query(samples[:-4])[0])<.008:
                    screening['visible_obstacle_rejected']+=1;continue
                candidate['ik_pregrasp']=ik_pre;candidate['ik_grasp']=ik_grasp;candidate['feasibility']='joint limits + calibrated robot FK IK + embodiment-specific hand/support-plane clearance + visible-depth swept-centre clearance; not complete hidden-scene collision proof'
                candidate['measured_robot_tcp']=pose(active_arm).tolist()
                candidate['score_components']={'arm_travel':float(np.linalg.norm(np.array(ik_pre['joints'])-joints)),'tcp_distance':.5*float(np.linalg.norm(pre-pose(arm)[:3])),'assembly_access':0. if robot_name=='xarm7' else candidate['assembly_compatibility']['cost'],'contact_friction_margin':candidate.get('contact_quality',{}).get('friction_margin_cost',0.) if robot_name=='xarm7' else 0.}
                candidate['score']=sum(candidate['score_components'].values())
                candidate['score_semantics']='Motion plus friction-margin heuristic, not a retention probability. G1 insertion-access cost is reported separately: an end grasp may need earlier release, but is not inherently an invalid pickup.' if robot_name=='xarm7' else 'Legacy motion plus assembly-access heuristic, not a retention probability'
                candidate['jaw_closing_axis_world']=Rotation.from_quat(np.array(q)[[1,2,3,0]]).apply([0,1,0]).tolist()
                candidate['estimated_grasp_pixel_xy']=project_world_points([candidate['position_m']],K,T)[0].tolist()
                if 'contact_pair_world_m' in candidate:candidate['estimated_contact_pair_pixel_xy']=project_world_points(candidate['contact_pair_world_m'],K,T).tolist()
                candidate['projection_source']='Public RGB-D/CAD candidate projected through current public camera calibration; not semantic identity or object GT'
                candidate['part_component']=component
                candidate['measured_component']={'center_m':points.mean(0).tolist(),'bounds_m':[points.min(0).tolist(),points.max(0).tolist()],'point_count':len(points)}
                valid.append(candidate)
                if len(valid)>=16:break
            valid.sort(key=lambda a:a['score'])
            valid=valid[:16] if robot_name=='xarm7' else valid[:8]
            for candidate in valid:
                key=f'grasp-{len(planned_grasps)}';candidate['candidate_id']=key;planned_grasps[key]=candidate
            component_uv=project_world_points(component_points,K,T);observed_bbox=np.r_[np.nanmin(component_uv,axis=0),np.nanmax(component_uv,axis=0)].tolist()
            screening.update(ik_stage_tested=tested,returned=len(valid),untested_after_candidate_limit=len(options)-tested,component_diagnostics=component_diagnostics)
            return {'status':'candidates' if valid else 'no_feasible_grasp','observed_component_image_bbox_xyxy':observed_bbox,'public_moving_CAD_extent_local_m':np.ptp(cad_points,axis=0).tolist(),'geometry_note':'CAD extents are in local-axis order, not necessarily bore diameter. A measured circular rim may be an inner opening, not the outer boundary. Check RGB and the CAD extents. Rim grasps naturally lie away from the part centre.','candidates':valid,'tested_candidates':tested,'generated_candidates':screening['initial_generated'],'screening':screening,'screening_meaning':{'contact_band_rejected':'Opposed contacts do not fit the pad band/width or hand clearance in mating_plane_grasp.','finger_body_collision_rejected':'Registered public CAD surface lies inside a finger collision hull at closure. This is not an IK failure.','registration_supported':'False means CAD contact and finger-body checks were not applied; unseen geometry remains unknown.'},'feedback':'No feasible candidate does not prove the part ungraspable. Inspect the failing stage and current geometry before changing approach. Manual moves do not inherit candidate geometry screening.','observation_frame':frame_index-1}
        if action=='execute_grasp':
            if c['candidate_id'] not in planned_grasps:raise ValueError('Unknown grasp candidate; generate current candidates first')
            candidate=planned_grasps[c['candidate_id']]
            if np.linalg.norm(pose(active_arm)[:3]-np.array(candidate['measured_robot_tcp'])[:3])>.003 or (Rotation.from_quat(pose(active_arm)[[4,5,6,3]])*Rotation.from_quat(np.array(candidate['measured_robot_tcp'])[[4,5,6,3]]).inv()).magnitude()>.03:raise ValueError('Robot moved since planning; generate fresh grasp candidates')
            p=np.array(candidate['position_m']);q=candidate['quaternion_wxyz'];axis=np.array(candidate['approach_axis_world']);pre=p-axis*candidate['pregrasp_distance_m'];stages=[]
            grip_force_targets.pop(active_arm,None)
            jaw_references[active_arm]=bumpless_open_reference(jaw_references[active_arm],measured_width(active_arm),.08)
            if torque_controller is not None:torque_controller._jaw_reference=bumpless_open_reference(torque_controller._jaw_reference,measured_width(active_arm),.08)
            widths[active_arm]=.08;settle_jaws()
            # Transit above observed geometry before following the selected approach axis.
            safe=max(pose(active_arm)[2],pre[2]+.025)
            for waypoint_index,target in enumerate(candidate.get('planned_waypoints_wxyz',[np.r_[pose(active_arm)[:2],safe,q],np.r_[pre[:2],safe,q],np.r_[pre,q],np.r_[p,q]])):
                preferred={active_arm:candidate['planned_waypoint_joints'][waypoint_index]} if candidate.get('planned_waypoint_joints') else None
                result=move({active_arm:target},preferred);stages.append(result)
                if result['status']!='reached':return {'status':'grasp_approach_incomplete','stages':stages}
            rotation=Rotation.from_quat(np.array(q)[[1,2,3,0]])
            part=candidate['part_component'];registered_pose=np.array(part['cad_registration']['world_from_cad']);public_com=(registered_pose@np.r_[part['uniform_density_com_local_m'],1])[:3]
            force_target=mass_based_grip_force(.045 if task=='peg' else cover_mass,public_com-pose(active_arm)[:3],rotation.apply([0,1,0]),candidate['observed_width_m'],minimum_force_n=4. if robot_name=='xarm7' else 2.) if c.get('grip_force_n') is None else float(c['grip_force_n'])
            if not .3<=force_target<=35.:raise ValueError('Per-finger force target must be 0.3..35N')
            grip_force_targets[active_arm]=force_target
            widths[active_arm]=0;jaw_settling=settle_jaws()
            record_g1_linkage('force_closure_settled')
            closure={'actual_width_m':measured_width(active_arm),'finger_forces_n':touch(active_arm),'jaw_closing_axis_pressure_n':jaw_pressure(active_arm),'target_per_finger_force_n':force_target,'settling':jaw_settling}
            if not jaw_settling['settled']:
                return {'status':'grasp_contact_incomplete','closure':closure,'stages':stages,'retention_verified':False,'note':'Pressure/bilateral contact did not settle; no automatic lift. Reobserve and adjust the grasp or pressure; direct Cartesian tools remain available.'}
            before_grasp=pose(active_arm).copy();grasp_geometry={'candidate':candidate,'grasp_tcp':before_grasp.tolist()}
            component=candidate['part_component'];registered=np.array(component['cad_registration']['world_from_cad']);cad_com=(registered@np.r_[component['uniform_density_com_local_m'],1])[:3]
            payload_estimates[active_arm]={'mass_kg':.045 if task=='peg' else cover_mass,'local_com_proxy_m':Rotation.from_quat(before_grasp[[4,5,6,3]]).inv().apply(cad_com-before_grasp[:3]),'source':'declared public mass and uniform-density closed CAD COM under current RGB-D registration; not simulator pose'}
            relative_rotation=Rotation.from_quat(before_grasp[[4,5,6,3]]).as_matrix().T@registered[:3,:3]
            payload_estimates[active_arm]['local_inertia_per_mass_m2']=relative_rotation@np.array(component['inertia_per_mass_local_m2'])@relative_rotation.T
            grasp_geometry['payload_estimate']={**payload_estimates[active_arm],'local_com_proxy_m':payload_estimates[active_arm]['local_com_proxy_m'].tolist(),'local_inertia_per_mass_m2':payload_estimates[active_arm]['local_inertia_per_mass_m2'].tolist()}
            lift=move({active_arm:np.r_[before_grasp[:3]+[0,0,candidate['lift_distance_m']],q]},verify_retention=True);stages.append(lift)
            record_g1_linkage('after_lift')
            if lift['status']=='grasp_retention_unconfirmed':
                return {'status':'grasp_retention_unconfirmed','closure':closure,'stages':stages,
                    'post_lift_width_m':measured_width(active_arm),'post_lift_finger_forces_n':touch(active_arm),
                    'retention_verified':False,'held_axis_features':[],
                    'note':'Closure pressure converged, but bilateral pressure disappeared during lift; motion held at measured pose. This does not prove insufficient force alone: reconsider contact pair and approach as well as force. Object identity/location is unknown. Inspect current RGB-D and replan; no held feature was generated.'}
            held_features=[]
            if measured_width(active_arm)>.001 and all(v>.02 for v in jaw_pressure(active_arm)):
                component=candidate['part_component'];reference=transform(before_grasp[:3],before_grasp[3:]);now=transform(pose(active_arm)[:3],pose(active_arm)[3:]);delta=now@np.linalg.inv(reference)
                for sign,index in [(-1,0),(1,1)]:
                    initial_point=np.array(component['endpoints_m'][index]);point=(delta@np.r_[initial_point,1])[:3];axis=delta[:3,:3]@np.array(component['axis_world'])*sign
                    key=f'held-feature-{len(feature_records)}';feature_records[key]={'position_m':point.tolist(),'axis_world':axis.tolist(),'measured_robot_tcp':pose(active_arm).tolist(),'rigid_grasp_reference_tcp':before_grasp.tolist(),'reference_position_m':initial_point.tolist(),'reference_axis_world':(np.array(component['axis_world'])*sign).tolist(),'source':'pregrasp observed component transported by measured TCP; rigid-retention assumption, not object GT'}
                    held_features.append({'feature_id':key,**feature_records[key]})
            if held_features:
                registration=component.get('cad_registration',{})
                if registration.get('rms_m',1)<.002 and registration.get('observed_surface_coverage',0)>.6 and registration.get('support_penetration_m',1)<.001:
                    # Reobserve the held part: closure can shift it relative to the TCP.
                    # Only public depth, camera calibration, CAD and measured robot pose.
                    prior=delta@np.array(registration['world_from_cad'])
                    inhand,current_tcp=reobserve_held_cad(prior,active_arm,'lift')
                    if inhand['status']=='visually_refined':
                        registration=inhand['registration'];delta=np.eye(4);before_grasp=current_tcp
                        refresh_payload_estimate(active_arm,component,np.array(registration['world_from_cad']),current_tcp)
                    grasp_geometry['inhand_registration']=inhand
                    model_pose=np.array(registration['world_from_cad']);world_point=(model_pose@np.r_[component['cad_feature_local'],1])[:3];world_axis=model_pose[:3,:3]@np.array(component['cad_feature_axis_local']);point=(delta@np.r_[world_point,1])[:3];axis=delta[:3,:3]@world_axis
                    key=f'held-cad-feature-{len(feature_records)}';feature_records[key]={'position_m':point.tolist(),'axis_world':axis.tolist(),'measured_robot_tcp':pose(active_arm).tolist(),'rigid_grasp_reference_tcp':before_grasp.tolist(),'reference_position_m':world_point.tolist(),'reference_axis_world':world_axis.tolist(),'source':'public CAD feature transformed by observed RGB-D registration and measured robot TCP; rigid-retention assumption','registration_rms_m':registration['rms_m'],'axial_protrusion_m':component['axial_protrusion_m'],'required_release_clearance_m':candidate.get('required_release_clearance_m',0.)}
                    feature_records[key]['cad_pose_at_reference']=model_pose.tolist()
                    held_features.append({'feature_id':key,**feature_records[key]})
            for feature in held_features:
                if feature['feature_id'].startswith('held-cad-feature-'):
                    feature_records[feature['feature_id']]['feature_provenance']=component['feature_provenance']
                    feature['feature_provenance']=component['feature_provenance']
                    feature['datum_note']='A declared CAD datum is not necessarily a physical seating face. Axis sign alone does not select the leading end; choose directed alignment only from geometric evidence.'
                point=np.array(feature['position_m']);camera_point=np.linalg.inv(T)@np.r_[point,1];uv=K@camera_point[:3]
                feature['estimated_pixel_xy']=(uv[:2]/uv[2]).tolist() if uv[2]>0 else None
                feature['projection_note']='Prediction from observed geometry and measured robot motion, not independent visual verification'
            return {'status':'grasp_attempt_complete','held_axis_features':held_features,'closure':closure,'post_lift_width_m':measured_width(active_arm),'post_lift_finger_forces_n':touch(active_arm),'retention_verified':False,'stages':stages,'public_grasp_geometry':grasp_geometry}
        if action=='assemble':
            if c['moving_feature_id'] not in feature_records or c['target_feature_id'] not in feature_records:raise ValueError('Unknown assembly feature; measure features or use a returned held feature first')
            moving=dict(feature_records[c['moving_feature_id']]);target_feature=feature_records[c['target_feature_id']]
            if 'rigid_grasp_reference_tcp' in moving:
                reference=np.array(moving['rigid_grasp_reference_tcp']);delta=transform(pose(active_arm)[:3],pose(active_arm)[3:])@np.linalg.inv(transform(reference[:3],reference[3:]));moving['position_m']=(delta@np.r_[moving['reference_position_m'],1])[:3].tolist();moving['axis_world']=(delta[:3,:3]@np.array(moving['reference_axis_world'])).tolist();moving['measured_robot_tcp']=pose(active_arm).tolist()
            if np.linalg.norm(pose(active_arm)[:3]-np.array(moving['measured_robot_tcp'])[:3])>.003:raise ValueError('Moving feature is stale after robot movement; remeasure')
            offset=float(c.get('seat_offset_m',0));radius=float(c.get('search_radius_m',.001))
            if not 0<=offset<=.1 or not 0<=radius<=.003:raise ValueError('Contact offset 0..0.1m and bounded search radius 0..3mm required')
            axis_alignment=c.get('axis_alignment','nearest');stages=[]
            transport_visual_registration={'status':'not_applicable'}
            require_retention='rigid_grasp_reference_tcp' in moving or min(jaw_pressure(active_arm))>.02
            for transit_pass in range(2):
                current=pose(active_arm).copy()
                release_clearance,pre,seat=assembly_targets(current,moving,target_feature,offset,axis_alignment)
                for target in [np.r_[current[:2],max(current[2],pre[2]+.04),current[3:]],np.r_[pre[:2],max(current[2],pre[2]+.04),pre[3:]],pre]:
                    result=move({active_arm:target},verify_retention=require_retention);stages.append(result)
                    if result['status']=='grasp_retention_unconfirmed':
                        expired=result.get('invalidated_feature_ids',[])+[key for key,value in feature_records.items() if 'rigid_grasp_reference_tcp' in value]
                        for key in expired:feature_records.pop(key,None)
                        grasp_geometry=None
                        return {'status':'grasp_retention_unconfirmed','stages':stages,'invalidated_feature_ids':expired,
                                'retention_verified':False,'note':'Bilateral pressure disappeared during assembly transport. Motion held; rigid held-feature estimates invalidated. Reobserve the part and fixture; this is not evidence of jam or successful seating.'}
                    if result['status']!='reached':return {'status':'alignment_incomplete','stages':stages,'transport_visual_registration':transport_visual_registration}
                if transit_pass:
                    transport_visual_registration['corrective_pregrasp_reached']=True
                    break
                if robot_name!='xarm7' or 'cad_pose_at_reference' not in moving or not grasp_geometry:break
                reference=np.array(moving['rigid_grasp_reference_tcp']);current_tcp=pose(active_arm).copy()
                prior=transform(current_tcp[:3],current_tcp[3:])@np.linalg.inv(transform(reference[:3],reference[3:]))@np.array(moving['cad_pose_at_reference'])
                transport_visual_registration,current_tcp=reobserve_held_cad(prior,active_arm,'transport')
                if transport_visual_registration['status']!='visually_refined':break
                component=grasp_geometry['candidate']['part_component']
                refined=np.array(transport_visual_registration['registration']['world_from_cad'])
                point=(refined@np.r_[component['cad_feature_local'],1])[:3];axis=refined[:3,:3]@np.array(component['cad_feature_axis_local'])
                moving.update(position_m=point.tolist(),axis_world=axis.tolist(),measured_robot_tcp=current_tcp.tolist(),rigid_grasp_reference_tcp=current_tcp.tolist(),reference_position_m=point.tolist(),reference_axis_world=axis.tolist(),cad_pose_at_reference=refined.tolist(),registration_rms_m=transport_visual_registration['registration']['rms_m'],source_observation_id=transport_visual_registration['observation_id'])
                expired=[key for key,value in feature_records.items() if key!=c['moving_feature_id'] and 'rigid_grasp_reference_tcp' in value]
                for key in expired:feature_records.pop(key,None)
                feature_records[c['moving_feature_id']]=dict(moving)
                transport_visual_registration['invalidated_other_held_feature_ids']=expired
                refresh_payload_estimate(active_arm,component,refined,current_tcp)
            alignment_stable=0
            for _ in range(400):
                advance();measured=pose(active_arm);position_error=np.linalg.norm(measured[:3]-pre[:3]);angular_error=(Rotation.from_quat(measured[[4,5,6,3]])*Rotation.from_quat(pre[[4,5,6,3]]).inv()).magnitude()
                alignment_stable=alignment_stable+1 if position_error<.0003 and angular_error<.003 else 0
                if alignment_stable>=8:break
            if alignment_stable<8:return {'status':'fine_alignment_not_converged','position_error_m':float(position_error),'angular_error_rad':float(angular_error),'stages':stages}
            baseline=sum(touch(active_arm));wrench_baseline=wrench_reader._wrench[:3].copy() if wrench_reader is not None else None
            if torque_controller is not None:torque_controller.force_reference=torque_controller.last_command_wrench[:3].copy();torque_controller.force_limit=3.
            axis=np.array(target_feature['axis_world']);seat_command=seat.copy();travel=float(np.linalg.norm(seat[:3]-pre[:3]));contact=False
            for fraction in np.linspace(0,1,max(2,round(travel/.0004))):
                target=pre.copy();target[:3]=pre[:3]*(1-fraction)+seat[:3]*fraction;targets[active_arm]=target
                for _ in range(60):
                    advance()
                    if sum(touch(active_arm))>baseline+3 or (wrench_reader is not None and np.linalg.norm(wrench_reader._wrench[:3]-wrench_baseline)>3.):contact=True;break
                    if np.linalg.norm(pose(active_arm)[:3]-target[:3])<.0003:break
                if contact or np.linalg.norm(pose(active_arm)[:3]-target[:3])>.004:
                    contact=True;break
            # Spiral search maintains a small axial reference bias. Once measured
            # axial progress is observed, retain that lateral correction and resume.
            lateral_correction=np.zeros(3);entry_progress=0.;search_samples=[]
            if contact and radius>0:
                search_origin=pose(active_arm).copy();u=np.cross(axis,[1.,0,0])
                if np.linalg.norm(u)<.1:u=np.cross(axis,[0.,1,0])
                u/=np.linalg.norm(u);v=np.cross(axis,u);found=False
                for theta in np.linspace(0,6*np.pi,48):
                    correction=radius*(theta/(6*np.pi))*(u*np.cos(theta)+v*np.sin(theta));candidate=search_origin.copy();candidate[:3]+=correction-axis*min(.001,max(0,float((search_origin[:3]-seat[:3])@axis)))
                    cartesian_references[active_arm]=CartesianReference(pose(active_arm),reference_speed_scale)
                    targets[active_arm]=candidate
                    excessive=False
                    for _ in range(20):
                        advance();entry_progress=float((search_origin[:3]-pose(active_arm)[:3])@axis)
                        if sum(touch(active_arm))>baseline+6 or (wrench_reader is not None and np.linalg.norm(wrench_reader._wrench[:3]-wrench_baseline)>3.):excessive=True;break
                        if entry_progress>.0007 and np.linalg.norm((pose(active_arm)[:3]-candidate[:3])-axis*((pose(active_arm)[:3]-candidate[:3])@axis))<.001:
                            found=True;lateral_correction=correction.copy();break
                    search_samples.append({'theta_rad':float(theta),'correction_m':correction.tolist(),'measured_axial_progress_m':entry_progress,'extra_total_finger_force_n':sum(touch(active_arm))-baseline})
                    if found:break
                    if excessive:
                        targets[active_arm]=pose(active_arm).copy();targets[active_arm][:3]+=axis*.0003
                        for _ in range(8):advance()
                if found:
                    seat[:3]+=lateral_correction;resumption=pose(active_arm).copy();distance=float(np.linalg.norm(seat[:3]-resumption[:3]))
                    for fraction in np.linspace(0,1,max(2,round(distance/.0003))):
                        target=seat.copy();target[:3]=resumption[:3]*(1-fraction)+seat[:3]*fraction;targets[active_arm]=target
                        for _ in range(60):
                            advance()
                            if sum(touch(active_arm))>baseline+6:break
                            if np.linalg.norm(pose(active_arm)[:3]-target[:3])<.0003:break
                        if sum(touch(active_arm))>baseline+6 or np.linalg.norm(pose(active_arm)[:3]-target[:3])>.004:break
            log('assembly-search.jsonl',{'source':'measured TCP and unfiltered total contact only','samples':search_samples,'retained_lateral_correction_m':lateral_correction.tolist(),'axial_progress_m':entry_progress})
            targets[active_arm]=pose(active_arm).copy()
            for _ in range(20):advance()
            release_ready=bool(release_clearance>0 and np.linalg.norm(pose(active_arm)[:3]-seat[:3])<.001)
            return {'status':'hand_clearance_boundary_reached' if release_ready else 'contact_search_complete' if contact else 'placement_depth_attempted','closed_hand_descent_blocked_by_predicted_geometry':release_ready,'release_pose_verified':False,'next_step':'The hand has reached its predicted opening-clearance boundary, NOT a verified insertion or supported release pose. Inspect the current part-to-target gap and support before choosing release. A large remaining gap can require regrasping; do not assume free-fall will seat the part. Do not descend further with this closed grasp into the predicted hand interference.' if release_ready else 'Inspect contact/alignment before release.','actual_tcp':pose(active_arm).tolist(),'seat_tcp_command':seat.tolist(),'stages':stages,'transport_visual_registration':transport_visual_registration,'release_automatic':False,'retained_clearance_before_release_m':release_clearance,'release_note':'This is a hand-clearance boundary, not a recommended release pose. The tool has not verified support or guided entry; choose subsequent actions from current observations.','search_axial_progress_m':entry_progress,'search_lateral_correction_m':lateral_correction.tolist(),'note':'Only independent native success proves seating; inspect current images and decide release.'}
        if action=='grasp_at':
            arm=c.get('arm','left')
            if arm not in arms:raise ValueError('Unknown arm')
            point=back_project(c['pixel_xy']);q=c['quaternion_wxyz'];stages=[];widths[arm]=.08
            for xyz in [point+[0,0,.08],point+[0,0,.01]]:
                r=move({arm:np.r_[xyz,q]});stages.append(r)
                if r['status']!='reached':return {'status':'grasp_approach_incomplete','stages':stages}
            widths[arm]=0
            for _ in range(20):advance()
            stages.append(move({arm:np.r_[pose(arm)[:3]+[0,0,.06],q]}))
            return {'status':'grasp_attempt_complete','actual_tcp':pose(arm).tolist(),'unfiltered_touch_n':touch(arm),'retention_verified':False,'stages':stages}
        raise ValueError('Unsupported action')
    error=None
    try:
        initialization_checkpoint('before_first_control_step')
        for _ in range(12):
            if snapshot()['terminal']:break
            advance()
        from task_profiles import public_skill_specification
        profile = {
            'robot': robot_name,
            'arms': list(arms),
            'task': task,
            'table_top_world_z_m': table_z,
            'frame': 'world metres, wxyz quaternions',
            'tcp_offsets': {a: info[a]['offset'].tolist() for a in arms},
            'tool_axis_convention': {
                'local_z': 'from palm towards finger tips; identity quaternion points fingers UP in world, not down',
                'local_y': 'jaw closing axis',
                'downward_quaternion_examples_wxyz': [[0, 1, 0, 0], [0, 0, 1, 0]],
                'assembly_object_axis': 'distinct from tool approach axis; use observed/tracked component axes',
            },
            'public_CAD_bolt_dimensions_m': {'shaft_length': .022, 'head_axial_thickness': .00415},
            'jaw_width_range_m': [0, .08],
            'jaw_width_semantics': 'width_m is the gap between inner finger faces, not the outside envelope of both fingers. A bore must contain both complete finger cross-sections plus clearance; gap < bore diameter is insufficient.' if robot_name=='panda' else 'width_m commands twice the driven prismatic joint displacement; measured_width is the same stroke proxy, not a calibrated physical pad gap. Observed R1 gap tracking and exact finger-envelope fit require checking.',
            'automatic_grasp_modes': 'plan_grasp generates compressive closing grasps, including local wall pinches. execute_grasp closes and lifts; it does not implement internal expansion. An OD larger than max opening rules out enclosing that OD, not all possible local wall grasps.' if robot_name=='panda' else 'Automatic R1 grasp candidate contact/pad adaptation is pending. plan_grasp provides public component/CAD fit diagnostics only; direct physical motion remains available.',
            'public_finger_geometry': json.loads((settings.ROOT/'public-robot-pad-inspection.json').read_text()) if robot_name=='panda' else {arm:model.public_summary() for arm,model in r1_geometry.items()},
            'finger_geometry_frame': 'Panda inspection bounds and pad Z are in the hand frame at zero jaw displacement, metres. Translate each finger along its joint axis for the current opening; transform through hand pose, not TCP pose. Full-body bounds are conservative, not a bore-fit certificate.' if robot_name=='panda' else 'Per-arm summary bounds are TCP-frame metres at each finger joint q=0. Translate each finger along its own reported opening axis using actual joint coordinates. Bounds are not pad patches or a bore-fit certificate.',
            'tcp_contact_semantics': 'TCP offset is from hand to TCP, not fingertip length. Panda pad hand-frame Z is about 0.09465..0.11215m; TCP hand-frame Z is 0.1034m. Compute pad/contact positions with the actual orientation.' if robot_name=='panda' else 'R1 wrist-to-TCP offset is [0.0079,0,0.09089]m. Static public finger meshes extend distally to wrist Z about 0.078355m: the TCP lies about 12.5mm beyond the tips along local +Z. Transform robot geometry by p_world = p_TCP_world + R_world_TCP * p_mesh_TCP. In a downward +Z approach, the distal tips are ABOVE the TCP by about 12.5mm, not below it. inspect_pose returns whole mesh world bounds for the specified orientation, not calibrated pad patches. Exact pad patches and physical gap remain uncalibrated; do not use Panda pad dimensions.',
            'hand_actuator_effort_limit_n': 70. if robot_name == 'panda' else None,
            'grasp_pressure_limit_per_finger_n': 35.,
            'jaw_pressure_source': 'measured force projected onto the jaw closing axis' if robot_name=='panda' else 'each measured finger force projected onto its own public-joint opening axis in world coordinates; unfiltered all-object contact, not retention or object identity',
            'payload_feedforward': 'declared mass plus RGB-D-registered uniform-density public CAD COM/inertia; removed on actual bilateral contact loss',
            'load_assumption': 'Moving-part mass is a declared-density public-CAD estimate, not measured; retention requires current observations',
            'moving_part_mass_estimate_status': os.environ.get('ASSEMBLY_COVER_MASS_STATUS', 'unspecified'),
            'touch': 'unfiltered all-object forces',
            'no_object_attachment': True,
            'physics_backend': physics_backend_info,
            'dynamic_mesh_sdf_resolution': int(os.environ.get('ASSEMBLY_SDF_RESOLUTION', '256')),
            'contact_solver_position_iterations': cfg.spawn.articulation_props.solver_position_iteration_count,
            'contact_solver_velocity_iterations': cfg.spawn.articulation_props.solver_velocity_iteration_count,
            'controller_mode': controller_mode,
            'measured_tcp_angular_stop_radps': angular_stop_limit,
            'measured_tcp_linear_stop_mps': linear_stop_limit,
            'controller': 'installed Isaac Lab OperationalSpaceController with inertial decoupling and robot/payload gravity compensation' if controller_mode == 'isaac_osc' else 'official Isaac Lab Panda high-PD joint drive + persistent Cartesian reference + local Jacobian DLS and null-space home posture' if controller_mode == 'reference_ik_high_pd' else 'persistent SE3 reference + Factory task-space impedance with implicit robot-mass PD discretization and real hand FT' if torque_controller is not None else 'persistent SE3 Cartesian reference, continuous robot-model IK, bounded joint tracking, Jacobian speed limits, gravity and bounded integral load compensation',
            **public_motion_reference_limits(controller_mode,reference_speed_scale),
            'robot_model_validation': kin_diagnostics,
            'generic_grasp': '6D public RGB-D candidates with robot-model IK screening; execute_grasp does not certify retention',
            'assembly_skill': public_skill_specification(),
            'task_mass_kg': .045 if task == 'peg' else cover_mass,
            'cover_material_model': os.environ.get('ASSEMBLY_COVER_MATERIAL', 'legacy-steel'),
            'evaluator_adapter': 'independent task evaluator',
            'native_success_only': True,
        }
        if robot_name=='xarm7':
            profile.update(jaw_width_range_m=[xarm_gripper.opening(.85),.08],
                automatic_grip_force_minimum_per_finger_n=4.,
                automatic_grip_force_minimum_status='Simulation empirical setting from 2N slip and 4N retained-lift trials; not calibrated real hardware. Explicit requests retain the 0.3..35N range.',
                jaw_width_semantics='Physical distal finger gap estimated from official G1 linkage and measured drive angle; minimum about 1.64mm. Smaller requests saturate at mechanical model limit, not zero gap.',
                automatic_grasp_modes='plan_grasp uses shared public RGB-D/CAD candidates, official G1 distal contact band, complete G1 collision mesh hull screening and robot-model IK. execute_grasp performs physical closing and lifting with measured jaw pressure. Internal expansion is not implemented. No grasp or retention is assumed.',
                tcp_contact_semantics='Official G1 TCP is 172mm along gripper-base +Z. Local +Y is closing direction; fingers remain parallel through 1:1 mimic linkage. Use inspect_pose for full gripper geometry clearance.',
                finger_geometry_frame='Official G1 collision mesh/linkage model; base to TCP [0,0,.172] metres. No Panda or R1 pad geometry applies.',
                jaw_pressure_source='Finger contact force projected onto measured G1 closing axis; all objects unfiltered. Not calibrated hardware tactile pressure or proof of retention.',
                sensor_configuration='Fixed RGB-D pinhole simulation for assumed RealSense deployment; not a calibrated physical camera. Wrench from simulated incoming gripper-base joint, moment shifted to TCP; not compensated hardware sensor.',
                hand_actuator_effort_limit_n=None)
        write(R/'private/robot-profile.json',profile);latest,depth,K,T=publish(0,{'status':'reset_ready'})
        for i in range(500):
            if snapshot()['terminal']:return
            path=R/'ipc'/f'command_{i:04d}.json'
            while not path.exists():
                if snapshot()['terminal']:return
                time.sleep(.05)
            c=json.loads(path.read_text());started=time.monotonic();step_before_action=step_index
            try:result=dispatch(c)
            except ValueError as e:
                recovered_targets,recovered_references=reset_motion_references({a:pose(a) for a in arms},reference_speed_scale)
                targets.update(recovered_targets);cartesian_references.update(recovered_references)
                joint_goal_poses.clear();joint_goals.clear();joint_bias.clear();motion_metrics.clear();commanded_joints[:]=robot.data.joint_pos
                result={'status':'execution_incomplete' if step_index>step_before_action else 'invalid_action','feedback':str(e),'executed_physics_steps':step_index-step_before_action,'actual_poses':{a:pose(a).tolist() for a in arms},'actual_widths_m':{a:measured_width(a) for a in arms},'recovery':'Motion references reset to measured poses; inspect partial movement before retrying.'}
            except RuntimeError:
                if not snapshot()['terminal']:raise
                result={'status':'interrupted','actual_poses':{a:pose(a).tolist() for a in arms},'actual_widths_m':{a:measured_width(a) for a in arms}}
            if torque_controller is not None:torque_controller.force_limit=None
            log('actions.jsonl',{'index':i,'command':c,'result':result,'elapsed_s':time.monotonic()-started})
            if snapshot()['terminal']:
                latest,depth,K,T=publish(i+1,result);write(R/'episode/public/final-state.json',latest);break
            latest,depth,K,T=publish(i+1,result)
    except BaseException as e:error=repr(e);write(R/'private/backend-failed.json',{'error':error,'traceback':traceback.format_exc()});terminate('backend_error',error=error)
    finally:
        if peg_process is not None:
            terminal=snapshot()['terminal'] or {}
            write(R/'private/peg-process-metrics.json',peg_process.result(termination_reason=terminal.get('source')))
            write(R/'private/peg-process-events.json',peg_process.events)
            peg_process_stream.close()
        video_error=recorder.close();write(R/'episode/backend-summary.json',{'native_success':native_success,'error':error,'physics_steps':step_index,'video_frames':recorder.frame_count,'video_error':video_error,'terminal':snapshot()['terminal'],'robot':robot_name,'task':task,'adapter_is_new':True,'privileged_setup_diagnostic':diagnostic});write(R/'private/evaluator-trace.json',trace)
        if profiler is not None:
            profiler.disable();profiler.dump_stats(str(R/'private/backend.pstats'))
        # All results are saved: skip only the timeline render-until-play callback during final teardown.
        sim._disable_app_control_on_stop_handle = True
        app.close()
if __name__=='__main__':
    try:main()
    except BaseException as e:write(R/'private/backend-failed.json',{'error':repr(e),'traceback':traceback.format_exc()});terminate('backend_startup_error',error=repr(e));raise
