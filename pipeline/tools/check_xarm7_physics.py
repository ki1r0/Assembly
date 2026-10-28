"""Measured xArm7/G1 articulation sweep; not a model assembly rollout."""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--usd',required=True)
parser.add_argument('--output',required=True)
parser.add_argument('--mode',choices=['gripper','arm'],default='gripper')
parser.add_argument('--diagnostic-friction-scale',type=float,default=1.)
parser.add_argument('--effort-friction',action='store_true')
parser.add_argument('--motion-record',type=Path,help='Private emergency telemetry for a fixture-free constant-target diagnostic')
parser.add_argument('--velocity-iterations',type=int,help='Diagnostic override; otherwise use embodiment configuration')
parser.add_argument('--physics-dt',type=float,default=1/120)
parser.add_argument('--gpu-dynamics',action='store_true',help='Match GPU PhysX/broadphase for fixture-free control diagnostics')
parser.add_argument('--diagnostic-arm-stiffness',type=float,default=400.)
parser.add_argument('--diagnostic-arm-damping',type=float,default=40.)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app
try:
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation,ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sensors import ContactSensor,ContactSensorCfg
    import math
    if not math.isfinite(args.physics_dt) or not 0<args.physics_dt<=1/120:
        raise ValueError('Diagnostic physics dt must be positive and at most 1/120 s')
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args.physics_dt,device='cpu'))
    if args.gpu_dynamics:
        import carb
        carb.settings.get_settings().set_int('/physics/cudaDevice',0)
        sim.get_physics_context().set_broadphase_type('GPU')
        sim.get_physics_context().enable_gpu_dynamics(True)
    from xarm_robot_cfg import xarm7_config,restore_g1_mimic_joints,apply_official_adjacent_collision_filters
    cfg=xarm7_config(args.usd,base_position=(0.,0.,0.))
    if not math.isfinite(args.diagnostic_arm_stiffness) or not 400<=args.diagnostic_arm_stiffness<=1000 or not math.isfinite(args.diagnostic_arm_damping) or not 40<=args.diagnostic_arm_damping<=100:
        raise ValueError('Diagnostic drive stiffness must be 400..1000 and damping 40..100')
    cfg.actuators['arm'].stiffness=args.diagnostic_arm_stiffness
    cfg.actuators['arm'].damping=args.diagnostic_arm_damping
    if args.velocity_iterations is not None:
        if not 1<=args.velocity_iterations<=255:raise ValueError('Velocity iterations must be in [1,255]')
        cfg.spawn.articulation_props.solver_velocity_iteration_count=args.velocity_iterations
    robot=Articulation(cfg)
    if not 0<=args.diagnostic_friction_scale<=1:raise ValueError('Diagnostic friction scale must be in [0,1]')
    friction_changes=[]
    for prim in sim.stage.Traverse():
        if prim.GetName() in {f'joint{i}' for i in range(1,8)}:
            attr=prim.GetAttribute('physxJoint:jointFriction')
            if attr and attr.Get() is not None:
                original=float(attr.Get());attr.Set(original*args.diagnostic_friction_scale)
                friction_changes.append({'joint':prim.GetName(),'original':original,'diagnostic':original*args.diagnostic_friction_scale})
    restored=restore_g1_mimic_joints(sim.stage)
    effort_friction=None
    if args.effort_friction:
        from xarm_robot_cfg import apply_urdf_effort_friction
        effort_friction=apply_urdf_effort_friction(sim.stage,Path(__file__).resolve().parents[1]/'assets/xarm7/xarm7_g1_simulation_assumption.urdf')
    filtered=apply_official_adjacent_collision_filters(sim.stage,Path(__file__).resolve().parents[1]/'third_party/xarm_ros2/xarm_moveit_config/srdf/_xarm7_macro.srdf.xacro')
    contact_links=['left_finger','right_finger','left_inner_knuckle','right_inner_knuckle','left_outer_knuckle','right_outer_knuckle','xarm_gripper_base_link']
    if args.mode=='arm':contact_links+=[f'link{i}' for i in range(1,8)]
    pattern='/World/Robot/(left_.*|right_.*|xarm_gripper_base_link'+('|link[1-7]' if args.mode=='arm' else '')+')'
    contacts=ContactSensor(ContactSensorCfg(prim_path=pattern,
        filter_prim_paths_expr=['/World/Robot/'+name for name in contact_links],update_period=0.,
        track_contact_points=True,max_contact_data_count_per_prim=1024))
    sim.reset()
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),robot.data.default_joint_vel.clone())
    targets=robot.data.default_joint_pos.clone()
    replay=None
    if args.motion_record:
        if args.mode!='arm':raise ValueError('Motion record requires arm diagnostic mode')
        replay=json.loads(args.motion_record.read_text().splitlines()[-1])
        initial=torch.tensor(replay['joint_positions'],dtype=targets.dtype)[None]
        targets=torch.tensor(replay['joint_position_targets'],dtype=targets.dtype)[None]
        if initial.shape!=robot.data.default_joint_pos.shape or targets.shape!=initial.shape:
            raise ValueError('Recorded articulation joint count differs from the model')
        robot.write_joint_state_to_sim(initial,torch.zeros_like(initial))
    drive,_=robot.find_joints('drive_joint')
    if len(drive)!=1:raise RuntimeError('Expected one G1 drive joint')
    records=[]
    controlled=drive if args.mode=='gripper' else robot.find_joints('joint2')[0]
    sequence=[0.,.2,.4,.6,.8,0.] if args.mode=='gripper' else [-.5,-.45,-.55,-.5]
    if replay:sequence=[None]
    for target in sequence:
        if target is not None:targets[:,controlled]=target
        transient=[]
        for tick in range(round(2/args.physics_dt)):
            robot.set_joint_position_target(targets)
            robot.set_joint_effort_target(robot.root_physx_view.get_gravity_compensation_forces())
            robot.write_data_to_sim();sim.step(render=False);robot.update(args.physics_dt);contacts.update(args.physics_dt)
            if replay:
                transient.append({'tick':tick,'joint_positions_rad':robot.data.joint_pos[0].tolist(),
                    'joint_velocities_radps':robot.data.joint_vel[0].tolist(),
                    'applied_torques_nm':robot.data.applied_torque[0].tolist(),
                    'max_robot_contact_force_n':float(torch.linalg.vector_norm(contacts.data.force_matrix_w,dim=-1).max()),
                    'max_body_angular_speed_radps':float(torch.linalg.vector_norm(robot.data.body_ang_vel_w,dim=-1).max())})
        records.append({'drive_target_rad':target,
            'fixture_free_constant_target_transient':transient if replay else None,
            'controlled_joint':'drive_joint' if args.mode=='gripper' else 'joint2',
            'joint_targets_rad':targets[0].tolist(),
            'joint_positions_rad':robot.data.joint_pos[0].tolist(),
            'joint_velocities_radps':robot.data.joint_vel[0].tolist(),
            'applied_joint_torques_nm':robot.data.applied_torque[0].tolist(),
            'contact_forces_world_n':contacts.data.force_matrix_w.tolist(),
            'body_poses_world_wxyz':{name:robot.data.body_state_w[0,index,:7].tolist()
                for index,name in enumerate(robot.body_names) if name in ('left_finger','right_finger','link7','link_tcp','xarm_gripper_base_link')}})
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'scope':'Physical articulation diagnostic, not assembly success',
        'physics_dt_s':args.physics_dt,
        'gpu_dynamics_enabled':sim.get_physics_context().is_gpu_dynamics_enabled(),
        'broadphase':sim.get_physics_context().get_broadphase_type(),
        'solver_velocity_iterations':cfg.spawn.articulation_props.solver_velocity_iteration_count,
        'motion_record':str(args.motion_record) if args.motion_record else None,
        'replay_scope':'Recorded posture and fixed drive targets; zero initial velocity, no fixtures, no Cartesian controller. Not a complete episode replay.' if replay else None,
        'mode':args.mode,'effort_friction':effort_friction,'diagnostic_friction_changes':friction_changes,'joint_effort_limits_nm':robot.data.joint_effort_limits[0].tolist(),
        'joint_stiffness':robot.data.joint_stiffness[0].tolist(),'joint_damping':robot.data.joint_damping[0].tolist(),
        'contact_sensor_bodies':contacts.body_names,'contact_filter_bodies':contact_links,
        'official_adjacent_collision_filters':filtered,'restored_mimic_joints':restored,'joint_names':robot.joint_names,'body_names':robot.body_names,'samples':records},indent=2))
    print('Measured articulation sweep saved:',output,flush=True)
except BaseException:
    import traceback
    failure=Path(args.output).with_suffix('.failure.txt')
    failure.parent.mkdir(parents=True,exist_ok=True)
    failure.write_text(traceback.format_exc())
    traceback.print_exc()
    raise
finally:
    # Final teardown must not enter Isaac Lab's render-until-play timeline callback.
    if 'sim' in globals():sim._disable_app_control_on_stop_handle = True
    app.close()
