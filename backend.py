"""Isaac Factory task service, continuous Cartesian tools and measured timing."""
import os,sys,time,json,traceback,dataclasses,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parent/'isaac_runtime';sys.path.insert(0,str(ROOT))
from common import R,write,log,snapshot,terminate,execution_observation
from m1_reach.control import begin_step,complete_step,start_decision

def main():
    import argparse
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');AppLauncher.add_app_launcher_args(p);args=p.parse_args()
    args.device='cpu';args.headless=True;args.enable_cameras=True;args.env_config=str(Path(__file__).resolve().parent/'config/bolt_insertion.yaml');args.output_dir=str(R/'episode/backend');args.observe_only=False;args.resume_state=None;args.seed=0
    scenario_name=os.environ.get('ASSEMBLY_CASE','nominal-aligned');scenario=json.loads((Path(__file__).resolve().parent/'config/scenarios.json').read_text())[scenario_name]
    delta=scenario['assembly_offset_m']
    calibration=json.loads((Path(__file__).resolve().parent/'config/fixture_calibration.json').read_text())
    # Offline fixture calibration stays behind the private task boundary.
    args.bench_task_config={'private_target_mapping':{'geometry':{
        'selected_hole_xy_m':[calibration['selected_hole_xy_m'][i]+delta[i] for i in range(2)],
        **{key:calibration[key]+delta[2] for key in ['cap_support_plane_z_m','casing_entry_plane_z_m','casing_interior_floor_z_m']},
        'assembly_frame_pos_m':[.50+delta[0],-.15+delta[1],.735896652+delta[2]]}}}

    from runtime.assembly_harness import factory_runtime as factory
    from runtime.assembly_harness.executor import _sensors
    from runtime.assembly_harness.agent import _capability_summary
    from runtime.bolt_harness.recording import EpisodeRGBVideoRecorder
    from visual_servo import CartesianTools
    import numpy as np
    timings=collections.defaultdict(lambda:{'calls':0,'seconds':0.,'max_s':0.})
    def timed(obj,name):
        original=getattr(obj,name)
        def wrapped(*a,**kw):
            t=time.perf_counter()
            try:return original(*a,**kw)
            finally:
                elapsed=time.perf_counter()-t;d=timings[name];d['calls']+=1;d['seconds']+=elapsed;d['max_s']=max(d['max_s'],elapsed)
        setattr(obj,name,wrapped)
    timed(factory,'render_robot_depth')
    from wrist_camera import install_wrist_camera,attach_wrist_capture
    original_scene_config=factory._apply_bench_task_scene_config
    def scene_config(cfg,task):
        original_scene_config(cfg,task)
        import numpy as np
        offset=np.asarray(scenario['assembly_offset_m'])
        cfg.assembly_frame_pos_m=tuple(float(x) for x in np.asarray(cfg.assembly_frame_pos_m)+offset)
        cfg.cover_pos_m=tuple(float(x) for x in np.asarray(cfg.cover_pos_m)+offset+calibration['cover_translation_correction_m'])
        cfg.bolt_reset_pos_m=tuple(float(x) for x in np.asarray(cfg.bolt_reset_pos_m)+scenario['bolt_offset_m'])
        if scenario['bolt_lying']:
            cfg.bolt_reset_pos_m=(*cfg.bolt_reset_pos_m[:2],cfg.table_top_z_m+.0101)
            cfg.bolt_reset_quat_wxyz=(1.,0.,0.,0.)
        cfg.bolt.init_state.pos=cfg.bolt_reset_pos_m
        cfg.bolt.init_state.rot=cfg.bolt_reset_quat_wxyz
        install_wrist_camera()
    factory._apply_bench_task_scene_config=scene_config
    runtime=factory.FactoryAssemblyRuntime.from_args(args);tools=CartesianTools(runtime)
    wrist,wrist_unsubscribe=attach_wrist_capture(runtime,R)
    # Direct simulation mode: observation holds do not impose grasp/contact authorization.
    runtime._stationary_hold_step=tools.hold
    def direct_observe():
        import math
        # One fresh camera acquisition, rather than eight guarded geometric holds.
        camera_hold_target=tools.pose()
        for _ in range(max(1,math.ceil(runtime.camera_period_s/runtime.limits.control_dt_s)+1)):
            tools.control_step(camera_hold_target)
        return runtime.observe_public(include_feature_overlay=True)
    runtime.observe_for_model=direct_observe
    for name in ('observe_public','_camera_features_for_frame','_obstacle_map'):timed(runtime,name)
    for name in ('solve_ik','check_robot_sweep_collision'):timed(runtime.driver,name)
    count=0;active=None;original_step=runtime.env.step
    def step(*a,**kw):
        nonlocal count
        t=time.monotonic();decision=snapshot()['decision_started_monotonic'] is not None
        if decision:begin_step(count)
        out=original_step(*a,**kw);count+=1
        runtime._capture_frame()
        if decision:complete_step(count,t)
        action=a[0] if a else kw.get('action');arr=action.detach().cpu().tolist() if hasattr(action,'detach') else str(action)
        log('physics.jsonl',{'step':count,'phase':'decision' if decision else 'startup','command':active,'action':arr,'step_elapsed_s':time.monotonic()-t})
        if decision and getattr(runtime._last_private_status,'task_success',False):terminate('native_success',step=count)
        return out
    runtime.env.step=step
    recorder=EpisodeRGBVideoRecorder(R/'episode/episode.mp4',fps=12);unsubscribe=runtime.subscribe_rgb_frames(lambda rgb,t:recorder.append(rgb,t))
    observation=execution_observation(runtime.observe_for_model());s=_sensors(observation)
    if s.get('rgbd_obstacle_map',{}).get('robot_self_filtered') is not True:raise RuntimeError('Initial public map is not self-filtered')
    write(R/'private/start-state.json',{'scenario':scenario_name,'scenario_setup':scenario,'resume_state':None,'grasped':s.get('grasped'),'tcp_pose':s['tcp_pose'],'gripper_width_m':s['gripper_width_m']})
    if s.get('grasped') is not False:raise RuntimeError('Expected unheld reset')
    profile={'touch_source':'unfiltered all-object contact forces; no Bolt identity','robot':'Factory fixed-base Panda 7DoF','joint_limits':runtime.driver.joint_limits,'execution_mode':'direct_simulation_cartesian_servo','execution_rules':{'geometric_grasp_authorization':False,'rgbd_sweep_rejection':False,'release_authorization':False,'contact_force_rejection':False,'native_motion_emergency_stop':True,'per_move_simulation_budget_s':20,'xy_convergence_tolerance_m':0.00012,'z_transport_tolerance_m':0.0007,'per_step_action_clipping':True},'public_part_dimensions':{'bolt_nominal_shaft_length_m':0.044,'bolt_nominal_radial_clearance_m':0.000337928269,'bolt_nominal_shaft_diameter_m':0.011767996864},'gripper_force_limit_n':runtime.limits.max_gripper_force_n,'pose_frame':'world, metres, quaternion wxyz','robot_base_world_z_m':float(runtime.env.cfg.table_top_z_m),'table_top_world_z_m':float(runtime.env.cfg.table_top_z_m),'camera_resolution':[int(os.environ.get('ASSEMBLY_CAMERA_WIDTH',960)),int(os.environ.get('ASSEMBLY_CAMERA_HEIGHT',720))],'fast_components_enabled':os.environ.get('ASSEMBLY_FAST_COMPONENTS')=='1','camera_pose_unchanged':True,'action_policy':'HRC visual geometry and Cartesian servo','wrist_camera':'real panda_hand-mounted 224x224 RGB; external main camera unchanged'}
    write(R/'private/robot-profile.json',profile)
    # Compare the SAME live robot frame and meshes, not an unrelated image benchmark.
    from runtime.assembly_harness import robot_geometry as fast,reference_robot_geometry as reference
    reference.RobotLinkMesh=fast.RobotLinkMesh
    params={'body_names':runtime.robot_body_names,'near_m':runtime.robot_depth_near_m,'far_m':runtime.robot_depth_far_m}
    frame=runtime._last_frame;transforms=runtime._robot_model_pose_snapshot['link_transforms_world']
    t=time.perf_counter();a=reference.render_robot_depth(runtime.robot_link_meshes,transforms,frame,**params);slow_s=time.perf_counter()-t
    t=time.perf_counter();b=fast.render_robot_depth(runtime.robot_link_meshes,transforms,frame,**params);fast_s=time.perf_counter()-t
    match=np.array_equal(np.isnan(a),np.isnan(b)) and np.allclose(a,b,rtol=0,atol=1e-7,equal_nan=True)
    write(R/'checks/raster-live-comparison.json',{'original_s':slow_s,'compiled_s':fast_s,'same_coverage_and_depth_within_1e-7m':bool(match)})
    if not match:raise RuntimeError('Compiled robot depth differs from reference')
    def publish(i,result):
        sensors=dict(_sensors(observation));sensors.pop('rgbd_obstacle_map',None);sensors['finger_joint_positions']=runtime.env._robot.data.joint_pos[0,runtime.env._finger_joint_ids].detach().cpu().tolist();caps,_=_capability_summary(observation)
        write(R/'ipc'/f'reply_{i:04d}.json',{'observation_id':observation['observation_id'],'sensors':sensors,'capabilities':caps,'features':observation['features'],'rgb_path':str(runtime._last_rgb_path),'overlay_path':observation['frames'][observation['frame_sequence'][-1]['frame_key']],'result':result,'native_success':runtime.check_task(),'touch_source':'unfiltered total finger forces; object identity unknown','wrist_camera':dict(wrist)})
    error=None;publish(0,{'status':'native_reset_ready'})
    try:
        if args.smoke:
            start_decision(600)
            commands=[{'action':'move_relative','translation_m':[-.03,.02,-.02]},{'action':'rotate','rotation_rad':[0,0,.0872664626]},{'action':'set_gripper','width_m':0.0},{'action':'set_gripper','width_m':.08}]
            for i,active in enumerate(commands):
                started=time.monotonic();result=tools.dispatch(active,observation);log('actions.jsonl',{'index':i,'command':active,'elapsed_s':time.monotonic()-started,'result':result});observation=execution_observation(runtime.observe_for_model());publish(i+1,result)
            terminate('smoke_completed')
        else:
            for i in range(500):
                path=R/'ipc'/f'command_{i:04d}.json'
                while not path.exists():
                    if snapshot()['terminal']:return
                    time.sleep(.05)
                active=json.loads(path.read_text());started=time.monotonic()
                try:result=tools.dispatch(active,observation)
                except ValueError as exc:result={'status':'invalid_action','terminal':False,'feedback':str(exc)}
                except RuntimeError:
                    if not snapshot()['terminal']:raise
                    result={'status':'interrupted','terminal':True,'feedback':'Episode endpoint reached; no further control steps.'}
                log('actions.jsonl',{'index':i,'command':active,'elapsed_s':time.monotonic()-started,'result':result})
                if result.get('terminal'):terminate('executor_safety_stop',status=result.get('status'))
                if snapshot()['terminal']:break
                if active['action'] not in {'inspect_features','back_project','localize_visual','inspect_aperture'}:observation=execution_observation(runtime.observe_for_model())
                publish(i+1,result)
    except BaseException as exc:
        error=repr(exc);(R/'private/backend-exception.txt').write_text(traceback.format_exc());terminate('backend_error',error=error)
    finally:
        if unsubscribe:unsubscribe()
        wrist_unsubscribe()
        video_error=recorder.close();write(R/'checks/timings.json',dict(timings))
        write(R/'episode/backend-summary.json',{'physics_steps':count,'video_frames':recorder.frame_count,'video_error':video_error,'native_success':runtime.check_task(),'error':error,'terminal':snapshot()['terminal']})
        # Post-endpoint diagnostic only: never exposed to the model or motion controller.
        try:
            adapter=runtime._evaluator;ev=adapter._evaluator;tol=ev._tolerances
            records=[]
            for item in adapter._trace:
                m=item.measurement
                predicates={'through_selected_cover_hole':m.through_selected_cover_hole,'entered_selected_casing_opening':m.entered_selected_casing_opening,'entry_depth_valid':m.casing_entry_depth_m>=tol.min_casing_entry_depth_m,'shaft_radial_valid':m.shaft_radial_error_m<=tol.max_shaft_radial_error_m,'cap_gap_valid':m.cap_support_gap_m<=tol.max_cap_support_gap_m,'cap_support_contact':m.cap_support_contact,'penetration_valid':m.penetration_m<=tol.max_penetration_m,'no_premature_bottoming':not m.premature_bottoming,'insertion_provenance_valid':m.insertion_provenance_valid,'controller_stopped':m.controller_stopped,'physically_held':m.physically_held,'robot_contact_or_support':m.robot_contact_or_support,'seat_ready':item.status.seat_ready,'task_success':item.status.task_success}
                records.append({'time_s':m.time_s,**{k:bool(v) for k,v in predicates.items()}})
            write(R/'private/posthoc-predicate-diagnostic.json',{'post_endpoint_only':True,'contains_target_coordinates':False,'counts_true':{k:sum(x[k] for x in records) for k in predicates} if records else {},'sample_count':len(records),'last_samples':records[-30:],'tail_held_samples':[x for x in records if x['physically_held']][-30:],'records':records,'semantic_progress':{k:v for k,v in adapter.checkpoint_progression().items() if isinstance(v,bool) or k in {'held_inserted_sample_count','pickup_held_streak'}},'temporal_evaluator_failed':bool(ev._failed),'temporal_evaluator_complete':bool(ev._complete)})
        except Exception as diag_exc:write(R/'private/posthoc-predicate-diagnostic-error.json',{'error':repr(diag_exc)})
        runtime.close()
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write(R/'private/backend-failed.json',{'error':repr(exc),'traceback':traceback.format_exc()});terminate('backend_startup_error',error=repr(exc));raise
