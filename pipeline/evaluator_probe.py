"""Privileged evaluator diagnostic. Never imported by the actor or used in model runs."""
import json,math
import numpy as np
from scipy.spatial.transform import Rotation
from common import R,write,log,snapshot

def run(runtime,tools,observation):
    from runtime.bolt_harness.measurement import CAP_UNDERSIDE_LOCAL_Y_M
    from settings import CAD_METERS_PER_SOURCE_UNIT
    from fixture_geometry import bolt_fixture_geometry
    calibration=json.loads((__import__('pathlib').Path(__file__).parent/'config/fixture_calibration.json').read_text())
    frame=np.asarray(runtime.env.cfg.assembly_frame_pos_m,dtype=float)
    geometry=bolt_fixture_geometry(calibration,frame,CAD_METERS_PER_SOURCE_UNIT)
    target=np.r_[geometry['selected_hole_xy_m'],geometry['cap_support_plane_z_m']]
    def geometry():
        body=runtime.env._bolt.data.root_pos_w[0].detach().cpu().numpy()
        q=runtime.env._bolt.data.root_quat_w[0].detach().cpu().numpy()
        rotation=Rotation.from_quat(q[[1,2,3,0]])
        return body+rotation.apply([0,CAP_UNDERSIDE_LOCAL_Y_M,0]),rotation.apply([0,1,0])
    def diagnostics(label):
        adapter=runtime._evaluator;m=adapter._trace[-1].measurement
        record={'phase':label,'native_success':runtime.check_task(),'measurement':__import__('dataclasses').asdict(m),'progress':adapter.checkpoint_progression(),'privileged':True}
        log('evaluator-probe.jsonl',record)
        return m
    # The grasp is still physical; oracle geometry is only a calibration controller.
    result=tools.grasp([755,385],observation);log('evaluator-probe.jsonl',{'phase':'physical_grasp','result':result,'privileged':True})
    if result['status']!='grasp_lifted':return
    cap,axis=geometry();q=tools.pose()[3:]
    tools.goto([tools.pose()[0],tools.pose()[1],1.0],q)
    # Align the actual shaft, not only the nominal hand orientation.
    for _ in range(4):
        cap,axis=geometry()
        current=Rotation.from_quat(tools.pose()[[4,5,6,3]])
        correction=Rotation.align_vectors([[0,0,1]],[axis])[0]
        desired=(correction*current).as_quat()[[3,0,1,2]]
        tools.goto(tools.pose()[:3],desired)
    cap,axis=geometry();tcp=tools.pose();tools.goto(tcp[:3]+np.r_[target[:2]-cap[:2],0],tcp[3:])
    diagnostics('aligned_above_socket')
    for step in range(2400):
        if snapshot()['terminal']:break
        cap,axis=geometry();pose=tools.pose();current=Rotation.from_quat(pose[[4,5,6,3]])
        correction=Rotation.align_vectors([[0,0,1]],[axis])[0]
        # Cap underside approaches the actual support plane; no object pose writes.
        error=target-cap;desired=pose.copy()
        desired[:2]+=np.clip(error[:2],-.0004,.0004)
        desired[2]+=np.clip(error[2],-.0004,.0002)
        desired[3:]=(correction*current).as_quat()[[3,0,1,2]]
        tools.control_step(desired)
        if step%30==0:diagnostics('controlled_descent')
        if step%180==0 and abs(error[2])<.0001 and np.linalg.norm(error[:2])<.00015:
            fixed=tools.pose().copy()
            for _ in range(120):
                tools.control_step(fixed)
                if runtime._last_private_status.seat_ready:break
            diagnostics('fixed_target_stability_check')
        if runtime._last_private_status.seat_ready:
            diagnostics('held_seat_ready');break
    m=diagnostics('before_release')
    if runtime._last_private_status.seat_ready:
        tools.command_width(.08,120)
        tools.goto(tools.pose()[:3]+[0,0,.08],tools.pose()[3:])
        tools.freeze(180)
        diagnostics('released_and_retracted')
    write(R/'checks/evaluator-probe-result.json',{'native_success':runtime.check_task(),'privileged_calibration_only':True,'not_a_model_rollout':True,'no_object_pose_writes':True})
