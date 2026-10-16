"""GLM-selected RGB-D targets and force-feedback motion, no part-pose oracle."""
import math,time
import numpy as np
from scipy.spatial.transform import Rotation
from actions import CartesianTools as DirectTools,array,target_pose
from common import snapshot,log

class CartesianTools(DirectTools):
    def __init__(self,runtime):
        super().__init__(runtime);self.home_orientation=self.pose()[3:].copy();self._entry_found=False;self._hold_bias=np.zeros(3)
    def control_step(self,target):
        import torch
        env=self.runtime.env;pose=self.pose();error=np.r_[target[:3]-pose[:3],(Rotation.from_quat(target[[4,5,6,3]])*Rotation.from_quat(pose[[4,5,6,3]]).inv()).as_rotvec()]
        # Bounded integral reference bias compensates steady load without changing native stops.
        error[:3]+=self._hold_bias
        # Limit the reference lead in vector norm; keep native speed/tracking stops.
        linear=error[:3]*min(1.,.0025/max(np.linalg.norm(error[:3]),1e-12))
        angular=error[3:]*min(1.,.008/max(np.linalg.norm(error[3:]),1e-12))
        env.commanded_tcp_pos[:]=env.fingertip_midpoint_pos;env.commanded_tcp_quat[:]=env.fingertip_midpoint_quat
        scale=np.r_[env.cfg.ctrl.pos_action_threshold,env.cfg.ctrl.rot_action_threshold]
        env.step(torch.as_tensor([np.clip(np.r_[linear,angular]/scale,-1,1).tolist()],dtype=torch.float32,device=env.device))
    def move(self,c,obs):
        start=self.pose();target=target_pose(c,start);steps=0;stable=0;self._hold_bias=np.zeros(3)
        for _ in range(round(20/self.ex.limits.control_dt_s)):
            if snapshot()['terminal']:break
            pose=self.pose();p=np.linalg.norm(target[:3]-pose[:3]);a=(Rotation.from_quat(target[[4,5,6,3]])*Rotation.from_quat(pose[[4,5,6,3]]).inv()).magnitude()
            stable=stable+1 if np.linalg.norm(target[:2]-pose[:2])<.00012 and abs(target[2]-pose[2])<.0007 and a<.003 else 0
            if stable>=8:break
            if p<.004:
                self._hold_bias=np.clip(self._hold_bias+(target[:3]-pose[:3])*self.ex.limits.control_dt_s*.15,-.0015,.0015)
            self.control_step(target);steps+=1
            if steps%120==0:log('servo-feedback.jsonl',{'target':target.tolist(),'measured':self.pose().tolist(),'integral_bias_m':self._hold_bias.tolist(),'position_error_m':p})
        result={'status':'reached' if stable>=8 else 'motion_timeout','target_tcp_pose_wxyz':target.tolist(),'measured_start_tcp_pose':start.tolist(),'measured_end_tcp_pose':self.pose().tolist(),'position_error_m':float(np.linalg.norm(target[:3]-self.pose()[:3])),'total_control_steps':steps,'integral_bias_m':self._hold_bias.tolist(),'terminal':False}
        log('servo-waypoints.jsonl',result);return result
    def freeze(self,steps=30,target=None):
        target=self.pose() if target is None else target
        for _ in range(steps):
            if snapshot()['terminal']:break
            self.control_step(target)
    def command_width(self,width,steps=90):
        self.runtime.env.set_gripper_target_width(width,self.ex.limits.max_gripper_force_n);self.freeze(steps)
    def goto(self,xyz,q=None):
        c={'action':'move_to','position_m':np.asarray(xyz).tolist()}
        if q is not None:c['quaternion_wxyz']=np.asarray(q).tolist()
        return self.move(c,{})
    def contacts(self):return tuple(self.runtime.env.get_public_executor_state()['finger_bolt_contacts'])
    def force(self):
        # Sensor-frame calibration is robot/workcell metadata, not a target position.
        q=np.asarray(self.runtime.env.cfg.assembly_frame_quat_wxyz)
        return Rotation.from_quat(q[[1,2,3,0]]).apply(array(self.runtime.env.public_wrench())[0,:3].astype(float))
    def dispatch(self,c,obs):
        if c.get('action')=='settle':
            duration=float(c.get('duration_s',2))
            if not .05<=duration<=5:raise ValueError('duration_s must be 0.05..5 simulation seconds')
            self.freeze(round(duration/self.ex.limits.control_dt_s));return {'status':'settled','duration_s':duration,'tcp_pose':self.pose().tolist(),'native_success':self.runtime.check_task()}
        if c.get('action')=='inspect_aperture':return self.aperture(c['feature_ref'],obs)
        if c.get('action')=='localize_visual':return self.localize(c['pixel_xy'],obs)
        if c.get('action')=='grasp_visual':return self.grasp(c['pixel_xy'],obs)
        if c.get('action')=='insert_visual':return self.insert(c['feature_ref'],obs)
        return super().dispatch(c,obs)
    def localize(self,pixel,obs):
        from runtime.assembly_harness.executor import _sensors
        s=_sensors(obs);d=np.load(s['camera_depth_path'],allow_pickle=False);h,w=d.shape;x,y=map(float,pixel);ix,iy=round(x),round(y)
        if not 0<=ix<w or not 0<=iy<h:raise ValueError('Pixel outside current raw image')
        seed_z=float(d[iy,ix]);k=np.array(s['camera_intrinsics']);t=np.array(s['camera_extrinsics'])
        if not math.isfinite(seed_z) or seed_z<=0:raise ValueError('Invalid pixel depth')
        seed=(t@np.r_[np.linalg.solve(k,[x,y,1])*seed_z,1])[:3]
        yy,xx=np.mgrid[max(0,iy-55):min(h,iy+56),max(0,ix-55):min(w,ix+56)];z=d[yy,xx];valid=np.isfinite(z)&(z>0)
        uv=np.stack([xx[valid],yy[valid],np.ones(valid.sum())]);pc=(t[:3,:3]@(np.linalg.solve(k,uv)*z[valid])+t[:3,3:4]).T
        table=float(self.runtime.env.cfg.table_top_z_m)
        # Same visible tabletop threshold at every pixel; no semantic object pose is queried.
        keep=(np.linalg.norm(pc[:,:2]-seed[:2],axis=1)<.025)&(pc[:,2]>table+.008)&(pc[:,2]<seed[2]+.018)
        points=pc[keep]
        if len(points)<8:raise ValueError('Too few visible above-table points near selected pixel; select the object itself')
        top=float(np.quantile(points[:,2],.9));head=points[points[:,2]>=top-.0004];center=(head[:,:2].min(0)+head[:,:2].max(0))/2
        outer=points[np.linalg.norm(points[:,:2]-center,axis=1)>.007]
        head_height=float(top-np.quantile(outer[:,2],.05)) if len(outer)>8 else None
        from cylinder_fit import fit_xy
        shaft=points[points[:,2]<top-(head_height if head_height is not None else .008)-.001]
        axis_fit=fit_xy(shaft,.011767996864/2,center)
        if axis_fit is not None:center=np.asarray(axis_fit['center_xy_m'])
        result={'status':'localized','source_observation_id':obs['observation_id'],'pixel_xy':[x,y],'selected_point_world_m':seed.tolist(),'cap_center_world_m':[float(center[0]),float(center[1]),top],'visible_bounds_world_m':[points.min(0).tolist(),points.max(0).tolist()],'point_count':len(points),'head_point_count':len(head),'observed_head_height_m':head_height,'shaft_axis_fit':axis_fit,'method':'current RGB-D backprojection and local above-table points; no simulator part transform'}
        log('visual-measurements.jsonl',result);return result
    def grasp(self,pixel,obs):
        measurement=self.localize(pixel,obs);cap=np.array(measurement['cap_center_world_m']);q=self.home_orientation.copy();stages=[]
        self.command_width(.08)
        # Geometry-generated waypoints relative to the selected visible surface, not scene coordinates.
        safe_z=max(self.pose()[2],cap[2]+.05)
        for name,p in [('align',[cap[0],cap[1],safe_z]),('pregrasp',[cap[0],cap[1],cap[2]+.035]),('grasp',[cap[0],cap[1],cap[2]+.002])]:
            result=self.goto(p,q);stages.append({'stage':name,**result})
            if result['status']!='reached':return {'status':'grasp_approach_incomplete','measurement':measurement,'stages':stages}
        self.command_width(0.,120);contact=self.contacts();grip=self.pose().copy()
        if not all(contact):return {'status':'no_bilateral_contact','measurement':measurement,'finger_contacts':contact,'measured_tcp_pose':grip.tolist(),'stages':stages}
        self.held_geometry={'cap_tcp_m':Rotation.from_quat(grip[[4,5,6,3]]).inv().apply(cap-grip[:3]),'nominal_length_m':.044,'observed_head_height_m':measurement.get('observed_head_height_m'),'grasp_quaternion_wxyz':grip[3:].copy(),'source_observation_id':obs['observation_id']}
        lift=self.goto(grip[:3]+[0,0,.055],q);self.freeze(30)
        return {'status':'grasp_lifted' if all(self.contacts()) and self.pose()[2]-grip[2]>.04 else 'grasp_not_retained','finger_contacts':self.contacts(),'measurement':measurement,'stages':stages,'lift':lift,'measured_tcp_pose':self.pose().tolist()}
    def measure_inhand(self):
        from common import execution_observation
        from runtime.assembly_harness.executor import _sensors
        from inhand_measurement import measure
        observation=execution_observation(self.runtime.observe_for_model())
        result=measure(_sensors(observation));log('inhand-measurements.jsonl',result)
        return result
    def align_inhand(self,reference_q=None):
        target_rotation=Rotation.from_quat(self.pose()[[4,5,6,3]] if reference_q is None else np.asarray(reference_q)[[1,2,3,0]]);history=[]
        for iteration in range(3):
            measurement=self.measure_inhand();history.append(measurement)
            axis=np.asarray(measurement['axis_world_up']);tilt=math.acos(float(np.clip(axis[2],-1,1)))
            if tilt<.0015:break
            correction=Rotation.align_vectors([[0,0,1]],[axis])[0]
            # Accumulate visual corrections on the previous reference to compensate steady servo bias.
            target_rotation=correction*target_rotation;target=self.pose();target[3:]=target_rotation.as_quat()[[3,0,1,2]]
            self.freeze(120,target)
        measurement=self.measure_inhand();history.append(measurement)
        pose=self.pose();current=Rotation.from_quat(pose[[4,5,6,3]])
        tip_tcp=current.inv().apply(np.asarray(measurement['estimated_visible_tip_world_m'])-pose[:3])
        return {'measurements':history,'target_quaternion_wxyz':target_rotation.as_quat()[[3,0,1,2]].tolist(),'tip_tcp_m':tip_tcp.tolist(),'final_tilt_deg':measurement['tilt_deg']}
    def aperture(self,ref,obs):
        from runtime.assembly_harness.executor import _sensors
        from aperture_measurement import measure_aperture
        features={f['feature_ref']:f for f in obs['features']}
        if ref not in features:raise ValueError('Choose a current feature_ref')
        result=measure_aperture(features[ref]['geometry'],_sensors(obs),self.runtime._last_rgb_path)
        result.update(feature_ref=ref,source_observation_id=obs['observation_id'])
        log('aperture-measurements.jsonl',result);return result
    def insert(self,ref,obs):
        env=self.runtime.env;gains=env.task_prop_gains.clone();deriv=env.task_deriv_gains.clone()
        try:return self._insert_feedback(ref,obs)
        finally:env.task_prop_gains[:]=gains;env.task_deriv_gains[:]=deriv
    def _insert_feedback(self,ref,obs):
        if not hasattr(self,'held_geometry') or not all(self.contacts()):return {'status':'no_verified_visual_grasp','feedback':'First establish a visual grasp with bilateral contact and lift.'}
        features={f['feature_ref']:f for f in obs['features']}
        if ref not in features:raise ValueError('Choose a current feature_ref')
        measurement=self.aperture(ref,obs);center=np.array(measurement['position_m'],float)
        if center.shape!=(3,):raise ValueError('Selected aperture has no measured position')
        pose=self.pose();q=self.held_geometry['grasp_quaternion_wxyz'];rotation=Rotation.from_quat(q[[1,2,3,0]])
        cap_offset=rotation.apply(self.held_geometry['cap_tcp_m']);head_height=self.held_geometry.get('observed_head_height_m')
        if head_height is None or not .002<head_height<.015:head_height=.008
        tip_offset=cap_offset-np.array([0,0,self.held_geometry['nominal_length_m']+head_height])
        xy=center[:2]-tip_offset[:2];stage_z=center[2]-tip_offset[2]+.006
        high=max(pose[2],stage_z+.035)
        stages=[]
        for p in [[pose[0],pose[1],high],[xy[0],xy[1],high]]:
            result=self.goto(p,q);stages.append(result)
            if result['status']!='reached':return {'status':'insert_approach_incomplete','stages':stages}
        try:inhand=self.align_inhand(q)
        except ValueError as exc:return {'status':'inhand_axis_unavailable','feedback':str(exc),'stages':stages}
        q=np.asarray(inhand['target_quaternion_wxyz']);rotation=Rotation.from_quat(self.pose()[[4,5,6,3]])
        tip_offset=rotation.apply(np.asarray(inhand['tip_tcp_m']));xy=center[:2]-tip_offset[:2]
        alignment=self.goto([xy[0],xy[1],self.pose()[2]],q);stages.append(alignment)
        self.freeze(60);baseline_samples=[];hold_target=self.pose()
        for _ in range(30):self.control_step(hold_target);baseline_samples.append(self.force())
        baseline=np.median(baseline_samples,axis=0);start_z=float(self.pose()[2]);contact_z=None;search_center=xy.copy();max_depth=.052;seated=False;peak=0.;trace=[];z_reference=start_z;recent=[];stalled=False;search_step=0;search_started_step=None;searched_cycles=0;best_entry_z=None;search_active=True;entry_anchor_z=None
        self.runtime.env.task_prop_gains[:,:2]*=.25;self.runtime.env.task_deriv_gains[:,:2]*=.5
        # Contact-guided local search around the model-selected aperture, as in zero-shot insertion literature.
        for step in range(6000):
            if snapshot()['terminal']:break
            pose=self.pose();delta_force=self.force()-baseline;norm_force=float(np.linalg.norm(delta_force));load=float(abs(delta_force[2]));peak=max(peak,norm_force)
            if norm_force>4.:
                return {'status':'contact_force_limit','peak_delta_force_n':peak,'descent_m':start_z-pose[2],'stages':stages}
            if not all(self.contacts()):return {'status':'grasp_lost_during_insert','descent_m':start_z-pose[2],'stages':stages}
            if load>.8 and contact_z is None:contact_z=float(pose[2]);entry_anchor_z=float(pose[2])
            desired=pose.copy();desired[3:]=q
            estimated_depth=float(center[2]-(pose[2]+tip_offset[2]))
            expected_shaft_depth=self.held_geometry['nominal_length_m']
            if contact_z is not None and search_active and entry_anchor_z-float(pose[2])>.003:
                search_active=False;search_center=pose[:2].copy();recent=[]
            if contact_z is not None and search_active and estimated_depth<expected_shaft_depth-.003:
                phase=search_step*.05;radius=min(.0015,.00005+phase*.000008)
                desired[:2]=search_center+[radius*math.cos(phase),radius*math.sin(phase)];search_step+=1
                if search_started_step is None:search_started_step=step
            else:desired[:2]=search_center if contact_z is not None else xy
            # Position/force hybrid: accumulate a slow descent, stop on load, retract if overloaded.
            # Baseline is measured in free air, before any descent/contact.
            if load<.8:z_reference-=.00008
            elif load>1.2:z_reference+=.00005
            z_reference=float(np.clip(z_reference,pose[2]-.0015,pose[2]+.0015));desired[2]=z_reference
            self.control_step(desired)
            depth=float(start_z-self.pose()[2])
            recent.append(float(self.pose()[2]));recent=recent[-120:]
            stalled=len(recent)==120 and max(recent)-min(recent)<.0002
            if step%30==0:
                item={'step':step,'tcp_pose':self.pose().tolist(),'delta_vertical_force_n':load,'delta_force_world_xyz_n':delta_force.tolist(),'delta_force_norm_n':norm_force,'descent_m':depth,'finger_contacts':self.contacts(),'contact_reference_z_m':contact_z,'stalled':stalled};trace.append(item);log('insertion-feedback.jsonl',item)
            if stalled:
                seated=estimated_depth>=expected_shaft_depth-.003
                if seated:break
                if not search_active:
                    search_active=True;search_center=self.pose()[:2].copy();entry_anchor_z=float(self.pose()[2]);search_step=0;search_started_step=step;recent=[]
                elif search_started_step is not None and step-search_started_step>1800:break
            if contact_z is not None and contact_z-self.pose()[2]>max_depth:break
            if contact_z is None and depth>.10:break
        self._entry_found=False
        # Stop lateral search and physically settle while still holding before handing control back.
        if seated:self.freeze(round(2.0/self.ex.limits.control_dt_s))
        return {'status':'contact_stalled_after_entry' if seated else 'contact_stalled_without_entry' if stalled else 'insertion_not_confirmed','target_measurement':measurement,'inhand_alignment':inhand,'descent_m':start_z-self.pose()[2],'peak_delta_force_n':peak,'seating_candidate':seated,'released':False,'held_settle_s':2.0 if seated else 0.0,'expected_shaft_depth_m':expected_shaft_depth,'spiral_search_steps':search_step,'estimated_tip_depth_m':float(center[2]-(self.pose()[2]+tip_offset[2])),'tip_estimate_note':'Assumes retained grasp offset and nominal length; slip/tilt can change it.','contact_entry_depth_m':None if contact_z is None else contact_z-float(self.pose()[2]),'feedback':'Measured contact/entry is not proof of seating. You choose release, adjust or retract; only independent native success proves completion.','native_success':self.runtime.check_task(),'trace':trace[-8:],'stages':stages}
