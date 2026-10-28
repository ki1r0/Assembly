"""LIBERO-style bounded Cartesian tools over the unchanged protected Factory runtime."""
import math,time,dataclasses
import numpy as np
from runtime.assembly_harness.depth_io import load_depth
from scipy.spatial.transform import Rotation
from common import execution_observation,snapshot

KINDS={'observe','move_to','move_relative','rotate','set_gripper','pick','execute_goal','back_project','inspect_features','ask_human_help','stop','finish'}
def vector(value,n,name):
    if not isinstance(value,list) or len(value)!=n or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in value):raise ValueError(name+' must be a finite numeric vector')
    return np.array(value,float)
def target_pose(command,current):
    current=vector(list(current),7,'current pose');target=current.copy();q=Rotation.from_quat(current[[4,5,6,3]])
    kind=command['action']
    if kind=='move_to':
        target[:3]=vector(command['position_m'],3,'position_m')
        if 'quaternion_wxyz' in command:
            v=vector(command['quaternion_wxyz'],4,'quaternion_wxyz')
            if abs(np.linalg.norm(v)-1)>1e-5:raise ValueError('quaternion must be unit length')
            target[3:]=v
    else:
        if kind=='move_relative':target[:3]+=vector(command['translation_m'],3,'translation_m')
        if 'rotation_rad' in command:
            v=vector(command['rotation_rad'],3,'rotation_rad');a=Rotation.from_rotvec(v)*q;xyzw=a.as_quat();target[3:]=xyzw[[3,0,1,2]]
    if np.linalg.norm(target[:3]-current[:3])>.1500001:raise ValueError('one motion is bounded to 150 mm; use another observed waypoint')
    angle=(Rotation.from_quat(target[[4,5,6,3]])*q.inv()).magnitude()
    if angle>math.pi/6+1e-7:raise ValueError('one rotation is bounded to 30 degrees')
    return target

def validate_command(c):
    if not isinstance(c,dict) or c.get('action') not in KINDS:raise ValueError('unknown action')
    fields={'observe':set(),'move_to':{'position_m','quaternion_wxyz'},'move_relative':{'translation_m','rotation_rad'},'rotate':{'rotation_rad'},'set_gripper':{'width_m'},'pick':{'feature_ref'},'execute_goal':{'goal'},'back_project':{'pixel_xy'},'inspect_features':{'feature_refs'},'ask_human_help':set(),'stop':set(),'finish':set()}
    if set(c)-fields[c['action']]-{'action','reasoning'}:raise ValueError('unexpected fields')
    for k in {'move_to':['position_m'],'move_relative':['translation_m'],'rotate':['rotation_rad'],'set_gripper':['width_m'],'pick':['feature_ref'],'execute_goal':['goal'],'back_project':['pixel_xy'],'inspect_features':['feature_refs']}.get(c['action'],[]):
        if k not in c:raise ValueError('missing '+k)
    if c['action']=='set_gripper':
        v=c['width_m']
        if isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or not 0<=v<=.08:raise ValueError('width_m must be between 0 and 0.08')
    if c['action']=='back_project':vector(c['pixel_xy'],2,'pixel_xy')
    return c

class CartesianTools:
    def __init__(self,runtime):self.runtime=runtime;self.ex=runtime.executor
    def fresh(self):return execution_observation(self.runtime.observe_public(include_feature_overlay=False))
    def move(self,c,observation):
        from runtime.assembly_harness.executor import _sensors,_quat_rotation_matrix,_rotation_vector
        from runtime.assembly_harness.agent import _capability_summary
        rt=self.runtime;target=target_pose(c,_sensors(observation)['tcp_pose']);initial=np.array(_sensors(observation)['tcp_pose']);passes=[];current=observation
        # Reissue the SAME absolute waypoint for bounded settling; never infer another waypoint.
        steps=0;limit=int(10/self.ex.limits.control_dt_s)
        for attempt in range(4):
            if snapshot()['terminal']:break
            s=_sensors(current);pose=np.array(s['tcp_pose']);remaining=limit-steps
            if remaining<=0:break
            if s.get('grasped') is False:
                rot=_quat_rotation_matrix(pose[3:]);call={'tool':'move_tcp','observation_id':current['observation_id'],'frame':'tcp','translation_delta_m':(rot.T@(target[:3]-pose[:3])).tolist(),'rotation_vector_rad':(rot.T@_rotation_vector(target[3:],pose[3:])).tolist(),'max_duration_s':min(10.,remaining*self.ex.limits.control_dt_s),'motion_budget':min(remaining,2000)}
                result=rt.execute_tcp_delta(call,current)
            else:
                caps,_=_capability_summary(current);held=caps.get('held_object_surface_candidate_refs',[])+caps.get('tracked_held_surface_refs',[]);planes=caps.get('plane_reference_refs',[])
                if not held or not planes:return {'status':'held_geometry_unverified','terminal':False,'feedback':'Held-object sweep geometry unavailable; observe or obtain new evidence.'}
                goal={'tool':'execute_goal','observation_id':current['observation_id'],'intent':'Reach the model-requested Cartesian waypoint.','stop_condition':'Reach and settle at the requested pose.','motion_regime':'free_space','constraints':[{'relation':'centered_on','moving_feature':held[0],'reference_feature':planes[0]}]}
                original=self.ex.connector.resolve
                def resolved(g,o):return dataclasses.replace(original(g,o),target_tcp_pose=tuple(target))
                self.ex.connector.resolve=resolved
                try:
                    result=rt._action_rejection(current)
                    if result is None:
                        d={'driver_step_count':0,'ik_call_count':0,'sweep_check_count':0};result=self.ex._execute_goal(goal,current,d,motion_budget_cap=remaining,force_guard=True);result['execution_diagnostics']=d
                finally:self.ex.connector.resolve=original
            passes.append(result);steps+=result.get('execution_diagnostics',{}).get('driver_step_count',0)
            if result.get('status')!='motion_timeout' or result.get('terminal'):break
            current=self.fresh()
        final=passes[-1] if passes else {'status':'interrupted','terminal':True}
        end=self.fresh() if not snapshot()['terminal'] else current
        return {**final,'target_tcp_pose_wxyz':target.tolist(),'measured_start_tcp_pose':initial.tolist(),'measured_end_tcp_pose':_sensors(end)['tcp_pose'],'settling_passes':passes,'total_control_steps':steps}
    def dispatch(self,c,obs):
        from runtime.assembly_harness.executor import _sensors,_obstacle_points,_bind_contact_surface_points
        from runtime.assembly_harness.agent import compile_grasp_tool_call,parse_tool_call,_capability_summary
        c=validate_command(c);action=c['action'];s=_sensors(obs)
        if action in {'move_to','move_relative','rotate'}:return self.move(c,obs)
        if action=='observe':return {'status':'observed'}
        if action=='inspect_features':
            refs=c['feature_refs']
            if not isinstance(refs,list) or not 1<=len(refs)<=6 or any(not isinstance(ref,str) for ref in refs):raise ValueError('inspect_features requires one to six refs')
            features={f['feature_ref']:f for f in obs['features']}
            if any(ref not in features for ref in refs):raise ValueError('unknown current feature_ref')
            return {'status':'measured','observation_id':obs['observation_id'],'features':[features[ref] for ref in refs]}
        if action=='back_project':
            x,y=vector(c['pixel_xy'],2,'pixel_xy');depth=load_depth(s['camera_depth_path']);ix,iy=int(round(x)),int(round(y))
            if not (0<=ix<depth.shape[1] and 0<=iy<depth.shape[0]):raise ValueError('pixel outside current frame')
            z=float(depth[iy,ix])
            if not math.isfinite(z) or z<=0:raise ValueError('no valid depth at selected pixel')
            k=np.array(s['camera_intrinsics']);t=np.array(s['camera_extrinsics']);p=t@np.r_[np.linalg.solve(k,[x,y,1])*z,1.]
            return {'status':'measured','pixel_xy':[x,y],'point_world_m':p[:3].tolist(),'source_observation_id':obs['observation_id']}
        if action=='pick':
            goal=compile_grasp_tool_call({'tool':'grasp','observation_id':obs['observation_id'],'target_feature_ref':c['feature_ref']},observation=obs)
            return self.runtime.execute_goal(goal,obs)
        if action=='execute_goal':
            goal=parse_tool_call(c['goal'],observation=obs,completed=False)
            if goal['tool']!='execute_goal':raise ValueError('only execute_goal is accepted in goal')
            return self.runtime.execute_goal(goal,obs)
        if action=='set_gripper':
            width=float(c['width_m'])
            if abs(width-s['gripper_width_m'])<=self.ex.limits.gripper_width_tolerance_m:return {'status':'gripper_unchanged','execution_diagnostics':{'driver_step_count':0,'ik_call_count':0,'sweep_check_count':0}}
            opening=width>s['gripper_width_m']+self.ex.limits.gripper_width_tolerance_m
            if opening and s.get('grasped') is not False and s.get('release_authorized') is not True:return {'status':'release_not_authorized','terminal':False}
            resolved=None;goal={'tool':'set_gripper','observation_id':obs['observation_id']}
            if not opening and width<s['gripper_width_m']-self.ex.limits.gripper_width_tolerance_m:
                if s.get('grasped') is not False:return {'status':'grasp_state_conflict','terminal':False}
                caps,_=_capability_summary(obs);candidates=[]
                from runtime.assembly_harness.executor import _rotation_vector
                for ref in caps.get('graspable_candidate_refs',[]):
                    try:
                        g=compile_grasp_tool_call({'tool':'grasp','observation_id':obs['observation_id'],'target_feature_ref':ref},observation=obs);r=self.ex.connector.resolve(g,obs)
                        if np.linalg.norm(np.array(r.target_tcp_pose[:3])-s['tcp_pose'][:3])<=self.ex.limits.position_tolerance_m and np.linalg.norm(_rotation_vector(r.target_tcp_pose[3:],s['tcp_pose'][3:]))<=self.ex.limits.orientation_tolerance_rad and abs(r.grasp_width_m-width)<=self.ex.limits.gripper_width_tolerance_m and r.allowed_contact_pairs:candidates.append((g,r))
                    except ValueError:continue
                if not candidates:return {'status':'grasp_not_aligned','terminal':False,'feedback':'Closing requires current aligned measured bilateral contact geometry and matching width. Use pick(current feature_ref) for geometry-checked approach and grasp.'}
                goal,resolved=candidates[0];_bind_contact_surface_points(resolved.allowed_contact_pairs,_obstacle_points(obs,max_age_s=self.ex.limits.max_rgbd_age_s))
            d={'driver_step_count':0,'ik_call_count':0,'sweep_check_count':0};self.ex._command_gripper(width,obs,closing=not opening,diagnostics=d,goal=goal,resolved=resolved)
            return {'status':'gripper_settled','execution_diagnostics':d}
        raise ValueError('terminal decisions are handled by the planner')
