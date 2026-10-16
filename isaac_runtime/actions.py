"""Simulation-only LIBERO-style Cartesian servo; no geometric action authorization."""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from protected_actions import vector, CartesianTools as ProtectedTools
from common import snapshot

KINDS={'observe','move_to','move_relative','rotate','set_gripper','back_project','inspect_features','ask_human_help','stop','finish'}
def validate_command(c):
    fields={'observe':set(),'move_to':{'position_m','quaternion_wxyz'},'move_relative':{'translation_m','rotation_rad'},'rotate':{'rotation_rad'},'set_gripper':{'width_m'},'back_project':{'pixel_xy'},'inspect_features':{'feature_refs'},'ask_human_help':set(),'stop':set(),'finish':set()}
    if not isinstance(c,dict) or c.get('action') not in KINDS:raise ValueError('unknown action; use direct motion and gripper tools')
    if set(c)-fields[c['action']]-{'action','reasoning'}:raise ValueError('unexpected fields')
    required={'move_to':'position_m','move_relative':'translation_m','rotate':'rotation_rad','set_gripper':'width_m','back_project':'pixel_xy','inspect_features':'feature_refs'}.get(c['action'])
    if required and required not in c:raise ValueError('missing '+required)
    if c['action']=='set_gripper':
        v=c['width_m']
        if isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or not 0<=v<=.08:raise ValueError('width_m must be between 0 and 0.08')
    if c['action']=='back_project':vector(c['pixel_xy'],2,'pixel_xy')
    return c

def target_pose(c,current):
    p=vector(np.asarray(current).tolist(),7,'current pose');q=Rotation.from_quat(p[[4,5,6,3]])
    if c['action']=='move_to':
        p[:3]=vector(c['position_m'],3,'position_m')
        if 'quaternion_wxyz' in c:
            v=vector(c['quaternion_wxyz'],4,'quaternion_wxyz')
            if abs(np.linalg.norm(v)-1)>1e-5:raise ValueError('quaternion must be unit length')
            p[3:]=v
    else:
        if c['action']=='move_relative':p[:3]+=vector(c['translation_m'],3,'translation_m')
        if 'rotation_rad' in c:p[3:]=(Rotation.from_rotvec(vector(c['rotation_rad'],3,'rotation_rad'))*q).as_quat()[[3,0,1,2]]
    return p

def array(t):return t.detach().cpu().numpy() if hasattr(t,'detach') else np.asarray(t)
class CartesianTools(ProtectedTools):
    def pose(self):
        env=self.runtime.env
        return np.r_[array(env.fingertip_midpoint_pos)[0],array(env.fingertip_midpoint_quat)[0]]
    def control_step(self,target):
        import torch
        env=self.runtime.env;pose=self.pose()
        error=np.r_[target[:3]-pose[:3],(Rotation.from_quat(target[[4,5,6,3]])*Rotation.from_quat(pose[[4,5,6,3]]).inv()).as_rotvec()]
        # Like a delta OSC command, integrate from measured TCP rather than winding up an unreachable target.
        env.commanded_tcp_pos[:]=env.fingertip_midpoint_pos
        env.commanded_tcp_quat[:]=env.fingertip_midpoint_quat
        scale=np.r_[env.cfg.ctrl.pos_action_threshold,env.cfg.ctrl.rot_action_threshold]
        step_limit=np.minimum(scale,np.array([.001]*3+[.01]*3))
        action=np.clip(error,-step_limit,step_limit)/scale
        env.step(torch.as_tensor([action.tolist()],dtype=torch.float32,device=env.device))
    def hold(self):self.control_step(self.pose())
    def move(self,c,observation):
        start=self.pose();target=target_pose(c,start);steps=0;settled=0
        limit=max(1,int(10/self.ex.limits.control_dt_s))
        for _ in range(limit):
            if snapshot()['terminal']:break
            pose=self.pose();pos=np.linalg.norm(target[:3]-pose[:3]);angle=(Rotation.from_quat(target[[4,5,6,3]])*Rotation.from_quat(pose[[4,5,6,3]]).inv()).magnitude()
            settled=settled+1 if pos<=.002 and angle<=.03 else 0
            if settled>=5:break
            self.control_step(target);steps+=1
        end=self.pose()
        return {'status':'reached' if settled>=5 else 'motion_timeout','terminal':False,'target_tcp_pose_wxyz':target.tolist(),'measured_start_tcp_pose':start.tolist(),'measured_end_tcp_pose':end.tolist(),'position_error_m':float(np.linalg.norm(target[:3]-end[:3])),'total_control_steps':steps,'execution_diagnostics':{'driver_step_count':steps,'ik_call_count':0,'sweep_check_count':0}}
    def dispatch(self,c,obs):
        c=validate_command(c)
        if c['action'] in {'move_to','move_relative','rotate'}:return self.move(c,obs)
        if c['action']=='set_gripper':
            self.runtime.env.set_gripper_target_width(float(c['width_m']),self.ex.limits.max_gripper_force_n)
            steps=0;target=self.pose()
            for _ in range(20):
                if snapshot()['terminal']:break
                self.control_step(target);steps+=1
            width=float(array(self.runtime.env.get_public_executor_state()['gripper_width_m']).reshape(-1)[0])
            return {'status':'gripper_commanded','requested_width_m':c['width_m'],'measured_width_m':width,'execution_diagnostics':{'driver_step_count':steps,'ik_call_count':0,'sweep_check_count':0}}
        return super().dispatch(c,obs)
