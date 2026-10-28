"""Robot-only adapter over installed NVIDIA Factory Cartesian torque control.

No task object, target object transform, evaluator or identity-filtered sensor
is accepted. Part-independent motion uses measured robot TCP and real hand FT.
"""
import numpy as np
from types import SimpleNamespace
from robot_kinematics import transform,PANDA_TCP_ANGULAR_SPEED_RADPS,PANDA_TCP_LINEAR_SPEED_MPS,joint_limit_violation

class FactoryCartesian:
    def __init__(self,robot,body,joints,grips,offset,stage,gravity_disabled):
        import torch
        from isaaclab_tasks.direct.factory import factory_control,factory_utils
        from isaaclab_tasks.direct.factory.factory_env_cfg import CtrlCfg
        from pxr import UsdPhysics
        self.torch=torch;self.control=factory_control;self.robot=robot;self.body=body;self.joints=joints;self.grips=grips;self.offset=torch.tensor([offset],dtype=torch.float32);self.gravity_disabled=gravity_disabled
        control_cfg=CtrlCfg()
        self.cfg=SimpleNamespace(scene=SimpleNamespace(num_envs=1),ctrl=SimpleNamespace(default_dof_pos_tensor=list(control_cfg.reset_joints),kp_null=control_cfg.kp_null,kd_null=control_cfg.kd_null))
        self.kp=torch.tensor([[1000.,1000.,1000.,30.,30.,30.]])
        self.kd=factory_utils.get_deriv_gains(self.kp,rot_deriv_scale=10.)
        self.force_limit=None;self.force_reference=np.zeros(3);self.last_command_wrench=np.zeros(6);self._wrench=np.zeros(6);self._bias=np.zeros(3);self._last_goal=None;self._debug_step=0;self._jaw_reference=None;self._rotation_bias=np.zeros(3)
        expected='/World/Robot/'+robot.body_names[body];joint=None
        for prim in stage.Traverse():
            if not str(prim.GetPath()).startswith('/World/Robot/') or not prim.IsA(UsdPhysics.Joint):continue
            candidate=UsdPhysics.Joint(prim)
            if list(map(str,candidate.GetBody1Rel().GetTargets()))==[expected]:joint=candidate;break
        if joint is None:raise RuntimeError('Public robot wrist joint frame unavailable')
        q=joint.GetLocalRot1Attr().Get();self.sensor_local=transform(joint.GetLocalPos1Attr().Get(),[q.GetReal(),*q.GetImaginary()])
        doc=robot.root_physx_view.get_link_incoming_joint_force.__doc__ or ''
        if 'child' not in doc.lower():raise RuntimeError('Installed wrench API did not confirm child frame')
    def measured(self):
        from isaaclab.utils.math import quat_apply
        torch=self.torch;body=self.robot.data.body_state_w[:,self.body];r=quat_apply(body[:,3:7],self.offset)
        position=body[:,:3]+r;quat=body[:,3:7];angular=body[:,10:13];linear=self.robot.data.body_link_lin_vel_w[:,self.body]+torch.cross(angular,r,dim=-1)
        jac=self.robot.root_physx_view.get_jacobians()[:,self.body-1 if self.robot.is_fixed_base else self.body][:,:,self.joints]
        shifted=jac[:,:3]+torch.cross(jac[:,3:].transpose(1,2),r[:,None,:],dim=-1).transpose(1,2)
        jac=torch.cat([shifted,jac[:,3:]],dim=1)
        return position,quat,linear,angular,jac
    def wrench_world(self):
        from scipy.spatial.transform import Rotation
        body=self.robot.data.body_state_w[0,self.body,:7].cpu().numpy();source=transform(body[:3],body[3:])@self.sensor_local;raw=self.robot.root_physx_view.get_link_incoming_joint_force().reshape(1,-1,6)[0,self.body].cpu().numpy();tcp=body[:3]+Rotation.from_quat(body[[4,5,6,3]]).apply(self.offset[0].cpu().numpy());force=source[:3,:3]@raw[:3];torque=source[:3,:3]@raw[3:]+np.cross(source[:3,3]-tcp,force)
        return np.r_[force,torque]
    def apply(self,reference,goal,width,dt,armed,grip_force_n=None,finger_force_n=0.):
        from scipy.spatial.transform import Rotation
        from m1_reach.control import terminate
        torch=self.torch;position,quat,linear,angular,jac=self.measured();actual=np.r_[position[0].cpu().numpy(),quat[0].cpu().numpy()]
        if armed and (float(torch.linalg.vector_norm(linear))>PANDA_TCP_LINEAR_SPEED_MPS or float(torch.linalg.vector_norm(angular))>PANDA_TCP_ANGULAR_SPEED_RADPS):
            terminate('executor_safety_stop',speed_mps=float(torch.linalg.vector_norm(linear)),angular_speed_radps=float(torch.linalg.vector_norm(angular)),source_measurement='measured robot velocity each physics tick');raise RuntimeError('Measured motion emergency stop')
        q=self.robot.data.joint_pos[0,self.joints];limits=self.robot.data.soft_joint_pos_limits[0,self.joints]
        if joint_limit_violation(q.cpu().numpy(),limits.cpu().numpy()):
            terminate('joint_limit_stop');raise RuntimeError('Measured joint limit stop')
        error=np.asarray(goal)[:3]-actual[:3]
        if np.linalg.norm(error)<.004:self._bias=np.clip(self._bias+error*dt*.15,-.0015,.0015)
        else:self._bias*=.98
        command=np.asarray(reference,float).copy();command[:3]+=self._bias
        if __import__('os').environ.get('ASSEMBLY_CONTROLLER')=='factory_damped':
            reference_rotation=Rotation.from_quat(command[[4,5,6,3]]);measured_rotation=Rotation.from_quat(actual[[4,5,6,3]])
            rotation_error=(reference_rotation*measured_rotation.inv()).as_rotvec()
            if np.linalg.norm(rotation_error)<.1:self._rotation_bias+=rotation_error*dt*.3
            self._rotation_bias*=min(1.,.04/max(np.linalg.norm(self._rotation_bias),1e-12))
            command[3:]=(Rotation.from_rotvec(self._rotation_bias)*reference_rotation).as_quat()[[3,0,1,2]]
        if np.linalg.norm(command[:3]-actual[:3])>.02 or (Rotation.from_quat(command[[4,5,6,3]])*Rotation.from_quat(actual[[4,5,6,3]]).inv()).magnitude()>.2:
            terminate('tcp_tracking_stop',commanded_tcp=command.tolist(),measured_tcp=actual.tolist(),measured_linear=linear[0].tolist(),measured_angular=angular[0].tolist());raise RuntimeError('Measured TCP tracking stop')
        mass=self.robot.root_physx_view.get_generalized_mass_matrices()[:,self.joints][:,:,self.joints]
        torques,wrench=self.control.compute_dof_torque(cfg=self.cfg,dof_pos=self.robot.data.joint_pos,dof_vel=self.robot.data.joint_vel,fingertip_midpoint_pos=position,fingertip_midpoint_quat=quat,fingertip_midpoint_linvel=linear,fingertip_midpoint_angvel=angular,jacobian=jac,arm_mass_matrix=mass,ctrl_target_fingertip_midpoint_pos=torch.tensor(command[None,:3],dtype=torch.float32),ctrl_target_fingertip_midpoint_quat=torch.tensor(command[None,3:],dtype=torch.float32),task_prop_gains=self.kp,task_deriv_gains=self.kd,device='cpu',dead_zone_thresholds=None)
        if self.force_limit is not None:
            force=wrench[0,:3].cpu().numpy();difference=force-self.force_reference;limited=self.force_reference+difference*min(1.,self.force_limit/max(np.linalg.norm(difference),1e-12));delta=torch.tensor(limited-force,dtype=torch.float32)
            torques[0,self.joints]+=jac[0,:3].T@delta;wrench[0,:3]=torch.tensor(limited,dtype=torch.float32)
        self.last_command_wrench=wrench[0].cpu().numpy().copy()
        # Stable implicit PD discretization: tiny authored distal inertias make
        # explicit J^T damping unstable at 120 Hz. Solve the robot mass/damping
        # system instead of adding fictitious inertia or weakening speed stops.
        damping=jac.transpose(1,2)@torch.diag_embed(self.kd)@jac
        stiffness=jac.transpose(1,2)@torch.diag_embed(self.kp)@jac
        effective=mass+dt*damping+dt*dt*stiffness
        acceleration=torch.linalg.solve(effective,torques[:,self.joints].unsqueeze(-1))
        torques[:,self.joints]=(mass@acceleration).squeeze(-1)
        if not self.gravity_disabled:torques+=self.robot.root_physx_view.get_gravity_compensation_forces()
        if not bool(torch.isfinite(torques).all()):
            terminate('nonfinite_control_stop');raise RuntimeError('Non-finite robot torque')
        torques[:,self.grips]=0.;effort_limits=self.robot.data.joint_effort_limits;torques=torch.clamp(torques,-effort_limits,effort_limits)
        measured_jaw=float(self.robot.data.joint_pos[0,self.grips].sum())
        if self._jaw_reference is None:self._jaw_reference=measured_jaw
        from gripper_control import jaw_reference_step
        self._jaw_reference=jaw_reference_step(self._jaw_reference,width,finger_force_n,grip_force_n,dt)
        targets=self.robot.data.joint_pos.clone();targets[:,self.grips]=self._jaw_reference/2
        self.robot.set_joint_position_target(targets);self.robot.set_joint_effort_target(torques)
        if self._debug_step<120:
            from common import R
            import json
            (R/'checks').mkdir(exist_ok=True)
            with (R/'checks/robot-only-torque-startup.jsonl').open('a') as stream:stream.write(json.dumps({'step':self._debug_step,'pose':actual.tolist(),'velocity':torch.cat([linear[0],angular[0]]).tolist(),'q':q.tolist(),'qd':self.robot.data.joint_vel[0,self.joints].tolist(),'torque':torques[0].tolist(),'jacobian':jac[0].tolist(),'mass':mass[0].tolist()})+'\n')
        self._debug_step+=1
        self._wrench=.8*self._wrench+.2*self.wrench_world()
