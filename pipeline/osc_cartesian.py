"""Task-independent adapter to installed Isaac Lab OperationalSpaceController.

Inputs: public robot dynamics, measured TCP/jaw state and declared payload.
No scene-object handles, private evaluator data or simulator object poses.
"""
import numpy as np
import os
from factory_cartesian import FactoryCartesian
from robot_kinematics import PANDA_TCP_ANGULAR_SPEED_RADPS,PANDA_TCP_LINEAR_SPEED_MPS,joint_limit_violation,payload_support_wrench

class OfficialOSC(FactoryCartesian):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from isaaclab.controllers import OperationalSpaceController,OperationalSpaceControllerCfg
        rotation_stiffness=float(os.environ.get('ASSEMBLY_OSC_ROTATION_STIFFNESS','100'))
        if not np.isfinite(rotation_stiffness) or not 100<=rotation_stiffness<=1000:
            raise ValueError('OSC rotation stiffness outside diagnostic range')
        cfg=OperationalSpaceControllerCfg(target_types=['pose_abs'],inertial_dynamics_decoupling=True,gravity_compensation=not self.gravity_disabled,motion_stiffness_task=[150.,150.,150.,rotation_stiffness,rotation_stiffness,rotation_stiffness],motion_damping_ratio_task=1.,nullspace_control='position')
        self.osc=OperationalSpaceController(cfg,num_envs=1,device='cpu')
        self.rest=self.robot.data.joint_pos[:,self.joints].clone();self.payload_estimate=None
    def apply(self,reference,goal,width,dt,armed,grip_force_n=None,finger_force_n=0.):
        from scipy.spatial.transform import Rotation
        from m1_reach.control import terminate
        from gripper_control import jaw_reference_step
        torch=self.torch;p,q,v,w,jac=self.measured();actual=torch.cat([p,q],dim=1)
        speed=float(torch.linalg.vector_norm(v));angular=float(torch.linalg.vector_norm(w))
        if armed and (speed>PANDA_TCP_LINEAR_SPEED_MPS or angular>PANDA_TCP_ANGULAR_SPEED_RADPS):
            terminate('executor_safety_stop',speed_mps=speed,angular_speed_radps=angular,source_measurement='measured robot velocity each physics tick');raise RuntimeError('Measured motion emergency stop')
        limits=self.robot.data.soft_joint_pos_limits[0,self.joints]
        if joint_limit_violation(self.robot.data.joint_pos[0,self.joints].cpu().numpy(),limits.cpu().numpy()):
            terminate('joint_limit_stop');raise RuntimeError('Measured joint limit stop')
        command=np.asarray(reference,float)
        angle=(Rotation.from_quat(command[[4,5,6,3]])*Rotation.from_quat(actual[0,[4,5,6,3]].cpu().numpy()).inv()).magnitude()
        if np.linalg.norm(command[:3]-p[0].cpu().numpy())>.02 or angle>.2:
            terminate('tcp_tracking_stop',commanded_tcp=command.tolist(),measured_tcp=actual[0].tolist(),
                translation_error_m=float(np.linalg.norm(command[:3]-p[0].cpu().numpy())),
                rotation_error_rad=float(angle),previous_controller_diagnostic=getattr(self,'last_tracking_diagnostic',None))
            raise RuntimeError('Measured TCP tracking stop')
        mass=self.robot.root_physx_view.get_generalized_mass_matrices()[:,self.joints][:,:,self.joints]
        gravity=self.robot.root_physx_view.get_gravity_compensation_forces()[:,self.joints]
        if self.payload_estimate is not None:
            from robot_kinematics import payload_joint_inertia
            payload=self.payload_estimate;rotation=Rotation.from_quat(q[0,[1,2,3,0]].cpu().numpy()).as_matrix()
            lever=rotation@payload['local_com_proxy_m'];inertia=payload['mass_kg']*rotation@payload['local_inertia_per_mass_m2']@rotation.T
            mass=mass+torch.tensor(payload_joint_inertia(jac[0].cpu().numpy(),payload['mass_kg'],lever,inertia)[None],dtype=torch.float32)
        self.osc.set_command(torch.tensor(command[None],dtype=torch.float32),current_ee_pose_b=actual)
        effort=self.osc.compute(jacobian_b=jac,current_ee_pose_b=actual,current_ee_vel_b=torch.cat([v,w],dim=1),mass_matrix=mass,gravity=gravity,current_joint_pos=self.robot.data.joint_pos[:,self.joints],current_joint_vel=self.robot.data.joint_vel[:,self.joints],nullspace_joint_pos_target=self.rest)
        motion_effort=effort-(gravity if not self.gravity_disabled else 0.)
        wrench=(torch.linalg.pinv(jac[0].T)@motion_effort[0]).cpu().numpy()
        if self.force_limit is not None:
            difference=wrench[:3]-self.force_reference;limited=self.force_reference+difference*min(1.,self.force_limit/max(np.linalg.norm(difference),1e-12))
            effort[0]+=jac[0,:3].T@torch.tensor(limited-wrench[:3],dtype=torch.float32);wrench[:3]=limited
        self.last_command_wrench=wrench.copy()
        if self.payload_estimate is not None:
            payload=self.payload_estimate;lever=Rotation.from_quat(q[0,[1,2,3,0]].cpu().numpy()).apply(payload['local_com_proxy_m'])
            effort[0]+=jac[0].T@torch.tensor(payload_support_wrench(payload['mass_kg'],lever),dtype=torch.float32)
        if not bool(torch.isfinite(effort).all()):
            terminate('nonfinite_control_stop');raise RuntimeError('Non-finite robot torque')
        torques=torch.zeros_like(self.robot.data.joint_pos);torques[:,self.joints]=effort
        bounds=self.robot.data.joint_effort_limits;torques=torch.clamp(torques,-bounds,bounds)
        self.last_tracking_diagnostic={'translation_error_m':float(np.linalg.norm(command[:3]-p[0].cpu().numpy())),
            'rotation_error_rad':float(angle),'joint_effort_before_clip_nm':effort[0].tolist(),
            'joint_effort_applied_nm':torques[0,self.joints].tolist(),
            'gravity_nm':gravity[0].tolist(),'payload_estimate_active':self.payload_estimate is not None,
            'command_wrench_world':self.last_command_wrench.tolist(),
            'jacobian_singular_values':torch.linalg.svdvals(jac[0]).tolist(),
            'joint_positions_rad':self.robot.data.joint_pos[0,self.joints].tolist(),
            'joint_velocities_radps':self.robot.data.joint_vel[0,self.joints].tolist()}
        measured=float(self.robot.data.joint_pos[0,self.grips].sum())
        if self._jaw_reference is None:self._jaw_reference=measured
        self._jaw_reference=jaw_reference_step(self._jaw_reference,width,finger_force_n,grip_force_n,dt)
        targets=self.robot.data.joint_pos.clone();targets[:,self.grips]=self._jaw_reference/2
        self.robot.set_joint_position_target(targets);self.robot.set_joint_effort_target(torques)
        self._wrench=.8*self._wrench+.2*self.wrench_world()
