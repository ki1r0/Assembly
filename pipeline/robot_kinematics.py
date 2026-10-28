"""IK from public authored robot joints, never from scene object transforms."""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

# Simulation disturbance stop, not a commanded-velocity rating. Commands stay <=0.15 rad/s.
# Contact responses clustered near the native wrist joint velocity limit 2.61 rad/s.
PANDA_TCP_ANGULAR_SPEED_RADPS = 3.0
PANDA_TCP_LINEAR_SPEED_MPS = .2
JOINT_LIMIT_SOLVER_TOLERANCE_RAD = 1e-4

def public_motion_reference_limits(controller_mode,speed_scale=1.):
    torque = controller_mode.startswith('factory') or controller_mode == 'isaac_osc'
    return {
        'joint_target_increment_rad_per_control': None if torque else .0025,
        'joint_reference_tracking_lead_limit_rad': None if torque else .008,
        'joint_integral_bias_limit_rad': None if torque else .02,
        'reference_cartesian_speed_mps': .010*speed_scale,
        'reference_angular_speed_radps': .060*speed_scale,
        'reference_translation_tracking_pause_m': .002,
        'reference_rotation_tracking_pause_rad': .02,
        'limit_semantics': 'Null joint-reference limits are not applicable to this torque controller. Reference speed is not guaranteed measured progress; inspect actual pose and tracking error.'}

def joint_limit_violation(position,limits):
    position=np.asarray(position);limits=np.asarray(limits)
    return not np.isfinite(position).all() or bool(np.any(position<limits[:,0]-JOINT_LIMIT_SOLVER_TOLERANCE_RAD) or np.any(position>limits[:,1]+JOINT_LIMIT_SOLVER_TOLERANCE_RAD))

def posture_reference(supplied,home,limits):
    home=np.asarray(home,float);limits=np.asarray(limits,float)
    if supplied is None:return home.copy()
    value=np.asarray(supplied,float)
    if value.shape!=home.shape or not np.isfinite(value).all():
        raise ValueError('Finite joint posture vector matching the arm required')
    if np.any(value<limits[:,0]) or np.any(value>limits[:,1]):
        raise ValueError('Joint posture must respect inset joint limits')
    return value.copy()

def nullspace_posture_increment(jacobian,current,reference,dt):
    jacobian=np.asarray(jacobian);current=np.asarray(current)
    null=np.eye(len(current))-np.linalg.pinv(jacobian)@jacobian
    return null@((np.asarray(reference)-current)*dt*.05)

def transform(p,q):
    out=np.eye(4);out[:3,:3]=Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix();out[:3,3]=p;return out

class RobotKinematics:
    def __init__(self,base,chain,joint_names,limits,offset):
        self.base=np.asarray(base);self.chain=chain;self.names=joint_names;self.limits=np.asarray(limits);self.offset=np.asarray(offset)
    @classmethod
    def from_stage(cls,stage,root,joint_names,limits,body,offset):
        from pxr import UsdPhysics,UsdGeom,Usd
        joints={}
        def quat(q):return [q.GetReal(),*q.GetImaginary()]
        for prim in stage.Traverse():
            if not str(prim.GetPath()).startswith(root+'/') or not prim.IsA(UsdPhysics.Joint):continue
            joint=UsdPhysics.Joint(prim);parents=joint.GetBody0Rel().GetTargets();children=joint.GetBody1Rel().GetTargets()
            if not parents or not children:continue
            left=transform(joint.GetLocalPos0Attr().Get(),quat(joint.GetLocalRot0Attr().Get()));right=transform(joint.GetLocalPos1Attr().Get(),quat(joint.GetLocalRot1Attr().Get()))
            axis=None
            if prim.IsA(UsdPhysics.RevoluteJoint):axis=np.eye(3)['XYZ'.index(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get())]
            joints[str(children[0])]=(str(parents[0]),left,right,prim.GetName(),axis)
        current=root+'/'+body;chain=[]
        while current in joints:
            parent,left,right,name,axis=joints[current];chain.append((left,np.linalg.inv(right),joint_names.index(name) if name in joint_names else None,axis));current=parent
        base=np.asarray(UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(stage.GetPrimAtPath(current))).T
        return cls(base,list(reversed(chain)),joint_names,limits,offset)
    def fk(self,q):
        value=self.base.copy()
        for left,right,index,axis in self.chain:
            rotation=np.eye(4)
            if index is not None:rotation[:3,:3]=Rotation.from_rotvec(axis*q[index]).as_matrix()
            value=value@left@rotation@right
        value[:3,3]+=value[:3,:3]@self.offset
        return value
    def jacobian(self,q):
        value=self.base.copy();axes={}
        for left,right,index,axis in self.chain:
            value=value@left;rotation=np.eye(4)
            if index is not None:
                axes[index]=(value[:3,3].copy(),value[:3,:3]@axis)
                rotation[:3,:3]=Rotation.from_rotvec(axis*q[index]).as_matrix()
            value=value@rotation@right
        tcp=value[:3,3]+value[:3,:3]@self.offset;jac=np.zeros((6,len(q)))
        for index,(origin,axis) in axes.items():jac[:3,index]=np.cross(axis,tcp-origin);jac[3:,index]=axis
        return jac
    def solve(self,target,q,configuration_clearances=None):
        target=np.asarray(target);initial=np.clip(q,self.limits[:,0]+1e-6,self.limits[:,1]-1e-6)
        def residual(value):
            actual=self.fk(value)
            error=np.r_[(actual[:3,3]-target[:3,3]),.15*Rotation.from_matrix(target[:3,:3]@actual[:3,:3].T).as_rotvec(),.001*(value-initial)]
            if configuration_clearances is not None:error=np.r_[error,np.minimum(np.asarray(configuration_clearances(value))-.001,0.)]
            return error
        result=least_squares(residual,initial,bounds=(self.limits[:,0],self.limits[:,1]),max_nfev=140,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        actual=self.fk(result.x);p=float(np.linalg.norm(actual[:3,3]-target[:3,3]));a=float(Rotation.from_matrix(target[:3,:3]@actual[:3,:3].T).magnitude())
        valid=configuration_clearances is None or min(configuration_clearances(result.x))>=0.
        if p>=.001 or a>=.015 or not valid:
            # Deterministic alternate seeds address bounded local IK minima;
            # every returned solution is still checked against the same limits.
            seeds=[(self.limits[:,0]+self.limits[:,1])/2]
            for sign in [-1,1]:
                seed=initial.copy();seed[0]+=sign*.6
                if len(seed)>3:seed[2]-=sign*.6
                seeds.append(np.clip(seed,self.limits[:,0]+1e-6,self.limits[:,1]-1e-6))
            for seed in seeds:
                retry=least_squares(residual,seed,bounds=(self.limits[:,0],self.limits[:,1]),max_nfev=180,ftol=1e-8,xtol=1e-8,gtol=1e-8)
                test=self.fk(retry.x);pp=float(np.linalg.norm(test[:3,3]-target[:3,3]));aa=float(Rotation.from_matrix(target[:3,:3]@test[:3,:3].T).magnitude())
                retry_valid=configuration_clearances is None or min(configuration_clearances(retry.x))>=0.
                if (retry_valid and not valid) or (retry_valid==valid and pp+.15*aa<p+.15*a):result,p,a,valid=retry,pp,aa,retry_valid
                if p<.001 and a<.015 and valid:break
        return {'reachable':p<.001 and a<.015 and valid,'position_error_m':p,'orientation_error_rad':a,'joints':result.x.tolist()}

def com_to_tcp_jacobian(jacobian,rotation_link_world,tcp_offset_link,com_position_link):
    """Shift world-axis SDK COM velocity rows to the link-defined TCP."""
    jac=np.asarray(jacobian,float);rotation=np.asarray(rotation_link_world,float)
    tcp=np.asarray(tcp_offset_link,float);com=np.asarray(com_position_link,float)
    if jac.ndim!=2 or jac.shape[0]!=6 or rotation.shape!=(3,3) or tcp.shape!=(3,) or com.shape!=(3,):
        raise ValueError('Expected 6xN Jacobian, 3x3 link rotation and two link-frame 3D offsets')
    if not all(np.isfinite(value).all() for value in [jac,rotation,tcp,com]):raise ValueError('Finite Jacobian and transforms required')
    r=rotation@(tcp-com)
    skew=np.array([[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]])
    return np.r_[jac[:3]-skew@jac[3:],jac[3:]]

def bounded_reference_increment(jacobian,offset_world,increment,dt,diagnostics=None):
    """Bound TCP reference velocity, since per-joint bounds alone do not do so."""
    jac=np.asarray(jacobian,float);dq=np.asarray(increment,float);r=np.asarray(offset_world,float)
    if jac.shape!=(6,len(dq)) or r.shape!=(3,):raise ValueError('Expected 6xN robot Jacobian and 3D TCP offset')
    skew=np.array([[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]])
    linear=(jac[:3]-skew@jac[3:])@dq;angular=jac[3:]@dq
    linear_scale=min(1.,.015*dt/max(np.linalg.norm(linear),1e-12));angular_scale=min(1.,.15*dt/max(np.linalg.norm(angular),1e-12))
    scale=min(linear_scale,angular_scale)
    if diagnostics is not None:diagnostics.update(tcp_linear_scale=float(linear_scale),tcp_angular_scale=float(angular_scale),tcp_applied_scale=float(scale),predicted_tcp_linear_speed_mps=float(np.linalg.norm(linear)*scale/dt),predicted_tcp_angular_speed_radps=float(np.linalg.norm(angular)*scale/dt),prediction_stage='after_cartesian_bound_before_joint_lead_clamp')
    return dq*scale

class CartesianReference:
    """Persistent Cartesian reference; interpolate in SE(3), not joint space."""
    def __init__(self,pose,speed_scale=1.):
        if not np.isfinite(speed_scale) or not 1.<=speed_scale<=2.:raise ValueError('Reference speed scale must be within 1..2')
        self.pose=np.array(pose,float);self.speed_scale=float(speed_scale)
    def advance(self,goal,measured,dt):
        goal=np.asarray(goal,float);measured=np.asarray(measured,float);reference=self.pose.copy()
        current_rotation=Rotation.from_quat(reference[[4,5,6,3]]);measured_rotation=Rotation.from_quat(measured[[4,5,6,3]])
        if np.linalg.norm(reference[:3]-measured[:3])>.002 or (current_rotation*measured_rotation.inv()).magnitude()>.02:return reference
        delta=goal[:3]-reference[:3];rv=(Rotation.from_quat(goal[[4,5,6,3]])*current_rotation.inv()).as_rotvec()
        fraction=min(1.,.010*self.speed_scale*dt/max(np.linalg.norm(delta),1e-12),.060*self.speed_scale*dt/max(np.linalg.norm(rv),1e-12))
        reference[:3]+=delta*fraction;reference[3:]=(Rotation.from_rotvec(rv*fraction)*current_rotation).as_quat()[[3,0,1,2]]
        self.pose=reference;return reference.copy()

def reset_motion_references(measured_poses,speed_scale=1.):
    targets={arm:np.asarray(pose,float).copy() for arm,pose in measured_poses.items()}
    references={arm:CartesianReference(pose,speed_scale) for arm,pose in targets.items()}
    return targets,references


def motion_reference_stale(reference,measured):
    reference=np.asarray(reference);measured=np.asarray(measured)
    angle=(Rotation.from_quat(reference[[4,5,6,3]])*Rotation.from_quat(measured[[4,5,6,3]]).inv()).magnitude()
    return np.linalg.norm(reference[:3]-measured[:3])>.002 or angle>.02


def payload_support_wrench(mass_kg,lever_world):
    """Static payload estimate from declared mass and observed COM proxy."""
    mass=float(mass_kg);lever=np.asarray(lever_world,float)
    if not np.isfinite(mass) or mass<0 or lever.shape!=(3,) or not np.isfinite(lever).all():raise ValueError('Finite nonnegative payload and measured 3D lever required')
    force=np.array([0.,0.,mass*9.81])
    return np.r_[force,np.cross(lever,force)]


def payload_joint_inertia(tcp_jacobian,mass_kg,com_lever_world,inertia_com_world):
    """Existing physical payload contribution; does not alter simulator inertia."""
    jac=np.asarray(tcp_jacobian,float);r=np.asarray(com_lever_world,float)
    skew=np.array([[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]])
    com_jac=jac[:3]-skew@jac[3:]
    return float(mass_kg)*com_jac.T@com_jac+jac[3:].T@np.asarray(inertia_com_world)@jac[3:]
