"""Simulation-only xArm7/G1 embodiment, using the converted official model."""


def diagnostic_initial_joints(values,urdf):
    import numpy as np
    import xml.etree.ElementTree as ET
    values=np.asarray(values,float)
    if values.shape!=(7,) or not np.isfinite(values).all():raise ValueError('Seven finite initial arm angles required')
    joints={joint.get('name'):joint for joint in ET.parse(urdf).getroot().findall('joint')}
    result={}
    for i,value in enumerate(values,1):
        name=f'joint{i}';limit=joints[name].find('limit')
        if not float(limit.get('lower'))+.005<=value<=float(limit.get('upper'))-.005:
            raise ValueError('Diagnostic initial arm angle outside inset URDF limits: '+name)
        result[name]=float(value)
    return result


def friction_tracking_bias(error_rad, friction_nm, stiffness_nm_per_rad):
    import numpy as np
    error=np.asarray(error_rad,dtype=float)
    friction=np.asarray(friction_nm,dtype=float)
    stiffness=np.asarray(stiffness_nm_per_rad,dtype=float)
    if error.shape!=friction.shape or error.shape!=stiffness.shape or not all(np.isfinite(x).all() for x in (error,friction,stiffness)) or (friction<0).any() or (stiffness<=0).any():
        raise ValueError('Finite matching joint vectors, nonnegative friction and positive stiffness required')
    # Smooth the Coulomb effort near zero error; retain the adapter's existing target/effort bounds.
    return friction/stiffness*np.tanh(error/.001)


def tracking_integral_increment(error_rad, previous_bias_rad, dt_s, near_goal, feedforward_rad=None):
    import numpy as np
    error=np.asarray(error_rad,dtype=float);previous=np.asarray(previous_bias_rad,dtype=float)
    if error.shape!=previous.shape or not np.isfinite(error).all() or not np.isfinite(previous).all() or not np.isfinite(dt_s) or dt_s<=0:
        raise ValueError('Finite joint vectors and positive control timestep required')
    feedforward=np.zeros_like(error) if feedforward_rad is None else np.asarray(feedforward_rad,dtype=float)
    if feedforward.shape!=error.shape or not np.isfinite(feedforward).all():raise ValueError('Finite matching feedforward required')
    decay=previous*np.expm1(-dt_s/2.5)
    if not near_goal:return decay
    increment=error*dt_s*.50
    demand=error+np.clip(previous+feedforward,-.02,.02)
    # Preserve the existing 8mrad actuator lead bound: unwind rather than
    # accumulating an integral that the bounded actuator cannot realize.
    outward_saturation=(np.abs(demand)>.008)&(increment*demand>0)
    return np.where(outward_saturation,decay,increment)


def tracking_integral_active(target,reference,current,reference_paused):
    import numpy as np
    from scipy.spatial.transform import Rotation
    def near(pose):
        return np.linalg.norm(pose[:3]-current[:3])<.004 and (Rotation.from_quat(pose[[4,5,6,3]])*Rotation.from_quat(current[[4,5,6,3]]).inv()).magnitude()<.03
    # A paused, nearby reference needs static compensation even far from the final goal.
    return near(target) or (reference_paused and near(reference))


def urdf_arm_friction(urdf):
    import math
    import xml.etree.ElementTree as ET
    result={}
    for joint in ET.parse(urdf).getroot().findall('joint'):
        name=joint.get('name')
        if name not in {f'joint{i}' for i in range(1,8)}:continue
        dynamics=joint.find('dynamics')
        friction=float(dynamics.get('friction','0')) if dynamics is not None else 0.
        damping=float(dynamics.get('damping','0')) if dynamics is not None else 0.
        if not all(math.isfinite(x) and x>=0 for x in (friction,damping)):
            raise ValueError('Invalid URDF joint dynamics')
        result[name]={'static_friction_nm':friction,'dynamic_friction_nm':friction,
                      'viscous_friction_nm_s_per_degree':damping*math.pi/180}
    if len(result)!=7:raise ValueError('Seven xArm7 dynamics records required')
    return result


def apply_urdf_effort_friction(stage,urdf,root='/World/Robot'):
    """Preserve URDF Nm friction instead of misusing a unitless PhysX coefficient."""
    from pxr import PhysxSchema,UsdPhysics,Sdf
    from omni.physx.bindings._physx import (JOINT_AXIS_ATTR_STATIC_FRICTION_EFFORT_ANGULAR,
        JOINT_AXIS_ATTR_DYNAMIC_FRICTION_EFFORT_ANGULAR,JOINT_AXIS_ATTR_VISCOUS_FRICTION_COEFFICIENT_ANGULAR)
    records=urdf_arm_friction(urdf)
    for name,record in records.items():
        prim=stage.GetPrimAtPath(root+'/joints/'+name)
        if not prim:raise ValueError('Missing arm joint '+name)
        prim.GetAttribute('physxJoint:jointFriction').Set(0.)
        # This installed schema is codeless: no generated pxr Python class.
        prim.AddAppliedSchema('PhysxJointAxisAPI:angular')
        for attribute,key in ((JOINT_AXIS_ATTR_STATIC_FRICTION_EFFORT_ANGULAR,'static_friction_nm'),
                              (JOINT_AXIS_ATTR_DYNAMIC_FRICTION_EFFORT_ANGULAR,'dynamic_friction_nm'),
                              (JOINT_AXIS_ATTR_VISCOUS_FRICTION_COEFFICIENT_ANGULAR,'viscous_friction_nm_s_per_degree')):
            prim.CreateAttribute(attribute,Sdf.ValueTypeNames.Float).Set(record[key])
    return records


def restore_g1_mimic_joints(stage,root='/World/Robot'):
    """Restore official 1:1 linkage if the URDF importer omitted its mimic API."""
    from pxr import PhysxSchema,UsdPhysics
    reference=stage.GetPrimAtPath(root+'/joints/drive_joint')
    if not reference:raise ValueError('G1 reference drive joint missing')
    restored=[]
    for name in ('left_finger_joint','right_finger_joint','left_inner_knuckle_joint',
                 'right_inner_knuckle_joint','right_outer_knuckle_joint'):
        prim=stage.GetPrimAtPath(root+'/joints/'+name)
        if not prim:raise ValueError('G1 mimic joint missing: '+name)
        # Remove independent rest-position drives that fight the mechanical coupling.
        prim.RemoveAPI(UsdPhysics.DriveAPI,'angular')
        mimic=PhysxSchema.PhysxMimicJointAPI.Apply(prim,'rotX')
        mimic.CreateReferenceJointRel().SetTargets([reference.GetPath()])
        mimic.CreateGearingAttr(-1.)
        mimic.CreateOffsetAttr(0.)
        restored.append(name)
    return restored


def apply_official_adjacent_collision_filters(stage,srdf,root='/World/Robot'):
    """Official adjacent pairs plus two measured interfering linkage pairs only."""
    from pxr import UsdPhysics
    pairs=[]
    for a,b,reason in official_collision_pairs(srdf):
        left,right=stage.GetPrimAtPath(root+'/'+a),stage.GetPrimAtPath(root+'/'+b)
        if not left or not right:continue
        relation=UsdPhysics.FilteredPairsAPI.Apply(left).CreateFilteredPairsRel()
        relation.AddTarget(right.GetPath())
        pairs.append([a,b,reason])
    return pairs


def official_collision_pairs(srdf):
    import xml.etree.ElementTree as ET
    pairs=[]
    for element in ET.parse(srdf).getroot().iter('disable_collisions'):
        a,b=[element.get(key).replace('${prefix}','') for key in ('link1','link2')]
        linkage=frozenset((a,b)) in {
            frozenset(('left_inner_knuckle','left_outer_knuckle')),
            frozenset(('right_inner_knuckle','right_outer_knuckle'))}
        if element.get('reason') not in {'Adjacent','Default'} and not (linkage and element.get('reason')=='Never'):continue
        pairs.append([a,b,element.get('reason')])
    return pairs


def xarm7_config(usd_path, base_position=(0.,0.,.68)):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    return ArticulationCfg(prim_path='/World/Robot',
        spawn=sim_utils.UsdFileCfg(usd_path=str(usd_path),activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,solver_position_iteration_count=192,
                solver_velocity_iteration_count=8)),
        init_state=ArticulationCfg.InitialStateCfg(pos=base_position,
            joint_pos={'joint[1357]':0.,'joint2':-.5,'joint4':1.,'joint6':.5,
                       '.*(gripper|finger|knuckle|drive).*':0.}),
        actuators={'arm':ImplicitActuatorCfg(joint_names_expr=['joint[1-7]'],
                                           stiffness=400.,damping=40.),
                   'gripper':ImplicitActuatorCfg(joint_names_expr=['drive_joint'],
                                               stiffness=100.,damping=10.)})
