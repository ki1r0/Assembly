"""Part-neutral jaw pressure feedback; physical effort limits remain in the robot."""
import numpy as np

def jaw_reference_step(reference,width,force_n,target_force_n,dt):
    if target_force_n is None:
        rate=np.clip((width-reference)/dt,-.005,.005)
    elif force_n<.05:
        rate=-.005
    else:
        rate=np.clip((force_n-target_force_n)*.0005,-.001,.001)
    return float(np.clip(reference+rate*dt,0.,.08))

def mass_based_grip_force(mass_kg,com_offset_world=None,closing_axis_world=None,jaw_width_m=.04,minimum_force_n=2.):
    # Declared mass + observed COM lever; friction 0.5, 1.3x margin and 6mm pad moment arm.
    if not np.isfinite(minimum_force_n) or not .3<=minimum_force_n<=35.:raise ValueError('Automatic minimum force must be within 0.3..35N')
    weight=mass_kg*9.81;tangential=weight/2;torsional=0.
    if com_offset_world is not None:
        axis=np.asarray(closing_axis_world,float);axis=axis/np.linalg.norm(axis)
        moment=np.cross(np.asarray(com_offset_world,float),[0.,0.,-weight])
        axial=float(moment@axis);torsional=abs(axial)/(2*.5*.006)
        tangential+=np.linalg.norm(moment-axis*axial)/max(.005,jaw_width_m)
    return float(np.clip(1.3*max(tangential/.5,torsional),minimum_force_n,35.))


def centering_increment(forces,target_force,closing_axis,dt,remaining_m,minimum_fraction=.7):
    left,right=map(float,forces)
    if min(left,right)>=minimum_fraction*target_force or max(left,right)<.5*target_force:return np.zeros(3)
    # Move toward the contacting jaw: unload that side and bring the other jaw in.
    return np.asarray(closing_axis)*np.sign(left-right)*min(.0002*dt,max(0.,remaining_m))


def bumpless_open_reference(reference,measured_width,commanded_width):
    """Remove force-mode preload when requesting an opening, not physical state."""
    return float(measured_width) if commanded_width>measured_width+1e-6 else float(reference)


def projected_jaw_pressure(forces_world,closing_axis_world):
    """Closing-axis component, excluding gravity/tangential contact forces.

    Still unfiltered all-object contact: not semantic identity or grasp proof.
    """
    forces=np.asarray(forces_world,float);axis=np.asarray(closing_axis_world,float)
    if forces.shape!=(2,3) or axis.shape!=(3,) or not np.isfinite(forces).all() or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-9:raise ValueError('Two finite finger forces and nonzero measured closing axis required')
    return np.abs(forces@(axis/np.linalg.norm(axis))).tolist()
