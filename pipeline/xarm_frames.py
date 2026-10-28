"""Pure unit/frame conversions; no robot connection or motion commands."""
import numpy as np
from scipy.spatial.transform import Rotation


def finite_vector(value, size):
    value = np.asarray(value, dtype=float)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f'Expected {size} finite values')
    return value


def sdk_pose_to_tcp_pose(pose_mm_rpy, *, is_radian):
    """xArm base-frame [mm, roll,pitch,yaw] to [metres,w,x,y,z].

    The SDK pose must already refer to the configured TCP; this does not
    compensate an unknown flange, gripper, or force-sensor offset.
    """
    pose = finite_vector(pose_mm_rpy, 6)
    if type(is_radian) is not bool:
        raise ValueError('Explicit SDK angle-unit boolean required')
    q = Rotation.from_euler('xyz', pose[3:], degrees=not is_radian).as_quat()
    return np.r_[pose[:3] / 1000., q[[3, 0, 1, 2]]].tolist()


def tcp_pose_to_sdk_pose(pose_m_wxyz, *, is_radian):
    """Inverse conversion for an already calibrated SDK TCP, not a motion API."""
    pose = finite_vector(pose_m_wxyz, 7)
    if type(is_radian) is not bool or np.linalg.norm(pose[3:]) < 1e-12:
        raise ValueError('Explicit angle units and nonzero quaternion required')
    q = pose[3:] / np.linalg.norm(pose[3:])
    rpy = Rotation.from_quat(q[[1, 2, 3, 0]]).as_euler('xyz', degrees=not is_radian)
    return np.r_[pose[:3] * 1000., rpy].tolist()


def transform_wrench(wrench_source_n_nm, target_from_source):
    """Transform [Fx,Fy,Fz,Tx,Ty,Tz] to the target frame and target origin.

    target_from_source maps source-coordinate points into target coordinates.
    Input must already be calibrated/tared in N and Nm. This does not remove
    tool gravity, bias, or robot acceleration and must not replace raw protection.
    """
    wrench = finite_vector(wrench_source_n_nm, 6)
    matrix = np.asarray(target_from_source, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Finite 4x4 frame transform required')
    rotation = matrix[:3, :3]
    if not np.allclose(matrix[3], [0, 0, 0, 1]) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-7):
        raise ValueError('Rigid right-handed frame transform required')
    force = rotation @ wrench[:3]
    torque = rotation @ wrench[3:] + np.cross(matrix[:3, 3], force)
    return np.r_[force, torque].tolist()
