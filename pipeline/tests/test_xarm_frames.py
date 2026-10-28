import numpy as np
import pytest
from xarm_frames import sdk_pose_to_tcp_pose, tcp_pose_to_sdk_pose, transform_wrench


@pytest.mark.parametrize('radians', [False, True])
def test_sdk_units_and_pose_roundtrip(radians):
    angles = [20., -30., 45.]
    if radians:
        angles = np.deg2rad(angles).tolist()
    source = [300., -200., 450., *angles]
    pose = sdk_pose_to_tcp_pose(source, is_radian=radians)
    assert np.allclose(pose[:3], [.3, -.2, .45])
    assert np.allclose(tcp_pose_to_sdk_pose(pose, is_radian=radians), source)


def test_wrench_rotation_and_lever_arm_have_correct_sign():
    matrix = np.array([[0., -1., 0., .1], [1., 0., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    result = transform_wrench([2., 0., 0., 0., 0., 0.], matrix)
    assert np.allclose(result, [0., 2., 0., 0., 0., .2])
    assert np.allclose(transform_wrench(result, np.linalg.inv(matrix)), [2., 0., 0., 0., 0., 0.])


def test_unknown_units_invalid_pose_and_nonrigid_frames_are_rejected():
    with pytest.raises(ValueError):
        sdk_pose_to_tcp_pose([0.] * 6, is_radian=None)
    with pytest.raises(ValueError):
        tcp_pose_to_sdk_pose([0.] * 7, is_radian=True)
    with pytest.raises(ValueError):
        transform_wrench([0.] * 6, np.diag([2., 1., 1., 1.]))
    with pytest.raises(ValueError):
        sdk_pose_to_tcp_pose([float('nan')] * 6, is_radian=True)
