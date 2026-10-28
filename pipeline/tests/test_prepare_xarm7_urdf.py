from pathlib import Path
import pytest
from tools.prepare_xarm7_urdf import generate


@pytest.mark.parametrize('gripper', ['none','ufactory-g1'])
def test_official_model_expands_without_real_hardware_plugins(tmp_path,gripper):
    pytest.importorskip('xacro')
    source=Path(__file__).resolve().parents[1]/'third_party/xarm_ros2'
    if not source.exists():
        pytest.skip('Official xArm description checkout is not installed')
    output=tmp_path/'robot.urdf'
    robot=generate(source,output,gripper)
    joints={x.attrib['name']:x for x in robot.findall('joint')}
    assert all(joints[f'joint{i}'].attrib['type']=='revolute' for i in range(1,8))
    assert not robot.findall('ros2_control')
    assert not robot.findall('gazebo')
    assert all(Path(x.attrib['filename']).is_file() for x in robot.iter('mesh'))
    assert any('gripper' in x.attrib['name'] for x in robot.findall('link')) == (gripper!='none')
    with pytest.raises(FileExistsError):
        generate(source,output,gripper)
