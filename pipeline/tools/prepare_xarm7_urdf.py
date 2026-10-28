"""Expand the official xArm7 model for offline simulation, without ROS drivers."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def generate(source, output, gripper='none'):
    import xacro
    import xacro.substitution_args

    source = Path(source).resolve()
    description = source/'xarm_description'
    original = xacro.substitution_args._find
    def find_package(resolved, expression, arguments, context):
        if arguments == ['xarm_description']:
            return resolved.replace('$('+expression+')',str(description))
        raise ValueError('Unexpected ROS package dependency: '+str(arguments))
    xacro.substitution_args._find = find_package
    try:
        document = xacro.process_file(str(description/'urdf/xarm_device.urdf.xacro'), mappings={
            'dof':'7', 'robot_type':'xarm', 'add_gripper':str(gripper=='ufactory-g1').lower(),
            'gripper_version':'G1', 'add_realsense_d435i':'false',
            'ros2_control_plugin':'mock_components/GenericSystem'})
    finally:
        xacro.substitution_args._find = original
    robot = ET.fromstring(document.toxml())
    # Offline Isaac imports only: never include a hardware/controller plugin.
    for element in list(robot):
        if element.tag in {'ros2_control', 'gazebo', 'transmission'}:
            robot.remove(element)
    for mesh in robot.iter('mesh'):
        name = mesh.attrib['filename']
        prefix = 'package://xarm_description/'
        if name.startswith(prefix):
            name = str(description/name[len(prefix):])
        if not Path(name.removeprefix('file://')).is_file():
            raise FileNotFoundError(name)
        mesh.set('filename', name)
    joints = {element.attrib['name']:element for element in robot.findall('joint')}
    for index in range(1,8):
        joint = joints['joint'+str(index)]
        if joint.attrib['type']!='revolute' or joint.find('limit') is None:
            raise ValueError('Missing arm joint type/limits')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Do not overwrite an existing robot model: '+str(output))
    output.parent.mkdir(parents=True,exist_ok=True)
    ET.indent(robot)
    ET.ElementTree(robot).write(output,encoding='utf-8',xml_declaration=True)
    return robot


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'third_party/xarm_ros2')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gripper',choices=['none','ufactory-g1'],default='none',
                        help='G1 is a simulation assumption until the actual gripper is confirmed')
    args=parser.parse_args()
    robot=generate(args.source,args.output,args.gripper)
    print(f'Offline model: {args.output}; links={len(robot.findall("link"))}; gripper={args.gripper}')
