"""Convert the official offline URDF using the installed Isaac Lab importer."""
import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--urdf',required=True)
parser.add_argument('--output-dir',required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app
try:
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    cfg=UrdfConverterCfg(
        asset_path=str(Path(args.urdf).resolve()),
        usd_dir=str(Path(args.output_dir).resolve()),usd_file_name='xarm7_g1.usd',
        fix_base=True,merge_fixed_joints=False,make_instanceable=False,
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=400.,damping=40.)))
    converter=UrdfConverter(cfg)
    print('Converted xArm7 USD:',converter.usd_path,flush=True)
finally:
    app.close()
