"""Measured hand-mounted RGB camera, independent of learned policies."""
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path

def install_wrist_camera():
    from runtime.bolt_harness.env import BoltInsertionEnv
    from isaaclab.sensors import Camera,CameraCfg
    import isaaclab.sim as sim_utils
    original=BoltInsertionEnv._setup_scene
    def setup(env):
        original(env)
        camera=Camera(CameraCfg(prim_path='/World/envs/env_.*/WristCamera',update_period=0.,update_latest_camera_pose=True,height=224,width=224,data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=16.,horizontal_aperture=20.955,clipping_range=(.01,3.)),offset=CameraCfg.OffsetCfg(pos=(0.,0.,0.),rot=(1.,0.,0.,0.),convention='ros')))
        env.scene.sensors['wrist_camera']=camera
    BoltInsertionEnv._setup_scene=setup

def attach_wrist_capture(runtime,run):
    import cv2
    from common import write
    latest={}
    def capture(rgb,timestamp):
        camera=runtime.env.scene.sensors['wrist_camera']
        import torch
        robot=runtime.env._robot;index=robot.body_names.index('panda_hand')
        body_position=robot.data.body_pos_w[0,index].detach().cpu().numpy()
        body_quaternion=robot.data.body_quat_w[0,index].detach().cpu().numpy()
        rotation=Rotation.from_quat(body_quaternion[[1,2,3,0]])
        position=body_position+rotation.apply([0.,-.075,.075])
        camera.set_world_poses(torch.as_tensor(position[None],device=runtime.env.device,dtype=torch.float32),torch.as_tensor(body_quaternion[None],device=runtime.env.device,dtype=torch.float32),convention='ros')
        runtime.env.sim.render()
        camera.update(0.,force_recompute=True)
        if np.linalg.norm(camera.data.pos_w[0].detach().cpu().numpy()-position)>1e-4:raise RuntimeError('Wrist camera extrinsics do not track measured hand FK: expected='+str(position.tolist())+' actual='+str(camera.data.pos_w[0].detach().cpu().tolist()))
        pixels=camera.data.output['rgb'][0,:,:,:3].detach().cpu().numpy().astype(np.uint8)
        path=Path(run)/'episode/wrist'/f'wrist_{runtime._last_frame.frame_id:08d}.png';path.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(path),pixels[:,:,::-1]):raise OSError('Unable to save actual wrist frame')
        latest.update(path=str(path),timestamp_s=float(timestamp),position_world_m=camera.data.pos_w[0].detach().cpu().tolist(),quaternion_world_ros_wxyz=camera.data.quat_w_ros[0].detach().cpu().tolist(),hand_position_world_m=body_position.tolist(),hand_quaternion_world_wxyz=body_quaternion.tolist(),hand_local_camera_offset_m=[0.,-.075,.075],intrinsics=camera.data.intrinsic_matrices[0].detach().cpu().tolist(),source='real camera follows measured hand FK at each acquisition; main camera unchanged')
        write(path.with_suffix('.json'),latest)
    unsubscribe=runtime.subscribe_rgb_frames(capture)
    return latest,unsubscribe
