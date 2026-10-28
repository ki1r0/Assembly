"""Real RPent Toolkit over the existing direct Isaac action backend."""
import settings
import json,time,os
from pathlib import Path
import numpy as np
from PIL import Image
from rpent.tools import Toolkit,ToolResult,tool
from rpent.session import EnvState
from rpent.memory.manager import MemoryManager
from rpent.dashboard.events import NullDashboardEventSink
from m1_reach.control import snapshot,terminate
from common import write,log

class IsaacToolkit(Toolkit):
    def __init__(self,run):
        self.run=Path(run);self.index=0;self.initial_delivered=False
        self.latest=json.loads((self.run/'ipc/reply_0000.json').read_text())
        super().__init__(dashboard_events=NullDashboardEventSink(),memory=MemoryManager(settings.ROOT/'memory'),state=EnvState(self.run/'episode/rpent'))
        for name in ['observe','back_project','inspect_features','move_to','move_relative','rotate','set_gripper','localize_visual','grasp_visual','insert_visual','view_region','inspect_aperture','settle']:self.add_tool(getattr(self,name))
    def solved(self):return bool(self.latest['native_success'])
    def rpc(self,command):
        if snapshot()['terminal']:return ToolResult(data={'episode_ended':True,'terminal':snapshot()['terminal']})
        write(self.run/'ipc'/f'command_{self.index:04d}.json',command);self.index+=1
        path=self.run/'ipc'/f'reply_{self.index:04d}.json'
        while not path.exists():
            self.raise_if_cancelled()
            if snapshot()['terminal']:return ToolResult(data={'episode_ended':True,'terminal':snapshot()['terminal']})
            if (self.run/'private/backend-failed.json').exists():raise RuntimeError('Isaac backend failed; see private/backend-failed.json')
            time.sleep(.05)
        self.latest=json.loads(path.read_text())
        log('ipc-tools.jsonl',{'command':command,'result':self.latest['result']})
        if self.solved():terminate('native_success')
        return ToolResult(data=self.latest['result'])
    def public_data(self):
        latest=self.latest;features=[]
        for f in latest['features']:
            g=f.get('geometry',{});features.append({k:f[k] for k in ['feature_ref','kind','role','overlay_label'] if k in f}|{'geometry':{k:v for k,v in g.items() if k in {'position_m','axis','axis_kind','normal','extent_m','grasp_width_m','pixel_centroid_xy','image_ellipse_candidate_px','image_circle_candidate_px','pixel_bbox_xyxy','model_image_pixel_centroid_xy'}},'uncertainty':{k:v for k,v in f.get('uncertainty',{}).items() if k in {'position_std_m','axis_std_rad','semantic_identity'}}})
        return {'task':'Pick up the loose bolt, insert its shaft through the upper-right bolt hole in the raised hub-cover annulus (about two o-clock from its large center bore in the fixed task-camera view) into the corresponding casing opening, seat the head on the cover, release, and withdraw the gripper. Complete the assembly task. Call observe first.','observation_id':latest['observation_id'],'robot_sensors':{k:v for k,v in latest['sensors'].items() if k in {'tcp_pose','joint_positions','gripper_width_m','grasped','finger_contacts','finger_contact_force_n','wrench','camera_intrinsics','camera_extrinsics','camera_pixel_convention'}},'last_result':latest['result'],'native_success':latest['native_success'],'wrist_camera':latest.get('wrist_camera'),'features':features,'image_order':['Fixed external camera raw RGB','Same camera and frame with geometric feature overlay; not a wrist camera']}
    def get_env_state(self,*,command,result,elapsed_s):
        data={'tool_result':result,**self.public_data()}
        if self.latest.get('wrist_camera',{}).get('path'):data['image_order'].append('Real hand-mounted wrist camera RGB')
        with self.state.record_step(state=data,terminated=self.solved(),command=command,result=result,elapsed_s=elapsed_s):
            self.state.save('agentview.png',np.array(Image.open(self.latest['rgb_path']).convert('RGB')))
            self.state.save('overlay.png',np.array(Image.open(self.latest['overlay_path']).convert('RGB')))
            self.state.save('observation.json',data)
        log('rpent-tools.jsonl',{'command':command,'result':result,'elapsed_s':elapsed_s,'observation_id':data['observation_id']})
        images=[Path(self.latest[k]).read_bytes() for k in ['rgb_path','overlay_path']]
        if self.latest.get('wrist_camera',{}).get('path'):
            images.append(Path(self.latest['wrist_camera']['path']).read_bytes())
        return ToolResult(data=data,images=images)
    @tool
    def observe(self)->ToolResult:
        """Read current RGB-D and robot state. First call delivers reset view; subsequent calls acquire a new frame."""
        if not self.initial_delivered:self.initial_delivered=True;return ToolResult(data={'status':'initial_observation'})
        return self.rpc({'action':'observe'})
    @tool(readonly=True)
    def back_project(self,pixel_xy:list[float])->ToolResult:
        """Measure a raw RGB pixel [x,y] using current depth and calibration; returns a world point, not a grasp pose."""
        return self.rpc({'action':'back_project','pixel_xy':pixel_xy})
    @tool(readonly=True)
    def inspect_features(self,feature_refs:list[str])->ToolResult:
        """Read full geometry for one to six current feature references; no motion or target selection."""
        return self.rpc({'action':'inspect_features','feature_refs':feature_refs})
    @tool
    def move_to(self,position_m:list[float],quaternion_wxyz:list[float]|None=None)->ToolResult:
        """Direct world TCP target in metres; optional unit quaternion [w,x,y,z]. Retain current orientation if omitted. Returns measured outcome; may time out after real motion."""
        c={'action':'move_to','position_m':position_m}
        if quaternion_wxyz is not None:c['quaternion_wxyz']=quaternion_wxyz
        return self.rpc(c)
    @tool
    def move_relative(self,translation_m:list[float],rotation_rad:list[float]|None=None)->ToolResult:
        """Direct world-axis TCP translation [dx,dy,dz], optional world rotation vector; no inferred route."""
        c={'action':'move_relative','translation_m':translation_m}
        if rotation_rad is not None:c['rotation_rad']=rotation_rad
        return self.rpc(c)
    @tool
    def rotate(self,rotation_rad:list[float])->ToolResult:
        """Rotate TCP using a world-axis rotation vector in radians; retain current position."""
        return self.rpc({'action':'rotate','rotation_rad':rotation_rad})
    @tool
    def set_gripper(self,width_m:float)->ToolResult:
        """Command jaw opening in metres, up to 0.08m; check the embodiment profile for minimum gap and physical linkage semantics. No grasp-candidate, bilateral-contact or release prerequisite. Commanding closure does not prove a grasp."""
        return self.rpc({'action':'set_gripper','width_m':width_m})

    @tool(readonly=True)
    def view_region(self,pixel_xy:list[float],size_px:int=160)->ToolResult:
        """Magnify a selected region of the current raw main image 3x. Returns raw crop origin and scale; no movement or target selection."""
        if len(pixel_xy)!=2 or not all(np.isfinite(pixel_xy)) or not 40<=size_px<=400:raise ValueError('Finite [x,y] and size_px 40..400 required')
        im=Image.open(self.latest['rgb_path']).convert('RGB');x,y=map(round,pixel_xy);left=max(0,x-size_px//2);top=max(0,y-size_px//2);right=min(im.width,left+size_px);bottom=min(im.height,top+size_px)
        if left>=right or top>=bottom:raise ValueError('Crop outside image')
        import io
        crop=im.crop((left,top,right,bottom));out=io.BytesIO();crop.resize((crop.width*3,crop.height*3)).save(out,format='PNG')
        return ToolResult(data={'source_observation_id':self.latest['observation_id'],'raw_origin_xy':[left,top],'scale':3,'pixel_mapping':'raw_pixel = origin + crop_pixel/3'},images=[out.getvalue()])
    @tool(readonly=True)
    def localize_visual(self,pixel_xy:list[float])->ToolResult:
        """Measure the selected small above-table object's top-face center from current RGB-D. Returns observed bounds and provenance, not private simulator poses."""
        return self.rpc({'action':'localize_visual','pixel_xy':pixel_xy})
    @tool
    def grasp_visual(self,pixel_xy:list[float])->ToolResult:
        """Grasp the model-selected upright small part using current depth. Computes top-face center, approaches with calibrated downward Panda orientation, closes fingers, checks contact and lifts. Uses no predefined scene positions; returns stages and measured contact evidence."""
        return self.rpc({'action':'grasp_visual','pixel_xy':pixel_xy})
    @tool
    def insert_visual(self,feature_ref:str)->ToolResult:
        """Insert a visually grasped upright 44mm bolt into the model-selected current aperture. Uses measured rim center and grasp geometry, then compliant descent/local spiral with measured wrench feedback. Returns promptly on stable contact; does not release automatically. You decide release/adjustment; only native success proves completion. No private target pose is used."""
        return self.rpc({'action':'insert_visual','feature_ref':feature_ref})

    @tool(readonly=True)
    def inspect_aperture(self,feature_ref:str)->ToolResult:
        """Refine the selected current aperture using its visible rim and local RGB-D top plane. Returns center, observed opening diameter and fit residual; estimates may include a counterbore and are not hidden dimensions."""
        return self.rpc({'action':'inspect_aperture','feature_ref':feature_ref})

    @tool
    def settle(self,duration_s:float=2.0)->ToolResult:
        """Hold measured TCP pose for 0.05..5 simulation seconds, advancing physical settling and independent success evaluation. No target/object pose is read."""
        return self.rpc({'action':'settle','duration_s':duration_s})
