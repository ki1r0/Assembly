"""Same RPent episode loop with robot-neutral Cartesian and dual-arm tools."""
import json,time
from pathlib import Path
import settings
from toolkit import IsaacToolkit
from rpent.tools import Toolkit,ToolResult,tool
from rpent.memory.manager import MemoryManager
from rpent.session import EnvState
from rpent.dashboard.events import NullDashboardEventSink

class RobotToolkit(IsaacToolkit):
    def __init__(self,run,trace_dir=None):
        self.run=Path(run);self.index=0;self.initial_delivered=False;self.recent_execution_evidence=[]
        self.latest=json.loads((self.run/'ipc/reply_0000.json').read_text())
        Toolkit.__init__(self,dashboard_events=NullDashboardEventSink(),memory=MemoryManager(settings.ROOT/'memory'),state=EnvState(trace_dir or self.run/'episode/rpent'))
        names=['observe','back_project','view_region','view_public_cad','move_to','move_relative','rotate','set_gripper','select_arm','settle','tool_orientation','inspect_pose','plan_grasp','execute_grasp','measure_feature','measure_object','measure_pattern','assemble']
        if self.latest.get('robot')=='r1':names+=['move_both','set_both_grippers']
        for name in names:self.add_tool(getattr(self,name))
    def solved(self):
        from m1_reach.control import snapshot
        terminal=snapshot()['terminal']
        return bool(terminal and terminal.get('source')=='native_success')
    def rpc(self,command):
        source_observation=getattr(self,'latest',{}).get('observation_id')
        result=super().rpc(command)
        from m1_reach.control import snapshot
        terminal=snapshot()['terminal']
        if terminal:
            final=self.run/'episode/public/final-state.json'
            for _ in range(20):
                if final.exists():
                    self.latest=json.loads(final.read_text());break
                time.sleep(.05)
        sensors=self.latest.get('sensors',{})
        outcome=self.latest.get('result',{})
        self.recent_execution_evidence=getattr(self,'recent_execution_evidence',[])
        self.recent_execution_evidence.append({'command':command,'source_observation_id':source_observation,'result_observation_id':self.latest.get('observation_id'),'status':outcome.get('status'),'screening':outcome.get('screening'),'measured':{k:sensors[k] for k in ['tcp_pose','gripper_width_m','finger_contact_force_n','jaw_closing_axis_pressure_n'] if k in sensors}})
        self.recent_execution_evidence=self.recent_execution_evidence[-8:]
        if command.get('action')=='plan_grasp':
            from grasp_feedback import public_grasp_feedback
            return ToolResult(data=public_grasp_feedback(result.data),images=result.images)
        return result
    def public_data(self):
        sensors=self.latest.get('sensors',{})
        return {
            'task':self.latest['task_instruction'],
            'observation_id':self.latest['observation_id'],
            'robot_sensors':{k:v for k,v in sensors.items() if k in {
                'tcp_pose','joint_positions','gripper_width_m','grasped',
                'finger_contacts','finger_contact_force_n','finger_net_force_vectors_world_n','jaw_closing_axis_pressure_n','wrench',
                'camera_intrinsics','camera_extrinsics','camera_pixel_convention',
            }},
            'features':[],
            'geometric_patterns':self.latest.get('geometric_patterns',[]),
            'arms':self.latest['arms'],
            'active_arm':self.latest['active_arm'],
            'recent_execution_evidence':getattr(self,'recent_execution_evidence',[]),
            'evidence_scope':'Current-episode commands and measured outcomes only; contact does not identify the touched object or prove retention. Occlusion leaves object state unknown, not absent.',
            'image_order':['Fixed external raw RGB','Same RGB with coordinate grid and unlabelled observed ellipse IDs; not semantic object identity'],
        }
    @tool
    def select_arm(self,arm:str)->ToolResult:
        """Select left or right arm for subsequent single-arm commands. Panda has only left."""
        return self.rpc({'action':'select_arm','arm':arm})
    @tool
    def move_both(self,left_position_m:list[float],right_position_m:list[float],left_quaternion_wxyz:list[float]|None=None,right_quaternion_wxyz:list[float]|None=None)->ToolResult:
        """Simultaneous physical Cartesian movement of both arms, world metres and wxyz. No object-following constraint."""
        return self.rpc({'action':'move_both','left_position_m':left_position_m,'right_position_m':right_position_m,'left_quaternion_wxyz':left_quaternion_wxyz,'right_quaternion_wxyz':right_quaternion_wxyz})
    @tool
    def set_both_grippers(self,left_width_m:float,right_width_m:float)->ToolResult:
        """Command independent jaw openings, 0..0.08m, without attaching objects."""
        return self.rpc({'action':'set_both_grippers','left_width_m':left_width_m,'right_width_m':right_width_m})
    @tool
    def grasp_at(self,pixel_xy:list[float],quaternion_wxyz:list[float],arm:str='left')->ToolResult:
        """Generic selected visible surface grasp: measure RGB-D at pixel, approach from above, close and lift. You supply arm and orientation. Reports measured TCP/touch only, not object identity or verified retention. Use direct tools for other approaches or dual-arm grasps."""
        return self.rpc({'action':'grasp_at','pixel_xy':pixel_xy,'quaternion_wxyz':quaternion_wxyz,'arm':arm})

    @tool(readonly=True)
    def view_public_cad(self,opposite_side:bool=False)->ToolResult:
        """View the task's named moving-part public CAD in local-axis projections; opposite_side shows reverse faces. Reference geometry only, not a live view, scene pose or target selection. Compare shapes when identity is uncertain."""
        import numpy as np
        from task_profiles import skill_specification
        from public_cad_preview import render_public_cad_preview
        specification=skill_specification()
        with np.load(settings.ROOT/'assets/public_cad_models'/specification['moving_cad_cache']) as model:
            png,metadata=render_public_cad_preview(model['points'],model['normals'],opposite_side)
        return ToolResult(data={'source':'public moving-part CAD, no scene pose','asset':specification['moving_cad_file'],'geometry':metadata},images=[png])
    @tool(readonly=True)
    def inspect_pose(self,position_m:list[float],quaternion_wxyz:list[float],gripper_width_m:float|None=None)->ToolResult:
        """Inspect your proposed TCP pose before motion: measured-model IK plus public hand/table clearance when available. R1 proposed opening is a nominal joint-stroke model, not calibrated pad gap. Negative clearance predicts hand/table overlap. No path, grasp or other-obstacle guarantee; no motion."""
        command={'action':'inspect_pose','position_m':position_m,'quaternion_wxyz':quaternion_wxyz}
        if gripper_width_m is not None:command['gripper_width_m']=gripper_width_m
        return self.rpc(command)
    @tool(readonly=True)
    def tool_orientation(self,approach_axis_world:list[float],closing_axis_world:list[float])->ToolResult:
        """Convert your chosen world finger-approach (+Z) and jaw-closing (+Y) directions into a unit wxyz quaternion. Pure arithmetic; no motion, target choice, IK or collision check. Returns actual orthogonal axes."""
        from assembly_skills import tool_orientation
        return ToolResult(data=tool_orientation(approach_axis_world,closing_axis_world))
    @tool(readonly=True)
    def plan_grasp(self,pixel_xy:list[float],bbox_xyxy:list[float])->ToolResult:
        """Generate 6D grasp poses for a model-selected current raw-image object/rim ROI. Returns screening counts for generation, contact-band, finger-body collision, support-plane, IK and visible-obstacle rejection. Read score_semantics: G1 score is motion cost only, not assembly suitability or retention probability. Compare assembly_compatibility separately. When full-opening release clearances are supplied, a clearance larger than axial_protrusion_m means the shaft cannot enter before the safe release plane; a low motion score does not fix that. Choose the directed-axis relation from public seating-face geometry, never just the smaller clearance. For a wide part select a grippable rim, not its entire diameter. No motion, identity or hidden-collision guarantee; manual moves do not inherit screening."""
        return self.rpc({'action':'plan_grasp','pixel_xy':pixel_xy,'bbox_xyxy':bbox_xyxy})
    @tool
    def execute_grasp(self,candidate_id:str,grip_force_n:float|None=None)->ToolResult:
        """Execute a generated compressive closing grasp, regulate jaw pressure, then lift; internal expansion is not implemented here. Optional grip_force_n is 0.3..35N closing-axis pressure per finger (tangential load excluded); default uses public mass, registered public-CAD COM lever and jaw width, assumed friction 0.5 and load margin. Reduce pressure if squeezing knocks the part away; if it slips while both jaws touch, increase gradually. A bounded tactile centering adjustment may occur before lift. Contact still does not prove retention; inspect images."""
        return self.rpc({'action':'execute_grasp','candidate_id':candidate_id,'grip_force_n':grip_force_n})
    @tool(readonly=True)
    def measure_feature(self,pixel_xy:list[float],radius_px:int=24)->ToolResult:
        """Measure a selected circular opening or plane from current RGB-D. For an opening supply its centre seed, not a point on its rim; RGB edges and outside depth locate the mouth surface. The initial ROI may expand up to radius 180px around your seed. Check circle_rms/source: a plane-ray fallback is not a verified hole centre. Returns no semantic identity."""
        return self.rpc({'action':'measure_feature','pixel_xy':pixel_xy,'radius_px':radius_px})
    @tool
    def assemble(self,moving_feature_id:str,target_feature_id:str,seat_offset_m:float=0.,search_radius_m:float=.001,axis_alignment:str='nearest')->ToolResult:
        """Align observed features, approach and perform bounded contact search. axis_alignment=nearest treats axes as unoriented lines and preserves the closest orientation; opposite axis signs do not force a 180-degree flip. Choose parallel or antiparallel explicitly only when evidence establishes the required directed-axis relation. Target axis points outward along approach/retreat; alignment mode never reverses retreat direction. Use a fresh measured feature or an execute_grasp held_axis_features feature_id. Held features are rigid-grasp estimates; check images and choose the intended seating face. seat_offset is visible/CAD feature-to-face distance, not a world pose (0..0.1m); search_radius is 0..0.003m. Whole-hand clearance is recomputed at the aligned pose. Does not release or claim success; you decide release and withdraw."""
        return self.rpc({'action':'assemble','moving_feature_id':moving_feature_id,'target_feature_id':target_feature_id,'seat_offset_m':seat_offset_m,'search_radius_m':search_radius_m,'axis_alignment':axis_alignment})

    @tool(readonly=True)
    def measure_object(self,pixel_xy:list[float],bbox_xyxy:list[float])->ToolResult:
        """Measure a selected visible RGB-D component's centroid, bounds and point covariance. Select a visible surface pixel inside its raw-image box. Occluded geometry is unknown; the result is not semantic identity, a full solid, or proof of physical support."""
        return self.rpc({'action':'measure_object','pixel_xy':pixel_xy,'bbox_xyxy':bbox_xyxy})

    @tool(readonly=True)
    def measure_pattern(self,center_pixel_xy:list[float],bbox_xyxy:list[float])->ToolResult:
        """Measure dark circular candidates and their image-plane clock hours relative to your chosen centre. 12 up, 3 right. Use for relative hole-pattern reasoning; candidates may include lettering and need image inspection. No target identity or evaluator coordinate is supplied."""
        return self.rpc({'action':'measure_pattern','center_pixel_xy':center_pixel_xy,'bbox_xyxy':bbox_xyxy})
