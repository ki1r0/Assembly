"""ERT-mediated access to the exact same robot capabilities used by ours."""
import json

from dewm import WorldMemory
from generic_toolkit import RobotToolkit
from rpent.tools import ToolResult, tool


DEWM_SYSTEM = """
This session uses a public-observation VLM-DEWM task adaptation, not a stateless
planner. Maintain environment memory with register_entity after measure_object
(visible component envelope) or measure_feature (measured rim/plane landmark).
Labels and reliability High/Medium/Low/Bad are your semantic judgement, not GT.
New landmarks are tentative until confirmed in two distinct observations.
Only measured geometry is stored; a visible envelope is not a full object
pose. Optional association_scores give your semantic similarity to remembered
entity IDs; measured geometry also gates matches. Reobserve after movement or occlusion; do not reuse old world
coordinates as current evidence. Uncertain is not absent.

All physical actions go through execute_ert. The five ERT fields are:
action_proposal: {"action": "move_relative", "translation_m": [0,0,0.01]},
world_belief: [{"pred":"Near", "args":["entity-0","entity-1"]}],
causal_assump: a list of expected post-action relations with the same structure,
confidence: a number in [0,1], fallback: your evidence-based recovery strategy.
Use only a shared tool name and its exact parameters in action_proposal. Its
parameters and safety semantics are unchanged from the capability list below.
Empty relation lists are allowed for initial sensing/motion, but provide no
verified semantic progress. Near means within 5cm; Coaxial means within 5mm and
5 degrees of an unoriented measured axis. These are belief checks, NOT seating
criteria or collision clearance. On/Inside/Holding cannot be verified from point
landmarks, so never claim them as supported facts. Gripper contact alone does
not identify or prove retention of an object.

After execute_ert, acquire fresh observe/measure_feature evidence, register the
reobserved landmarks, then call verify_action. Failed or unknown predictions are
not committed. Read the deterministic discrepancy and choose recovery yourself.
The world memory is not an evaluator and cannot certify task completion. Existing
independent success reporting and all physical safety protections still apply.
"""


MOTION_TOOLS = ('move_to', 'move_relative', 'rotate', 'set_gripper', 'select_arm',
                'settle', 'execute_grasp', 'assemble', 'move_both', 'set_both_grippers')


class DEWMToolkit(RobotToolkit):
    def __init__(self, run):
        self.world = WorldMemory()
        self.measured_features = {}
        self.motion_epoch = 0
        super().__init__(run)
        self.world.task_instruction=self.latest['task_instruction']
        self.motion_tools = {name: self._tools.pop(name) for name in MOTION_TOOLS if name in self._tools}
        for name in ('register_entity', 'execute_ert', 'verify_action'):
            self.add_tool(getattr(self, name))

    def rpc(self, command):
        source_observation_id=self.latest['observation_id']
        result = super().rpc(command)
        payload = result.data
        if command['action'] == 'measure_feature' and payload.get('status') == 'measured' and payload.get('feature_id'):
            self.measured_features[payload['feature_id']] = {
                'observation_id': source_observation_id,
                'motion_epoch': self.motion_epoch,
                'position_m': payload['position_m'], 'axis_world': payload.get('axis_world'),
                # An explicit conservative proxy, not a calibrated pose covariance.
                'position_std_m': max(.005, float(payload.get('plane_rms_m') or 0.), float(payload.get('circle_rms_m') or 0.)),
            }
        if command['action']=='measure_object' and payload.get('status')=='object_measured':
            import numpy as np
            key='object-geometry-'+str(self.index)
            self.measured_features[key]={
                **{k:payload[k] for k in ('position_m','bounds_world_m','position_covariance_m2')},
                'position_std_m':float(np.sqrt(np.trace(payload['position_covariance_m2'])/3)),
                'motion_epoch':self.motion_epoch,'observation_id':source_observation_id}
            result.data['geometry_id']=key
        return result

    def public_data(self):
        return super().public_data() | {'dewm': self.world.context()}

    def capability_context(self):
        return json.dumps({name: {'description': item.description, 'parameters': item.input_schema}
                           for name, item in self.motion_tools.items()})

    def save_world(self):
        from common import write
        write(self.run/'episode/dewm-world-memory.json', self.world.context())

    @tool
    def register_entity(self, label: str, feature_id: str, reliability: str, association_scores: dict[str,float]|None=None)->ToolResult:
        """Associate current measure_feature/measure_object geometry with a semantic entity. Use returned feature_id or geometry_id. Reliability is High/Medium/Low/Bad. Optional entity-ID similarity scores in [0,1] resolve semantic ambiguity; geometry cannot be supplied or invented."""
        measured = self.measured_features.get(feature_id)
        if measured is None or measured['motion_epoch'] != self.motion_epoch:
            return ToolResult(data={'error': 'A current measure_feature result is required; predicted held features are not independent observations'})
        ids = self.world.update([measured | {'label': label, 'feature_id': feature_id, 'reliability': reliability,'association_scores':association_scores or {}}],
                                measured['observation_id'])
        self.save_world()
        return ToolResult(data={'entity_ids': ids, 'memory': self.world.context()})

    @tool
    def execute_ert(self, ert: dict)->ToolResult:
        """Validate an ERT then invoke its shared capability, with exactly the original argument schema and physical safety. Requires post-action verification before another proposal."""
        try:
            command = ert.get('action_proposal', {})
            declaration = self.motion_tools.get(command.get('action'))
            if declaration is None:
                raise ValueError('Unknown shared action capability')
            arguments = {key: value for key, value in command.items() if key != 'action'}
            if set(arguments)-set(declaration.args_schema.model_fields):
                raise ValueError('Unexpected shared action parameter')
            validated = declaration.args_schema.model_validate(arguments).model_dump()
            self.world.propose(ert, self.latest['observation_id'], self.motion_tools)
        except (ValueError, TypeError, KeyError) as error:
            return ToolResult(data={'error': str(error), 'executed': False})
        try:
            result = declaration(**validated)
        except Exception as error:
            self.motion_epoch += 1
            self.world.executed({'error': str(error)}, self.latest['observation_id'])
            self.save_world()
            raise
        self.motion_epoch += 1
        self.world.executed(result.data, self.latest['observation_id'])
        self.save_world()
        return ToolResult(data=result.data | {'ert_requires_post_observation_verification': True}, images=result.images)

    @tool
    def verify_action(self)->ToolResult:
        """Compare ERT expected relations with fresh measured world memory; unknown evidence is not success. Returns a deterministic discrepancy for your recovery decision."""
        try:
            result = self.world.verify(self.latest['observation_id'])
        except ValueError as error:
            return ToolResult(data={'error': str(error), 'verified': False})
        self.save_world()
        from common import log
        log('dewm-verification.jsonl', result)
        return ToolResult(data=result)
