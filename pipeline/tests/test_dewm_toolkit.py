import settings
from dewm import WorldMemory
from dewm_toolkit import DEWMToolkit
from rpent.tools import ToolResult, tool
from generic_toolkit import RobotToolkit


def fake_toolkit():
    toolkit = object.__new__(DEWMToolkit)
    toolkit.world = WorldMemory()
    toolkit.measured_features = {}
    toolkit.motion_epoch = 0
    toolkit.latest = {'observation_id': 'frame-1'}
    toolkit.save_world = lambda: None
    calls = []
    @tool
    def move_relative(translation_m: list[float])->ToolResult:
        """Test the shared motion delegation without any simulator or GT."""
        calls.append(translation_m)
        toolkit.latest = {'observation_id': 'frame-2'}
        return ToolResult(data={'status': 'moved'})
    toolkit.motion_tools = {'move_relative': move_relative}
    return toolkit, calls


def proposal():
    return {'action_proposal': {'action': 'move_relative', 'translation_m': [0, 0, .01]},
            'world_belief': [], 'causal_assump': [], 'confidence': .8,
            'fallback': 'Observe again before another action'}


def test_ert_delegates_original_tool_and_rejects_hidden_parameters():
    toolkit, calls = fake_toolkit()
    bad = proposal()
    bad['action_proposal']['private_target_pose'] = [1, 2, 3]
    assert toolkit.execute_ert(bad).is_error
    assert calls == []
    assert toolkit.world.pending is None
    result = toolkit.execute_ert(proposal())
    assert result.data['status'] == 'moved'
    assert calls == [[0, 0, .01]]
    assert toolkit.motion_epoch == 1
    assert toolkit.execute_ert(proposal()).is_error
    assert len(calls) == 1


def test_register_requires_measured_not_model_provided_or_pre_motion_geometry():
    toolkit, calls = fake_toolkit()
    assert toolkit.register_entity('ring', 'invented', 'High').is_error
    toolkit.measured_features['observed'] = {
        'position_m': [0, 0, 0], 'position_std_m': .005,
        'observation_id': 'frame-1', 'motion_epoch': 0}
    assert not toolkit.register_entity('ring', 'observed', 'High').is_error
    toolkit.execute_ert(proposal())
    assert toolkit.register_entity('ring', 'observed', 'High').is_error
    assert toolkit.world.entities['entity-0']['lifecycle'] == 'Uncertain'


def test_object_geometry_keeps_the_acquisition_frame_not_the_later_reply(monkeypatch):
    toolkit,_=fake_toolkit()
    toolkit.index=1
    def rpc(self,command):
        self.latest={'observation_id':'later-reply-frame'}
        return ToolResult(data={'status':'object_measured','position_m':[0,0,0],
            'bounds_world_m':[[-.01,-.01,-.01],[.01,.01,.01]],
            'position_covariance_m2':[[.0001,0,0],[0,.0001,0],[0,0,.0001]]})
    monkeypatch.setattr(RobotToolkit,'rpc',rpc)
    result=toolkit.rpc({'action':'measure_object'})
    geometry=toolkit.measured_features[result.data['geometry_id']]
    assert geometry['observation_id']=='frame-1'
    assert geometry['motion_epoch']==0
