import pytest
from dewm import WorldMemory, envelope_iou
import numpy as np


def observation(feature, x, label='ring', reliability='High'):
    return {'feature_id': feature, 'label': label, 'reliability': reliability,
            'position_m': [x, 0, 0], 'position_std_m': .01, 'axis_world': [0, 0, 1]}


def ert():
    relation = {'pred': 'Near', 'args': ['entity-0', 'entity-1']}
    return {'action_proposal': {'action': 'move_relative', 'translation_m': [0, 0, .01]},
            'world_belief': [relation], 'causal_assump': [relation],
            'confidence': .8, 'fallback': 'Acquire another public observation and diagnose'}


def test_association_is_one_to_one_and_same_frame_cannot_shrink_uncertainty():
    memory = WorldMemory()
    memory.update([observation('a', 0), observation('b', .03)], 'frame-1')
    original = memory.context()
    assert memory.update([observation('a', 0)], 'frame-1') == []
    assert memory.context() == original
    assert memory.update([observation('c', .001), observation('d', .029)], 'frame-2') == ['entity-0', 'entity-1']
    assert len(memory.entities) == 2
    assert memory.entities['entity-0']['variance_m2'] < .0001


def test_bad_measurement_has_paper_near_freeze_gain_and_unmatched_entries_are_tentative():
    memory = WorldMemory()
    assert memory.update([observation('bad', 0, reliability='Bad')], 'f1') == ['entity-0']
    assert memory.entities['entity-0']['variance_m2'] == pytest.approx(.01)
    assert memory.entities['entity-0']['lifecycle'] == 'Tentative'
    assert memory.update([observation('a', 1)], 'f2') == ['entity-1']


def test_prediction_requires_fresh_observed_evidence_and_controller_success():
    memory = WorldMemory()
    memory.update([observation('initial-a', 0), observation('initial-b', .03)], 'initial')
    memory.update([observation('a', 0), observation('b', .03)], 'f1')
    memory.propose(ert(), 'f1', {'move_relative'})
    memory.executed({'status': 'moved'}, 'f2')
    with pytest.raises(ValueError, match='new public'):
        memory.verify('f1')
    result = memory.verify('f2')
    assert not result['commit']
    assert len(result['discrepancy']['unknown_expected_relations']) == 1
    memory.update([observation('c', 0), observation('d', .03)], 'f3')
    memory.propose(ert(), 'f3', {'move_relative'})
    memory.executed({'status': 'moved'}, 'f4')
    memory.update([observation('e', 0), observation('f', .03)], 'f4')
    assert memory.verify('f4')['commit']
    memory.propose(ert(), 'f4', {'move_relative'})
    memory.executed({'status': 'safety_stop'}, 'f5')
    memory.update([observation('g', 0), observation('h', .03)], 'f5')
    assert not memory.verify('f5')['commit']


def test_contact_cannot_be_inferred_from_a_landmark_envelope():
    memory = WorldMemory()
    memory.update([observation('initial-a', 0), observation('initial-b', 0)], 'initial')
    memory.update([observation('a', 0), observation('b', 0)], 'f1')
    assert memory.relation({'pred': 'On', 'args': ['entity-0', 'entity-1']}, 'f1') is None
    proposal = ert()
    proposal['world_belief'][0]['pred'] = 'On'
    with pytest.raises(ValueError, match='geometric support'):
        memory.propose(proposal, 'f1', {'move_relative'})


def test_sequential_public_measurements_can_verify_after_both_reobserved():
    memory = WorldMemory()
    memory.update([observation('initial-a', 0), observation('initial-b', .03, label='shaft')], 'initial')
    memory.update([observation('a', 0)], 'f1')
    memory.update([observation('b', .03, label='shaft')], 'f2')
    memory.propose(ert(), 'f2', {'move_relative'})
    memory.executed({'status': 'moved'}, 'f3')
    memory.update([observation('c', 0)], 'f4')
    assert memory.relation({'pred': 'Near', 'args': ['entity-0', 'entity-1']}, 'f4') is None
    memory.update([observation('d', .03, label='shaft')], 'f5')
    assert memory.verify('f5')['commit']


def test_invalid_or_unknown_entities_and_hallucinated_actions_are_rejected():
    memory = WorldMemory()
    memory.update([observation('initial-a', 0), observation('initial-b', 0)], 'initial')
    memory.update([observation('a', 0), observation('b', 0)], 'f1')
    proposal = ert()
    proposal['world_belief'][0]['args'][1] = 'private_target_pose'
    with pytest.raises(ValueError, match='existing entity'):
        memory.propose(proposal, 'f1', {'move_relative'})
    with pytest.raises(ValueError, match='exposed shared capability'):
        memory.propose(ert(), 'f1', {'teleport'})


def test_one_frame_and_duplicate_feature_aliases_do_not_confirm_an_entity():
    memory = WorldMemory()
    memory.update([observation('a', 0), observation('a', 0)], 'f1')
    assert len(memory.entities) == 1
    memory.update([observation('same-frame-alias', 0)], 'f1')
    assert memory.entities['entity-0']['observed_frames'] == 1
    assert memory.entities['entity-0']['lifecycle'] == 'Tentative'
    memory.update([observation('new-frame', 0)], 'f2')
    assert memory.entities['entity-0']['lifecycle'] == 'Active'


def test_full_covariance_fusion_and_measured_envelope_iou_association():
    memory=WorldMemory()
    first=observation('body-a',0)
    first.update(position_covariance_m2=np.diag([.0001,.0002,.0003]).tolist(),bounds_world_m=[[-.01,-.01,-.01],[.01,.01,.01]])
    memory.update([first],'f1')
    second=first | {'feature_id':'body-b','position_m':[.001,0,0]}
    assert memory.update([second],'f2')==['entity-0']
    assert np.allclose(memory.entities['entity-0']['covariance_m2'],np.diag([.00005,.0001,.00015]))
    assert np.allclose(memory.entities['entity-0']['position_m'],[.0005,0,0])
    assert envelope_iou(first['bounds_world_m'],second['bounds_world_m'])==pytest.approx(1)
    disjoint=second | {'feature_id':'body-c','bounds_world_m':[[.02,-.01,-.01],[.04,.01,.01]]}
    assert memory.update([disjoint],'f3')==['entity-1']


def test_invalid_covariance_cannot_enter_world_memory():
    memory=WorldMemory()
    invalid=observation('a',0) | {'position_covariance_m2':np.diag([-1,1,1]).tolist()}
    with pytest.raises(ValueError,match='Positive definite'):
        memory.update([invalid],'f1')
    assert memory.entities=={}
