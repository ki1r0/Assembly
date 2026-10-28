from grasp_replay import recorded_retreat
from grasp_replay import replay_selection
import pytest


def test_retreat_is_recorded_after_grasp_and_before_another_plan():
    def rows(actions):return [{'command':{'action':a,'translation_m':[0.,0.,.1]}} for a in actions]
    assert recorded_retreat(rows(['move_relative','execute_grasp','observe','move_relative']))['action']=='move_relative'
    assert recorded_retreat(rows(['execute_grasp','plan_grasp','move_relative'])) is None
    assert recorded_retreat(rows(['observe','move_relative'])) is None


def test_alternative_diagnostic_selection_keeps_original_record_untouched():
    candidates=[{'candidate_id':'grasp-0'},{'candidate_id':'grasp-2'}]
    records=[{'command':{'action':'plan_grasp'},'result':{'candidates':candidates}},
             {'command':{'action':'execute_grasp','candidate_id':'grasp-0'}}]
    assert replay_selection(records)[2] is candidates[0]
    _,grasp,reference=replay_selection(records,'grasp-2')
    assert grasp['candidate_id']=='grasp-2' and reference is candidates[1]
    assert records[-1]['command']['candidate_id']=='grasp-0'
    with pytest.raises(ValueError):replay_selection(records,'grasp-99')
