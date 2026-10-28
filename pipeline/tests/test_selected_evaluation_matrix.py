import json

import settings
from task_profiles import SUPPORTED_TASKS


def test_user_confirmed_twelve_rollouts_include_both_peg_initial_orientations():
    matrix = json.loads((settings.ROOT/'config/selected_five_task_evaluation.json').read_text())
    assert len(matrix['methods']) * len(matrix['cases']) == 12
    assert len({row['id'] for row in matrix['cases']}) == 6
    assert len({row['task'] for row in matrix['cases']}) == 5
    assert all(row['task'] in SUPPORTED_TASKS for row in matrix['cases'])
    assert {row['case'] for row in matrix['cases'] if row['task']=='peg'} == {'nominal-aligned', 'bolt-horizontal-pose'}


def test_peg_support_is_a_hub_cover_not_a_second_bolt(monkeypatch):
    from task_profiles import skill_specification
    monkeypatch.setenv('ASSEMBLY_TASK', 'peg')
    skill = skill_specification()
    assert skill['moving_cad_file'] == 'M6 Hub Bolt.usd'
    assert skill['fixture_cover_asset'] == 'parts/Hub Cover Output.usd'
    assert skill['fixture_cover_asset'] != 'parts/' + skill['moving_cad_file']
