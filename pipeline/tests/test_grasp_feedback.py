import json

import settings
from grasp_feedback import public_grasp_feedback
from rpent.tools import ToolResult


def test_every_option_survives_model_serialization_without_changing_execution_plan():
    source = {'status': 'candidates', 'candidates': [
        {'candidate_id': f'grasp-{i}', 'position_m': [i, 0, 1],
         'feasibility': 'geometric estimate, not retention proof',
         'contact_quality': {'supported': True, 'required_friction_coefficient': .2},
         'assembly_compatibility': {'guided_entry_before_release_m': .02},
         'planned_waypoint_joints': [[.1] * 7] * 100,
         'part_component': {'registration': list(range(1000))}}
        for i in range(16)]}
    before = json.dumps(source)
    assert '[truncated]' in ToolResult(data=source).to_text()
    message = ToolResult(data=public_grasp_feedback(source)).to_text()
    decoded = json.loads(message)
    assert len(decoded['candidates']) == 16
    assert decoded['candidates'][-1]['candidate_id'] == 'grasp-15'
    assert decoded['candidates'][0]['assembly_compatibility'] == source['candidates'][0]['assembly_compatibility']
    assert json.dumps(source) == before

    assert decoded['shared_part_component'] == source['candidates'][0]['part_component']
    assert decoded['metadata_shared_by_all_candidates']['feasibility'] == source['candidates'][0]['feasibility']
