import pytest
from peg_completion_feedback import peg_process_completed


def good_result():
    return dict(evaluation_status='uncalibrated',initial_state_valid=True,
                process_provenance=True,candidate_physical_success=True,
                goal_predicates=[True]*5,assembly_success=None)


def test_online_physical_completion_does_not_certify_benchmark_score():
    result=good_result()
    assert peg_process_completed(result)
    assert result['assembly_success'] is None
    assert result['evaluation_status']=='uncalibrated'


@pytest.mark.parametrize('changes',[
    {'evaluation_status':'privileged_diagnostic'}, {'initial_state_valid':False},
    {'process_provenance':False}, {'candidate_physical_success':False},
    {'goal_predicates':[True,True,True,True,None]},
    {'goal_predicates':[True,True,True,True,False]},
])
def test_incomplete_or_privileged_evidence_cannot_complete_online_task(changes):
    assert not peg_process_completed(good_result()|changes)
