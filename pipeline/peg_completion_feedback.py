"""Online development completion from CAPE physical evidence, not benchmark certification."""


def peg_process_completed(result):
    """Do not turn calibration probes, invalid resets or partial predicates into completion."""
    goals=result.get('goal_predicates')
    return (
        result.get('evaluation_status') in {'valid','uncalibrated'}
        and result.get('initial_state_valid') is True
        and result.get('process_provenance') is True
        and result.get('candidate_physical_success') is True
        and isinstance(goals,list) and len(goals)==5
        and all(goal is True for goal in goals)
    )
