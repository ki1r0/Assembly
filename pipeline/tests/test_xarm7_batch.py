from tools.run_xarm7_batch import commands


def test_three_scenarios_use_same_ours_pipeline_and_selected_condition(tmp_path):
    rows=list(commands(tmp_path,'2','input_error',1800))
    assert len(rows)==3
    assert {(case['task'],case['case']) for case,_,_ in rows}=={
        ('peg','nominal-aligned'),('peg','bolt-horizontal-pose'),('hub-cover','nominal-aligned')}
    for _,run,command in rows:
        assert command[command.index('--robot')+1]=='xarm7'
        assert command[command.index('--method')+1]=='ours'
        assert command[command.index('--condition')+1]=='input_error'
        assert command[command.index('--budget-s')+1]=='1800'
        assert '--rollout' in command
        assert '--verifier-calibration' not in command
        assert run.parent==tmp_path


def test_integration_comparison_is_explicit_for_every_trial(tmp_path):
    rows=list(commands(tmp_path,'3','clean',1800,physics_hz=240))
    assert len(rows)==3
    for _,_,command in rows:
        assert command[command.index('--xarm-physics-hz')+1]=='240'
    for case,_,command in rows:
        assert ('--peg-mvp' in command)==(case['task']=='peg')
