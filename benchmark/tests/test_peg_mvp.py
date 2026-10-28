import pytest
from assembly_bench.peg_mvp import PegProcessEvaluator


def sample(t, **overrides):
    return dict(time_s=t, target_controlled=False, lifted_from_support=False,
        in_target_neighborhood=True, through_selected_cover_hole=True,
        entered_selected_casing_opening=True, shaft_radial_error_m=0.00005,
        axis_tilt_rad=0.01, casing_entry_depth_m=0.012, cap_support_gap_m=0.00001,
        cap_support_contact=True, penetration_m=0., premature_bottoming=False,
        robot_released=True, robot_withdrawn=True, relative_linear_speed_mps=0.,
        relative_angular_speed_radps=0., **overrides)


def run_seat(rate, **options):
    evaluator = PegProcessEvaluator(initial_state_valid=True, **options)
    for i in range(2 * rate + 1):
        s = sample(i / rate)
        if i / rate <= .15:
            s.update(target_controlled=True, lifted_from_support=True, robot_released=False,
                     robot_withdrawn=False)
        evaluator.observe(s)
    return evaluator


@pytest.mark.parametrize('rate', [120, 480])
def test_seat_after_release_is_valid_without_held_seat_dwell(rate):
    result = run_seat(rate, calibrated=True).result()
    assert result['assembly_success'] is True
    assert result['checkpoints']['withdrawn_stable']['ever_reached']


def test_unknown_calibration_and_privileged_probes_never_become_scores():
    assert run_seat(120).result()['assembly_success'] is None
    assert run_seat(120, calibrated=True, privileged=True).result()['assembly_success'] is None


def test_off_hole_projection_is_not_reported_as_valid_insertion_progress():
    evaluator = PegProcessEvaluator()
    row = sample(0.)
    row.update(through_selected_cover_hole=False, casing_entry_depth_m=.055)
    evaluator.observe(row)
    assert evaluator.result()['max_entry_m'] == .055
    assert evaluator.result()['max_valid_entry_m'] is None
    evaluator.observe(sample(.01))
    assert evaluator.result()['max_valid_entry_m'] == .012


def test_recovery_is_allowed_and_historical_success_not_latched():
    e = run_seat(120, calibrated=True)
    bad = sample(2.01)
    bad.update(through_selected_cover_hole=False)
    assert e.observe(bad)['assembly_success'] is False
    assert e.result()['checkpoints']['withdrawn_stable']['ever_reached']
    for i in range(1, 151):
        e.observe(sample(2.01 + i / 120))
    assert e.result()['assembly_success'] is True


def test_missing_evidence_and_time_gaps_cannot_complete_dwell():
    e = run_seat(120, calibrated=True)
    missing = sample(2.01)
    missing['cap_support_contact'] = None
    assert e.observe(missing)['assembly_success'] is None
    assert e.observe(sample(4.))['assembly_success'] is None
    with pytest.raises(ValueError):
        e.observe(sample(4.))


@pytest.mark.parametrize('key,value', [('through_selected_cover_hole', False),
    ('cap_support_contact', False), ('penetration_m', .003), ('robot_released', False),
    ('robot_withdrawn', False), ('cap_support_gap_m', -.003)])
def test_invalid_seats_never_pass(key, value):
    e = PegProcessEvaluator(calibrated=True, initial_state_valid=True)
    initial = sample(0.)
    initial['target_controlled'] = True
    e.observe(initial)
    for i in range(1, 241):
        s = sample(i / 120)
        s[key] = value
        e.observe(s)
    assert e.result()['assembly_success'] is False


def test_reset_release_and_preplaced_part_do_not_gain_provenance():
    e = PegProcessEvaluator(calibrated=True, initial_state_valid=True)
    for i in range(241):
        e.observe(sample(i / 120))
    assert e.result()['assembly_success'] is False
    assert not e.result()['process_provenance']
    assert e.result()['seat_predicate_stable'] is True


def test_invalid_initial_scene_cannot_be_a_formal_success():
    e = run_seat(120, calibrated=True)
    e.initial_state_valid = False
    e.observe(sample(2.01))
    assert e.result()['assembly_success'] is None
    assert e.result()['evaluation_status'] == 'invalid_scene'


def test_offline_rescore_preserves_unknown_and_never_overwrites(tmp_path):
    import json
    from assembly_bench.peg_mvp import rescore_run
    run = tmp_path / 'run'
    (run / 'private').mkdir(parents=True)
    (run / 'episode').mkdir()
    (run / 'episode/backend-summary.json').write_text(json.dumps({
        'task': 'peg', 'native_success': False, 'terminal': {'source': 'test'}}))
    (run / 'private/peg-process-measurements.jsonl').write_text(
        '\n'.join(json.dumps(sample(i / 120)) for i in range(241)))
    output = tmp_path / 'score'
    result = rescore_run(run, output)
    assert result['assembly_success'] is None
    assert (output / 'episode_report.html').exists()
    with pytest.raises(FileExistsError):
        rescore_run(run, output)
    (run / 'episode/backend-summary.json').write_text(json.dumps({
        'task': 'peg', 'native_success': False, 'error': 'camera failed',
        'terminal': {'source': 'backend_error'}}))
    failed = rescore_run(run, tmp_path / 'failed-score')
    assert failed['evaluation_status'] == 'runtime_error'
    assert failed['assembly_success'] is None


def test_instances_are_explicit_and_held_out_variants_balanced():
    from pathlib import Path
    from assembly_bench.peg_mvp_runner import selected_instances
    path = Path(__file__).resolve().parents[1] / 'data/instances.json'
    rows = selected_instances(path, 'test')
    assert len(rows) == 20
    assert sum(row['case'] == 'nominal-aligned' for row in rows) == 10
    assert len(selected_instances(path, 'dev')) == 4
    assert selected_instances(path,'dev',instance_ids=['dev-h0'])[0]['case']=='bolt-horizontal-pose'
    with pytest.raises(ValueError):
        selected_instances(path,'dev',instance_ids=['test-u00'])
    with pytest.raises(ValueError):
        selected_instances(path, 'test', -1)


def test_native_raycast_hit_uses_installed_collision_attribute():
    from types import SimpleNamespace
    from assembly_bench.peg_mvp import scene_query_hit
    hit = SimpleNamespace(collision='/World/Cover/node_/mesh_', distance=.00201,
                          normal=(0, 0, 1), position=(.46, -.07, .746))
    result = scene_query_hit(hit, (0, 0, 1), 3)
    assert result['collider'] == hit.collision
    assert result['gap_m'] == pytest.approx(.00001)
    assert result['normal_alignment'] == 1.


def test_explicit_geometric_support_mode_does_not_fabricate_force():
    from assembly_bench.peg_mvp import PegCriteria
    e = PegProcessEvaluator(PegCriteria(support_evidence='collider_geometry'),
                            calibrated=True, initial_state_valid=True)
    for i in range(241):
        s = sample(i / 120)
        s.update(cap_support_contact=False, cap_bearing_geometry_supported=True)
        if i <= 18:
            s.update(target_controlled=True, robot_released=False, robot_withdrawn=False)
        e.observe(s)
    assert e.result()['assembly_success'] is True
    assert e.result()['criteria']['support_evidence'] == 'collider_geometry'
    s = sample(2.01)
    s.update(cap_support_contact=True, cap_bearing_geometry_supported=None)
    assert e.observe(s)['assembly_success'] is None


def test_test_load_is_privileged_and_recorded_for_the_actual_interval():
    from assembly_bench.peg_mvp import diagnostic_load_force
    assert diagnostic_load_force('peg-seat-probe', True, .1) == .2
    assert diagnostic_load_force('peg-seat-probe', True, .5) == 0.
    assert diagnostic_load_force(None, False, .1) == 0.
    with pytest.raises(ValueError):
        diagnostic_load_force(None, True, .1)


def test_endpoint_calibration_splits_have_disjoint_seeds_and_explicit_labels():
    from assembly_bench.peg_mvp_runner import calibration_rows
    development=calibration_rows('development')
    acceptance=calibration_rows('acceptance')
    assert len(development)==30 and len(acceptance)==10
    assert sum(row['expected_seat'] for row in development)==10
    assert sum(not row['expected_seat'] for row in development)==20
    assert {row['probe_seed'] for row in development}.isdisjoint(row['probe_seed'] for row in acceptance)
    assert len({row['instance_id'] for row in development+acceptance})==40


def test_batch_unknowns_do_not_become_failures_or_boost_success_denominator():
    from assembly_bench.peg_mvp_runner import aggregate_trials
    result=aggregate_trials([
        {'status':'ended','evaluation_status':'valid','assembly_success':True},
        {'status':'ended','evaluation_status':'valid','assembly_success':False},
        {'status':'ended','evaluation_status':'uncalibrated','assembly_success':None},
    ],4)
    assert result['success_rate']==.5
    assert result['scored_coverage']==.5
    assert result['verified_delivery_rate']==.25
    assert result['unscored_ended_count']==1
    assert aggregate_trials([],4)['success_rate'] is None


def test_crashed_negative_probe_is_not_a_correct_calibration_result():
    from assembly_bench.peg_mvp_runner import endpoint_calibration_match
    row=dict(returncode=0,termination_reason='evaluator_calibration_complete',
             seat_predicate_stable=False,expected_seat=False)
    assert endpoint_calibration_match(row) is True
    assert endpoint_calibration_match(dict(row,returncode=1)) is None
    assert endpoint_calibration_match(dict(row,termination_reason='backend_error')) is None
    assert endpoint_calibration_match(dict(row,termination_reason='wallclock_budget')) is None
    assert endpoint_calibration_match(dict(row,seat_predicate_stable=None)) is None


def test_actual_collider_fit_can_disprove_old_circular_approximation():
    from assembly_bench.peg_mvp import PegCriteria
    e=PegProcessEvaluator(PegCriteria(support_evidence='collider_geometry',
        aperture_evidence='collider_sweep'),calibrated=True,initial_state_valid=True)
    for i in range(241):
        s=sample(i/120)
        s.update(entered_selected_casing_opening=False,shaft_radial_error_m=.0003268,
            penetration_m=.0001578,non_casing_penetration_bound_m=0.,
            casing_sweep_clear=True,cap_bearing_geometry_supported=True)
        if i<=18:s.update(target_controlled=True,robot_released=False,robot_withdrawn=False)
        e.observe(s)
    assert e.result()['assembly_success'] is True
    s['time_s']=2.01;s['casing_sweep_clear']=False
    assert e.observe(s)['assembly_success'] is False
    s['time_s']=2.02;s['casing_sweep_clear']=None
    assert e.observe(s)['assembly_success'] is None
