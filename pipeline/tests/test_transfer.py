"""Task routing and public-data boundary tests for the cross-task adapter."""
import json
import pytest
import numpy as np
from pathlib import Path
import settings
from task_profiles import generic,instruction,public_skill_specification
from generic_toolkit import RobotToolkit

def test_task_and_robot_routing(monkeypatch):
    monkeypatch.delenv('ASSEMBLY_ACTION_BACKEND',raising=False)
    monkeypatch.setenv('ASSEMBLY_ROBOT','panda');monkeypatch.setenv('ASSEMBLY_TASK','peg');assert generic();assert 'two o-clock' in instruction()
    monkeypatch.setenv('ASSEMBLY_ACTION_BACKEND','generic');assert generic()
    monkeypatch.delenv('ASSEMBLY_ACTION_BACKEND')
    monkeypatch.setenv('ASSEMBLY_ROBOT','r1');assert generic()
    monkeypatch.setenv('ASSEMBLY_TASK','hub-cover');assert 'hub cover' in instruction()
    monkeypatch.setenv('ASSEMBLY_TASK','hub-cover-input');assert 'input hub cover' in instruction()
    monkeypatch.setenv('ASSEMBLY_TASK','hub-cover-small');assert 'small hub cover' in instruction()

@pytest.mark.parametrize('task',['hub-cover-input','hub-cover-small'])
def test_hub_variant_public_skill_omits_evaluator_targets(monkeypatch,task):
    monkeypatch.setenv('ASSEMBLY_TASK',task)
    profile=public_skill_specification()
    private=json.loads((settings.ROOT/'config/task_evaluator_profiles.json').read_text())[task]
    assert profile['moving_part']
    assert profile['moving_cad_file']
    assert 'expected_root_offset_m' not in profile
    assert 'target_socket' not in profile
    assert not any(value in json.dumps(profile) for value in private.values() if isinstance(value,str))

def test_cover_mass_uses_transformed_cad_cache_for_non_output_variant(tmp_path):
    from tools.calibrate_part_mass import cover_mass_profile
    cache=tmp_path/'cover_input.npz'
    np.savez(cache,volume_m3=.0003,uniform_density_com_local_m=[.001,.002,.003],geometry_scale_m_per_coordinate=.001)
    result=cover_mass_profile(tmp_path,'polypropylene-assumption','Hub Cover Input.stl',cache)
    assert result['mass_kg']==pytest.approx(.27)
    assert result['uniform_density_com_local_m']==[.001,.002,.003]
    assert result['volume_source']=='USD hierarchy-transformed public CAD triangle cache'

def test_non_output_cover_requires_transformed_geometry_cache(tmp_path):
    from tools.calibrate_part_mass import cover_mass_profile
    with pytest.raises(ValueError,match='hierarchy-transformed'):
        cover_mass_profile(tmp_path,'polypropylene-assumption','Hub Cover Input.stl')

@pytest.mark.parametrize('robot',['r1','panda'])
def test_actor_receives_generic_tools_not_bolt_macros(tmp_path,robot):
    (tmp_path/'ipc').mkdir();state={'robot':robot,'native_success':False,'observation_id':'frame','result':{},'sensors':{},'features':[],'active_arm':'left','arms':{'left':{'tcp_pose':[0,0,1,1,0,0,0]}},'task_instruction':'Place cover','private_evaluator':{'true_target':[1,2,3]}}
    (tmp_path/'ipc/reply_0000.json').write_text(json.dumps(state));toolkit=RobotToolkit(tmp_path)
    try:
        public=toolkit.public_data();assert public['task']=='Place cover';assert 'left' in public['arms'];assert 'private_evaluator' not in public;assert ('move_both' in toolkit._tools) is (robot=='r1');assert 'insert_visual' not in toolkit._tools;assert 'grasp_visual' not in toolkit._tools;assert 'plan_grasp' in toolkit._tools;assert 'execute_grasp' in toolkit._tools;assert 'assemble' in toolkit._tools
    finally:toolkit.close()

def test_original_evaluator_import_and_positive_logic():
    from runtime.bolt_harness.evaluator import IndependentBoltSeatingEvaluator,BoltSeatTolerances,BoltSeatMeasurement
    import yaml
    cfg=yaml.safe_load((settings.ROOT/'config/bolt_insertion.yaml').read_text());evaluator=IndependentBoltSeatingEvaluator(BoltSeatTolerances(**cfg['tolerances']))
    # Synthetic known-good trace checks logic only, not simulation/contact calibration.
    good=dict(through_selected_cover_hole=True,entered_selected_casing_opening=True,casing_entry_depth_m=.025,shaft_radial_error_m=0,cap_support_gap_m=0,cap_support_contact=True,penetration_m=0,premature_bottoming=False,insertion_provenance_valid=True,controller_stopped=True,axial_position_m=.06,relative_linear_speed_mps=0,relative_angular_speed_radps=0)
    for i in range(200):
        held=i<60;status=evaluator.observe(BoltSeatMeasurement(time_s=i*.01,physically_held=held,robot_contact_or_support=held,**good))
    assert status.task_success

def test_original_evaluator_does_not_pass_without_release_or_in_wrong_hole():
    from runtime.bolt_harness.evaluator import IndependentBoltSeatingEvaluator,BoltSeatTolerances,BoltSeatMeasurement
    import yaml
    config=yaml.safe_load((settings.ROOT/'config/bolt_insertion.yaml').read_text())
    for wrong_hole in [False,True]:
        evaluator=IndependentBoltSeatingEvaluator(BoltSeatTolerances(**config['tolerances']))
        for i in range(200):
            status=evaluator.observe(BoltSeatMeasurement(time_s=i*.01,through_selected_cover_hole=not wrong_hole,entered_selected_casing_opening=not wrong_hole,casing_entry_depth_m=.025,shaft_radial_error_m=.03 if wrong_hole else 0,cap_support_gap_m=0,cap_support_contact=True,penetration_m=0,premature_bottoming=False,physically_held=True,insertion_provenance_valid=True,controller_stopped=True,robot_contact_or_support=True,axial_position_m=.06,relative_linear_speed_mps=0,relative_angular_speed_radps=0))
        assert not status.task_success
        assert status.seat_ready is (not wrong_hole)

def test_hub_pending_calibration_never_becomes_native_success():
    from hub_evaluator import IndependentSeatEvaluator,SeatMeasurement
    result=IndependentSeatEvaluator(calibration_status='pending').evaluate(SeatMeasurement(axial_depth_m=.062,radial_error_m=0,tilt_deg=0,penetration_m=0,released=True,stable=True,contact_valid=True,settle_speed_mps=0,settle_window_s=1))
    assert result.physical_criteria_passed
    assert not result.task_success
