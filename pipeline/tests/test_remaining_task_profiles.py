import json

import numpy as np
import pytest

import settings
from task_profiles import HUB_TASKS, SUPPORTED_TASKS, instruction, public_skill_specification


TASKS = {
    "m10-casing-bolt": ("M10 Casing Bolt.usd", "m10_bolt.npz", "parts/Casing Top.usd"),
    "m10-casing-nut": ("M10 Casing Nut.usd", "m10_nut.npz", "parts/M10 Casing Bolt.usd"),
    "casing-top-to-base": ("Casing Top.usd", "casing_top.npz", "parts/Casing Base.usd"),
    "oil-level-indicator": ("Oil Level Indicator.usd", "oil_indicator.npz", "parts/Casing Base.usd"),
    "breather-plug": ("Breather Plug.usd", "breather_plug.npz", "parts/Casing Base.usd"),
    "bearing-32013-onto-output-shaft": (
        "Bearing 32013 Inner Assy.usd",
        "bearing_32013.npz",
        "parts/Output Shaft.usd",
    ),
    "bearing-transfer-onto-shaft": (
        "Bearing 32205 Inner Assy.usd",
        "bearing_32205_transfer.npz",
        "parts/Transfer Shaft.usd",
    ),
    "bearing-input-bottom-onto-shaft": (
        "Bearing 30304 Inner Assy.usd",
        "bearing_30304.npz",
        "parts/Input Shaft.usd",
    ),
    "bearing-input-top-onto-shaft": (
        "Bearing 32205 Inner Assy.usd",
        "bearing_32205_input.npz",
        "parts/Input Shaft.usd",
    ),
}


def test_transfer_bearing_exposes_original_negative_x_seat_and_wide_face_leads():
    from scipy.spatial.transform import Rotation
    profile=json.loads((settings.ROOT/'config/task_evaluator_profiles.json').read_text())['bearing-transfer-onto-shaft']
    fixture=Rotation.from_euler('xyz',profile['fixture_rotation_xyz_deg'],degrees=True)
    moving=fixture*Rotation.from_euler('xyz',profile['candidate_reset_hint']['moving_rotation_xyz_deg'],degrees=True)
    assert np.allclose(fixture.apply([-1.,0,0]),[0,0,1])
    assert np.allclose(moving.apply([1.,0,0]),[0,0,-1])
    assert profile['physical_gt_verified'] is False


def test_oil_indicator_uses_cad_y_axis_without_resolving_port_or_fixture(monkeypatch):
    monkeypatch.setenv("ASSEMBLY_TASK", "oil-level-indicator")
    skill = public_skill_specification()
    assert skill["moving_feature_axis_local"] == [0.0, 1.0, 0.0]
    assert "remain unconfirmed" in skill["feature_provenance"]
    with np.load(settings.ROOT / "assets/public_cad_models/oil_indicator.npz") as cache:
        points = cache["points"]
        normals = cache["normals"]
    shoulder = (np.abs(points[:, 1] - 0.002149) < 0.000002) & (normals[:, 1] > 0.999)
    assert int(shoulder.sum()) > 100
    profile = json.loads(
        (settings.ROOT / "config/task_evaluator_profiles.json").read_text()
    )["oil-level-indicator"]
    assert profile["fixture_rotation_xyz_deg"] == [0.0, 0.0, 0.0]
    assert profile["candidate_reset_hint"]["moving_rotation_xyz_deg"] == [0.0, 0.0, 0.0]
    assert "target_socket" not in profile


def test_input_bottom_bearing_exposes_negative_y_seat_with_wide_face_leading():
    from scipy.spatial.transform import Rotation

    profile = json.loads(
        (settings.ROOT / "config/task_evaluator_profiles.json").read_text()
    )["bearing-input-bottom-onto-shaft"]
    fixture = Rotation.from_euler("xyz", profile["fixture_rotation_xyz_deg"], degrees=True)
    relative = Rotation.from_euler(
        "xyz", profile["candidate_reset_hint"]["moving_rotation_xyz_deg"], degrees=True
    )
    assert profile["fixture_rotation_xyz_deg"] == [-90.0, 0.0, 0.0]
    assert profile["candidate_reset_hint"]["moving_rotation_xyz_deg"] == [0.0, 0.0, 90.0]
    assert np.allclose(fixture.apply([0.0, -1.0, 0.0]), [0.0, 0.0, 1.0])
    assert np.allclose(relative.apply([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0])
    assert np.allclose((fixture * relative).apply([1.0, 0.0, 0.0]), [0.0, 0.0, -1.0])
    assert profile["allow_provisional_open_mesh_mass"] is True


@pytest.mark.parametrize("task", ["oil-level-indicator", "bearing-input-bottom-onto-shaft"])
def test_source_mapping_corrections_do_not_enable_uncalibrated_success(task):
    from relation_evaluator import PendingRelationEvaluator, build_relation_evaluator

    profile = json.loads(
        (settings.ROOT / "config/task_evaluator_profiles.json").read_text()
    )[task]
    assert profile["calibration_status"] == "pending"
    assert profile["evaluation_status"] == "pending"
    assert profile["status"] == "pending_calibration"
    assert profile["physical_gt_verified"] is False
    evaluator = build_relation_evaluator(task, profile, None, settings.CAD_METERS_PER_SOURCE_UNIT)
    assert isinstance(evaluator, PendingRelationEvaluator)


@pytest.mark.parametrize("task,expected", TASKS.items())
def test_task_prompt_public_asset_and_private_pending_profile(monkeypatch, task, expected):
    asset, cache, fixture = expected
    monkeypatch.setenv("ASSEMBLY_TASK", task)
    prompt = instruction()
    public = public_skill_specification()
    private = json.loads(
        (settings.ROOT / "config/task_evaluator_profiles.json").read_text()
    )[task]

    assert task in SUPPORTED_TASKS
    assert "Call observe first" in prompt
    assert public["moving_cad_file"] == asset
    assert public["moving_cad_cache"] == cache
    assert private["fixture_asset"] == fixture
    assert private["calibration_status"] == "pending"
    assert private["evaluation_status"] == "pending"
    assert private["physical_gt_verified"] is False
    for key in ("fixture_asset", "candidate_reset_hint", "calibration_file"):
        assert key not in public


def test_threaded_nut_prompt_does_not_equate_push_fit_with_success(monkeypatch):
    monkeypatch.setenv("ASSEMBLY_TASK", "m10-casing-nut")
    prompt = instruction().lower()
    assert "rotate" in prompt
    assert "not treat pushing" in prompt
    assert public_skill_specification()["moving_feature_axis_local"] == [0.0, 1.0, 0.0]


def test_breather_scene_exposes_side_port_without_target_coordinates(monkeypatch):
    from scipy.spatial.transform import Rotation

    monkeypatch.setenv("ASSEMBLY_TASK", "breather-plug")
    prompt = instruction().lower()
    profile = json.loads(
        (settings.ROOT / "config/task_evaluator_profiles.json").read_text()
    )["breather-plug"]
    fixture_rotation = Rotation.from_euler(
        "xyz", profile["fixture_rotation_xyz_deg"], degrees=True
    )

    assert "side-facing breather port" in prompt
    assert "shaft bore" in prompt
    assert "safely retract the arm" in prompt
    assert "observe again before concluding" in prompt
    assert np.allclose(fixture_rotation.apply([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-7)
    assert "0.55" not in prompt
    assert "feature-" not in prompt


@pytest.mark.parametrize("task", ["hub-cover", "hub-cover-input", "hub-cover-small"])
def test_hub_cover_prompts_use_public_geometry_not_historical_fit_answers(monkeypatch, task):
    monkeypatch.setenv("ASSEMBLY_TASK", task)
    prompt = instruction().lower()

    assert "current public" in prompt
    assert "scene coordinates" in prompt
    assert "historical fit poses" in prompt


def test_m10_nut_scene_reset_aligns_public_bore_and_bolt_axes():
    from scipy.spatial.transform import Rotation

    profiles = json.loads(
        (settings.ROOT / "config/task_evaluator_profiles.json").read_text()
    )
    profile = profiles["m10-casing-nut"]
    fixture_rotation = Rotation.from_euler(
        "xyz", profile["fixture_rotation_xyz_deg"], degrees=True
    )
    reset_rotation = Rotation.from_euler(
        "xyz", profile["candidate_reset_hint"]["moving_rotation_xyz_deg"], degrees=True
    )
    local_phase = Rotation.from_euler(
        "y", profile["candidate_reset_hint"].get("moving_local_phase_y_deg", 0.0), degrees=True
    )
    bolt_axis_world = fixture_rotation.apply([0.0, 1.0, 0.0])
    nut_axis_world = (fixture_rotation * reset_rotation * local_phase).apply([0.0, 1.0, 0.0])

    assert float(nut_axis_world @ bolt_axis_world) > 0.999


def test_all_declared_moving_axes_match_public_collision_mesh_principal_axes():
    profiles = json.loads(
        (settings.ROOT / "config/assembly_skills.json").read_text()
    )
    long_axis_tasks = {
        "peg",
        "output-shaft-into-base",
        "transfer-shaft-into-base",
        "input-shaft-into-base",
        "m10-casing-bolt",
        "oil-level-indicator",
    }
    assert set(profiles) == SUPPORTED_TASKS

    for task, profile in profiles.items():
        cache = np.load(settings.ROOT / "assets/public_cad_models" / profile["moving_cad_cache"])
        points = np.asarray(cache["points"], dtype=np.float64)
        centered = points - points.mean(axis=0)
        covariance = centered.T @ centered / len(centered)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        axis = np.asarray(profile["moving_feature_axis_local"], dtype=np.float64)
        axis /= np.linalg.norm(axis)
        variance_ratio = (
            float(axis @ covariance @ axis) - float(eigenvalues[0])
        ) / max(float(eigenvalues[-1] - eigenvalues[0]), 1e-12)
        principal_alignment = float(np.max(np.abs(eigenvectors.T @ axis)))

        assert principal_alignment >= 0.95, task
        if task in long_axis_tasks:
            assert variance_ratio >= 0.92, (task, variance_ratio)
        else:
            assert variance_ratio <= 0.08, (task, variance_ratio)


@pytest.mark.parametrize(
    "robot,expected_capability,forbidden_capability",
    [
        ("panda", "only the panda single arm is available", "available r1 has two arms"),
        ("r1", "available r1 has two arms", "only the panda single arm is available"),
    ],
)
def test_casing_mate_prompt_matches_available_robot_arms(
    monkeypatch, robot, expected_capability, forbidden_capability
):
    monkeypatch.setenv("ASSEMBLY_TASK", "casing-top-to-base")
    monkeypatch.setenv("ASSEMBLY_ROBOT", robot)

    prompt = instruction().lower()

    assert expected_capability in prompt
    assert forbidden_capability not in prompt
    assert "request specific human assistance" in prompt


def test_all_manifest_tasks_have_a_registered_runner_profile():
    manifest = json.loads(
        (settings.ROOT / "config/gearbox_single_step_tasks.json").read_text()
    )
    task_ids = {
        item.get("existing_task_id", item["id"]) for item in manifest["tasks"]
    }
    assert task_ids == SUPPORTED_TASKS
    assert set(manifest["excluded_scene_setup_actions"]) == {
        "focus",
        "unfocus",
        "flip",
        "upright",
        "inspect",
    }
    assert manifest["current_cad_scale_m_per_source_unit"] == 0.001
    assert manifest["historical_scene_smoke_scale_m_per_source_unit"] == 0.002
    assert "invalid as current-scale" in manifest["historical_scene_smoke_status"]
    smoke = manifest["current_scale_scene_smoke_batch"]
    assert set(smoke["runs_by_manifest_task_id"]) == {item["id"] for item in manifest["tasks"]}
    stale_tasks = set(smoke["stale_manifest_task_ids"])
    assert len(smoke["runs_by_manifest_task_id"]) == 18
    assert smoke["status"] == "passed_all_deduplicated_tasks"
    assert smoke["prior_batch_status"] == "passed_all_deduplicated_tasks"
    assert stale_tasks == set()
    assert smoke["superseded_runs_by_manifest_task_id"] == {
        "m10-casing-nut": "runs/20261021T135756796967Z"
    }
    assert smoke["runs_by_manifest_task_id"]["m10-casing-nut"] == (
        "runs/20261021T152053180127Z"
    )
    assert smoke["cad_meters_per_source_unit"] == manifest["current_cad_scale_m_per_source_unit"]
    assert smoke["model_called"] is False
    assert smoke["robot_action_count"] == 0
    assert smoke["physics_steps_per_run"] == 72
    assert smoke["video_frames_per_run"] == 3
    assert smoke["rgbd_initial_observation_verified"] is True
    assert smoke["assembly_success_claimed"] is False
    rollout = smoke["representative_full_rollout_attempt"]
    assert rollout["task"] == "output-gear-onto-shaft"
    assert rollout["status"] == "timed_out_before_grasp"
    assert rollout["run"] == "runs/20261021T152614254020Z"
    assert rollout["plan_grasp_generated_candidate_counts"] == [0] * 8
    diagnostic = rollout["offline_public_geometry_diagnostic"]
    assert diagnostic["registration_supported"] is True
    assert diagnostic["pca_candidate_count"] == 0
    assert diagnostic["antipodal_candidate_count_with_current_normal_sign_filter"] == 0
    assert diagnostic["bidirectional_normal_diagnostic"]["after_complete_finger_cad_collision_filter"] == 0
    assert diagnostic["private_scene_pose_used"] is False
    assert rollout["pickup_observed"] is False
    assert rollout["native_success"] is False
    assert rollout["assembly_success_claimed"] is False
    assert rollout["evaluator_status"] == "pending_calibration"
    hub_rollout = smoke["representative_hub_cover_input_full_rollout_attempt"]
    assert hub_rollout["task"] == "hub-cover-input"
    assert hub_rollout["status"] == "timed_out_after_grasp_attempt_before_assembly"
    assert hub_rollout["run"] == "runs/20261021T155102609308Z"
    assert hub_rollout["plan_grasp_generated_candidate_counts"] == [8]
    assert hub_rollout["retention_verified"] is False
    assert hub_rollout["assembly_tool_calls"] == 0
    assert hub_rollout["pickup_assessment"] == "evidence_supports_lift_but_retention_not_certified"
    assert hub_rollout["native_success"] is False
    assert hub_rollout["assembly_success_claimed"] is False
    assert hub_rollout["evaluator_status"] == "pending_calibration"
    hub_task = next(item for item in manifest["tasks"] if item["id"] == "hub-cover-input")
    assert hub_task["last_full_rollout_attempt"]["run_dir"] == hub_rollout["run"]
    plug_rollout = smoke["representative_breather_plug_full_rollout_attempt"]
    assert plug_rollout["task"] == "breather-plug"
    assert plug_rollout["status"] == "timed_out_during_grasp_approach_before_closure"
    assert plug_rollout["run"] == "runs/20261021T160755039658Z"
    assert plug_rollout["execute_grasp_status"] == "interrupted"
    assert plug_rollout["final_gripper_width_m"] > 0.079
    assert plug_rollout["grasp_contact_observed"] is False
    assert plug_rollout["assembly_tool_calls"] == 0
    assert plug_rollout["terminal_source"] == "wallclock_budget"
    assert plug_rollout["native_success"] is False
    assert plug_rollout["assembly_success_claimed"] is False
    plug_retry = smoke["representative_breather_plug_1200s_full_rollout_attempt"]
    assert plug_retry["task"] == "breather-plug"
    assert plug_retry["run"] == "runs/20261021T162217447771Z"
    assert plug_retry["budget_s"] == 1200
    assert plug_retry["status"] == "grasped_and_lifted_then_assembly_interrupted_by_wallclock_budget"
    assert plug_retry["grasp_stage_statuses"] == ["reached"] * 6
    assert plug_retry["executor_retention_verified"] is False
    assert plug_retry["assembly_tool_calls"] == 1
    assert plug_retry["assembly_result"] == "interrupted"
    assert plug_retry["posthoc_evaluator_only"]["pickup_observed"] is True
    assert plug_retry["posthoc_evaluator_only"]["task_success"] is False
    assert plug_retry["evaluator_status"] == "pending_calibration"
    assert plug_retry["video_fully_decoded"] is True
    port_scene_smoke = smoke["representative_breather_plug_side_port_visible_scene_smoke"]
    assert port_scene_smoke["run"] == "runs/20261021T165255944197Z"
    assert port_scene_smoke["status"] == "passed_scene_initialization_only"
    assert port_scene_smoke["model_called"] is False
    assert port_scene_smoke["robot_action_count"] == 0
    help_retry = smoke["representative_breather_plug_1800s_rotated_scene_rollout_attempt"]
    assert help_retry["run"] == "runs/20261021T165623571466Z"
    assert help_retry["terminal_source"] == "model_ask_human_help"
    assert help_retry["assembly_tool_calls"] == 0
    assert help_retry["posthoc_evaluator_only"]["physical_gt_verified"] is False
    recovery_retry = smoke["representative_breather_plug_1800s_recovery_prompt_rollout_attempt"]
    assert recovery_retry["run"] == "runs/20261021T171626975199Z"
    assert recovery_retry["physics_steps"] == 56076
    assert recovery_retry["grasp_attempts"][1]["candidate_id"] == "grasp-8"
    assert recovery_retry["grasp_attempts"][1]["retention_verified"] is False
    assert recovery_retry["assembly_tool_calls"] == 0
    assert recovery_retry["terminal_source"] == "wallclock_budget"
    assert recovery_retry["posthoc_evaluator_only"]["task_success"] is False
    assert recovery_retry["evaluator_status"] == "pending_calibration"
    assert recovery_retry["post_run_public_frame_inspection"]["moving_part_visible_near_table"] is True
    assert recovery_retry["post_run_public_frame_inspection"]["frame_delivered_to_model"] is False
    plug_task = next(item for item in manifest["tasks"] if item["id"] == "breather-plug")
    assert plug_task["last_full_rollout_attempt"]["run_dir"] == recovery_retry["run"]
    assert plug_task["previous_full_rollout_attempt"]["run_dir"] == help_retry["run"]
    assert plug_task["full_rollout_attempt_history"] == [
        plug_rollout["run"],
        plug_retry["run"],
        help_retry["run"],
        recovery_retry["run"],
    ]
    r1_smoke = smoke["representative_r1_scene_smoke"]
    assert r1_smoke["task"] == "output-gear-onto-shaft"
    assert r1_smoke["status"] == "passed_scene_initialization_only"
    assert r1_smoke["robot"] == "r1"
    assert r1_smoke["run"] == "runs/20261021T154611991002Z"
    assert r1_smoke["robot_action_count"] == 0
    assert r1_smoke["assembly_success_claimed"] is False
    for task in manifest["tasks"]:
        current_run = smoke["runs_by_manifest_task_id"][task["id"]]
        assert task["last_scene_smoke"]["run_dir"] == current_run
        assert task["last_scene_smoke"]["status"] == "passed_current_scale_scene_smoke"
    assert smoke["launch_id_by_manifest_task_id"] == {
        item["id"]: item["existing_task_id"]
        for item in manifest["tasks"]
        if "existing_task_id" in item
    }


def test_relation_tasks_declare_their_physics_evaluator_kind():
    profiles = json.loads(
        (settings.ROOT / "config/task_evaluator_profiles.json").read_text()
    )
    relation_profiles = {task: profile for task, profile in profiles.items() if "fixture_asset" in profile}
    assert set(relation_profiles) == set(profiles) - {"hub-cover", "hub-cover-input", "hub-cover-small"}
    assert all(profile.get("evaluator_kind") for profile in relation_profiles.values())
    assert relation_profiles["m10-casing-nut"]["evaluator_kind"] == "threaded_engagement"
    assert relation_profiles["casing-top-to-base"]["evaluator_kind"] == "multi_surface_mate"
    assert all(
        profile["evaluator_kind"] == "rigid_pose_seat"
        for task, profile in relation_profiles.items()
        if task not in {"m10-casing-nut", "casing-top-to-base"}
    )


def test_every_manifest_task_has_one_supported_evaluator_route():
    import yaml
    from fixture_geometry import physical_calibration_matches_scale
    from relation_evaluator import PendingRelationEvaluator, build_relation_evaluator

    manifest = json.loads(
        (settings.ROOT / "config/gearbox_single_step_tasks.json").read_text()
    )
    profiles = json.loads(
        (settings.ROOT / "config/task_evaluator_profiles.json").read_text()
    )
    task_ids = {
        item.get("existing_task_id", item["id"]) for item in manifest["tasks"]
    }

    assert task_ids == SUPPORTED_TASKS
    assert set(profiles) == SUPPORTED_TASKS - {"peg"}
    assert set(HUB_TASKS) == {"hub-cover", "hub-cover-input", "hub-cover-small"}
    for task in HUB_TASKS:
        profile = profiles[task]
        assert profile["target_socket"]
        assert profile["calibration_file"]
        assert len(profile["expected_root_offset_m"]) == 3
        assert len(profile["expected_rotation_xyz_deg"]) == 3
        calibration_path = settings.ROOT / "config" / profile["calibration_file"]
        calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else {}
        calibrated = physical_calibration_matches_scale(
            calibration, manifest["current_cad_scale_m_per_source_unit"]
        )
        assert calibrated == (task in {'hub-cover-small','hub-cover'})
        if calibrated:
            assert calibration['positive']['regraded_status'] == 'PASS'
            assert calibration['positive']['real_contact_samples'] > 0
            assert len(calibration['negatives']) >= 2
            assert all(case['regraded_status'] == 'FAIL' for case in calibration['negatives'])

    for task, profile in profiles.items():
        assert profile["calibration_file"]
        if task not in HUB_TASKS:
            assert profile["fixture_asset"]
            assert profile["evaluator_kind"] in {
                "rigid_pose_seat",
                "threaded_engagement",
                "multi_surface_mate",
            }
            assert profile["evaluation_status"] == "pending"
            assert profile["calibration_status"] == "pending"
            assert profile["physical_gt_verified"] is False
            assert isinstance(
                build_relation_evaluator(
                    task,
                    profile,
                    None,
                    manifest["current_cad_scale_m_per_source_unit"],
                ),
                PendingRelationEvaluator,
            )

    bolt = yaml.safe_load((settings.ROOT / "config/bolt_insertion.yaml").read_text())
    bolt_calibration = bolt["evaluator_calibration"]
    assert bolt_calibration["status"] == "historical_scale_mismatch_pending_revalidation"
    assert bolt_calibration["cad_meters_per_source_unit"] != manifest[
        "current_cad_scale_m_per_source_unit"
    ]
    assert not physical_calibration_matches_scale(
        bolt_calibration, manifest["current_cad_scale_m_per_source_unit"]
    )


def test_manifest_covers_each_source_combine_step_exactly_once():
    import yaml

    manifest = json.loads(
        (settings.ROOT / "config/gearbox_single_step_tasks.json").read_text()
    )
    source_paths = [
        (settings.ROOT / manifest["source_sequence"]).resolve(),
        (settings.ROOT / manifest["source_subassemblies"]).resolve(),
    ]
    if not all(path.exists() for path in source_paths):
        pytest.skip("Canonical assembly source YAML is not available in this checkout")
    source_steps = []
    for path in source_paths:
        source_steps.extend(yaml.safe_load(path.read_text()).get("steps", []))
    source_ids = {step["id"] for step in source_steps}
    expected = {step["id"] for step in source_steps if step.get("action") == "combine"}
    excluded = {
        step["id"]
        for step in source_steps
        if step.get("action") in set(manifest["excluded_scene_setup_actions"])
    }
    mapped = [step for task in manifest["tasks"] for step in task["source_steps"]]
    assert set(mapped) == expected
    assert len(mapped) == len(set(mapped))
    assert expected.isdisjoint(excluded)
    assert expected | excluded == source_ids


def test_pending_physical_evaluator_records_measurements_without_success():
    from relation_evaluator import PendingRelationEvaluator, RelationMeasurement

    measurement = RelationMeasurement(
        simulation_time_s=4.0,
        moving_position_world_m=[0.1, 0.2, 0.3],
        moving_quaternion_world_wxyz=[1.0, 0.0, 0.0, 0.0],
        fixture_position_world_m=[0.0, 0.0, 0.0],
        fixture_quaternion_world_wxyz=[1.0, 0.0, 0.0, 0.0],
        moving_linear_velocity_world_mps=[0.0, 0.0, 0.0],
        moving_angular_velocity_world_radps=[0.0, 0.0, 0.0],
        released=True,
        gripper_contact_forces_n=[0.0, 0.0],
        fixture_contact_force_n=3.0,
        maximum_penetration_m=0.0,
        stable_window_s=0.5,
        stable=True,
        pickup_observed=True,
    )
    result = PendingRelationEvaluator(
        "m10-casing-bolt", "m10_casing_bolt_evaluator_calibration.json"
    ).observe(measurement)

    assert result["task_success"] is False
    assert result["evaluator_status"] == "pending_calibration"
    assert result["physical_gt_verified"] is False
    assert result["measurement"]["fixture_contact_force_n"] == 3.0
