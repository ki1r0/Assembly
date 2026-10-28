"""Data and prompt checks for the pending gear-to-shaft task profiles."""
import json

import pytest

import settings
from task_profiles import instruction, public_skill_specification


GEAR_TASKS = {
    "output-gear-onto-shaft": {
        "fixture_asset": "parts/Output Shaft.usd",
        "candidate_expected_root_offset_m": [0.0, 0.0, 0.0155],
    },
    "transfer-gear-onto-shaft": {
        "fixture_asset": "parts/Transfer Shaft.usd",
        "candidate_expected_root_offset_m": [0.0, 0.0, 0.0221],
    },
}


@pytest.mark.parametrize("task,expected", GEAR_TASKS.items())
def test_gear_prompt_and_public_skill_are_task_specific(monkeypatch, task, expected):
    monkeypatch.setenv("ASSEMBLY_TASK", task)
    prompt = instruction()
    public = public_skill_specification()

    assert task.split("-")[0] in prompt
    assert "slide it along the shaft" in prompt
    assert "historical fit poses" in prompt
    assert public["task_family"] == "shaft_subassembly"
    assert public["moving_feature_local_m"] == [0.0, 0.0, 0.0]
    assert public["moving_feature_axis_local"] == [0.0, 0.0, 1.0]

    private = json.loads(
        (settings.ROOT / "config/task_evaluator_profiles.json").read_text()
    )[task]
    assert private["fixture_asset"] == expected["fixture_asset"]
    assert private["fixture_world_xy_m"] == [0.55, -0.15]
    assert private["candidate_expected_root_offset_m"] == expected[
        "candidate_expected_root_offset_m"
    ]
    assert private["candidate_expected_rotation_xyz_deg"] == [0.0, 0.0, 0.0]
    assert private["evaluation_status"] == "pending"
    assert private["calibration_status"] == "pending"
    assert private["physical_gt_verified"] is False

    for key in (
        "fixture_asset",
        "fixture_rotation_xyz_deg",
        "fixture_world_xy_m",
        "candidate_expected_root_offset_m",
        "candidate_expected_rotation_xyz_deg",
        "candidate_reset_hint",
        "calibration_file",
        "evaluation_status",
    ):
        assert key not in public
