import json

import pytest

import settings
from task_profiles import SHAFT_TASKS, instruction, public_skill_specification


SHAFTS = {
    "output-shaft-into-base": ("Output Shaft.usd", "shaft_output.npz", [0.0, 1.0, 0.0]),
    "transfer-shaft-into-base": ("Transfer Shaft.usd", "shaft_transfer.npz", [1.0, 0.0, 0.0]),
    "input-shaft-into-base": ("Input Shaft.usd", "shaft_input.npz", [0.0, 1.0, 0.0]),
}


@pytest.mark.parametrize("task,expected", SHAFTS.items())
def test_shaft_task_prompt_public_geometry_and_private_pending_profile(monkeypatch, task, expected):
    asset, cache, axis = expected
    # Offline root-CAD metadata; episode-specific compound paths are covered by
    # test_shaft_compound_integration, independent of other tests' environments.
    monkeypatch.delenv("ASSEMBLY_RUN_DIR", raising=False)
    monkeypatch.setenv("ASSEMBLY_TASK", task)
    public = public_skill_specification()
    prompt = instruction()
    private = json.loads(
        (settings.ROOT / "config/task_evaluator_profiles.json").read_text()
    )[task]

    assert task in SHAFT_TASKS
    assert "casing base" in prompt
    assert "historical reset poses" in prompt
    assert public["moving_cad_file"] == asset
    assert public["moving_cad_cache"] == cache
    assert public["moving_feature_local_m"] == [0.0, 0.0, 0.0]
    assert public["moving_feature_axis_local"] == axis
    assert private["fixture_asset"] == "parts/Casing Base.usd"
    assert private["calibration_status"] == "pending"
    assert private["physical_gt_verified"] is False
    for key in ("fixture_asset", "fixture_world_xy_m", "candidate_reset_hint", "calibration_file"):
        assert key not in public


def test_input_shaft_mass_profile_is_explicitly_provisional_for_open_mesh(tmp_path):
    import numpy as np
    from tools.calibrate_part_mass import cad_mass_profile

    cache = tmp_path / "open-shaft.npz"
    np.savez(
        cache,
        volume_m3=np.asarray(7.3e-4),
        uniform_density_com_local_m=np.zeros(3),
        inertia_per_mass_local_m2=np.eye(3),
        closed_mesh_validated=np.asarray(False),
        collision_geometry_selected=np.asarray(True),
        geometry_scale_m_per_coordinate=np.asarray(0.001),
    )
    with pytest.raises(ValueError, match="closed-mesh validation"):
        cad_mass_profile(cache, 7800, "steel-density-assumption")

    provisional = cad_mass_profile(
        cache, 7800, "steel-density-assumption", allow_open_mesh=True
    )
    assert provisional["mass_estimate_status"] == "provisional_density_assumption_from_nonwatertight_collision_mesh"
    assert provisional["closed_mesh_validated"] is False
