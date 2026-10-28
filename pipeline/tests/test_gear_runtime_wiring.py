import json

import numpy as np
import pytest

from tools.calibrate_part_mass import cad_mass_profile


def test_gear_mass_uses_transformed_public_cad_and_declared_density(tmp_path):
    cache = tmp_path / "gear.npz"
    np.savez(
        cache,
        volume_m3=np.asarray(1e-4),
        uniform_density_com_local_m=np.asarray([0.01, -0.02, 0.03]),
        inertia_per_mass_local_m2=np.eye(3) * 1e-4,
        closed_mesh_validated=np.asarray(True),
        collision_geometry_selected=np.asarray(True),
        geometry_scale_m_per_coordinate=np.asarray(0.001),
    )

    profile = cad_mass_profile(cache, 7800, "steel-density-assumption")

    assert profile["mass_kg"] == 0.78
    assert profile["volume_m3"] == 1e-4
    assert "assumption" in profile["material_status"]
    assert "placeholder" in profile["mass_source"]
    assert profile["collision_geometry_selected"] is True


def test_public_cad_volume_marks_open_mesh_unvalidated():
    from tools.calibrate_part_mass import uniform_solid_properties

    open_tetrahedron = np.asarray(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, 0, 0], [0, 0, 1], [1, 0, 0]],
        ],
        dtype=float,
    )
    assert uniform_solid_properties(open_tetrahedron)["closed_mesh_validated"] is False


def test_gear_mass_rejects_unclosed_geometry_cache(tmp_path):
    cache = tmp_path / "unclosed.npz"
    np.savez(
        cache,
        volume_m3=np.asarray(1e-4),
        uniform_density_com_local_m=np.zeros(3),
        inertia_per_mass_local_m2=np.eye(3),
        closed_mesh_validated=np.asarray(False),
        collision_geometry_selected=np.asarray(True),
        geometry_scale_m_per_coordinate=np.asarray(0.001),
    )
    with pytest.raises(ValueError, match="closed-mesh validation"):
        cad_mass_profile(cache, 7800, "steel-density-assumption")


def test_generic_public_observation_excludes_private_evaluator_fields():
    from generic_toolkit import RobotToolkit

    toolkit = object.__new__(RobotToolkit)
    toolkit.latest = {
        "task_instruction": "assemble the visible parts",
        "observation_id": "frame-1",
        "robot": "panda",
        "active_arm": "left",
        "arms": {"left": {"tcp_pose": [0.0] * 7}},
        "sensors": {"tcp_pose": [0.0] * 7, "native_success": True},
        "geometric_patterns": [],
        "features": [],
        "native_success": True,
        "result": {
            "task_success": True,
            "physical_gt_verified": True,
            "evaluator_status": "PASS",
            "measurement": {"fixture_contact_force_n": 5.0},
        },
    }

    public = toolkit.public_data()
    serialized = json.dumps(public)

    assert public["task"] == "assemble the visible parts"
    assert "native_success" not in serialized
    assert "task_success" not in serialized
    assert "physical_gt_verified" not in serialized
    assert "evaluator_status" not in serialized
    assert "fixture_contact_force_n" not in serialized


def test_generic_solved_state_uses_endpoint_not_private_observation(monkeypatch):
    from generic_toolkit import RobotToolkit
    import m1_reach.control

    toolkit = object.__new__(RobotToolkit)
    toolkit.latest = {"native_success": True}
    monkeypatch.setattr(m1_reach.control, "snapshot", lambda: {"terminal": None})
    assert toolkit.solved() is False

    monkeypatch.setattr(
        m1_reach.control,
        "snapshot",
        lambda: {"terminal": {"source": "executor_safety_stop"}},
    )
    assert toolkit.solved() is False

    monkeypatch.setattr(
        m1_reach.control,
        "snapshot",
        lambda: {"terminal": {"source": "native_success"}},
    )
    assert toolkit.solved() is True


def test_generic_rpc_does_not_attach_private_terminal_outcome_to_tool_result(monkeypatch, tmp_path):
    from generic_toolkit import RobotToolkit, ToolResult
    import generic_toolkit
    import m1_reach.control

    final_state = tmp_path / "episode/public/final-state.json"
    final_state.parent.mkdir(parents=True)
    final_state.write_text(
        json.dumps({
            "task_instruction": "task",
            "observation_id": "final",
            "native_success": True,
            "result": {"evaluator_status": "PASS"},
        })
    )
    toolkit = object.__new__(RobotToolkit)
    toolkit.run = tmp_path
    toolkit.latest = {"observation_id": "before"}
    monkeypatch.setattr(
        generic_toolkit.IsaacToolkit,
        "rpc",
        lambda _self, _command: ToolResult(data={"action_status": "reached"}),
    )
    monkeypatch.setattr(
        m1_reach.control,
        "snapshot",
        lambda: {"terminal": {"source": "native_success"}},
    )

    result = toolkit.rpc({"action": "move_to"})

    serialized = json.dumps(result.to_dict())
    assert result.data == {"action_status": "reached"}
    assert toolkit.latest["observation_id"] == "final"
    for private_key in ("native_success", "task_success", "evaluator_status", "stop_reason"):
        assert private_key not in serialized
