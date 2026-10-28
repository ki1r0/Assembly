import task_profiles
import json
import numpy as np
from scipy.spatial.transform import Rotation
import settings
from fixture_geometry import bearing_middle_probe_pose
import pytest


def test_input_top_bearing_instruction_names_middle_not_end_shoulder(monkeypatch):
    monkeypatch.setenv('ASSEMBLY_TASK', 'bearing-input-top-onto-shaft')
    prompt = task_profiles.instruction()
    assert 'middle shoulder' in prompt
    assert 'not the first shoulder near the exposed shaft end' in prompt
    assert 'Do not use historical fit coordinates' in prompt
    assert 'force the bearing past an incompatible step' in prompt
    assert '2.732481' not in prompt


def test_middle_seat_fixture_exposes_the_downward_leading_wide_face():
    profile = json.loads((settings.ROOT/'config/task_evaluator_profiles.json').read_text())['bearing-input-top-onto-shaft']
    fixture = Rotation.from_euler('xyz', profile['fixture_rotation_xyz_deg'], degrees=True)
    relative = Rotation.from_euler('z', -90, degrees=True)
    assert np.allclose(fixture.apply([0, 1, 0]), [0, 0, 1])
    assert np.allclose((fixture*relative).apply([1, 0, 0]), [0, 0, -1])


def test_middle_probe_negatives_change_only_declared_initial_geometry():
    fixture = Rotation.from_euler('x', 90, degrees=True)
    p, q = bearing_middle_probe_pose([.55, -.15, .78], fixture, 'bearing-middle-seat-probe')
    offset, oq = bearing_middle_probe_pose([.55, -.15, .78], fixture, 'bearing-middle-offset-probe')
    flipped, fq = bearing_middle_probe_pose([.55, -.15, .78], fixture, 'bearing-middle-flipped-probe')
    assert np.allclose(np.array(offset)-p, [.025, 0, 0])
    assert np.allclose(oq, q)
    assert np.allclose(flipped, p)
    rotation = Rotation.from_quat(np.array(fq)[[1, 2, 3, 0]])
    assert np.allclose(rotation.apply([1, 0, 0]), [0, 0, 1])
    with pytest.raises(ValueError, match='Unsupported'):
        bearing_middle_probe_pose([0, 0, 0], fixture, 'unknown')
