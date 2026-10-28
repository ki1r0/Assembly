import json
import pytest

import settings
from fixture_geometry import m10_seat_probe_geometry


@pytest.mark.cad_assets
def test_m10_flange_and_table_clearance_are_derived_from_current_cad():
    root = settings.ROOT
    g = m10_seat_probe_geometry(root/'assets/public_cad_models/casing_top.npz', root/'assets/public_cad_models/m10_bolt.npz')
    assert .0058 < g['bore_radius_m'] < .0061
    assert .0516 < g['cap_underside_local_y_m'] < .0520
    assert .109 < g['shank_length_m'] < .111
    profile = json.loads((root/'config/task_evaluator_profiles.json').read_text())['m10-casing-bolt']
    casing_bottom = -.027948326
    tip_above_table = -casing_bottom + .002 + profile['fixture_table_clearance_m'] + g['flange_z_m'] - g['shank_length_m']
    assert tip_above_table > .005
    assert profile['physical_gt_verified'] is False
