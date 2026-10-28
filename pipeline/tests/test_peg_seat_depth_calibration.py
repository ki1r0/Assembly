import json

import yaml

import settings


def test_current_scale_seated_peg_can_meet_entry_depth_without_overdriving():
    fixture = json.loads((settings.ROOT / 'config/fixture_calibration.json').read_text())
    config = yaml.safe_load((settings.ROOT / 'config/bolt_insertion.yaml').read_text())
    scale = settings.CAD_METERS_PER_SOURCE_UNIT
    # Public authored tip and cap-underside local Y coordinates.
    shank = (8.92500114440918 - (-13.07499885559082)) * scale
    entry_gap = (fixture['cap_support_plane_offset_from_assembly_frame_source_units']
                 - fixture['casing_entry_plane_offset_from_assembly_frame_source_units']) * scale
    seated_entry = shank - entry_gap
    assert .0119 < seated_entry < .0121
    assert 0 < config['tolerances']['min_casing_entry_depth_m'] <= seated_entry
    assert config['evaluator_calibration']['status'] != 'calibrated'
