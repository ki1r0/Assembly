import json

import pytest

import settings
from fixture_geometry import bolt_fixture_geometry, physical_calibration_matches_scale


def test_bolt_fixture_datums_resolve_from_source_units_at_current_scale():
    calibration = json.loads((settings.ROOT / "config/fixture_calibration.json").read_text())
    frame = (0.5, -0.15, 0.68 + 27.948326 * settings.CAD_METERS_PER_SOURCE_UNIT)

    geometry = bolt_fixture_geometry(calibration, frame, settings.CAD_METERS_PER_SOURCE_UNIT)

    assert geometry["selected_hole_xy_m"] == pytest.approx((0.4600026000045, -0.0698678023422))
    assert geometry["cap_support_plane_z_m"] == pytest.approx(0.74589665211084)
    assert geometry["casing_entry_plane_z_m"] == pytest.approx(0.735896652)
    assert geometry["casing_interior_floor_z_m"] == pytest.approx(0.71489648426416)
    assert geometry["assembly_frame_pos_m"] == frame


def test_fixture_geometry_rejects_stale_scale_data():
    calibration = json.loads((settings.ROOT / "config/fixture_calibration.json").read_text())

    with pytest.raises(ValueError, match="scale does not match"):
        bolt_fixture_geometry(calibration, (0.5, -0.15, 0.708), 0.002)


def test_old_physical_evaluator_calibration_cannot_grade_current_scale():
    import yaml

    task = yaml.safe_load((settings.ROOT / "config/bolt_insertion.yaml").read_text())
    old_calibration = task["evaluator_calibration"]
    assert physical_calibration_matches_scale(old_calibration, settings.CAD_METERS_PER_SOURCE_UNIT) is False
    assert physical_calibration_matches_scale(
        {**old_calibration, "cad_meters_per_source_unit": settings.CAD_METERS_PER_SOURCE_UNIT},
        settings.CAD_METERS_PER_SOURCE_UNIT,
    ) is True
    assert physical_calibration_matches_scale(
        {**old_calibration, "cad_meters_per_source_unit": settings.CAD_METERS_PER_SOURCE_UNIT,
         "physical_negatives_rejected": False},
        settings.CAD_METERS_PER_SOURCE_UNIT,
    ) is False


def test_m6_prompt_dimensions_follow_the_current_source_scale():
    import yaml

    task = yaml.safe_load((settings.ROOT / "config/bolt_insertion.yaml").read_text())
    dimensions = task["task_spec"]["nominal_dimensions"]

    assert dimensions["bolt_diameter_m"] == pytest.approx(2 * 2.941999216 * settings.CAD_METERS_PER_SOURCE_UNIT)
    assert dimensions["bolt_length_m"] == pytest.approx(22.00000104494393 * settings.CAD_METERS_PER_SOURCE_UNIT)

    skill = json.loads((settings.ROOT / "config/assembly_skills.json").read_text())["peg"]
    assert skill["moving_feature_local_m"][1] == pytest.approx(8.92500114440918 * settings.CAD_METERS_PER_SOURCE_UNIT)


def test_bolt_grasp_criterion_is_inside_current_cad_cap_bounds():
    import yaml
    from runtime.bolt_harness.measurement import (
        BOLT_TOP_LOCAL_Y_M,
        CAP_UNDERSIDE_LOCAL_Y_M,
        BoltSeatSamplingCriteria,
        CapSupportContactCriteria,
    )

    task = yaml.safe_load((settings.ROOT / "config/bolt_insertion.yaml").read_text())
    criteria = dict(task["sampling_criteria"])
    criteria["cap_support"] = CapSupportContactCriteria(**criteria["cap_support"])
    parsed = BoltSeatSamplingCriteria(**criteria)

    assert CAP_UNDERSIDE_LOCAL_Y_M <= parsed.min_cap_grasp_local_y_m <= BOLT_TOP_LOCAL_Y_M


def test_scale_sensitive_skill_datums_match_reexported_public_cad_planes():
    inspection = json.loads((settings.ROOT / "public-cad-inspection.json").read_text())
    skills = json.loads((settings.ROOT / "config/assembly_skills.json").read_text())

    assert inspection["_metadata"]["cad_meters_per_source_unit"] == settings.CAD_METERS_PER_SOURCE_UNIT
    bolt_planes = [plane for plane, _area in inspection["M6 Hub Bolt.usd"]["normal_y_planar_surface_area_m2"]]
    assert any(abs(skills["peg"]["moving_feature_local_m"][1] - plane) < 1e-6 for plane in bolt_planes)
    assert skills["hub-cover"]["moving_feature_local_m"][1] == pytest.approx(-0.0029705)
    cover_planes = [plane for plane, _area in inspection["Hub Cover Output.usd"]["normal_y_planar_surface_area_m2"]]
    assert any(abs(skills["hub-cover"]["moving_feature_local_m"][1] - plane) < 1e-6 for plane in cover_planes)
