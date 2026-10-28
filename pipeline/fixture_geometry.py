"""Scale-aware placement of the bolt task's private fixture geometry."""

from __future__ import annotations

from math import isclose, isfinite
from typing import Mapping, Sequence


def bolt_fixture_geometry(
    calibration: Mapping[str, object],
    assembly_frame_pos_m: Sequence[float],
    cad_meters_per_source_unit: float,
) -> dict[str, object]:
    """Resolve source-coordinate fixture datums into one current world frame."""
    scale = float(cad_meters_per_source_unit)
    frame = tuple(float(value) for value in assembly_frame_pos_m)
    if not isfinite(scale) or scale <= 0.0 or len(frame) != 3 or not all(isfinite(v) for v in frame):
        raise ValueError("scale and assembly frame must be finite; scale must be positive")
    if not isclose(float(calibration.get("cad_meters_per_source_unit", -1.0)), scale, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("fixture calibration scale does not match current CAD scale")

    hole_offset = tuple(float(value) for value in calibration["selected_hole_offset_from_assembly_frame_source_units"])
    z_offsets = {
        "cap_support_plane_z_m": float(calibration["cap_support_plane_offset_from_assembly_frame_source_units"]),
        "casing_entry_plane_z_m": float(calibration["casing_entry_plane_offset_from_assembly_frame_source_units"]),
        "casing_interior_floor_z_m": float(calibration["casing_interior_floor_offset_from_assembly_frame_source_units"]),
    }
    if len(hole_offset) != 2 or not all(isfinite(value) for value in (*hole_offset, *z_offsets.values())):
        raise ValueError("fixture source-coordinate offsets must be finite")
    return {
        "selected_hole_xy_m": (frame[0] + hole_offset[0] * scale, frame[1] + hole_offset[1] * scale),
        **{name: frame[2] + offset * scale for name, offset in z_offsets.items()},
        "assembly_frame_pos_m": frame,
    }


def physical_calibration_matches_scale(calibration: Mapping[str, object], scale: float) -> bool:
    """Only allow native success from physical calibration for the active CAD scale."""
    try:
        recorded_scale = float(calibration["cad_meters_per_source_unit"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        isclose(recorded_scale, float(scale), rel_tol=0.0, abs_tol=1e-12)
        and calibration.get("physical_positive") is True
        and calibration.get("physical_negatives_rejected") is True
    )


def bearing_middle_probe_pose(fixture_position, fixture_rotation, mode):
    """Private initial-pose probes; never an actor target or runtime correction."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    if mode not in {'bearing-middle-seat-probe', 'bearing-middle-offset-probe', 'bearing-middle-flipped-probe'}:
        raise ValueError('Unsupported bearing probe')
    position = np.asarray(fixture_position) + fixture_rotation.apply([0., .002732481, 0.]) + [0., 0., .002]
    rotation = fixture_rotation * Rotation.from_euler('z', -90, degrees=True)
    if mode == 'bearing-middle-offset-probe':
        position[0] += .025
    if mode == 'bearing-middle-flipped-probe':
        rotation = Rotation.from_euler('x', 180, degrees=True) * rotation
    return position.tolist(), rotation.as_quat()[[3, 0, 1, 2]].tolist()


def m10_seat_probe_geometry(casing_cache, bolt_cache):
    """Private diagnostic: fit one CAD bore and its adjacent supporting flange."""
    import numpy as np

    with np.load(casing_cache) as data:
        points, normals = data['points'], data['normals']
    # Source socket 1 only selects the neighbourhood; actual seat comes from CAD.
    radius = np.linalg.norm(points[:, :2] - [.060, .120], axis=1)
    wall = points[(radius < .007) & (abs(normals[:, 2]) < .1), :2]
    face = points[(radius > .007) & (radius < .012) & (normals[:, 2] > .99), 2]
    if len(wall) < 12 or len(face) < 6:
        raise ValueError('M10 probe lacks bore-wall or flange samples')
    fit = np.linalg.lstsq(np.c_[2*wall, np.ones(len(wall))], (wall*wall).sum(1), rcond=None)[0]
    bore_radius = float(np.sqrt(fit[2] + fit[:2] @ fit[:2]))
    with np.load(bolt_cache) as data:
        p, n = data['points'], data['normals']
    # The broad underside is the positive-Y face opposite the cap's outer top.
    underside = p[(n[:, 1] < -.99) & (p[:, 1] > 0.), 1]
    if not len(underside) or not .005 < bore_radius < .007:
        raise ValueError('M10 probe geometry inconsistent with this CAD')
    cap_y = float(np.median(underside))
    support_z = float(np.median(face))
    return {'root_offset_m': [*fit[:2].tolist(), support_z-cap_y],
            'bore_radius_m': bore_radius, 'flange_z_m': support_z,
            'cap_underside_local_y_m': cap_y,
            'shank_length_m': cap_y-float(p[:, 1].min())}
