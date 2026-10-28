from io import BytesIO
import itertools
import json

import numpy as np
from PIL import Image
import pytest

from public_cad_preview import render_public_cad_preview


def samples():
    points = np.array(list(itertools.product((-.02, .02), (-.03, .03), (-.04, .04))))
    return points, points / np.linalg.norm(points, axis=1)[:, None]


def test_png_metadata_and_deterministic_local_reference():
    points, normals = samples()
    png, metadata = render_public_cad_preview(points, normals)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    image = Image.open(BytesIO(png))
    assert image.format == "PNG"
    assert image.size == (1000, 860)
    assert metadata["image_size_px"] == list(image.size)
    np.testing.assert_allclose(metadata["sample_extents_m"], [.04, .06, .08])
    np.testing.assert_allclose(metadata["sample_bounds_m"], [points.min(0), points.max(0)])
    assert metadata["sample_count"] == 8
    assert len(metadata["views"]) == 4
    assert all(view["front_facing_samples"] > 0 for view in metadata["views"])
    assert "not live camera or scene pose" in metadata["reference"]
    assert "points" not in metadata and "normals" not in metadata
    json.dumps(metadata)
    assert render_public_cad_preview(points, normals)[0] == png
    assert render_public_cad_preview(points[::-1], normals[::-1])[0] == png
    pixels = np.asarray(image)[130:420, 20:480].astype(int)
    assert np.count_nonzero((pixels[:, :, 2] - pixels[:, :, 0]) > 20) > 0


def test_opposite_views_keep_local_geometry_but_reverse_viewer():
    points,normals=samples()
    _,front=render_public_cad_preview(points,normals)
    png,back=render_public_cad_preview(points,normals,opposite_side=True)
    assert back['sample_bounds_m']==front['sample_bounds_m']
    for first,second in zip(front['views'],back['views']):
        np.testing.assert_allclose(second['viewer_direction_local'],-np.array(first['viewer_direction_local']))
        assert '-' in second['label']
    assert png.startswith(b'\x89PNG')


def test_ring_cavity_is_not_filled_in_top_view():
    angles = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    ring = np.column_stack((.03 * np.cos(angles), .03 * np.sin(angles), np.full(240, .004)))
    points = np.concatenate((ring, ring * [1, 1, -1]))
    normals = np.concatenate((np.tile([0, 0, 1.], (240, 1)), np.tile([0, 0, -1.], (240, 1))))
    png, _ = render_public_cad_preview(points, normals)
    pixels = np.asarray(Image.open(BytesIO(png)))
    # Top view panel's projected local origin remains background, not hull fill.
    assert np.all(pixels[640:658, 244:262] == [252, 252, 248])
    top = pixels[510:800, 20:480].astype(int)
    assert np.count_nonzero(top[:, :, 2] - top[:, :, 0] > 20) > 100


@pytest.mark.parametrize("case", ["shape", "empty", "mismatch", "nan", "inf_normal", "zero_normal", "flat"])
def test_malformed_inputs(case):
    points, normals = samples()
    if case == "shape":
        points = points[:, :2]
    elif case == "empty":
        points, normals = np.empty((0, 3)), np.empty((0, 3))
    elif case == "mismatch":
        normals = normals[:-1]
    elif case == "nan":
        points[0, 0] = np.nan
    elif case == "inf_normal":
        normals[0, 0] = np.inf
    elif case == "zero_normal":
        normals[0] = 0
    elif case == "flat":
        points[:, 2] = 0
    with pytest.raises(ValueError):
        render_public_cad_preview(points, normals)
