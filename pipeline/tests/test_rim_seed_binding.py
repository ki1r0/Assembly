import numpy as np
from rim_measurement import seed_matches_ellipse


def test_short_arc_cannot_masquerade_as_a_closed_rim():
    from rim_measurement import circle_boundary_supported
    angles=np.linspace(0,np.deg2rad(78),81)
    arc=np.c_[np.cos(angles),np.sin(angles)]*.045
    assert not circle_boundary_supported(np.r_[arc,arc[::-1]])
    angles=np.linspace(0,2*np.pi,80,endpoint=False)
    ring=np.c_[np.cos(angles),np.sin(angles)]*.045
    assert circle_boundary_supported(ring)
    assert not circle_boundary_supported(ring[:40])


def test_robust_circle_keeps_broad_support_and_reports_outliers():
    from rim_measurement import fit_rim_circle
    angles=np.linspace(0,2*np.pi,200,endpoint=False)
    ring=np.c_[np.cos(angles),np.sin(angles)]*.05
    contaminated=ring.copy();contaminated[::15]*=1.15
    fit=fit_rim_circle(contaminated)
    assert fit is not None
    centre,radius,rms,support,untrimmed=fit
    assert np.linalg.norm(centre)<.0002
    assert abs(radius-.05)<.0002 and rms<.001
    assert .9<support<1 and untrimmed>.001
    assert fit_rim_circle(contaminated[:40]) is None
    contaminated[::2]*=1.5
    assert fit_rim_circle(contaminated) is None


def test_development_episode_cannot_rebind_to_circle_37_pixels_away():
    assert not seed_matches_ellipse([469,392],[505.2045,398.5],[7.2984,9.0157],np.eye(2))


def test_seed_at_center_or_visible_rim_can_measure_that_opening():
    assert seed_matches_ellipse([469,392],[469,392],[6,10],np.eye(2))
    assert seed_matches_ellipse([472,392],[469,392],[6,10],np.eye(2))
    assert not seed_matches_ellipse([480,392],[469,392],[6,10],np.eye(2))


def test_residual_accounts_for_pixel_resolution_not_task_specific_geometry():
    from rim_measurement import ellipse_residual_supported
    assert ellipse_residual_supported(.10123,[5.0829,6.3592])
    assert not ellipse_residual_supported(.10123,[50.,63.])
    assert not ellipse_residual_supported(.5,[5.0829,6.3592])


def test_real_public_rgbd_keeps_selected_hole_and_its_rim_plane():
    import gzip
    import json
    from pathlib import Path
    from PIL import Image
    from rim_measurement import observed_rim
    root=Path(__file__).parent/'fixtures/cape_peg_rim_dev01'
    with gzip.open(root/'depth.npy.gz','rb') as stream:depth=np.load(stream)
    camera=json.loads((root/'camera.json').read_text())
    result=observed_rim(np.array(Image.open(root/'rgb.png')),depth,
        np.array(camera['intrinsics']),np.array(camera['world_from_camera']),[469,392],15)
    assert result is not None
    assert result['image_seed_distance_px']<2
    assert abs(result['position_m'][2]-.7459)<.0002
    assert .003<result['estimated_radius_m']<.0045
