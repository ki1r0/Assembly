import numpy as np
import pytest
from assembly_skills import observed_object_envelope


def test_envelope_is_computed_only_from_actual_visible_points():
    points=np.array([[x,y,z] for x in [0.,.01,.02] for y in [0.,.03] for z in [0.,.04]])
    result=observed_object_envelope(points)
    assert np.allclose(result['position_m'],points.mean(0))
    assert np.allclose(result['bounds_world_m'],[points.min(0),points.max(0)])
    assert np.allclose(result['position_covariance_m2'],np.cov(points,rowvar=False)+np.eye(3)*1e-6)
    assert not result['complete_volume_observed']
    assert result['point_count']==12


def test_sparse_or_invalid_geometry_cannot_become_an_object_envelope():
    for points in (np.zeros((11,3)),np.full((12,3),np.nan)):
        with pytest.raises(ValueError,match='measured'):
            observed_object_envelope(points)
