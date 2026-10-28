import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hub_evaluator import datum_equivalent_root


@pytest.mark.parametrize('angle',[0,45,90,180,270])
def test_coaxial_mating_datum_is_independent_of_off_axis_cad_origin(angle):
    datum=np.array([-.000000571,-.0029705,-.00366934])
    reference=Rotation.from_euler('xyz',[-90,180,0],degrees=True).as_matrix()
    rotation=reference@Rotation.from_euler('y',angle,degrees=True).as_matrix()
    target=np.array([.5,-.10658141,.739])
    actual=target+reference@datum-rotation@datum
    np.testing.assert_allclose(datum_equivalent_root(actual,rotation,reference,datum),target,atol=1e-14)


def test_rotating_about_wrong_root_does_not_hide_real_bore_offset():
    datum=np.array([-.000000571,-.0029705,-.00366934])
    reference=Rotation.from_euler('xyz',[-90,180,0],degrees=True).as_matrix()
    rotated=reference@Rotation.from_euler('y',180,degrees=True).as_matrix()
    root=np.array([.5,-.10658141,.739])
    effective=datum_equivalent_root(root,rotated,reference,datum)
    assert np.linalg.norm(effective[:2]-root[:2])>.007
    assert np.linalg.norm(effective[:2]-root[:2])>.004


def test_nonfinite_datum_is_not_a_valid_measurement():
    with pytest.raises(ValueError):datum_equivalent_root([0,0,0],np.eye(3),np.eye(3),[np.nan,0,0])
