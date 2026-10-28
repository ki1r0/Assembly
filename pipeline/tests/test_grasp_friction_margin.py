import numpy as np
import pytest
from xarm_gripper import antipodal_contact_quality


def quality_for_required_friction(mu):
    normal = np.array([1.,mu,0.]);normal /= np.linalg.norm(normal)
    return antipodal_contact_quality([[1,0,0],[-1,0,0]],[normal,-normal])


def test_margin_cost_distinguishes_near_boundary_contacts_without_rejecting_them():
    near = quality_for_required_friction(.485)
    robust = quality_for_required_friction(.265)
    assert near['supported'] and robust['supported']
    assert near['friction_margin_cost'] > robust['friction_margin_cost'] + 2.
    assert quality_for_required_friction(0)['friction_margin_cost'] == pytest.approx(0)
    assert quality_for_required_friction(.51)['friction_margin_cost'] is None
