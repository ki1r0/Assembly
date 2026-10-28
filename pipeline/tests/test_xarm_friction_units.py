import math
from pathlib import Path
import pytest
import xml.etree.ElementTree as ET
from xarm_robot_cfg import urdf_arm_friction


def test_official_friction_is_nm_and_viscous_units_are_converted():
    urdf=Path(__file__).resolve().parents[1]/'assets/xarm7/xarm7_g1_simulation_assumption.urdf'
    if not urdf.exists():pytest.skip('Offline xArm7 model unavailable')
    records=urdf_arm_friction(urdf)
    assert len(records)==7
    source={joint.get('name'):joint for joint in ET.parse(urdf).getroot().findall('joint')}
    for name,record in records.items():
        assert record['static_friction_nm']==1.
        assert record['dynamic_friction_nm']==1.
        damping=float(source[name].find('dynamics').get('damping'))
        assert record['viscous_friction_nm_s_per_degree']==pytest.approx(damping*math.pi/180)
