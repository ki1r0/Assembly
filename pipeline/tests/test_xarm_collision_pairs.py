from pathlib import Path
import pytest
from xarm_robot_cfg import official_collision_pairs


def test_only_measured_linkage_pairs_use_official_never_exclusions():
    source=Path(__file__).resolve().parents[1]/'third_party/xarm_ros2/xarm_moveit_config/srdf/_xarm7_macro.srdf.xacro'
    if not source.exists():pytest.skip('Official robot description unavailable')
    pairs=official_collision_pairs(source)
    never={frozenset((a,b)) for a,b,reason in pairs if reason=='Never'}
    assert never=={frozenset(('left_inner_knuckle','left_outer_knuckle')),
                   frozenset(('right_inner_knuckle','right_outer_knuckle'))}
    assert not any(frozenset((a,b))==frozenset(('left_finger','right_finger')) for a,b,_ in pairs)
    assert not any(a.startswith('/World') or b.startswith('/World') for a,b,_ in pairs)
