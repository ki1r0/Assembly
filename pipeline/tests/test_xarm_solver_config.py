import sys
from types import ModuleType, SimpleNamespace

from xarm_robot_cfg import xarm7_config


def test_xarm_configuration_retains_self_collision_and_resolves_wrist_velocity(monkeypatch):
    modules={name:ModuleType(name) for name in
             ('isaaclab','isaaclab.sim','isaaclab.assets','isaaclab.actuators')}
    for name in ('UsdFileCfg','RigidBodyPropertiesCfg','ArticulationRootPropertiesCfg'):
        setattr(modules['isaaclab.sim'],name,lambda **kwargs:SimpleNamespace(**kwargs))
    class Configuration(SimpleNamespace):
        InitialStateCfg=SimpleNamespace
    modules['isaaclab.assets'].ArticulationCfg=Configuration
    modules['isaaclab.actuators'].ImplicitActuatorCfg=SimpleNamespace
    modules['isaaclab'].sim=modules['isaaclab.sim']
    for name,module in modules.items():monkeypatch.setitem(sys.modules,name,module)
    cfg=xarm7_config('official-model.usd')
    assert cfg.spawn.articulation_props.enabled_self_collisions is True
    assert cfg.spawn.rigid_props.disable_gravity is False
    assert cfg.spawn.articulation_props.solver_velocity_iteration_count==8
    assert cfg.actuators['arm'].stiffness==400.
    assert cfg.actuators['arm'].damping==40.
