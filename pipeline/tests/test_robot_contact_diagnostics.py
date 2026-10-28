import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def assignment(name):
    tree=ast.parse((Path(__file__).parents[1]/'generic_backend.py').read_text())
    return next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))


def test_each_rigid_body_has_its_own_filtered_sensor():
    paths=['/World/Robot/link1','/World/Robot/link2']
    filters=['/World/Table/geometry/mesh','/World/Bolt/collision_cap']
    namespace=dict(robot_contact_paths=paths,diagnostic_filter_paths=filters,
        ContactSensorCfg=lambda **kw:SimpleNamespace(**kw),ContactSensor=lambda cfg:cfg)
    exec(compile(ast.Module(body=[assignment('diagnostic_contacts')],type_ignores=[]),'contacts','exec'),namespace)
    sensors=namespace['diagnostic_contacts']
    assert list(sensors)==paths
    assert sensors[paths[0]] is not sensors[paths[1]]
    for path,sensor in sensors.items():
        assert sensor.prim_path==path and sensor.filter_prim_paths_expr==filters


def test_filter_discovery_includes_instanced_robot_collision_meshes():
    source=ast.unparse(assignment('diagnostic_filter_paths'))
    assert 'Usd.TraverseInstanceProxies()' in source
    assert "'/World/Robot/'" in source


def test_table_and_self_contacts_do_not_count_as_held_target_force():
    matrices=[np.array([[[[100.,0,0],[0,3.,4.],[0,200.,0]]]]),
              np.array([[[[200.,0,0],[0,0,2.],[0,400.,0]]]])]
    sensors={str(i):SimpleNamespace(data=SimpleNamespace(force_matrix_w=m)) for i,m in enumerate(matrices)}
    torch=SimpleNamespace(linalg=SimpleNamespace(vector_norm=lambda x,dim:np.linalg.norm(x,axis=dim)))
    namespace=dict(torch=torch,diagnostic_contacts=sensors,diagnostic_target_indices=[1])
    exec(compile(ast.Module(body=[assignment('robot_target_force')],type_ignores=[]),'force','exec'),namespace)
    assert np.isclose(namespace['robot_target_force'],7.)
