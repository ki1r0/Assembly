"""Static CAD/init integration only: no Isaac application, GPU or simulation."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest

import settings
from shaft_compound import build_shaft_compound, compound_cad_cache, prepare_initial_shaft_compound
from task_profiles import (SHAFT_TASKS, public_skill_specification, shaft_compound_paths,
                           shaft_compound_specification, skill_specification)
from tools.calibrate_part_mass import cad_mass_profile

pytest.importorskip('pxr')
from pxr import Gf, Usd, UsdGeom, UsdPhysics

SOURCE = Path('/home/sunsiliang/Human-AI-Collab')
CACHES = settings.ROOT / 'assets/public_cad_models'
MEMBER_CACHES = {
    'Output_Shaft': 'shaft_output', 'Transfer_Shaft': 'shaft_transfer', 'Input_Shaft': 'shaft_input',
    'Output_Gear': 'gear_output', 'Transfer_Gear': 'gear_transfer',
    'Bearing_Output_Top': 'bearing_32013', 'Bearing_Output_Bottom': 'bearing_32013',
    'Bearing_Transfer': 'bearing_32205_transfer', 'Bearing_Input_Bottom': 'bearing_30304',
    'Bearing_Input_Top': 'bearing_32205_input',
}


@pytest.fixture
def local_assets():
    if not (SOURCE / 'assets/parts/Output Shaft.usd').exists() or not (CACHES / 'shaft_output.npz').exists():
        pytest.skip('Local public CAD and existing member caches unavailable')
    return SOURCE / 'assets'


def root_mass(task, density):
    kind = task.split('-')[0]
    cache = CACHES / f'shaft_{kind}.npz'
    return cache, cad_mass_profile(cache, density, 'custom-density-assumption',
                                  allow_open_mesh=kind == 'input')


@pytest.mark.parametrize('task', sorted(SHAFT_TASKS))
def test_public_recipe_matches_original_source_not_casing_layout(local_assets, task):
    _, original = build_shaft_compound(task, local_assets, SOURCE)
    stage, public = build_shaft_compound(task, local_assets,
                                         compound_spec=shaft_compound_specification(task))
    assert public == original
    assert stage.GetDefaultPrim().GetPath() == '/Assembly'
    text = json.dumps(shaft_compound_specification(task))
    assert all(word not in text for word in ('Casing_Base', 'fixture_world', 'candidate_reset', '-40.0'))


@pytest.mark.parametrize('task', sorted(SHAFT_TASKS))
def test_effective_geometry_mass_com_inertia_and_actor_cache(local_assets, tmp_path, monkeypatch, task):
    density = 6150.  # Must not be silently replaced by default steel density.
    legacy_cache, legacy = root_mass(task, density)
    result = prepare_initial_shaft_compound(task, local_assets, tmp_path, density,
        legacy['material'], allow_open_mesh=task.startswith('input'))
    profile, provenance = result['mass_profile'], result['provenance']
    assert profile['density_kg_m3'] == pytest.approx(density)
    assert profile['material'] == legacy['material']
    assert profile['mass_kg'] > legacy['mass_kg']
    assert profile['calibration_status'] == 'pending'
    assert 'missing_bearing_plugs' in profile['assembly_placement_status']
    assert profile['closed_mesh_validated'] is (not task.startswith('input'))
    assert ('provisional' in profile['mass_estimate_status']) == task.startswith('input')
    # Independent reference: transform EXISTING member cache properties, then
    # sum masses and apply the parallel-axis theorem. No aggregate seam test.
    masses, centers, inertias = [], [], []
    for member in provenance['members']:
        with np.load(CACHES / (MEMBER_CACHES[member['name']] + '.npz')) as raw:
            transform = np.array(member['shaft_from_component_row_source_units'])
            rotation = transform[:3, :3].T
            masses.append(float(raw['volume_m3']) * density)
            centers.append(rotation @ raw['uniform_density_com_local_m'] + transform[3, :3] * .001)
            inertias.append(rotation @ raw['inertia_per_mass_local_m2'] @ rotation.T)
    total = sum(masses)
    center = np.average(centers, axis=0, weights=masses)
    inertia = np.zeros((3, 3))
    for mass, com, local_inertia in zip(masses, centers, inertias):
        delta = com - center
        inertia += mass * (local_inertia + np.dot(delta, delta)*np.eye(3) - np.outer(delta, delta))
    assert profile['mass_kg'] == pytest.approx(total, rel=1e-10)
    np.testing.assert_allclose(profile['uniform_density_com_local_m'], center, atol=1e-12)
    np.testing.assert_allclose(np.array(profile['inertia_per_mass_local_m2'])*total, inertia, atol=1e-12)
    stage = Usd.Stage.Open(result['asset_path'])
    root = stage.GetDefaultPrim()
    assert sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse()) == 1
    assert sum(p.HasAPI(UsdPhysics.MassAPI) for p in stage.Traverse()) == 1
    mass_api = UsdPhysics.MassAPI(root)
    assert mass_api.GetMassAttr().Get() == pytest.approx(total, rel=1e-6)
    np.testing.assert_allclose(np.array(mass_api.GetCenterOfMassAttr().Get())*.001, center, atol=1e-8)
    principal = mass_api.GetPrincipalAxesAttr().Get()
    rotation = np.array(Gf.Matrix3d(Gf.Rotation(Gf.Quatd(principal)))).T
    diagonal = np.array(mass_api.GetDiagonalInertiaAttr().Get()) * .001**2
    np.testing.assert_allclose(rotation @ np.diag(diagonal) @ rotation.T, inertia, atol=1e-8)
    # Reproduce the backend's one-time scale with an ordinary USD reference.
    world = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Xform.Define(world, '/World/Cover')
    prim.GetPrim().GetReferences().AddReference(result['asset_path'])
    prim.AddScaleOp().Set((.001, .001, .001))
    cache = UsdGeom.XformCache()
    vertices = []
    for p in world.Traverse():
        if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI):
            mesh = np.array(UsdGeom.Mesh(p).GetPointsAttr().Get())
            tr = np.array(cache.GetLocalToWorldTransform(p)).T
            vertices.append(mesh @ tr[:3, :3].T + tr[:3, 3])
    vertices = np.concatenate(vertices)
    monkeypatch.setenv('ASSEMBLY_TASK', task)
    monkeypatch.setenv('ASSEMBLY_RUN_DIR', str(tmp_path))
    public = public_skill_specification()
    spec = skill_specification()
    assert Path(spec['moving_asset_path']) == Path(result['asset_path'])
    # This is exactly the path expression used by plan_grasp and view_public_cad.
    actor_cache = CACHES / public['moving_cad_cache']
    assert actor_cache == Path(result['cache_path'])
    with np.load(actor_cache) as data:
        assert data['points'].shape == data['normals'].shape == (16000, 3)
        np.testing.assert_allclose(data['uniform_density_com_local_m'], center, atol=1e-12)
        np.testing.assert_allclose(data['inertia_per_mass_local_m2']*total, inertia, atol=1e-12)
        np.testing.assert_allclose(data['bounds_extent_m'], np.ptp(vertices, axis=0), atol=1e-12)
        assert np.all(data['points'] >= vertices.min(0) - 1e-7)
        assert np.all(data['points'] <= vertices.max(0) + 1e-7)
        assert list(data['member_names']) == [m['name'] for m in provenance['members']]
    assert 'Assembly.usd' in public['moving_cad_file']
    assert all(key not in public for key in ('candidate_reset_hint', 'fixture_world_xy_m', 'calibration_file'))


def test_input_open_mesh_still_requires_explicit_existing_exception(local_assets, tmp_path):
    legacy_cache, legacy = root_mass('input-shaft-into-base', 7800)
    with pytest.raises(ValueError, match='closed-mesh validation'):
        prepare_initial_shaft_compound('input-shaft-into-base', local_assets, tmp_path,
                                       7800., legacy['material'])


def test_coincident_closed_members_remain_closed_and_additive(tmp_path):
    from tools.calibrate_part_mass import closed_mesh_validated
    stage = Usd.Stage.CreateInMemory()
    vertices = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=float)
    faces = np.array([[0,2,1], [0,1,3], [1,2,3], [2,0,3]])
    assert closed_mesh_validated(vertices[faces])
    # An aggregate edge count would wrongly reject these coincident components.
    assert not closed_mesh_validated(np.concatenate([vertices[faces]] * 3))
    names = ['Transfer_Shaft', 'Transfer_Gear', 'Bearing_Transfer']
    for name in names:
        mesh = UsdGeom.Mesh.Define(stage, '/Assembly/' + name + '/mesh')
        mesh.CreatePointsAttr(vertices.tolist())
        mesh.CreateFaceVertexCountsAttr([3]*4)
        mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    provenance = {'coordinate_scale_m': .001, 'input_shaft_open_mesh_provisional': False,
                  'members': [{'name': name} for name in names]}
    path = tmp_path / 'coincident.npz'
    properties = compound_cad_cache(stage, provenance, path)
    with np.load(path) as cache:
        assert bool(cache['closed_mesh_validated'])
        assert cache['member_closed_mesh_validated'].all()
        assert float(cache['volume_m3']) == pytest.approx(3 * 1e-9 / 6)
        assert float(cache['volume_m3']) == pytest.approx(sum(p['volume_m3'] for p in properties))


def test_backend_initialization_loads_prepared_compound_before_spawn(local_assets, tmp_path, monkeypatch):
    # Execute the real initialization branch, without importing/starting Isaac.
    task = 'output-shaft-into-base'
    monkeypatch.setenv('ASSEMBLY_TASK', task)
    monkeypatch.setenv('ASSEMBLY_RUN_DIR', str(tmp_path))
    monkeypatch.setenv('ASSEMBLY_COVER_MATERIAL', 'custom-density-assumption')
    legacy_cache, legacy = root_mass(task, 6050.)
    prepared = prepare_initial_shaft_compound(task, local_assets, tmp_path, 6050., legacy['material'])
    monkeypatch.setenv('ASSEMBLY_COVER_MASS_KG', str(prepared['mass_profile']['mass_kg']))
    monkeypatch.setenv('ASSEMBLY_COVER_MASS_STATUS', prepared['mass_profile']['mass_estimate_status'])
    tree = ast.parse((settings.ROOT / 'generic_backend.py').read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    branch = next(n for n in main.body if isinstance(n, ast.If) and
                  ast.unparse(n.test) == 'task in SHAFT_TASKS')
    reset = next(n for n in ast.walk(main) if isinstance(n, ast.Call) and
                 ast.unparse(n.func) == 'sim.reset')
    assert branch.lineno < reset.lineno

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    import os
    namespace = dict(task=task, SHAFT_TASKS=SHAFT_TASKS, assets=local_assets,
                     R=tmp_path, settings=settings, cover_mass=legacy['mass_kg'],
                     skill_data=skill_specification(), evaluator_profile={}, os=os, json=json, write=write)
    exec(compile(ast.Module(body=[branch], type_ignores=[]), '<backend-initialization>', 'exec'), namespace)
    effective = namespace['compound_initialization']
    assert namespace['cover_mass'] == effective['mass_profile']['mass_kg']
    assert float(os.environ['ASSEMBLY_COVER_MASS_KG']) == namespace['cover_mass']
    assert effective['mass_profile']['density_kg_m3'] == pytest.approx(6050.)
    assert effective == prepared
    assert json.loads((tmp_path / 'checks/moving-part-mass-profile.json').read_text()) == effective['mass_profile']
    # The actual unchanged spawn call must receive the effective spec, not raw JSON.
    call = next(n for n in main.body if isinstance(n, ast.Assign) and
                any(isinstance(t, ast.Name) and t.id == 'cover' for t in n.targets))
    captured = []
    namespace.update(part=lambda *args: captured.append(args), skill_data=skill_specification(),
                     cover_pos=[0,0,1], cover_quat=[1,0,0,0], is_relation_task=True)
    exec(compile(ast.Module(body=[call], type_ignores=[]), '<backend-spawn>', 'exec'), namespace)
    assert captured[0][1] == effective['asset_path']
    assert captured[0][-1] == effective['mass_profile']['mass_kg']


@pytest.mark.parametrize('task', sorted(SHAFT_TASKS))
def test_launcher_prepares_compound_directly_with_selected_density(local_assets, tmp_path, monkeypatch, task):
    from types import SimpleNamespace
    from tools.prepare_cad_cache import prepare_shaft_compound
    import os

    monkeypatch.setenv('ASSEMBLY_TASK', task)
    monkeypatch.setenv('ASSEMBLY_RUN_DIR', str(tmp_path))
    # No Docker/Isaac process is allowed in this static test. The installed host
    # USD runtime is used by prepare_shaft_compound, just as launch.py will do.
    def forbidden(*args, **kwargs):
        raise AssertionError('Static preparation unexpectedly attempted subprocess execution')
    monkeypatch.setattr('tools.prepare_cad_cache.subprocess.run', forbidden)
    tree = ast.parse((settings.ROOT / 'launch.py').read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    branch = next(n for n in main.body if isinstance(n, ast.If) and
                  ast.unparse(n.test) == 'args.task in SHAFT_TASKS')
    namespace = dict(args=SimpleNamespace(task=task, steel_density_kg_m3=6350.),
        SHAFT_TASKS=SHAFT_TASKS, prepare_shaft_compound=prepare_shaft_compound,
        skill_specification=skill_specification, assets=local_assets, run=tmp_path,
        evaluator_profiles={task: {'allow_provisional_open_mesh_mass': task.startswith('input')}},
        os=os)
    exec(compile(ast.Module(body=[branch], type_ignores=[]), '<launcher-compound>', 'exec'), namespace)
    assert namespace['cover_material']['density_kg_m3'] == pytest.approx(6350.)
    assert namespace['moving_asset'] == str(shaft_compound_paths(task, tmp_path)[0])
    assert namespace['moving_cache'] == str(shaft_compound_paths(task, tmp_path)[1])
    assert Path(namespace['moving_asset']).is_relative_to(tmp_path)
    assert Path(namespace['moving_cache']).exists()
    saved = json.loads((tmp_path / 'checks/initial-shaft-compound.json').read_text())
    assert saved['mass_profile'] == namespace['cover_material']
