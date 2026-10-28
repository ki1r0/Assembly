"""Initialization-only compound shaft CAD; no simulation or runtime attachment.

Source fit offsets describe internal subassembly layout, never a casing target.
Missing bearing plugs are reported, not fabricated. Their source fit is used as
an explicit provisional CAD-root placement, not a successful strict combine.
The returned stage is unspawned and must only be consumed before scene reset.
"""
import ast
from pathlib import Path

import numpy as np
import yaml


_SHAFTS = {
    "output-shaft-into-base": ("Output_Shaft", "parts/Output Shaft.usd"),
    "transfer-shaft-into-base": ("Transfer_Shaft", "parts/Transfer Shaft.usd"),
    "input-shaft-into-base": ("Input_Shaft", "parts/Input Shaft.usd"),
}
_CHILDREN = {
    "Output_Gear": "parts/Output Gear.usd",
    "Transfer_Gear": "parts/Transfer Gear.usd",
    "Bearing_Output_Top": "CAD_edit/Bearing 32013 Inner Assy.usd",
    "Bearing_Output_Bottom": "CAD_edit/Bearing 32013 Inner Assy.usd",
    "Bearing_Transfer": "CAD_edit/Bearing 32205 Inner Assy.usd",
    "Bearing_Input_Bottom": "CAD_edit/Bearing 30304 Inner Assy.usd",
    "Bearing_Input_Top": "CAD_edit/Bearing 32205 Inner Assy.usd",
}


def _source_offsets(path):
    tree = ast.parse(Path(path).read_text())
    wanted = {"HARDCODED_FIT_OFFSETS", "HARDCODED_CHILD_LOCAL_TRANSLATE",
              "HARDCODED_CHILD_LOCAL_Z", "HARDCODED_CHILD_LOCAL_ROTATE_Y"}
    return {node.target.id: ast.literal_eval(node.value) for node in tree.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
            and node.target.id in wanted}


def build_shaft_compound(task, asset_root, source_root=None, meters_per_source_unit=.001,
                         compound_spec=None):
    """Return (in-memory USD stage, provenance) for one initial shaft assembly.

    Geometry stays in source coordinates, with metersPerUnit explicitly set.
    Existing backend spawning must still scale coordinates once, as for raw CAD.
    One dynamic RigidBodyAPI is authored at /Assembly; child rigid bodies, joints
    and placeholder masses are removed. Collision mesh selection honors ancestor
    APIs; visual meshes are collision fallback only if the source has no enabled
    colliders. No mass/inertia or physical calibration is invented here.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    if task not in _SHAFTS:
        raise ValueError("Only the three shaft-into-base initial assemblies are supported")
    if not np.isfinite(meters_per_source_unit) or meters_per_source_unit <= 0:
        raise ValueError("Finite positive source coordinate scale required")
    asset_root = Path(asset_root)
    parent_name, parent_file = _SHAFTS[task]
    if compound_spec is None:
        source_root = Path(source_root)
        steps = yaml.safe_load((source_root / "assembly/shaft_subassembly_steps.yaml").read_text())["steps"]
        steps = [s for s in steps if s["parent"] == parent_name]
        offsets = _source_offsets(source_root / "runtime/magic_assembly.py")
    else:
        if compound_spec['shaft_root'] != parent_name:
            raise ValueError("Compound recipe does not match shaft root")
        steps = [dict(id=m['source_step'], child=m['name'], parent=parent_name,
                      plug='plug_main', socket='socket_gear') for m in compound_spec['members']]
        offsets = {'HARDCODED_FIT_OFFSETS': {
            (m['name'], parent_name, 'plug_main', 'socket_gear'): m for m in compound_spec['members']}}
    shaft_kind = parent_name.split("_")[0]
    expected = {c for c in _CHILDREN if c.startswith(shaft_kind + "_") or
                c.startswith("Bearing_" + shaft_kind)}
    if {s["child"] for s in steps} != expected or len(steps) != len(expected):
        raise ValueError("Source sequence does not contain the expected shaft components")
    output = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(output, float(meters_per_source_unit))
    UsdGeom.SetStageUpAxis(output, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(output, "/Assembly").GetPrim()
    output.SetDefaultPrim(root)
    records = []

    def open_asset(relative):
        stage = Usd.Stage.Open(str(asset_root / relative))
        if stage is None or not stage.GetDefaultPrim():
            raise ValueError(f"Missing public CAD/default prim: {relative}")
        return stage

    parent_stage = open_asset(parent_file)

    def frame(stage, name):
        prim = stage.GetPrimAtPath(stage.GetDefaultPrim().GetPath().AppendChild(name))
        if not prim:
            return None
        cache = UsdGeom.XformCache()
        return cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(stage.GetDefaultPrim()).GetInverse()

    def add(name, relative, source, local, status, step_id=None):
        wrapper = UsdGeom.Xform.Define(output, f"/Assembly/{name}")
        wrapper.AddTransformOp().Set(local)
        destination = wrapper.GetPath().AppendChild("Geometry")
        if not Sdf.CopySpec(source.Flatten(), source.GetDefaultPrim().GetPath(),
                            output.GetRootLayer(), destination):
            raise ValueError(f"Unable to copy CAD {relative}")
        geometry = output.GetPrimAtPath(destination)
        meshes = [p for p in Usd.PrimRange(geometry) if p.IsA(UsdGeom.Mesh)]
        selected = []
        for mesh in meshes:
            p = mesh
            while p and p.GetPath().HasPrefix(destination):
                if p.HasAPI(UsdPhysics.CollisionAPI):
                    if UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get():
                        selected.append(mesh)
                    break
                p = p.GetParent()
        # Explicitly disabled colliders must not silently become visual fallback.
        has_collision_api = any(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(geometry))
        if has_collision_api and not selected:
            raise ValueError(f"Source has no enabled collision geometry: {relative}")
        fallback = not has_collision_api
        if fallback:
            selected = meshes
        if not selected:
            raise ValueError(f"No collision meshes: {relative}")
        for p in list(Usd.PrimRange(geometry)):
            if p.IsA(UsdPhysics.Joint):
                output.RemovePrim(p.GetPath())
                continue
            for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI, UsdPhysics.CollisionAPI,
                        UsdPhysics.MeshCollisionAPI, UsdPhysics.ArticulationRootAPI):
                if p.HasAPI(api):
                    p.RemoveAPI(api)
        for mesh in selected:
            UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr("sdf")
        records.append({"name": name, "asset": relative, "source_step": step_id,
                        "placement_status": status, "shaft_from_component_row_source_units":
                        np.asarray(local).tolist(), "collision_mesh_count": len(selected),
                        "collision_source": "visual_fallback" if fallback else "authored"})

    add(parent_name, parent_file, parent_stage, Gf.Matrix4d(1), "shaft_root")
    for step in steps:
        child = step["child"]
        key = (child, parent_name, step["plug"], step["socket"])
        cfg = offsets["HARDCODED_FIT_OFFSETS"][key]
        if any(key in offsets.get(name, {}) for name in offsets if name != "HARDCODED_FIT_OFFSETS"):
            raise ValueError(f"Additional source child override needs explicit handling: {child}")
        rx, ry, rz = cfg["rotate_xyz"]
        rotation = (Gf.Rotation(Gf.Vec3d(1, 0, 0), rx) *
                    Gf.Rotation(Gf.Vec3d(0, 1, 0), ry) *
                    Gf.Rotation(Gf.Vec3d(0, 0, 1), rz))
        fit = Gf.Matrix4d(1).SetRotate(rotation)
        fit.SetTranslateOnly(Gf.Vec3d(*cfg["translate"]))
        source = open_asset(_CHILDREN[child])
        plug, socket = frame(source, step["plug"]), frame(parent_stage, step["socket"])
        if socket is None:
            raise ValueError(f"Missing authored shaft socket for {child}")
        if plug is None:
            if not child.startswith("Bearing_") or not np.allclose(np.asarray(socket), np.eye(4)):
                raise ValueError(f"Cannot interpret missing plug for {child}")
            local, status = fit, "provisional_source_fit_as_CAD_root_missing_bearing_plug"
        else:
            local, status = plug.GetInverse() * fit * socket, "authored_plug_fit_socket"
        add(child, _CHILDREN[child], source, local, status, step["id"])
    UsdPhysics.RigidBodyAPI.Apply(root).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.RigidBodyAPI(root).CreateKinematicEnabledAttr(False)
    return output, {
        "task": task, "shaft_root": parent_name, "members": records,
        "coordinate_scale_m": float(meters_per_source_unit),
        "scope": "initial compound rigid CAD only; no casing target or runtime attachment",
        "calibration_status": "pending; source fits are not physically validated seats",
        "mass_status": "caller must supply compound mass/COM/inertia; source placeholder masses removed",
        "input_shaft_open_mesh_provisional": task == "input-shaft-into-base",
    }


def compound_cad_cache(stage, provenance, cache_path, sample_count=16000):
    """Sample the actual compound colliders and sum member solid properties.

    Mass is additive, not a boolean union. Closure is checked per member, never
    on coincident assembly seams. Input-shaft open geometry stays provisional.
    """
    from pxr import UsdGeom, UsdPhysics
    from tools.calibrate_part_mass import uniform_solid_properties

    scale = provenance['coordinate_scale_m']
    cache = UsdGeom.XformCache()
    all_triangles, properties = [], []
    for member in provenance['members']:
        triangles = []
        prefix = '/Assembly/' + member['name'] + '/'
        for prim in stage.Traverse():
            if not str(prim.GetPath()).startswith(prefix) or not prim.IsA(UsdGeom.Mesh):
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            mesh = UsdGeom.Mesh(prim)
            transform = np.asarray(cache.GetLocalToWorldTransform(prim)).T
            vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
            vertices = (vertices @ transform[:3, :3].T + transform[:3, 3]) * scale
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
            start = 0
            for size in mesh.GetFaceVertexCountsAttr().Get():
                face = indices[start:start + size]
                start += size
                for i in range(1, size - 1):
                    triangles.append(vertices[face[[0, i, i + 1]]])
        triangles = np.asarray(triangles, dtype=float)
        prop = uniform_solid_properties(triangles)
        if not prop['closed_mesh_validated'] and not (
                provenance['input_shaft_open_mesh_provisional'] and member['name'] == 'Input_Shaft'):
            raise ValueError(f"Unexpected nonwatertight compound member: {member['name']}")
        properties.append({'name': member['name'], **prop})
        all_triangles.append(triangles)
    volume = sum(p['volume_m3'] for p in properties)
    com = sum(p['volume_m3'] * p['uniform_density_com_local_m'] for p in properties) / volume
    inertia = np.zeros((3, 3))
    for prop in properties:
        delta = prop['uniform_density_com_local_m'] - com
        inertia += prop['volume_m3'] * (prop['inertia_per_mass_local_m2'] +
                    np.dot(delta, delta) * np.eye(3) - np.outer(delta, delta)) / volume
    if not np.isfinite(inertia).all() or np.linalg.eigvalsh(inertia).min() <= 0:
        raise ValueError('Compound inertia is not finite positive definite')
    triangles = np.concatenate(all_triangles)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1) / 2
    valid = areas > 0
    triangles, cross, areas = triangles[valid], cross[valid], areas[valid]
    rng = np.random.default_rng(18)
    choice = rng.choice(len(triangles), sample_count, p=areas / areas.sum())
    selected = triangles[choice]
    u, v = np.sqrt(rng.random((sample_count, 1))), rng.random((sample_count, 1))
    points = (1-u)*selected[:, 0] + u*(1-v)*selected[:, 1] + u*v*selected[:, 2]
    normals = cross[choice] / (2 * areas[choice, None])
    closed = all(p['closed_mesh_validated'] for p in properties)
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, points=points.astype(np.float32), normals=normals.astype(np.float32),
        bounds_extent_m=np.ptp(triangles.reshape(-1, 3), axis=0),
        volume_m3=volume, uniform_density_com_local_m=com, inertia_per_mass_local_m2=inertia,
        closed_mesh_validated=closed, collision_geometry_selected=True,
        collision_geometry_source='compound_selected_member_collision_meshes',
        geometry_scale_m_per_coordinate=scale,
        member_names=np.array([p['name'] for p in properties]),
        member_volumes_m3=np.array([p['volume_m3'] for p in properties]),
        member_closed_mesh_validated=np.array([p['closed_mesh_validated'] for p in properties]))
    return properties


def prepare_initial_shaft_compound(task, asset_root, run_dir, density_kg_m3,
                                   material, allow_open_mesh=False,
                                   meters_per_source_unit=.001):
    """Build effective initial USD/cache/mass before reset, without a simulator.

    Use the launcher's selected task density, including custom density, for all
    members. No bare-shaft cache/mass or unmounted source checkout is required.
    """
    from pxr import Gf, UsdPhysics
    from scipy.spatial.transform import Rotation
    from task_profiles import shaft_compound_paths, shaft_compound_specification
    from tools.calibrate_part_mass import cad_mass_profile

    if not np.isfinite(density_kg_m3) or density_kg_m3 <= 0:
        raise ValueError('Task density must be finite and positive')
    stage, provenance = build_shaft_compound(task, asset_root,
        meters_per_source_unit=meters_per_source_unit,
        compound_spec=shaft_compound_specification(task))
    usd_path, cache_path = shaft_compound_paths(task, run_dir)
    properties = compound_cad_cache(stage, provenance, cache_path)
    mass_profile = cad_mass_profile(cache_path, density_kg_m3, material, allow_open_mesh=allow_open_mesh)
    mass_profile.update(
        mass_source='sum of member CAD volumes times the unchanged task density; parallel-axis COM/inertia',
        member_properties=[{key: value.tolist() if isinstance(value, np.ndarray) else value
                            for key, value in prop.items()} for prop in properties],
        assembly_placement_status='provisional_source_fits_missing_bearing_plugs',
        calibration_status='pending')
    # USD values are in source-coordinate units. The existing spawner scales the
    # asset by metres/source-unit once, converting COM and inertia to SI units.
    mass = UsdPhysics.MassAPI.Apply(stage.GetDefaultPrim())
    mass.CreateMassAttr(mass_profile['mass_kg'])
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*(np.array(mass_profile['uniform_density_com_local_m']) / meters_per_source_unit)))
    inertia = np.array(mass_profile['inertia_per_mass_local_m2']) * mass_profile['mass_kg']
    diagonal, axes = np.linalg.eigh(inertia)
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    x, y, z, w = Rotation.from_matrix(axes).as_quat()
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*(diagonal / meters_per_source_unit**2)))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    stage.GetRootLayer().Export(str(usd_path))
    provenance['mass_status'] = mass_profile['mass_estimate_status']
    result = {'asset_path': str(usd_path), 'cache_path': str(cache_path),
              'mass_profile': mass_profile, 'provenance': provenance}
    import json
    (Path(run_dir) / 'checks/initial-shaft-compound.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
