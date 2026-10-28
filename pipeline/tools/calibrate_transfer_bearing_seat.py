"""Privileged, actor-independent transfer-bearing seat probe; never enables success.

Default mode only inspects CAD. --simulate explicitly starts a standalone Isaac
probe, following evaluator_probe.py's privileged diagnostic separation. No online
backend, actor, task profile or calibration file is modified.
"""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCALE = .001
DT = 1 / 120


@dataclass(frozen=True)
class Limits:
    radial_m: float = .00005
    tilt_deg: float = .2
    axial_gap_m: float = .0002
    penetration_m: float = .0001
    contact_region_m: float = .0002
    support_force_n: float = .1
    linear_speed_mps: float = .001
    angular_speed_radps: float = .01
    dwell_s: float = .5


def inspect_geometry(asset_root):
    """Read real hierarchy-transformed triangles, not cached point-cloud bounds."""
    from pxr import Usd, UsdGeom

    def triangles(relative):
        stage = Usd.Stage.Open(str(Path(asset_root) / relative))
        cache = UsdGeom.XformCache()
        result = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            p = np.array(mesh.GetPointsAttr().Get(), float)
            matrix = np.array(cache.GetLocalToWorldTransform(prim))
            p = (p @ matrix[:3, :3] + matrix[3, :3]) * SCALE
            indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
            offset = 0
            for size in mesh.GetFaceVertexCountsAttr().Get():
                face = indices[offset:offset + size]
                offset += size
                for j in range(1, size - 1):
                    result.append(p[face[[0, j, j + 1]]])
        t = np.asarray(result)
        normal = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        lengths = np.linalg.norm(normal, axis=1)
        valid = lengths > 0
        return t[valid], normal[valid] / lengths[valid, None]

    bearing, bn = triangles('CAD_edit/Bearing 32205 Inner Assy.usd')
    shaft, sn = triangles('parts/Transfer Shaft.usd')
    wide_x = float(bearing[:, :, 0].max())
    face = bearing[(bn[:, 0] > .99999) & (abs(bearing[:, :, 0].mean(1) - wide_x) < 1e-7)]
    # Select the negative-facing annulus adjacent to the negative end journal.
    shoulder = shaft[(sn[:, 0] < -.99999) & (shaft[:, :, 0].mean(1) > -.04)
                     & (shaft[:, :, 0].mean(1) < -.03)]
    if not len(face) or not len(shoulder):
        raise ValueError('Required wide bearing face / negative-X shoulder missing')
    shoulder_x = float(np.median(shoulder[:, :, 0]))
    face_r = np.linalg.norm(face[:, :, 1:], axis=2)
    shoulder_r = np.linalg.norm(shoulder[:, :, 1:], axis=2)
    annulus = [float(max(face_r.min(), shoulder_r.min())),
               float(min(face_r.max(), shoulder_r.max()))]
    if annulus[0] >= annulus[1]:
        raise ValueError('Bearing face and shoulder have no radial overlap')
    journal = shaft[(shaft[:, :, 0].mean(1) > -.049)
                    & (shaft[:, :, 0].mean(1) < -.034) & (abs(sn[:, 0]) < .01)]
    q = journal.reshape(-1, 3)
    q = q[np.linalg.norm(q[:, 1:], axis=1) > .01]
    fit = least_squares(lambda x: np.linalg.norm(q[:, 1:] - x[:2], axis=1) - x[2],
                        [0, 0, .0125])
    # The sampled cache only estimates bore-axis centre; its uncertainty is logged.
    with np.load(ROOT / 'assets/public_cad_models/bearing_32205_transfer.npz') as f:
        if not np.isclose(float(f['geometry_scale_m_per_coordinate']), SCALE, rtol=0, atol=1e-12):
            raise ValueError('Bearing cache scale mismatch')
        p, n = f['points'].astype(float), f['normals'].astype(float)
    radius = np.linalg.norm(p[:, 1:], axis=1)
    mask = (p[:, 0] > .002) & (p[:, 0] < .018) & (radius < .013)
    mask &= (abs(n[:, 0]) < .01) & ((p[:, 1:] * n[:, 1:]).sum(1) < 0)
    q = p[mask]
    bore = least_squares(lambda x: np.linalg.norm(q[:, 1:] - x[:2], axis=1) - x[2],
                         [0, 0, .0125])
    return dict(scale_m_per_source_unit=SCALE, bearing_wide_face_x_m=wide_x,
                shaft_shoulder_x_m=shoulder_x, shoulder_annulus_radius_m=annulus,
                shaft_axis_yz_m=fit.x[:2].tolist(), bore_axis_yz_m=bore.x[:2].tolist(),
                candidate_relative_position_m=[shoulder_x - wide_x,
                                               *(fit.x[:2] - bore.x[:2])],
                shaft_journal_diameter_m=float(2 * fit.x[2]),
                bore_circle_fit_rms_m=float(np.sqrt(np.mean(bore.fun ** 2))),
                source_offset_axial_residual_m=float(shoulder_x - wide_x + .050563),
                candidate_relative_orientation_wxyz=[1, 0, 0, 0],
                contact_measurement='Filtered PhysX per-contact force, normal, point and signed separation; missing data is UNKNOWN, not zero penetration.',
                clearance_warning='Micrometre-scale cached radial clearance is inconclusive; real contacts and signed separation remain required.')


def measure(bearing_pose, fixture_pose, velocity, contacts, released, geometry, limits):
    """Pure geometric/physical checks, independent of the pending native evaluator."""
    try:
        b, f, v = [np.asarray(x, float) for x in (bearing_pose, fixture_pose, velocity)]
        if b.shape != (7,) or f.shape != (7,) or v.shape != (6,):
            raise ValueError('Invalid pose/velocity shape')
        if not np.isfinite(np.r_[b, f, v]).all() or contacts is None:
            raise ValueError('Missing/nonfinite pose, velocity or contact buffers')
        if not isinstance(released, (bool, np.bool_)):
            raise ValueError('Measured constraint release state required')
        if abs(np.linalg.norm(b[3:]) - 1) > .001 or abs(np.linalg.norm(f[3:]) - 1) > .001:
            raise ValueError('Unit quaternions required')
        br = Rotation.from_quat(b[[4, 5, 6, 3]])
        fr = Rotation.from_quat(f[[4, 5, 6, 3]])
        relative = fr.inv() * br
        axis = relative.apply([1, 0, 0])
        centre = fr.inv().apply(b[:3] + br.apply([0, *geometry['bore_axis_yz_m']]) - f[:3])
        radial = float(np.linalg.norm(centre[1:] - geometry['shaft_axis_yz_m']))
        tilt = float(np.degrees(np.arccos(np.clip(axis[0], -1, 1))))
        face = fr.inv().apply(b[:3] + br.apply([geometry['bearing_wide_face_x_m'],
                                               *geometry['bore_axis_yz_m']]) - f[:3])
        gap = float(geometry['shaft_shoulder_x_m'] - face[0])
        penetration, support, raw = 0., 0., []
        outward = fr.apply([-1, 0, 0])
        for contact in contacts:
            force, separation = float(contact['force_n']), float(contact['separation_m'])
            point, normal = np.asarray(contact['point_w_m'], float), np.asarray(contact['normal_w'], float)
            if point.shape != (3,) or normal.shape != (3,) or not np.isfinite(np.r_[force, separation, point, normal]).all() or np.linalg.norm(normal) < 1e-9:
                raise ValueError('Invalid raw contact')
            normal = normal / np.linalg.norm(normal)
            local = fr.inv().apply(point - f[:3])
            r = np.linalg.norm(local[1:] - geometry['shaft_axis_yz_m'])
            lo, hi = geometry['shoulder_annulus_radius_m']
            shoulder = abs(local[0] - geometry['shaft_shoulder_x_m']) <= limits.contact_region_m
            shoulder = shoulder and lo - limits.contact_region_m <= r <= hi + limits.contact_region_m
            alignment = abs(float(normal @ outward))
            if shoulder and alignment >= .9 and separation <= limits.axial_gap_m:
                support += abs(force) * alignment
            penetration = max(penetration, -separation)
            raw.append(dict(contact, in_shoulder_region=bool(shoulder)))
        checks = dict(coaxiality=radial <= limits.radial_m, orientation=tilt <= limits.tilt_deg,
                      axial_shoulder=abs(gap) <= limits.axial_gap_m,
                      shoulder_contact=support >= limits.support_force_n,
                      nonpenetrating=penetration <= limits.penetration_m,
                      released=bool(released), linear_speed=np.linalg.norm(v[:3]) <= limits.linear_speed_mps,
                      angular_speed=np.linalg.norm(v[3:]) <= limits.angular_speed_radps)
        return dict(status='measured', checks={k: bool(x) for k, x in checks.items()},
                    instantaneous_seat=bool(all(checks.values())), radial_error_m=radial,
                    tilt_deg=tilt, axial_gap_m=gap, maximum_penetration_m=penetration,
                    shoulder_support_force_n=support, contacts=raw)
    except (ValueError, TypeError, KeyError) as error:
        return dict(status='UNKNOWN', instantaneous_seat=False, error=str(error))


def stable_seat(history, limits):
    required = int(np.ceil(limits.dwell_s / DT))
    return len(history) >= required and all(x.get('instantaneous_seat', False) for x in history[-required:])


def trial_plan():
    return [dict(name='drop_1mm', kind='positive', gap_m=.001, offset_m=0., tilt_deg=0.),
            dict(name='drop_3mm', kind='positive', gap_m=.003, offset_m=0., tilt_deg=0.),
            dict(name='lateral', kind='negative', gap_m=0., offset_m=.003, tilt_deg=0.),
            dict(name='height', kind='negative', gap_m=.008, offset_m=0., tilt_deg=0.),
            dict(name='tilt', kind='negative', gap_m=0., offset_m=0., tilt_deg=8.),
            dict(name='unreleased', kind='negative', gap_m=0., offset_m=0., tilt_deg=0.)]


def summarize(trials, geometry, limits):
    results = []
    failures = dict(lateral='coaxiality', height='axial_shoulder', tilt='orientation', unreleased='released')
    for case in trial_plan():
        history = trials.get(case['name'], [])
        if case['kind'] == 'positive':
            # Must actually have fallen after release, not initialized at the seat.
            fallen = bool(history and history[0].get('axial_gap_m', 0) - history[-1].get('axial_gap_m', 0) > case['gap_m'] / 2)
            ok = fallen and stable_seat(history, limits)
        else:
            # Perturbations can self-correct later. Only the measured perturbed
            # snapshot is a negative; never mislabel a repaired final seat.
            first = history[0] if history else {}
            ok = first.get('status') == 'measured' and first.get('checks', {}).get(failures[case['name']]) is False
            ok = ok and not first.get('instantaneous_seat', True)
            if case['name'] == 'unreleased':
                # A held object far from the seat is not a meaningful release
                # negative. The holder may carry load, so contact force alone
                # need not be positive in this constrained trial.
                checks = first.get('checks', {})
                ok = ok and all(checks.get(k) is True for k in (
                    'coaxiality', 'orientation', 'axial_shoulder', 'nonpenetrating',
                    'linear_speed', 'angular_speed'))
            else:
                ok = ok and first.get('checks', {}).get('released') is True
        results.append(dict(case, probe_case_passed=bool(ok), samples=len(history)))
    return dict(privileged_calibration_only=True, not_a_model_rollout=True,
                native_success=False, calibration_status='pending', physical_gt_verified=False,
                geometry=geometry, proposed_limits=asdict(limits), trials=results,
                physical_probe_passed=all(x['probe_case_passed'] for x in results),
                approval_note='Never auto-publish calibration. Review real raw-contact traces and positive/negative results. Runtime pickup/release and all existing calibration protections remain required; this no-robot seat probe does not validate pickup.')


def simulate(args, geometry, limits):
    """Lazy Isaac imports: offline tests never start Kit or touch a GPU."""
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device='cpu').app
    try:
        # Kit supplies pxr in the Isaac image; bootstrap it before reading USD.
        if geometry is None:
            geometry = inspect_geometry(args.asset_root)
        import carb
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Sdf
        sys.path.insert(0, str(ROOT / 'isaac_runtime'))
        from runtime.bolt_harness.measurement import read_filtered_contact_reports

        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device='cpu', render_interval=120))
        carb.settings.get_settings().set_int('/physics/cudaDevice', 0)
        sim.get_physics_context().set_broadphase_type('GPU')
        sim.get_physics_context().enable_gpu_dynamics(True)
        # Match generic_backend's visual-fallback bearing collider and SDF setup.
        wrapper_path = args.output_dir / 'private/bearing_wrapper.usda'
        wrapper_path.parent.mkdir(parents=True, exist_ok=True)
        wrapper = Usd.Stage.CreateNew(str(wrapper_path))
        prim = UsdGeom.Xform.Define(wrapper, '/Asset').GetPrim()
        prim.GetReferences().AddReference(str(args.asset_root / 'CAD_edit/Bearing 32205 Inner Assy.usd'), '/World')
        UsdPhysics.RigidBodyAPI.Apply(prim)
        wrapper.SetDefaultPrim(prim)
        for prim in wrapper.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('sdf')
                PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim).CreateSdfResolutionAttr(256)
        wrapper.GetRootLayer().Save()
        with np.load(ROOT / 'assets/public_cad_models/bearing_32205_transfer.npz') as f:
            mass = float(f['volume_m3']) * 7800.
        rotation = Rotation.from_euler('y', 90, degrees=True)
        quat = rotation.as_quat()[[3, 0, 1, 2]].tolist()
        objects, sensors, joints, fixtures = {}, {}, {}, {}
        for i, case in enumerate(trial_plan()):
            base = '/World/Probe' + str(i)
            UsdGeom.Xform.Define(sim.stage, base)
            fp = np.array([i * .25, 0., .2])
            relative = np.array(geometry['candidate_relative_position_m'])
            relative[0] -= case['gap_m']
            relative[1] += case['offset_m']
            bp = fp + rotation.apply(relative)
            bq = (rotation * Rotation.from_euler('z', case['tilt_deg'], degrees=True)).as_quat()[[3, 0, 1, 2]].tolist()
            static = sim_utils.UsdFileCfg(usd_path=str(args.asset_root / 'parts/Transfer Shaft.usd'), scale=(SCALE,) * 3,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.00005, rest_offset=0))
            fixture = RigidObject(RigidObjectCfg(prim_path=base + '/Shaft', spawn=static,
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(fp), rot=tuple(quat))))
            for prim in Usd.PrimRange(sim.stage.GetPrimAtPath(base + '/Shaft')):
                if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')
            dynamic = sim_utils.UsdFileCfg(usd_path=str(wrapper_path), scale=(SCALE,) * 3, activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=32, solver_velocity_iteration_count=8),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.00005, rest_offset=0))
            body = RigidObject(RigidObjectCfg(prim_path=base + '/Bearing', spawn=dynamic,
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(bp), rot=tuple(bq))))
            holder = sim_utils.CuboidCfg(size=(.001,) * 3,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True))
            holder.func(base + '/Holder', holder, translation=tuple(bp), orientation=tuple(bq))
            joint = UsdPhysics.FixedJoint.Define(sim.stage, base + '/Support')
            joint.CreateBody0Rel().SetTargets([Sdf.Path(base + '/Holder')])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(base + '/Bearing')])
            joint.CreateJointEnabledAttr(True)
            sensor = ContactSensor(ContactSensorCfg(prim_path=base + '/Bearing',
                filter_prim_paths_expr=[base + '/Shaft'], track_contact_points=True,
                max_contact_data_count_per_prim=1024, update_period=0))
            objects[case['name']] = (body, fixture)
            sensors[case['name']], joints[case['name']] = sensor, joint
            fixtures[case['name']] = np.r_[fp, quat]
        sim.reset()
        histories = {c['name']: [] for c in trial_plan()}
        trace = args.output_dir / 'private/evaluator-probe.jsonl'
        with trace.open('w') as stream:
            for step in range(int(args.duration_s / DT)):
                if step == 1:
                    for name, joint in joints.items():
                        if name != 'unreleased':
                            joint.GetJointEnabledAttr().Set(False)
                sim.step(render=False)
                for case in trial_plan():
                    name = case['name']; body, fixture = objects[name]
                    body.update(DT); fixture.update(DT); sensors[name].update(DT)
                    if step < 1:
                        continue
                    state = body.data.root_state_w[0].detach().cpu().numpy()
                    try:
                        raw = read_filtered_contact_reports(sensors[name]._contact_physx_view,
                            dt_s=DT, sensor_index=0, filter_index=0)
                        contacts = [asdict(x) for x in raw]
                        if len(raw) >= 1024:
                            raise ValueError('Contact buffer may be saturated')
                        released = not bool(joints[name].GetJointEnabledAttr().Get())
                        record = measure(state[:7], fixtures[name], state[7:13], contacts, released, geometry, limits)
                    except Exception as error:
                        record = dict(status='UNKNOWN', instantaneous_seat=False, error=str(error))
                    record.update(step=step, time_s=step * DT, root_state_w=state.tolist(),
                                  support_constraint_enabled=bool(joints[name].GetJointEnabledAttr().Get()))
                    histories[name].append(record)
                    stream.write(json.dumps(dict(trial=name, **record), allow_nan=False) + '\n')
        result = summarize(histories, geometry, limits)
        result.update(simulation_executed=True, mass_kg=mass, density_assumption_kg_m3=7800, raw_trace=str(trace),
                      object_pose_writes_after_initialization=0, cameras=0, robots=0,
                      release_measured_from='USD fixed support joint enabled state; no robot pickup certification')
        if result['physical_probe_passed']:
            # Geometric/contact validation precedes measured reference extraction.
            # This remains an unapproved candidate, never a deployable calibration.
            count = int(np.ceil(limits.dwell_s / DT))
            measured_positions, measured_rotations = [], []
            for name in ['drop_1mm', 'drop_3mm']:
                fp = fixtures[name]
                fr = Rotation.from_quat(fp[[4, 5, 6, 3]])
                for sample in histories[name][-count:]:
                    state = np.array(sample['root_state_w'])
                    measured_positions.append(fr.inv().apply(state[:3] - fp[:3]))
                    measured_rotations.append(fr.inv() * Rotation.from_quat(state[[4, 5, 6, 3]]))
            reference = np.median(measured_positions, axis=0)
            orientation = Rotation.concatenate(measured_rotations).mean().as_quat()[[3, 0, 1, 2]]
            result['unapproved_measured_reference'] = dict(
                calibration_status='pending', physical_gt_verified=False,
                reference_relative_pose=dict(position_m=reference.tolist(), orientation_wxyz=orientation.tolist()),
                position_range_m=np.ptp(measured_positions, axis=0).tolist(),
                provenance='Two physically dropped, released, CAD-coaxial, shoulder-contacting, nonpenetrating stable windows; not the source magic matrix.',
                missing_validation='Robot pickup/release and deployed evaluator calibration approval remain outstanding.')
        # Preserve measured results even if Kit shutdown stalls after the trials.
        (args.output_dir / 'evaluator-probe-result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        return result
    finally:
        app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--simulate', action='store_true')
    parser.add_argument('--duration-s', type=float, default=4.)
    args = parser.parse_args()
    if not np.isfinite(args.duration_s) or not 1 <= args.duration_s <= 30:
        parser.error('duration-s must be finite and within 1..30')
    args.asset_root = args.asset_root.resolve(); args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    geometry = None
    limits = Limits()
    try:
        if not args.simulate:
            geometry = inspect_geometry(args.asset_root)
        result = simulate(args, geometry, limits) if args.simulate else dict(
            privileged_calibration_only=True, native_success=False, physical_gt_verified=False,
            calibration_status='pending', simulation_executed=False, geometry=geometry,
            planned_trials=trial_plan(), proposed_limits=asdict(limits))
    except Exception as error:
        result = dict(privileged_calibration_only=True, native_success=False,
                      physical_gt_verified=False, calibration_status='pending',
                      physical_probe_passed=False, status='UNKNOWN', geometry=geometry,
                      error=f'{type(error).__name__}: {error}')
    (args.output_dir / 'evaluator-probe-result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if not args.simulate or result['physical_probe_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
