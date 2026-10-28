"""Launch an owned Isaac container and the host-side RPent planner."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import settings
from task_profiles import SUPPORTED_TASKS,SHAFT_TASKS,HUB_TASKS,skill_specification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['nominal-aligned', 'bolt-position-shift', 'bolt-horizontal-pose', 'assembly-position-shift'], default='nominal-aligned')
    parser.add_argument('--run-dir')
    parser.add_argument('--robot',choices=['panda','r1','xarm7'],default='panda')
    parser.add_argument('--task',choices=sorted(SUPPORTED_TASKS),default='peg')
    parser.add_argument('--method',choices=['ours','vlm-dewm'],default='ours')
    parser.add_argument('--condition',choices=['clean','environment','input_error'],default='clean')
    parser.add_argument('--action-backend',choices=['auto','generic'],default='generic',help='Force the shared Cartesian/IK backend, including for Panda peg')
    parser.add_argument('--controller',choices=['reference_ik','reference_ik_high_pd','factory','factory_damped','isaac_osc'],default='reference_ik_high_pd',help='Robot-only torque adapter or earlier IK-reference diagnostic adapter')
    parser.add_argument('--cover-material',choices=['legacy-steel','aluminum-6061-assumption','polypropylene-assumption'],default='polypropylene-assumption',help='Declared material model; aluminum mass is computed from closed public STL volume, not chosen to fit the robot')
    parser.add_argument('--gear-density-kg-m3','--steel-density-kg-m3',dest='steel_density_kg_m3',type=float,default=7800.,help='Declared steel-density assumption used with authored transformed collision meshes; not measured material')
    parser.add_argument('--collision-backend',choices=['gpu-sdf','cpu'],default='gpu-sdf',help='SDF collision requires GPU dynamics; control tensors may remain on CPU')
    parser.add_argument('--sdf-resolution',type=int,choices=[256,512,1024],default=256,help='Dynamic mesh collision precision; preserves CAD geometry')
    parser.add_argument('--gpu', default='0', help='Host NVIDIA GPU index')
    parser.add_argument('--xarm-physics-hz',type=int,choices=[120,240,480],default=480,help='xArm integration comparison; safety monitoring remains every physics tick')
    parser.add_argument('--xarm-diagnostic-reference-speed-scale',type=float,default=1.,help='Adapter-only reference-speed comparison; tracking and emergency limits unchanged')
    parser.add_argument('--physics-threads',type=int,choices=[1,2,4,8,16,32,64],help='Optional PhysX worker count; integration rate and solver iterations unchanged')
    parser.add_argument('--cpu-threads',type=int,choices=[1,2,4,8,16,32,64],help='Optional Carbonite/TBB worker limit for concurrent simulations')
    parser.add_argument('--verifier-calibration',choices=['hub-positive','hub-positive-from-clearance','hub-positive-near-seat','hub-positive-cad-datum','hub-yaw-positive','hub-yaw-origin-negative','hub-offset-negative','hub-tilt-negative','bearing-middle-seat-probe','bearing-middle-offset-probe','bearing-middle-flipped-probe','peg-seat-probe','peg-offset-probe','m10-seat-probe'],help='Privileged setup diagnostic, excluded from model results')
    parser.add_argument('--recovery-validation',action='store_true',help='Invalid-reference handling and measured small-move recovery diagnostic')
    parser.add_argument('--adapter-validation',action='store_true',help='Measured robot-only motion/gripper validation; no model assembly score')
    parser.add_argument('--adapter-start-joints',type=float,nargs=7,help='Diagnostic-only initial xArm arm angles; never moves or initializes parts')
    parser.add_argument('--osc-diagnostic-rotation-stiffness',type=float,default=100.,help='Adapter-only posture tracking experiment; normal OSC remains 100')
    parser.add_argument('--scene-smoke',action='store_true',help='Start the selected Isaac scene, capture its initial public observation, and exit without an LLM call or robot action')
    parser.add_argument('--profile-backend',action='store_true',help='Write a Python execution profile for backend performance diagnosis')
    parser.add_argument('--rollout', action='store_true', help='Run assembly; otherwise run a bounded wiring smoke test')
    parser.add_argument('--peg-mvp',action='store_true',help='Record independent private peg process metrics; does not certify calibration')
    parser.add_argument('--peg-mvp-module',type=Path,default=settings.ROOT.parent/'benchmark/assembly_bench/peg_mvp.py')
    parser.add_argument('--peg-offset-m',type=float,nargs=2,default=[0.,0.],metavar=('X','Y'))
    parser.add_argument('--peg-contact-diagnostic',action='store_true',help='Privileged comparison with numerical body stabilization disabled')
    parser.add_argument('--peg-load-probe',action='store_true',help='Privileged 0.2N downward pulse, then unloaded settling; never a policy rollout')
    parser.add_argument('--peg-probe-seed',type=int)
    parser.add_argument('--peg-probe-inverted',action='store_true')
    parser.add_argument('--peg-sim-budget-s',type=float,default=600.)
    parser.add_argument('--budget-s', type=float)
    parser.add_argument('--xarm-diagnostic-arm-stiffness',type=float,default=400.)
    parser.add_argument('--xarm-diagnostic-arm-damping',type=float,default=40.)
    args = parser.parse_args()
    if args.robot=='xarm7' and args.xarm_physics_hz==120 and not (args.adapter_validation or args.scene_smoke):
        parser.error('120Hz xArm physics is diagnostic-only until physical stability is validated')
    if args.verifier_calibration in {'hub-yaw-positive','hub-yaw-origin-negative'} and args.task!='hub-cover':
        parser.error('Yaw datum diagnostics require hub-cover')
    if args.adapter_start_joints is not None:
        if not args.adapter_validation or args.robot!='xarm7' or args.rollout or args.verifier_calibration:
            parser.error('Initial arm override is only for xArm adapter diagnostics, never a policy rollout')
        from xarm_robot_cfg import diagnostic_initial_joints
        diagnostic_initial_joints(args.adapter_start_joints,settings.ROOT/'assets/xarm7/xarm7_g1_simulation_assumption.urdf')
    if args.peg_mvp and (args.task!='peg' or args.robot not in {'panda','xarm7'}):
        parser.error('Peg process recording supports Panda or xArm7 peg tasks only')
    import math
    if not math.isfinite(args.xarm_diagnostic_reference_speed_scale) or not 1<=args.xarm_diagnostic_reference_speed_scale<=2:parser.error('Reference speed scale must be within 1..2')
    if args.xarm_diagnostic_reference_speed_scale!=1 and not (args.robot=='xarm7' and args.adapter_validation and not args.rollout):parser.error('Reference speed changes are adapter diagnostics only')
    if not math.isfinite(args.xarm_diagnostic_arm_stiffness) or not 400<=args.xarm_diagnostic_arm_stiffness<=1000 or not math.isfinite(args.xarm_diagnostic_arm_damping) or not 40<=args.xarm_diagnostic_arm_damping<=100:
        parser.error('xArm diagnostic stiffness must be 400..1000 and damping 40..100')
    if (args.xarm_diagnostic_arm_stiffness!=400 or args.xarm_diagnostic_arm_damping!=40) and not (args.robot=='xarm7' and args.adapter_validation and not args.rollout):
        parser.error('Changed xArm gains are only available in adapter diagnostics')
    if not math.isfinite(args.osc_diagnostic_rotation_stiffness) or not 100<=args.osc_diagnostic_rotation_stiffness<=1000:
        parser.error('OSC diagnostic rotation stiffness must be within 100..1000')
    if args.osc_diagnostic_rotation_stiffness!=100 and not (args.adapter_validation and args.controller=='isaac_osc'):
        parser.error('Changed OSC stiffness is only available in adapter diagnostics')
    if not math.isfinite(args.peg_sim_budget_s) or args.peg_sim_budget_s<=0:
        parser.error('Peg simulation budget must be finite and positive')
    if not all(math.isfinite(v) and abs(v)<=.01 for v in args.peg_offset_m):
        parser.error('Peg instance XY offsets must be finite and within 10mm')
    if any(args.peg_offset_m) and (not args.peg_mvp or args.verifier_calibration):
        parser.error('XY instance offsets require a non-privileged peg MVP episode')
    if args.peg_contact_diagnostic and not (args.peg_mvp and args.verifier_calibration):
        parser.error('Contact comparison is only a privileged peg MVP calibration diagnostic')
    if args.peg_load_probe and not (args.peg_mvp and args.verifier_calibration=='peg-seat-probe' and not args.peg_contact_diagnostic):
        parser.error('Load probe requires the standard privileged seat calibration only')
    if (args.peg_probe_seed is not None or args.peg_probe_inverted) and not (args.peg_mvp and args.verifier_calibration in {'peg-seat-probe','peg-offset-probe'}):
        parser.error('Probe pose variants are only initial conditions for privileged peg calibration')
    if args.robot=='xarm7' and args.controller in {'factory','factory_damped','isaac_osc'}:
        parser.error('xArm7 uses reference_ik_high_pd; Panda torque adapters are not compatible')
    if args.verifier_calibration in {'bearing-middle-seat-probe','bearing-middle-offset-probe','bearing-middle-flipped-probe'} and args.task!='bearing-input-top-onto-shaft':
        parser.error('bearing-middle-seat-probe requires bearing-input-top-onto-shaft')
    if args.verifier_calibration in {'peg-seat-probe','peg-offset-probe'} and args.task!='peg':
        parser.error('peg probes require the peg task')
    if args.verifier_calibration=='m10-seat-probe' and args.task!='m10-casing-bolt':
        parser.error('m10-seat-probe requires m10-casing-bolt')
    if args.verifier_calibration and args.verifier_calibration not in {'bearing-middle-seat-probe','bearing-middle-offset-probe','bearing-middle-flipped-probe','peg-seat-probe','peg-offset-probe','m10-seat-probe'} and args.task not in HUB_TASKS:
        parser.error('--verifier-calibration requires a hub-cover task')
    if args.robot!='panda' and args.controller.startswith('factory'):parser.error('Factory torque adapter is Panda-only; use reference_ik for the legacy R1 diagnostic')
    budget = args.budget_s if args.budget_s is not None else (600 if args.rollout else 120)
    if budget <= 0:
        parser.error('--budget-s must be positive')
    if args.steel_density_kg_m3<=0:
        parser.error('--steel-density-kg-m3 must be positive')
    if args.scene_smoke and (args.rollout or args.verifier_calibration or args.recovery_validation or args.adapter_validation):
        parser.error('--scene-smoke cannot be combined with rollout or validation modes')
    required_env = ['ASSEMBLY_ASSET_ROOT']
    if not args.scene_smoke:
        required_env += ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL']
    for key in required_env:
        if not os.environ.get(key):
            parser.error(f'Set {key}; see .env.example')
    os.environ.update(ASSEMBLY_ROBOT=args.robot, ASSEMBLY_TASK=args.task, ASSEMBLY_ACTION_BACKEND=args.action_backend,ASSEMBLY_CONTROLLER=args.controller)
    assets = Path(os.environ['ASSEMBLY_ASSET_ROOT']).expanduser().resolve()
    skill_profiles=json.loads((settings.ROOT/'config/assembly_skills.json').read_text())
    skill=skill_profiles[args.task]
    moving_asset=skill.get('moving_asset_path','parts/'+skill['moving_cad_file'])
    moving_cache=skill['moving_cad_cache']
    moving_assets={'hub-cover':('parts/Hub Cover Output.usd','Hub Cover Output.stl','cover.npz'),'hub-cover-input':(moving_asset,'Hub Cover Input.stl',moving_cache),'hub-cover-small':(moving_asset,'Hub Cover Small.stl',moving_cache)}
    selected_moving=moving_assets.get(args.task,moving_assets['hub-cover'] if args.task=='peg' else (moving_asset,None,moving_cache))
    evaluator_profiles=json.loads((settings.ROOT/'config/task_evaluator_profiles.json').read_text())
    fixture=evaluator_profiles.get(args.task,{}).get('fixture_asset','parts/Casing Top.usd')
    required = [fixture, selected_moving[0], 'vendor/remote_mirror/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd']
    if args.task=='peg':required.append('parts/M6 Hub Bolt.usd')
    if args.robot=='xarm7' and not (settings.ROOT/'assets/xarm7/usd/xarm7_g1.usd').is_file():
        parser.error('Prepare the official xArm7 G1 USD before launch')
    missing = [str(assets / name) for name in required if not (assets / name).is_file()]
    if missing:
        parser.error('Missing external assets: ' + ', '.join(missing))
    from tools.prepare_cad_cache import prepare,prepare_shaft_compound
    if args.task not in SHAFT_TASKS or args.robot=='panda':prepare()
    tag = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run = settings.configure(args.run_dir or settings.ROOT / 'runs' / tag)
    if (run / 'ipc/reply_0000.json').exists() or list((run / 'ipc').glob('command_*.json')):
        raise ValueError('Choose a new run directory; episode IPC must not be reused')
    metal_tasks={'output-gear-onto-shaft','transfer-gear-onto-shaft','output-shaft-into-base','transfer-shaft-into-base','input-shaft-into-base','m10-casing-bolt','m10-casing-nut','casing-top-to-base','oil-level-indicator','breather-plug','bearing-32013-onto-output-shaft','bearing-transfer-onto-shaft','bearing-input-bottom-onto-shaft','bearing-input-top-onto-shaft'}
    if args.task in SHAFT_TASKS:
        material='steel-density-assumption' if abs(args.steel_density_kg_m3-7800.)<1e-6 else 'custom-density-assumption'
        allow_open=bool(evaluator_profiles.get(args.task,{}).get('allow_provisional_open_mesh_mass',False))
        compound=prepare_shaft_compound(args.task,assets,run,args.steel_density_kg_m3,material,
                                        allow_open_mesh=allow_open)
        cover_material=compound['mass_profile']
        skill=skill_specification()
        moving_asset=skill['moving_asset_path'];moving_cache=skill['moving_cad_cache']
    elif args.task in metal_tasks:
        from tools.calibrate_part_mass import cad_mass_profile
        material='steel-density-assumption' if abs(args.steel_density_kg_m3-7800.)<1e-6 else 'custom-density-assumption'
        allow_open=bool(evaluator_profiles.get(args.task,{}).get('allow_provisional_open_mesh_mass',False))
        cover_material=cad_mass_profile(settings.ROOT/'assets/public_cad_models'/selected_moving[2],args.steel_density_kg_m3,material,allow_open_mesh=allow_open)
    else:
        from tools.calibrate_part_mass import cover_mass_profile
        cover_material=cover_mass_profile(assets,args.cover_material,selected_moving[1],settings.ROOT/'assets/public_cad_models'/selected_moving[2])
    (run/'checks/moving-part-mass-profile.json').write_text(json.dumps(cover_material,indent=2))
    name = 'assembly-agent-' + tag.lower()
    cmd = ['docker', 'run', '-d', '--name', name, '--gpus', 'device=' + args.gpu, '--ipc', 'host', '--network', 'host']
    for entry in ['ASSEMBLY_COLLISION_BACKEND='+args.collision_backend,'ASSEMBLY_SDF_RESOLUTION='+str(args.sdf_resolution),'ASSEMBLY_COVER_MASS_KG='+str(cover_material['mass_kg']),'ASSEMBLY_COVER_MATERIAL='+cover_material['material'],'ASSEMBLY_COVER_MASS_STATUS='+cover_material.get('mass_estimate_status',cover_material.get('material_status','unspecified')), 'ASSEMBLY_CONTROLLER='+args.controller,'ASSEMBLY_CASE=' + args.case, 'ASSEMBLY_ROBOT='+args.robot, 'ASSEMBLY_TASK='+args.task, 'ASSEMBLY_ACTION_BACKEND='+args.action_backend, 'ASSEMBLY_ASSET_ROOT=' + str(assets), 'ASSEMBLY_FAST_COMPONENTS=1', 'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1', 'ACCEPT_EULA=Y', 'OMNI_KIT_ACCEPT_EULA=YES', 'ASSEMBLY_RUN_DIR=' + str(run), 'ASSEMBLY_CAMERA_WIDTH=' + os.environ.get('ASSEMBLY_CAMERA_WIDTH', '960'), 'ASSEMBLY_CAMERA_HEIGHT=' + os.environ.get('ASSEMBLY_CAMERA_HEIGHT', '720')]:
        cmd += ['-e', entry]
    cmd+=['-e','ASSEMBLY_INPUT_CONDITION='+args.condition]
    cmd+=['-e','ASSEMBLY_XARM_PHYSICS_HZ='+str(args.xarm_physics_hz)]
    cmd+=['-e','ASSEMBLY_XARM_DIAGNOSTIC_ARM_STIFFNESS='+str(args.xarm_diagnostic_arm_stiffness),'-e','ASSEMBLY_XARM_DIAGNOSTIC_ARM_DAMPING='+str(args.xarm_diagnostic_arm_damping)]
    if args.adapter_start_joints is not None:cmd+=['-e','ASSEMBLY_ADAPTER_START_JOINTS='+json.dumps(args.adapter_start_joints)]
    cmd+=['-e','ASSEMBLY_XARM_REFERENCE_SPEED_SCALE='+str(args.xarm_diagnostic_reference_speed_scale)]
    if args.physics_threads is not None:cmd+=['-e','ASSEMBLY_PHYSICS_THREADS='+str(args.physics_threads)]
    if args.cpu_threads is not None:cmd+=['-e','ASSEMBLY_CPU_THREADS='+str(args.cpu_threads)]
    if args.peg_mvp:
        module=args.peg_mvp_module.expanduser().resolve()
        if not module.is_file():parser.error('Missing shared assembly_bench peg MVP evaluator')
        source_copy=run/'private/peg-mvp-source.py'
        with source_copy.open('x') as stream:stream.write(module.read_text())
        cmd+=['-e','ASSEMBLY_PEG_MVP=1','-e','ASSEMBLY_PEG_SIM_BUDGET_S='+str(args.peg_sim_budget_s),'-e','ASSEMBLY_PEG_OFFSET_M='+json.dumps(args.peg_offset_m),'-v',f'{source_copy}:/opt/cape_peg/peg_mvp.py:ro']
        (run/'checks/peg-mvp-instance.json').write_text(json.dumps({'variant':args.case,
            'bolt_xy_offset_m':args.peg_offset_m,'privileged':bool(args.verifier_calibration or args.adapter_validation or args.recovery_validation or args.scene_smoke),
            'simulation_budget_s':args.peg_sim_budget_s,
            'scoring_status':'development_calibration_pending'},indent=2))
        if args.adapter_validation or args.recovery_validation or args.scene_smoke:
            cmd+=['-e','ASSEMBLY_PEG_VALIDATION_ONLY=1']
    if args.peg_contact_diagnostic:cmd+=['-e','ASSEMBLY_PEG_CONTACT_DIAGNOSTIC=1']
    cmd+=['-e','ASSEMBLY_OSC_ROTATION_STIFFNESS='+str(args.osc_diagnostic_rotation_stiffness)]
    if args.peg_load_probe:cmd+=['-e','ASSEMBLY_PEG_LOAD_PROBE=1']
    if args.peg_probe_seed is not None:cmd+=['-e','ASSEMBLY_PEG_PROBE_SEED='+str(args.peg_probe_seed)]
    if args.peg_probe_inverted:cmd+=['-e','ASSEMBLY_PEG_PROBE_INVERTED=1']
    if args.verifier_calibration:cmd += ['-e','ASSEMBLY_DIAGNOSTIC='+args.verifier_calibration]
    for mount in [f'{settings.ROOT}:{settings.ROOT}:ro', f'{run}:{run}', f'{assets}:{assets}:ro']:
        cmd += ['-v', mount]
    if os.environ.get('ASSEMBLY_PERSISTENT_ISAAC_CACHE','1')=='1':
        # Named volumes retain image-provided cache contents and survive owned container removal.
        cache_image=os.environ.get('ISAAC_DOCKER_IMAGE','nvcr.io/nvidia/isaac-lab:2.3.0')
        cache_prefix='assembly-agent-'+cache_image.replace('/','-').replace(':','-')
        for label,destination in [('kit','/isaac-sim/kit/cache'),('ov','/root/.cache/ov'),('gl','/root/.cache/nvidia/GLCache'),('compute','/root/.nv/ComputeCache')]:
            cmd+=['-v',f'{cache_prefix}-{label}:{destination}']
    # A read-only source mount would otherwise expose the host planner's key file.
    if (settings.ROOT / '.env').is_file():
        cmd += ['-v', f'/dev/null:{settings.ROOT / ".env"}:ro']
    if args.robot=='r1':
        r1=Path(os.environ['ASSEMBLY_R1_ROOT']).expanduser().resolve()
        cmd+=['-e','ASSEMBLY_R1_ROOT='+str(r1),'-v',f'{r1}:{r1}:ro']
    from task_profiles import generic
    backend = settings.ROOT / ('generic_backend.py' if generic() else 'backend.py')
    if args.peg_mvp:
        backend_copy=run/'private/peg-mvp-backend-source.py'
        with backend_copy.open('x') as stream:stream.write(backend.read_text())
        backend=backend_copy
        cmd+=['-e','PYTHONPATH='+str(settings.ROOT)]
    if args.profile_backend:cmd+=['-e','ASSEMBLY_PROFILE_BACKEND=1']
    if args.adapter_validation and os.environ.get('ASSEMBLY_CONTACT_AUDIT')=='1':cmd+=['-e','ASSEMBLY_CONTACT_AUDIT=1']
    command = 'umask 000; exec /isaac-sim/python.sh ' + shlex.quote(str(backend)) + ' --headless --enable_cameras --device cpu'
    if args.cpu_threads is not None:
        command+=' --kit_args '+shlex.quote(f'--/plugins/carb.tasking.plugin/threadCount={args.cpu_threads} --/plugins/omni.tbb.globalcontrol/maxThreadCount={args.cpu_threads}')
    cmd += ['--entrypoint', '/bin/bash', os.environ.get('ISAAC_DOCKER_IMAGE', 'nvcr.io/nvidia/isaac-lab:2.3.0'), '-c', command]
    planner = [str(settings.PYTHON), str(settings.ROOT / 'planner.py'), '--run-dir', str(run), '--budget-s', str(budget),'--method',args.method]
    if args.verifier_calibration:
        planner=[str(settings.PYTHON),str(settings.ROOT/'validate_evaluator.py'),'--run-dir',str(run),'--budget-s',str(budget)]
    elif args.recovery_validation:
        planner=[str(settings.PYTHON),str(settings.ROOT/'validate_recovery.py'),'--run-dir',str(run),'--budget-s',str(budget)]
    elif args.adapter_validation:
        planner=[str(settings.PYTHON),str(settings.ROOT/'validate_adapter.py'),'--run-dir',str(run),'--budget-s',str(budget)]
    elif args.scene_smoke:
        planner += ['--scene-smoke']
    elif not args.rollout:
        planner += ['--integration-smoke']
    configuration = {'implementation':'panda-unified-skills/gpu-sdf-collision-cpu-control-v33','source_code_dir':str(settings.ROOT),'cad_meters_per_source_unit':settings.CAD_METERS_PER_SOURCE_UNIT,'request_store':os.environ.get('ASSEMBLY_REQUEST_STORE'),'controller':args.controller,'collision_backend':args.collision_backend,'dynamic_mesh_sdf_resolution':args.sdf_resolution,'video_fps':5,'observation_resolution':[960,720],'moving_asset':moving_asset,'moving_cad_cache':moving_cache,'moving_part_mass_profile':cover_material,'scene_fixture_asset':fixture,'scenario': args.case, 'planner': 'scene-smoke-no-model' if args.scene_smoke else 'RPent ApiAgentLoop', 'action_policy': 'none: initial observation only' if args.scene_smoke else ('public RGB-D/CAD 6D grasp + bounded robot-model IK trajectory + relational contact assembly' if generic() else 'HRC visual geometry and Cartesian servo'), 'docker_command': cmd, 'planner_command': planner, 'rollout': args.rollout,'scene_smoke':args.scene_smoke,'robot':args.robot,'task':args.task,'model':os.environ.get('OPENAI_MODEL'),'model_base_url':os.environ.get('OPENAI_BASE_URL'),'adapter':'generic' if generic() else 'original_factory'}
    configuration['method']=args.method
    configuration['condition']=args.condition
    configuration['diagnostic_initial_arm_joints_rad']=args.adapter_start_joints
    (run / 'configuration.json').write_text(json.dumps(configuration, indent=2))
    print(run, flush=True)
    started = False
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        started = True
        with (run / 'private/planner.log').open('w') as stream:
            with subprocess.Popen(planner, stdout=stream, stderr=subprocess.STDOUT) as process:
                deadline = time.monotonic() + 600 + budget + 60
                while process.poll() is None:
                    state = subprocess.run(['docker', 'inspect', '--format', '{{.State.Running}}', name], capture_output=True, text=True)
                    if state.stdout.strip() != 'true':
                        failure = run / 'private/backend-failed.json'
                        if not failure.exists():
                            failure.write_text(json.dumps({'error': 'Isaac container exited; see private/backend.log'}))
                    if time.monotonic() > deadline:
                        process.terminate()
                        raise TimeoutError('Planner exceeded startup plus episode budget')
                    time.sleep(2)
                returncode = process.returncode
        if returncode:
            raise RuntimeError('RPent planner failed; see ' + str(run / 'private/planner.log'))
    finally:
        if started:
            subprocess.run(['docker', 'stop', '-t', '15', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with (run / 'private/backend.log').open('w') as stream:
                subprocess.run(['docker', 'logs', name], stdout=stream, stderr=subprocess.STDOUT)
            subprocess.run(['docker', 'rm', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([str(settings.PYTHON), str(settings.ROOT / 'export_activity.py'), str(run)], check=True)


if __name__ == '__main__':
    main()
