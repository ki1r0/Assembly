"""Launch an owned Isaac container and the host-side RPent planner."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['nominal-aligned', 'bolt-position-shift', 'bolt-horizontal-pose', 'assembly-position-shift'], default='nominal-aligned')
    parser.add_argument('--run-dir')
    parser.add_argument('--gpu', default='0', help='Host NVIDIA GPU index')
    parser.add_argument('--rollout', action='store_true', help='Run assembly; otherwise run a bounded wiring smoke test')
    parser.add_argument('--budget-s', type=float)
    args = parser.parse_args()
    budget = args.budget_s if args.budget_s is not None else (600 if args.rollout else 120)
    if budget <= 0:
        parser.error('--budget-s must be positive')
    for key in ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL', 'ASSEMBLY_ASSET_ROOT']:
        if not os.environ.get(key):
            parser.error(f'Set {key}; see .env.example')
    assets = Path(os.environ['ASSEMBLY_ASSET_ROOT']).expanduser().resolve()
    required = ['parts/M6 Hub Bolt.usd', 'parts/Casing Top.usd', 'parts/Hub Cover Output.usd', 'vendor/remote_mirror/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd']
    missing = [str(assets / name) for name in required if not (assets / name).is_file()]
    if missing:
        parser.error('Missing external assets: ' + ', '.join(missing))
    tag = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run = settings.configure(args.run_dir or settings.ROOT / 'runs' / tag)
    if (run / 'ipc/reply_0000.json').exists() or list((run / 'ipc').glob('command_*.json')):
        raise ValueError('Choose a new run directory; episode IPC must not be reused')
    name = 'assembly-agent-' + tag.lower()
    cmd = ['docker', 'run', '-d', '--name', name, '--gpus', 'device=' + args.gpu, '--ipc', 'host', '--network', 'host']
    for entry in ['ASSEMBLY_CASE=' + args.case, 'ASSEMBLY_ASSET_ROOT=' + str(assets), 'ASSEMBLY_FAST_COMPONENTS=1', 'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1', 'ACCEPT_EULA=Y', 'OMNI_KIT_ACCEPT_EULA=YES', 'ASSEMBLY_RUN_DIR=' + str(run), 'ASSEMBLY_CAMERA_WIDTH=' + os.environ.get('ASSEMBLY_CAMERA_WIDTH', '960'), 'ASSEMBLY_CAMERA_HEIGHT=' + os.environ.get('ASSEMBLY_CAMERA_HEIGHT', '720')]:
        cmd += ['-e', entry]
    for mount in [f'{settings.ROOT}:{settings.ROOT}:ro', f'{run}:{run}', f'{assets}:{assets}:ro']:
        cmd += ['-v', mount]
    # A read-only source mount would otherwise expose the host planner's key file.
    if (settings.ROOT / '.env').is_file():
        cmd += ['-v', f'/dev/null:{settings.ROOT / ".env"}:ro']
    backend = settings.ROOT / 'backend.py'
    command = 'umask 000; exec /isaac-sim/python.sh ' + shlex.quote(str(backend)) + ' --headless --enable_cameras --device cpu'
    cmd += ['--entrypoint', '/bin/bash', os.environ.get('ISAAC_DOCKER_IMAGE', 'nvcr.io/nvidia/isaac-lab:2.3.0'), '-c', command]
    planner = [str(settings.PYTHON), str(settings.ROOT / 'planner.py'), '--run-dir', str(run), '--budget-s', str(budget)]
    if not args.rollout:
        planner += ['--integration-smoke']
    configuration = {'scenario': args.case, 'planner': 'RPent ApiAgentLoop', 'action_policy': 'HRC visual geometry and Cartesian servo', 'docker_command': cmd, 'planner_command': planner, 'rollout': args.rollout}
    (run / 'configuration.json').write_text(json.dumps(configuration, indent=2))
    print(run, flush=True)
    started = False
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        started = True
        with (run / 'private/planner.log').open('w') as stream:
            result = subprocess.run(planner, stdout=stream, stderr=subprocess.STDOUT, timeout=600 + budget + 60)
        if result.returncode:
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
