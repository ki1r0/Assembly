"""Sequential current-harness exploration; preserve every task and failure."""
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    batch = Path(sys.argv[1]).resolve()
    gpu = sys.argv[2]
    batch.mkdir(parents=True, exist_ok=False)
    for line in (ROOT / '.env').read_text().splitlines():
        fields = shlex.split(line, comments=True)
        if fields and fields[0] == 'export':
            fields = fields[1:]
        if fields and '=' in fields[0]:
            key, value = fields[0].split('=', 1)
            os.environ[key] = value
    os.environ['ASSEMBLY_ASSET_ROOT'] = str(Path(os.environ['ASSEMBLY_ASSET_ROOT']).expanduser().resolve())
    # R1 asset root must be provided by the caller for R1-only tasks.
    tasks = json.loads((ROOT / 'config/gearbox_single_step_tasks.json').read_text())['tasks']
    pending = {'oil-level-indicator': 'User awaits drawing or explicit installation definition',
               'm10-casing-nut': 'User keeps non-helical threaded asset pending'}
    aliases = {'peg-m6-hub-bolt': 'peg', 'hub-cover-output': 'hub-cover'}
    dual = {'output-shaft-into-base', 'transfer-shaft-into-base',
            'input-shaft-into-base', 'casing-top-to-base', 'output-gear-onto-shaft'}
    status = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'budget_s_per_task': 1800, 'gpu': gpu, 'status': 'running',
              'scope': '18 deduplicated tasks; two user-pending exceptions; exploration, not equal-budget baseline',
              'tasks': []}
    def save():
        (batch / 'batch-status.json').write_text(json.dumps(status, indent=2))
    save()
    for index, entry in enumerate(tasks, 1):
        name = entry['id']
        row = {'index': index, 'manifest_id': name}
        status['tasks'].append(row)
        if name in pending:
            row.update(status='pending', reason=pending[name]); save(); continue
        task = aliases.get(name, name)
        robot = 'r1' if task in dual else 'panda'
        run = batch / f'{index:02d}-{task}'
        row.update(task=task, robot=robot, run_dir=str(run), status='running')
        save()
        command = [str(ROOT / '.venv/bin/python'), str(ROOT / 'launch.py'),
                   '--task', task, '--robot', robot, '--rollout', '--gpu', gpu,
                   '--budget-s', '1800', '--run-dir', str(run)]
        with (batch / f'{index:02d}-{task}.log').open('w') as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        row['returncode'] = result.returncode
        summary = run / 'episode/backend-summary.json'
        if summary.exists():
            data = json.loads(summary.read_text())
            row.update(status=data.get('terminal', {}).get('source', 'ended'),
                       native_success=data.get('native_success', False),
                       physics_steps=data.get('physics_steps'), video_frames=data.get('video_frames'),
                       video_error=data.get('video_error'))
        else:
            row['status'] = 'infrastructure_failure_no_terminal_summary'
        save()
    status['status'] = 'finished_attempts_not_success_claim'
    status['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()


if __name__ == '__main__':
    main()
