"""Physical hub-seat calibration trials, explicitly separate from model rollouts."""
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for line in (ROOT / '.env').read_text().splitlines():
        fields = shlex.split(line, comments=True)
        if fields and fields[0] == 'export':fields = fields[1:]
        if fields and '=' in fields[0]:
            key, value = fields[0].split('=', 1)
            os.environ[key] = value
    os.environ['ASSEMBLY_ASSET_ROOT'] = str(Path(os.environ['ASSEMBLY_ASSET_ROOT']).expanduser().resolve())
    record = {'type': 'privileged physical calibration, not autonomous assembly',
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'status': 'running', 'trials': [],
              'remaining_scope': ['peg M6 bolt', 'M10 casing bolt', 'input-top bearing task18, user-selected middle seat B'],
              'note': 'Drop-to-seat diagnostics do not certify pickup or full robot assembly; no calibration is automatically enabled.'}
    tasks = sys.argv[3].split(',') if len(sys.argv)>3 else ['hub-cover-small', 'hub-cover', 'bearing-input-top-onto-shaft']
    for task in tasks:
        modes = {'bearing-input-top-onto-shaft':['bearing-middle-seat-probe','bearing-middle-offset-probe','bearing-middle-flipped-probe'], 'peg':['peg-seat-probe','peg-offset-probe'], 'm10-casing-bolt':['m10-seat-probe']}.get(task,['hub-positive-near-seat'])
        if len(sys.argv)>5:modes=sys.argv[5].split(',')
        for mode in modes:
            run = output / (task + '-' + mode)
            row = {'task': task, 'mode': mode, 'run_dir': str(run), 'status': 'running'}
            record['trials'].append(row)
            (output / 'status.json').write_text(json.dumps(record, indent=2))
            command=[str(ROOT / '.venv/bin/python'), str(ROOT / 'launch.py'),
                '--task', task, '--robot', sys.argv[6] if len(sys.argv)>6 else 'panda', '--gpu', sys.argv[2],
                '--sdf-resolution', sys.argv[4] if len(sys.argv)>4 else '256',
                '--verifier-calibration', mode, '--budget-s', '120', '--run-dir', str(run)]
            if len(sys.argv)>7:
                if sys.argv[7]!='peg-mvp' or task!='peg':raise ValueError('Unsupported optional calibration mode')
                command+=['--peg-mvp']
            with (output / (run.name + '.log')).open('w') as log:
                result = subprocess.run(command,
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            row.update(status='ended_requires_review', returncode=result.returncode)
            summary = run / 'episode/backend-summary.json'
            if summary.exists():row['summary'] = json.loads(summary.read_text())
            (output / 'status.json').write_text(json.dumps(record, indent=2))
    record['status'] = 'seat_probes_finished_requires_physical_review'
    (output / 'status.json').write_text(json.dumps(record, indent=2))


if __name__ == '__main__':main()
