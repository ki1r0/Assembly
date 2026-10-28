"""Three current-harness xArm7 physical attempts; keep failures and no auto-retry."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def commands(output,gpu,condition,budget,physics_hz=480):
    suite=json.loads((ROOT/'config/xarm7_robustness_suite.json').read_text())
    for index,case in enumerate(suite['cases'],1):
        name=f'{index:02d}-{case["task"]}-{case["case"]}'
        run=Path(output)/name
        yield case,run,[str(ROOT/'.venv/bin/python'),str(ROOT/'launch.py'),
            '--robot','xarm7','--method','ours','--controller','reference_ik_high_pd',
            '--task',case['task'],'--case',case['case'],'--condition',condition,
            '--rollout','--gpu',str(gpu),'--budget-s',str(budget),'--xarm-physics-hz',str(physics_hz),'--run-dir',str(run)]+(['--peg-mvp'] if case['task']=='peg' else [])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--gpu',default='1')
    parser.add_argument('--condition',choices=['clean','environment','input_error'],default='clean')
    parser.add_argument('--budget-s',type=float,default=1800.)
    parser.add_argument('--physics-hz',type=int,choices=[240,480],default=480)
    args=parser.parse_args()
    if not args.budget_s>0:parser.error('Positive budget required')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    for line in (ROOT/'.env').read_text().splitlines():
        fields=shlex.split(line,comments=True)
        if fields and fields[0]=='export':fields=fields[1:]
        if fields and '=' in fields[0]:
            key,value=fields[0].split('=',1);os.environ[key]=value
    os.environ['ASSEMBLY_ASSET_ROOT'] = str(Path(os.environ['ASSEMBLY_ASSET_ROOT']).expanduser().resolve())
    record={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'scope':'Simulation with assumed xArm7/G1/RGB-D and simulated wrench; not real hardware',
            'condition':args.condition,'physics_hz':args.physics_hz,'status':'running','trials':[]}
    def save():
        temporary=output/'batch-status.json.tmp'
        temporary.write_text(json.dumps(record,indent=2));temporary.replace(output/'batch-status.json')
    save()
    for case,run,command in commands(output,args.gpu,args.condition,args.budget_s,args.physics_hz):
        row={**case,'run_dir':str(run),'status':'running','command':command}
        record['trials'].append(row);save()
        with (output/(run.name+'.log')).open('w') as stream:
            result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        row['returncode']=result.returncode
        summary=run/'episode/backend-summary.json'
        if summary.exists():
            row['summary']=json.loads(summary.read_text())
            row['status']='ended_requires_review'
        else:row['status']='infrastructure_failure_without_summary'
        save()
    record['status']='attempts_finished_not_success_claim'
    record['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()


if __name__=='__main__':main()
