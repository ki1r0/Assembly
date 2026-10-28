"""Exercise malformed references and recovery on real backend, without assembly."""
import argparse,time,json
from pathlib import Path
import settings
from common import write
from m1_reach.control import start_decision,terminate,snapshot
from generic_toolkit import RobotToolkit
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--budget-s',type=float,default=90);a=p.parse_args();run=settings.configure(a.run_dir);deadline=time.monotonic()+600
while not (run/'ipc/reply_0000.json').exists():
 if (run/'private/backend-failed.json').exists():raise RuntimeError('backend initialization failed')
 if time.monotonic()>deadline:raise TimeoutError('backend startup timeout')
 time.sleep(.2)
start_decision(a.budget_s);tools=RobotToolkit(run);rows=[]
try:
 for command in [{'action':'assemble','moving_feature_id':'invalid-ref','target_feature_id':'invalid-ref','search_radius_m':.004},{'action':'execute_grasp','candidate_id':'invalid-candidate'},{'action':'move_relative','translation_m':[-.005,0,0]}]:
  tools.rpc(command);rows.append({'command':command,'result':tools.latest['result']})
 passed=[row['result'].get('status') for row in rows]==['invalid_action','invalid_action','reached'] and not snapshot()['terminal']
 write(run/'checks/recovery-validation.json',{'passed':passed,'results':rows,'no_assembly_policy':True})
 if not snapshot()['terminal']:terminate('protocol_validation_complete',passed=passed)
finally:tools.close()
