"""Explicitly privileged setup-only evaluator diagnostic; never a model rollout."""
import argparse,json,time
import settings
from common import write
from m1_reach.control import start_decision,terminate,snapshot
from generic_toolkit import RobotToolkit
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--budget-s',type=float,default=60);a=p.parse_args();run=settings.configure(a.run_dir);started=time.monotonic()
while not (run/'ipc/reply_0000.json').exists():
 if (run/'private/backend-failed.json').exists():raise RuntimeError('backend initialization failed')
 if time.monotonic()-started>600:raise TimeoutError('startup timeout')
 time.sleep(.2)
start_decision(a.budget_s);toolkit=RobotToolkit(run)
try:
 for i in range(6):
  toolkit.rpc({'action':'settle','duration_s':1.})
  if snapshot()['terminal']:break
 if not snapshot()['terminal']:terminate('evaluator_calibration_complete')
finally:toolkit.close()
