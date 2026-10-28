"""Robot-only measured validation, not a model assembly capability result."""
import argparse,json,time
import numpy as np
import settings
from common import write
from m1_reach.control import start_decision,terminate,snapshot
from generic_toolkit import RobotToolkit
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--budget-s',type=float,default=180);a=p.parse_args();run=settings.configure(a.run_dir)
started=time.monotonic()
while not (run/'ipc/reply_0000.json').exists():
 if (run/'private/backend-failed.json').exists():raise RuntimeError('backend initialization failed')
 if time.monotonic()-started>600:raise TimeoutError('startup timeout')
 time.sleep(.2)
start_decision(a.budget_s);toolkit=RobotToolkit(run);results=[]
replay=__import__('os').environ.get('ASSEMBLY_ADAPTER_GRASP_REPLAY')
if replay:
 try:
  records=[json.loads(line) for line in open(replay)]
  from grasp_replay import replay_selection
  plan,grasp,reference=replay_selection(records,__import__('os').environ.get('ASSEMBLY_ADAPTER_GRASP_CANDIDATE_ID'))
  for command in [plan,grasp]:
   if command['action']=='execute_grasp':
    from grasp_replay import match_replay_candidate
    matched=match_replay_candidate(reference,toolkit.latest['result'].get('candidates',[]))
    if matched is None:
     results.append({'status':'matching_candidate_not_available','reference_candidate':reference})
     terminate('adapter_grasp_replay_candidate_changed')
     break
    command=dict(command,candidate_id=matched['candidate_id'])
    force_override=__import__('os').environ.get('ASSEMBLY_ADAPTER_GRIP_FORCE_N')
    if force_override is not None:command['grip_force_n']=float(force_override)
   toolkit.rpc(command)
   results.append({'command':command,'result':toolkit.latest['result']})
   if snapshot()['terminal']:break
  if __import__('os').environ.get('ASSEMBLY_ADAPTER_REPLAY_RETREAT')=='1' and not snapshot()['terminal'] and toolkit.latest['result'].get('status') in {'grasp_retention_unconfirmed','grasp_contact_incomplete'}:
   from grasp_replay import recorded_retreat
   retreat=recorded_retreat(records)
   if retreat is not None:
    toolkit.rpc(retreat)
    results.append({'command':retreat,'result':toolkit.latest['result'],'scope':'Replay of recorded public retreat after failed grasp; not an autonomous policy outcome'})
  write(run/'checks/grasp-replay.json',{'source':replay,'results':results,
    'scope':'Deterministic replay of public plan selection and grasp request with fresh RGB-D candidates; not an autonomous policy result or assembly success.'})
  if not snapshot()['terminal']:terminate('adapter_grasp_replay_complete')
 finally:toolkit.close()
 raise SystemExit(0)
if __import__('os').environ.get('ASSEMBLY_ADAPTER_GRIP_ONLY')=='1':
 try:
  for width in [.03,.06]:
   toolkit.rpc({'action':'set_gripper','width_m':width})
   results.append({'requested_width_m':width,'result':toolkit.latest['result']})
   if snapshot()['terminal']:break
  write(run/'checks/gripper-linkage-validation.json',{'results':results,'scope':'Unloaded gripper-only mechanical validation; not grasp or assembly success'})
  if not snapshot()['terminal']:terminate('gripper_linkage_validation_complete')
 finally:toolkit.close()
 raise SystemExit(0)
if toolkit.latest.get('robot')=='r1':
 try:
  initial={arm:np.array(state['tcp_pose']) for arm,state in toolkit.latest['arms'].items()}
  for arm in ['left','right']:
   toolkit.rpc({'action':'select_arm','arm':arm})
   for delta in [.005,0.]:
    target=initial[arm].copy();target[0]+=delta
    toolkit.rpc({'action':'move_to','position_m':target[:3].tolist(),'quaternion_wxyz':target[3:].tolist()})
    results.append({'kind':'independent_arm','arm':arm,'result':toolkit.latest['result']})
    if snapshot()['terminal']:break
   if snapshot()['terminal']:break
  if not snapshot()['terminal']:
   for delta in [.005,0.]:
    command={'action':'move_both'}
    for arm in ['left','right']:
     target=initial[arm].copy();target[2]+=delta
     command[arm+'_position_m']=target[:3].tolist();command[arm+'_quaternion_wxyz']=target[3:].tolist()
    toolkit.rpc(command);results.append({'kind':'both_arms','result':toolkit.latest['result']})
    if snapshot()['terminal']:break
  if not snapshot()['terminal']:
   for width in [.02,.06]:
    toolkit.rpc({'action':'set_both_grippers','left_width_m':width,'right_width_m':width})
    measured={arm:state['gripper_width_m'] for arm,state in toolkit.latest['arms'].items()}
    results.append({'kind':'both_grippers','target_m':width,'measured_m':measured,'result':toolkit.latest['result']})
  passed=len(results)==8 and not snapshot()['terminal'] and all(row['result'].get('status') in {'reached','gripper_commanded'} and all(error<.001 for error in row['result'].get('errors_m',{}).values()) and all(abs(width-row['target_m'])<.003 for width in row.get('measured_m',{}).values()) for row in results)
  write(run/'checks/adapter-validation.json',{'passed_unloaded':passed,'results':results,'scope':'Independent and simultaneous 5mm translations and both grippers only; not payload, grasp geometry or assembly verification.'})
  if not snapshot()['terminal']:terminate('adapter_validation_complete',passed=passed)
 finally:toolkit.close()
 raise SystemExit(0)
try:
 initial=np.array(toolkit.latest['sensors']['tcp_pose'])
 for axis in range(3):
  for sign in [-1,1]:
   target=initial.copy();target[axis]+=.01*sign
   out=toolkit.rpc({'action':'move_to','position_m':target[:3].tolist(),'quaternion_wxyz':target[3:].tolist()});result=toolkit.latest['result'];results.append({'kind':'translation','axis':axis,'sign':sign,'result':result})
   if snapshot()['terminal']:break
   toolkit.rpc({'action':'move_to','position_m':initial[:3].tolist(),'quaternion_wxyz':initial[3:].tolist()})
  if snapshot()['terminal']:break
 if not snapshot()['terminal']:
  for axis in range(3):
   vector=[0.,0,0];vector[axis]=.08
   toolkit.rpc({'action':'rotate','rotation_rad':vector});results.append({'kind':'rotation','axis':axis,'result':toolkit.latest['result']})
   if snapshot()['terminal']:break
   toolkit.rpc({'action':'move_to','position_m':initial[:3].tolist(),'quaternion_wxyz':initial[3:].tolist()})
 if not snapshot()['terminal']:
  for axis in range(3):
   target=initial.copy();target[axis]+=[-.08,.08,-.10][axis]
   toolkit.rpc({'action':'move_to','position_m':target[:3].tolist(),'quaternion_wxyz':target[3:].tolist()});results.append({'kind':'workspace_translation','axis':axis,'result':toolkit.latest['result']})
   if snapshot()['terminal']:break
   toolkit.rpc({'action':'move_to','position_m':initial[:3].tolist(),'quaternion_wxyz':initial[3:].tolist()})
  if not snapshot()['terminal']:
   for axis in range(3):
    vector=[0.,0,0];vector[axis]=.4
    toolkit.rpc({'action':'rotate','rotation_rad':vector});results.append({'kind':'workspace_rotation','axis':axis,'result':toolkit.latest['result']})
    if snapshot()['terminal']:break
    toolkit.rpc({'action':'move_to','position_m':initial[:3].tolist(),'quaternion_wxyz':initial[3:].tolist()})
 if not snapshot()['terminal']:
  for width in [.02,.08]:
   toolkit.rpc({'action':'set_gripper','width_m':width});results.append({'kind':'gripper','command':width,'result':toolkit.latest['result']})
 passed=all(r['result'].get('status') in {'reached','gripper_commanded'} and all(m['max_cross_track_error_m']<.004 for m in r['result'].get('path_metrics',{}).values()) for r in results) and len(results)==17 and not snapshot()['terminal']
 write(run/'checks/adapter-validation.json',{'passed_unloaded':passed,'results':results,'payload_validation':'Not covered by unloaded motion. Actual grasp/load trials are recorded separately.','safety_stop_validation':'Terminal rejection covered by unit test; no unsafe real excursion deliberately commanded.'})
 if not snapshot()['terminal']:terminate('adapter_validation_complete',passed=passed)
finally:toolkit.close()
