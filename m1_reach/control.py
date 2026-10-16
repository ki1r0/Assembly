"""Evaluator-only atomic episode endpoint shared by runner, planner and worker."""
import contextlib, fcntl, json, os, time
from pathlib import Path

@contextlib.contextmanager
def locked():
    path=Path(os.environ['M1_CONTROL_PATH']);path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data=json.loads(path.read_text()) if path.exists() else {'terminal':None,'decision_started_monotonic':None,'deadline_monotonic':None,'last_native_step':0,'sequence':0}
        yield data
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data));tmp.replace(path)
        fcntl.flock(lock,fcntl.LOCK_UN)

def event(data,kind,**extra):
    data['sequence']+=1
    record={'event':kind,'monotonic':time.monotonic(),'wall_time':time.time(),'sequence':data['sequence'],**extra}
    path=Path(os.environ['M1_CONTROL_PATH']).with_suffix('.events.jsonl')
    with path.open('a') as f:f.write(json.dumps(record)+'\n')
    return record

def _terminate(data,source,**extra):
    record=event(data,'endpoint_candidate',source=source,last_native_step=data['last_native_step'],**extra)
    if data['terminal'] is None:data['terminal']=record
    return data['terminal']

def terminate(source,**extra):
    with locked() as data:
        if data['terminal'] is None and data['deadline_monotonic'] is not None and time.monotonic() >= data['deadline_monotonic']:
            _terminate(data,'wallclock_budget')
        return _terminate(data,source,**extra)

def start_decision(seconds):
    with locked() as data:
        if data['decision_started_monotonic'] is None:
            now=time.monotonic();data.update(decision_started_monotonic=now,deadline_monotonic=now+seconds)
            event(data,'decision_started',budget_s=seconds,deadline_monotonic=now+seconds)

def snapshot():
    with locked() as data:return dict(data)

def remaining():
    data=snapshot();return max(.001,data['deadline_monotonic']-time.monotonic())

def begin_step(step):
    blocked=False
    with locked() as data:
        now=time.monotonic()
        if data['terminal'] is None and data['deadline_monotonic'] is not None and now>=data['deadline_monotonic']:_terminate(data,'wallclock_budget')
        if data['terminal'] is not None:
            event(data,'step_rejected',task_step=step+1);blocked=True
        else:event(data,'physical_step_start',task_step=step+1)
    if blocked:raise RuntimeError('Episode ended; no new physical step may start.')

def complete_step(step,started):
    with locked() as data:
        data['last_native_step']=step
        event(data,'physical_step_complete',task_step=step,started_monotonic=started)
