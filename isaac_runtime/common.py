import json,time,os
from pathlib import Path
R=Path(os.environ.get("ASSEMBLY_RUN_DIR",Path(__file__).resolve().parent))
os.environ['M1_CONTROL_PATH']=str(R/'private/control.json')
from m1_reach.control import snapshot,terminate,remaining

def write(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,default=str,indent=2));tmp.replace(p)
def log(name,data):
    with (R/'episode'/name).open('a') as f:f.write(json.dumps({'wall_time':time.time(),'monotonic':time.monotonic(),**data},ensure_ascii=False,default=str)+'\n')

def execution_observation(raw):
    """Use the original public projection expected at the runtime action boundary."""
    from runtime.assembly_harness.agent import project_observation
    raw=dict(raw)
    # Control-step observations may have fewer than eight frames and no overlay.
    # The single current RGB is sufficient for public-state normalization here.
    raw.pop('frame_sequence',None)
    frames=raw['frames'];key=next(reversed(frames));raw['frames']={key:frames[key]}
    projected,_=project_observation(raw)
    projected['frame_sequence']=[{'frame_key':key}]
    sensors=projected['robot_state']['sensors']
    if 'gripper_width' in sensors:sensors['gripper_width_m']=sensors['gripper_width']
    return projected
