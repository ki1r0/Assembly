"""Public IPC continuity for a separately labelled planner recovery segment."""
import json
from pathlib import Path


def continuation_cursor(run):
    ipc=Path(run)/'ipc'
    commands=sorted(ipc.glob('command_*.json'))
    next_index=max((int(p.stem.split('_')[-1])+1 for p in commands),default=0)
    return next_index,ipc/f'reply_{next_index:04d}.json'


def public_history(run):
    path=Path(run)/'episode/actions.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    fields={'status','feedback','actual_poses','actual_widths_m','closure','post_lift_width_m',
            'post_lift_finger_forces_n','retention_verified','point_world_m','feature_id',
            'position_m','axis_world','estimated_radius_m','next_step'}
    return [{'command':row['command'],'result':{k:v for k,v in row['result'].items() if k in fields}} for row in rows[-32:]]
