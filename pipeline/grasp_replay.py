"""Bind a diagnostic replay to geometry rather than frame-local candidate IDs."""
import numpy as np
from scipy.spatial.transform import Rotation


def replay_selection(records,candidate_id=None):
    plan=next(row['command'] for row in records if row['command']['action']=='plan_grasp')
    grasp=dict(next(row['command'] for row in records if row['command']['action']=='execute_grasp'))
    if candidate_id is not None:grasp['candidate_id']=candidate_id
    candidates=next(row['result']['candidates'] for row in records if row['command']==plan)
    reference=next((c for c in candidates if c['candidate_id']==grasp['candidate_id']),None)
    if reference is None:raise ValueError('Requested diagnostic candidate is absent from the recorded public plan')
    return plan,grasp,reference


def recorded_retreat(records):
    after_grasp=False
    for row in records:
        command=row['command']
        if command['action']=='execute_grasp':after_grasp=True
        elif after_grasp and command['action']=='plan_grasp':return None
        elif after_grasp and command['action']=='move_relative':return dict(command)
    return None


def match_replay_candidate(reference, candidates):
    reference_position=np.asarray(reference['position_m'])
    reference_rotation=Rotation.from_quat(np.asarray(reference['quaternion_wxyz'])[[1,2,3,0]])
    matches=[]
    for candidate in candidates:
        distance=float(np.linalg.norm(np.asarray(candidate['position_m'])-reference_position))
        rotation=Rotation.from_quat(np.asarray(candidate['quaternion_wxyz'])[[1,2,3,0]])
        angle=float((rotation*reference_rotation.inv()).magnitude())
        if distance<=.001 and angle<=.03:
            matches.append((distance+.01*angle,candidate))
    return min(matches,key=lambda row:row[0])[1] if matches else None
