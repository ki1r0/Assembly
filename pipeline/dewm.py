"""Public-observation DEWM mechanisms adapted from arXiv:2602.15549.

Landmark envelopes are not complete object poses or physical contact evidence.
The actor supplies semantic labels/reliability; geometry must come from tools.
"""
from copy import deepcopy
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


RELIABILITY_GAIN = {'High': 1., 'Medium': .5, 'Low': .1, 'Bad': .01}


def vector(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError('Expected a finite world-metre 3-vector')
    return result


def envelope_iou(left, right):
    a,b=np.asarray(left,dtype=float),np.asarray(right,dtype=float)
    if any(x.shape!=(2,3) or not np.isfinite(x).all() or (x[1]<x[0]).any() for x in (a,b)):
        raise ValueError('Finite ordered world bounds required')
    intersection=float(np.prod(np.maximum(0.,np.minimum(a[1],b[1])-np.maximum(a[0],b[0]))))
    union=float(np.prod(a[1]-a[0])+np.prod(b[1]-b[0])-intersection)
    return intersection/union if union>0 else 0.


class WorldMemory:
    def __init__(self, task_instruction=''):
        self.entities = {}
        self.observations = set()
        self.working_memory = []
        self.constraint_state = 'Unknown'
        self.pending = None
        self.task_instruction=task_instruction
        self.verified_relations=[]

    def update(self, observations, observation_id):
        """Geometric gating, semantic association, Hungarian match, information fusion."""
        if not isinstance(observation_id, str) or not observation_id:
            raise ValueError('A real observation ID is required')
        measurements = []
        submitted = set()
        for item in observations:
            if not item.get('label') or not item.get('feature_id'):
                raise ValueError('Measured feature ID and semantic label required')
            gain = RELIABILITY_GAIN[item['reliability']]
            position = vector(item['position_m'])
            sigma = float(item['position_std_m'])
            if not math.isfinite(sigma) or sigma <= 0:
                raise ValueError('Positive finite measurement uncertainty required')
            covariance=np.asarray(item.get('position_covariance_m2',np.eye(3)*sigma**2),dtype=float)
            if covariance.shape!=(3,3) or not np.isfinite(covariance).all() or not np.allclose(covariance,covariance.T) or np.linalg.eigvalsh(covariance).min()<=0:
                raise ValueError('Positive definite symmetric measured covariance required')
            if 'bounds_world_m' in item:envelope_iou(item['bounds_world_m'],item['bounds_world_m'])
            key = (observation_id, item['feature_id'])
            if key in self.observations or key in submitted:
                continue
            submitted.add(key)
            measurements.append((item, position, covariance, gain, key))
        ids = list(self.entities)
        costs = np.full((len(measurements), len(ids)), 1e9)
        for i, (item, position, covariance, gain, _) in enumerate(measurements):
            for j, entity_id in enumerate(ids):
                old = self.entities[entity_id]
                distance = float(np.linalg.norm(position - old['position_m']))
                displacement=position-old['position_m']
                normalized=float(displacement@np.linalg.solve(covariance+old['covariance_m2'],displacement))
                has_bounds='bounds_world_m' in item
                if has_bounds != ('bounds_world_m' in old):continue
                semantic=float(item.get('association_scores',{}).get(entity_id,1. if old['label']==item['label'] else 0.))
                if not math.isfinite(semantic) or not 0<=semantic<=1:raise ValueError('Semantic association score must be in [0,1]')
                if gain > 0 and semantic>=.5 and normalized <= 11.345 and distance <= .05:
                    if has_bounds:
                        cost=.7*(1-envelope_iou(item['bounds_world_m'],old['bounds_world_m']))+.3*(1-semantic)
                        if cost<=.55:costs[i,j]=cost
                    else:costs[i, j] = normalized
        matches = {}
        if costs.size:
            rows, cols = linear_sum_assignment(costs)
            matches = {int(i): ids[int(j)] for i, j in zip(rows, cols) if costs[i, j] < 1e9}
        updated = []
        for i, (item, position, covariance, gain, key) in enumerate(measurements):
            self.observations.add(key)
            if gain == 0:
                continue
            entity_id = matches.get(i)
            if entity_id is None:
                entity_id = f'entity-{len(self.entities)}'
                entity = {'label': item['label'], 'position_m': position.tolist(),
                          'covariance_m2':(covariance/gain).tolist(),
                          'variance_m2':float(np.trace(covariance/gain)/3), 'observed_frames': 1}
                self.entities[entity_id] = entity
            else:
                entity = self.entities[entity_id]
                if entity['observation_id'] == observation_id:
                    updated.append(entity_id)
                    continue
                old_information = np.linalg.inv(entity['covariance_m2'])
                new_information = gain*np.linalg.inv(covariance)
                fused_covariance=np.linalg.inv(old_information+new_information)
                position = fused_covariance@(old_information@entity['position_m']+new_information@position)
                entity.update(position_m=position.tolist(), covariance_m2=fused_covariance.tolist(),variance_m2=float(np.trace(fused_covariance)/3),
                              observed_frames=entity['observed_frames']+1)
            entity.update(observation_id=observation_id, feature_id=item['feature_id'],
                          reliability=item['reliability'], lifecycle='Active' if entity['observed_frames']>=2 else 'Tentative',
                          geometry_scope='observed visible envelope, not complete hidden volume' if 'bounds_world_m' in item else 'observed landmark, not a complete object pose')
            if 'bounds_world_m' in item:entity['bounds_world_m']=deepcopy(item['bounds_world_m'])
            if item.get('axis_world') is not None:
                axis = vector(item['axis_world'])
                if np.linalg.norm(axis) < 1e-8:
                    raise ValueError('Nonzero observed axis required')
                entity['axis_world'] = (axis/np.linalg.norm(axis)).tolist()
            updated.append(entity_id)
        return updated

    def invalidate_after_motion(self):
        # Occlusion/motion is uncertainty, never evidence of absence.
        self.verified_relations=[]
        for entity in self.entities.values():
            entity['lifecycle'] = 'Uncertain'
            covariance=np.asarray(entity['covariance_m2'])+np.eye(3)*.005**2
            entity['covariance_m2']=covariance.tolist()
            entity['variance_m2']=float(np.trace(covariance)/3)

    def relation(self, relation, observation_id):
        """Three-valued geometry check; unsupported contact/containment remains unknown."""
        args = relation.get('args', [])
        if len(args) != 2 or any(key not in self.entities for key in args):
            raise ValueError('Relation needs two existing entity IDs')
        a, b = [self.entities[key] for key in args]
        if any(x['lifecycle'] != 'Active' for x in (a, b)):
            return None
        if relation['pred'] == 'Near':
            return bool(np.linalg.norm(vector(a['position_m'])-vector(b['position_m'])) < .05)
        if relation['pred'] == 'Coaxial':
            if 'axis_world' not in a or 'axis_world' not in b:
                return None
            axis = vector(b['axis_world'])
            displacement = vector(a['position_m'])-vector(b['position_m'])
            radial = np.linalg.norm(displacement-np.dot(displacement, axis)*axis)
            angle = math.degrees(math.acos(float(np.clip(abs(np.dot(a['axis_world'], axis)), 0, 1))))
            return bool(radial <= .005 and angle <= 5)
        return None

    def propose(self, ert, observation_id, allowed_actions):
        required = {'action_proposal', 'world_belief', 'causal_assump', 'confidence', 'fallback'}
        if not isinstance(ert, dict) or set(ert) != required:
            raise ValueError('ERT requires exactly the five declared fields')
        if self.pending is not None:
            raise ValueError('Verify the previous action before proposing another')
        confidence = ert['confidence']
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('ERT confidence must be finite in [0,1]')
        if not isinstance(ert['fallback'], str) or not ert['fallback'].strip():
            raise ValueError('Explicit recovery strategy required')
        action = ert['action_proposal']
        if not isinstance(action, dict) or action.get('action') not in allowed_actions:
            raise ValueError('Proposed action must use an exposed shared capability')
        for key in ('world_belief', 'causal_assump'):
            if not isinstance(ert[key], list):
                raise ValueError(key+' must be a list of structured relation assertions')
            for assertion in ert[key]:
                result = self.relation(assertion, observation_id)
                if key == 'world_belief' and result is not True:
                    raise ValueError('World belief lacks current public geometric support')
        self.pending = {'ert': deepcopy(ert), 'source_observation_id': observation_id,
                        'controller_result': None, 'result_observation_id': None}
        return deepcopy(action)

    def executed(self, outcome, result_observation_id):
        if self.pending is None:
            raise ValueError('No pending ERT action')
        self.pending.update(controller_result=deepcopy(outcome), result_observation_id=result_observation_id)
        self.invalidate_after_motion()

    def verify(self, observation_id):
        if self.pending is None or self.pending['controller_result'] is None:
            raise ValueError('No executed action to verify')
        if observation_id == self.pending['source_observation_id']:
            raise ValueError('Post-action verification requires a new public observation')
        pending = self.pending
        checks = [{'relation': relation, 'observed': self.relation(relation, observation_id)}
                  for relation in pending['ert']['causal_assump']]
        outcome = pending['controller_result']
        status = str(outcome.get('status', '')).lower()
        rejected = bool(outcome.get('error')) or any(word in status for word in ('reject', 'fail', 'stop', 'timeout', 'abort'))
        confirmed = bool(checks) and all(item['observed'] is True for item in checks) and not rejected
        discrepancy = {'missing_expected_relations': [item['relation'] for item in checks if item['observed'] is False],
                       'unknown_expected_relations': [item['relation'] for item in checks if item['observed'] is None],
                       'controller_failure': rejected}
        record = pending | {'verification_observation_id': observation_id,
                            'commit': confirmed, 'discrepancy': discrepancy,
                            'scope': 'Geometric belief verification only, not physical task success'}
        # Predictions are never inserted as facts: failed hypotheses need no physical rollback.
        self.working_memory.append(record)
        if confirmed:self.verified_relations=deepcopy(pending['ert']['causal_assump'])
        self.pending = None
        self.constraint_state = 'VerifiedGeometry' if confirmed else 'NeedsDiagnosis'
        return deepcopy(record)

    def context(self):
        return deepcopy({'environment_memory': self.entities, 'constraint_state': self.constraint_state,
                         'task_instruction':self.task_instruction,
                         'semantic_graph':{'zones':['workcell'],
                             'entities':{key:{'label':value['label'],'lifecycle':value['lifecycle'],'zone':'workcell'} for key,value in self.entities.items()},
                             'verified_geometric_relations':self.verified_relations,
                             'zone_scope':'Logical task workspace, not a measured containment claim'},
                         'pending_action': self.pending, 'recent_working_memory': self.working_memory[-6:]})
