import ast
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_kinematics import transform


def run_transport(supported=True,lose_contact=False):
    tree=ast.parse((Path(__file__).parents[1]/'generic_backend.py').read_text())
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='transit_pass')
    wrapper=ast.parse("def probe():\n global grasp_geometry\n transport_visual_registration={'status':'not_applicable'}\n pass\n return transport_visual_registration\n")
    wrapper.body[0].body[2]=loop
    state=np.array([0.,0.,.5,1.,0.,0.,0.]);calls=[];observations=[];payload=[]
    reference=state.copy();moving={'position_m':[0.,.002,.1],'axis_world':[0,1,0],
        'rigid_grasp_reference_tcp':reference.tolist(),'cad_pose_at_reference':np.eye(4).tolist()}
    component={'cad_feature_local':[0.,.002,0.],'cad_feature_axis_local':[0.,1.,0.]}
    refined=np.eye(4);refined[:3,:3]=Rotation.from_euler('x',2,degrees=True).as_matrix();refined[:3,3]=[.002,.003,.35]
    records={'held-cad-0':dict(moving),'old-held':{'rigid_grasp_reference_tcp':reference.tolist()},'target':{'position_m':[0,0,.3]}}
    def targets(current,feature,target,offset,alignment):
        pre=np.r_[np.array(feature['position_m'])[:2],.4,1.,0.,0.,0.]
        seat=pre.copy();seat[2]=.3
        return .003,pre,seat
    def move(request,verify_retention):
        assert verify_retention
        calls.append(np.array(request['left']));state[:]=request['left']
        return {'status':'grasp_retention_unconfirmed' if lose_contact else 'reached'}
    def observe(prior,arm,stage):
        observations.append((prior,arm,stage))
        return {'status':'visually_refined' if supported else 'rejected',
            'registration':{'world_from_cad':refined.tolist(),'rms_m':.0008},'observation_id':'held-transport-1'},state.copy()
    namespace=dict(np=np,Rotation=Rotation,transform=transform,pose=lambda arm:state.copy(),
        assembly_targets=targets,moving=moving,target_feature=records['target'],offset=0.,axis_alignment='nearest',
        stages=[],active_arm='left',require_retention=True,move=move,feature_records=records,
        grasp_geometry={'candidate':{'part_component':component}},robot_name='xarm7',
        reobserve_held_cad=observe,c={'moving_feature_id':'held-cad-0'},
        refresh_payload_estimate=lambda *args:payload.append(args))
    exec(compile(ast.fix_missing_locations(wrapper),'transport','exec'),namespace)
    result=namespace['probe']()
    return result,moving,records,calls,observations,payload,refined,namespace


@pytest.mark.parametrize('supported',[True,False])
def test_transport_relocalizes_once_without_using_rejected_estimates(supported):
    result,moving,records,calls,observations,payload,refined,_=run_transport(supported)
    assert len(observations)==1 and observations[0][2]=='transport'
    assert len(calls)==(6 if supported else 3)
    assert len(payload)==int(supported)
    if supported:
        expected=(refined@np.array([0.,.002,0.,1.]))[:3]
        np.testing.assert_allclose(moving['position_m'],expected)
        np.testing.assert_allclose(moving['axis_world'],refined[:3,:3]@[0,1,0])
        assert 'old-held' not in records and 'target' in records
        assert result['corrective_pregrasp_reached']
        assert records['held-cad-0']['source_observation_id']=='held-transport-1'
    else:
        assert moving['position_m']==[0.,.002,.1]
        assert 'old-held' in records


def test_retention_loss_still_holds_and_invalidates_before_visual_correction():
    result,_,records,calls,observations,payload,_,namespace=run_transport(lose_contact=True)
    assert result['status']=='grasp_retention_unconfirmed'
    assert len(calls)==1 and not observations and not payload
    assert namespace['grasp_geometry'] is None
    assert list(records)==['target']
