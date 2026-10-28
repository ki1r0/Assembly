from pathlib import Path
import ast
import pytest

import numpy as np
from scipy.spatial.transform import Rotation

from assembly_skills import antipodal_grasp_candidates
from assembly_skills import aligned_tcp
from xarm_gripper import G1Geometry


def test_model_tool_description_matches_g1_motion_only_score():
    tree=ast.parse((Path(__file__).parents[1]/'generic_toolkit.py').read_text())
    function=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='plan_grasp')
    description=ast.get_docstring(function)
    assert 'G1 score is motion cost only' in description
    assert 'axial_protrusion_m' in description
    assert 'never just the smaller clearance' in description


@pytest.mark.cad_assets
def test_steep_side_proposal_survives_actual_pad_and_full_opening_screen():
    root=Path(__file__).parents[1]
    geometry=G1Geometry(root/'assets/xarm7/xarm7_g1_simulation_assumption.urdf')
    with np.load(root/'assets/public_cad_models/bolt.npz') as cad:
        points,normals=cad['points'],cad['normals']
    # Public RGB-D registration from the flat-bolt failure; no simulator pose.
    rotation=np.array([[-.9998310362525594,.00278325997153082,.018170096589299793],
                       [.00461964847766551,.994791976980458,.10182132086776291],
                       [-.017792071101375852,.10188805621486749,-.9946367508827937]])
    points=points@rotation.T+[.2999911689685885,.24993374692500842,.6840926526564184]
    normals=normals@rotation.T
    proposals=antipodal_grasp_candidates(points,normals,.68,max_candidates=4096,
        pad_offset_m=geometry.contact_offset_m,clearance_test=lambda candidate,table:True,
        approach_tilts=(-1.2,1.2,-1.3,1.3,-1.4,1.4))
    passed,counts=geometry.screen_candidates(proposals,points,normals,.68,True)
    assert counts['finger_body_collision_rejected']>0
    assert passed
    transverse=[]
    for candidate in passed:
        R=Rotation.from_quat(np.array(candidate['quaternion_wxyz'])[[1,2,3,0]]).as_matrix()
        closed=geometry.drive_for_opening(candidate['observed_width_m'])
        assert geometry.swept_support_clearance(candidate['position_m'],R,
            geometry.drive_for_opening(.08),closed,[0,0,1],[0,0,.68])>=.001
        if abs(R[:,1]@rotation[:,1])<.1:transverse.append(candidate)
    assert transverse
    datum=np.array([.2999911689685885,.24993374692500842,.6840926526564184])+rotation@np.array([0,.00892500114440918,0])
    candidate=transverse[0]
    clearances=geometry.mating_release_clearances(candidate,datum,rotation[:,1])
    assert clearances['antiparallel']<.01
    for mode in ['parallel','antiparallel']:
        aligned=aligned_tcp(np.r_[candidate['position_m'],candidate['quaternion_wxyz']],
            {'position_m':datum,'axis_world':rotation[:,1]},
            {'position_m':[0,0,0],'axis_world':[0,0,1]},0.,mode)
        R=Rotation.from_quat(aligned[[4,5,6,3]]).as_matrix()
        expected=max(0.,.003-geometry.swept_support_clearance(aligned[:3],R,
            geometry.drive_for_opening(candidate['observed_width_m']),geometry.drive_for_opening(.08),[0,0,1],[0,0,0]))
        np.testing.assert_allclose(clearances[mode],expected,atol=1e-12)
