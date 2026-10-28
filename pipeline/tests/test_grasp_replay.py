from grasp_replay import match_replay_candidate


def candidate(identifier, x=0.):
    return {'candidate_id':identifier,'position_m':[x,0.,1.],
            'quaternion_wxyz':[1.,0.,0.,0.]}


def test_frame_local_id_cannot_redirect_diagnostic_to_another_grasp():
    old=candidate('grasp-0')
    assert match_replay_candidate(old,[candidate('grasp-0',.013)]) is None
    new=candidate('grasp-7',.0001)
    assert match_replay_candidate(old,[candidate('grasp-0',.013),new]) is new


def test_changed_orientation_is_not_the_same_physical_grasp():
    changed=candidate('grasp-0');changed['quaternion_wxyz']=[0.,1.,0.,0.]
    assert match_replay_candidate(candidate('old'),[changed]) is None
