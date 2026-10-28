import json
from planner_recovery import continuation_cursor,public_history


def test_recovery_waits_for_pending_action_reply_without_replaying_commands(tmp_path):
    ipc=tmp_path/'ipc';ipc.mkdir()
    (ipc/'command_0000.json').write_text('{}')
    (ipc/'command_0001.json').write_text('{}')
    (ipc/'reply_0001.json').write_text('{}')
    index,reply=continuation_cursor(tmp_path)
    assert index==2
    assert reply==ipc/'reply_0002.json'
    assert not reply.exists()


def test_recovery_history_uses_only_public_outcomes_not_private_grading(tmp_path):
    folder=tmp_path/'episode';folder.mkdir()
    rows=[{'command':{'action':'move_to'},'result':{'status':'reached','actual_poses':{'left':[0]*7},'private_evaluator_pose':[1,2,3],'native_success':True}}]*40
    (folder/'actions.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    history=public_history(tmp_path)
    assert len(history)==32
    assert history[0]['result']=={'status':'reached','actual_poses':{'left':[0]*7}}
