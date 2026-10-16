"""Offline tests for the retained interfaces and actor/evaluator boundary."""
import threading
from pathlib import Path

import numpy as np
import pytest
import settings
from cylinder_fit import fit_observed_xy
from rpent.robots.components import BaseVLAClient, BaseVLAFacade
from rpent.utils.rpc.http_rpc import HttpRpcClient, HttpRpcServer
from toolkit import IsaacToolkit
from m1_reach import control


def test_generic_vla_http_round_trip():
    class Policy(BaseVLAFacade):
        def predict(self, obs, options=None):
            return {'actions': np.asarray(obs['state']) + options['offset']}

    policy = Policy()
    server = HttpRpcServer(('127.0.0.1', 0), policy._dispatch)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    client = HttpRpcClient(f'http://127.0.0.1:{server.server_address[1]}')
    try:
        result = BaseVLAClient(client).predict({'state': np.array([1., 2., 3.])}, {'offset': .5})
        np.testing.assert_allclose(result['actions'], [1.5, 2.5, 3.5])
        with pytest.raises(Exception, match='not.*found|Unknown|unknown|Unsupported|unsupported'):
            client.call('missing.route')
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
        policy.close()


def test_actor_observation_drops_private_geometry():
    toolkit = IsaacToolkit.__new__(IsaacToolkit)
    toolkit.latest = {
        'observation_id': 'frame1', 'native_success': False, 'result': {'status': 'observed'},
        'sensors': {'tcp_pose': [0, 0, 1, 1, 0, 0, 0], 'finger_contacts': [False, True], 'true_bolt_pose': [9, 9, 9], 'seat_ready': True},
        'features': [{'feature_ref': 'f1', 'geometry': {'position_m': [1, 2, 3], 'private_target_pose': [9, 9, 9]}, 'uncertainty': {'semantic_identity': 'unknown', 'oracle_identity': 'Bolt'}}],
        'private_evaluator': {'correct_hole': 'secret'},
    }
    public = toolkit.public_data()
    assert public['robot_sensors']['finger_contacts'] == [False, True]
    assert 'true_bolt_pose' not in public['robot_sensors']
    assert 'seat_ready' not in public['robot_sensors']
    assert 'private_target_pose' not in public['features'][0]['geometry']
    assert 'oracle_identity' not in public['features'][0]['uncertainty']
    assert 'private_evaluator' not in public


def test_first_terminal_endpoint_cannot_be_overwritten(tmp_path, monkeypatch):
    monkeypatch.setenv('M1_CONTROL_PATH', str(tmp_path / 'control.json'))
    control.start_decision(60)
    control.terminate('ask_human_help', reason='need repositioning')
    control.terminate('native_success')
    assert control.snapshot()['terminal']['source'] == 'ask_human_help'


def test_observed_bore_fit_recovers_larger_partial_rim():
    rng = np.random.default_rng(42)
    center = np.array([.43, -.14])
    radius = .0072
    theta = np.linspace(-1.2, 1.2, 90)
    xy = center + radius * np.column_stack([np.cos(theta), np.sin(theta)])
    xy += rng.normal(0, .000005, xy.shape)
    points = np.column_stack([xy, np.full(len(xy), .8)])
    fit = fit_observed_xy(points, center + [.001, .0003], radius)
    assert fit is not None
    np.testing.assert_allclose(fit['center_xy_m'], center, atol=.00005)
    assert abs(fit['measured_radius_m'] - radius) < .00005
    assert fit_observed_xy(points[:4], center, radius) is None


def test_memory_rejects_private_paths_and_symlink_escape(tmp_path, monkeypatch):
    from rpent.memory.manager import MemoryManager
    from rpent.utils.logging import init_output_dir
    monkeypatch.setenv('M1_AUTONOMOUS', '1')
    init_output_dir(tmp_path / 'public')
    memory = tmp_path / 'memory'
    memory.mkdir()
    (memory / 'MEMORY.md').write_text('Use observed images')
    private = tmp_path / 'private/evaluator.json'
    private.parent.mkdir()
    private.write_text('{"true_pose": [1, 2, 3]}')
    (memory / 'escaped.json').symlink_to(private)
    manager = MemoryManager(memory)
    assert manager.authorize_read(memory / 'MEMORY.md') == memory / 'MEMORY.md'
    for path in [private, memory / 'escaped.json']:
        with pytest.raises(PermissionError):
            manager.authorize_read(path)
    with pytest.raises(PermissionError):
        manager.authorize_write(memory / 'MEMORY.md')


def test_actual_agent_loop_rejects_false_success_then_stops(tmp_path, monkeypatch):
    import json
    from PIL import Image
    from pydantic_ai.models.function import DeltaToolCall, FunctionModel
    from rpent.planner.api_loop import ApiAgentLoop
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.utils.logging import init_output_dir

    monkeypatch.setenv('M1_CONTROL_PATH', str(tmp_path / 'private/control.json'))
    monkeypatch.setenv('M1_RECORD_DIR', str(tmp_path / 'activity'))
    monkeypatch.setenv('M1_AUTONOMOUS', '1')
    monkeypatch.setenv('PYDANTIC_AI_NO_BANNER', '1')
    (tmp_path / 'ipc').mkdir()
    image = tmp_path / 'frame.png'
    Image.new('RGB', (16, 16)).save(image)
    reply = {'observation_id': 'frame1', 'native_success': False, 'result': {'status': 'ready'}, 'sensors': {}, 'features': [], 'rgb_path': str(image), 'overlay_path': str(image)}
    (tmp_path / 'ipc/reply_0000.json').write_text(json.dumps(reply))
    init_output_dir(tmp_path / 'episode/rpent')
    toolkit = IsaacToolkit(tmp_path)
    count = 0

    async def stream(messages, info):
        nonlocal count
        count += 1
        if count == 1:
            yield {0: DeltaToolCall(name='finish', json_args=json.dumps({'status': 'success', 'summary': 'unsupported claim'}))}
        else:
            yield {0: DeltaToolCall(name='stop', json_args=json.dumps({'evidence': 'No simulated action executed', 'uncertainty': 'Offline wiring test', 'reason': 'test_complete'}))}

    try:
        result = ApiAgentLoop(model=FunctionModel(stream_function=stream), dashboard_events=NullDashboardEventSink(), timeout_s=30).solve(system_prompt='Offline protocol test', user_message='Exercise completion validation', toolkit=toolkit, max_turns=4)
        assert result.error is None
        assert count == 2
        assert result.finish_result['status'] == 'failure'
        assert control.snapshot()['terminal']['source'] == 'model_stop'
        assert not list((tmp_path / 'ipc').glob('command_*.json'))
    finally:
        toolkit.close()


def test_launcher_keeps_api_credentials_out_of_simulator_and_cleans_up(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace
    import launch
    assets = tmp_path / 'assets with spaces'
    names = ['parts/M6 Hub Bolt.usd', 'parts/Casing Top.usd', 'parts/Hub Cover Output.usd', 'vendor/remote_mirror/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd']
    for name in names:
        path = assets / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setenv('OPENAI_API_KEY', 'secret-for-offline-test')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://provider.example/v1')
    monkeypatch.setenv('OPENAI_MODEL', 'vision-model')
    monkeypatch.setenv('ASSEMBLY_ASSET_ROOT', str(assets))
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    (checkout / '.env').write_text('OPENAI_API_KEY=secret-for-offline-test\n')
    monkeypatch.setattr(launch.settings, 'ROOT', checkout)
    run = tmp_path / 'episode outside checkout'
    monkeypatch.setattr(sys, 'argv', ['launch.py', '--run-dir', str(run), '--rollout', '--gpu', '0'])
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launch.subprocess, 'run', execute)
    launch.main()
    docker = next(command for command in calls if command[:2] == ['docker', 'run'])
    assert 'OPENAI_API_KEY' not in str(docker)
    assert 'secret-for-offline-test' not in (run / 'configuration.json').read_text()
    assert f'{assets}:{assets}:ro' in docker
    assert f'{run}:{run}' in docker
    assert f'/dev/null:{checkout / ".env"}:ro' in docker
    assert any(command[:2] == ['docker', 'stop'] for command in calls)
    assert any(command[:2] == ['docker', 'rm'] for command in calls)
    assert json.loads((run / 'configuration.json').read_text())['rollout'] is True
