from types import SimpleNamespace

import pytest
import numpy as np

from camera_readiness import update_ready_camera
from camera_readiness import configure_full_frame_window


def test_full_frame_window_has_no_missing_overscan_bounds_or_pixel_crop():
    values={}
    configure_full_frame_window(SimpleNamespace(set_float=lambda key,value:values.update({key:value})))
    bounds=[values[f'/rtx/dataWindowNDC/{i}'] for i in range(4)]
    assert bounds==[0.,0.,1.,1.]
    assert bounds[2]-bounds[0]==1 and bounds[3]-bounds[1]==1


def camera(failures):
    namespace = {}
    exec(compile('def _resize_data_for_overscan():\n    return None - None\n', 'annotator_utils.py', 'exec'), namespace)
    calls = []

    def update(dt, force_recompute):
        calls.append((dt, force_recompute))
        if len(calls) <= failures:
            namespace['_resize_data_for_overscan']()

    return SimpleNamespace(update=update), calls


def test_retries_only_render_without_adding_simulation_or_camera_time():
    c, calls = camera(2)
    rendered = []
    update_ready_camera(SimpleNamespace(render=lambda: rendered.append(True)), c, .2)
    assert calls == [(.2, True), (0, True), (0, True)]
    assert len(rendered) == 2


def test_persistent_metadata_failure_still_raises():
    c, calls = camera(99)
    with pytest.raises(TypeError):
        update_ready_camera(SimpleNamespace(render=lambda: None), c, .2)
    assert len(calls) == 8


def test_unrelated_type_error_is_not_retried():
    def update(*args, **kwargs):
        raise TypeError('other bug')
    def render():
        pytest.fail('unrelated error must not render/retry')
    with pytest.raises(TypeError, match='other bug'):
        update_ready_camera(SimpleNamespace(render=render), SimpleNamespace(update=update), 0)


@pytest.mark.parametrize('failures', [2, 99])
def test_empty_native_depth_annotator_is_bounded_render_only_retry(failures):
    namespace = {}
    exec(compile("def _process_annotator_output():\n    raise RuntimeError(\"shape '[720, 960, 1]' is invalid for input of size 0\")\n",
                 '/workspace/isaaclab/isaaclab/sensors/camera/camera.py', 'exec'), namespace)
    calls = []; renders = []
    def update(dt, force_recompute):
        calls.append(dt)
        if len(calls) <= failures:
            namespace['_process_annotator_output']()
    sim = SimpleNamespace(render=lambda: renders.append(True))
    sensor = SimpleNamespace(update=update)
    if failures > 8:
        with pytest.raises(RuntimeError):
            update_ready_camera(sim, sensor, .2)
        assert len(calls) == 8
    else:
        update_ready_camera(sim, sensor, .2)
        assert calls == [.2, 0, 0]
    assert len(renders) == len(calls) - 1


def test_unrelated_empty_shape_error_is_not_retried():
    def update(*args, **kwargs):
        raise RuntimeError('invalid for input of size 0')
    with pytest.raises(RuntimeError):
        update_ready_camera(SimpleNamespace(render=lambda: pytest.fail('unexpected retry')),
                            SimpleNamespace(update=update), 0)


@pytest.mark.parametrize('failures',[2,99])
def test_empty_overscan_buffer_is_render_only_bounded_retry(failures):
    namespace={'np':np}
    exec(compile('def _resize_data_for_overscan():\n data=np.empty((0,))\n return data.shape[1]\n','annotator_utils.py','exec'),namespace)
    calls=[];renders=[]
    def update(dt,force_recompute):
        calls.append(dt)
        if len(calls)<=failures:namespace['_resize_data_for_overscan']()
    sim=SimpleNamespace(render=lambda:renders.append(True))
    sensor=SimpleNamespace(update=update)
    if failures==99:
        with pytest.raises(IndexError):update_ready_camera(sim,sensor,.2)
        assert len(calls)==8
    else:
        update_ready_camera(sim,sensor,.2)
        assert calls==[.2,0,0]
    assert len(renders)==len(calls)-1


def test_nonempty_malformed_overscan_buffer_still_raises():
    namespace={'np':np}
    exec(compile('def _resize_data_for_overscan():\n data=np.zeros(4)\n return data.shape[1]\n','annotator_utils.py','exec'),namespace)
    with pytest.raises(IndexError) as caught:
        update_ready_camera(SimpleNamespace(render=lambda:pytest.fail('must not hide malformed data')),
            SimpleNamespace(update=lambda *args,**kwargs:namespace['_resize_data_for_overscan']()),0)
    assert '(4,)' in caught.value.__notes__[0]
