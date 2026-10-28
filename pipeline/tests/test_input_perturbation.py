import numpy as np
import pytest
from input_perturbation import InputPerturbation


def test_clean_input_is_unchanged_and_never_modifies_native_arrays():
    source=np.array([[1.,np.nan],[0.,2.]])
    snapshot=source.copy()
    noise=InputPerturbation({})
    assert np.array_equal(noise.depth(source,'f1'),source,equal_nan=True)
    assert np.array_equal(source,snapshot,equal_nan=True)
    wrench=np.arange(6,dtype=float)
    assert np.array_equal(noise.wrench(wrench),wrench)


def test_noise_is_seeded_and_reused_for_one_acquisition():
    cfg={'seed':42,'depth_std_m':.001,'depth_dropout_probability':.1,'force_std_n':.1,'torque_std_nm':.005}
    a,b=InputPerturbation(cfg),InputPerturbation(cfg)
    source=np.ones((30,30))
    first=a.depth(source,'f1')
    assert np.array_equal(first,b.depth(source,'f1'),equal_nan=True)
    assert np.array_equal(first,a.depth(source,'f1'),equal_nan=True)
    assert not np.array_equal(first,a.depth(source,'f2'),equal_nan=True)
    assert np.isnan(first).any()
    assert np.all(source==1)


def test_invalid_noise_cannot_enter_measurements():
    with pytest.raises(ValueError):InputPerturbation({'depth_dropout_probability':2})
    with pytest.raises(ValueError):InputPerturbation({'force_std_n':float('nan')})
