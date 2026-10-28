"""Reject experimental drive gains outside their diagnostic scope before startup."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('arguments,message', [
    (['--robot', 'xarm7', '--rollout', '--xarm-diagnostic-reference-speed-scale', '2'], 'adapter diagnostics only'),
    (['--robot', 'xarm7', '--adapter-validation', '--xarm-diagnostic-reference-speed-scale', 'nan'], 'Reference speed scale must be'),
    (['--robot', 'xarm7', '--rollout', '--xarm-diagnostic-arm-stiffness', '800'], 'only available in adapter diagnostics'),
    (['--robot', 'xarm7', '--rollout', '--xarm-physics-hz', '120'], 'diagnostic-only'),
    (['--robot', 'panda', '--adapter-validation', '--xarm-diagnostic-arm-stiffness', '800'], 'only available in adapter diagnostics'),
    (['--robot', 'xarm7', '--adapter-validation', '--xarm-diagnostic-arm-stiffness', 'nan'], 'stiffness must be'),
    (['--robot', 'xarm7', '--adapter-validation', '--xarm-diagnostic-arm-damping', '10'], 'stiffness must be'),
])
def test_reject_invalid_gain_scope_before_container_start(arguments, message):
    result = subprocess.run([sys.executable, str(ROOT/'launch.py'), *arguments],
                            cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 2
    assert message in result.stderr
