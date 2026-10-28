import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize('value', ['nan', '99', '1001'])
def test_invalid_diagnostic_gains_are_rejected_before_simulation(value):
    launch = Path(__file__).resolve().parents[1] / 'launch.py'
    result = subprocess.run([sys.executable, str(launch), '--osc-diagnostic-rotation-stiffness', value],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert 'stiffness must be within' in result.stderr


def test_diagnostic_gain_cannot_silently_change_a_policy_rollout():
    launch = Path(__file__).resolve().parents[1] / 'launch.py'
    result = subprocess.run([sys.executable, str(launch), '--controller', 'isaac_osc', '--rollout',
                             '--osc-diagnostic-rotation-stiffness', '1000'], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'only available in adapter diagnostics' in result.stderr
