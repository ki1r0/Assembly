# Development

Use a Python 3.11/3.12 host virtual environment and install `requirements-dev.txt`. Run `python -m pytest -q` for offline boundary, RPC and geometry tests. `python launch.py --help` must work without Isaac installed on the host. The simulator runs in its own container.

Keep model inference separate from physical action adaptation. Preserve public/private data boundaries, independent native success and native motion emergency stops. Do not put credentials, trajectories, evaluator poses or scene-specific answer coordinates into prompts or memory. Runtime artifacts belong under ignored `runs/`; source changes apply to future episodes and do not require a whole project copy per rollout.

For simulator changes run the bounded integration smoke test with your own assets/API configuration, then record the actual commands, result and limitations. Offline passing tests do not establish simulator rollout success. Do not distribute CAD/vendor assets or application code without resolving its license.
