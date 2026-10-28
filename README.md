# Current xArm package (2026-10-28)

The current simulation xArm pipeline is in `pipeline/`, with an independent peg benchmark in `benchmark/`. Start with [the deployment guide](docs/SETUP_CN.md) and [hardware integration boundaries](docs/REAL_ROBOT_MIGRATION_CN.md). Real hardware drivers and calibration are not yet validated. The original top-level code below is retained as the historical assembly-agent-core.

# Assembly Agent Core

GLM/OpenAI-compatible vision-language planning through RPent, with RGB-D geometry, reusable grasp/insertion skills, and HRC Cartesian servo control in Isaac Lab. [中文说明](README_CN.md)

This source checkout retains RPent's model-independent VLA client/backend interfaces. No learned-policy weights, model-specific inference server, normalized LIBERO action mapping, credentials, experiment trajectories, or CAD assets are bundled. The default execution path is the existing geometry/servo controller, not a trained VLA.

## Setup

For the complete tested procedure, expected artifacts, provider model IDs and troubleshooting, see [the reproduction guide](docs/REPRODUCE_CN.md).

Use Linux, Python 3.11 or 3.12 for the host planner, NVIDIA Container Toolkit, Docker, and an RTX GPU compatible with Isaac Sim 5.1. The simulation image is `nvcr.io/nvidia/isaac-lab:2.3.0`; pulling/running it requires accepting NVIDIA's terms. Python dependencies below are for the host, not the container.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
# Edit .env with your endpoint, model identifier, key and asset directory.
set -a
source .env
set +a
python launch.py --help
python -m pytest -q
python launch.py --gpu 0 --budget-s 120
# Full assembly episode:
python launch.py --gpu 0 --rollout --budget-s 600
```

The default launch is an integration smoke test: observe, read the empty memory index, move 5 mm, open the gripper, and stop. `--rollout` selects the assembly task. Other `--case` choices are `bolt-position-shift`, `bolt-horizontal-pose`, and `assembly-position-shift`. `--run-dir` must identify a fresh episode directory. Camera resolution can be configured through the environment; lowering it can reduce perception accuracy.

## Source map

| Path | Responsibility |
| --- | --- |
| `planner.py`, `SYSTEM.md` | Model configuration, task and RPent agent loop |
| `toolkit.py` | Public observations, actor tools, IPC |
| `visual_servo.py` | RGB-D grasp, transport and insertion feedback skills |
| `aperture_measurement.py`, `cylinder_fit.py`, `inhand_measurement.py` | Image/depth measurements |
| `backend.py`, `wrist_camera.py` | Simulator process, observed state and camera capture |
| `isaac_runtime/` | HRC execution, scene, contacts and private native evaluator |
| `m1_reach/` | Existing episode decision/budget protocol and API recording |
| `third_party/rpent/` | Locally adapted RPent core and generic Env/VLA/RPC interfaces |
| `config/` | Scene variants and private fixture/evaluator geometry |

See [architecture and data boundaries](docs/ARCHITECTURE.md), [external assets](docs/ASSETS.md), [VLA extension](docs/VLA_INTERFACE.md), [known limitations](docs/LIMITATIONS.md), and [provenance/licensing](THIRD_PARTY_NOTICES.md).

## Outputs and scope

Real setup/rollout evidence is recorded in [the validation report](docs/SETUP_ROLLOUT_REPORT_CN.md).

Each episode creates `runs/<timestamp>/`: configuration, public RGB-D artifacts, tool traces, recorded API request/response bodies, video, and private backend/evaluator diagnostics. The launcher stops and removes only its own container. API credentials stay in the host planner environment; the launcher masks the root `.env` file inside the simulator container. Keep other credential files outside source and asset directories. Logs can contain images and conversation content even though authentication headers are not recorded; review them before sharing. Runtime outputs and `.env` are ignored by Git.

Independent native task success is the completion criterion. Process exit, model text, closed fingers and seating candidates do not establish success. Packaging/import tests are not evidence that a fresh assembly rollout succeeds. Earlier final four geometry/servo episodes did not meet native success; this repository does not claim benchmark completion.

The GitHub destination is [ki1r0/Assembly](https://github.com/ki1r0/Assembly). Runtime artifacts and local credentials are excluded from source control. RPent is Apache-2.0; the inherited HRC/application code has no supplied redistribution license. Resolve its license before public distribution, and do not upload third-party CAD/vendor assets without permission.
