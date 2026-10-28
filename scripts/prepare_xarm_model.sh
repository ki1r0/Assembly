#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
pipeline_dir="$repo_dir/pipeline"
source_dir="$pipeline_dir/third_party/xarm_ros2"
if [[ ! -d "$source_dir" ]]; then
    git clone https://github.com/xArm-Developer/xarm_ros2.git "$source_dir"
    git -C "$source_dir" checkout ac736c92b02de7ca7e0be6f737d77d46707464d8
fi
"$pipeline_dir/.venv/bin/python" "$pipeline_dir/tools/prepare_xarm7_urdf.py" --source "$source_dir" --gripper ufactory-g1 --output "$pipeline_dir/assets/xarm7/xarm7_g1_simulation_assumption.urdf"
docker run --rm --gpus "device=${GPU:-0}" --entrypoint /bin/bash \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES \
    -v "$pipeline_dir:$pipeline_dir" \
    "${ISAAC_DOCKER_IMAGE:-nvcr.io/nvidia/isaac-lab:2.3.0}" \
    -c 'exec /isaac-sim/python.sh "$1/tools/convert_xarm7_isaac.py" --urdf "$1/assets/xarm7/xarm7_g1_simulation_assumption.urdf" --output-dir "$1/assets/xarm7/usd" --headless' _ "$pipeline_dir"
