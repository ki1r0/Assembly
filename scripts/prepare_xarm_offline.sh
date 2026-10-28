#!/usr/bin/env bash
# Public URDF/mesh preparation; no simulator or hardware connection.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$repo_dir/pipeline/third_party/xarm_ros2"
if [[ ! -d "$source_dir" ]]; then
    git clone https://github.com/xArm-Developer/xarm_ros2.git "$source_dir"
    git -C "$source_dir" checkout ac736c92b02de7ca7e0be6f737d77d46707464d8
fi
"${PYTHON:-python}" "$repo_dir/pipeline/tools/prepare_xarm7_urdf.py" \
    --source "$source_dir" --gripper ufactory-g1 \
    --output "$repo_dir/pipeline/assets/xarm7/xarm7_g1_simulation_assumption.urdf"
