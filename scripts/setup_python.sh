#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
"${PYTHON:-python3.11}" -m venv "$repo_dir/pipeline/.venv"
"$repo_dir/pipeline/.venv/bin/python" -m pip install -r "$repo_dir/pipeline/requirements-dev.txt" xacro==2.1.1
