#!/usr/bin/env bash
# Export locally licensed USD geometry; never download or publish assembly CAD.
set -euo pipefail
: "${ASSEMBLY_ASSET_ROOT:?Set ASSEMBLY_ASSET_ROOT to your licensed assembly assets}"
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
pipeline_dir="$repo_dir/pipeline"
asset_dir="$(cd -- "$ASSEMBLY_ASSET_ROOT" && pwd)"
docker run --rm --entrypoint /bin/bash \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES \
    -e ASSEMBLY_ASSET_ROOT="$asset_dir" -e ASSEMBLY_CAD_SCALE_M_PER_UNIT=0.001 \
    -v "$pipeline_dir:$pipeline_dir" -v "$asset_dir:$asset_dir:ro" \
    "${ISAAC_DOCKER_IMAGE:-nvcr.io/nvidia/isaac-lab:2.3.0}" \
    -c 'exec /isaac-sim/python.sh "$1/tools/export_public_cad.py" --output "$1/assets/public_cad_models"' _ "$pipeline_dir"
