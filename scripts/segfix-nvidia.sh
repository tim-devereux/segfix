#!/usr/bin/env bash
# Launch segfix with rendering offloaded to a discrete NVIDIA GPU (Linux
# PRIME/Optimus laptops). Activate the environment segfix is installed in
# before running this (e.g. `conda activate segfix`).
set -euo pipefail

export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia

exec segfix "$@"
