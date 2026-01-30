#!/bin/bash
set -e

# Location of the Pixi environment (assumes script is run inside 'pixi run')
if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: This script must be run within a Pixi environment (pixi run ...)"
    exit 1
fi

# Ensure we are in the script's directory (the source root for this check)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[BuildCheck] Using Environment: $CONDA_PREFIX"
echo "[BuildCheck] Working Directory: $PWD"
echo "[BuildCheck] NVCC: $(which nvcc)"

# Explicitly set the CUDA compiler to the one in the environment
export CUDACXX=$CONDA_PREFIX/bin/nvcc

# Configure
cmake -G Ninja -S . -B build \
    -DCMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc \
    -DCUDAToolkit_ROOT=$CONDA_PREFIX

# Build
cmake --build build

# Run
echo "[BuildCheck] Executing..."
./build/check_app