# NVIDIA TensorRT Docker SSH Environment Setup

## HEADER
- **Purpose**: Guide for resolving SSH environment variable issues in NVIDIA NGC containers
- **Status**: Active
- **Date**: 2025-10-14
- **Dependencies**: Docker, NVIDIA NGC containers, SSH server
- **Target**: Developers working with NVIDIA TensorRT in containerized environments

## Problem

When using NVIDIA NGC containers (TensorRT, Triton Server, etc.) with SSH access, Python imports that work via `docker exec` fail via SSH due to missing environment variables, particularly `LD_LIBRARY_PATH`.

## Root Cause

Docker ENV variables are inherited by `docker exec` but not by SSH login shells. SSH creates fresh login shells that only read standard shell initialization files, not Docker's ENV variables.

## Quick Solution

Add missing environment variables to `/etc/shinit_v2`:

```bash
docker exec <container> bash -c 'cat >> /etc/shinit_v2 << "EOF"

# NVIDIA library paths for TensorRT and CUDA
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/usr/local/tensorrt/lib/:/opt/tritonserver/backends/tensorrtllm:/usr/local/tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64}
export LIBRARY_PATH=${LIBRARY_PATH:-/usr/local/cuda/lib64/stubs:}
EOF
'
```

## Best Practice for Dockerfiles

```dockerfile
FROM nvcr.io/nvidia/tritonserver:25.08-py3

# Ensure BASH_ENV is preserved
ENV BASH_ENV=/etc/bash.bashrc

# Add NVIDIA library paths to shinit_v2 for SSH compatibility
RUN echo '' >> /etc/shinit_v2 && \
    echo '# Custom: Ensure LD_LIBRARY_PATH is set for SSH sessions' >> /etc/shinit_v2 && \
    echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/usr/local/tensorrt/lib/:/opt/tritonserver/backends/tensorrtllm:/usr/local/tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64}' >> /etc/shinit_v2
```

## Testing

```bash
# Test docker exec
docker exec <container> python3 -c "import tensorrt"

# Test non-interactive SSH
ssh -p <port> root@<host> 'python3 -c "import tensorrt"'

# Test interactive SSH
ssh -p <port> root@<host>
python3 -c "import tensorrt"
```

## References

- Full analysis: `/workspace/code/model-inference-perf/about-nvidia-trt-docker-bash-env.md`
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
