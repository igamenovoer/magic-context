---
name: pixi-install-nvidia
description: Guidance for installing NVIDIA, CUDA, and GPU-accelerated packages (PyTorch, TensorFlow, RAPIDS) using Pixi. Use when a user needs to set up a GPU-enabled environment or resolve CUDA dependency issues in a Pixi project.
---

# Pixi Install NVIDIA

## Overview

This skill provides optimized workflows for setting up GPU-accelerated environments using Pixi. It handles channel configuration, dependency resolution for CUDA-enabled packages, and environment setup for libraries like PyTorch, TensorFlow, and RAPIDS.

## Workflow

### 1. Channel Configuration
Ensure the correct channels are present in `pyproject.toml`. The order determines priority.
*   **Command**: `pixi project channel add nvidia` (and `pytorch` if needed).
*   **Priority**: `nvidia` and `pytorch` should generally be prioritized over `conda-forge` for GPU builds.

### 2. Adding GPU Packages
Use specific versions to ensure compatibility between CUDA and the framework.

#### PyTorch (Recommended)
```bash
pixi add pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

#### CUDA Toolkit Components
Instead of the massive `cuda-toolkit`, consider adding only what's needed:
```bash
pixi add cuda-compiler cuda-libraries-dev
```

### 3. Verification
After installation, verify GPU visibility:
```bash
pixi run python -c "import torch; print(torch.cuda.is_available())"
```

## Troubleshooting & References

*   **CUDA Version Mismatch**: Check `nvidia-smi` on the host to ensure the installed `pytorch-cuda` version is supported by the host driver.
*   **Library Loading Issues**: If `libcuda.so` or `libcudart.so` are not found, ensure the environment is activated (`pixi shell`).
*   **Detailed Package List**: See [nvidia-packages.md](references/nvidia-packages.md) for a comprehensive list of available NVIDIA and GPU libraries.

## Resources

### references/
*   **[nvidia-packages.md](references/nvidia-packages.md)**: Comprehensive guide on channels, packages, and versioning for NVIDIA ecosystems.
