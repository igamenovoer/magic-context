# NVIDIA and CUDA Packages for Pixi

## Common Packages

### CUDA Toolkit and Compilers
*   **`cuda-toolkit`**: The full toolkit (conda-forge). Often better to install specific components.
*   **`cuda-compiler`**: NVCC and other compiler tools.
*   **`cuda-libraries`**: Runtime libraries (libcudart, etc.).
*   **`cuda-libraries-dev`**: Development headers and static libraries.
*   **`cuda-nvcc`**: Specifically the NVCC compiler.

### Deep Learning Frameworks
*   **`pytorch`**:
    *   Conda: `pixi add pytorch torchvision torchaudio -c pytorch -c nvidia` (standard way).
    *   Conda-Forge: `pixi add pytorch` (often has `cpu` and `cuda` variants).
*   **`tensorflow`**: `pixi add tensorflow`.

### GPU Accelerated Libraries
*   **`cupy`**: NumPy-like API for NVIDIA GPUs. `pixi add cupy`.
*   **`cuml` / `cudf` / `cugraph` (RAPIDS)**: Accelerated data science. `pixi add cudf cuml -c rapidsai -c conda-forge -c nvidia`.

## PyPI-specific NVIDIA Packages

When using `pypi-dependencies` in Pixi, NVIDIA provides modular packages (often with `-cu11` or `-cu12` suffixes).

### Common PyPI Packages
*   **`nvidia-cuda-runtime-cu12`**: CUDA Runtime.
*   **`nvidia-cudnn-cu12`**: cuDNN.
*   **`nvidia-cublas-cu12`**: cuBLAS.
*   **`nvidia-curand-cu12`**: cuRAND.

### PyTorch via PyPI
To use PyTorch from PyPI with a specific CUDA version:
```bash
pixi add --pypi torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
```

## Channel Management
Order matters in `pyproject.toml` or `pixi.toml`:
1.  `nvidia`
2.  `pytorch` (if using)
3.  `rapidsai` (if using)
4.  `conda-forge`

## Environment Variables
Some packages need hints about where CUDA is located if not automatically detected:
*   `CUDA_HOME`
*   `LD_LIBRARY_PATH`

In Pixi, you can set these in `[tool.pixi.tasks]` or rely on Pixi's activation scripts which usually handle this for conda-installed CUDA.

## Example: Minimal GPU PyTorch Setup
```toml
[tool.pixi.project]
channels = ["pytorch", "nvidia", "conda-forge"]
platforms = ["linux-64"]

[tool.pixi.dependencies]
python = "3.11.*"
pytorch = "*"
torchvision = "*"
pytorch-cuda = "11.8.*" # Or 12.1.*
```
