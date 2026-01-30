# NVIDIA and CUDA Packages for Pixi

## Replacing System Packages (APT/Sudo) with Pixi

The primary advantage of using Pixi for NVIDIA development is **User-Space Management**. You do not need `sudo` or root access to install CUDA toolkits, compilers, or libraries. This ensures:
*   **Reproducibility**: Everyone uses the exact same version, regardless of the host OS.
*   **Isolation**: Project dependencies don't conflict with system drivers or other projects.
*   **No Root Required**: Safe to run on shared HPC clusters.

### Migration Guide: APT vs. Pixi

| Category | System (Sudo APT) | Pixi (User Space) | Notes |
| :--- | :--- | :--- | :--- |
| **Compiler** | `sudo apt install nvidia-cuda-toolkit` | `pixi add cuda-nvcc=12.1` | Explicitly pin versions like `=12.1`. |
| **Libraries** | `sudo apt install libcudnn8-dev` | `pixi add cudnn=8.9` | Ensure channel `nvidia` is active. |
| **Multi-GPU** | `sudo apt install libnccl2` | `pixi add nccl=2.18` | |
| **Profiling** | `sudo apt install nsight-compute` | `pixi add nsight-compute` | |
| **Full Toolkit**| `sudo apt install cuda` | `pixi add cuda-toolkit=12.1` | Prefer smaller components in Pixi. |
| **Drivers** | `sudo apt install nvidia-driver-535` | *(System Only)* | Drivers **must** be on the host. |

---

## CUDA Toolkit and Compilers (Conda-Forge & NVIDIA Channels)

When using the `nvidia` channel or `conda-forge`, you can install specific components of the CUDA toolkit.

> **Tip:** Always pin your CUDA version (e.g., `cuda-toolkit=12.1`) to ensure compatibility with your host driver and other libraries.

### Core Components
*   **`cuda-toolkit`**: The full toolkit (all libraries, compilers, tools).
    *   **Note**: While available on PyPI as `cuda-toolkit`, in Pixi it is **strongly recommended** to use the Conda package from the `nvidia` channel for better environment integration:
        ```bash
        pixi add cuda-toolkit -c nvidia
        ```
*   **`cuda-compiler`**: NVCC and other compiler tools.
*   **`cuda-libraries`**: Runtime libraries (libcudart, etc.).
*   **`cuda-libraries-dev`**: Development headers and static libraries.
*   **`cuda-nvcc`**: Specifically the NVCC compiler.
*   **`cuda-cudart`**: CUDA Runtime API.
*   **`cuda-cupti`**: CUDA Profiling Tools Interface.
*   **`cuda-nvtx`**: NVIDIA Tools Extension.

### Libraries
*   **`libcublas`** / **`libcublas-dev`**: GPU-accelerated BLAS.
*   **`libcufft`** / **`libcufft-dev`**: FFT libraries.
*   **`libcurand`** / **`libcurand-dev`**: Random number generation.
*   **`libcusolver`** / **`libcusolver-dev`**: Linear algebra solver.
*   **`libcusparse`** / **`libcusparse-dev`**: Sparse matrix library.

## RAPIDS (NVIDIA Accelerated Data Science)

RAPIDS libraries provide GPU-accelerated alternatives to standard data science tools.

*   **`cudf`**: GPU DataFrame library (pandas-like).
*   **`cuml`**: GPU Machine Learning algorithms (scikit-learn-like).
*   **`cugraph`**: GPU Graph Analytics (NetworkX-like).
*   **`cuspatial`**: GPU Spatial Analysis.
*   **`cuxfilter`**: GPU accelerated cross-filtering.

Installation via Pixi:
```bash
# Example: Install RAPIDS 24.02
pixi add cudf=24.02 cuml=24.02 -c rapidsai -c conda-forge -c nvidia
```

## PyPI-specific NVIDIA Packages

NVIDIA publishes many packages to PyPI, typically with modular naming. These are useful when you cannot use Conda packages or need specific versions.

### Mapping PyPI to APT

Use this table to translate system-level `apt` requirements into their user-space `pip`/`pixi` equivalents.

| Component | System (Sudo APT) | PyPI (Pixi/Pip) | Notes |
| :--- | :--- | :--- | :--- |
| **Full Toolkit** | `cuda` | `cuda-toolkit` | Monolithic installer. Prefer Conda. |
| **CUDA Runtime** | `libcudart12` | `nvidia-cuda-runtime-cu12==12.1.105` | Core runtime libraries. |
| **Compiler (NVCC)**| `nvidia-cuda-toolkit` | `nvidia-cuda-nvcc-cu12==12.1.105` | Just the compiler, not the full toolkit. |
| **cuDNN** | `libcudnn8` | `nvidia-cudnn-cu12==8.9.2.26` | Deep learning primitives. |
| **cuBLAS** | `libcublas-12-0` | `nvidia-cublas-cu12==12.1.3.1` | GPU BLAS libraries. |
| **cuRAND** | `libcurand-12-0` | `nvidia-curand-cu12==10.3.2.106` | Random number generation. |
| **cuSOLVER** | `libcusolver-12-0` | `nvidia-cusolver-cu12==11.4.5.107` | Linear algebra solver. |
| **cuSPARSE** | `libcusparse-12-0` | `nvidia-cusparse-cu12==12.1.0.106` | Sparse matrix library. |
| **NCCL** | `libnccl2` | `nvidia-nccl-cu12==2.18.1` | Multi-GPU communication. |
| **NVML (mgmt)** | `libnvidia-ml1` | `nvidia-ml-py` | Python bindings for GPU management. |

### CUDA Runtime & Math Libraries
*   **`nvidia-cuda-runtime-cu12`**: CUDA Runtime.
*   **`nvidia-cuda-nvcc-cu12`**: NVCC Compiler.
*   **`nvidia-cublas-cu12`**: cuBLAS.
*   **`nvidia-cudnn-cu12`**: cuDNN (Deep Neural Network library).
*   **`nvidia-curand-cu12`**: cuRAND.
*   **`nvidia-cusolver-cu12`**: cuSOLVER.
*   **`nvidia-cusparse-cu12`**: cuSPARSE.
*   **`nvidia-nccl-cu12`**: NCCL (Multi-GPU communication).

### Management & Tools
*   **`nvidia-ml-py`**: Python bindings for NVML (NVIDIA Management Library) - useful for monitoring GPU status programmatically.

## Channel Management

For the best compatibility with NVIDIA packages, prioritize the `nvidia` channel or `rapidsai`.

Recommended channel order in `pyproject.toml` or `pixi.toml`:
1.  `nvidia` (Official NVIDIA packages)
2.  `rapidsai` (If using RAPIDS)
3.  `conda-forge` (Community maintained, often has good CUDA repackaging)

## Environment Variables

When working with these packages, ensure your environment variables are set correctly if the tools don't auto-detect paths.

*   **`CUDA_HOME`**: Often required by source builds.
*   **`LD_LIBRARY_PATH`**: May need to include paths to `lib` directories of installed packages if using system-level linking.