# Compiling CUDA Code with Pixi

This guide explains how to compile `.cu` (CUDA C++) files using the CUDA toolkit managed by Pixi. This approach ensures you are using a consistent, user-space compiler that is independent of the system's `/usr/local/cuda` installation.

## 1. Prerequisites

You need the CUDA compiler (`nvcc`) and potentially the runtime libraries.

### Minimal Setup (Compiler Only)
If you just need `nvcc`:
```bash
pixi add cuda-nvcc -c nvidia
```

### Full Toolkit Setup
For complex projects requiring all libraries and tools:
```bash
pixi add cuda-toolkit -c nvidia
```
*Note: Always use the `nvidia` channel for the most reliable toolchain.*

## 2. Environment Activation

When you run `pixi shell` or `pixi run`, Pixi automatically configures the environment variables:
*   **`PATH`**: Updated to include the `bin` directory where `nvcc` resides.
*   **`LD_LIBRARY_PATH`**: Updated to include the `lib` directory for runtime libraries.
*   **`CUDA_HOME`** / **`CUDA_PATH`**: *Note: These are NOT always automatically set by Conda packages. You may need to set them explicitly if your build script relies on them.*

To find the location of your Pixi-managed CUDA installation:
```bash
pixi run which nvcc
# Output example: /path/to/project/.pixi/envs/default/bin/nvcc
```

## 3. Basic Compilation with NVCC

Create a simple `hello.cu` file:

```cpp
#include <stdio.h>

__global__ void helloCUDA() {
    printf("Hello form GPU thread %d!\n", threadIdx.x);
}

int main() {
    helloCUDA<<<1, 5>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

Compile it using `pixi run`:

```bash
pixi run nvcc hello.cu -o hello
```

Run the binary:
```bash
pixi run ./hello
```

## 4. Using CMake

If your project uses CMake, you need to ensure it finds the Pixi-managed CUDA compiler, not the system one.

### CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.18)
project(MyCudaProject LANGUAGES CXX CUDA)

add_executable(hello hello.cu)
```

### Building with Pixi

For the most robust configuration, explicitly point CMake to your Pixi environment's compiler and toolkit root.

```bash
# Recommended: Explicitly set the compiler path using the Pixi environment variable
pixi run cmake -S . -B build \
    -DCMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc \
    -DCUDAToolkit_ROOT=$CONDA_PREFIX

pixi run cmake --build build
```

*   **`CMAKE_CUDA_COMPILER`**: Tells CMake exactly which `nvcc` executable to use.
*   **`CUDAToolkit_ROOT`**: Helps `FindCUDAToolkit` locate libraries and headers within the Conda environment.

## 5. Troubleshooting

### "nvcc: command not found"
*   Ensure you have added `cuda-nvcc` or `cuda-toolkit` to your `pixi.toml`.
*   Ensure you are running inside `pixi shell` or using `pixi run`.

### "fatal error: cuda_runtime.h: No such file or directory"
*   This header comes with the toolkit. Ensure your installation is complete.
*   If using a minimal setup, you might need `cuda-libraries-dev` or `cuda-cudart-dev`.
    ```bash
    pixi add cuda-libraries-dev -c nvidia
    ```

### Linker Errors (`-lcudart` not found)
*   Ensure `LD_LIBRARY_PATH` is correct.
*   In some setups, you might need to manually link the library path in your compilation command:
    ```bash
    nvcc hello.cu -o hello -L$CONDA_PREFIX/lib -lcudart
    ```

## 6. References

*   **[NVIDIA NVCC Documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)**: Official guide for the CUDA compiler driver.
*   **[CMake FindCUDAToolkit](https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html)**: Documentation for CMake's CUDA toolkit detection module.
*   **[NVIDIA Conda Channel](https://anaconda.org/nvidia)**: Official source for the `cuda-toolkit` and related packages.
*   **[CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)**: Broad system requirements and environment variable references (useful for understanding `LD_LIBRARY_PATH` and `CUDA_HOME`).
