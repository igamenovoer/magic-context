---
name: pixi-make-cu-build-env
description: Guides the agent to setup a new or existing Pixi environment for compiling C++ and CUDA code. It ensures the correct compilers, toolkits, and CMake configurations are in place for a robust user-space build.
---

# Pixi Make CUDA Build Env

## Trigger

Use this skill when the user asks to:
- "Setup a CUDA build environment in this project"
- "Prepare the current Pixi project for compiling .cu files"
- "Add CUDA compilation support to <project_path>"

## Workflow

### 1. Requirements Gathering
**Mandatory**: You MUST identify:
1.  **Project Context**: The existing Pixi project path. (Default to current working directory if valid).
2.  **Manifest File**: Identify if the project uses `pixi.toml` or `pyproject.toml`.
3.  **Target Environment Name**: "Which Pixi environment should this be set up in?"
    *   *Default*: If not specified, assume the `default` environment.
    *   *Named*: If the user provides a name (e.g., `cuda-12`), use that feature.
4.  **CUDA Version**: "Which CUDA version do you need?" (e.g., 11.8, 12.1).
5.  **Host Tools**: Check if `cmake`, `ninja`, and a C++ compiler are available and suitable.
    *   *Action*: Run `cmake --version`, `ninja --version`, and check for `gcc`/`clang` (Linux) or `cl` (Windows).
    *   *Decision*: If missing or old, include them in the Pixi environment.

**Note**: Do not ask for extra libraries or tools. Only install `cuda-toolkit` unless the user explicitly requests extras (like `cudnn`, `nsight`).

### 2. Adding Dependencies
Add the core build tools and the CUDA toolchain to the **existing project**. Use the `nvidia` channel and **explicitly pin** the requested version.

**Command Logic**:
*   **Default Environment**: Do NOT use `--feature`.
*   **Named Environment**: Use `--feature <ENV_NAME>` (this will create the feature/environment if it doesn't exist).

```bash
# Core Build Tools (Install if missing/unsuitable on host)
pixi add --manifest-path <PROJECT_PATH>/<MANIFEST_FILE> [--feature <ENV_NAME>] cmake ninja cxx-compiler make pkg-config

# CUDA Toolchain (Standard libs + Compiler)
pixi project channel add nvidia --manifest-path <PROJECT_PATH>/<MANIFEST_FILE>
pixi add --manifest-path <PROJECT_PATH>/<MANIFEST_FILE> [--feature <ENV_NAME>] cuda-toolkit=<VERSION> cuda-nvcc=<VERSION> -c nvidia
```

*Optional Extras (Only if requested):*
```bash
pixi add --manifest-path <PROJECT_PATH>/<MANIFEST_FILE> [--feature <ENV_NAME>] cudnn=<VERSION> nccl=<VERSION> -c nvidia
```

*Optional Tools (Only if requested):*
```bash
pixi add --manifest-path <PROJECT_PATH>/<MANIFEST_FILE> [--feature <ENV_NAME>] nsight-compute -c nvidia
```

### 3. Configuring Build Tasks
Add a standard CMake build task to `<MANIFEST_FILE>` that properly points to the user-space compilers. This avoids conflicts with system-installed CUDA.

**Task Definition:**
```toml
[tool.pixi.tasks]
configure = { cmd = "cmake -G Ninja -S . -B build -DCMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc -DCUDAToolkit_ROOT=$CONDA_PREFIX", env = { CUDACXX = "$CONDA_PREFIX/bin/nvcc" } }
build = "cmake --build build"
test = "ctest --test-dir build"
```

### 4. Verification (Automated)
To verify the setup, create a self-contained test in `<project-dir>/build-check`.

**1. Create Directory Structure:**
```bash
mkdir -p build-check
```

**2. Create Source Files:**
*   `build-check/hello.cu`:
    ```cpp
    #include <stdio.h>
    __global__ void cuda_hello(){ printf("Hello from GPU!\n"); }
    void run_cuda_hello() { cuda_hello<<<1,1>>>(); cudaDeviceSynchronize(); }
    ```
*   `build-check/hello.cpp`:
    ```cpp
    #include <iostream>
    void run_cuda_hello();
    int main() { std::cout << "Hello from C++!" << std::endl; run_cuda_hello(); return 0; }
    ```
*   `build-check/CMakeLists.txt`:
    ```cmake
    cmake_minimum_required(VERSION 3.18)
    project(BuildCheck LANGUAGES CXX CUDA)
    add_executable(check_app hello.cpp hello.cu)
    set_target_properties(check_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    ```

**3. Create Build Script (`build-check/build-and-run.sh`):**
```bash
#!/bin/bash
set -e
# Ensure we use the Pixi environment's nvcc
export CUDACXX=$CONDA_PREFIX/bin/nvcc
cmake -G Ninja -S . -B build \
    -DCMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc \
    -DCUDAToolkit_ROOT=$CONDA_PREFIX
cmake --build build
./build/check_app
```
*(Create `.bat` equivalent for Windows)*

**4. Execute:**
Run the script using the configured environment:
```bash
chmod +x build-check/build-and-run.sh
pixi run --manifest-path <MANIFEST_FILE> [--feature <ENV_NAME>] ./build-check/build-and-run.sh
```

## Best Practices
*   **Never mix channels** for CUDA: Stick to `nvidia` for `cuda-*` packages.
*   **Explicit Compilation**: Always prefer passing `-DCMAKE_CUDA_COMPILER` over relying on implicit path resolution, as CMake might pick up `/usr/bin/nvcc`.
*   **User Space**: Remind the user that this setup requires NO `sudo` and works on any Linux machine with a driver.
