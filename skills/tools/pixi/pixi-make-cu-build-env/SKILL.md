---
name: pixi-make-cu-build-env
description: Guides the agent to setup a new or existing Pixi environment for compiling C++ and CUDA code. It ensures the correct compilers, toolkits, and CMake configurations are in place for a robust user-space build.
---

# Pixi Make CUDA Build Env

## Trigger

Use this skill when the user asks to:
- "Setup a CUDA build environment with Pixi"
- "Prepare this project for compiling .cu files"
- "Initialize a C++ and CUDA project using Pixi"

## Workflow

### 1. Requirements Gathering
**Mandatory**: You MUST ask the user for the specific CUDA version (e.g., 11.8, 12.1, 12.4).
*   **Version**: "Which CUDA version do you need?" (Do not assume a default).
*   **Libraries**: Ask if they need `cudnn`, `nccl`, `cutlass`, etc.

### 2. Environment Initialization
If the project doesn't exist:
```bash
pixi init <project_name>
cd <project_name>
```

### 3. Adding Dependencies
Add the core build tools and the CUDA toolchain. **Crucially**, use the `nvidia` channel for CUDA components and **explicitly pin** the requested version.

```bash
# Core Build Tools
pixi add cmake ninja cxx-compiler make pkg-config

# CUDA Toolchain (Replace <VERSION> with user input, e.g., 12.1)
pixi project channel add nvidia
pixi add cuda-toolkit=<VERSION> cuda-nvcc=<VERSION> -c nvidia
```

*Optional Libraries:*
```bash
pixi add cudnn=<VERSION> libcublas-dev=<VERSION> libcurand-dev=<VERSION> -c nvidia
```

### 4. Configuring Build Tasks
Add a standard CMake build task to `pixi.toml` (or `pyproject.toml`) that properly points to the user-space compilers. This avoids conflicts with system-installed CUDA.

**Task Definition:**
```toml
[tool.pixi.tasks]
configure = { cmd = "cmake -G Ninja -S . -B build -DCMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc -DCUDAToolkit_ROOT=$CONDA_PREFIX", env = { CUDACXX = "$CONDA_PREFIX/bin/nvcc" } }
build = "cmake --build build"
test = "ctest --test-dir build"
```

### 5. Verification
Create a minimal `hello.cu` and `CMakeLists.txt` to verify the setup if requested.

**Minimal CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.18)
project(SanityCheck LANGUAGES CXX CUDA)
add_executable(hello hello.cu)
```

## Best Practices
*   **Never mix channels** for CUDA: Stick to `nvidia` for `cuda-*` packages.
*   **Explicit Compilation**: Always prefer passing `-DCMAKE_CUDA_COMPILER` over relying on implicit path resolution, as CMake might pick up `/usr/bin/nvcc`.
*   **User Space**: Remind the user that this setup requires NO `sudo` and works on any Linux machine with a driver.
