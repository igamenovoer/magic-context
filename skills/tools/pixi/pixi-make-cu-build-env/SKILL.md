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
Add the core build tools and the CUDA toolchain to the **existing project**.

**Best Practice**: Always add the `nvidia` channel to the project configuration *before* installing packages. This avoids flag ordering issues and ensures consistent dependency resolution.

**Command Logic**:
*   **Default Environment**: Do NOT use `--feature`.
*   **Named Environment**: Use `--feature <ENV_NAME>` (this will create the feature/environment if it doesn't exist).

```bash
# 1. Setup Channels
pixi project channel add nvidia --manifest-path <PROJECT_PATH>/<MANIFEST_FILE>

# 2. Core Build Tools (Install if missing/unsuitable on host)
pixi add --manifest-path <PROJECT_PATH>/<MANIFEST_FILE> [--feature <ENV_NAME>] cmake ninja cxx-compiler make pkg-config

# 3. CUDA Toolchain (Standard libs + Compiler)
pixi add --manifest-path <PROJECT_PATH>/<MANIFEST_FILE> [--feature <ENV_NAME>] cuda-toolkit=<VERSION> cuda-nvcc=<VERSION>
```

*Optional Extras (Only if requested):*
```bash
pixi add --manifest-path <PROJECT_PATH>/<MANIFEST_FILE> [--feature <ENV_NAME>] cudnn=<VERSION> nccl=<VERSION>
```

*Optional Tools (Only if requested):*
```bash
pixi add --manifest-path <PROJECT_PATH>/<MANIFEST_FILE> [--feature <ENV_NAME>] nsight-compute
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
To verify the setup, deploy the self-contained test suite from the skill's resource directory.

**1. Deploy Test Suite:**
Copy the demo code from the magic-context repository to the project:
```bash
mkdir -p build-check
cp -r magic-context/skills/tools/pixi/pixi-make-cu-build-env/demo-code/* build-check/
chmod +x build-check/build-and-run.sh
```

**2. Execute:**
Run the script using the configured environment. Ensure you run it from the `build-check` directory so CMake finds the source.
```bash
pixi run --manifest-path <MANIFEST_FILE> [--feature <ENV_NAME>] bash -c "cd build-check && ./build-and-run.sh"
```
*   **Note**: If the host has no GPU or an incompatible driver, the **build step** should still succeed, but the binary execution will fail. In this case, verify that `build-check/build/check_app` exists to confirm the compilation setup.

## Troubleshooting

### CMake Errors
*   **Source Directory**: If CMake complains "does not appear to contain CMakeLists.txt", ensure you are running the build script from inside the `build-check` directory (see Execution step above).
*   **Pthread Failure**: `Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed` is often normal; CMake usually finds `pthread` in the next step.
*   **Architecture Warnings**: `nvcc warning : Support for offline compilation...` indicates the default architecture might be older. You can safely ignore this warning for verification purposes.

### Runtime Errors
*   **Driver Mismatch**: If the kernel fails to launch, check `nvidia-smi`. The host driver must support the installed CUDA Toolkit version.

## Best Practices
*   **Never mix channels** for CUDA: Stick to `nvidia` for `cuda-*` packages.
*   **Explicit Compilation**: Always prefer passing `-DCMAKE_CUDA_COMPILER` over relying on implicit path resolution, as CMake might pick up `/usr/bin/nvcc`.
*   **User Space**: Remind the user that this setup requires NO `sudo` and works on any Linux machine with a driver.
