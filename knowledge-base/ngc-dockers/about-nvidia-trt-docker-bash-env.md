# NVIDIA NGC Docker Container Environment Variables and SSH Access

## Problem Statement

When building custom Docker images based on NVIDIA NGC containers (such as TensorRT, Triton Server, PyTorch, TensorFlow, etc.), you may encounter a critical issue:

**Python imports work via `docker exec` but fail via SSH login.**

### Example Symptom

```bash
# Works fine
docker exec <container> python3 -c "import tensorrt"  # SUCCESS

# Fails with ImportError
ssh root@container python3 -c "import tensorrt"
# ImportError: libnvinfer.so.10: cannot open shared object file: No such file or directory
```

This affects not just TensorRT, but potentially any NVIDIA library that relies on environment variables like `LD_LIBRARY_PATH`.

---

## Root Cause Analysis

### Why Docker ENV Variables Don't Work with SSH

Docker's `ENV` directive in Dockerfiles sets environment variables that are:

1. **Inherited by `docker exec`** - Because `docker exec` runs processes directly in the container's environment, which includes all ENV variables set in the Dockerfile
2. **NOT inherited by SSH login shells** - SSH creates a fresh login shell that only reads standard shell initialization files (`/etc/profile`, `~/.bashrc`, etc.), not Docker's ENV variables

This is a fundamental architectural difference between these two execution contexts:

| Execution Method | Environment Source |
|-----------------|-------------------|
| `docker exec` | Container's ENV (from Dockerfile) + parent process environment |
| SSH login | Shell initialization files only (`/etc/profile`, `/etc/bash.bashrc`, `~/.bashrc`) |

### The Missing Variable

The critical missing variable in SSH sessions is typically `LD_LIBRARY_PATH`, which tells the dynamic linker where to find shared libraries:

```bash
# Via docker exec (PRESENT):
LD_LIBRARY_PATH=/usr/local/tensorrt/lib/:/opt/tritonserver/backends/tensorrtllm:/usr/local/tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Via SSH (MISSING):
LD_LIBRARY_PATH=    # Empty or undefined
```

Without this, the Python interpreter cannot locate NVIDIA's shared libraries (`.so` files) like:
- `libnvinfer.so.10` (TensorRT)
- `libcudart.so` (CUDA Runtime)
- `libnccl.so` (NCCL)
- etc.

---

## NVIDIA NGC Container Architecture

### The NVIDIA Solution: `/etc/shinit_v2`

NVIDIA NGC containers use a custom initialization mechanism to ensure environment variables are set across all shell contexts:

#### 1. **BASH_ENV Variable**

NVIDIA sets in their Dockerfile:
```dockerfile
ENV BASH_ENV=/etc/bash.bashrc
```

This tells bash to source `/etc/bash.bashrc` for **all** shell invocations, including non-interactive ones.

#### 2. **Modified `/etc/bash.bashrc`**

The `/etc/bash.bashrc` file includes at the very top:
```bash
test -f /etc/shinit_v2 && source /etc/shinit_v2
```

This sources the custom initialization script before anything else.

#### 3. **The `/etc/shinit_v2` Script**

This NVIDIA-created script performs:
- **CUDA compatibility checks** using `/usr/local/bin/cudaCheck`
- **Driver version detection** from `/proc/driver/nvidia/version`
- **Conditional library path setup** based on CUDA compatibility mode
- **NCCL and MPI configuration** for multi-GPU setups

Example excerpt from `/etc/shinit_v2`:
```bash
#!/bin/sh
NV_DRIVER_VERS=$(sed -n 's/^NVRM.*Kernel Module\( for [a-z0-9_]*\| \) *\([^() ]*\).*$/\2/p' /proc/driver/nvidia/version 2>/dev/null)

export _CUDA_COMPAT_PATH=${_CUDA_COMPAT_PATH:-/usr/local/cuda/compat}

# Check CUDA compatibility and set up library paths
_CUDA_COMPAT_STATUS="$(LD_LIBRARY_PATH="${_CUDA_COMPAT_REALLIB}" \
                      /usr/local/bin/cudaCheck 2>/dev/null)"

if [ "${_CUDA_COMPAT_STATUS}" = "CUDA Driver OK" ]; then
  export LD_LIBRARY_PATH="${_CUDA_COMPAT_REALLIB}${LD_LIBRARY_PATH:+":${LD_LIBRARY_PATH}"}"
fi

# ... additional configuration
```

**However**, `/etc/shinit_v2` in NVIDIA base images does **NOT** include the full `LD_LIBRARY_PATH` that's defined in Dockerfile ENV variables. This is the source of the SSH problem.

---

## Why This Architecture Exists

### Historical Context

The pattern originates from **Alpine Linux** containers, where the Almquist shell (ash) uses an `ENV` variable to point to a shell initialization file:

```dockerfile
ENV ENV=/etc/shinit
RUN echo 'export PATH=$PATH:/custom/path' > /etc/shinit
```

NVIDIA adapted this pattern for bash-based containers:
- Alpine uses `ENV=/etc/shinit` (for ash/dash shells)
- NVIDIA uses `BASH_ENV=/etc/bash.bashrc` (for bash shells)

### Why Not Just Use `/etc/profile.d/`?

The `/etc/profile.d/` directory is **only sourced by login shells** (`bash -l`), not by:
- Non-interactive shells (`ssh host 'command'`)
- Subshells spawned by scripts
- Some automated tools and CI/CD pipelines

By using `BASH_ENV` → `/etc/bash.bashrc` → `/etc/shinit_v2`, NVIDIA ensures initialization across **all bash invocation types**.

---

## Shell Invocation Types and When They Load Config Files

Understanding shell types is crucial to solving this issue:

| Shell Type | Invocation Example | Loads `/etc/profile`? | Loads `/etc/bash.bashrc`? | Loads `BASH_ENV`? |
|-----------|-------------------|----------------------|---------------------------|-------------------|
| Interactive login | `ssh user@host` (without command) | ✅ Yes | ✅ Yes (via /etc/profile) | ✅ Yes |
| Interactive non-login | `bash` (in terminal) | ❌ No | ✅ Yes | ✅ Yes |
| Non-interactive remote | `ssh user@host 'command'` | ❌ No | ⚠️ **Only if BASH_ENV is set** | ✅ Yes |
| `docker exec` | `docker exec container bash` | ❌ No | ✅ Yes | ✅ Yes |

**Key Insight**: Without `BASH_ENV=/etc/bash.bashrc`, non-interactive SSH commands (`ssh host 'python ...'`) will NOT source `/etc/bash.bashrc`, causing environment variables to be missing.

---

## Solutions

### Solution 1: Add Missing Variables to `/etc/shinit_v2` (Recommended)

Since `/etc/shinit_v2` is already being sourced for all shell types, append the missing environment variables there:

```bash
docker exec <container> bash -c 'cat >> /etc/shinit_v2 << "EOF"

# NVIDIA library paths for TensorRT and CUDA
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/usr/local/tensorrt/lib/:/opt/tritonserver/backends/tensorrtllm:/usr/local/tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64}
export LIBRARY_PATH=${LIBRARY_PATH:-/usr/local/cuda/lib64/stubs:}
EOF
'
```

**Why this works:**
- `/etc/shinit_v2` is sourced by `/etc/bash.bashrc` before the `[ -z "$PS1" ] && return` early exit
- Applies to all shell types: login, non-login, interactive, non-interactive
- Uses `${VAR:-default}` to avoid overwriting if already set

**Testing:**
```bash
# Test non-interactive SSH
ssh -p 22222 root@localhost 'python3 -c "import tensorrt; print(tensorrt.__version__)"'

# Test interactive SSH
ssh -p 22222 root@localhost
python3 -c "import tensorrt"

# Verify docker exec still works
docker exec <container> python3 -c "import tensorrt"
```

### Solution 2: Add to `/etc/profile.d/` (Partial Solution)

Create a file in `/etc/profile.d/`:

```bash
cat > /etc/profile.d/nvidia-env.sh << 'EOF'
export LD_LIBRARY_PATH=/usr/local/tensorrt/lib/:/opt/tritonserver/backends/tensorrtllm:/usr/local/tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:
EOF
```

**Limitations:**
- ⚠️ Only works for **login shells**
- ❌ Does NOT work for `ssh host 'command'` (non-interactive)
- ❌ May not work in all CI/CD environments

### Solution 3: Modify Dockerfile (Best for Custom Images)

When building custom images based on NVIDIA NGC containers, add to your Dockerfile:

```dockerfile
FROM nvcr.io/nvidia/tritonserver:25.08-py3

# Ensure BASH_ENV is still set (inherit from base image)
ENV BASH_ENV=/etc/bash.bashrc

# Add NVIDIA library paths to shinit_v2
RUN echo '' >> /etc/shinit_v2 && \
    echo '# Custom: Ensure LD_LIBRARY_PATH is set for SSH sessions' >> /etc/shinit_v2 && \
    echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/usr/local/tensorrt/lib/:/opt/tritonserver/backends/tensorrtllm:/usr/local/tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64}' >> /etc/shinit_v2 && \
    echo 'export LIBRARY_PATH=${LIBRARY_PATH:-/usr/local/cuda/lib64/stubs:}' >> /etc/shinit_v2

# Rest of your custom configuration
# ...
```

**Benefits:**
- ✅ Permanent solution that survives container rebuilds
- ✅ Works across all shell contexts
- ✅ Documents the fix in your infrastructure-as-code

### Solution 4: Use `/etc/environment` (Not Recommended)

```bash
cat >> /etc/environment << 'EOF'
LD_LIBRARY_PATH=/usr/local/tensorrt/lib/:/opt/tritonserver/backends/tensorrtllm:/usr/local/tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
LIBRARY_PATH=/usr/local/cuda/lib64/stubs:
EOF
```

**Limitations:**
- `/etc/environment` is read by PAM (Pluggable Authentication Modules) for SSH sessions
- Does NOT support variable expansion or conditional logic
- Not sourced by `docker exec`
- Requires SSH to be configured with `UsePAM yes`

---

## Best Practices for Custom Images Based on NVIDIA NGC Containers

### 1. **Preserve BASH_ENV**

Always ensure `BASH_ENV=/etc/bash.bashrc` is set. If you override ENV variables, re-export it:

```dockerfile
ENV BASH_ENV=/etc/bash.bashrc
```

### 2. **Verify `/etc/shinit_v2` Exists and is Sourced**

Check that `/etc/bash.bashrc` contains:
```bash
test -f /etc/shinit_v2 && source /etc/shinit_v2
```

### 3. **Add Critical Environment Variables to `/etc/shinit_v2`**

For any custom library paths or environment variables your application needs:

```dockerfile
RUN echo 'export MY_CUSTOM_PATH=/opt/myapp/lib:${MY_CUSTOM_PATH}' >> /etc/shinit_v2
```

### 4. **Test All Access Methods**

Verify your environment works in all contexts:

```bash
# Test 1: docker exec
docker exec <container> python3 -c "import tensorrt"

# Test 2: Non-interactive SSH
ssh -p <port> root@<host> 'python3 -c "import tensorrt"'

# Test 3: Interactive SSH
ssh -p <port> root@<host>
python3 -c "import tensorrt"

# Test 4: SSH with explicit command and login shell
ssh -p <port> root@<host> -t 'bash -l -c "python3 -c \"import tensorrt\""'
```

### 5. **Document the Behavior**

Add comments to your Dockerfile explaining the BASH_ENV setup:

```dockerfile
# NVIDIA NGC containers use BASH_ENV to ensure environment variables
# are available in all shell contexts (docker exec, SSH, etc.)
# We add our custom variables to /etc/shinit_v2 to maintain this behavior
RUN echo 'export CUSTOM_VAR=value' >> /etc/shinit_v2
```

---

## Debugging Environment Issues

### Compare Environments

```bash
# Via docker exec
docker exec <container> env | sort > docker-exec-env.txt

# Via SSH
ssh -p <port> root@<host> 'env' | sort > ssh-env.txt

# Compare
diff docker-exec-env.txt ssh-env.txt
```

### Check Which Files Are Sourced

Add debugging to your shell initialization:

```bash
# Add to /etc/bash.bashrc
echo "Sourcing /etc/bash.bashrc" >&2

# Add to /etc/shinit_v2
echo "Sourcing /etc/shinit_v2" >&2

# Add to ~/.bashrc
echo "Sourcing ~/.bashrc" >&2
```

Then observe output:
```bash
ssh -p <port> root@<host> 'echo test'
# Should show which files were sourced
```

### Trace Shell Initialization

```bash
# Start bash with tracing
ssh -p <port> root@<host> 'bash -x -c "python3 -c \"import sys; print(sys.version)\""'
```

### Check Library Dependencies

```bash
# Find what libraries Python is trying to load
ssh -p <port> root@<host> 'ldd /usr/bin/python3'

# Check if TensorRT library exists
ssh -p <port> root@<host> 'find /usr -name "libnvinfer.so*" 2>/dev/null'

# Test library loading directly
ssh -p <port> root@<host> 'LD_LIBRARY_PATH=/usr/local/tensorrt/lib python3 -c "import tensorrt"'
```

---

## Common Pitfalls

### ❌ Don't Rely on Dockerfile ENV Alone

```dockerfile
# This will NOT work for SSH sessions
ENV LD_LIBRARY_PATH=/usr/local/tensorrt/lib
```

### ❌ Don't Use `source` in Non-Interactive Scripts

```bash
# This fails in non-interactive shells
ssh host 'source /etc/profile && python3 script.py'
```

### ❌ Don't Assume `/etc/profile.d/` Always Runs

```bash
# Only runs for login shells
cat > /etc/profile.d/myenv.sh << 'EOF'
export MY_VAR=value
EOF
```

### ❌ Don't Forget to Test SSH Access

Many developers only test with `docker exec` and discover SSH issues in production.

---

## Technical Deep Dive: Why NVIDIA Doesn't Include LD_LIBRARY_PATH in shinit_v2

### The Design Decision

NVIDIA's `/etc/shinit_v2` focuses on **dynamic** environment setup:
- CUDA compatibility mode detection
- Driver version checking
- Conditional library path adjustment

The static `LD_LIBRARY_PATH` is set in Dockerfile ENV because:
1. It's consistent across all containers of the same image
2. ENV variables are inherited by `docker exec` automatically
3. Most NVIDIA users interact with containers via `docker exec`, not SSH

### The Gap

NVIDIA **did not anticipate** (or prioritize) the SSH server use case, where:
- Users install `openssh-server` in the container
- SSH sessions don't inherit Docker ENV variables
- The static `LD_LIBRARY_PATH` needs to be in shell init files

### The Consequence

Custom images that add SSH servers need to **manually bridge this gap** by adding static environment variables to `/etc/shinit_v2` or equivalent shell initialization files.

---

## Summary

### Key Takeaways

1. **Docker ENV ≠ Shell Environment**: Variables set in Dockerfile ENV are not automatically available to SSH login shells
2. **NVIDIA's Solution**: Uses `BASH_ENV` → `/etc/bash.bashrc` → `/etc/shinit_v2` for cross-context initialization
3. **The Fix**: Add static environment variables (especially `LD_LIBRARY_PATH`) to `/etc/shinit_v2` when building custom images with SSH
4. **Test Thoroughly**: Verify your environment in all contexts: `docker exec`, SSH, non-interactive shells, etc.

### Quick Reference

| Problem | Solution |
|---------|----------|
| TensorRT import fails via SSH | Add `LD_LIBRARY_PATH` to `/etc/shinit_v2` |
| Custom library not found | Add library path to `/etc/shinit_v2` |
| Environment var missing in SSH | Add export to `/etc/shinit_v2` |
| Works in docker exec, not SSH | Check if `BASH_ENV=/etc/bash.bashrc` is set |
| Non-interactive SSH fails | Use `/etc/shinit_v2`, not `/etc/profile.d/` |

---

## Additional Resources

- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Bash Startup Files](https://www.gnu.org/software/bash/manual/html_node/Bash-Startup-Files.html)
- [Docker ENV Reference](https://docs.docker.com/reference/dockerfile/#env)
- [SSH Environment Variables](https://man.openbsd.org/sshd_config#PermitUserEnvironment)

---

**Last Updated**: 2025-10-14
**Author**: Based on debugging session with Claude Code
**Version**: 1.0
