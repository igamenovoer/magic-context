# Initialize Python Project with Pixi

This guide describes how to set up a new Python project using Pixi package manager with pyproject.toml format.

## Prerequisites

- `pixi` must be installed on the system
- `git` for version control
- GitHub CLI (`gh`) if working with GitHub repositories

## Setup Steps

### 1. Initialize Pixi Project

```bash
pixi init --format pyproject
```

This creates a `pyproject.toml` file that pixi will use for project configuration.

### 2. Configure Python Version and Platforms

**Determine Current Python Version:**

Use pixi to check available Python versions:

```bash
pixi search python
```

This will show the latest available versions. Look for the highest stable version (e.g., 3.14.0, 3.13.x) and then select **one major/minor version behind** for better package compatibility and stability.

For example:
- If Python 3.14.x is the latest → use Python 3.13
- If Python 3.13.x is the latest → use Python 3.12

**Configure pyproject.toml:**

Edit `pyproject.toml` to:
- Set the required Python version in `[project]` section
- Add Python dependency in `[tool.pixi.dependencies]` section
- Configure supported platforms in `[tool.pixi.workspace]` section

Example configuration with Windows/Linux support (using Python 3.12 as one version behind 3.13):

```toml
[project]
requires-python = ">= 3.12"

[tool.pixi.workspace]
channels = ["conda-forge"]
platforms = ["linux-64", "win-64"]

[tool.pixi.dependencies]
python = "3.12.*"
```

### 3. Add PyPI Dependencies

Use `pixi add` with the `--pypi` flag to add Python packages:

```bash
pixi add --pypi <package1> <package2> <package3> ...
```

Common packages for a typical Python project:
- **Core utilities**: attrs, click, omegaconf, pydantic
- **Scientific computing**: scipy, numpy
- **Computer vision**: opencv-python
- **Configuration**: hydra-core
- **Development tools**: mypy, ruff, pytest
- **Documentation**: mkdocs-material

Example:
```bash
pixi add --pypi attrs click omegaconf scipy opencv-python hydra-core mypy ruff mkdocs-material pydantic
```

### 4. Install Dependencies

After configuration, install all dependencies:

```bash
pixi install
```

## Optional Libraries

### PyTorch Installation

PyTorch is a popular deep learning framework that benefits from GPU acceleration via CUDA. Follow these steps to install PyTorch from the official source.

**Reference:** For the latest installation instructions and available CUDA versions, visit:
- Official PyTorch Installation Guide: <https://pytorch.org/get-started/locally/>
- PyTorch Previous Versions: <https://pytorch.org/get-started/previous-versions/>

**Determine CUDA Version:**

First, check the available CUDA version on your system (if you have an NVIDIA GPU):

```bash
nvidia-smi
```

Look at the "CUDA Version" in the output (e.g., 12.6, 12.4, 11.8). This indicates the maximum CUDA version your driver supports.

**Important Notes:**
- If the user specifies a CUDA version requirement, use that version
- Otherwise, use the CUDA version that matches or is slightly lower than what `nvidia-smi` reports
- For CPU-only installations, skip the CUDA version check

**Configure PyTorch in pyproject.toml:**

PyTorch should be installed from the official PyTorch wheel repository using PyPI, not from conda-forge. Add the following to your `pyproject.toml`:

```toml
[tool.pixi.pypi-dependencies]
torch = { version = "*", index = "https://download.pytorch.org/whl/cu126" }
torchvision = { version = "*", index = "https://download.pytorch.org/whl/cu126" }
torchaudio = { version = "*", index = "https://download.pytorch.org/whl/cu126" }
```

**Replace the CUDA version in the URL** based on your requirements:
- CUDA 12.6: `cu126`
- CUDA 12.4: `cu124`
- CUDA 12.1: `cu121`
- CUDA 11.8: `cu118`
- CPU only: `cpu`

**Example for CUDA 12.4:**
```toml
[tool.pixi.pypi-dependencies]
torch = { version = "*", index = "https://download.pytorch.org/whl/cu124" }
torchvision = { version = "*", index = "https://download.pytorch.org/whl/cu124" }
torchaudio = { version = "*", index = "https://download.pytorch.org/whl/cu124" }
```

**Example for CPU only:**
```toml
[tool.pixi.pypi-dependencies]
torch = { version = "*", index = "https://download.pytorch.org/whl/cpu" }
torchvision = { version = "*", index = "https://download.pytorch.org/whl/cpu" }
torchaudio = { version = "*", index = "https://download.pytorch.org/whl/cpu" }
```

**Verification:**

After installation, verify PyTorch is working correctly:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
```

## Project Structure

A typical initialized project will have:
```
project-root/
├── pyproject.toml          # Project configuration and dependencies
├── .pixi/                  # Pixi environment (git-ignored)
├── pixi.lock              # Lock file for reproducible environments
├── .gitattributes         # Git attributes
└── README.md              # Project documentation
```

## Adding Submodules

To add Git submodules (e.g., for shared prompt libraries):

```bash
git submodule add -b <branch> git@github.com:<user>/<repo>.git <path>
```

Example:
```bash
git submodule add -b main git@github.com:igamenovoer/magic-context.git magic-context
```

## Platform Support

Common platform identifiers:
- `linux-64`: 64-bit Linux
- `win-64`: 64-bit Windows
- `osx-64`: 64-bit macOS (Intel)
- `osx-arm64`: 64-bit macOS (Apple Silicon)

## Notes

- Use `pixi run <command>` to run commands in the pixi environment
- Use `pixi shell` to activate the environment in your current shell
- The `pyproject.toml` format integrates well with standard Python tooling
- Pixi manages both conda and PyPI packages in a unified way
