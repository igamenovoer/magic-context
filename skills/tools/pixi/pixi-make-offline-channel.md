---
name: pixi-make-offline-channel
description: Use when the user wants to create a self-hosted, offline-installable Conda channel (mirror) containing a specific subset of packages using Pixi.
---

# Pixi Make Offline Channel

## Trigger

Use this skill when the user asks to:
- "Create a local conda channel with pixi"
- "Mirror a subset of conda packages for offline use"
- "Download conda packages and dependencies using pixi"
- "Create a self-hosted conda-forge mirror"

## Overview

This skill guides the user to creating a **custom, self-hosted Conda channel** (mirror) that contains only a selected subset of packages but includes **all transitive dependencies**. This ensures the channel is fully installable on offline machines.

The workflow uses Pixi's lockfile (`pixi.lock`) as the source of truth for dependency resolution and `rattler-index` for generating channel metadata.

## Workflow

### 1. Setup Builder Project

Create a temporary "builder" project to define the package set. This project will not be used for running code, but for resolving the dependency tree.

```bash
mkdir my-mirror-builder
cd my-mirror-builder
pixi init
```

### 2. Define Package Subset

Add the target packages you want to mirror. Also add the necessary build tools (`rattler-index` and `pyyaml`) to the environment.

```bash
# Target packages to mirror
pixi add pandas scikit-learn pytorch

# Tools required for the mirroring process
pixi add rattler-index pyyaml requests
```

### 3. Fetch Artifacts (The "Mirror Script")

Use a Python script to parse `pixi.lock`, extract the exact URLs for all resolved dependencies, and download them into a standard channel structure (e.g., `channel/linux-64/`).

**Key Steps for the Script:**
1.  Read `pixi.lock` (YAML format).
2.  Iterate through `packages`.
3.  Download the file at `url` to `output_dir/<platform>/<filename>`.

*Ref: See `context/hints/howto-create-self-hosted-conda-subset-repo-with-pixi.md` for the complete `build_mirror.py` script.*

### 4. Index the Channel

After downloading all artifacts, use `rattler-index` to generate the `repodata.json`. This turns the directory into a valid Conda channel.

```bash
pixi run rattler-index local-channel
```

### 5. Verify & Consume

The channel is now ready. It can be used via `file://` or hosted via HTTP.

**Verification:**
```bash
# Create a test env using ONLY the local channel
conda create -n test-offline -c file://$(pwd)/local-channel pandas --offline --override-channels
```

## Tips

*   **Multi-Platform**: To support multiple platforms (e.g., `linux-64` AND `osx-arm64`), manually edit `pixi.toml` to include them (`platforms = ["linux-64", "osx-arm64"]`) and run `pixi lock` before running the download script.
*   **Updates**: To update the mirror, simply `pixi update` the builder project, re-run the download script (it should skip existing files), and re-run `rattler-index`.
*   **Pixi-Pack**: If the goal is just to *move* one environment, suggest `pixi-pack` instead. This skill is for creating a *reusable channel*.

## Resources

*   **[howto-create-self-hosted-conda-subset-repo-with-pixi.md](../../../../context/hints/howto-create-self-hosted-conda-subset-repo-with-pixi.md)**: Detailed step-by-step guide with the full Python script.
