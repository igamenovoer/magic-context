---
name: explore-dnn-model
description: Manual invocation only; use only when the user explicitly requests `explore-dnn-model` by name. Explore how to run a given DNN model checkpoint in the current Python environment by locating weights + upstream source code, resolving dependencies with user confirmation, running reproducible experiments under `tmp/`, and producing reports + a step-by-step tutorial with inputs/outputs.
---

# Explore DNN Model

## Minimum Required Inputs (Hard Requirement)

To use this skill, the user must provide:
- A model checkpoint (file or directory path, or a URL that can be downloaded into a local file/dir), **and**
- At least one of:
  - Model name, or
  - Upstream repository link, or
  - A checked-out source code path in the workspace

This is required so there are enough keywords and artifacts to identify the canonical implementation and infer the correct preprocessing/postprocessing.

## Goals

This skill has four goals:

1) Verify that the given DNN model can work (inference or training; default focus is **inference**) in the *current* Python environment of the workspace.
2) Determine how to use it (inference or training; default is **inference**) by reading the upstream source code and producing minimal, reproducible runs.
3) Write a report covering:
   - Input and output contracts (formats, shapes, dtypes, preprocessing/postprocessing)
   - Benchmarks and performance profiling (latency/throughput/memory, device details)
   - User-provided metrics/targets (e.g., accuracy, mAP, IoU, F1, latency budget), and whether/how they are met
4) Write a step-by-step how-to tutorial so the user can reproduce usage end-to-end.

Before changing anything, detect how the environment is managed by checking for:
- `pixi.toml` and/or `pyproject.toml` (Pixi-managed project)
- `.venv/` (venv-managed project)

## Dependency Policy (Ask Once, Then Apply)

If any dependency is missing:
- Do **not** install it automatically *without user confirmation*.
- List the missing packages (and versions/constraints if known) and ask the developer how to proceed.
- Provide clear options, let the developer choose, then proceed with the chosen approach.
- Once the developer confirms an approach, apply it for **all** newly required packages (no need to ask approval per package).

### Version Strategy

- First attempt: use the **latest versions** resolved by the selected package manager (`pixi`, `pip`, `uv`).
- If that fails (import/runtime errors, incompatibilities): fall back to the **specific versions/constraints** documented by the model’s upstream source code or docs.

### Preferred Options (in order)

**Pixi-managed env**
- Ask the user to choose one:
  - Modify the current Pixi environment by adding deps to the relevant manifest (`pixi.toml` / `pyproject.toml`).
  - Create a new Pixi environment specifically to test this model.
- Then use `pixi install`/`pixi run ...` to execute.
- Prefer **PyPI** packages over **conda-forge** when both are available.
- Avoid direct `pip install ...` into the Pixi environment unless the developer explicitly requests it.

**`.venv`-managed env**
Ask the user to choose one:
- Install deps via `pip` (or `uv pip`) into the current `.venv`.
- Create a new venv specifically for this model (keeps the repo venv clean).

## Inputs to Collect (ask if missing)

- Model task/modality if unclear (classification/detection/segmentation/embedding/audio/video/etc.)
- Checkpoint path (file/dir) and format (`.pt`, `.pth`, `.onnx`, `.engine`, etc.)
- Any known I/O contract details (expected resolution, channel order, normalization, label mapping), if the user has them
- CPU-only requirement (only if the user explicitly requests CPU-only)
- Optional: user-provided metrics/targets to evaluate (quality and/or performance)

Notes:
- Determine framework/runtime automatically from checkpoint type + upstream code/docs + what’s available in the current Python environment.
- If hardware is unspecified, default to using hardware acceleration when available (CUDA GPU, ROCm GPU, Apple MPS, etc.). Use CPU-only only if the user requested it.
- If unspecified, the default objective is to confirm the model runs end-to-end from input → output (prefer real inputs found in the workspace; synthesize as a fallback) and record end-to-end timing.

## Core Workflow

### 0) Confirm artifacts and pick the target environment

- Confirm the minimum required inputs are present:
  - Checkpoint path/URL is accessible.
  - Model name/repo/source path is provided so you can derive keywords and locate canonical docs/code.
- Detect environment type:
  - If both Pixi and `.venv` exist, ask the user which one should be treated as the “current” environment for this exploration.
- Device default:
  - If the user did not request CPU-only, use hardware acceleration when available (CUDA/ROCm/MPS/etc.).

### 1) Locate and read the upstream source code/docs

- Use online search to find the canonical implementation:
  - Official GitHub repo, paper, model card, or vendor docs.
- Download the relevant source code (pin a tag/commit when possible) and identify:
  - The exact inference entrypoints (scripts/modules), model class, preprocessing, postprocessing, and label mapping.
  - Any config files required to construct the model (YAML/JSON/TOML).
- Do not “guess” preprocessing/postprocessing: confirm from code and/or reference examples.

### 2) Derive required dependencies

Before downloading the checkpoint or changing the environment, determine the minimal dependencies required to run the model by using (in priority order):
- Upstream source code (setup files, `requirements*.txt`, `pyproject.toml`, import graph).
- Upstream docs/model card (pinned versions, known-good combos).
- Checkpoint type (e.g., `.onnx` implies ONNX Runtime; `.pt/.pth` implies PyTorch; `.engine` implies TensorRT).

Make a concise dependency list covering:
- Runtime/framework (e.g., `torch`, `onnxruntime`, `opencv-python`)
- Model-specific libs (e.g., `ultralytics`, `timm`, `transformers`, `mmengine`, etc.)
- Utility deps used by the official inference path (e.g., `numpy`, `Pillow`, `pyyaml`)
- Optional acceleration deps (CUDA/TensorRT) separated from the CPU baseline

### 3) Resolve missing dependencies (with user choice)

- Check whether each required dependency is available in the current environment.
- If anything is missing, ask the user which path to take:
  - **Pixi:** modify current manifest to add deps, or create a new Pixi env for this model.
  - **Venv:** install into current `.venv`, or create a new venv for this model.
- After the user confirms, apply the decision for all required packages (no per-package prompts).
- Use the **Version Strategy** above (latest first; fall back to pinned versions if needed).
- After dependency changes, run a quick smoke test:
  - Imports for the core runtime stack
  - Minimal “load model” path (without a full benchmark yet)

### 4) Ensure the checkpoint exists locally

- Prefer storing checkpoints under the repo-level `checkpoints/` directory (gitignored).
- Record provenance in a short note:
  - Source URL(s), version/commit/tag, download date, file size, and (if feasible) SHA256.
- If the checkpoint is “model code + weights” (e.g., a HF repo), still treat the weights as the artifact that must be present and verifiable.

### 5) Create an experiment workspace under `tmp/`

Default experiment directory:

`{{workspace}}/tmp/{{experiment-slug}}-<time>`

If the user specifies a different location/name, use the user-provided one instead.

Create the experiment root directory, then immediately create a written plan in the experiment root **before** implementing anything else:

- Create `{{experiment-dir}}/plan.md` with:
  - A short summary of what is being tested (model, checkpoint, task, runtime, device)
  - A TODO checklist (`- [ ] ...`) for each major step (deps, checkpoint, inputs, inference run, timing/benchmark, reporting, tutorial)
  - Any assumptions/unknowns discovered from the checkpoint/source/docs
- As you progress, keep `plan.md` up to date by checking off items (`- [x] ...`) so users can see current status.

Then create the standard directory layout:

```
tmp/<experiment-dir>/
  plan.md
  scripts/      # throwaway but reproducible scripts (committed if useful)
  inputs/       # downloaded/synthesized test inputs
  outputs/      # raw outputs: images/json/npy/logs
  reports/      # markdown notes: what was tried, params, results
  tutorial/
    step-by-step.md
    inputs/     # tutorial-specific inputs (small, redistributable if possible)
    outputs/    # tutorial-specific outputs
```

Conventions:
- Use relative paths from `tmp/<experiment-dir>` in scripts so the folder is movable.
- Keep scripts small and single-purpose (`01_download_inputs.py`, `10_infer.py`, `20_visualize.py`, …).
- Run Python via the selected environment manager:
  - Pixi: `pixi run python ...`
  - Venv: use the venv’s Python (avoid system Python)

### 6) Collect or synthesize inputs

- First try to find suitable inputs already present in the workspace (e.g., under `datasets/`, `downloads/`, or other project-specific data dirs) based on what you learned from the checkpoint/source code (task, modality, expected resolution, file types).
- If no suitable inputs exist locally, synthesize minimal inputs that satisfy the model contract (e.g., generated images, random tensors saved in the expected container format, short synthetic video).
- Save all chosen/generated inputs under `tmp/<experiment-dir>/inputs/` and copy a minimal subset to `tmp/<experiment-dir>/tutorial/inputs/`.

### 7) Run minimal, traceable inference experiments (default: inference + end-to-end timing)

- Start with a single known-good example (from upstream repo) if available.
- Save every “input → output” mapping:
  - Inputs: the exact file(s) used + preprocessing parameters.
  - Outputs: raw model outputs + any decoded/visualized artifacts.
  - Command line + environment notes (device, precision, batch size).
- Measure end-to-end timing by default:
  - At minimum: one cold run + a small number of warm runs (record mean/median).
- If the model is accessed via HTTP/gRPC, save request/response payloads (sanitized) under `reports/` and/or `outputs/`.

### 7b) (Optional) Training sanity check

If the user asks to validate training (or if inference is insufficient to validate “works”):
- Start with a minimal configuration (single batch / tiny subset) to confirm the forward + backward pass runs.
- Record key configs (optimizer, LR, batch size, mixed precision) and any dataset assumptions.
- Do not run long trainings unless the user explicitly requests it.

### 8) Write reports (while iterating)

In `tmp/<experiment-dir>/reports/`, maintain markdown notes that answer:
- How to load the checkpoint (exact code snippet / API)
- Exact input format + preprocessing
- Output format + postprocessing
- Known pitfalls (dtype, channel order, normalization, resize/letterbox, NMS, thresholds)
- A small table that links:
  - input filename → script/command → output filenames

Also include:
- **Benchmark & profiling** results:
  - CPU/GPU model, RAM/VRAM, OS, Python version, key library versions
  - Latency breakdown if possible (preprocess / model / postprocess)
  - Throughput (items/s) and peak memory/VRAM
- **User metrics** (if provided):
  - The metric definition + measurement method
  - Results on the chosen evaluation inputs
  - Any deltas vs the user’s targets and suggested next experiments

### 9) Produce a tutorial

Create `tmp/<experiment-dir>/tutorial/step-by-step.md` as the final deliverable:
- Step-by-step setup (env, deps, checkpoint placement)
- How to run the scripts (exact commands)
- How to reproduce at least one end-to-end run
- Include small tutorial artifacts:
  - `tutorial/inputs/*` and `tutorial/outputs/*` (keep them minimal and redistributable)
- Prefer referencing `scripts/` as the canonical implementation; the tutorial should explain the “why” and “how”, not duplicate code.

If you need more structure for authoring the tutorial, use the `make-program-tutorial` skill and its template.

## Guardrails

- Do not commit large checkpoints or huge outputs; keep them under gitignored paths (`checkpoints/`, `tmp/`).
- Respect upstream licenses; record the repo URL + commit/tag in `reports/`.
- Avoid modifying runtime code under `src/` unless the user explicitly requests integration; keep exploration isolated to `tmp/<experiment-dir>`.
