# Make Model Experiment Directory

This guide shows how to set up only the parts needed for a model experiment: Hydra config subdirectories, the `models/` directory, and the `datasets/` directory. It mirrors the patterns documented in `context/hints/nv-profile-kb/about-profile-project-structure.md` while focusing on these three areas.

## Hydra Subdirs

Use a grouped config tree under `conf/` so experiments compose cleanly via Hydra defaults. Create the following structure and minimal files for a new model. The tree below shows typical groups, per‑stage output presets, and model‑specific output overlays, plus sampling presets and Torch Profiler presets.

```
conf/                                           # Hydra config tree (grouped, swappable)
├── config.yaml                                 # Root defaults list that mounts groups below
├── dataset/                                    # Dataset identity + sampling controls
│   ├── <dataset-name>.yaml                     # Dataset root + subset filelist for this experiment
│   └── sampling/                               # Sampling presets mounted as dataset.sampling
│       ├── default.yaml                        # Deterministic order, basic epoch/sample knobs
│       └── random.yaml                         # Randomized order or alternate sampling strategy
├── hardware/                                   # Device selection and hardware options
│   └── single_gpu.yaml                         # Example: select a single CUDA device (optional)
├── model/                                      # Model configs grouped by arch/infer variants
│   └── <model-name>/                           # All configs for a single model family
│       ├── arch/                               # Architecture + preprocessing defaults
│       │   └── <model-name>.default.yaml       # Baseline arch preset for the model
│       ├── infer/                              # Inference behavior (temperature, max tokens, ...)
│       │   └── <model-name>.default.yaml       # Baseline inference preset for the model
│       └── output/                             # Model-specific output overlays (optional)
│           ├── torch/                          # Stage: torch_profiler — extra model-specific outputs
│           │   └── default.yaml                # Overlay merged with shared output/torch/default.yaml
│           └── direct/                         # Stage: direct_inference — extra model-specific outputs
│               └── default.yaml                # Overlay merged with shared output/direct/default.yaml
├── output/                                     # Per-stage output presets (model-agnostic)
│   ├── torch/                                  # Stage: torch_profiler — CSVs, traces, visualizations
│   │   └── default.yaml                        # Shared defaults for torch_profiler outputs
│   └── direct/                                 # Stage: direct_inference — predictions/visualizations
│       └── default.yaml                        # Shared defaults for direct inference outputs
├── profiling/                                  # Profiler presets (example only)
│   └── torch/                                  # Example: Torch Profiler group
│       └── <torch-profiler-preset>.yaml        # e.g., torch-profiler.default.yaml (min/max variants possible)
├── pipeline/                                   # Future-friendly: swappable pipeline behaviors
│   ├── default.yaml                            # Baseline enabling static_analysis + torch_profiler
│   └── <specific-pipeline>.yaml                # e.g., stage1.yaml or deep_profile.yaml (optional)
└── inference_runtime/                          # Inference runtime layer presets
    ├── pytorch.yaml                            # Example: PyTorch runtime knobs
    ├── tensorrt.yaml                           # Example: TensorRT-LLM runtime knobs
    └── onnxruntime.yaml                        # Example: ONNX Runtime knobs
```

Minimal example for `conf/config.yaml` to mount your model configs alongside dataset and profiling presets:

```yaml
defaults:
  - dataset: omnidocbench
  - dataset/sampling@dataset.sampling: default
  - model/<model-name>/arch@model: <model-name>.default
  - model/<model-name>/infer@infer: <model-name>.default
  - output/torch@pipeline.torch_profiler.output: default
  - model/<model-name>/output/torch@pipeline.torch_profiler.output.extra.<model-name>: default
  - output/direct@pipeline.direct_inference.output: default
  - model/<model-name>/output/direct@pipeline.direct_inference.output.extra.<model-name>: default
  - profiling/torch@pipeline.torch_profiler: torch-profiler.default
  - _self_

hydra:
  run:
    dir: ${hydra:runtime.cwd}/tmp/profile-output/${now:%Y%m%d-%H%M%S}
  output_subdir: null
  job:
    chdir: true
```

Notes:
- Keep model‑specific defaults in `conf/model/<model-name>/{arch,infer}/...` and swap them via CLI overrides when needed.
- Place additional shared knobs in `conf/output/`, `conf/profiling/`, and `conf/inference_runtime/` as your pipelines evolve.

### Optional: Pipeline Group (future‑friendly)

To make the pipeline configuration in `conf/config.yaml` more flexible and swappable, add a dedicated Hydra group under `conf/pipeline/`. This lets you select different end‑to‑end pipeline behaviors (enable/disable stages, outputs) via a single defaults entry without editing the root config.

Structure:

```
conf/
└── pipeline/
    ├── default.yaml                   # Baseline pipeline behavior
    └── <specific-pipeline>.yaml       # e.g., stage1.yaml, deep_profile.yaml
```

Minimal `conf/pipeline/default.yaml` (Torch Profiler only here; add others as needed):

```yaml
pipeline:
  direct_inference:
    enable: false
  static_analysis:
    enable: true
    write_reports: true
  torch_profiler:
    # Mirror the selected Torch Profiler preset’s enabled flag
    enable: ${pipeline.torch_profiler.enabled}
```

Mount it in `conf/config.yaml` alongside your model/dataset and Torch Profiler preset:

```yaml
defaults:
  - dataset: <dataset-name>
  - dataset/sampling@dataset.sampling: default
  - model/<model-name>/arch@model: <model-name>.default
  - model/<model-name>/infer@infer: <model-name>.default
  - output/torch@pipeline.torch_profiler.output: default
  - model/<model-name>/output/torch@pipeline.torch_profiler.output.extra.<model-name>: default
  - profiling/torch@pipeline.torch_profiler: torch-profiler.default
  - pipeline: default                # ← single switch to pick a pipeline behavior
  - _self_
```

Tips:
- Create `conf/pipeline/<specific>.yaml` variants (e.g., `stage1.yaml`) to toggle stage flags or outputs for particular workflows, then select with `pipeline=<specific>`.
- You can extend the pipeline entries later (e.g., add Nsight sections) without touching `config.yaml`; keep Nsight as an optional preset mounted only when needed.

Example: add Nsight presets later (optional)

You can add Nsight Systems/Compute presets outside of the basic Torch Profiler setup. Example structure and defaults:

```
conf/
└── profiling/
    ├── nsys/
    │   └── nsys.default.yaml
    └── ncu/
        └── ncu.default.yaml
```

Mount them in `conf/config.yaml` only when needed:

```yaml
defaults:
  - profiling/nsys@pipeline.nsys: nsys.default
  - profiling/ncu@pipeline.ncu: ncu.default
```

## Models

The `models/` directory holds model weights/tokenizers as symlinks or submodules. Prefer symlinks to external storage for large artifacts. Typical layout and bootstrapping flow:

```
models/
├── bootstrap.sh                 # Creates per-model symlinks
├── bootstrap.yaml               # Bootstrap configuration (paths, checks)
└── <model-name> -> /path/to/snapshots/<model-name>  # Symlink to host storage
```

Bootstrapping options:
- Workspace orchestrator: from repo root, run `./bootstrap.sh --yes` to call component bootstraps in sequence.
- Per‑model bootstrap: run `./models/bootstrap.sh --yes` with a configured `bootstrap.yaml` and environment variables such as `HF_SNAPSHOTS_ROOT` or `MODELS_ROOT`.
- Manual symlink (example): `ln -s /nvme/hf-cache/<model-name> models/<model-name>`.

Policy:
- Do not commit symlinks that point to external storage; create them via bootstrap scripts per host.
- Document any required files or validation in `models/bootstrap.yaml` and keep the script thin.

## Datasets

Organize both original datasets and derived variants in a consistent, documented structure. Each dataset lives under its own folder with a `source-data` symlink and optional variant subdirectories.

```
datasets/
└── <dataset-name>/
    ├── bootstrap.sh                     # Creates source-data symlink and optional prep
    ├── bootstrap.yaml                   # Dataset bootstrap configuration
    ├── source-data -> /path/to/<dataset>  # Symlink to external storage
    ├── README.md                        # Dataset documentation
    ├── metadata.yaml                    # Optional dataset facts
    └── <variant-name>/                  # Optional processed/filtered variants
        ├── data/
        └── metadata.yaml
```

Bootstrapping options:
- Per‑dataset bootstrap: `DATASETS_ROOT=/mnt/datasets ./datasets/<dataset-name>/bootstrap.sh --yes` to create or refresh the `source-data` link and perform optional preparation.
- Workspace orchestrator: `./bootstrap.sh --yes` from repo root to run dataset then model bootstraps.
- Manual symlink: `ln -s /mnt/datasets/<dataset-name> datasets/<dataset-name>/source-data`.

Symlink policy:
- Never commit `datasets/<dataset-name>/source-data` symlinks; they are environment‑specific and must be created per host.

Recommended ignores:

```gitignore
# datasets/.gitignore
*/source-data
*/source-data/**
```

Example steps to add a new dataset:
1) Create the directory: `mkdir -p datasets/<dataset-name>`
2) Run the dataset bootstrap (preferred) or create the `source-data` symlink manually
3) Add `metadata.yaml` and `README.md` documenting provenance and structure
4) Add a Hydra entry for it under `conf/dataset/<dataset-name>.yaml` and reference it in `conf/config.yaml`

That’s it — with these three areas in place, you can start composing experiments via Hydra defaults and keep large artifacts out of version control via symlinks and local paths.

### Dataset Subdir Template (recommended pattern)

The following template mirrors a proven layout and bootstrap flow and can be used for any `<dataset-name>` by substituting placeholders and configuring the bootstrap accordingly.

Layout:

```
datasets/<dataset-name>/
├── README.md                          # Dataset overview and bootstrap instructions
├── bootstrap.sh                       # Creates source-data symlink and optional extraction
├── bootstrap.yaml                     # Bootstrap configuration (env roots, target subdir, zips)
└── source-data -> /path/to/<dataset>  # Symlink to external storage (created by bootstrap.sh)
```

Bootstrap configuration template (`datasets/<dataset-name>/bootstrap.yaml`):

```yaml
env:
  data_root_env: DATASETS_ROOT                         # Env var pointing to a base datasets directory
  default_data_root: /workspace/datasets               # Fallback base directory when env is not set
dataset:
  source_subdir: <external-dataset-subdir-or-snapshot> # Resolved under the chosen base directory
  repo_link_name: source-data                           # Name of the repo-local symlink
  zips:                                                 # Optional archives to extract in target dir
    - <maybe-images.zip>
    - <maybe-pdfs.zip>
  unzip_targets:                                        # Optional map zip -> extraction subdir name
    <maybe-images.zip>: images
    <maybe-pdfs.zip>: pdfs
```

Typical bootstrap usage:
- Per-dataset: `DATASETS_ROOT=/mnt/datasets ./datasets/<dataset-name>/bootstrap.sh --yes` to create or refresh the `source-data` link and perform optional extraction.
- Workspace orchestrator: `./bootstrap.sh --yes` from repo root to run dataset then model bootstraps.
- Manual symlink: `ln -s /mnt/datasets/<external-dataset-subdir-or-snapshot> datasets/<dataset-name>/source-data`.

Expected tree under `source-data/` (example):

```
source-data/
├── images/ or images.zip
├── pdfs/ or pdfs.zip
├── README.md
├── README_EN.md
├── metafile.yaml
└── <metadata-or-index>.json
```

Notes:
- If only archives are present (e.g., `images.zip`, `pdfs.zip`), configure `zips` and `unzip_targets` and let the bootstrap script extract them once so downstream glob patterns like `images/*.png` work out of the box.
