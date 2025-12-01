# LLM Profiling Project Structure Guide

This guide provides a comprehensive reference structure for a systematic LLM profiling project using **Pixi** (with `pyproject.toml` as the manifest), **Hydra+OmegaConf** for configuration management, and comprehensive profiling tools (PyTorch profiler + NVTX + NVML + Nsight Systems/Compute).

## About This Guide

**Important**: All code examples, configuration patterns, and directory structures in this guide are **recommendations**, not requirements. Your project should adapt and modify these patterns to fit your specific needs, constraints, and preferences. Use this as a starting point and reference, not as a rigid specification.

### Placeholder Naming Convention

Throughout this guide, placeholder names are shown in angle brackets:
- `<your-project>` or `<your-project-name>`: Your project name (e.g., `llm-profiler`, `gpu-bench`)
- `<your_project>`: Your Python package name using underscores (e.g., `llmprof`, `gpu_bench`)
- Replace these placeholders with your actual project/package names when implementing

---

# Complete Project Layout

```
<your-project>/
├── pyproject.toml                 # Pixi env + tasks + project metadata
├── bootstrap.sh                   # Optional workspace orchestrator (calls component bootstraps)
├── conf/                          # Hydra config tree (config groups)
│   ├── config.yaml                # top-level defaults list
│   ├── hydra/
│   ├── model/                     # Model configs grouped by arch/infer variants
│   │   └── <model-name>/
│   │       ├── arch/
│   │       └── infer/
│   ├── dataset/                   # Dataset configs (roots, subsets, sampling)
│   ├── runtime/                   # Framework/runtime layer
│   ├── hardware/                  # GPU affinity & caps
│   └── profiling/                 # Profiler toggles & presets
├── models/                        # Model weights/tokenizers (symlinks or submodules)
│   ├── bootstrap.sh               # Model bootstrap (creates <model-name> symlink)
│   ├── bootstrap.yaml             # Model bootstrap configuration
│   └── <processed>/               # Optional processed artifacts
├── datasets/                      # Dataset organization
│   └── <dataset-name>/
│       ├── bootstrap.sh           # Creates `source-data` symlink (and optional prep)
│       ├── bootstrap.yaml         # Dataset bootstrap configuration
│       ├── source-data -> /path/to/<dataset>/   # Symlink to data on host
│       ├── README.md              # Dataset documentation
│       ├── metadata.yaml          # (optional) dataset facts
│       └── <variant-name>/        # (optional) preprocessed/filtered variants
├── third_party/                   # Reference sources (read-only; symlinks/submodules)
│   ├── github/
│   └── hf/
├── src/
│   └── <your_project>/            # Python package
│       ├── profiling/             # harness, exporters, NVTX/NSYS/NCU helpers
│       │   └── parsers/
│       ├── runners/               # CLI runners used by Hydra entries
│       └── data/                  # dataset utilities
├── scripts/                       # Utility scripts (prep, analysis, small tools)
├── tests/                         # Unit/integration/manual tests
├── docs/                          # Documentation and guides
├── context/                       # Knowledge base, hints, how-tos
├── magic-context/                 # Templates/snippets (submodule or folder)
└── .specify/                      # Project constitution/templates (optional)
```

## Design Rationale

* **Hydra** grouped configs under `conf/<group>/<option>.yaml` enable composable, swappable configurations with top-level `defaults` list; the templated output/run directory pattern (e.g., `runs/${experiment}/${model.name}/…`) ensures every run is self-contained and reproducible. ([Hydra][1])
* **Pixi** can use **`pyproject.toml`** as a single manifest with **tasks** for common operations like `pixi run profile`. ([Prefix Dev][2])
* **`third_party/`** provides a clear boundary for vendored reference code. It can
  be a collection of symlinks to local checkouts or Git submodules/subtrees; the
  key is treating it as read-only.
* **`models/`** can be symlinks to external storage or tracked via Git submodules
  when using small repos/LFS pointers. Prefer symlinks for large weight files.
* **`datasets/` structure** organizes both original data (via symlinks) and derived variants (subsets, preprocessed versions) in a consistent, documented manner, with metadata and per-dataset documentation making it easy to track provenance and transformations.

# Configuration Files

These are example configurations demonstrating the recommended patterns. Customize them to match your project's tooling, dependencies, and workflow preferences.

## `pyproject.toml` (what it is and a minimal example)

What it is: single manifest for project metadata, dependencies, Pixi workspace, and convenient tasks (so you can run `pixi run <task>`).

```toml
[project]
name = "<your-project-name>"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["torch", "omegaconf", "hydra-core"]

[tool.pixi.workspace]
channels  = ["conda-forge"]
platforms = ["linux-64"]

[tool.pixi.tasks]
# Replace <your command> with your actual CLI
profile = "python -m <your_project>.cli experiment=baseline profiling=minimal"
nsys    = "nsys profile -o runs/nsys <your command>"
ncu     = "ncu -o runs/ncu <your command>"
```

Keep tasks short and descriptive; add more as needed. ([Prefix Dev][2])

## Hydra Configuration Structure

### `conf/config.yaml` (what it is: top‑level defaults/composition)

```yaml
# Compose from config groups (swap via CLI)
defaults:
  - hydra: default
  - model/<model-name>/arch: <model-name>.default
  - model/<model-name>/infer: <model-name>.default
  - runtime: pytorch
  - profiling: minimal
  - _self_
experiment: baseline
```

Hydra's **defaults list** composes configs from groups; you can swap any piece via CLI overrides (e.g., `model=llama3_70b dataset=c4 profiling=full`). ([Hydra][3])

### `conf/hydra/default.yaml` (what it is: run directory & job behavior)

```yaml
hydra:
  run:
    dir: runs/${experiment}/${model.name}/${now:%Y-%m-%d_%H-%M-%S}
  job:
    name: ${experiment}
    chdir: true
```

These fields control where Hydra writes `.hydra/` configs, logs, and where your code runs (CWD). ([Hydra][4])

### `conf/model/<model-name>/arch/<model-name>.<arch-variant>.yaml` (architecture & preprocessing)

```yaml
model:
  name: qwen2_5_7b
  path: ${hydra:runtime.cwd}/models/qwen2_5_7b   # symlink, folder, or submodule
  dtype: bf16
  preprocess:
    enable: true
    base_size: 1024
    image_size: 640
    crop_mode: true
    patch_size: 16
    downsample_ratio: 4
```

### `conf/model/<model-name>/infer/<model-name>.<inference-variant>.yaml` (inference defaults)

```yaml
infer:
  temperature: 0.0
  max_new_tokens: 8192
  no_repeat_ngram_size: 20
  do_sample: false
```

### `conf/dataset/<name>.yaml` (what it is: dataset root, subset, sampling)

```yaml
name: openwebtext
root: ${hydra:runtime.cwd}/datasets/openwebtext/source-data
subset_filelist: null  # or a repo-relative filelist
sampling:
  num_epochs: 1
  num_samples_per_epoch: 3
  randomize: false
```

### `conf/runtime/pytorch.yaml` (what it is: runtime and basic parameters)

```yaml
type: pytorch
batch_size: 1
num_new_tokens: 64
torch_profiler: { enabled: false }
```

### `conf/hardware/single_gpu.yaml` (what it is: device selection)

```yaml
device_index: 0
```

### `conf/profiling/minimal.yaml` (what it is: profiler toggles)

```yaml
nsys: { enable: false }
ncu:  { enable: false }
nvml: { enable: true }
```

### `conf/profiling/full.yaml` (what it is: enable profilers with options)

```yaml
nsys: { enable: true, args: "<nsys args>" }
ncu:  { enable: true }
nvml: { enable: true }
```

# Datasets Directory Structure

The `datasets/` directory organizes both original datasets and their derived variants in a consistent, well-documented structure.

## Directory Organization Pattern

Each dataset follows this structure:

```
datasets/
└── <dataset-name>/
    ├── source-data -> /path/to/original/dataset  # Symlink to the official/downloaded dataset
    ├── metadata.yaml                              # Dataset metadata
    ├── README.md                                  # Documentation
    └── <variant-name>/                            # Processed variants (optional)
        ├── data/                                  # Variant data files
        └── metadata.yaml                          # Variant-specific metadata (optional)
```

## Key Files and Their Purposes

### `source-data` (symlink)
- Points to the actual location of the downloaded/official dataset
- Keeps large data files outside the repo while maintaining easy access
- Allows different environments to point to different storage locations

### Bootstrapping Symlinks (Cross-Host)

Prefer decentralized bootstraps so each component owns its linking logic:

- Dataset bootstrap (per dataset):
  - `datasets/<dataset-name>/bootstrap.sh` reads `datasets/<dataset-name>/bootstrap.yaml`.
  - Resolves a base directory from an env var (e.g., `$DATASETS_ROOT`) with a default fallback.
  - Creates `source-data` symlink. Optionally prepares data (e.g., extracting `*.zip` in-place) when needed.
  - Example:
    - `DATASETS_ROOT=/mnt/datasets ./datasets/<dataset-name>/bootstrap.sh --yes`

- Model bootstrap (all models in one place):
  - `models/bootstrap.sh` reads `models/bootstrap.yaml`.
  - Resolves a base directory from an env var (e.g., `$HF_SNAPSHOTS_ROOT` or `$MODELS_ROOT`).
  - Creates the model symlink (e.g., `models/<model-name>`). Optionally validates required files.
  - Example:
    - `HF_SNAPSHOTS_ROOT=/nvme/hf-cache ./models/bootstrap.sh --yes`

- Workspace orchestrator (optional):
  - `./bootstrap.sh` can call component bootstraps in sequence (datasets first, then models) for convenience.
  - Keep it thin; the source of truth lives next to each component.

### Symlink Policy

- Do not track or commit symlinks that point to external storage (e.g., host datasets, HF cache, mounted volumes). These links are environment‑specific and must be created by the bootstrap scripts on every developer/CI host.
- If a symlink’s target resides inside the repository (rare; e.g., an internal layout alias), it may be versioned, but prefer real directories. Document the rationale in the local README.
- Datasets: never commit `source-data` symlinks; they are created by `datasets/<dataset>/bootstrap.sh`.
- Models: do not commit `models/<model-name>` symlinks; they are created by `models/bootstrap.sh`.

### `metadata.yaml` (what it is: dataset facts for provenance)

Short example:

```yaml
name: openwebtext
version: "1.0"
source: https://huggingface.co/datasets/openwebtext
splits: { train: 8013769 }
notes: "Downloaded with HF datasets; source-data is read-only."
```

### `README.md`
Should document:
- Dataset overview and purpose
- Source and licensing information
- How to obtain/download the original data
- Available variants and their purposes
- Any preprocessing steps or transformations applied
- Usage examples

- First-level directory tree of `source-data/` (top-level only), for quick
  orientation. Example:

  ```
  source-data/
  ├── images/
  ├── pdfs/
  ├── README.md
  ├── README_EN.md
  ├── metafile.yaml
  └── OmniDocBench.json
  ```

### Variant directories (e.g., `subset-1k/`, `tokenized-gpt2/`)
Store processed versions of the dataset:
- **Subsets**: Smaller samples for quick testing (e.g., `subset-1k/`, `dev-split/`)
- **Preprocessed**: Tokenized, normalized, or otherwise transformed data (e.g., `tokenized-gpt2/`, `cleaned/`)
- **Format conversions**: Different file formats or organizations (e.g., `parquet/`, `tfrecord/`)

Each variant can have its own `metadata.yaml` documenting:
```yaml
parent: source-data
created: "2024-01-20"
transform: "Random sample of 1000 examples from training set"
size:
  samples: 1000
  disk_mb: 47.5
```

### Git Ignore for External Source Data

Keep external dataset content out of version control by ignoring `source-data`
symlinks/directories at the dataset root. Place the following in
`datasets/.gitignore`:

```gitignore
*/source-data
*/source-data/**
```

Similarly, you can ignore top‑level model links (optional) if contributors sometimes accidentally add them:

```gitignore
# models/.gitignore
/*
!.gitignore
!README.md
!bootstrap.sh
!bootstrap.yaml
```

## Example: Complete Dataset Entry

```
datasets/openwebtext/
├── source-data -> /data/datasets/openwebtext/
├── metadata.yaml
├── README.md
├── subset-1k/                    # Small subset for testing
│   ├── data/
│   │   └── samples.jsonl
│   └── metadata.yaml
├── tokenized-gpt2/               # Preprocessed with GPT-2 tokenizer
│   ├── data/
│   │   ├── train.bin
│   │   └── val.bin
│   └── metadata.yaml
└── analysis/                     # Optional: analysis artifacts
    ├── statistics.json
    └── samples.html
```

## Dataset Management Workflows

### Adding a new dataset

1. Download/obtain the dataset to your storage location (e.g., `/data/datasets/mydataset/`)
2. Create dataset directory: `mkdir -p datasets/mydataset`
3. Prefer running the dataset bootstrap:
   - `DATASETS_ROOT=/data/datasets ./datasets/<dataset-name>/bootstrap.sh`
   - Or create a symlink manually: `ln -s /data/datasets/mydataset datasets/mydataset/source-data`
4. Create `metadata.yaml` with dataset information
5. Create `README.md` with documentation
6. Add corresponding Hydra config: `conf/dataset/mydataset.yaml`

### Creating a variant

1. Run preprocessing script: `python scripts/prepare_dataset.py --dataset openwebtext --variant subset-1k`
2. The script creates `datasets/openwebtext/subset-1k/` with processed data
3. Optionally create variant metadata: `datasets/openwebtext/subset-1k/metadata.yaml`
4. Use in Hydra config: `dataset=openwebtext dataset.variant=subset-1k`

### Recommended scripts (`scripts/prepare_dataset.py` example structure)

```python
# scripts/prepare_dataset.py
import argparse
from pathlib import Path
import yaml

def create_subset(dataset_root: Path, variant_name: str, num_samples: int):
    """Create a subset variant of a dataset."""
    source = dataset_root / "source-data"
    variant_dir = dataset_root / variant_name
    variant_dir.mkdir(exist_ok=True)

    # Your subsetting logic here
    # ...

    # Write metadata
    metadata = {
        "parent": "source-data",
        "created": datetime.now().isoformat(),
        "transform": f"Random sample of {num_samples} examples",
        "size": {"samples": num_samples}
    }
    with open(variant_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    args = parser.parse_args()

    dataset_root = Path("datasets") / args.dataset
    create_subset(dataset_root, args.variant, args.num_samples)
```

## Best Practices

1. **Keep source data read-only**: Never modify files under `source-data/`
2. **Document transformations**: Each variant should have clear documentation of what was done
3. **Use consistent naming**: Follow patterns like `subset-{size}`, `tokenized-{tokenizer}`, `split-{split_name}`
4. **Track provenance**: Include creation dates, parent references, and transformation details in metadata
5. **Symlink external storage**: Keep large files outside the repo, use symlinks for access
6. **Version control metadata**: Commit `metadata.yaml` and `README.md`, but typically `.gitignore` the actual data files

# Code Structure

The following are example implementations showing recommended patterns. Your project should implement these according to its specific requirements.

## Core Implementation Files

### `src/<your_project>/profiling/harness.py`

Drop in the harness we built earlier (NVTX context manager, `run_torch_profiler`, `NVMLSampler`, and the "advise NSYS/NCU command" helpers). PyTorch Profiler & NVTX usage matches the official docs/workflows. ([PyTorch Docs][5])

### `src/<your_project>/runners/base.py`

```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class Runner(ABC):
    def __init__(self, cfg: Dict):
        self.cfg = cfg

    @abstractmethod
    def prefill(self, prompt_ids): ...
    @abstractmethod
    def decode(self, num_new_tokens: int): ...
```

### `src/<your_project>/runners/pytorch_runner.py` (sketch)

```python
import torch
from .base import Runner

class PyTorchRunner(Runner):
    def __init__(self, cfg):
        super().__init__(cfg)
        # TODO: load your model from cfg.model.path
        self.device = f"cuda:{cfg.hardware.device_index}"
        # self.model = ...
        # self.tokenizer = ...

    def prefill(self, prompt_ids):
        # with torch.inference_mode(): ...
        pass

    def decode(self, n_new):
        for _ in range(n_new):
            # single-step decode; use NVTX in caller
            pass
```

### `src/<your_project>/cli.py` (Hydra entry + profiling glue)

```python
import os
import hydra
from omegaconf import DictConfig, OmegaConf

from <your_project>.profiling.harness import (
    nvtx_range, NVMLSampler, run_torch_profiler, prof_to_csv
)

def build_runner(cfg):
    if cfg.runtime.type == "pytorch":
        from <your_project>.runners.pytorch_runner import PyTorchRunner
        return PyTorchRunner(cfg)
    # elif cfg.runtime.type == "tensorrtllm": ...
    raise ValueError(f"Unknown runtime: {cfg.runtime.type}")

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    os.environ.setdefault("PROJECT_ROOT", os.getcwd())

    runner = build_runner(cfg)

    nvml = None
    if cfg.profiling.nvml.enable:
        nvml = NVMLSampler(device_index=cfg.hardware.device_index,
                           interval_s=cfg.profiling.nvml.interval_s,
                           out_csv="nvml.csv")
        nvml.start()

    def one_step(_):
        with nvtx_range("prefill"): runner.prefill(prompt_ids=None)
        with nvtx_range("decode"):  runner.decode(cfg.runtime.num_new_tokens)

    if getattr(cfg.runtime, "torch_profiler", {}).get("enabled", False):
        prof = run_torch_profiler(
            one_step,
            steps=cfg.runtime.torch_profiler.steps,
            warmup=cfg.runtime.torch_profiler.warmup,
            out_dir="tb"
        )
        if cfg.profiling.outputs.write_ops_csv:
            prof_to_csv(prof, "ops.csv")
    else:
        # plain run (still places NVTX ranges for NSYS)
        one_step(0)

    if nvml:
        nvml.stop(); nvml.join()

if __name__ == "__main__":
    main()
```

This keeps your **end-to-end phases NVTX-annotated** for Nsight Systems (`--capture-range=nvtx`) and dumps a **PyTorch Profiler** Chrome trace + `ops.csv` when enabled. ([NVIDIA Docs][6])

### `scripts/snapshot_hf.py`

```python
# Utility to snapshot HF repos into third_party/hf/ (source files only, no weights)
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5",
    local_dir="third_party/hf/Qwen/Qwen2.5",
    allow_patterns=["*.py", "LICENSE*"],  # keep only source, not big weights
    revision="main"                        # or specific commit/revision
)
```

# Managing External Dependencies (Reference Source Code)

The `third_party/` directory houses upstream model implementations as read-only reference code. Here are four recommended approaches to manage these dependencies. Choose the approach that best fits your workflow:

## Option 1: Git Submodules (Recommended for Pinning)

**Add & pin reference repos:**

```bash
git submodule add https://github.com/huggingface/transformers third_party/github/transformers
git submodule add https://github.com/vllm-project/vllm      third_party/github/vllm
git submodule update --init --recursive

# Pin to a specific commit for reproducibility
git -C third_party/github/transformers checkout <commit-sha>
git -C third_party/github/vllm checkout <commit-sha>
git add .gitmodules third_party/github
git commit -m "Add reference sources as submodules"
```

**Hide local edits in status (reference-only behavior):**

```bash
git config -f .gitmodules submodule.third_party/github/transformers.ignore dirty
git config -f .gitmodules submodule.third_party/github/vllm.ignore dirty
git add .gitmodules && git commit -m "Ignore dirty submodule worktrees"
```

**Shallow/fast clones for large repos:**

```bash
git submodule update --init --depth 1 -- third_party/github/transformers
```

([Git Submodules][1], [Git Documentation][3])

## Option 2: Git Subtree (Normal Folder, No Submodule UX)

```bash
git subtree add --prefix third_party/github/transformers \
  https://github.com/huggingface/transformers main --squash

# Later update:
git subtree pull --prefix third_party/github/transformers \
  https://github.com/huggingface/transformers main --squash
```

Subtrees behave like regular directories with no `.gitmodules`, at the cost of larger superproject history. ([Debian Manpages][4])

## Option 3: Sparse-Checkout (For Parts of Giant Monorepos)

If you only need specific paths (e.g., `src/transformers/models/llama/`):

```bash
git clone https://github.com/huggingface/transformers third_party/github/transformers
cd third_party/github/transformers
git sparse-checkout init --cone
git sparse-checkout set src/transformers/models/llama
```

This keeps your working tree tiny while the repo remains complete. ([Git Sparse-Checkout][5])

## Option 0: Symlinks (Fastest Pointer)

If you already have local checkouts or shared code on disk, create simple
symlinks under `third_party/`:

```bash
ln -s /opt/src/transformers third_party/github/transformers
ln -s /opt/src/vllm         third_party/github/vllm
```

Pros: no Git metadata in your repo, very quick to set up. Cons: not portable for
collaborators unless they replicate the layout.

## Option 4: Hugging Face Hub Snapshots (No Git History)

For model repos on the Hub that include reference implementation files alongside weights:

```python
# scripts/snapshot_hf.py
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5",
    local_dir="third_party/hf/Qwen/Qwen2.5",
    allow_patterns=["*.py", "LICENSE*"],  # keep only source, not big weights
    revision="main"                        # or specific commit/revision
)
```

This uses HF's content-addressed cache and avoids Git LFS entirely. ([Hugging Face Docs][6])

## Guard Rails (Optional but Handy)

* **Pre-commit**: Add hooks so contributors don't accidentally edit `third_party/**` or add new submodules without review. ([pre-commit.com][8])
* **.gitmodules defaults**: Set `submodule.<name>.update = none` for truly frozen submodules, and `submodule.<name>.ignore = dirty` for quiet status. ([Git gitmodules][9])
* **Licenses**: Copy upstream `LICENSE` files alongside the referenced code to keep compliance obvious.

## Which Option When?

* **You want history + easy pinning** → **Submodules** (default)
* **You want a normal folder, no special Git UX** → **Subtree**
* **You only need a slice of a big monorepo** → **Sparse-checkout**
* **You just want reference `.py` files from HF, no weights** → **Hub snapshot**

# Typical Workflows

These are example workflows using the recommended structure. Adapt commands and patterns to your project's needs.

## Profiling Operations

**Plain run (minimal stats):**

```bash
pixi run profile
```

**Nsight Systems timeline** (NVTX-gated):

```bash
pixi run nsys
```

Uses `--capture-range=nvtx` so only your labeled region is captured. ([NVIDIA Docs][6])

**Nsight Compute deep-dive** (kernel metrics for roofline/MFU):

```bash
pixi run ncu
```

Tweak metrics/filters as you focus on attention/GEMMs. ([NVIDIA Docs][7])

**Swap models/runtimes via Hydra overrides:**

```bash
pixi run profile model=llama3_70b runtime=vllm profiling=full
```

## External Dependency Management

**Initialize submodules:**

```bash
pixi run externals:init
```

**Update submodules:**

```bash
pixi run externals:update
```

**Snapshot HF reference code:**

```bash
pixi run externals:snapshot
```

# Key Design Principles

These principles are **recommendations** based on common patterns. Adapt them to your project's specific requirements:

* **Config groups** provide composability: patterns like `profiling/{minimal,full,roofline-only}`; `hardware/{single_gpu,multi_gpu}`; `runtime/{pytorch,vllm,tensorrtllm}`; `model/*`; `dataset/*` can be mixed and matched. Hydra's defaults list enables reproducible runs and easy parameter sweeps. ([Hydra][1])
* **Run directory templates** (via `hydra.run.dir`) can organize all artifacts—NVML CSV, `ops.csv`, TensorBoard traces, NSYS `.qdrep`, NCU reports—under structured paths like `runs/<exp>/<model>/<timestamp>`. ([Hydra][4])
* **Reference code pinning** via `third_party/` (using submodules/subtrees/snapshots) keeps experiments reproducible and audit-friendly by tracking exact upstream versions.
* **Symlinked external assets**: Both `models/` (weights) and `datasets/` (data) use symlinks to external storage, decoupling the repo from large binary files and enabling flexible storage management.
* **Dataset variants** provide a structured way to maintain original data alongside preprocessed versions, subsets, and transformations, with metadata tracking provenance.
* **Optional monitoring**: For continuous node monitoring, tools like **DCGM Exporter** (Prometheus `/metrics`) complement per-run profiling artifacts with fleet-level GPU telemetry. ([NVIDIA Docs][8])

# Summary

This guide presents a comprehensive reference structure for LLM profiling projects. Remember:

- **All patterns are recommendations**: Adapt the structure, configurations, and workflows to your specific needs
- **Key organizational principles**:
  - Use Hydra for composable, reproducible configurations
  - Symlink external assets (models, datasets) to decouple from large files
  - Track reference code in `third_party/` for reproducibility
  - Organize datasets with source data + documented variants
  - Structure runs with templated output directories
- **Flexibility is essential**: Your project may require different tools, different structures, or different workflows—that's expected and encouraged

Use this as a starting point and evolve it based on your requirements, team preferences, and constraints.

# References

[1]: https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/?utm_source=chatgpt.com "Grouping config files"
[2]: https://prefix-dev.github.io/pixi/dev/python/pyproject_toml/ "pyproject.toml - Pixi by prefix.dev"
[3]: https://hydra.cc/docs/advanced/defaults_list/?utm_source=chatgpt.com "The Defaults List"
[4]: https://hydra.cc/docs/configure_hydra/workdir/?utm_source=chatgpt.com "Customizing working directory pattern"
[5]: https://docs.pytorch.org/docs/stable/profiler.html?utm_source=chatgpt.com "torch.profiler — PyTorch 2.9 documentation"
[6]: https://docs.nvidia.com/nsight-systems/UserGuide/index.html?utm_source=chatgpt.com "User Guide — nsight-systems"
[7]: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html?utm_source=chatgpt.com "4. Nsight Compute CLI"
[8]: https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html?utm_source=chatgpt.com "DCGM Exporter — NVIDIA GPU Telemetry 1.0.0 ..."
[9]: https://git-scm.com/docs/gitmodules?utm_source=chatgpt.com "Git - gitmodules Documentation"
