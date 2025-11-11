# Decentralized Bootstrap Scripts (Per-Component Symlinks)

## Purpose
- Prefer small, component-scoped bootstrap scripts living next to assets
  (e.g., `datasets/<name>/bootstrap.sh`, `models/bootstrap.sh`).
- Each script reads a local `bootstrap.yaml`, creates a repo symlink, and may do
  light prep (e.g., zip extraction for datasets).
- Keep UX consistent for interactive and batch usage.

## Configuration (YAML)
- Minimal, per-component schemas parsed with `yq`.

### Dataset example (`datasets/omnidocbench/bootstrap.yaml`)
- Keys:
  - `env.data_root_env`: env var for base datasets root (e.g., `DATASETS_ROOT`).
  - `env.default_data_root`: fallback base (e.g., `/workspace/datasets`).
  - `dataset.source_subdir`: directory under base (e.g., `OpenDataLab___OmniDocBench`).
  - `dataset.repo_link_name`: repo-local link name (e.g., `source-data`).
  - `dataset.zips`: list of zips to detect (e.g., `images.zip`, `pdfs.zip`).
  - `dataset.unzip_targets`: map zip->expected subdir name. Used to decide if
    extraction is needed; extraction occurs in-place at the dataset root.

```yaml
env:
  data_root_env: DATASETS_ROOT
  default_data_root: /workspace/datasets
dataset:
  source_subdir: OpenDataLab___OmniDocBench
  repo_link_name: source-data
  zips: [images.zip, pdfs.zip]
  unzip_targets:
    images.zip: images
    pdfs.zip: pdfs
```

### Model example (`models/bootstrap.yaml`)
- Keys:
  - `env.model_root_env`: env var for base models root (e.g., `HF_SNAPSHOTS_ROOT`).
  - `env.default_model_root`: fallback base.
  - `model.source_subdir`: directory under base (e.g., HF snapshot hash).
  - `model.repo_link_name`: repo-local link name (e.g., `deepseek-ocr`).
  - `model.required_files`: optional quick sanity list.

```yaml
env:
  model_root_env: HF_SNAPSHOTS_ROOT
  default_model_root: /home/user/.cache/huggingface/hub/models--org--proj/snapshots
model:
  source_subdir: <snapshot-hash>
  repo_link_name: deepseek-ocr
  required_files:
    - config.json
    - tokenizer.json
    - model.safetensors.index.json
```

## Behavior

### Candidate discovery
- Derive target from env var (preferred) or default base.
- Print base, target, and repo symlink path.

### Prompting + validation
- Ask before replacing an existing symlink (auto-yes with `--yes`).
- Models: optionally validate `required_files` and warn on missing.

### Zip extraction (datasets only)
- If the corresponding subdir exists and is non-empty, do nothing.
- If missing/empty and the zip exists, prompt to extract; extract in-place to the dataset root (`unzip -d <root>`).

### Non‑interactive mode
- `--yes`: accept defaults, auto-overwrite links, and auto-extract if needed.

### Cleanup mode
- `--clean`: remove established symlinks without touching source data.
  - Idempotent: succeeds even if the link does not exist.
  - Skips discovery/validation/extraction.
  - If your script also creates internal symlinks (e.g., a dev-copy), clean those as well.

## Implementation Notes
- Bash (`set -euo pipefail`).
- Dependencies: `yq`, `ln`, `mkdir`, `realpath`; datasets may also use `unzip`.
- Optional ANSI colors; respect `NO_COLOR`.
- Reuse helpers like `require_cmd` and `confirm`.

## CLI Examples
- Run all: `./bootstrap.sh --yes`
- Dataset only: `datasets/omnidocbench/bootstrap.sh --yes`
- Model only: `models/bootstrap.sh --yes`
- Cleanup links: add `--clean` (idempotent)

## Bash Examples

### Dataset bootstrap skeleton
```bash
#!/usr/bin/env bash
set -euo pipefail

require_cmd() { for c in "$@"; do command -v "$c" >/dev/null || { echo "missing: $c" >&2; exit 127; }; done; }
require_cmd yq realpath ln mkdir
command -v unzip >/dev/null && HAVE_UNZIP=true || HAVE_UNZIP=false

ASSUME_YES=false; CLEAN_ONLY=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes) ASSUME_YES=true; shift;;
    --clean) CLEAN_ONLY=true; shift;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="$SCRIPT_DIR/bootstrap.yaml"

DATA_ROOT_ENV=$(yq -r '.env.data_root_env' "$CFG")
DEFAULT_DATA_ROOT=$(yq -r '.env.default_data_root' "$CFG")
SRC_SUBDIR=$(yq -r '.dataset.source_subdir' "$CFG")
LINK_NAME=$(yq -r '.dataset.repo_link_name' "$CFG")

set +u; BASE=${!DATA_ROOT_ENV-}; set -u; BASE="${BASE:-$DEFAULT_DATA_ROOT}"
TARGET="$BASE/$SRC_SUBDIR"
LINK_PATH="$SCRIPT_DIR/$LINK_NAME"

if $CLEAN_ONLY; then
  echo "Cleaning dataset link: $LINK_PATH"
  if [[ -L "$LINK_PATH" ]]; then rm -f -- "$LINK_PATH"; elif [[ -e "$LINK_PATH" ]]; then echo "exists but not a symlink: $LINK_PATH" >&2; else echo "already clean"; fi
  exit 0
fi

echo "Dataset bootstrap: $LINK_PATH -> $TARGET"
mkdir -p "$(dirname "$LINK_PATH")"
if [[ -e "$LINK_PATH" || -L "$LINK_PATH" ]]; then
  if $ASSUME_YES; then rm -rf -- "$LINK_PATH"; else read -r -p "Replace link? [y/N]: " a; [[ ${a,,} == y* ]] || exit 0; rm -rf -- "$LINK_PATH"; fi
fi
ln -s -- "$TARGET" "$LINK_PATH"

if $HAVE_UNZIP; then
  mapfile -t ZIPS < <(yq -r '.dataset.zips[]?' "$CFG" 2>/dev/null || true)
  for z in "${ZIPS[@]}"; do
    subdir=$(yq -r --arg z "$z" '.dataset.unzip_targets[$z] // ""' "$CFG" 2>/dev/null || true)
    [[ -z "$subdir" ]] && subdir="${z%.zip}"
    dest="$TARGET/$subdir"
    [[ -d "$dest" && -n "$(find "$dest" -mindepth 1 -print -quit 2>/dev/null)" ]] && continue
    if [[ -f "$TARGET/$z" ]]; then
      if $ASSUME_YES || { read -r -p "Extract $z in $TARGET? [y/N]: " a; [[ ${a,,} == y* ]]; }; then
        unzip -q -o "$TARGET/$z" -d "$TARGET"
      fi
    fi
  done
fi
```

### Model bootstrap skeleton
```bash
#!/usr/bin/env bash
set -euo pipefail

require_cmd() { for c in "$@"; do command -v "$c" >/dev/null || { echo "missing: $c" >&2; exit 127; }; done; }
require_cmd yq realpath ln mkdir

ASSUME_YES=false; CLEAN_ONLY=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes) ASSUME_YES=true; shift;;
    --clean) CLEAN_ONLY=true; shift;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="$SCRIPT_DIR/bootstrap.yaml"

ROOT_ENV=$(yq -r '.env.model_root_env' "$CFG")
DEF_ROOT=$(yq -r '.env.default_model_root' "$CFG")
SUBDIR=$(yq -r '.model.source_subdir' "$CFG")
LINK=$(yq -r '.model.repo_link_name' "$CFG")

set +u; BASE=${!ROOT_ENV-}; set -u; BASE="${BASE:-$DEF_ROOT}"
TARGET="$BASE/$SUBDIR"; LINK_PATH="$SCRIPT_DIR/$LINK"

if $CLEAN_ONLY; then
  echo "Cleaning model link: $LINK_PATH"
  if [[ -L "$LINK_PATH" ]]; then rm -f -- "$LINK_PATH"; elif [[ -e "$LINK_PATH" ]]; then echo "exists but not a symlink: $LINK_PATH" >&2; else echo "already clean"; fi
  exit 0
fi

echo "Model bootstrap: $LINK_PATH -> $TARGET"
mkdir -p "$(dirname "$LINK_PATH")"
if [[ -e "$LINK_PATH" || -L "$LINK_PATH" ]]; then
  if $ASSUME_YES; then rm -rf -- "$LINK_PATH"; else read -r -p "Replace link? [y/N]: " a; [[ ${a,,} == y* ]] || exit 0; rm -rf -- "$LINK_PATH"; fi
fi
ln -s -- "$TARGET" "$LINK_PATH"

if yq -e '.model.required_files' "$CFG" >/dev/null 2>&1; then
  mapfile -t REQ < <(yq -r '.model.required_files[]' "$CFG")
  miss=(); for f in "${REQ[@]}"; do [[ -e "$TARGET/$f" ]] || miss+=("$f"); done
  ((${#miss[@]})) && echo "Warning: missing in target: ${miss[*]}" >&2 || echo "OK: required present"
fi
```
