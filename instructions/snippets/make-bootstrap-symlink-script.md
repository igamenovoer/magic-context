# Bootstrap Symlink Script Pattern (Data Linking)

## Purpose
- Provide a reusable pattern for a shell script that bootstraps symlinks from a project repo into external data directories.
- Optimize UX for both interactive and non-interactive runs.

## Configuration (YAML)
- Use a small YAML schema parsed with `yq` to avoid hardcoding paths:
  - `env.data_root_env`: name of the env var for the preferred root (e.g., `CONTAINER_DATA_ROOT`).
  - `env.default_data_root`: fallback root when the env var is unset (e.g., `/extra/data`).
  - `data_items[]`: each item describes a dataset or asset group:
    - `id`: unique identifier (e.g., `imageset_v1`).
    - `source_dir`: directory name expected under the data root.
    - `repo_link_path`: repo-relative symlink path to create.
    - `required_files[]`: file names that must exist inside `source_dir`.
    - `optional_files[]`: file names that may exist (do not block linking).
    - `procssed_dir`: repo path where processed/derived outputs will be placed (created if missing).

### YAML Example
```yaml
env:
  data_root_env: CONTAINER_DATA_ROOT
  default_data_root: /extra/data
data_items:
  - id: imageset_v1
    source_dir: imageset_v1
    repo_link_path: datasets/imageset_v1
    required_files: [manifest.json]
    optional_files: [classes.txt]
    procssed_dir: datasets/processed/imageset_v1
```

## Behavior

### 1) Candidate Discovery + Status Printing
- Compute two candidates per item: `/${env.data_root_env}/${source_dir}` (if env set) and `/${env.default_data_root}/${source_dir}`.
- Print exactly what is checked, using absolute paths and explicit status:
  - `DIR <abs-path> [OK|MISSING]`
  - For `DIR=OK`, print one `FILE <abs-path> [OK|MISSING]` line per required file.

### 2) Expected Layout Preview
- Print a minimal directory tree built from YAML:
  - `<source_dir>/`
    - `<required_file> (required)`
    - `<optional_file> (optional)`

### 3) Prompting + Validation
- Prompt on a new line with a short label (e.g., `Path:`).
- Trim whitespace.
- Empty input:
  - If a valid candidate was detected: accept it.
  - Else: skip this item.
- Non-empty input: normalize to absolute path and validate that `DIR` exists and all `required_files` exist; otherwise print the specific issue and re‑prompt.

### 4) Non‑Interactive Mode
- `--yes`/`-y`: accept detected defaults automatically; auto‑overwrite conflicting links.
- `--strict` (with `--yes` only): if any item has no valid detected path, exit non‑zero so callers can fail fast.

### 5) Symlink Creation
- Create or update the `repo_link_path` to point to the chosen directory.
- Ask before overwrite (skipped when `--yes`).
- Ensure `procssed_dir` exists.

## Implementation Notes
- **Language**: bash (`set -euo pipefail`).
- **Dependencies**: `yq` (mikefarah), `ln`, `mkdir`, `realpath`.
- **Colors**: optional ANSI colors to highlight headers, statuses `[OK]/[MISSING]`, and actions. Respect `NO_COLOR` to disable.
- **Functions**:
  - `require_cmd`: verify dependencies.
  - `confirm_overwrite`: interactive overwrite with auto‑yes when `--yes`.

## CLI Examples
- Interactive: `bash ./bootstrap-symlinks.sh`
- Non‑interactive, skip missing: `bash ./bootstrap-symlinks.sh --yes`
- Non‑interactive, fail on missing: `bash ./bootstrap-symlinks.sh --yes --strict`

## Acceptance Criteria (Quality Bar)
- Idempotent: re‑running is safe; either no‑ops or updates links per confirmation.
- Clear output: every checked path printed with explicit status.
- Config‑driven: no hardcoded data paths; adapts when YAML changes.

## Bash Examples

### Parse Config, Args, and Setup Colors
```bash
#!/usr/bin/env bash
set -euo pipefail

require_cmd() { for c in "$@"; do command -v "$c" >/dev/null || { echo "missing: $c" >&2; exit 127; }; done; }
require_cmd yq realpath ln mkdir

ASSUME_YES=false; STRICT_FAIL=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes) ASSUME_YES=true; shift;;
    --strict) STRICT_FAIL=true; shift;;
    -h|--help) echo "usage: $0 [--yes] [--strict]"; exit 0;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

RED=""; GREEN=""; YELLOW=""; BLUE=""; BOLD=""; DIM=""; RESET=""
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BLUE=$'\033[34m'; BOLD=$'\033[1m'; DIM=$'\033[2m'; RESET=$'\033[0m'
fi

PACK="data-pack.yaml"
DATA_ROOT_ENV=$(yq -r '.env.data_root_env' "$PACK")
DEFAULT_DATA_ROOT=$(yq -r '.env.default_data_root' "$PACK")
set +u; ENV_DATA_ROOT=${!DATA_ROOT_ENV-}; set -u
SUGGEST_ROOT="${ENV_DATA_ROOT:-$DEFAULT_DATA_ROOT}"

echo "Using data root: ${BOLD}${SUGGEST_ROOT}${RESET} (env ${DATA_ROOT_ENV} preferred)"
```

### Candidate Discovery + Printing Status
```bash
print_checks() {
  local base="$1" srcdir="$2"; shift 2
  local req=("$@")
  local dir_raw="${base%/}/$srcdir"
  local dir_abs="$(realpath -m -- "$dir_raw")"
  if [[ -d "$dir_raw" ]]; then
    echo "  DIR  $dir_abs  [${GREEN}OK${RESET}]"
    for f in "${req[@]}"; do
      local file_raw="$dir_raw/$f"; local file_abs="$(realpath -m -- "$file_raw")"
      [[ -f "$file_raw" ]] && echo "  FILE $file_abs  [${GREEN}OK${RESET}]" \
                            || echo "  FILE $file_abs  [${RED}MISSING${RESET}]"
    done
  else
    echo "  DIR  $dir_abs  [${RED}MISSING${RESET}]"
  fi
}

detected_path=""
detect_valid() {
  local base="$1" srcdir="$2"; shift 2
  local req=("$@")
  local dir_raw="${base%/}/$srcdir"
  [[ -d "$dir_raw" ]] || return 1
  for f in "${req[@]}"; do [[ -f "$dir_raw/$f" ]] || return 1; done
  detected_path="$dir_raw"; return 0
}
```

### Prompting + Validation Loop
```bash
read_one_item() {
  local id="$1" srcdir="$2" linkpath="$3" processed="$4"; shift 4
  local req=("$@")

  echo "\n${BOLD}=== $id ===${RESET}"
  echo "Candidates:"
  if [[ -n "${ENV_DATA_ROOT:-}" ]]; then
    print_checks "$ENV_DATA_ROOT" "$srcdir" "${req[@]}"
  else
    echo "  ${DATA_ROOT_ENV} is unset"
  fi
  print_checks "$DEFAULT_DATA_ROOT" "$srcdir" "${req[@]}"

  detected_path=""; if [[ -n "${ENV_DATA_ROOT:-}" ]]; then detect_valid "$ENV_DATA_ROOT" "$srcdir" "${req[@]}" || true; fi
  if [[ -z "$detected_path" ]]; then detect_valid "$DEFAULT_DATA_ROOT" "$srcdir" "${req[@]}" || true; fi

  echo "Expected layout:"; echo "  $srcdir/"; for f in "${req[@]}"; do echo "    - $f (required)"; done

  local chosen=""
  if $ASSUME_YES; then
    if [[ -n "$detected_path" ]]; then
      chosen=$(realpath -m -- "$detected_path"); echo "Using detected: $chosen"
    else
      $STRICT_FAIL && { echo "ERROR: no detected path for $id"; exit 3; } || { echo "Skip $id"; return; }
    fi
  else
    while true; do
      echo "Enter absolute path (Enter=accept detected; blank=skip if none):"; read -r -p "Path: " inp || true
      local trim="$(printf '%s' "$inp" | sed 's/^\s\+//;s/\s\+$//')"
      if [[ -z "$trim" ]]; then
        if [[ -n "$detected_path" ]]; then chosen=$(realpath -m -- "$detected_path"); echo "Using detected: $chosen"; break; else echo "Skip $id"; return; fi
      else
        chosen=$(realpath -m -- "$trim")
      fi
      [[ -d "$chosen" ]] || { echo "Not a directory: $chosen"; continue; }
      local miss=(); for f in "${req[@]}"; do [[ -f "$chosen/$f" ]] || miss+=("$f"); done
      (( ${#miss[@]} == 0 )) || { echo "Missing: ${miss[*]}"; continue; }
      break
    done
  fi

  mkdir -p "$(dirname "$linkpath")"
  if [[ -e "$linkpath" || -L "$linkpath" ]]; then
    if $ASSUME_YES; then rm -rf -- "$linkpath"; else read -r -p "Overwrite $linkpath? (y/N): " ans; [[ ${ans,,} == y* ]] || return; rm -rf -- "$linkpath"; fi
  fi
  ln -s -- "$chosen" "$linkpath"; echo "Linked: $linkpath -> $chosen"
  mkdir -p -- "$processed"; echo "Ensured processed dir: $processed"
}
```

### Iterate YAML Items
```bash
COUNT=$(yq -r '.data_items | length' "$PACK")
for ((i=0; i<COUNT; i++)); do
  id=$(yq -r ".data_items[$i].id" "$PACK")
  src=$(yq -r ".data_items[$i].source_dir" "$PACK")
  link=$(yq -r ".data_items[$i].repo_link_path" "$PACK")
  proc=$(yq -r ".data_items[$i].procssed_dir" "$PACK")
  mapfile -t req < <(yq -r ".data_items[$i].required_files[]" "$PACK")
  read_one_item "$id" "$src" "$link" "$proc" "${req[@]}"
done
```
