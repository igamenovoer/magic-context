#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Create or overwrite WORKDIR/product/ from an existing WORKDIR/.

This generates a deliverable directory:
  WORKDIR/product/
    <project_name>/      (must include pixi.toml or pyproject.toml)
    res/                 (optional; from WORKDIR/resources/)
    pkg-cache/           (from WORKDIR/pixi-cache/)
    envs.sh              (manual env setup)
    bootstrap-project.sh (sources envs.sh then runs pixi install)

Required tools: rsync

Usage:
  make-archive-ready-product.sh --workdir <workdir> [--project-name <name>] [options]

Options:
  --workdir <dir>           WORKDIR created by create-workdir scripts (required)
  --project-name <name>     Project dir name under WORKDIR/project/ (auto-detect if unique)
  --env KEY=VALUE           Adds an extra export to product/envs.sh (repeatable)
  --exclude <pattern>       Additional rsync exclude when copying project into product (repeatable)
  --help                    Show this help

Notes:
  - This script errs on the side of including files. Use --exclude to trim.
  - The Pixi manifest must exist in the product project dir (pixi.toml or pyproject.toml).
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

WORKDIR=""
PROJECT_NAME=""
EXCLUDES=()
EXTRA_ENVS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workdir)
      WORKDIR="${2:-}"; shift 2;;
    --project-name)
      PROJECT_NAME="${2:-}"; shift 2;;
    --exclude)
      EXCLUDES+=("${2:-}"); shift 2;;
    --env)
      EXTRA_ENVS+=("${2:-}"); shift 2;;
    --help|-h)
      usage; exit 0;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      echo "Run with --help for usage." >&2
      exit 2;;
  esac
done

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: Missing required command: $1" >&2; exit 127; }
}
need_cmd rsync

if [[ -z "$WORKDIR" ]]; then
  echo "ERROR: --workdir is required" >&2
  exit 2
fi

if [[ ! -d "$WORKDIR/project" ]]; then
  echo "ERROR: WORKDIR does not look valid (missing $WORKDIR/project)" >&2
  exit 2
fi

if [[ -z "$PROJECT_NAME" ]]; then
  mapfile -t candidates < <(find "$WORKDIR/project" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
  if [[ "${#candidates[@]}" -eq 1 ]]; then
    PROJECT_NAME="${candidates[0]}"
  else
    echo "ERROR: --project-name not provided and could not auto-detect a single project under $WORKDIR/project" >&2
    echo "Found: ${candidates[*]:-(none)}" >&2
    exit 2
  fi
fi

SRC_PROJECT_DIR="$WORKDIR/project/$PROJECT_NAME"
if [[ ! -d "$SRC_PROJECT_DIR" ]]; then
  echo "ERROR: Project not found in WORKDIR: $SRC_PROJECT_DIR" >&2
  exit 2
fi

SRC_MANIFEST_OK="false"
if [[ -f "$SRC_PROJECT_DIR/pixi.toml" || -f "$SRC_PROJECT_DIR/pyproject.toml" ]]; then
  SRC_MANIFEST_OK="true"
fi
if [[ "$SRC_MANIFEST_OK" != "true" ]]; then
  echo "ERROR: Project dir must contain pixi.toml or pyproject.toml: $SRC_PROJECT_DIR" >&2
  exit 2
fi

PRODUCT_DIR="$WORKDIR/product"
rm -rf "$PRODUCT_DIR"
mkdir -p "$PRODUCT_DIR"

RSYNC_EXCLUDES=(
  --exclude '.git/'
  --exclude '.pixi/'
  --exclude '.env'
  --exclude '__pycache__/'
  --exclude '.pytest_cache/'
  --exclude '.mypy_cache/'
  --exclude '.ruff_cache/'
  --exclude '.venv/'
)
for pat in "${EXCLUDES[@]}"; do
  RSYNC_EXCLUDES+=(--exclude "$pat")
done

mkdir -p "$PRODUCT_DIR/$PROJECT_NAME"
rsync -a --delete "${RSYNC_EXCLUDES[@]}" "$SRC_PROJECT_DIR/" "$PRODUCT_DIR/$PROJECT_NAME/"

if [[ -d "$WORKDIR/resources" ]]; then
  mkdir -p "$PRODUCT_DIR/res"
  rsync -a --delete "$WORKDIR/resources/" "$PRODUCT_DIR/res/"
fi

if [[ -d "$WORKDIR/pixi-cache" ]]; then
  mkdir -p "$PRODUCT_DIR/pkg-cache"
  rsync -a --delete "$WORKDIR/pixi-cache/" "$PRODUCT_DIR/pkg-cache/"
else
  echo "ERROR: WORKDIR is missing pixi-cache (expected $WORKDIR/pixi-cache)" >&2
  exit 2
fi

cat >"$PRODUCT_DIR/envs.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

PRODUCT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export PIXI_CACHE_DIR="$PRODUCT_ROOT/pkg-cache"
export PRODUCT_RES_DIR="$PRODUCT_ROOT/res"
EOF

for kv in "${EXTRA_ENVS[@]}"; do
  if [[ "$kv" != *=* ]]; then
    echo "ERROR: --env must be KEY=VALUE, got: $kv" >&2
    exit 2
  fi
  key="${kv%%=*}"
  value="${kv#*=}"
  if [[ -z "$key" ]]; then
    echo "ERROR: --env must be KEY=VALUE, got empty KEY" >&2
    exit 2
  fi
  printf 'export %s=%q\n' "$key" "$value" >>"$PRODUCT_DIR/envs.sh"
done

cat >"$PRODUCT_DIR/bootstrap-project.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Bootstrap an extracted product directory.

This script:
  - sources ./envs.sh
  - runs: pixi install

Usage:
  bootstrap-project.sh [--project-name <name>] [--pixi <pixi>] [--] [pixi_install_args...]

Options:
  --project-name <name>   Project directory name under product/ (auto-detect if unique)
  --pixi <pixi>           Pixi executable to use (default: pixi)
  --help                  Show this help
USAGE
}

PROJECT_NAME=""
PIXI_BIN="pixi"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-name)
      PROJECT_NAME="${2:-}"; shift 2;;
    --pixi)
      PIXI_BIN="${2:-}"; shift 2;;
    --)
      shift; break;;
    *)
      break;;
  esac
done

PRODUCT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$PRODUCT_ROOT/envs.sh"

if [[ -z "$PROJECT_NAME" ]]; then
  mapfile -t candidates < <(
    find "$PRODUCT_ROOT" -mindepth 1 -maxdepth 1 -type d \
      ! -name 'pkg-cache' ! -name 'res' ! -name 'out' ! -name 'helpers' -printf '%f\n' | sort
  )
  if [[ "${#candidates[@]}" -eq 1 ]]; then
    PROJECT_NAME="${candidates[0]}"
  else
    echo "ERROR: --project-name not provided and could not auto-detect a single project directory in product root." >&2
    echo "Found: ${candidates[*]:-(none)}" >&2
    exit 2
  fi
fi

cd "$PRODUCT_ROOT/$PROJECT_NAME"

"$PIXI_BIN" install "$@"
EOF

chmod +x "$PRODUCT_DIR/envs.sh" "$PRODUCT_DIR/bootstrap-project.sh"

echo "OK: product created at: $PRODUCT_DIR" >&2
echo "Project: $PRODUCT_DIR/$PROJECT_NAME" >&2
