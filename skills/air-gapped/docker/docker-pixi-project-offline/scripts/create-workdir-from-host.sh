#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Create a WORKDIR from a host Pixi project directory.

Stages a filtered copy of the project into:
  WORKDIR/project/<project_name>/
Optionally stages resources into:
  WORKDIR/resources/
Then runs, inside Docker (with network):
  pixi install
  pixi run <verification> (optional)

It also installs helper scripts into:
  WORKDIR/helpers/

Required tools: docker, rsync

Usage:
  create-workdir-from-host.sh --image <image> --project-dir <dir> [--workdir <workdir>] [options]

Options:
  --image <image>                 Docker image that has pixi installed (required)
  --project-dir <dir>             Host path to project (required)
  --workdir <dir>                 Output workdir on host (default: <workspace>/tmp/workdir-<ts>; will be overwritten)
  --container-workdir-path <dir>  Base path used inside the container (default: /workdir)
  --project-name <name>           Overrides detected project name (default: basename(project-dir))
  --resources-dir <dir>           Optional host path to resources to stage into WORKDIR/resources
  --exclude <pattern>             Additional rsync exclude (repeatable)

Verification (pick at most one; if omitted, only runs pixi install):
  --verify-task <task>            Runs: pixi run <task>
  --verify-run -- <cmd...>        Runs: pixi run <cmd...>
                                 Note: --verify-run must be the last option.

Help:
  --help                          Show this help

Examples:
  create-workdir-from-host.sh --image myimg:latest --project-dir ./app --workdir ./tmp/wd --verify-task smoke
  create-workdir-from-host.sh --image myimg:latest --project-dir ./app --workdir ./tmp/wd --verify-run -- python -c 'print("ok")'
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

IMAGE=""
PROJECT_DIR=""
WORKDIR=""
CONTAINER_WORKDIR_PATH="/workdir"
PROJECT_NAME=""
RESOURCES_DIR=""
EXCLUDES=()

VERIFY_MODE="none" # none|task|run
VERIFY_TASK=""
VERIFY_RUN=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE="${2:-}"; shift 2;;
    --project-dir)
      PROJECT_DIR="${2:-}"; shift 2;;
    --workdir)
      WORKDIR="${2:-}"; shift 2;;
    --container-workdir-path)
      CONTAINER_WORKDIR_PATH="${2:-}"; shift 2;;
    --project-name)
      PROJECT_NAME="${2:-}"; shift 2;;
    --resources-dir)
      RESOURCES_DIR="${2:-}"; shift 2;;
    --exclude)
      EXCLUDES+=("${2:-}"); shift 2;;
    --verify-task)
      VERIFY_MODE="task"
      VERIFY_TASK="${2:-}"
      shift 2;;
    --verify-run)
      VERIFY_MODE="run"
      shift
      if [[ "${1:-}" != "--" ]]; then
        echo "ERROR: --verify-run requires a '--' delimiter" >&2
        echo "Example: --verify-run -- python -c 'print(\"ok\")'" >&2
        exit 2
      fi
      shift
      if [[ $# -eq 0 ]]; then
        echo "ERROR: --verify-run requires at least one command argument after --" >&2
        exit 2
      fi
      VERIFY_RUN=("$@")
      break;;
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
need_cmd docker
need_cmd rsync

workspace_root() {
  if command -v git >/dev/null 2>&1; then
    git rev-parse --show-toplevel 2>/dev/null && return 0
  fi
  pwd
}

default_workdir() {
  local root ts
  root="$(workspace_root)"
  ts="$(date +%s)"
  echo "$root/tmp/workdir-$ts"
}

if [[ -z "$IMAGE" || -z "$PROJECT_DIR" ]]; then
  echo "ERROR: --image and --project-dir are required" >&2
  echo "Run with --help for usage." >&2
  exit 2
fi

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "ERROR: --project-dir does not exist or is not a directory: $PROJECT_DIR" >&2
  exit 2
fi

if [[ -z "$PROJECT_NAME" ]]; then
  PROJECT_NAME="$(basename "$PROJECT_DIR")"
fi

if [[ -z "$WORKDIR" ]]; then
  WORKDIR="$(default_workdir)"
fi

if [[ -z "$CONTAINER_WORKDIR_PATH" || "$CONTAINER_WORKDIR_PATH" == "/" ]]; then
  echo "ERROR: --container-workdir-path must be a non-root absolute-ish path (got: $CONTAINER_WORKDIR_PATH)" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

rm -rf "$WORKDIR"
mkdir -p \
  "$WORKDIR/project/$PROJECT_NAME" \
  "$WORKDIR/resources" \
  "$WORKDIR/pixi-cache" \
  "$WORKDIR/out" \
  "$WORKDIR/helpers" \
  "$WORKDIR/product"

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

rsync -a --delete "${RSYNC_EXCLUDES[@]}" "$PROJECT_DIR/" "$WORKDIR/project/$PROJECT_NAME/"

if [[ -n "$RESOURCES_DIR" ]]; then
  if [[ ! -d "$RESOURCES_DIR" ]]; then
    echo "ERROR: --resources-dir is not a directory: $RESOURCES_DIR" >&2
    exit 2
  fi
  rsync -a --delete "$RESOURCES_DIR/" "$WORKDIR/resources/"
fi

# Install helpers into WORKDIR/helpers (for portability with the generated workdir)
cp -f "$SCRIPT_DIR/make-archive-ready-product.sh" "$WORKDIR/helpers/make-archive-ready-product.sh"
chmod +x "$WORKDIR/helpers/make-archive-ready-product.sh"

MODE_MARKER="__none__"
DOCKER_VERIFY_ARGS=()
if [[ "$VERIFY_MODE" == "task" ]]; then
  if [[ -z "$VERIFY_TASK" ]]; then
    echo "ERROR: --verify-task requires a non-empty task name" >&2
    exit 2
  fi
  MODE_MARKER="__task__"
  DOCKER_VERIFY_ARGS=("$VERIFY_TASK")
elif [[ "$VERIFY_MODE" == "run" ]]; then
  MODE_MARKER="__run__"
  DOCKER_VERIFY_ARGS=("${VERIFY_RUN[@]}")
fi

docker run --rm \
  -v "$WORKDIR/project:$CONTAINER_WORKDIR_PATH/project" \
  -v "$WORKDIR/resources:$CONTAINER_WORKDIR_PATH/resources" \
  -v "$WORKDIR/out:$CONTAINER_WORKDIR_PATH/out" \
  -v "$WORKDIR/pixi-cache:$CONTAINER_WORKDIR_PATH/pixi-cache" \
  -w "$CONTAINER_WORKDIR_PATH/project/$PROJECT_NAME" \
  "$IMAGE" \
  bash -lc '
set -euo pipefail
container_workdir="${1:?missing container_workdir}"
shift || true

export PIXI_CACHE_DIR="$container_workdir/pixi-cache"
pixi install 2>&1 | tee "$container_workdir/out/pixi-install.log"

mode="${1:-__none__}"
shift || true
case "$mode" in
  __none__)
    echo "No verification requested." | tee "$container_workdir/out/verify.log"
    ;;
  __task__)
    task="${1:?missing task name}"
    pixi run "$task" 2>&1 | tee "$container_workdir/out/verify.log"
    ;;
  __run__)
    if [[ $# -lt 1 ]]; then
      echo "ERROR: missing verify command args" >&2
      exit 2
    fi
    pixi run "$@" 2>&1 | tee "$container_workdir/out/verify.log"
    ;;
  *)
    echo "ERROR: unknown verify mode: $mode" >&2
    exit 2
    ;;
esac
' bash "$CONTAINER_WORKDIR_PATH" "$MODE_MARKER" "${DOCKER_VERIFY_ARGS[@]}"

echo "OK: WORKDIR created at: $WORKDIR" >&2
echo "Next: $WORKDIR/helpers/make-archive-ready-product.sh --workdir \"$WORKDIR\" --project-name \"$PROJECT_NAME\"" >&2
