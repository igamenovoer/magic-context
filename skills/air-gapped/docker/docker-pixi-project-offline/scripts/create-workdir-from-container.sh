#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Create a host WORKDIR from a project that already exists inside a running container.

Inside the container, this script creates --container-workdir-path with:
  <container-workdir>/project/<project_name>/
  <container-workdir>/resources/              (optional; from --resources-path-in-container)
  <container-workdir>/pixi-cache/
  <container-workdir>/out/
Then runs (with network, unless the container is already restricted):
  pixi install
  pixi run <verification> (optional)
Finally, it copies <container-workdir> back to the host WORKDIR using docker cp.

Required tools (host): docker
Required tools (container): bash, pixi

Usage:
  create-workdir-from-container.sh --container-id <id> --project-path-in-container <path> [--workdir <workdir>] [options]

Options:
  --container-id <id>               Running container id/name (required)
  --project-path-in-container <p>   Path to project inside container (required)
  --workdir <dir>                   Output workdir on host (default: <workspace>/tmp/workdir-<ts>; will be overwritten)
  --container-workdir-path <p>      Work directory path to use inside the container (default: /tmp/pixi-offline-workdir)
  --project-name <name>             Overrides detected project name (default: basename(project-path-in-container))
  --resources-path-in-container <p> Optional path to resources inside container copied to <container-workdir>/resources/

Verification (pick at most one; if omitted, only runs pixi install):
  --verify-task <task>              Runs: pixi run <task>
  --verify-run -- <cmd...>          Runs: pixi run <cmd...>
                                   Note: --verify-run must be the last option.

Help:
  --help                            Show this help

Examples:
  create-workdir-from-container.sh --container-id app --project-path-in-container /srv/app --workdir ./tmp/wd --verify-task smoke
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

CONTAINER_ID=""
PROJECT_PATH_IN_CONTAINER=""
WORKDIR=""
CONTAINER_WORKDIR_PATH="/tmp/pixi-offline-workdir"
PROJECT_NAME=""
RESOURCES_PATH_IN_CONTAINER=""

VERIFY_MODE="none" # none|task|run
VERIFY_TASK=""
VERIFY_RUN=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --container-id)
      CONTAINER_ID="${2:-}"; shift 2;;
    --project-path-in-container)
      PROJECT_PATH_IN_CONTAINER="${2:-}"; shift 2;;
    --workdir)
      WORKDIR="${2:-}"; shift 2;;
    --container-workdir-path)
      CONTAINER_WORKDIR_PATH="${2:-}"; shift 2;;
    --project-name)
      PROJECT_NAME="${2:-}"; shift 2;;
    --resources-path-in-container)
      RESOURCES_PATH_IN_CONTAINER="${2:-}"; shift 2;;
    --verify-task)
      VERIFY_MODE="task"
      VERIFY_TASK="${2:-}"
      shift 2;;
    --verify-run)
      VERIFY_MODE="run"
      shift
      if [[ "${1:-}" != "--" ]]; then
        echo "ERROR: --verify-run requires a '--' delimiter" >&2
        exit 2
      fi
      shift
      if [[ $# -eq 0 ]]; then
        echo "ERROR: --verify-run requires command args after --" >&2
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

if [[ -z "$CONTAINER_ID" || -z "$PROJECT_PATH_IN_CONTAINER" ]]; then
  echo "ERROR: --container-id and --project-path-in-container are required" >&2
  echo "Run with --help for usage." >&2
  exit 2
fi

if [[ -z "$PROJECT_NAME" ]]; then
  PROJECT_NAME="$(basename "$PROJECT_PATH_IN_CONTAINER")"
fi

if [[ -z "$WORKDIR" ]]; then
  WORKDIR="$(default_workdir)"
fi

if [[ -z "$CONTAINER_WORKDIR_PATH" || "$CONTAINER_WORKDIR_PATH" == "/" ]]; then
  echo "ERROR: --container-workdir-path must be a non-root path (got: $CONTAINER_WORKDIR_PATH)" >&2
  exit 2
fi

MODE_MARKER="__none__"
VERIFY_ARGS=()
if [[ "$VERIFY_MODE" == "task" ]]; then
  if [[ -z "$VERIFY_TASK" ]]; then
    echo "ERROR: --verify-task requires a non-empty task name" >&2
    exit 2
  fi
  MODE_MARKER="__task__"
  VERIFY_ARGS=("$VERIFY_TASK")
elif [[ "$VERIFY_MODE" == "run" ]]; then
  MODE_MARKER="__run__"
  VERIFY_ARGS=("${VERIFY_RUN[@]}")
fi

# Prepare the container workdir and run install/verify.
docker exec "$CONTAINER_ID" bash -lc '
set -euo pipefail
project_path="${1:?missing project_path}"
project_name="${2:?missing project_name}"
resources_path="${3:-}"
container_workdir="${4:?missing container_workdir}"
mode="${5:-__none__}"
shift 5 || true

if [[ -z "$container_workdir" || "$container_workdir" == "/" ]]; then
  echo "ERROR: invalid container_workdir: $container_workdir" >&2
  exit 2
fi

rm -rf "$container_workdir"
mkdir -p "$container_workdir/project/$project_name" "$container_workdir/resources" "$container_workdir/pixi-cache" "$container_workdir/out"

cp -a "$project_path/." "$container_workdir/project/$project_name/"
if [[ -n "$resources_path" && -d "$resources_path" ]]; then
  cp -a "$resources_path/." "$container_workdir/resources/"
fi

cd "$container_workdir/project/$project_name"
export PIXI_CACHE_DIR="$container_workdir/pixi-cache"
pixi install 2>&1 | tee "$container_workdir/out/pixi-install.log"

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
' bash "$PROJECT_PATH_IN_CONTAINER" "$PROJECT_NAME" "$RESOURCES_PATH_IN_CONTAINER" "$CONTAINER_WORKDIR_PATH" "$MODE_MARKER" "${VERIFY_ARGS[@]}"

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
docker cp "$CONTAINER_ID:$CONTAINER_WORKDIR_PATH/." "$WORKDIR"

# Install helper scripts into WORKDIR/helpers on host (for portability).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$WORKDIR/helpers" "$WORKDIR/product"
cp -f "$SCRIPT_DIR/make-archive-ready-product.sh" "$WORKDIR/helpers/make-archive-ready-product.sh"
chmod +x "$WORKDIR/helpers/make-archive-ready-product.sh"

echo "OK: WORKDIR created at: $WORKDIR" >&2
echo "Next: $WORKDIR/helpers/make-archive-ready-product.sh --workdir \"$WORKDIR\" --project-name \"$PROJECT_NAME\"" >&2
