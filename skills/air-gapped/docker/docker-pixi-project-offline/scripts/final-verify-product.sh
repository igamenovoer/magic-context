#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Final verification of a product directory in an offline/no-network container.

Runs (inside a container with --network none):
  ./bootstrap-project.sh
  pixi run <verification> (optional)

Logs are written to the provided --out-dir on the host.

Required tools: docker

Usage:
  final-verify-product.sh --image <image> --product-dir <dir> --out-dir <dir> [options]

Options:
  --image <image>             Docker image that has pixi installed (required)
  --product-dir <dir>         Path to product/ on host (required)
  --out-dir <dir>             Host directory for logs (required; created if missing)
  --project-name <name>       Project directory name under product/ (recommended)

Verification (pick at most one; if omitted, only runs bootstrap):
  --verify-task <task>        Runs: pixi run <task>
  --verify-run -- <cmd...>    Runs: pixi run <cmd...>
                             Note: --verify-run must be the last option.

Help:
  --help                      Show this help

Example:
  final-verify-product.sh --image myimg:latest --product-dir ./tmp/wd/product --out-dir ./tmp/wd/out --project-name app --verify-task smoke
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

IMAGE=""
PRODUCT_DIR=""
OUT_DIR=""
PROJECT_NAME=""

VERIFY_MODE="none" # none|task|run
VERIFY_TASK=""
VERIFY_RUN=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE="${2:-}"; shift 2;;
    --product-dir)
      PRODUCT_DIR="${2:-}"; shift 2;;
    --out-dir)
      OUT_DIR="${2:-}"; shift 2;;
    --project-name)
      PROJECT_NAME="${2:-}"; shift 2;;
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

if [[ -z "$IMAGE" || -z "$PRODUCT_DIR" || -z "$OUT_DIR" ]]; then
  echo "ERROR: --image, --product-dir, and --out-dir are required" >&2
  exit 2
fi
if [[ ! -d "$PRODUCT_DIR" ]]; then
  echo "ERROR: --product-dir is not a directory: $PRODUCT_DIR" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

MODE_MARKER="__none__"
VERIFY_ARGS=()
if [[ "$VERIFY_MODE" == "task" ]]; then
  if [[ -z "$PROJECT_NAME" ]]; then
    echo "ERROR: --project-name is required when using --verify-task" >&2
    exit 2
  fi
  if [[ -z "$VERIFY_TASK" ]]; then
    echo "ERROR: --verify-task requires a non-empty task name" >&2
    exit 2
  fi
  MODE_MARKER="__task__"
  VERIFY_ARGS=("$VERIFY_TASK")
elif [[ "$VERIFY_MODE" == "run" ]]; then
  if [[ -z "$PROJECT_NAME" ]]; then
    echo "ERROR: --project-name is required when using --verify-run" >&2
    exit 2
  fi
  MODE_MARKER="__run__"
  VERIFY_ARGS=("${VERIFY_RUN[@]}")
fi

docker run --rm --network none \
  -v "$PRODUCT_DIR:/product" \
  -v "$OUT_DIR:/out" \
  -w /product \
  "$IMAGE" \
  bash -lc '
set -euo pipefail
project_name="${1:-}"
mode="${2:-__none__}"
shift 2 || true

./bootstrap-project.sh ${project_name:+--project-name "$project_name"} 2>&1 | tee /out/bootstrap.offline.log

case "$mode" in
  __none__)
    echo "No verification requested." | tee /out/verify.offline.log
    ;;
  __task__)
    task="${1:?missing task name}"
    cd "$project_name"
    pixi run "$task" 2>&1 | tee /out/verify.offline.log
    ;;
  __run__)
    if [[ $# -lt 1 ]]; then
      echo "ERROR: missing verify command args" >&2
      exit 2
    fi
    cd "$project_name"
    pixi run "$@" 2>&1 | tee /out/verify.offline.log
    ;;
  *)
    echo "ERROR: unknown verify mode: $mode" >&2
    exit 2
    ;;
esac
' bash "$PROJECT_NAME" "$MODE_MARKER" "${VERIFY_ARGS[@]}"

echo "OK: final verification complete; logs in: $OUT_DIR" >&2
