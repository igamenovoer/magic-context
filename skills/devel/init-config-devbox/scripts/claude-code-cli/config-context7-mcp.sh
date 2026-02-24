#!/usr/bin/env sh
set -eu

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)

# Self-contained helpers (inlined per script).
lr_os() {
  # outputs: linux | macos | unknown
  case "$(uname -s 2>/dev/null || echo unknown)" in
    Linux) echo linux ;;
    Darwin) echo macos ;;
    *) echo unknown ;;
  esac
}

lr_arch() {
  # outputs: amd64 | arm64 | <raw>
  case "$(uname -m 2>/dev/null || echo unknown)" in
    x86_64|amd64) echo amd64 ;;
    aarch64|arm64) echo arm64 ;;
    *) uname -m 2>/dev/null || echo unknown ;;
  esac
}

lr_has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

lr_shell_quote() {
  # Single-quote a string for POSIX shells.
  # Example: abc'def -> 'abc'"'"'def'
  printf "'%s'" "$(printf "%s" "${1-}" | sed "s/'/'\"'\"'/g")"
}

lr_set_proxy_env() {
  proxy="${1-}"
  if [ -n "$proxy" ]; then
    export HTTP_PROXY="$proxy" HTTPS_PROXY="$proxy"
    export http_proxy="$proxy" https_proxy="$proxy"
  fi
}

lr_init_component_log() {
  LR_COMPONENT_NAME="${1-unknown}"
  LR_CAPTURE_LOG_FILE="${2-}"
  LR_DRY_RUN="${3-0}"

  if [ -n "${LRAI_MASTER_OUTPUT_DIR:-}" ]; then
    LR_ROOT="$LRAI_MASTER_OUTPUT_DIR"
  else
    LR_ROOT="$(pwd)/lanren-cache"
  fi

  LR_LOG_DIR="$LR_ROOT/logs/$LR_COMPONENT_NAME"
  LR_PKG_DIR="$LR_ROOT/packages/$LR_COMPONENT_NAME"
  mkdir -p "$LR_LOG_DIR" "$LR_PKG_DIR"

  ts="$(date +%Y%m%d_%H%M%S 2>/dev/null || date)"
  LR_LOG_FILE="$LR_LOG_DIR/$LR_COMPONENT_NAME-$ts.log"
  : >"$LR_LOG_FILE"

  if [ -n "$LR_CAPTURE_LOG_FILE" ]; then
    cap_dir="$(dirname "$LR_CAPTURE_LOG_FILE" 2>/dev/null || echo .)"
    mkdir -p "$cap_dir"
    : >"$LR_CAPTURE_LOG_FILE"
  fi
}

lr_log() {
  msg="$*"
  if [ -n "${LR_CAPTURE_LOG_FILE:-}" ]; then
    printf '%s\n' "$msg" | tee -a "$LR_LOG_FILE" "$LR_CAPTURE_LOG_FILE"
  else
    printf '%s\n' "$msg" | tee -a "$LR_LOG_FILE"
  fi
}

lr_warn() {
  lr_log "WARNING: $*"
}

lr_err() {
  lr_log "ERROR: $*"
}

lr_die() {
  lr_err "$*"
  exit 1
}

lr_run_impl() {
  if [ "${LR_DRY_RUN:-0}" -eq 1 ]; then
    return 0
  fi
  tmp_rc="$(mktemp "${TMPDIR:-/tmp}/lanren-ai-rc.XXXXXX" 2>/dev/null || echo "${TMPDIR:-/tmp}/lanren-ai-rc.$$")"

  # Stream output in real-time while logging, capture exit code
  if [ -n "${LR_CAPTURE_LOG_FILE:-}" ]; then
    { "$@" 2>&1; echo $? >"$tmp_rc"; } | tee -a "$LR_LOG_FILE" "$LR_CAPTURE_LOG_FILE"
  else
    { "$@" 2>&1; echo $? >"$tmp_rc"; } | tee -a "$LR_LOG_FILE"
  fi

  rc="$(cat "$tmp_rc" 2>/dev/null || echo 1)"
  rm -f "$tmp_rc" 2>/dev/null || true
  return "$rc"
}

lr_run() {
  # Runs a command and logs its combined output while preserving the command exit code.
  # Usage: lr_run <cmd> [args...]
  lr_log "+ $*"
  lr_run_impl "$@"
}

lr_run_masked() {
  # Runs a command but logs a caller-provided, redacted description instead of the full argv.
  # Usage: lr_run_masked "<safe description>" <cmd> [args...]
  desc="${1-}"
  shift || true
  lr_log "+ $desc"
  lr_run_impl "$@"
}

lr_download() {
  # Usage: lr_download <url> <dest_path>
  url="$1"
  dest="$2"
  if lr_has_cmd curl; then
    lr_run curl -fsSL "$url" -o "$dest"
    return $?
  fi
  if lr_has_cmd wget; then
    lr_run wget -q "$url" -O "$dest"
    return $?
  fi
  lr_die "Neither curl nor wget is available for downloading: $url"
}

lr_sudo() {
  # Echo a sudo prefix if needed/available.
  if [ "$(id -u 2>/dev/null || echo 1)" -eq 0 ]; then
    echo ""
    return 0
  fi
  if lr_has_cmd sudo; then
    echo "sudo"
    return 0
  fi
  echo ""
}

usage() {
  cat <<'EOF'
Usage: ./config-context7-mcp.sh [options]

Configure the Context7 MCP server for Claude Code CLI (user scope).

By default this uses a runner (bunx preferred, otherwise npx) so you don't
need a global install:
  claude mcp add-json -s user context7 '<json>'

Options:
  --from-official           Use official npm registry instead of China mirror
  --dry-run                 Print what would change, without installing/configuring
  --capture-log-file PATH   Also write logs to PATH
  -h, --help                Show this help
EOF
}

dry_run=0
capture_log_file=""
from_official=0

while [ $# -gt 0 ]; do
  case "$1" in
    --from-official) from_official=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    --capture-log-file) capture_log_file="${2-}"; shift 2 ;;
    --capture-log-file=*) capture_log_file="${1#*=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

component_name=$(basename "$script_dir")
lr_init_component_log "$component_name" "$capture_log_file" "$dry_run"

lr_log "=== Configure Context7 MCP for Claude Code (user scope) ==="
lr_log ""

if ! lr_has_cmd claude; then
  lr_die "Claude Code CLI ('claude') is not on PATH. Install it first."
fi

# Choose a runner to launch the MCP server (no global install required).
# Preference on Linux/macOS: bunx (if bun is available) > npx
runner=""
runner_kind=""
if lr_has_cmd bunx; then
  runner="bunx"
  runner_kind="bun"
elif lr_has_cmd npx; then
  runner="npx"
  runner_kind="node"
else
  lr_die "No suitable runner found. Install bun (bunx) or Node.js (npx) first."
fi

if [ "$runner_kind" = "node" ] && ! lr_has_cmd node; then
  lr_die "Node.js is required for npx, but node is not available on PATH."
fi

base_package_name="@upstash/context7-mcp"
package_name="${base_package_name}"
package_name_latest="${base_package_name}@latest"

# Registry configuration: default to China mirror
primary_registry="https://registry.npmmirror.com"
official_registry="https://registry.npmjs.org"

if [ "$from_official" -eq 1 ]; then
  registry="$official_registry"
  lr_log "Using official npm registry: $registry"
else
  registry="$primary_registry"
  lr_log "Using China npm mirror: $registry"
fi

if [ "$runner_kind" = "bun" ]; then
    if bun pm -g ls 2>/dev/null | grep -q "${package_name}"; then
       lr_log "Package ${package_name} is already installed globally. Skipping installation."
    else
       lr_log "Installing $package_name_latest globally via Bun..."
       lr_run bun add -g "$package_name_latest" --registry "$registry" || lr_die "bun failed to install $package_name_latest"
    fi
elif [ "$runner_kind" = "node" ]; then
    if npm list -g --depth=0 "${package_name}" >/dev/null 2>&1; then
       lr_log "Package ${package_name} is already installed globally. Skipping installation."
    else
       lr_log "Installing $package_name_latest globally via npm..."
       lr_sudo_prefix="$(lr_sudo)"
       if [ -n "$lr_sudo_prefix" ]; then
         lr_run "$lr_sudo_prefix" npm install -g "$package_name_latest" --registry "$registry" || lr_die "npm failed to install $package_name_latest"
       else
         lr_run npm install -g "$package_name_latest" --registry "$registry" || lr_die "npm failed to install $package_name_latest"
       fi
    fi
fi

lr_log "Configuring Context7 MCP using runner: $runner ($package_name)"

scope="user"
mcp_name="context7"

lr_log "Removing existing '$mcp_name' server in scope '$scope' (if any)..."
lr_run claude mcp remove -s "$scope" "$mcp_name" || true

lr_log "Building JSON configuration..."
if ! lr_has_cmd node; then
  lr_die "Node.js is required to build JSON config. Install Node.js (or run with an existing JSON string)."
fi

if [ "$runner" = "npx" ]; then
  json="$(node -e 'console.log(JSON.stringify({type:"stdio",command:process.argv[1],args:["-y",process.argv[2]],env:{}}))' "$runner" "$package_name")"
else
  json="$(node -e 'console.log(JSON.stringify({type:"stdio",command:process.argv[1],args:[process.argv[2]],env:{}}))' "$runner" "$package_name")"
fi

lr_log "Adding '$mcp_name' server in scope '$scope' via add-json..."
lr_run_masked "claude mcp add-json -s $scope $mcp_name <json>" claude mcp add-json -s "$scope" "$mcp_name" "$json" || lr_die "Failed to add Context7 MCP server."

lr_log "Context7 MCP server has been configured. Verify with: claude mcp list"
