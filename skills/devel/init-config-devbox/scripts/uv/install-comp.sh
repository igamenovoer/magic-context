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
  cmd="${1-}"
  [ -n "$cmd" ] || return 1

  # Primary: respect existing PATH.
  command -v "$cmd" >/dev/null 2>&1 && return 0

  # Secondary: try common user bin dirs for non-login shells.
  for d in "$HOME/.local/bin" "$HOME/bin" "$HOME/.pixi/bin" "$HOME/.bun/bin"; do
    if [ -x "$d/$cmd" ]; then
      export PATH="$d:${PATH:-}"
      return 0
    fi
  done

  return 1
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
Usage: ./install-comp.sh [options]

Install uv (Astral) on Linux/macOS.

Options:
  --proxy URL               HTTP/HTTPS proxy for downloads/package managers
  --accept-defaults         Use non-interactive defaults where possible
  --from-official           Prefer official sources (uv already uses official)
  --force                   Reinstall even if already installed
  --capture-log-file PATH   Also write logs to PATH
  --dry-run                 Print what would change, without installing
  -h, --help                Show this help
EOF
}

proxy=""
accept_defaults=0
from_official=0
force=0
capture_log_file=""
dry_run=0

while [ $# -gt 0 ]; do
  case "$1" in
    --proxy|--proxy-url) proxy="${2-}"; shift 2 ;;
    --proxy=*) proxy="${1#*=}"; shift ;;
    --accept-defaults) accept_defaults=1; shift ;;
    --from-official) from_official=1; shift ;;
    --force) force=1; shift ;;
    --capture-log-file) capture_log_file="${2-}"; shift 2 ;;
    --capture-log-file=*) capture_log_file="${1#*=}"; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

component_name=$(basename "$script_dir")
lr_init_component_log "$component_name" "$capture_log_file" "$dry_run"
lr_log ""
lr_log "=== Installing uv (Astral) ==="
lr_log ""

lr_set_proxy_env "$proxy"
[ "$accept_defaults" -eq 1 ] && lr_log "Using --accept-defaults."
[ "$from_official" -eq 1 ] && lr_log "Using --from-official (uv already uses official sources)."

if lr_has_cmd uv && [ "$force" -ne 1 ]; then
  lr_log "uv is already available on PATH. Use --force to reinstall."
  exit 0
fi

os="$(lr_os)"

installed=0

if [ "$os" = "macos" ] && lr_has_cmd brew; then
  if [ "$force" -eq 1 ]; then
    lr_run brew reinstall uv && installed=1 || installed=0
  else
    lr_run brew install uv && installed=1 || installed=0
  fi
fi

if [ "$installed" -ne 1 ]; then
  installer_url="https://astral.sh/uv/install.sh"
  installer_path="$LR_PKG_DIR/uv-install.sh"

  lr_log "Downloading official uv installer script from $installer_url ..."
  lr_download "$installer_url" "$installer_path" || lr_die "Failed to download uv installer."
  lr_run chmod +x "$installer_path" || true

  lr_log "Running uv installer script: $installer_path"
  lr_run sh "$installer_path" || lr_die "uv installer script failed."
fi

if ! lr_has_cmd uv; then
  lr_die "uv installation completed, but 'uv' is not on PATH."
fi

lr_log "uv installed successfully."

# Ensure uv tool bin is on PATH (uv provides a helper to update shell config)
lr_log "Running: uv tool update-shell (adds uv tool bin dir to PATH in shell config)"
if lr_run uv tool update-shell; then
  lr_log "uv tool update-shell completed. Restart your shell for PATH changes to take effect."
else
  lr_warn "'uv tool update-shell' failed. You may need to add ~/.local/bin to PATH manually."
fi
