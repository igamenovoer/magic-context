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

Install Claude Code CLI (@anthropic-ai/claude-code) via npm on Linux/macOS and
mark onboarding as completed.

Options:
  --proxy URL               HTTP/HTTPS proxy for npm network access
  --accept-defaults         Use non-interactive defaults where possible (no-op)
  --from-official           Use official npm registry (registry.npmjs.org)
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
lr_log "=== Installing Claude Code CLI ==="
lr_log ""

lr_set_proxy_env "$proxy"
[ "$accept_defaults" -eq 1 ] && lr_log "Using --accept-defaults (no-op for this installer)."

# Prefer user-space Node.js installs from this repo's node component.
export PATH="${HOME}/.local/bin:${PATH:-}"

if ! lr_has_cmd node; then
  lr_die "Node.js is not available on PATH. Install Node.js first (components/nodejs/install-comp.sh)."
fi
if ! lr_has_cmd npm; then
  lr_die "npm is not available on PATH. Reinstall Node.js with npm support."
fi

package_name="@anthropic-ai/claude-code"
use_bun=0

if lr_has_cmd bun; then
  use_bun=1
  lr_log "Bun detected. Using bun for installation."
  if lr_has_cmd claude && [ "$force" -ne 1 ]; then
    # bun pm ls -g is human readable, use grep
    if bun pm ls -g 2>/dev/null | grep -q "$package_name"; then
       lr_log "Claude Code CLI is already available on PATH (claude found). Use --force to reinstall."
       exit 0
    fi
  fi
else
  if lr_has_cmd claude && [ "$force" -ne 1 ]; then
    lr_log "Claude Code CLI is already available on PATH (claude found). Use --force to reinstall."
    exit 0
  fi
fi

if true; then
  primary_registry="https://registry.npmmirror.com"
  official_registry="https://registry.npmjs.org"

  if [ "$from_official" -eq 1 ]; then
    registry="$official_registry"
    lr_log "Using official npm registry: $registry"
  else
    registry="$primary_registry"
    lr_log "Using China npm mirror: $registry"
    lr_log "Will fall back to official registry if the mirror fails."
  fi

  install_rc=0
  if [ "$use_bun" -eq 1 ]; then
    # Bun installation
    lr_log "Running: bun add -g ${package_name}@latest --registry $registry"
    lr_run bun add -g "${package_name}@latest" --registry "$registry" || install_rc=$?

    if [ "$install_rc" -ne 0 ] && [ "$from_official" -ne 1 ]; then
      lr_warn "bun add via mirror failed (exit code $install_rc). Retrying against official registry: $official_registry"
      lr_run bun add -g "${package_name}@latest" --registry "$official_registry" || lr_die "bun failed to install $package_name."
    elif [ "$install_rc" -ne 0 ]; then
      lr_die "bun failed to install $package_name (exit code $install_rc)."
    fi
  else
    # npm installation
    sudo_prefix="$(lr_sudo)"
    use_sudo=0
    npm_prefix="$(npm prefix -g 2>/dev/null || true)"
    npm_global_bin_dir=""
    [ -n "$npm_prefix" ] && npm_global_bin_dir="$npm_prefix/bin"
    # Make the npm global bin dir discoverable for the remainder of this script.
    if [ -n "$npm_global_bin_dir" ]; then
      export PATH="$npm_global_bin_dir:${PATH:-}"
    fi
    if [ -n "$npm_prefix" ] && [ ! -w "$npm_prefix" ] && [ -n "$sudo_prefix" ]; then
      use_sudo=1
      lr_log "Global npm prefix is not writable ($npm_prefix); using sudo for global install."
    fi

    if [ "$use_sudo" -eq 1 ]; then
      lr_run "$sudo_prefix" npm install -g "$package_name" --registry "$registry" || install_rc=$?
    else
      lr_run npm install -g "$package_name" --registry "$registry" || install_rc=$?
    fi

    if [ "$install_rc" -ne 0 ] && [ "$from_official" -ne 1 ]; then
      lr_warn "npm install via mirror failed (exit code $install_rc). Retrying against official registry: $official_registry"
      if [ "$use_sudo" -eq 1 ]; then
        lr_run "$sudo_prefix" npm install -g "$package_name" --registry "$official_registry" || lr_die "npm failed to install $package_name."
      else
        lr_run npm install -g "$package_name" --registry "$official_registry" || lr_die "npm failed to install $package_name."
      fi
    elif [ "$install_rc" -ne 0 ]; then
      lr_die "npm failed to install $package_name (exit code $install_rc)."
    fi
  fi

  # For user-space Node/Bun installs, the global bin dir is typically not on PATH.
  # Link `claude` into ~/.local/bin so it is available alongside node/npm/bun.
  user_bin_dir="${HOME}/.local/bin"
  target_bin=""

  if [ "$use_bun" -eq 1 ]; then
    bun_bin_dir="$(bun pm bin -g 2>/dev/null || true)"
    if [ -n "$bun_bin_dir" ] && [ -x "$bun_bin_dir/claude" ]; then
       target_bin="$bun_bin_dir/claude"
    elif [ -x "$HOME/.bun/bin/claude" ]; then
       target_bin="$HOME/.bun/bin/claude"
    fi
  else
    if [ -n "$npm_global_bin_dir" ] && [ -x "$npm_global_bin_dir/claude" ]; then
       target_bin="$npm_global_bin_dir/claude"
    fi
  fi

  if [ -n "$target_bin" ]; then
    mkdir -p "$user_bin_dir" || true
    lr_log "Linking claude into: $user_bin_dir"
    lr_run ln -sf "$target_bin" "$user_bin_dir/claude" || true
    export PATH="$user_bin_dir:${PATH:-}"
  fi
fi

if lr_has_cmd claude; then
  lr_log "Claude Code CLI installed successfully: $(claude --version 2>/dev/null || true)"
else
  lr_warn "'claude' command not found on PATH after installation."
  lr_warn "Ensure your global npm bin directory is on PATH."
fi

lr_log ""
lr_log "Configuring Claude Code to skip onboarding..."

# Resolve claude binary, even if PATH is not refreshed yet.
claude_cmd=""
if lr_has_cmd claude; then
  claude_cmd="claude"
fi

if [ -z "$claude_cmd" ]; then
  lr_die "Claude Code CLI ('claude') was not found after installation. Ensure your global npm bin directory is on PATH."
fi

config_file="${HOME:-}/.claude.json"
lr_log "Config file: $config_file"

if [ "$dry_run" -eq 1 ]; then
  lr_log "Dry-run: would set hasCompletedOnboarding=true in $config_file"
else
  lr_log "Updating hasCompletedOnboarding in $config_file ..."
  lr_run node - "$config_file" <<'NODE'
const fs = require("fs");
const path = process.argv[2];
let obj = {};
try {
  if (fs.existsSync(path)) {
    const text = fs.readFileSync(path, "utf8").trim();
    if (text) obj = JSON.parse(text);
  }
} catch {
  obj = {};
}
obj.hasCompletedOnboarding = true;
fs.writeFileSync(path, JSON.stringify(obj, null, 2), { encoding: "utf8" });
NODE
  lr_log "Claude Code onboarding has been marked as completed."
fi

lr_log "Claude Code onboarding/login should now be skipped on this host."
