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

lr_user_prefix() {
  # Prefer user-space installs.
  # Default: ~/.local (respects XDG if set).
  if [ -n "${XDG_DATA_HOME:-}" ]; then
    # XDG_DATA_HOME typically is ~/.local/share; keep tools in ~/.local/bin.
    echo "${HOME}/.local"
    return 0
  fi
  echo "${HOME}/.local"
}

lr_node_platform() {
  os="$1"
  arch="$2"
  case "$os" in
    linux)
      case "$arch" in
        amd64) echo "linux-x64" ;;
        arm64) echo "linux-arm64" ;;
        *) lr_die "Unsupported CPU architecture for Node.js tarball: $arch" ;;
      esac
      ;;
    macos)
      case "$arch" in
        amd64) echo "darwin-x64" ;;
        arm64) echo "darwin-arm64" ;;
        *) lr_die "Unsupported CPU architecture for Node.js tarball: $arch" ;;
      esac
      ;;
    *) lr_die "Unsupported OS for Node.js tarball install: $os" ;;
  esac
}

lr_node_latest_lts_version() {
  # Prints something like: v20.11.1
  index_url="$1"
  index_path="$2"

  # IMPORTANT: This function is often called via command substitution $(...).
  # Do not print logs to stdout here; only print the version.
  if [ "${LR_DRY_RUN:-0}" -eq 1 ]; then
    return 1
  fi

  # Download without lr_run/lr_log to avoid stdout pollution.
  {
    printf '%s\n' "+ (silent) download $index_url -> $index_path"
    if lr_has_cmd curl; then
      curl -fsSL "$index_url" -o "$index_path"
    elif lr_has_cmd wget; then
      wget -q "$index_url" -O "$index_path"
    else
      printf '%s\n' "ERROR: Neither curl nor wget is available for downloading: $index_url"
      exit 1
    fi
  } >>"$LR_LOG_FILE" 2>&1

  if [ -n "${LR_CAPTURE_LOG_FILE:-}" ]; then
    {
      printf '%s\n' "+ (silent) download $index_url -> $index_path"
    } >>"$LR_CAPTURE_LOG_FILE" 2>&1 || true
  fi

  # Extract the first release where "lts" is not false.
  # This is a best-effort JSON scrape to keep dependencies minimal.
  awk '
    BEGIN { v = "" }
    /"version"[[:space:]]*:[[:space:]]*"v[0-9]+\.[0-9]+\.[0-9]+"/ {
      line=$0
      sub(/.*"version"[[:space:]]*:[[:space:]]*"/, "", line)
      sub(/".*/, "", line)
      v=line
    }
    /"lts"[[:space:]]*:/ {
      if (v != "" && $0 !~ /false/) { print v; exit 0 }
    }
  ' "$index_path"
}

usage() {
  cat <<'EOF'
Usage: ./install-comp.sh [options]

Install Node.js (LTS preferred) on Linux/macOS in user-space (no sudo).

Options:
  --proxy URL               HTTP/HTTPS proxy for downloads/package managers
  --accept-defaults         Use non-interactive defaults where possible
  --from-official           Prefer official sources (affects guidance only)
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
lr_log "=== Installing Node.js (LTS preferred) ==="
lr_log ""

lr_set_proxy_env "$proxy"

if lr_has_cmd node && [ "$force" -ne 1 ]; then
  lr_log "Node.js is already available on PATH (node found). Use --force to reinstall."
  exit 0
fi

os="$(lr_os)"
arch="$(lr_arch)"
installed=0

prefix="$(lr_user_prefix)"
bin_dir="$prefix/bin"
opt_dir="$prefix/opt"
mkdir -p "$bin_dir" "$opt_dir"

node_platform="$(lr_node_platform "$os" "$arch")"

if [ "$from_official" -eq 1 ]; then
  dist_base="https://nodejs.org/dist"
else
  # Mirrors are useful for CN networks; this mirror follows upstream layout.
  dist_base="https://mirrors.tuna.tsinghua.edu.cn/nodejs-release"
fi

if [ "$dry_run" -eq 1 ]; then
  lr_log "Dry-run: would install latest Node.js LTS tarball (no sudo)."
  lr_log "- Platform: $node_platform"
  lr_log "- Prefix: $prefix"
  lr_log "- Would resolve LTS from: $dist_base/index.json"
  lr_log "- Would extract into: $opt_dir/node-<version>-$node_platform"
  lr_log "- Would symlink into: $bin_dir (node/npm/npx/corepack)"
  lr_log "Dry-run complete (no changes made)."
  exit 0
fi

index_url="$dist_base/index.json"
index_path="$LR_PKG_DIR/node-index.json"
lr_log "Fetching Node.js release index to resolve latest LTS..."
node_version="$(lr_node_latest_lts_version "$index_url" "$index_path" 2>/dev/null || true)"

if [ -z "$node_version" ]; then
  lr_warn "Could not resolve latest LTS version from index.json."
  lr_warn "You can re-run with --from-official, or install manually from the URLs below."
else
  tar_name="node-$node_version-$node_platform.tar.gz"
  tar_url="$dist_base/$node_version/$tar_name"
  tar_path="$LR_PKG_DIR/$tar_name"
  install_dir="$opt_dir/node-$node_version-$node_platform"

  lr_log "Installing Node.js $node_version ($node_platform) to user-space: $install_dir"

  if [ "$force" -eq 1 ] && [ -d "$install_dir" ]; then
    lr_log "Removing existing install (force enabled): $install_dir"
    lr_run rm -rf "$install_dir" || true
  fi

  if [ ! -d "$install_dir" ]; then
    lr_download "$tar_url" "$tar_path"
    lr_run mkdir -p "$install_dir" || true
    # Strip the leading directory from the tarball (node-vX.Y.Z-<platform>/...)
    lr_run tar -xzf "$tar_path" -C "$install_dir" --strip-components=1 && installed=1 || installed=0
  else
    lr_log "Install directory already exists; skipping extract. Use --force to reinstall."
    installed=1
  fi

  if [ "$installed" -eq 1 ]; then
    lr_log "Linking node/npm/npx/corepack into: $bin_dir"
    lr_run ln -sf "$install_dir/bin/node" "$bin_dir/node" || true
    lr_run ln -sf "$install_dir/bin/npm" "$bin_dir/npm" || true
    lr_run ln -sf "$install_dir/bin/npx" "$bin_dir/npx" || true
    if [ -x "$install_dir/bin/corepack" ]; then
      lr_run ln -sf "$install_dir/bin/corepack" "$bin_dir/corepack" || true
    fi
  fi
fi

if [ "$installed" -eq 1 ]; then
  # The install prefers user-space ($prefix/bin). That directory may not be on PATH
  # for non-interactive shells, so validate using the expected symlink first.
  export PATH="$bin_dir:${PATH:-}"

  if [ -x "$bin_dir/node" ]; then
    node_ver="$("$bin_dir/node" --version 2>/dev/null || true)"
    if [ -n "$node_ver" ]; then
      lr_log "Node.js installed successfully: $node_ver"
      exit 0
    fi
  fi

  if lr_has_cmd node; then
    lr_log "Node.js installed successfully: $(node --version 2>/dev/null || true)"
    exit 0
  fi
fi

lr_log ""
lr_log "Manual installation guidance:"
lr_log "- This script prefers user-space installs (no sudo). Ensure \"$prefix/bin\" is on PATH."
lr_log "  Example (bash/zsh): export PATH=\"$prefix/bin:\$PATH\""
lr_log "- Official downloads: https://nodejs.org/en/download"
lr_log "- Official dist index (for latest LTS): https://nodejs.org/dist/index.json"
lr_log "- China mirror (same layout): https://mirrors.tuna.tsinghua.edu.cn/nodejs-release/"
lr_log ""
if [ "$installed" -eq 1 ] && [ -e "$bin_dir/node" ]; then
  lr_die "Node.js appears installed, but 'node' is not discoverable on PATH (missing $prefix/bin on PATH)."
fi

lr_die "Node.js installation did not complete successfully."
