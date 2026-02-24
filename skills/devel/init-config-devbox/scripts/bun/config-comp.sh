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
Usage: ./config-comp.sh [options]

Configure Bun global registry mirror by editing ~/.bunfig.toml.

Options:
  --mirror <cn|official>     Registry mirror to use
  --dry-run                  Print what would change, without writing files
  --capture-log-file PATH    Also write logs to PATH
  -h, --help                 Show this help
EOF
}

mirror=""
dry_run=0
capture_log_file=""

while [ $# -gt 0 ]; do
  case "$1" in
    --mirror) mirror="${2-}"; shift 2 ;;
    --mirror=*) mirror="${1#*=}"; shift ;;
    --dry-run) dry_run=1; shift ;;
    --capture-log-file) capture_log_file="${2-}"; shift 2 ;;
    --capture-log-file=*) capture_log_file="${1#*=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [ -z "$mirror" ]; then
  echo "Missing required --mirror." >&2
  usage
  exit 1
fi

case "$mirror" in
  cn) registry_url="https://registry.npmmirror.com" ;;
  official) registry_url="https://registry.npmjs.org" ;;
  *) echo "Invalid --mirror: $mirror" >&2; usage; exit 1 ;;
esac

component_name=$(basename "$script_dir")
lr_init_component_log "$component_name" "$capture_log_file" "$dry_run"

bunfig_path="${HOME:-}/.bunfig.toml"
lr_log "=== Configuring Bun Global Mirror ==="
lr_log "Target file: $bunfig_path"
lr_log "Selected Mirror: $mirror ($registry_url)"

if [ "$dry_run" -eq 1 ]; then
  lr_log "Dry-run: would ensure the following in ~/.bunfig.toml:"
  lr_log "[install]"
  lr_log "registry = \"$registry_url\""
  exit 0
fi

content=""
if [ -f "$bunfig_path" ]; then
  content="$(cat "$bunfig_path" 2>/dev/null || true)"
fi

tmp_file="$(mktemp "${TMPDIR:-/tmp}/bunfig.toml.XXXXXX" 2>/dev/null || echo "${TMPDIR:-/tmp}/bunfig.toml.$$")"

registry_line="registry = \"$registry_url\""

if printf '%s\n' "$content" | grep -Eq '^[[:space:]]*registry[[:space:]]*='; then
  printf '%s\n' "$content" | awk -v repl="$registry_line" '
    BEGIN {done=0}
    {
      if (!done && $0 ~ /^[[:space:]]*registry[[:space:]]*=/) {
        print repl
        done=1
        next
      }
      print
    }
  ' >"$tmp_file"
  lr_log "Updated existing registry configuration."
elif printf '%s\n' "$content" | grep -Eq '^[[:space:]]*\\[install\\][[:space:]]*$'; then
  printf '%s\n' "$content" | awk -v repl="$registry_line" '
    BEGIN {done=0}
    {
      print
      if (!done && $0 ~ /^[[:space:]]*\\[install\\][[:space:]]*$/) {
        print repl
        done=1
      }
    }
  ' >"$tmp_file"
  lr_log "Added registry configuration to existing [install] section."
else
  {
    printf '%s\n' "$content"
    [ -n "$content" ] && printf '\n'
    printf '[install]\n%s\n' "$registry_line"
  } >"$tmp_file"
  lr_log "Created [install] section with registry configuration."
fi

mv -f "$tmp_file" "$bunfig_path"
lr_log "Configuration complete."
