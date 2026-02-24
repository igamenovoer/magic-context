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

Configure uv global mirror (PyPI index URL) on Linux/macOS.

Options:
  --mirror <cn|aliyun|tuna|official>   Mirror preset (default: cn -> aliyun)
  --dry-run                            Print what would change, without writing files
  --capture-log-file PATH              Also write logs to PATH
  -h, --help                           Show this help
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
  mirror="cn"
fi

case "$mirror" in
  cn) mirror="aliyun" ;;
  aliyun|tuna|official) ;;
  *) echo "Invalid --mirror: $mirror" >&2; usage; exit 1 ;;
esac

case "$mirror" in
  aliyun) registry_url="https://mirrors.aliyun.com/pypi/simple/" ;;
  tuna) registry_url="https://pypi.tuna.tsinghua.edu.cn/simple" ;;
  official) registry_url="https://pypi.org/simple" ;;
esac

component_name=$(basename "$script_dir")
lr_init_component_log "$component_name" "$capture_log_file" "$dry_run"
lr_log "=== Configuring uv Global Mirror ==="
lr_log "Selected Mirror: $mirror ($registry_url)"

config_root="${XDG_CONFIG_HOME:-${HOME:-}/.config}"
config_dir="$config_root/uv"
config_path="$config_dir/uv.toml"

lr_log "Config file: $config_path"

if [ "$dry_run" -eq 1 ]; then
  lr_log "Dry-run: would ensure $config_dir exists and set:"
  lr_log "index-url = \"$registry_url\""
  exit 0
fi

mkdir -p "$config_dir"
content=""
if [ -f "$config_path" ]; then
  content="$(cat "$config_path" 2>/dev/null || true)"
fi

tmp_file="$(mktemp "${TMPDIR:-/tmp}/uv.toml.XXXXXX" 2>/dev/null || echo "${TMPDIR:-/tmp}/uv.toml.$$")"

if printf '%s\n' "$content" | grep -Eq '^[[:space:]]*index-url[[:space:]]*='; then
  printf '%s\n' "$content" | awk -v url="$registry_url" '
    BEGIN {done=0}
    {
      if (!done && $0 ~ /^[[:space:]]*index-url[[:space:]]*=/) {
        print "index-url = \"" url "\""
        done=1
        next
      }
      print
    }
  ' >"$tmp_file"
  lr_log "Updated existing index-url configuration."
else
  # Append
  if [ -n "$content" ] && [ "$(printf '%s' "$content" | tail -c 1 2>/dev/null || echo "")" != "" ]; then
    : # best-effort; we'll ensure trailing newline below
  fi
  {
    printf '%s\n' "$content"
    [ -n "$content" ] && printf '\n'
    printf 'index-url = "%s"\n' "$registry_url"
  } >"$tmp_file"
  lr_log "Added index-url configuration."
fi

mv -f "$tmp_file" "$config_path"
lr_log "Configuration complete."
