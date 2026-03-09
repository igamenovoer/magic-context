#!/usr/bin/env sh
set -eu

usage() {
  cat <<'EOF'
Usage: ./install-comp.sh [options]

Install Claude Code CLI (@anthropic-ai/claude-code) with Bun or npm.

Options:
  --proxy URL         HTTP/HTTPS proxy to export for this run
  --from-official     Use https://registry.npmjs.org instead of npmmirror
  --force             Reinstall even if claude is already available
  --skip-onboarding   Also run config-skip-login.sh after install
  --dry-run           Print actions without changing the system
  -h, --help          Show this help
EOF
}

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

log() {
  printf '%s\n' "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

set_proxy_env() {
  proxy="${1-}"
  if [ -n "$proxy" ]; then
    export HTTP_PROXY="$proxy" HTTPS_PROXY="$proxy"
    export http_proxy="$proxy" https_proxy="$proxy"
  fi
}

ensure_user_bins_on_path() {
  for dir in "$HOME/.local/bin" "$HOME/bin"; do
    if [ -d "$dir" ]; then
      PATH="$dir:$PATH"
    fi
  done
  export PATH
}

ensure_node_bins_on_path() {
  if has_cmd npm; then
    npm_bin="$(npm bin -g 2>/dev/null || true)"
    if [ -n "$npm_bin" ] && [ -d "$npm_bin" ]; then
      PATH="$npm_bin:$PATH"
      export PATH
    fi
  fi
}

proxy=""
from_official=0
force=0
skip_onboarding=0
dry_run=0

while [ $# -gt 0 ]; do
  case "$1" in
    --proxy|--proxy-url)
      proxy="${2-}"
      shift 2
      ;;
    --proxy=*|--proxy-url=*)
      proxy="${1#*=}"
      shift
      ;;
    --from-official)
      from_official=1
      shift
      ;;
    --force)
      force=1
      shift
      ;;
    --skip-onboarding)
      skip_onboarding=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "Unknown argument: $1"
      ;;
  esac
done

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
package_name='@anthropic-ai/claude-code'
mirror_registry='https://registry.npmmirror.com'
official_registry='https://registry.npmjs.org'

set_proxy_env "$proxy"
ensure_user_bins_on_path

if has_cmd claude && [ "$force" -ne 1 ]; then
  log "claude is already available on PATH. Use --force to reinstall."
  exit 0
fi

runner=''
if has_cmd bun; then
  runner='bun'
elif has_cmd npm; then
  runner='npm'
else
  die "Neither bun nor npm is available. Install Bun or Node.js first."
fi

registry="$mirror_registry"
if [ "$from_official" -eq 1 ]; then
  registry="$official_registry"
fi

if [ "$dry_run" -eq 1 ]; then
  log "Dry-run: would install $package_name using $runner from $registry"
  if [ "$skip_onboarding" -eq 1 ]; then
    log "Dry-run: would run $script_dir/config-skip-login.sh"
  fi
  exit 0
fi

if [ "$runner" = 'bun' ]; then
  log "Installing $package_name with bun from $registry"
  if ! bun add -g "$package_name" --registry "$registry"; then
    if [ "$from_official" -eq 1 ]; then
      die "bun installation failed via official registry"
    fi
    log "Mirror install failed. Retrying with $official_registry"
    bun add -g "$package_name" --registry "$official_registry"
  fi
else
  log "Installing $package_name with npm from $registry"
  if ! npm install -g "$package_name" --registry "$registry"; then
    if [ "$from_official" -eq 1 ]; then
      die "npm installation failed via official registry"
    fi
    log "Mirror install failed. Retrying with $official_registry"
    npm install -g "$package_name" --registry "$official_registry"
  fi
fi

ensure_node_bins_on_path
ensure_user_bins_on_path

has_cmd claude || die "Installation finished, but 'claude' is still not on PATH. Check your global package bin directory."
log "Installed Claude Code CLI: $(command -v claude)"

if [ "$skip_onboarding" -eq 1 ]; then
  sh "$script_dir/config-skip-login.sh"
fi