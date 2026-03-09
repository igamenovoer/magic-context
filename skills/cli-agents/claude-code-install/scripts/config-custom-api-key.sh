#!/usr/bin/env sh
set -eu

usage() {
  cat <<'EOF'
Usage: ./config-custom-api-key.sh [options]

Create a launcher in ~/.local/bin/<alias-name> that exports Anthropic-compatible
environment variables and then runs claude.

Options:
  --alias-name NAME         Launcher name to create
  --base-url URL            Optional custom Anthropic-compatible endpoint
  --api-key KEY             Store the API key in the launcher file
  --api-key-env ENV_NAME    Read the API key from ENV_NAME at runtime
  --primary-model NAME      Optional primary model override
  --secondary-model NAME    Optional secondary model override
  --dry-run                 Print actions without changing the system
  -h, --help                Show this help
EOF
}

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

quote_sh() {
  printf "'%s'" "$(printf '%s' "${1-}" | sed "s/'/'\"'\"'/g")"
}

log() {
  printf '%s\n' "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

alias_name=''
base_url=''
api_key=''
api_key_env=''
primary_model=''
secondary_model=''
dry_run=0

while [ $# -gt 0 ]; do
  case "$1" in
    --alias-name)
      alias_name="${2-}"
      shift 2
      ;;
    --alias-name=*)
      alias_name="${1#*=}"
      shift
      ;;
    --base-url)
      base_url="${2-}"
      shift 2
      ;;
    --base-url=*)
      base_url="${1#*=}"
      shift
      ;;
    --api-key)
      api_key="${2-}"
      shift 2
      ;;
    --api-key=*)
      api_key="${1#*=}"
      shift
      ;;
    --api-key-env)
      api_key_env="${2-}"
      shift 2
      ;;
    --api-key-env=*)
      api_key_env="${1#*=}"
      shift
      ;;
    --primary-model)
      primary_model="${2-}"
      shift 2
      ;;
    --primary-model=*)
      primary_model="${1#*=}"
      shift
      ;;
    --secondary-model)
      secondary_model="${2-}"
      shift 2
      ;;
    --secondary-model=*)
      secondary_model="${1#*=}"
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

has_cmd claude || die "claude is not on PATH. Install Claude Code first."

[ -n "$alias_name" ] || die "--alias-name is required"
printf '%s' "$alias_name" | grep -Eq '^[A-Za-z0-9_-]+$' || die "Alias name may only contain letters, digits, underscores, and hyphens"

if [ -n "$base_url" ]; then
  printf '%s' "$base_url" | grep -Eq '^https?://' || die "--base-url must start with http:// or https://"
fi

if [ -n "$api_key" ] && [ -n "$api_key_env" ]; then
  die "Use either --api-key or --api-key-env, not both"
fi

if [ -z "$api_key" ] && [ -z "$api_key_env" ]; then
  die "One of --api-key or --api-key-env is required"
fi

if [ -n "$api_key_env" ]; then
  printf '%s' "$api_key_env" | grep -Eq '^[A-Za-z_][A-Za-z0-9_]*$' || die "--api-key-env must be a valid environment variable name"
fi

if [ -n "$primary_model" ] && [ -z "$secondary_model" ]; then
  secondary_model="$primary_model"
fi

bin_dir="$HOME/.local/bin"
launcher_path="$bin_dir/$alias_name"

if [ "$dry_run" -eq 1 ]; then
  log "Dry-run: would create launcher at $launcher_path"
  exit 0
fi

mkdir -p "$bin_dir"

base_url_q="$(quote_sh "$base_url")"
api_key_q="$(quote_sh "$api_key")"
primary_q="$(quote_sh "$primary_model")"
secondary_q="$(quote_sh "$secondary_model")"

{
  printf '%s\n' '#!/usr/bin/env sh'
  printf '%s\n' 'set -eu'
  if [ -n "$base_url" ]; then
    printf '%s\n' "export ANTHROPIC_BASE_URL=$base_url_q"
  fi
  if [ -n "$api_key_env" ]; then
    printf '%s\n' "export ANTHROPIC_API_KEY=\"\${$api_key_env:?Environment variable $api_key_env is required}\""
  else
    printf '%s\n' "export ANTHROPIC_API_KEY=$api_key_q"
  fi
  if [ -n "$primary_model" ]; then
    printf '%s\n' "export ANTHROPIC_MODEL=$primary_q"
    printf '%s\n' "export ANTHROPIC_DEFAULT_OPUS_MODEL=$primary_q"
    printf '%s\n' "export ANTHROPIC_DEFAULT_SONNET_MODEL=$primary_q"
    printf '%s\n' "export ANTHROPIC_DEFAULT_HAIKU_MODEL=$secondary_q"
    printf '%s\n' "export CLAUDE_CODE_SUBAGENT_MODEL=$secondary_q"
  fi
  printf '%s\n' 'exec claude --dangerously-skip-permissions "$@"'
} >"$launcher_path"

chmod +x "$launcher_path"
log "Created launcher: $launcher_path"
log "Ensure ~/.local/bin is on PATH before using $alias_name"