#!/usr/bin/env sh
set -eu

usage() {
  cat <<'EOF'
Usage: ./config-skip-login.sh [options]

Configure Codex to use a custom provider in config.toml and disable built-in login.

Options:
  --provider-id NAME    Provider id to configure (default: codex-custom)
  --base-url URL        Base URL for the provider (default: https://api.openai.com/v1)
  --env-key NAME        Environment variable name for the API key (default: OPENAI_API_KEY)
  --dry-run             Print actions without changing the system
  -h, --help            Show this help
EOF
}

has_cmd() { command -v "$1" >/dev/null 2>&1; }
log() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

provider_id='codex-custom'
base_url='https://api.openai.com/v1'
env_key='OPENAI_API_KEY'
dry_run=0

while [ $# -gt 0 ]; do
  case "$1" in
    --provider-id) provider_id="${2-}"; shift 2 ;;
    --provider-id=*) provider_id="${1#*=}"; shift ;;
    --base-url) base_url="${2-}"; shift 2 ;;
    --base-url=*) base_url="${1#*=}"; shift ;;
    --env-key) env_key="${2-}"; shift 2 ;;
    --env-key=*) env_key="${1#*=}"; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "Unknown argument: $1" ;;
  esac
done

has_cmd codex || die "codex is not on PATH. Install Codex CLI first."
printf '%s' "$provider_id" | grep -Eq '^[A-Za-z0-9_-]+$' || die "Invalid provider id."
printf '%s' "$env_key" | grep -Eq '^[A-Za-z_][A-Za-z0-9_]*$' || die "Invalid env key name."
printf '%s' "$base_url" | grep -Eq '^https?://' || die "Base URL must start with http:// or https://"

codex_home="${CODEX_HOME:-$HOME/.codex}"
config_path="$codex_home/config.toml"

if [ "$dry_run" -eq 1 ]; then
  log "Dry-run: would configure provider '$provider_id' in $config_path"
  exit 0
fi

mkdir -p "$codex_home"
touch "$config_path"

tmp_file="$(mktemp "${TMPDIR:-/tmp}/codex-skip-login.XXXXXX" 2>/dev/null || echo "${TMPDIR:-/tmp}/codex-skip-login.$$")"

awk -v provider="$provider_id" '
  BEGIN {skip=0; printed=0; has_root=0}
  /^[[:space:]]*\[model_providers\][[:space:]]*$/ {has_root=1}
  /^[[:space:]]*model_provider[[:space:]]*=/ {
    if (printed==0) {
      print "model_provider = \"" provider "\""
      printed=1
    }
    next
  }
  $0 ~ "^[[:space:]]*\\[model_providers\\." provider "\\][[:space:]]*$" {skip=1; next}
  skip==1 {
    if ($0 ~ /^[[:space:]]*\[/) { skip=0 } else { next }
  }
  { print }
  END {
    if (printed==0) print "model_provider = \"" provider "\""
    if (has_root==0) {
      print ""
      print "[model_providers]"
    }
  }
' "$config_path" > "$tmp_file"

mv -f "$tmp_file" "$config_path"

{
  printf '\n[model_providers.%s]\n' "$provider_id"
  printf '%s\n' 'name = "Custom OpenAI-compatible endpoint"'
  printf 'base_url = "%s"\n' "$base_url"
  printf 'env_key = "%s"\n' "$env_key"
  printf 'env_key_instructions = "Set %s in your environment before launching codex."\n' "$env_key"
  printf '%s\n' 'requires_openai_auth = false'
  printf '%s\n' 'wire_api = "responses"'
} >> "$config_path"

log "Updated $config_path for provider '$provider_id'"