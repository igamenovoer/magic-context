#!/usr/bin/env sh
set -eu

usage() {
  cat <<'EOF'
Usage: ./config-custom-api-key.sh [options]

Create a shell function or optional launcher for Codex with a custom endpoint and API key,
and update CODEX_HOME/config.toml to disable built-in login for the matching provider.

Options:
  --alias-name NAME         Alias or launcher name to create
  --provider-id NAME        Provider id for config.toml (defaults to alias name)
  --base-url URL            Optional custom OpenAI-compatible endpoint
  --api-key KEY             Store the API key directly in the generated command
  --api-key-env ENV_NAME    Read the API key from ENV_NAME at runtime
  --profile-file PATH       Explicit shell profile file to update
  --separate-launcher       Create ~/.local/bin/<alias-name> instead of a profile function
  --dry-run                 Print actions without changing the system
  -h, --help                Show this help
EOF
}

has_cmd() { command -v "$1" >/dev/null 2>&1; }
quote_sh() { printf "'%s'" "$(printf '%s' "${1-}" | sed "s/'/'\"'\"'/g")"; }
log() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

detect_profile_file() {
  shell_name="$(basename "${SHELL:-}")"
  case "$shell_name" in
    bash) printf '%s\n' "$HOME/.bashrc" ;;
    zsh) printf '%s\n' "$HOME/.zshrc" ;;
    *) return 1 ;;
  esac
}

update_profile_block() {
  profile_path="$1"
  begin_marker="$2"
  end_marker="$3"
  block_file="$4"
  mkdir -p "$(dirname "$profile_path")"
  touch "$profile_path"
  tmp_file="$(mktemp "${TMPDIR:-/tmp}/codex-profile.XXXXXX" 2>/dev/null || echo "${TMPDIR:-/tmp}/codex-profile.$$")"
  awk -v begin="$begin_marker" -v end="$end_marker" '
    $0==begin {skip=1; next}
    $0==end {skip=0; next}
    skip!=1 {print}
  ' "$profile_path" > "$tmp_file"
  mv -f "$tmp_file" "$profile_path"
  cat "$block_file" >> "$profile_path"
}

alias_name=''
provider_id=''
base_url=''
api_key=''
api_key_env=''
profile_file=''
separate_launcher=0
dry_run=0

while [ $# -gt 0 ]; do
  case "$1" in
    --alias-name) alias_name="${2-}"; shift 2 ;;
    --alias-name=*) alias_name="${1#*=}"; shift ;;
    --provider-id) provider_id="${2-}"; shift 2 ;;
    --provider-id=*) provider_id="${1#*=}"; shift ;;
    --base-url) base_url="${2-}"; shift 2 ;;
    --base-url=*) base_url="${1#*=}"; shift ;;
    --api-key) api_key="${2-}"; shift 2 ;;
    --api-key=*) api_key="${1#*=}"; shift ;;
    --api-key-env) api_key_env="${2-}"; shift 2 ;;
    --api-key-env=*) api_key_env="${1#*=}"; shift ;;
    --profile-file) profile_file="${2-}"; shift 2 ;;
    --profile-file=*) profile_file="${1#*=}"; shift ;;
    --separate-launcher) separate_launcher=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "Unknown argument: $1" ;;
  esac
done

has_cmd codex || die "codex is not on PATH. Install Codex CLI first."
[ -n "$alias_name" ] || die "--alias-name is required"
printf '%s' "$alias_name" | grep -Eq '^[A-Za-z0-9_-]+$' || die "Alias name may only contain letters, digits, underscores, and hyphens"
if [ -z "$provider_id" ]; then provider_id="$alias_name"; fi
printf '%s' "$provider_id" | grep -Eq '^[A-Za-z0-9_-]+$' || die "Provider id may only contain letters, digits, underscores, and hyphens"
if [ "$provider_id" = 'openai' ] || [ "$provider_id" = 'OpenAI' ]; then die "Provider id 'openai' is reserved; choose a different value."; fi
if [ -n "$base_url" ]; then printf '%s' "$base_url" | grep -Eq '^https?://' || die "Base URL must start with http:// or https://"; fi
if [ -n "$api_key" ] && [ -n "$api_key_env" ]; then die "Use either --api-key or --api-key-env, not both"; fi
if [ -z "$api_key" ] && [ -z "$api_key_env" ]; then die "One of --api-key or --api-key-env is required"; fi
if [ -n "$api_key_env" ]; then printf '%s' "$api_key_env" | grep -Eq '^[A-Za-z_][A-Za-z0-9_]*$' || die "--api-key-env must be a valid environment variable name"; fi

if [ "$separate_launcher" -ne 1 ] && [ -z "$profile_file" ]; then
  profile_file="$(detect_profile_file || true)"
  [ -n "$profile_file" ] || die "Could not infer a shell profile file. Pass --profile-file or use --separate-launcher."
fi

codex_home="${CODEX_HOME:-$HOME/.codex}"
config_path="$codex_home/config.toml"
effective_base_url="$base_url"
if [ -z "$effective_base_url" ]; then effective_base_url='https://api.openai.com/v1'; fi

if [ "$dry_run" -eq 1 ]; then
  if [ "$separate_launcher" -eq 1 ]; then
    log "Dry-run: would create launcher ~/.local/bin/$alias_name and update $config_path"
  else
    log "Dry-run: would update profile $profile_file and update $config_path"
  fi
  exit 0
fi

mkdir -p "$codex_home"
sh "$(dirname "$0")/config-skip-login.sh" --provider-id "$provider_id" --base-url "$effective_base_url" --env-key OPENAI_API_KEY

base_url_q="$(quote_sh "$base_url")"
api_key_q="$(quote_sh "$api_key")"

if [ "$separate_launcher" -eq 1 ]; then
  launcher_dir="$HOME/.local/bin"
  launcher_path="$launcher_dir/$alias_name"
  mkdir -p "$launcher_dir"
  {
    printf '%s\n' '#!/usr/bin/env sh'
    printf '%s\n' 'set -eu'
    if [ -n "$base_url" ]; then printf '%s\n' "export OPENAI_BASE_URL=$base_url_q"; fi
    if [ -n "$api_key_env" ]; then
      printf '%s\n' "export OPENAI_API_KEY=\"\${$api_key_env:?Environment variable $api_key_env is required}\""
    else
      printf '%s\n' "export OPENAI_API_KEY=$api_key_q"
    fi
    printf '%s\n' 'exec codex --search "$@"'
  } > "$launcher_path"
  chmod +x "$launcher_path"
  log "Created launcher: $launcher_path"
else
  begin_marker="# BEGIN: Codex custom endpoint ($alias_name)"
  end_marker="# END: Codex custom endpoint ($alias_name)"
  block_file="$(mktemp "${TMPDIR:-/tmp}/codex-block.XXXXXX" 2>/dev/null || echo "${TMPDIR:-/tmp}/codex-block.$$")"
  {
    printf '\n%s\n' "$begin_marker"
    printf '%s\n' "$alias_name() {"
    if [ -n "$base_url" ]; then printf '  export OPENAI_BASE_URL=%s\n' "$base_url_q"; fi
    if [ -n "$api_key_env" ]; then
      printf '  export OPENAI_API_KEY="${%s:?%s is required}"\n' "$api_key_env" "$api_key_env"
    else
      printf '  export OPENAI_API_KEY=%s\n' "$api_key_q"
    fi
    printf '%s\n' '  codex --search "$@"'
    printf '%s\n' '}'
    printf '%s\n' "$end_marker"
  } > "$block_file"
  update_profile_block "$profile_file" "$begin_marker" "$end_marker" "$block_file"
  rm -f "$block_file"
  log "Updated profile: $profile_file"
fi

log "Updated Codex provider config: $config_path"