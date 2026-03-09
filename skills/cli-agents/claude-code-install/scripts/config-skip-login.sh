#!/usr/bin/env sh
set -eu

usage() {
  cat <<'EOF'
Usage: ./config-skip-login.sh [options]

Mark Claude Code onboarding as completed by setting
hasCompletedOnboarding=true in ~/.claude.json.

Options:
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

dry_run=0

while [ $# -gt 0 ]; do
  case "$1" in
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
has_cmd node || die "node is required to update ~/.claude.json safely."

config_file="$HOME/.claude.json"

if [ "$dry_run" -eq 1 ]; then
  log "Dry-run: would set hasCompletedOnboarding=true in $config_file"
  exit 0
fi

node - "$config_file" <<'NODE'
const fs = require('fs');
const target = process.argv[2];
let data = {};
try {
  if (fs.existsSync(target)) {
    const text = fs.readFileSync(target, 'utf8').trim();
    if (text) {
      data = JSON.parse(text);
    }
  }
} catch {
  data = {};
}
data.hasCompletedOnboarding = true;
fs.writeFileSync(target, JSON.stringify(data, null, 2) + '\n', 'utf8');
NODE

log "Updated $config_file with hasCompletedOnboarding=true"