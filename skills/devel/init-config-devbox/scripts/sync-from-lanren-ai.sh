#!/usr/bin/env bash
set -euo pipefail

REPO="igamenovoer/lanren-ai"
REF="main"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/sync-from-lanren-ai.sh [options]

Sync vendored setup scripts from raw GitHub URLs (no git clone/checkout).

Options:
  --repo <owner/name>   GitHub repo (default: igamenovoer/lanren-ai)
  --ref <ref>           Branch/tag/commit (default: main)
  --dry-run             Print planned downloads only
  -h, --help            Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      REPO="${2:-}"
      shift 2
      ;;
    --repo=*)
      REPO="${1#*=}"
      shift
      ;;
    --ref)
      REF="${2:-}"
      shift 2
      ;;
    --ref=*)
      REF="${1#*=}"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$REPO" || -z "$REF" ]]; then
  echo "repo/ref cannot be empty" >&2
  exit 1
fi

ROOT_DIR="$(CDPATH= cd "$(dirname "$0")/.." && pwd)"
BASE_URL="https://raw.githubusercontent.com/${REPO}/${REF}/components"

FILES=(
  "nodejs/README.md"
  "nodejs/install-comp.sh"
  "nodejs/config-comp.sh"
  "bun/README.md"
  "bun/install-comp.sh"
  "bun/config-comp.sh"
  "uv/README.md"
  "uv/install-comp.sh"
  "uv/config-comp.sh"
  "pixi/README.md"
  "pixi/install-comp.sh"
  "pixi/config-comp.sh"
  "claude-code-cli/README.md"
  "claude-code-cli/install-comp.sh"
  "claude-code-cli/config-skip-login.sh"
  "claude-code-cli/config-custom-api-key.sh"
  "claude-code-cli/config-tavily-mcp.sh"
  "claude-code-cli/config-context7-mcp.sh"
)

download() {
  local url="$1"
  local out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$out"
    return
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -q "$url" -O "$out"
    return
  fi
  echo "Need curl or wget to download: $url" >&2
  exit 1
}

echo "[sync] repo: $REPO"
echo "[sync] ref:  $REF"
echo "[sync] root: $ROOT_DIR/scripts"

for rel in "${FILES[@]}"; do
  src="${BASE_URL}/${rel}"
  dst="${ROOT_DIR}/scripts/${rel}"
  mkdir -p "$(dirname "$dst")"

  echo "[sync] $src -> $dst"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    download "$src" "$dst"
  fi
done

if [[ "$DRY_RUN" -eq 0 ]]; then
  find "${ROOT_DIR}/scripts" -type f -name "*.sh" -exec chmod +x {} +

  cat > "${ROOT_DIR}/scripts/README.md" <<EOF
# Vendored setup scripts

These scripts are synced from raw GitHub URLs (no git clone/checkout):
- repo: https://github.com/${REPO}
- raw base: https://raw.githubusercontent.com/${REPO}/${REF}/components
- ref: ${REF}

Vendored components in this workspace:
- scripts/nodejs
- scripts/bun
- scripts/uv
- scripts/pixi
- scripts/claude-code-cli

Sync command:
- bash scripts/sync-from-lanren-ai.sh --ref ${REF}

Purpose:
- Keep host installation/bootstrap scripts local so future runs do not require cloning the full lanren-ai repository.
EOF
fi

echo "[sync] done"
