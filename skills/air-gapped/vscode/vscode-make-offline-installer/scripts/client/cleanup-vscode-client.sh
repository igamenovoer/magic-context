#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Cleanup optional files on an air-gapped client after installation/update.

Deletes selected package artifacts under a directory. Use only after verifying VS Code + extensions work.

Usage:
  cleanup-vscode-client.sh --package-dir <DIR> [--remove-installers] [--remove-vsix] [--dry-run]

Args:
  --package-dir       Directory containing offline installers/archives and/or VSIX files.
  --remove-installers Remove common VS Code installer/archive files (*.exe, *.msi, *.zip, *.dmg, *.deb, *.rpm, *.tar.gz).
  --remove-vsix       Remove *.vsix files.
  --dry-run           Print what would be removed.
EOF
}

package_dir=""
remove_installers="0"
remove_vsix="0"
dry_run="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --package-dir) package_dir="${2:-}"; shift 2 ;;
        --remove-installers) remove_installers="1"; shift 1 ;;
        --remove-vsix) remove_vsix="1"; shift 1 ;;
        --dry-run) dry_run="1"; shift 1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "${package_dir}" ]]; then
    echo "Missing --package-dir." >&2
    usage
    exit 2
fi

if [[ ! -d "${package_dir}" ]]; then
    echo "Package dir not found: ${package_dir}" >&2
    exit 1
fi

patterns=()
if [[ "${remove_installers}" -eq 1 ]]; then
    patterns+=( "*.exe" "*.msi" "*.zip" "*.dmg" "*.deb" "*.rpm" "*.tar.gz" )
fi
if [[ "${remove_vsix}" -eq 1 ]]; then
    patterns+=( "*.vsix" )
fi

if [[ ${#patterns[@]} -eq 0 ]]; then
    echo "Nothing selected. Use --remove-installers and/or --remove-vsix." >&2
    exit 2
fi

echo "==> Cleanup in: ${package_dir}"

find_args=( "${package_dir}" -type f "(" )
for i in "${!patterns[@]}"; do
    pat="${patterns[$i]}"
    if [[ "${i}" -gt 0 ]]; then
        find_args+=( -o )
    fi
    find_args+=( -name "${pat}" )
done
find_args+=( ")" )

set +e
mapfile -t matches < <(find "${find_args[@]}" 2>/dev/null)
set -e

if [[ ${#matches[@]} -eq 0 ]]; then
    echo "==> Nothing matched."
    exit 0
fi

for f in "${matches[@]}"; do
    if [[ "${dry_run}" -eq 1 ]]; then
        echo "  DRY-RUN: ${f}"
    else
        echo "  REMOVE: ${f}"
        rm -f -- "${f}"
    fi
done

echo "==> Done."
