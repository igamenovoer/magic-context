#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Install local (client-side) VS Code extensions from a directory of .vsix files.

Usage:
  install-vscode-client-extensions.sh [--extensions-dir <DIR>] [--channel auto|stable|insider] [--required-id <ID> ...] [--kit-dir <DIR>]

Args:
  --extensions-dir   Directory containing *.vsix files. If omitted, defaults to ./extensions/local relative to this script.
  --channel          Which VS Code binary to use. Default: auto (prefer code, else code-insiders).
  --required-id      Extension ID that must be installed (repeatable).
                    Default: ms-vscode-remote.remote-ssh
  --kit-dir          Optional kit root override (folder containing extensions/, scripts/).
EOF
}

extensions_dir=""
channel="auto"
required_ids=( "ms-vscode-remote.remote-ssh" )
kit_dir=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --extensions-dir) extensions_dir="${2:-}"; shift 2 ;;
        --channel) channel="${2:-}"; shift 2 ;;
        --required-id) required_ids+=( "${2:-}" ); shift 2 ;;
        --kit-dir) kit_dir="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "${extensions_dir}" ]]; then
    if [[ -z "${kit_dir}" ]]; then
        script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
        kit_dir="$(cd -- "${script_dir}/../.." && pwd)"
    else
        kit_dir="$(cd -- "${kit_dir}" && pwd)"
    fi

    os="$(uname -s | tr '[:upper:]' '[:lower:]' || true)"
    if [[ "${os}" == "linux" ]]; then
        m="$(uname -m || true)"
        arch="x64"
        case "${m}" in
            x86_64|amd64) arch="x64" ;;
            aarch64|arm64) arch="arm64" ;;
            *) arch="x64" ;;
        esac

        if [[ -d "${kit_dir}/extensions/local-linux-${arch}" ]]; then
            extensions_dir="${kit_dir}/extensions/local-linux-${arch}"
        else
            extensions_dir="${kit_dir}/extensions/local"
        fi
    else
        extensions_dir="${kit_dir}/extensions/local"
    fi
fi

if [[ ! -d "${extensions_dir}" ]]; then
    echo "Extensions dir not found: ${extensions_dir}" >&2
    exit 1
fi

resolve_code_cmd() {
    local want="$1"
    local stable=""
    local insiders=""
    stable="$(command -v code 2>/dev/null || true)"
    insiders="$(command -v code-insiders 2>/dev/null || true)"

    case "${want}" in
        stable)
            [[ -n "${stable}" ]] && echo "${stable}" && return 0
            echo "Requested channel=stable but 'code' is not on PATH." >&2
            return 1
            ;;
        insider|insiders)
            [[ -n "${insiders}" ]] && echo "${insiders}" && return 0
            echo "Requested channel=insider but 'code-insiders' is not on PATH." >&2
            return 1
            ;;
        auto)
            [[ -n "${stable}" ]] && echo "${stable}" && return 0
            [[ -n "${insiders}" ]] && echo "${insiders}" && return 0
            echo "Neither 'code' nor 'code-insiders' found on PATH." >&2
            return 1
            ;;
        *)
            echo "Invalid --channel: ${want} (expected auto|stable|insider)" >&2
            return 1
            ;;
    esac
}

code_cmd="$(resolve_code_cmd "${channel}")"

shopt -s nullglob
vsix_files=( "${extensions_dir}"/*.vsix )
shopt -u nullglob

if [[ ${#vsix_files[@]} -eq 0 ]]; then
    echo "No .vsix files found in: ${extensions_dir}" >&2
    exit 1
fi

echo "==> Installing ${#vsix_files[@]} extension(s) from: ${extensions_dir}"
for f in "${vsix_files[@]}"; do
    echo "  - $(basename "${f}")"
    "${code_cmd}" --install-extension "${f}" --force >/dev/null
done

echo "==> Installed extensions (local):"
installed="$("${code_cmd}" --list-extensions --show-versions)"
echo "${installed}" | sed 's/^/  /'

for rid in "${required_ids[@]}"; do
    rid="$(printf '%s' "${rid}" | xargs)"
    [[ -z "${rid}" ]] && continue
    if ! printf '%s\n' "${installed}" | grep -qiF "${rid}"; then
        echo "Required extension not installed: ${rid}" >&2
        exit 1
    fi
done
