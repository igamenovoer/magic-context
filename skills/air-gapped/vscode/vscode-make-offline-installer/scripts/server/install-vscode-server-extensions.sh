#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Install remote-side VS Code extensions on a Linux server from a directory of VSIX files.

Usage:
  install-vscode-server-extensions.sh [--commit <COMMIT>] [--extensions-dir <DIR>] [--user <USERNAME>] [--kit-dir <DIR>]

Args:
  --commit         VS Code commit hash (40 chars). If omitted, tries to read it from ./manifest/vscode.json relative to this script.
  --extensions-dir Directory containing *.vsix. If omitted, defaults to ./extensions/remote if present, else ./extensions/local (relative to this script).
  --user           Install for this Linux user. Default: executing user.
  --kit-dir        Optional kit root override (folder containing manifest/, extensions/, scripts/).

Notes:
  - Requires that the server for COMMIT is already extracted:
      ~/.vscode-server/cli/servers/Stable-<COMMIT>/server/bin/code-server
  - Uses a shared extensions dir:
      ~/.vscode-server/extensions
  - Safe to re-run for updates (uses --force).
EOF
}

commit=""
extensions_dir=""
install_user=""
kit_dir=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --commit) commit="${2:-}"; shift 2 ;;
        --extensions-dir) extensions_dir="${2:-}"; shift 2 ;;
        --user) install_user="${2:-}"; shift 2 ;;
        --kit-dir) kit_dir="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "${kit_dir}" ]]; then
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    kit_dir="$(cd -- "${script_dir}/../.." && pwd)"
else
    kit_dir="$(cd -- "${kit_dir}" && pwd)"
fi

read_commit_from_manifest() {
    local manifest=""
    for manifest in "${kit_dir}/manifest/vscode.json" "${kit_dir}/manifest/vscode.local.json"; do
        [[ -f "${manifest}" ]] || continue
        grep -oE '"commit"[[:space:]]*:[[:space:]]*"[0-9a-f]{40}"' "${manifest}" 2>/dev/null | head -n 1 | sed -E 's/.*"([0-9a-f]{40})".*/\\1/' || true
        return 0
    done
    return 1
}

if [[ -z "${commit}" ]]; then
    commit="$(read_commit_from_manifest || true)"
fi

if [[ -z "${extensions_dir}" ]]; then
    detect_arch() {
        local m=""
        m="$(uname -m || true)"
        case "${m}" in
            x86_64|amd64) echo "x64" ;;
            aarch64|arm64) echo "arm64" ;;
            *) echo "x64" ;;
        esac
    }

    arch="$(detect_arch)"

    if ls "${kit_dir}/extensions/remote-linux-${arch}"/*.vsix >/dev/null 2>&1; then
        extensions_dir="${kit_dir}/extensions/remote-linux-${arch}"
    elif ls "${kit_dir}/extensions/remote"/*.vsix >/dev/null 2>&1; then
        extensions_dir="${kit_dir}/extensions/remote"
    else
        extensions_dir="${kit_dir}/extensions/local"
    fi
fi

if [[ -z "${commit}" || -z "${extensions_dir}" ]]; then
    echo "Missing required arguments and could not auto-detect from kit layout." >&2
    usage
    exit 2
fi

if [[ ! -d "${extensions_dir}" ]]; then
    echo "Extensions dir not found: ${extensions_dir}" >&2
    exit 1
fi

current_user="$(id -un)"
if [[ -z "${install_user}" ]]; then
    install_user="${current_user}"
fi

if [[ "${install_user}" != "${current_user}" && "$(id -u)" -ne 0 ]]; then
    echo "ERROR: --user '${install_user}' differs from executing user '${current_user}'." >&2
    echo "Run this script as root (or via sudo) to install for another user." >&2
    exit 1
fi

resolve_home_dir() {
    local user="$1"
    local home=""
    if command -v getent >/dev/null 2>&1; then
        home="$(getent passwd "${user}" | cut -d: -f6 || true)"
    fi
    if [[ -z "${home}" && -r /etc/passwd ]]; then
        home="$(awk -F: -v u="${user}" '$1==u{print $6}' /etc/passwd | head -n 1 || true)"
    fi
    echo "${home}"
}

target_home="$(resolve_home_dir "${install_user}")"
if [[ -z "${target_home}" ]]; then
    echo "ERROR: Could not resolve home directory for user: ${install_user}" >&2
    exit 1
fi

vscode_server_dir="${target_home}/.vscode-server"
server_bin="${vscode_server_dir}/cli/servers/Stable-${commit}/server/bin/code-server"
extensions_target_dir="${vscode_server_dir}/extensions"

if [[ ! -x "${server_bin}" ]]; then
    echo "ERROR: VS Code Server binary not found/executable: ${server_bin}" >&2
    exit 1
fi

mkdir -p "${extensions_target_dir}"

install_one() {
    local vsix="$1"
    "${server_bin}" --install-extension "${vsix}" --force --extensions-dir "${extensions_target_dir}" >/dev/null
}

shopt -s nullglob
vsix_files=( "${extensions_dir}"/*.vsix )
shopt -u nullglob

if [[ ${#vsix_files[@]} -eq 0 ]]; then
    echo "==> No .vsix files found in: ${extensions_dir}"
    exit 0
fi

echo "==> Installing ${#vsix_files[@]} remote extension(s) for user: ${install_user}"
for vsix in "${vsix_files[@]}"; do
    echo "  - $(basename "${vsix}")"
    if [[ "$(id -u)" -eq 0 ]]; then
        su -s /bin/bash -c "$(printf '%q ' "${server_bin}") --install-extension $(printf '%q ' "${vsix}") --force --extensions-dir $(printf '%q ' "${extensions_target_dir}") >/dev/null" "${install_user}"
    else
        install_one "${vsix}"
    fi
done

echo "==> Installed remote extensions:"
if [[ "$(id -u)" -eq 0 ]]; then
    su -s /bin/bash -c "$(printf '%q ' "${server_bin}") --list-extensions --show-versions --extensions-dir $(printf '%q ' "${extensions_target_dir}") || true" "${install_user}"
else
    "${server_bin}" --list-extensions --show-versions --extensions-dir "${extensions_target_dir}" || true
fi
