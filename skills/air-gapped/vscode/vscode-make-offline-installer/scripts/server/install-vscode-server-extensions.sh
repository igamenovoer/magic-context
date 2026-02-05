#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Install remote-side VS Code extensions on a Linux server from a directory of VSIX files.

Usage:
  install-vscode-server-extensions.sh --commit <COMMIT> --extensions-dir <DIR> [--user <USERNAME>]

Args:
  --commit         VS Code commit hash (40 chars)
  --extensions-dir Directory containing *.vsix
  --user           Install for this Linux user. Default: executing user.

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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --commit) commit="${2:-}"; shift 2 ;;
        --extensions-dir) extensions_dir="${2:-}"; shift 2 ;;
        --user) install_user="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "${commit}" || -z "${extensions_dir}" ]]; then
    echo "Missing required arguments." >&2
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

