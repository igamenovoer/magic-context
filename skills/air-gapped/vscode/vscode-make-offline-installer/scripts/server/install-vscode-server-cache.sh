#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Install Microsoft VS Code Server + Remote-SSH cache files for an air-gapped Linux server.

This script handles only installation (cache placement + extraction). Configure and extensions are separate scripts.

Usage:
  install-vscode-server-cache.sh --commit <COMMIT> --server-tar <PATH> --cli-tar <PATH> [--user <USERNAME>] [--force]

Args:
  --commit      VS Code build commit hash (40 chars; line 2 of `code --version` on the client)
  --server-tar  Path to server tarball (vscode-server-linux-<arch>-<COMMIT>.tar.gz)
  --cli-tar     Path to CLI tarball (vscode-cli-alpine-<arch>-<COMMIT>.tar.gz)
  --user        Install for this Linux user (writes into ~<user>/.vscode-server). Default: executing user.
  --force       Overwrite cache files and re-extract even if already present.

Notes:
  - If --user differs from the executing user, run as root/admin (or via sudo) so ownership can be fixed.
EOF
}

commit=""
server_tar=""
cli_tar=""
install_user=""
force="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --commit) commit="${2:-}"; shift 2 ;;
        --server-tar) server_tar="${2:-}"; shift 2 ;;
        --cli-tar) cli_tar="${2:-}"; shift 2 ;;
        --user) install_user="${2:-}"; shift 2 ;;
        --force) force="1"; shift 1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "${commit}" || -z "${server_tar}" || -z "${cli_tar}" ]]; then
    echo "Missing required arguments." >&2
    usage
    exit 2
fi

if [[ ! "${commit}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Commit must be 40 lowercase hex chars: ${commit}" >&2
    exit 1
fi

if [[ ! -f "${server_tar}" ]]; then
    echo "Server tarball not found: ${server_tar}" >&2
    exit 1
fi

if [[ ! -f "${cli_tar}" ]]; then
    echo "CLI tarball not found: ${cli_tar}" >&2
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
server_root="${vscode_server_dir}/cli/servers/Stable-${commit}/server"

cli_cache="${vscode_server_dir}/vscode-cli-${commit}.tar.gz"
cli_done="${vscode_server_dir}/vscode-cli-${commit}.tar.gz.done"
server_cache="${vscode_server_dir}/vscode-server.tar.gz"

echo "==> Installing VS Code Server cache"
echo "  USER  : ${install_user}"
echo "  HOME  : ${target_home}"
echo "  COMMIT: ${commit}"
echo "  DIR   : ${vscode_server_dir}"

mkdir -p "${vscode_server_dir}"
mkdir -p "${server_root}"

copy_if_needed() {
    local src="$1"
    local dst="$2"
    if [[ "${force}" -eq 1 ]]; then
        cp -f "${src}" "${dst}"
        return 0
    fi
    if [[ -f "${dst}" ]]; then
        return 0
    fi
    cp -f "${src}" "${dst}"
}

echo "==> Placing cache files"
copy_if_needed "${cli_tar}" "${cli_cache}"
copy_if_needed "${cli_cache}" "${cli_done}"
# The server tarball cache name is not commit-specific; always overwrite so it matches --commit.
cp -f "${server_tar}" "${server_cache}"

echo "==> Extracting server"
if [[ "${force}" -eq 0 && -f "${server_root}/bin/code-server" ]]; then
    echo "  - Already extracted: ${server_root}/bin/code-server"
else
    rm -rf "${server_root}"
    mkdir -p "${server_root}"
    tar -xzf "${server_tar}" --strip-components=1 -C "${server_root}"
fi

if [[ "$(id -u)" -eq 0 ]]; then
    chown -R "${install_user}:${install_user}" "${vscode_server_dir}" || true
fi

echo "==> Done"
