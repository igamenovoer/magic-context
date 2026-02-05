#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Configure VS Code Server state under ~/.vscode-server for an air-gapped Linux server.

This script is separate from installation and extension installation.

Usage:
  configure-vscode-server.sh --commit <COMMIT> [--user <USERNAME>] [--settings-file <PATH>]

What it does:
  - Ensures ~/.vscode-server/data/Machine/settings.json exists (default: {})
  - Touches the server readiness marker: cli/servers/Stable-<COMMIT>/server/.ready

Args:
  --commit  VS Code commit hash (40 chars)
  --user    Configure for this Linux user. Default: executing user.
  --settings-file Optional JSON file to copy to data/Machine/settings.json

Notes:
  - If --user differs from the executing user, run as root/admin (or via sudo).
EOF
}

commit=""
install_user=""
settings_file=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --commit) commit="${2:-}"; shift 2 ;;
        --user) install_user="${2:-}"; shift 2 ;;
        --settings-file) settings_file="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "${commit}" ]]; then
    echo "Missing --commit." >&2
    usage
    exit 2
fi

current_user="$(id -un)"
if [[ -z "${install_user}" ]]; then
    install_user="${current_user}"
fi

if [[ "${install_user}" != "${current_user}" && "$(id -u)" -ne 0 ]]; then
    echo "ERROR: --user '${install_user}' differs from executing user '${current_user}'." >&2
    echo "Run this script as root (or via sudo) to configure for another user." >&2
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
settings_dir="${vscode_server_dir}/data/Machine"
settings_json="${settings_dir}/settings.json"

mkdir -p "${settings_dir}"
if [[ -n "${settings_file}" ]]; then
    if [[ ! -f "${settings_file}" ]]; then
        echo "ERROR: settings file not found: ${settings_file}" >&2
        exit 1
    fi
    cp -f "${settings_file}" "${settings_json}"
else
    if [[ ! -f "${settings_json}" ]]; then
        echo '{}' > "${settings_json}"
    fi
fi

mkdir -p "${server_root}"
touch "${server_root}/.ready"

if [[ "$(id -u)" -eq 0 ]]; then
    chown -R "${install_user}:${install_user}" "${vscode_server_dir}" || true
fi

echo "==> Configured: ${vscode_server_dir}"
