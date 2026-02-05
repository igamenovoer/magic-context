#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Cleanup VS Code Server artifacts under ~/.vscode-server on a Linux server.

This is optional and should be run only after verifying Remote-SSH works for the desired COMMIT(s).

Usage:
  cleanup-vscode-server.sh [--user <USERNAME>] [--remove-cache-tarballs] [--remove-old-servers] [--keep-commit <COMMIT> ...]

Args:
  --user                 Cleanup for this Linux user. Default: executing user.
  --remove-cache-tarballs Remove ~/.vscode-server/vscode-server.tar.gz and vscode-cli-*.tar.gz(.done)
  --remove-old-servers   Remove extracted servers under cli/servers/Stable-* except those listed via --keep-commit
  --keep-commit          Commit hash to keep when using --remove-old-servers (repeatable)

Notes:
  - If --user differs from the executing user, run as root/admin (or via sudo).
EOF
}

install_user=""
remove_cache="0"
remove_old="0"
keep_commits=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --user) install_user="${2:-}"; shift 2 ;;
        --remove-cache-tarballs) remove_cache="1"; shift 1 ;;
        --remove-old-servers) remove_old="1"; shift 1 ;;
        --keep-commit) keep_commits+=( "${2:-}" ); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

current_user="$(id -un)"
if [[ -z "${install_user}" ]]; then
    install_user="${current_user}"
fi

if [[ "${install_user}" != "${current_user}" && "$(id -u)" -ne 0 ]]; then
    echo "ERROR: --user '${install_user}' differs from executing user '${current_user}'." >&2
    echo "Run this script as root (or via sudo) to cleanup for another user." >&2
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
if [[ ! -d "${vscode_server_dir}" ]]; then
    echo "==> Nothing to cleanup (missing): ${vscode_server_dir}"
    exit 0
fi

if [[ "${remove_cache}" -eq 1 ]]; then
    echo "==> Removing cache tarballs"
    rm -f "${vscode_server_dir}/vscode-server.tar.gz" || true
    rm -f "${vscode_server_dir}/vscode-cli-"*.tar.gz || true
    rm -f "${vscode_server_dir}/vscode-cli-"*.tar.gz.done || true
fi

if [[ "${remove_old}" -eq 1 ]]; then
    servers_dir="${vscode_server_dir}/cli/servers"
    if [[ -d "${servers_dir}" ]]; then
        echo "==> Removing old extracted servers under: ${servers_dir}"
        for d in "${servers_dir}"/Stable-*; do
            [[ -d "${d}" ]] || continue
            commit="${d##*/Stable-}"
            keep="0"
            for k in "${keep_commits[@]}"; do
                if [[ "${k}" == "${commit}" ]]; then
                    keep="1"
                    break
                fi
            done
            if [[ "${keep}" -eq 1 ]]; then
                echo "  - KEEP: Stable-${commit}"
            else
                echo "  - REMOVE: Stable-${commit}"
                rm -rf "${d}"
            fi
        done
    fi
fi

if [[ "$(id -u)" -eq 0 ]]; then
    chown -R "${install_user}:${install_user}" "${vscode_server_dir}" || true
fi

echo "==> Done"

