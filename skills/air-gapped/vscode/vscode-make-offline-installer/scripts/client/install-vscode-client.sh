#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Install VS Code on an air-gapped Linux desktop client (Ubuntu recommended) using a pre-downloaded package.

Usage:
  install-vscode-client.sh [--installer-path <PATH>] [--channel auto|stable|insider] [--kit-dir <DIR>]

Args:
  --installer-path   Path to offline VS Code package (recommended: .deb for Ubuntu Desktop). If omitted, tries to locate one under ./clients/ relative to this script.
  --channel          Used only for printing the currently installed VS Code (auto|stable|insider). Default: auto.
  --kit-dir          Optional kit root override (folder containing clients/, extensions/, scripts/).

Notes:
  - Ubuntu Desktop: prefer the "linux-deb-<arch>" artifact and install via dpkg.
  - If dpkg reports missing dependencies, you must also provide those dependency .deb files offline.
EOF
}

installer_path=""
channel="auto"
kit_dir=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --installer-path) installer_path="${2:-}"; shift 2 ;;
        --channel) channel="${2:-}"; shift 2 ;;
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

if [[ -z "${installer_path}" ]]; then
    detect_arch() {
        local m=""
        m="$(uname -m || true)"
        case "${m}" in
            x86_64|amd64) echo "x64" ;;
            aarch64|arm64) echo "arm64" ;;
            *) echo "x64" ;;
        esac
    }

    pick_matching() {
        local arch="$1"
        shift
        local f=""
        for f in "$@"; do
            local b=""
            b="$(basename "${f}")"
            case "${arch}" in
                x64)
                    if [[ "${b}" =~ (amd64|x86_64|x64) ]]; then
                        printf '%s\n' "${f}"
                        return 0
                    fi
                    ;;
                arm64)
                    if [[ "${b}" =~ (arm64|aarch64) ]]; then
                        printf '%s\n' "${f}"
                        return 0
                    fi
                    ;;
            esac
        done
        if [[ $# -gt 0 ]]; then
            printf '%s\n' "$1"
            return 0
        fi
        return 1
    }

    try_dir() {
        local d="$1"
        [[ -d "${d}" ]] || return 1

        shopt -s nullglob
        local debs=( "${d}"/*.deb )
        local rpms=( "${d}"/*.rpm )
        shopt -u nullglob

        if [[ ${#debs[@]} -gt 0 ]]; then
            installer_path="$(pick_matching "${arch}" "${debs[@]}")"
            return 0
        fi
        if [[ ${#rpms[@]} -gt 0 ]]; then
            installer_path="$(pick_matching "${arch}" "${rpms[@]}")"
            return 0
        fi
        return 1
    }

    arch="$(detect_arch)"

    # Preferred (platform-reflecting) layout produced by download scripts:
    #   clients/linux-deb-x64/, clients/linux-deb-arm64/, clients/linux-rpm-x64/, ...
    try_dir "${kit_dir}/clients/linux-deb-${arch}" || true
    if [[ -z "${installer_path}" ]]; then
        try_dir "${kit_dir}/clients/linux-rpm-${arch}" || true
    fi

    # Legacy layout:
    if [[ -z "${installer_path}" ]]; then
        try_dir "${kit_dir}/clients/linux" || true
    fi
fi

if [[ -z "${installer_path}" ]]; then
    echo "Missing --installer-path." >&2
    usage
    exit 2
fi

if [[ ! -f "${installer_path}" ]]; then
    echo "Installer not found: ${installer_path}" >&2
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
            return 1
            ;;
        insider|insiders)
            [[ -n "${insiders}" ]] && echo "${insiders}" && return 0
            return 1
            ;;
        auto)
            [[ -n "${stable}" ]] && echo "${stable}" && return 0
            [[ -n "${insiders}" ]] && echo "${insiders}" && return 0
            return 1
            ;;
        *)
            return 1
            ;;
    esac
}

code_cmd=""
if code_cmd="$(resolve_code_cmd "${channel}" 2>/dev/null)"; then
    set +e
    ver_out="$("${code_cmd}" --version 2>/dev/null)"
    set -e
    if [[ -n "${ver_out}" ]]; then
        echo "==> Current VS Code:"
        echo "${ver_out}" | sed -n '1,3p' | sed 's/^/    /'
    fi
else
    echo "==> VS Code not found on PATH (may not be installed yet)."
fi

ext="${installer_path##*.}"
ext="$(printf '%s' "${ext}" | tr '[:upper:]' '[:lower:]')"

if [[ "${ext}" != "deb" && "${ext}" != "rpm" ]]; then
    echo "ERROR: Unsupported installer type for Linux automation: ${installer_path}" >&2
    echo "Provide a .deb (Ubuntu/Debian) or .rpm (Fedora/RHEL) package." >&2
    exit 1
fi

if [[ "${ext}" == "deb" ]]; then
    echo "==> Installing VS Code .deb via dpkg..."
    if command -v sudo >/dev/null 2>&1; then
        set +e
        sudo dpkg -i "${installer_path}"
        rc="$?"
        set -e
    else
        set +e
        dpkg -i "${installer_path}"
        rc="$?"
        set -e
    fi

    if [[ "${rc}" -ne 0 ]]; then
        echo "ERROR: dpkg install failed (exit ${rc})." >&2
        echo "If this was due to missing dependencies, you must stage those dependency .deb files offline and install them too." >&2
        exit "${rc}"
    fi

    echo "==> Done."
    exit 0
fi

echo "==> Installing VS Code .rpm..."
if command -v sudo >/dev/null 2>&1; then
    sudo rpm -Uvh "${installer_path}"
else
    rpm -Uvh "${installer_path}"
fi

echo "==> Done."
