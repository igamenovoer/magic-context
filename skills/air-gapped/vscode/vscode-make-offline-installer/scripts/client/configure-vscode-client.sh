#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Configure VS Code client settings for air-gapped Remote-SSH use.

Writes/merges keys into the user's settings.json:
  "update.mode": "manual"
  "extensions.autoCheckUpdates": false
  "extensions.autoUpdate": false
  "remote.SSH.localServerDownload": "off" (default)

Usage:
  configure-vscode-client.sh [--channel auto|stable|insider] [--settings-path <PATH>] [--remote-ssh-local-server-download off|always]

Args:
  --channel   Which settings path to target when --settings-path is not provided. Default: auto (stable).
  --settings-path  Override settings.json path directly.
  --remote-ssh-local-server-download  off|always. Default: off.

Notes:
  - Requires python3 (for safe JSON merge).
EOF
}

channel="auto"
settings_path=""
remote_ssh_local_server_download="off"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --channel) channel="${2:-}"; shift 2 ;;
        --settings-path) settings_path="${2:-}"; shift 2 ;;
        --remote-ssh-local-server-download) remote_ssh_local_server_download="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ "${remote_ssh_local_server_download}" != "off" && "${remote_ssh_local_server_download}" != "always" ]]; then
    echo "Invalid --remote-ssh-local-server-download: ${remote_ssh_local_server_download} (expected off|always)" >&2
    exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required but not found on PATH." >&2
    exit 1
fi

os="$(uname -s | tr '[:upper:]' '[:lower:]')"

resolve_settings_path() {
    local want="$1"

    if [[ "${os}" == "darwin" ]]; then
        case "${want}" in
            insider|insiders) printf '%s\n' "${HOME}/Library/Application Support/Code - Insiders/User/settings.json" ;;
            stable|auto) printf '%s\n' "${HOME}/Library/Application Support/Code/User/settings.json" ;;
            *) return 1 ;;
        esac
        return 0
    fi

    # Linux (Ubuntu Desktop recommended)
    case "${want}" in
        insider|insiders) printf '%s\n' "${HOME}/.config/Code - Insiders/User/settings.json" ;;
        stable|auto) printf '%s\n' "${HOME}/.config/Code/User/settings.json" ;;
        *) return 1 ;;
    esac
}

target="${settings_path}"
if [[ -z "${target}" ]]; then
    target="$(resolve_settings_path "${channel}")"
fi

dir="$(dirname "${target}")"
mkdir -p "${dir}"

python3 - "${target}" "${remote_ssh_local_server_download}" <<'PY'
import json
import os
import sys
import tempfile

path = sys.argv[1]
remote_download = sys.argv[2]

def read_settings(p: str) -> dict:
    if not os.path.exists(p):
        return {}
    raw = ""
    with open(p, "r", encoding="utf-8") as f:
        raw = f.read()
    if raw.strip() == "":
        return {}
    obj = json.loads(raw)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise SystemExit(f"settings.json is not a JSON object: {p}")
    return obj

settings = read_settings(path)
settings["update.mode"] = "manual"
settings["extensions.autoCheckUpdates"] = False
settings["extensions.autoUpdate"] = False
settings["remote.SSH.localServerDownload"] = remote_download

tmp_dir = os.path.dirname(path)
fd, tmp_path = tempfile.mkstemp(prefix="settings.", suffix=".json.tmp", dir=tmp_dir)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp_path, path)
finally:
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except Exception:
        pass
PY

echo "==> Updated: ${target}"
echo "    update.mode = manual"
echo "    extensions.autoCheckUpdates = false"
echo "    extensions.autoUpdate = false"
echo "    remote.SSH.localServerDownload = ${remote_ssh_local_server_download}"
