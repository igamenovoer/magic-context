# VS Code Air-gapped Offline Kit (Remote-SSH)

This folder is an offline kit for Microsoft VS Code Remote-SSH.

## Release pin

- Channel: `{{CHANNEL}}`
- Version: `{{VERSION}}` (if known)
- Commit: `{{COMMIT}}` (must match line 2 of `code --version` on the client)

## What is in this kit

### Client installers/archives

Client packages are grouped under `./clients/<os>-<arch>/` (for example `win32-x64`, `darwin-universal`, `linux-x64`).

{{CLIENT_INVENTORY}}

### Server artifacts (headless Linux)

VS Code Server tarballs:

{{SERVER_INVENTORY}}

VS Code CLI tarballs:

{{CLI_INVENTORY}}

### Extensions (VSIX)

- Local (client/UI side): `./extensions/local-<TARGET>/` (examples: `./extensions/local-win32-x64/`, `./extensions/local-linux-x64/`)
- Remote (server extension host): `./extensions/remote-linux-<arch>/` (examples: `./extensions/remote-linux-x64/`, `./extensions/remote-linux-arm64/`)

## Install on an air-gapped desktop client

### 0) Copy the kit

Copy this entire folder to the client machine (USB/NAS).

### 1) Install VS Code (client)

{{CLIENT_INSTALL_SECTIONS}}

### 2) Configure VS Code for air-gapped stability

Set these in user settings (disables auto-updates and prevents Remote-SSH downloads):

- `"update.mode": "manual"`
- `"extensions.autoCheckUpdates": false`
- `"extensions.autoUpdate": false`
- `"remote.SSH.localServerDownload": "off"`

{{CONFIGURE_HELPERS}}

### 3) Install local (client-side) extensions from VSIX

Remote-SSH must be installed locally (`ms-vscode-remote.remote-ssh`).

{{LOCAL_EXT_HELPERS}}

## Install on an air-gapped headless Linux server (Remote-SSH target)

These steps must be run on the target Linux server (or via SSH to it). They pre-place the cache files so Remote-SSH does not try to download anything.

### 1) Install the server cache + extract the server

{{SERVER_INSTALL_NOTE}}

### 2) Configure server state (settings + readiness marker)

```bash
sudo bash scripts/server/configure-vscode-server.sh --user "<USERNAME>"
```

### 3) Install remote-side extensions from VSIX (optional but common)

```bash
sudo bash scripts/server/install-vscode-server-extensions.sh \
  --user "<USERNAME>"
```

### 4) Verify on the server

```bash
test -f ~/.vscode-server/vscode-cli-{{COMMIT}}.tar.gz.done
test -f ~/.vscode-server/vscode-server.tar.gz
test -x ~/.vscode-server/cli/servers/Stable-{{COMMIT}}/server/bin/code-server
~/.vscode-server/cli/servers/Stable-{{COMMIT}}/server/bin/code-server --list-extensions --show-versions --extensions-dir ~/.vscode-server/extensions
```

## Connect (desktop client)

In VS Code: run “Remote-SSH: Connect to Host...”. The Remote-SSH log should indicate it found an existing server installation and should not attempt downloads.

{{SERVER_ARTIFACTS_WARNING}}

