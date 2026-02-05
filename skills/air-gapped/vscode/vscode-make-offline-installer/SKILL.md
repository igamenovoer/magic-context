---
name: vscode-make-offline-installer
description: 'Build a VS Code air-gapped "offline kit" for Remote-SSH: desktop VS Code installers/archives for Windows/macOS/Linux, matching VS Code Server+CLI tarballs for headless Linux (commit-aligned), and pinned extension .vsix bundles for both local (client) and remote (server) sides. Use when you need a reproducible offline deployment, or when you want to mirror the VS Code release+extensions currently installed on a host.'
---

# VS Code Make Offline Installer (Air-gapped Remote-SSH)

Prepare an offline, reproducible bundle that lets:
- Desktop clients (Windows/macOS/Linux) install VS Code and extensions without internet, and
- A headless Linux server run the matching VS Code Server, so Remote-SSH connects without downloading anything,
- With extensions installed on the correct side (local UI vs remote extension host), pinned to known-good versions.

This skill assumes Microsoft VS Code + Remote-SSH (not Coder "code-server").

## What to ask the user

- Target release:
  - Either:
    - Explicit: `CHANNEL` (`stable` or `insider`), and `COMMIT` (required), plus optional `VERSION`, or
    - Mirror local host: `TARGET_RELEASE=local-installed` (agent discovers `CHANNEL`, `VERSION`, `COMMIT`, and local extensions)
- Client targets:
  - OS + arch list (examples: `win32-x64-user`, `darwin-universal`, `linux-deb-x64`)
  - If unspecified, assume the client runs in the same environment as the host running this workflow.
  - If the host is headless, still assume the same OS family + arch as the host; assume the user will add a desktop environment themselves if needed.
- Server targets:
  - Linux arch list (`x64`, `arm64`)
  - You do not need to pre-decide the server user account; the install script supports `--user <username>` and defaults to the executing user.
- Extensions:
  - `extensions_local`: extension IDs + pinned versions
  - `extensions_remote`: extension IDs + pinned versions
  - Download policy (recommended): try Open VSX first, then Marketplace, else skip the extension
  - Required: `extensions_local` must include `ms-vscode-remote.remote-ssh`
- Assumptions for this skill:
  - Clients have SSH access to servers (Remote-SSH is the connectivity path).
  - `tar` is available on the Linux server.
  - Disable VS Code + extension auto-updates via the included post-install script.

## Key invariants (don't skip)

- **VS Code "version" is not enough**: Remote-SSH must match the **client `COMMIT`** (build hash).
- **Remote-SSH extension is mandatory on the client**: air-gapped remote development over SSH requires `ms-vscode-remote.remote-ssh` to be installed locally from a `.vsix`.
- **Remote-SSH offline requires cache placement** on the remote server:
  - `~/.vscode-server/vscode-cli-<COMMIT>.tar.gz.done`
  - `~/.vscode-server/vscode-server.tar.gz`
  - `~/.vscode-server/cli/servers/Stable-<COMMIT>/server/` extracted contents
- **Extensions have two installs**:
  - Local UI side: `code --install-extension <file.vsix>`
  - Remote side: `~/.vscode-server/.../bin/code-server --install-extension <file.vsix>`

## Outputs (recommended kit layout)

Create a single folder you can copy via USB/NAS:

```
vscode-airgap-kit/
  manifest/
    vscode.json                     # version+commit+platforms+sha256
    vscode.local.json               # optional: discovery export from a host (channel/version/commit)
    extensions.local.txt            # id@version list
    extensions.remote.txt           # id@version list
  clients/
    windows/
    macos/
    linux/
  server/
    linux-x64/
    linux-arm64/
    cli/
  extensions/
    local/
    remote/
  scripts/
    wan/                            # run on WAN-connected prep host
    client/                         # run on air-gapped desktop client
    server/                         # copy+run on air-gapped headless Linux server
```

Manifest example: `references/vscode-airgap-manifest.example.json`

## Workflow

### Script groups (3 parts)

This skill ships scripts in 3 groups:

1) WAN prep host (internet-connected):
- `scripts/wan/discover-local-vscode.ps1`
- `scripts/wan/download-vscode-artifacts.ps1`
- `scripts/wan/download-vsix-bundle.ps1`

2) Air-gapped client (desktop):
- Install VS Code: `scripts/client/install-vscode-client.ps1`
- Install local extensions: `scripts/client/install-vscode-client-extensions.ps1`
- Configure VS Code: `scripts/client/configure-vscode-client.ps1`
- Cleanup packages (optional): `scripts/client/cleanup-vscode-client.ps1`

3) Air-gapped headless Linux server:
- Install server cache + extract: `scripts/server/install-vscode-server-cache.sh`
- Configure server state: `scripts/server/configure-vscode-server.sh`
- Install remote extensions: `scripts/server/install-vscode-server-extensions.sh`
- Cleanup old versions/cache (optional): `scripts/server/cleanup-vscode-server.sh`

Installation, configuration, and cleanup are intentionally split because they have different purposes and options.

When preparing `vscode-airgap-kit/`, copy the relevant script folders from this skill into `vscode-airgap-kit/scripts/` so they travel with the offline packages.

Windows note (agent vs user):
- When the AI agent runs a `.ps1`, use one-off execution policy bypass, e.g. `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File <script.ps1> ...` (do not ask the user to change global ExecutionPolicy).
- For user-facing deliverables, keep a same-basename `.bat` launcher next to each `.ps1` so end users can run it without dealing with ExecutionPolicy/permission issues.

### 0) Discovery option: mirror the host's currently installed VS Code release

If the user says to target the release currently used by the host, discover it first (do not guess).

Windows (PowerShell):
- Run `scripts/wan/discover-local-vscode.ps1` to export `VERSION`, `COMMIT`, and the local extension list:
  - Agent: `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts\\wan\\discover-local-vscode.ps1 -OutDir .\\manifest`
  - User: `scripts\\wan\\discover-local-vscode.bat -OutDir .\\manifest`
  - Outputs:
    - `.\\manifest\\vscode.local.json`
    - `.\\manifest\\extensions.local.txt`
  - Ensure `ms-vscode-remote.remote-ssh` is installed locally before exporting if the offline client must use Remote-SSH.

macOS (Terminal):
- Stable:
  - `/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code --version`
- Insiders:
  - `/Applications/Visual Studio Code - Insiders.app/Contents/Resources/app/bin/code-insiders --version`
- Local extensions:
  - `code --list-extensions --show-versions > extensions.local.txt`

Linux (Terminal):
- Version+commit:
  - `code --version` (or `code-insiders --version`)
- Local extensions:
  - `code --list-extensions --show-versions > extensions.local.txt`

Derive `CHANNEL` from which binary is used (`code` = stable, `code-insiders` = insider). Keep `COMMIT` as the compatibility key for server downloads.

If the user does not specify `client targets`, assume the client target matches the host OS family + arch (even if the host is currently headless).

### 1) Pick `VERSION` and `COMMIT` (match everything to `COMMIT`)

On any machine that can install VS Code with internet, install the exact VS Code build you intend to deploy, then record:

```sh
code --version
```

Keep:
- Line 1: `VERSION` (example: `1.106.2`)
- Line 2: `COMMIT` (example: `1e3c50d64110be466c0b4a45222e81d2c9352888`)

If you must "standardize" across multiple client OSes, standardize on `COMMIT` (it is the real compatibility key for the server).

### 2) Download VS Code clients (Windows/macOS/Linux)

Download from the Microsoft update endpoint (commit-pinned):

```
https://update.code.visualstudio.com/commit:<COMMIT>/<PLATFORM>/<CHANNEL>
```

Common `PLATFORM` values:
- Windows: `win32-x64-user`, `win32-arm64-user`
- macOS: `darwin-universal` (recommended), `darwin-arm64`
- Linux:
  - Archives: `linux-x64`, `linux-arm64`
  - Debian/Ubuntu: `linux-deb-x64`, `linux-deb-arm64`
  - RHEL/Fedora: `linux-rpm-x64`, `linux-rpm-arm64`

Save the downloaded files under `clients/<os>/` and record SHA256 hashes in `manifest/vscode.json`.

Optional helper (WAN prep) to download commit-pinned artifacts and write `manifest/vscode.json`:

```powershell
pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts\\wan\\download-vscode-artifacts.ps1 `
  -Commit "<COMMIT>" -Channel stable -OutDir .\\vscode-airgap-kit `
  -ClientPlatforms @("win32-x64-user") `
  -ServerArch @("x64","arm64")
```

### 3) Download VS Code Server + VS Code CLI (Linux)

Download the server tarballs (commit-pinned):

```
https://update.code.visualstudio.com/commit:<COMMIT>/server-linux-x64/<CHANNEL>
https://update.code.visualstudio.com/commit:<COMMIT>/server-linux-arm64/<CHANNEL>
```

Download the CLI tarballs (commit-pinned):

```
https://update.code.visualstudio.com/commit:<COMMIT>/cli-alpine-x64/<CHANNEL>
https://update.code.visualstudio.com/commit:<COMMIT>/cli-alpine-arm64/<CHANNEL>
```

Put them under:
- `server/linux-x64/` and `server/linux-arm64/`
- `server/cli/`

### 4) Freeze and download extensions as `.vsix` (local + remote)

Pin versions first, then download `.vsix` files.

Required client extension for SSH remote development:
- `ms-vscode-remote.remote-ssh` (do not skip)

Recommended "pinning" flow (do this on an online staging environment):
1. Install VS Code client (`COMMIT`) on a desktop.
2. Connect to a similar Linux server once (online) and install/configure extensions until it works.
3. Export the exact extension versions:
   - Local: `code --list-extensions --show-versions > extensions.local.txt`
   - Remote (over SSH): run the server's `code-server` binary:
     - `~/.vscode-server/cli/servers/Stable-<COMMIT>/server/bin/code-server --list-extensions --show-versions`
     - Save output as `extensions.remote.txt`

Downloading `.vsix` (preferred order):
- 1) Open VSX (https://open-vsx.org/) (preferred when available):
  - `ovsx get publisher.extension@<version> -o <file>.vsix`
- 2) Marketplace fallback:
  - Use the "vspackage" download endpoint described in `quick-tools/install-vscode-offline/howto-download-vscode-extensions.md`
- 3) If neither source has the extension, skip it and record it in your manifest/logs.

Optional helper (Windows-friendly) to download pinned VSIX in bulk:

```powershell
pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts\\wan\\download-vsix-bundle.ps1 -InputList .\\manifest\\extensions.local.txt -OutDir .\\extensions\\local
pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts\\wan\\download-vsix-bundle.ps1 -InputList .\\manifest\\extensions.remote.txt -OutDir .\\extensions\\remote -RequiredIds @()
```

Place downloaded `.vsix` files into:
- `extensions/local/`
- `extensions/remote/`

Validate every `.vsix` is a ZIP (fast corruption check):
- `unzip -t file.vsix` (Linux/macOS) or
- `python -c "import zipfile; zipfile.ZipFile('file.vsix').testzip()"` (any)

### 5) Install VS Code on air-gapped clients (desktop)

On each client OS:
1. Install VS Code from the offline installer/archive.
2. Disable update checks + extension auto-updates (required for air-gapped stability):
   - Run the post-install config script:
     - `pwsh -NoProfile -File scripts\\client\\configure-vscode-client.ps1 -Channel auto`
3. Install local extensions:
   - `code --install-extension /path/to/extensions/local/<x>.vsix --force`

Recommended split scripts on the air-gapped client:

```powershell
# 1) Install VS Code (best-effort automation; Windows supported)
pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts\\client\\install-vscode-client.ps1 -InstallerPath .\\clients\\windows\\VSCodeUserSetup-x64-*.exe

# 2) Configure VS Code settings (disable updates, set Remote-SSH behavior)
pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts\\client\\configure-vscode-client.ps1 -Channel auto

# 3) Install local extensions from VSIX (includes required Remote-SSH)
pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts\\client\\install-vscode-client-extensions.ps1 -ExtensionsDir .\\extensions\\local -Channel auto
```

Remote-SSH settings guidance:
- Fully air-gapped (server already has cache files): set `"remote.SSH.localServerDownload": "off"`
- Client online but server offline (rare): set `"remote.SSH.localServerDownload": "always"`

### 6) Install VS Code Server on headless Linux (air-gapped)

Prefer the cache-based method so Remote-SSH never attempts downloads.

Copy the `scripts/server/` folder to the headless server and run the scripts there.

Recommended split scripts on the server:

```bash
# 1) Install (place cache files + extract server)
sudo bash scripts/server/install-vscode-server-cache.sh \
  --commit "<COMMIT>" --user "<USERNAME>" \
  --server-tar "/path/to/vscode-server-linux-x64-<COMMIT>.tar.gz" \
  --cli-tar "/path/to/vscode-cli-alpine-x64-<COMMIT>.tar.gz"

# 2) Configure (create settings.json, touch .ready)
sudo bash scripts/server/configure-vscode-server.sh --commit "<COMMIT>" --user "<USERNAME>"
# Optional: provide a prebuilt settings.json
# sudo bash scripts/server/configure-vscode-server.sh --commit "<COMMIT>" --user "<USERNAME>" --settings-file "/path/to/settings.json"

# 3) Install remote-side extensions (safe to re-run for updates)
sudo bash scripts/server/install-vscode-server-extensions.sh \
  --commit "<COMMIT>" --user "<USERNAME>" \
  --extensions-dir "/path/to/extensions/remote"
```

Notes:
- If `--user` is omitted, it installs for the executing user.
- If `--user` is different from the executing user, run the script as root/admin (or via `sudo`) so it can write into that user's home and fix ownership.

Option B: manual placement (must match filenames exactly):
- Copy CLI tarball to `~/.vscode-server/vscode-cli-<COMMIT>.tar.gz`
- Duplicate it to `~/.vscode-server/vscode-cli-<COMMIT>.tar.gz.done`
- Copy server tarball to `~/.vscode-server/vscode-server.tar.gz`
- Extract server to `~/.vscode-server/cli/servers/Stable-<COMMIT>/server/` (strip top-level folder)

### 7) Verify end-to-end (no downloads)

On the server:
- `test -f ~/.vscode-server/vscode-cli-<COMMIT>.tar.gz.done`
- `test -f ~/.vscode-server/vscode-server.tar.gz`
- `test -x ~/.vscode-server/cli/servers/Stable-<COMMIT>/server/bin/code-server`
- `~/.vscode-server/cli/servers/Stable-<COMMIT>/server/bin/code-server --list-extensions --extensions-dir ~/.vscode-server/extensions`

From a desktop client:
- Connect via "Remote-SSH: Connect to Host..."
- Watch the Remote-SSH output: it should report it found an existing installation and should not attempt downloads.

## Common failure modes

- Client/server mismatch: `COMMIT` differs -> client tries to download a different server build.
- Wrong filenames: `vscode-server-linux-x64-<COMMIT>.tar.gz` is not detected unless it is copied/renamed to `~/.vscode-server/vscode-server.tar.gz`.
- Wrong architecture: x64 server tarball on arm64 host (or vice versa).
- Extensions not visible remotely: installed only on the client; install them via server `code-server` too (or install "from VSIX" while connected to the remote window).

## Updating (client and server)

When you need to update VS Code (new `COMMIT`) and/or extension versions:

1) On the WAN prep host:
   - Re-run `scripts/wan/download-vscode-artifacts.ps1` for the new commit and overwrite the kit (or produce a new kit folder).
   - Re-run `scripts/wan/download-vsix-bundle.ps1` for updated pinned extension lists.

2) On the air-gapped client:
   - Re-run `scripts/client/install-vscode-client.ps1` with the new installer/archive (updates in-place on most platforms).
   - Re-run `scripts/client/configure-vscode-client.ps1` (idempotent).
   - Re-run `scripts/client/install-vscode-client-extensions.ps1` with updated VSIX (forces install).

3) On the headless server:
   - Run `scripts/server/install-vscode-server-cache.sh --commit <NEW_COMMIT> ...` (keeps old extracted servers unless you later clean up).
   - Run `scripts/server/configure-vscode-server.sh --commit <NEW_COMMIT> ...`
   - Run `scripts/server/install-vscode-server-extensions.sh --commit <NEW_COMMIT> ...` (forces extension installs).
   - Optionally remove old commits/cache with `scripts/server/cleanup-vscode-server.sh` after verification.
