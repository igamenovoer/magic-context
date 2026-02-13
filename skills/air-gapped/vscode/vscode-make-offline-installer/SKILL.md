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

## Triggering (manual)

This skill must be triggered manually by skill name (for example: `vscode-make-offline-installer`).
Do not auto-trigger this workflow based on “similar” requests; ask the user to explicitly invoke it.

## What to ask the user

- Target release:
  - Either:
    - Explicit: `CHANNEL` (`stable` or `insider`), and `COMMIT` (required), plus optional `VERSION`, or
    - Mirror local host: `TARGET_RELEASE=local-installed` (agent discovers `CHANNEL`, `VERSION`, `COMMIT`, and local extensions)
  - Note: server-side cache/install scripts currently assume the Stable server directory layout (`~/.vscode-server/.../Stable-<COMMIT>`). If you target `CHANNEL=insider`, confirm the expected server paths for Insiders and adjust scripts/paths as needed.
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
- Optional: Testing environment (to validate the kit end-to-end):
  - Option A: an actual remote Linux server reachable over SSH
    - Provide as `username@ip` / `username@hostname`, or an SSH config host alias (e.g. `myserver` from `~/.ssh/config`).
    - Used to verify the server-side cache placement and remote-side extension installs (the VS Code Server `bin/code-server`) without needing any internet access.
  - Option B: a Docker image or an existing container to use as a disposable test host
    - If provided, treat it as a fully-controlled testing sandbox (destructive actions are OK inside the container; just do not delete the image).
    - Used to run/exercise the `scripts/server/*.sh` flow and sanity-check the extracted VS Code Server `bin/code-server` + remote extension installs.
- Safety (default behavior):
  - **Never** install/modify client-side VS Code or client-side extensions on the current host as part of testing; generating the kit must not change the user's dev environment.
  - **Never** attempt to verify server-side `code-server` or remote-side extension installs unless the user explicitly provides a testing environment (SSH host or Docker image/container) for this task.
- Assumptions for this skill:
  - Clients have SSH access to servers (Remote-SSH is the connectivity path).
  - `tar` is available on the Linux server.
  - Disable VS Code + extension auto-updates via the included post-install script.

## Key invariants (don't skip)

- **VS Code "version" is not enough**: Remote-SSH must match the **client `COMMIT`** (build hash).
- **Remote-SSH extension is mandatory on the client**: air-gapped remote development over SSH requires `ms-vscode-remote.remote-ssh` to be installed locally from a `.vsix`.
- **Remote-SSH offline requires cache placement** on the remote server:
  > Remote-SSH boots a VS Code Server on the remote host.
  > If the server for this exact `COMMIT` is missing, it normally downloads and extracts it during the first connection.
  > In an air-gapped environment that download fails, so you must pre-place the tarballs/marker files and extracted server under `~/.vscode-server/` for the target user (the provided `scripts/server/install-vscode-server-cache.sh` does this).
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
  README.md                         # installation instructions for the selected client/server platforms (generated)
  manifest/                         # metadata + inventories (what this kit contains)
    vscode.json                     # commit/channel + downloaded artifacts + sha256
    vscode.local.json               # optional: discovery export from a host (channel/version/commit)
    extensions.local.txt            # local-side extension id@version pins
    extensions.remote.txt           # remote-side extension id@version pins
  clients/                          # VS Code desktop installers/archives per OS
    windows/                        # e.g., VSCodeUserSetup-*.exe
    macos/                          # e.g., VSCode-*.zip
    linux/                          # e.g., .deb/.rpm/.tar.gz
  server/                           # VS Code Server + CLI artifacts for air-gapped Remote-SSH
    linux-x64/                      # vscode-server-linux-x64-<COMMIT>.tar.gz
    linux-arm64/                    # vscode-server-linux-arm64-<COMMIT>.tar.gz
    cli/                            # vscode-cli-alpine-<arch>-<COMMIT>.tar.gz
  extensions/                       # offline extension bundles (.vsix)
    local/                          # install on the client (UI side): `code --install-extension`
    remote/                         # install on the server (extension host): `code-server --install-extension`
  scripts/                          # helper install/config scripts to run in each environment
    wan/                            # run on WAN-connected prep host
    client/                         # run on air-gapped desktop client
    server/                         # copy+run on air-gapped headless Linux server
```

Manifest example: `references/vscode-airgap-manifest.example.json`

## Workflow

### Script groups (3 parts)

This skill ships scripts in 3 groups:

1) WAN prep host (internet-connected):
- `scripts/wan/discover-local-vscode.ps1` — detect installed VS Code `channel`/`version`/`commit` and export the local extension inventory.
- `scripts/wan/download-vscode-artifacts.ps1` — download commit-pinned VS Code client installers/archives and the matching server+CLI tarballs into the kit, plus SHA256s.
- `scripts/wan/download-vsix-bundle.ps1` — download pinned extension `.vsix` files for offline install (Open VSX first, Marketplace fallback) and write a report.
- `scripts/wan/export-kit-readme-context.ps1` — export `manifest/readme.context.json` (inventory + snippets) to help the agent fill the kit `README.md` template manually; also stages `scripts/client/` + `scripts/server/` into the kit by default (disable via `-NoStageScripts`). This script does **not** generate the final `README.md`.
- Download robustness: if `aria2c` is available on the host, the WAN download scripts prefer it for resumable downloads; otherwise they fall back to PowerShell HTTP downloads.

2) Air-gapped client (desktop):
- Install VS Code:
  - Windows: `scripts/client/install-vscode-client.ps1` — launch the offline installer (interactive by default; `-Silent` best-effort).
  - Linux (Ubuntu Desktop): `scripts/client/install-vscode-client.sh` — install the offline `.deb` via `dpkg` (recommended client artifact: `linux-deb-<arch>`).
- Install local extensions (client/UI side):
  - PowerShell: `scripts/client/install-vscode-client-extensions.ps1`
  - Bash: `scripts/client/install-vscode-client-extensions.sh`
- Configure VS Code (disable auto-updates + set `remote.SSH.localServerDownload`):
  - PowerShell: `scripts/client/configure-vscode-client.ps1`
  - Bash: `scripts/client/configure-vscode-client.sh`
- Cleanup packages (optional):
  - PowerShell: `scripts/client/cleanup-vscode-client.ps1`
  - Bash: `scripts/client/cleanup-vscode-client.sh`

3) Air-gapped headless Linux server:
- Install server cache + extract: `scripts/server/install-vscode-server-cache.sh` — pre-place the server/CLI cache files and extract the server for the target `COMMIT`.
- Configure server state: `scripts/server/configure-vscode-server.sh` — create `data/Machine/settings.json` (optional override) and touch the `.ready` marker.
- Install remote extensions: `scripts/server/install-vscode-server-extensions.sh` — install all `.vsix` in a folder into the remote extension host via `code-server`.
- Cleanup old versions/cache (optional): `scripts/server/cleanup-vscode-server.sh` — remove cache tarballs and/or old extracted servers once you’ve verified Remote-SSH works.

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

### 4) Freeze and download extensions as `.vsix`

Pin versions first, then download the matching `.vsix` files.

#### 4a) Client-side extensions (local / UI side)

Required client extension for SSH remote development:
- `ms-vscode-remote.remote-ssh` (do not skip)

Export the exact client-side extension versions:
- `code --list-extensions --show-versions > extensions.local.txt`

Place downloaded `.vsix` files into:
- `extensions/local/`

#### 4b) Server-side extensions (remote / extension host side)

If the user provided a testing environment for this task, prefer using it to produce `extensions.remote.txt` (this captures what actually ends up running remotely).

Option A (recommended): use the provided test environment to generate `extensions.remote.txt`
- SSH host:
  - `ssh <host> "~/.vscode-server/cli/servers/Stable-<COMMIT>/server/bin/code-server --list-extensions --show-versions" > extensions.remote.txt`
- Docker image/container:
  - Use it as a disposable sandbox to run the server install scripts, then run:
    - `~/.vscode-server/cli/servers/Stable-<COMMIT>/server/bin/code-server --list-extensions --show-versions > extensions.remote.txt`

Option B (fallback): no test environment provided
- Start from the extension IDs you intend to install remotely and pin versions to be “as close as possible” to the client-side pins:
  - First choice: use the **exact same version** as the local pin for that extension ID.
  - If that exact `id@version` cannot be downloaded as a `.vsix` from Open VSX or Marketplace, pick the **nearest older version** than the local pin:
    - Define “nearest older” as the highest available version that is `<` the local version (semantic-version compare).
    - Practical workflow: try downloading the remote list with `scripts/wan/download-vsix-bundle.ps1`, and when an item fails, decrement to the next older published version and retry until it succeeds.

Place downloaded `.vsix` files into:
- `extensions/remote/`

#### Download `.vsix` files (applies to both local + remote lists)

Recommended pinning flow (do this on an online staging environment):
1. Install VS Code client (`COMMIT`) on a desktop staging machine.
2. Connect to a similar Linux server once (online) and install/configure extensions until it works.
3. Export `extensions.local.txt` (client) and `extensions.remote.txt` (server).

Downloading `.vsix` (preferred order):
- 1) Open VSX (https://open-vsx.org/) (preferred when available):
  - `ovsx get publisher.extension@<version> -o <file>.vsix`
- 2) Marketplace fallback:
  - Use the Marketplace "vspackage" endpoint (as implemented by `scripts/wan/download-vsix-bundle.ps1`):
    - `https://marketplace.visualstudio.com/_apis/public/gallery/publishers/<publisher>/vsextensions/<name>/<version>/vspackage`
- 3) If neither source has the extension, skip it and record it in your manifest/logs.
  - If you expected it to exist but the links/endpoints look broken (404/redirect loops), try a quick web search for the extension ID + version + “vsix” and update the download URL/source notes you store in the manifest.

Optional helper (Windows-friendly) to download pinned VSIX in bulk:

```powershell
pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts\\wan\\download-vsix-bundle.ps1 -InputList .\\manifest\\extensions.local.txt -OutDir .\\extensions\\local
pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts\\wan\\download-vsix-bundle.ps1 -InputList .\\manifest\\extensions.remote.txt -OutDir .\\extensions\\remote -RequiredIds @()
```

Validate every `.vsix` is a ZIP (fast corruption check):
- `unzip -t file.vsix` (Linux/macOS) or
- `python -c "import zipfile; zipfile.ZipFile('file.vsix').testzip()"` (any)

### 5) Fill kit `README.md` template (recommended)

Keep detailed install/runbook instructions in the kit output (not in this `SKILL.md`). This skill ships a Markdown template that the **agent** fills in as part of the run (no auto-generated final README).

Template:
- `references/kit-readme.template.md` (edit this template to customize wording)

Recommended workflow:
1. Copy the template into the kit root as `README.md`.
2. Fill all `{{...}}` placeholders using:
   - Selected `CHANNEL`/`COMMIT` (and optional `VERSION`),
   - The actual kit contents (filenames under `clients/`, `server/`, `extensions/`, and the staged `scripts/` folder if included),
   - Any existing kit metadata under `manifest/` (for example `vscode.json`, `vscode.local.json`, `extensions.*.txt`, and VSIX download reports).
3. (Optional) Export an agent-facing fill-context JSON (inventory + snippets) into `<KitDir>/manifest/readme.context.json`:

```powershell
pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts\\wan\\export-kit-readme-context.ps1 -KitDir .\\vscode-airgap-kit
```

Notes:
- `export-kit-readme-context.ps1` does **not** generate the final `README.md`; it only exports context to help fill the template and (by default) stages `scripts/client/` and `scripts/server/` into `<KitDir>/scripts/`. Use `-NoStageScripts` to skip staging.

### 6) Optional: verify end-to-end (requires a provided test environment)

Only do verification if the user explicitly provided a test environment (SSH host or Docker image/container). Stage 1 is fully headless; Stage 2 (optional) is a user-driven VS Code GUI check performed while offline.

Verification stages:

#### Stage 1: Server-side validation (requires the provided test server env)

Goal: prove the kit’s VS Code Server artifacts can be installed/extracted and the server-side `code-server` binary runs and can manage extensions.

1) Install/extract on the provided server env (SSH host or disposable container) using the kit’s server scripts:
- Run `scripts/server/install-vscode-server-cache.sh` (cache placement + extraction)
- Run `scripts/server/configure-vscode-server.sh` (settings + readiness marker)
- Optionally run `scripts/server/install-vscode-server-extensions.sh` with `./extensions/remote/` (VSIX installs)

2) Headless sanity checks on the server env (example; substitute `<COMMIT>`):
```bash
COMMIT="<COMMIT>"
SERVER_BIN="$HOME/.vscode-server/cli/servers/Stable-$COMMIT/server/bin/code-server"
EXT_DIR="$HOME/.vscode-server/extensions"

test -x "$SERVER_BIN"
"$SERVER_BIN" -h
"$SERVER_BIN" --list-extensions --show-versions --extensions-dir "$EXT_DIR"
"$SERVER_BIN" --install-extension "/path/to/kit/extensions/remote/<some>.vsix" --force --extensions-dir "$EXT_DIR"
"$SERVER_BIN" --list-extensions --show-versions --extensions-dir "$EXT_DIR"
```

3) Optional: try starting a local listener on the server env
- Use `"$SERVER_BIN" -h` to discover the correct flags, bind to `127.0.0.1`, then probe with `curl -I` (or use SSH port-forwarding). Keep it on localhost and do not expose it publicly.

#### Stage 2: Desktop-side Remote-SSH validation (user GUI; optional)

Goal: validate the full Remote-SSH experience with the user’s VS Code desktop app, while the internet is disconnected (so any attempted downloads fail fast and are visible).

Agent preparation (do before asking the user to test):
- Ensure Stage 1 is complete: the server cache files are in place, the server is extracted, and `code-server` is runnable for the target `COMMIT`.
- Ensure server SSH access works (for a Docker image/container test env, ensure SSH is reachable from the user’s desktop):
  - SSH host: confirm `ssh <host> 'echo ok'` succeeds (from the user’s desktop if possible).
  - Container: start/maintain an SSH-accessible container instance (do not delete the image), and provide the user with `username@host:port` (and any key/password) needed to connect.
- Ensure remote-side extensions are installed as intended (optional but common) via `scripts/server/install-vscode-server-extensions.sh` using the kit’s `./extensions/remote/`.
- Do not install/modify client-side VS Code or client-side extensions on the agent host “for testing”; the user performs the GUI validation on their desktop environment (ideally a dedicated test profile/machine).

User instructions (GUI-driven, but offline):
1) On the desktop client, ensure VS Code has the Remote-SSH extension installed from the kit (`ms-vscode-remote.remote-ssh` from `./extensions/local/`), and apply the kit’s client config (disable auto-updates + set `remote.SSH.localServerDownload` to `off`).
2) Disconnect the desktop client from the internet (airplane mode or unplug). Keep LAN access to the SSH server if applicable.
3) In VS Code, open **View → Output**, select **Remote - SSH** in the dropdown, then run **Remote-SSH: Connect to Host...** and connect to the test host.
4) Verify (from the Remote - SSH output/log):
   - It finds an existing server install for the exact `COMMIT` and does not attempt to download server bits.
   - Remote window opens successfully and remote extensions (if provided) are present.
5) If it fails, re-enable internet only long enough to capture complete logs (Output: Remote - SSH), then go offline again and iterate by adjusting the kit/server cache based on those logs.

## Updating (client and server)

When you need to update VS Code (new `COMMIT`) and/or extension versions:

1) On the WAN prep host:
   - Re-run `scripts/wan/download-vscode-artifacts.ps1` for the new commit and overwrite the kit (or produce a new kit folder).
   - Re-run `scripts/wan/download-vsix-bundle.ps1` for updated pinned extension lists.

2) On the air-gapped client:
   - Re-run `scripts/client/install-vscode-client.ps1` (Windows) or `scripts/client/install-vscode-client.sh` (Linux). If the kit layout is intact, you can omit the installer path and the script will auto-locate it relative to the script location.
   - Re-run `scripts/client/configure-vscode-client.ps1` or `scripts/client/configure-vscode-client.sh` (idempotent).
   - Re-run `scripts/client/install-vscode-client-extensions.ps1` or `scripts/client/install-vscode-client-extensions.sh`. If the kit layout is intact, you can omit the extensions dir and it will default to `./extensions/local` relative to the script location (forces install).

3) On the headless server:
   - Run `scripts/server/install-vscode-server-cache.sh` (keeps old extracted servers unless you later clean up). If the kit layout is intact, you can omit `--commit` and tarball paths and it will auto-detect from `./manifest/` and `./server/` relative to the script location.
   - Run `scripts/server/configure-vscode-server.sh` (can auto-detect `--commit` from `./manifest/`).
   - Run `scripts/server/install-vscode-server-extensions.sh` (can auto-detect `--commit`, and defaults `--extensions-dir` to `./extensions/remote` if present, else `./extensions/local`).
   - Optionally remove old commits/cache with `scripts/server/cleanup-vscode-server.sh` after verification.

## Troubleshooting

This section is based on real-world friction observed while mirroring a Windows host into an air-gapped kit.

### Downloads are slow, restart from zero, or fail due to partial files / file locks

- Prefer `aria2c` (if installed): the WAN download scripts will automatically use it for resumable downloads; verify with `Get-Command aria2c`.
- If a previous run left temp artifacts (`*.tmp`, `*.part`, `*.aria2`) and retries behave oddly, stop any stray `pwsh` processes that might still be downloading, then re-run with `-Force` (where supported).

### `download-vscode-artifacts.ps1` errors: `A positional parameter cannot be found that accepts argument 'arm64'`

- This can happen when passing array parameters through wrappers/quoting.
- Workaround: invoke via `pwsh -Command "& 'scripts\\wan\\download-vscode-artifacts.ps1' ... -ServerArch @('x64','arm64')"` (so PowerShell parses the array expression reliably), or run it directly inside an interactive PowerShell session.

### VSIX download “succeeds” but the file is not a real VSIX (ZIP validation fails)

Symptoms:
- You see `.vsix` files but install fails or the script reports `bad_file` / `required_bad_file`.

Actions:
- Re-run the VSIX download; Open VSX / Marketplace endpoints can return HTML/redirect/error payloads for some IDs/versions.
- Use local fallbacks when mirroring a host:
  - VS Code’s cache directory (e.g. `%APPDATA%\\Code\\CachedExtensionVSIXs`), if present.
  - Export from the installed extension folder under `%USERPROFILE%\\.vscode\\extensions\\...`.

### Remote bundle contains Windows-only VSIX (Linux Remote-SSH target cannot install them)

Symptoms:
- Remote install fails on Linux for extensions that are platform-specific on Windows (examples seen: `ms-vscode.cpptools`, `ms-dotnettools.csharp`, `charliermarsh.ruff`).

Actions:
- When building `extensions/remote/`, do not rely only on `extension/package.json` fields; some packages only encode platform targeting in `extension.vsixmanifest` via `TargetPlatform="win32-x64"`.
- Prefer a deterministic “derive remote bundle” step that:
  - Parses each VSIX `extension.vsixmanifest` `TargetPlatform`, and
  - Excludes `win32-*` / `darwin-*` targets from `extensions/remote/`.

### Docker/container testbed seems “stuck” installing remote extensions

Symptoms:
- Installing a large number of VSIX into the remote extension host via `code-server --install-extension ...` takes a very long time.

Actions:
- For discovery and packaging, prefer deriving `extensions/remote/` from VSIX metadata (manifest-based), and reserve full remote installs for small validation sets or explicitly requested end-to-end verification.
- If you do validate with Docker, keep the container commands simple and inspect progress with `docker logs` / `docker top` rather than complex shell pipelines.

### Cross-shell quoting pitfalls (`/dev/null`, `grep`, `true`) when running `docker exec ... bash -lc "..."`

- Keep Linux redirection/pipelines entirely inside the quoted `bash -lc "..."` string.
- Avoid composing long `bash -lc` command strings with nested quoting from PowerShell; prefer multiple smaller `docker exec ...` calls or write outputs to files under a bind-mounted kit directory.
