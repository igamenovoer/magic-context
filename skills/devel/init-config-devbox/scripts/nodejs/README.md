# Node.js

> **Note:** Before running the `.ps1` script, please run the `<workspace>/enable-ps1-permission.bat` script once to allow PowerShell script execution.

This component installs Node.js (preferring the LTS build).

## Preferred installation (winget)

- LTS channel:
  ```powershell
  winget install -e --id OpenJS.NodeJS.LTS
  ```
- Fallback (current):
  ```powershell
  winget install -e --id OpenJS.NodeJS
  ```

## China-friendly installation (mirrors)

If you need to download installers or binaries manually, use a known mirror:

- **Tsinghua nodejs-release mirror**  
  - https://mirrors.tuna.tsinghua.edu.cn/nodejs-release/
  - Can be used with tools like `nvm`, `nvs`, or `volta` by setting `NVM_NODEJS_ORG_MIRROR` / `NODE_MIRROR`.
- **npm registry mirror for packages** (after Node is installed):
  ```powershell
  npm config set registry https://registry.npmmirror.com
  npm config get registry
  ```

## Configuration

- To configure the global package registry (mirror), run the configuration script:
  ```powershell
  .\config-comp.ps1 -Mirror cn
  # or
  .\config-comp.ps1 -Mirror official
  ```
- This runs `npm config set registry ...` to use `registry.npmmirror.com` (for `cn`) or `registry.npmjs.org` (for `official`).

Our `install-comp` script will:

- Prefer `winget install` as above.
- If direct installers are needed, try `nodejs-release` mirrors first, then fall back to `https://nodejs.org/en/download`.
- Respect `--proxy / -Proxy` and `--from-official` (skipping mirrors and using the official Node.js download site).

## Official installation

- Official downloads:
  - https://nodejs.org/en/download
- Package manager docs:
  - https://nodejs.org/en/download/package-manager

## Linux/macOS (POSIX) scripts

- Install:
  ```bash
  cd components/nodejs
  sh ./install-comp.sh --dry-run
  sh ./install-comp.sh --accept-defaults
  ```
- Configure npm registry mirror:
  ```bash
  sh ./config-comp.sh --mirror cn
  # or
  sh ./config-comp.sh --mirror official
  ```
