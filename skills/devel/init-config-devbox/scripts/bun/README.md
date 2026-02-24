# Bun (JavaScript runtime)

> **Note:** Before running the `.ps1` script, please run the `<workspace>/enable-ps1-permission.bat` script once to allow PowerShell script execution.

This component installs [Bun](https://bun.sh), an all‑in‑one JavaScript/TypeScript runtime, bundler, and test runner.

## Recommended installation on Windows

Bun’s own installer ensures both `bun` and `bunx` are set up correctly and that `~\.bun\bin` is added to `PATH`. This avoids current issues with the winget package where `bunx` is not available on the command line and `~\.bun\bin` is not added to `PATH` (see the `Oven-sh.Bun` package issue in the winget repository).

- Recommended command:
  ```powershell
  powershell -ExecutionPolicy Bypass -Command "irm https://bun.sh/install.ps1 | iex"
  ```
- This installs Bun into `%USERPROFILE%\.bun\bin` and updates your shell profile so `bun` and `bunx` are directly runnable in PowerShell.

## Installation via winget

- You can still install Bun via winget:
  ```powershell
  winget install -e --id Oven-sh.Bun
  ```
- However, due to current packaging limitations:
  - `bunx` may not be available as a separate command.
  - `%USERPROFILE%\.bun\bin` may not be added to `PATH` automatically.
- Our installer will typically either:
  - Prefer the official `install.ps1` flow, or
  - Run the official installer as a post-step after `winget` to ensure `bunx` and `PATH` are configured correctly.

## China-friendly installation notes

- Behind China networks, combine the official installer with a proxy:
  ```powershell
  $env:HTTP_PROXY = "<proxy-url>"
  $env:HTTPS_PROXY = "<proxy-url>"
  powershell -ExecutionPolicy Bypass -Command "irm https://bun.sh/install.ps1 | iex"
  ```
- You may also use a corporate GitHub/file accelerator if available to improve access to `bun.sh` and GitHub releases.

## Official installation

- Bun’s official installation instructions are at:
  - https://bun.sh
  - Windows install docs: https://bun.sh/docs/installation
- Our `install-comp` script for Bun will:
  - Ensure the official Bun installer has been run so `bun` and `bunx` work as expected in PowerShell.
  - Use `winget` as appropriate for package management, but correct any PATH / `bunx` issues via the official installer when necessary.
  - Respect `--proxy / -Proxy` for downloads.
  - Accept `--from-official` to force use of official URLs even if a local accelerator/proxy is configured.

## Configuration

- To configure the global package registry (mirror), run the configuration script:
  ```powershell
  .\config-comp.ps1 -Mirror cn
  # or
  .\config-comp.ps1 -Mirror official
  ```
- This updates `~/.bunfig.toml` to use `registry.npmmirror.com` (for `cn`) or `registry.npmjs.org` (for `official`).

## Linux/macOS (POSIX) scripts

- Install:
  ```bash
  cd components/bun
  sh ./install-comp.sh --dry-run
  sh ./install-comp.sh
  # optional:
  # sh ./install-comp.sh --proxy "http://127.0.0.1:7890"
  ```
- Configure registry mirror:
  ```bash
  sh ./config-comp.sh --mirror cn
  # or
  sh ./config-comp.sh --mirror official
  ```
- Notes:
  - The installer uses Bun’s official install script (`https://bun.sh/install`).
  - `config-comp.sh` updates `~/.bunfig.toml`.
