# pixi (prefix.dev)

> **Note:** Before running the `.ps1` script, please run the `<workspace>/enable-ps1-permission.bat` script once to allow PowerShell script execution.

This component installs [pixi](https://pixi.sh), a cross-platform package and workflow manager from prefix.dev.

## Preferred installation (winget)

- Winget package:
  ```powershell
  winget install -e --id prefix-dev.pixi
  ```

## China-friendly installation

- Pixi is distributed primarily from prefix.dev and GitHub; there is no widely recognized dedicated China mirror at this time.
- For slower networks:
  - Use `--proxy / -Proxy` so the installer can set `HTTP_PROXY` / `HTTPS_PROXY`.
  - If your environment exposes a generic HTTP file proxy or GitHub accelerator, the installer may use it when downloading `pixi` from `https://pixi.sh` or GitHub releases.

Our `install-comp` script will:

- Prefer `winget install -e --id prefix-dev.pixi` when available.
- Fall back to the official install script:
  ```powershell
  iwr https://pixi.sh/install.ps1 -UseBasicParsing | iex
  ```
- Honor `--proxy` and `--from-official` (which mainly controls whether a custom mirror/proxy is used versus direct access).

## Official installation

- Official docs:
  - https://pixi.sh
  - Installation guide: https://prefix-dev.github.io/pixi/dev/installation/

## Configuration

- To configure global mirrors (conda-forge and PyPI), run the configuration script:
  ```powershell
  # Set both to China mirrors (Tsinghua)
  .\config-comp.ps1 -Mirror cn

  # Set specific mirrors
  .\config-comp.ps1 -MirrorConda tuna -MirrorPypi aliyun

  # Reset to official
  .\config-comp.ps1 -Mirror official
  ```
- Parameters:
  - `-Mirror`: Sets both Conda and PyPI (e.g., `cn` sets both to `tuna`).
  - `-MirrorConda`: Overrides Conda mirror (`cn`, `official`, `tuna`).
  - `-MirrorPypi`: Overrides PyPI mirror (`cn`, `official`, `aliyun`, `tuna`).
- Configuration is saved to `%APPDATA%\pixi\config.toml` via `pixi config`.

## Linux/macOS (POSIX) scripts

- Install:
  ```bash
  cd components/pixi
  sh ./install-comp.sh --dry-run
  sh ./install-comp.sh
  ```
- Configure mirrors:
  ```bash
  sh ./config-comp.sh --mirror cn
  sh ./config-comp.sh --mirror-conda tuna --mirror-pypi aliyun
  sh ./config-comp.sh --mirror official
  ```
