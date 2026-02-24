# uv (Astral)

> **Note:** Before running the `.ps1` script, please run the `<workspace>/enable-ps1-permission.bat` script once to allow PowerShell script execution.

This component installs [uv](https://astral.sh/uv), an extremely fast Python package and project manager.

## Preferred installation (winget)

- Winget package:
  ```powershell
  winget install -e --id astral-sh.uv
  ```

## China-friendly installation

- uv is distributed from `astral.sh` and GitHub; there is no dedicated China mirror, but:
  - `winget` plus a configured proxy is usually the easiest path.
  - You can also run the official installer behind a proxy:
    ```powershell
    $env:HTTP_PROXY = "<proxy-url>"
    $env:HTTPS_PROXY = "<proxy-url>"
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

Our `install-comp` script will:

- Prefer `winget install -e --id astral-sh.uv`.
- Fall back to the official installer script (`https://astral.sh/uv/install.ps1`).
- Respect `--proxy / -Proxy` and `--from-official` (direct Astral/GitHub vs any enterprise proxy/mirror).

## Official installation

- Official docs:
  - https://docs.astral.sh/uv/getting-started/installation/

## Configuration

- To configure the global package index (mirror), run the configuration script:
  ```powershell
  .\config-comp.ps1 -Mirror cn
  # or
  .\config-comp.ps1 -Mirror aliyun
  # or
  .\config-comp.ps1 -Mirror tuna
  # or
  .\config-comp.ps1 -Mirror official
  ```
- This updates `%APPDATA%\uv\uv.toml` to set `index-url`.
- `cn` defaults to `aliyun`.

## Linux/macOS (POSIX) scripts

- Install:
  ```bash
  cd components/uv
  sh ./install-comp.sh --dry-run
  sh ./install-comp.sh
  ```
- Configure PyPI mirror:
  ```bash
  sh ./config-comp.sh --mirror cn
  sh ./config-comp.sh --mirror tuna
  sh ./config-comp.sh --mirror official
  ```
- Note: `config-comp.sh` writes `~/.config/uv/uv.toml` (or `$XDG_CONFIG_HOME/uv/uv.toml`).
