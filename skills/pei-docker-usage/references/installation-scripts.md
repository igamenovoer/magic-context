# Installation Scripts Reference

This document provides usage details for the system installation scripts included with PeiDocker. These scripts are designed to be used within `user_config.yml` (typically in `custom.on_build` or `custom.on_first_run`) to set up development environments.

## Common Features
Most scripts support:
*   `--user <username>`: Install for a specific user (requires root privileges).
*   **China Mirrors**: Options to configure domestic mirrors for faster downloads in China.

---

## Pixi (`stage-2/system/pixi/install-pixi.bash`)

Installs the [Pixi](https://pixi.sh) package manager.

### Usage
```bash
./install-pixi.bash [OPTIONS]
```

### Options
| Option | Description |
| :--- | :--- |
| `--cache-dir=<path>` | Sets a custom cache directory. |
| `--install-dir=<path>` | Sets a custom installation directory. |
| `--pypi-repo <name>` | Configures PyPI mirror (`tuna`, `aliyun`, `official`). |
| `--conda-repo <name>` | Configures Conda mirror (`tuna`, `official`). |
| `--installer-url <url>` | Override the installer script URL (`official`, `cn`, or custom URL). |

### Example (China)
```yaml
custom:
  on_build:
    - 'stage-2/system/pixi/install-pixi.bash --pypi-repo=tuna --conda-repo=tuna'
```

### More Examples

**Persistent Cache:**
```yaml
custom:
  on_first_run:
    - 'stage-2/system/pixi/install-pixi.bash --cache-dir=/hard/volume/pixi-cache'
```

**Custom Installation Directory (Shared):**
```yaml
custom:
  on_first_run:
    - 'stage-2/system/pixi/install-pixi.bash --install-dir=/hard/volume/pixi'
```

**Pixi with Persistent Storage (Full Scenario):**
Configures external volumes for storage and persists the Pixi cache.
```yaml
stage_2:
  storage:
    app:
      type: auto-volume
    workspace:
      type: auto-volume
    data:
      type: auto-volume
  
  mount:
    home_developer:
      type: auto-volume
      dst_path: /home/developer
  
  custom:
    on_first_run:
      # Install pixi with cache directory in external storage
      - 'stage-2/system/pixi/install-pixi.bash --cache-dir=/hard/volume/app/pixi-cache'
      - 'stage-2/system/pixi/create-env-common.bash'
```

---

## uv (`stage-1/system/uv/install-uv.sh`)

Installs [uv](https://github.com/astral-sh/uv), a fast Python package installer and manager.

### Usage
```bash
./install-uv.sh [OPTIONS]
```

### Options
| Option | Description |
| :--- | :--- |
| `--install-dir <path>` | Custom installation directory. |
| `--pypi-repo <name>` | Configures PyPI mirror (`tuna`, `aliyun`, `official`). |
| `--installer-url <url>` | Override the installer script URL (`official`, `cn`, or custom URL). |

### Example (China)
```yaml
custom:
  on_build:
    - 'stage-1/system/uv/install-uv.sh --pypi-repo=tuna'
```

---

## Node.js (`stage-2/system/nodejs/install-nodejs.sh`)

Installs Node.js LTS using NVM. **Requires NVM to be installed first.**

### Prerequisites
*   Run `install-nvm.sh` first.

### Usage
```bash
./install-nodejs.sh [OPTIONS]
```

### Options
| Option | Description |
| :--- | :--- |
| `--user <username>` | Target user. |
| `--version <ver>` | Node.js version (default: `lts`). |
| `--nvm-dir <dir>` | Custom NVM directory. |

---

## NVM (`stage-2/system/nodejs/install-nvm.sh`)

Installs [NVM](https://github.com/nvm-sh/nvm) (Node Version Manager).

### Usage
```bash
./install-nvm.sh [OPTIONS]
```

### Options
| Option | Description |
| :--- | :--- |
| `--install-dir <dir>` | Custom installation directory. |
| `--with-cn-mirror` | Configures npm registry and NVM mirrors (git, nodejs binary) to use Chinese mirrors. |
| `--version <ver>` | NVM version tag. |

### Example (China)
```yaml
custom:
  on_build:
    # 1. Install NVM with mirror support (run as target user for correct permissions)
    - "su - peid -c 'bash stage-2/system/nodejs/install-nvm.sh --with-cn-mirror'"
    # 2. Install Node.js
    - 'stage-2/system/nodejs/install-nodejs.sh --user peid'
```

---

## Bun (`stage-2/system/bun/install-bun.sh`)

Installs the [Bun](https://bun.sh) runtime.

### Usage
```bash
./install-bun.sh [OPTIONS]
```

### Options
| Option | Description |
| :--- | :--- |
| `--install-dir <dir>` | Custom installation directory. |
| `--npm-repo <url>` | Configures default npm registry in `bunfig.toml` (e.g., `https://registry.npmmirror.com`). |
| `--installer-url <url>` | Override the installer script URL (`official`, `cn`, or custom URL). |

### Example (China)
```yaml
custom:
  on_build:
    - 'stage-2/system/bun/install-bun.sh --npm-repo https://registry.npmmirror.com'
```
