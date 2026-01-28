# Pixi Installation Script (`install-pixi.bash`)

This script installs and configures the [Pixi](https://pixi.sh) package manager inside the container. It is typically run in the `stage-2` build phase via `user_config.yml`.

## Usage

```bash
./install-pixi.bash [OPTIONS]
```

## Options

| Option | Description |
| :--- | :--- |
| `--cache-dir=<path>` | Sets a custom cache directory. Useful for persisting cache in a volume. |
| `--install-dir=<path>` | Sets a custom installation directory (default: `~/.pixi`). |
| `--pypi-repo <name>` | Configures the PyPI mirror. Supported: `tuna`, `aliyun`, `official`. |
| `--conda-repo <name>` | Configures the Conda mirror. Supported: `tuna`, `official`. |
| `--verbose` | Enables verbose output for debugging. |

## Script Usage Examples

### Basic Installation
```yaml
custom:
  on_first_run:
    - 'stage-2/system/pixi/install-pixi.bash'
```

### Using Mirrors (China)
```yaml
custom:
  on_first_run:
    - 'stage-2/system/pixi/install-pixi.bash --pypi-repo=tuna --conda-repo=tuna'
```

### Persistent Cache
```yaml
custom:
  on_first_run:
    - 'stage-2/system/pixi/install-pixi.bash --cache-dir=/hard/volume/pixi-cache'
```

### Custom Installation Directory (Shared)
```yaml
custom:
  on_first_run:
    - 'stage-2/system/pixi/install-pixi.bash --install-dir=/hard/volume/pixi'
```

## Full Configuration Scenarios

### Pixi with Persistent Storage
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

### Pixi for Machine Learning (GPU)
Setup for ML/DL development with GPU support and pre-installed libraries.

```yaml
stage_1:
  device:
    type: gpu

stage_2:
  device:
    type: gpu
    
  storage:
    app:
      type: auto-volume
    workspace:
      type: auto-volume  
    data:
      type: auto-volume
  
  mount:
    models:
      type: auto-volume
      dst_path: /models
    home_mldev:
      type: auto-volume
      dst_path: /home/mldev
  
  environment:
    - 'NVIDIA_VISIBLE_DEVICES=all'
    - 'NVIDIA_DRIVER_CAPABILITIES=all'
  
  custom:
    on_first_run:
      - 'stage-2/system/pixi/install-pixi.bash'
      - 'stage-2/system/pixi/create-env-common.bash'
      - 'stage-2/system/pixi/create-env-ml.bash'
```