# User Config Reference (`user_config.yml`)

The `user_config.yml` file is the heart of a PeiDocker project. It defines your environment across two build stages.

## Root Structure

```yaml
stage_1:
  # System-level configuration (OS, SSH, Proxy, Apt)
  ...

stage_2:
  # Application-level configuration (App install, Volumes, Dev tools)
  ...
```

## Stage Configuration

Both `stage_1` and `stage_2` support the following sections. Note that some sections (like `ssh` and `apt`) are typically only used in `stage_1`.

### `image` (Required in Stage 1)
Defines the Docker image inputs and outputs.

```yaml
image:
  base: ubuntu:24.04        # Base image (Required in Stage 1)
  output: my-app:stage-1    # Tag for the built image
```

### `ssh` (Stage 1 Only)
Configures the internal SSH server.

```yaml
ssh:
  enable: true              # Enable SSH server
  port: 22                  # Container's internal SSH port
  host_port: 2222           # Port exposed on the host
  users:
    username:               # Username to create
      password: 'password'  # Password (required, NO spaces or commas)
      uid: 1000             # Optional: Force specific UID
      pubkey_file: '~'      # Optional: Path to public key ('~' uses system key)
```

### `proxy`
Configures HTTP/HTTPS proxies for build and run time.

```yaml
proxy:
  address: host.docker.internal # Proxy host
  port: 7890                    # Proxy port
  enable_globally: false        # If true, sets http_proxy ENV vars globally
  remove_after_build: false     # If true, unsets proxy vars after build
  use_https: false              # If true, uses http://... for https_proxy
```

### `apt` (Stage 1 Only)
Configures APT package manager sources.

```yaml
apt:
  repo_source: 'tuna'           # Mirrors: 'tuna', 'aliyun', '163', 'ustc', 'cn'
  keep_repo_after_build: true   # Keep source list after build
  use_proxy: false              # Use configured proxy for apt-get
```

### `storage` (Stage 2 Only)
Defines the dynamic storage locations. PeiDocker standardizes on three paths: `/soft/app`, `/soft/data`, `/soft/workspace`.

```yaml
storage:
  app:
    type: auto-volume       # Options: auto-volume, manual-volume, host, image
  data:
    type: host
    host_path: /local/data  # Required for type: host
  workspace:
    type: manual-volume
    volume_name: my-vol     # Required for type: manual-volume
```

### `mount`
Mounts additional volumes or paths to arbitrary locations in the container.

**Warning:** Do not use `workspace`, `app`, or `data` as keys in this section. These are reserved keywords. If used, PeiDocker will force their destination path to the internal storage location (e.g., `/hard/volume/workspace`), ignoring your `dst_path`.

```yaml
mount:
  my_config:                # Arbitrary name (do NOT use 'workspace', 'app', or 'data')
    type: host
    host_path: ./config
    dst_path: /etc/myapp    # Destination in container
```

### `custom`
Defines scripts to run at various lifecycle events.

```yaml
custom:
  on_build:                 # Runs during `docker build`
    - 'stage-2/custom/install.sh'
  on_first_run:             # Runs ONCE when container first starts
    - 'stage-2/custom/setup.sh'
  on_every_run:             # Runs every time container starts
    - 'stage-2/custom/startup.sh'
  on_user_login:            # Runs when user SSHs in
    - 'stage-2/custom/welcome.sh'
```

### `device`
Configures hardware access (GPU).

```yaml
device:
  type: gpu                 # Options: cpu, gpu
```

### `ports`
Exposes additional ports.

```yaml
ports:
  - "8080:80"               # host:container
  - "3000-3005:3000-3005"   # ranges
```

### `environment`
Sets environment variables. Supports both list and dictionary formats.

```yaml
# List format (recommended for simple vars)
environment:
  - 'NODE_ENV=production'
  - 'API_KEY=${API_KEY}'    # Supports variable substitution

# Dictionary format
environment:
  NODE_ENV: production
  API_KEY: "${API_KEY}"
```
