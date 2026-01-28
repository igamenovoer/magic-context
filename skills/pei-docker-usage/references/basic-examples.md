# Examples

[](){#basic-ssh}
## Basic Ubuntu image with SSH support

This is the most basic example of creating an image with SSH support. The image is based on `ubuntu:24.04` and has three users: `me`, `you`, and `root`. The passwords for the users are `123456`, `654321`, and `root` respectively. The SSH server is running on port `22` and mapped to host port `2222`. 

To accelerate apt installation in China, the `apt` source is set to `tuna`. For other options, see the full [config file](user-config-reference.md) for documentation. If this option is omitted, the default source will be used.

**Note:** This example uses password-only authentication. For advanced SSH key configuration (including inline keys and file-based keys), see the [SSH Key Configuration](#ssh-key-configuration) section below. For environment variable integration, see [Advanced Examples](advanced-examples.md).

```yaml
# user_config.yml under project directory

stage_1:
  image:
    base: ubuntu:24.04
    output: pei-image:stage-1
  ssh:
    enable: true
    port: 22
    host_port: 2222
    users:
      me:
        password: '123456'
      you:
        password: '654321'
      root:
        password: root
  apt:
    repo_source: tuna
```

## SSH Key Configuration

While password authentication works well for basic usage, you can also configure SSH keys for more secure access. PeiDocker supports using your existing system SSH keys with the `~` syntax for automatic discovery.

### Using System SSH Keys

When you specify `~` for `pubkey_file` or `privkey_file`, PeiDocker will automatically discover and use your system SSH keys from your home directory (`~/.ssh/`). The discovery follows this priority order: `id_rsa`, `id_dsa`, `id_ecdsa`, `id_ed25519`.

```yaml
# user_config.yml
stage_1:
  ssh:
    enable: true
    port: 22
    host_port: 2222
    users:
      me:
        password: '123456'
        pubkey_file: '~'        # Auto-discovers your system public key
      admin:
        password: 'admin123'
        privkey_file: '~'       # Auto-discovers your system private key
  # ... other settings ...
```

### Important Notes

- **Public keys**: When using `pubkey_file: '~'`, your system public key will be added to the container's `authorized_keys`, allowing passwordless SSH access
- **Private keys**: When using `privkey_file: '~'`, your system private key will be copied to the container using standard SSH filenames (`id_rsa`, `id_ecdsa`, etc.)
- **Encrypted private keys**: If your private key is encrypted, it will be copied as-is without decryption attempts (no passphrase prompts during build)
- **Manual public key generation**: For encrypted private keys, you can manually generate the public key inside the container: `ssh-keygen -y -f ~/.ssh/id_rsa > ~/.ssh/id_rsa.pub`
- **Absolute paths**: You can also use absolute paths like `/home/user/.ssh/id_rsa.pub` instead of the `~` syntax

With this configuration, you can SSH into the container using your existing SSH keys:

```bash
# Using your private key (if pubkey_file was configured)
ssh -p 2222 me@127.0.0.1

# Or specify your key explicitly
ssh -i ~/.ssh/id_rsa -p 2222 admin@127.0.0.1
```

### SSH Key Methods

PeiDocker supports multiple methods for configuring SSH keys for users, giving you flexibility in how you manage authentication. All SSH key processing (such as public key generation from private keys) happens inside the container during the build process, ensuring cross-platform compatibility.

#### Method 1: Public Key File (Legacy)
Reference an existing public key file relative to the installation directory:

```yaml
users:
  myuser:
    password: 'mypassword'
    pubkey_file: 'stage-1/system/ssh/keys/my-key.pub'
```

#### Method 2: Inline Public Key Text
Provide the public key content directly in the configuration:

```yaml
users:
  myuser:
    password: 'mypassword'
    pubkey_text: 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC7vbqajDhA... user@host'
```

#### Method 3: Inline Private Key Text
Provide the private key content directly - the public key will be automatically generated inside the container:

```yaml
users:
  myuser:
    password: 'mypassword'
    privkey_text: |
      -----BEGIN OPENSSH PRIVATE KEY-----
      b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAABlwAAAAdzc2g...
      -----END OPENSSH PRIVATE KEY-----
```

#### Method 4: Private Key File
Reference a private key file - the public key will be automatically generated inside the container:

```yaml
users:
  myuser:
    password: 'mypassword'
    privkey_file: 'stage-1/system/ssh/keys/my-private-key'
```

### Example Configurations

#### Basic SSH setup with inline public key:
```yaml
stage_1:
  ssh:
    enable: true
    host_port: 2222
    users:
      developer:
        password: 'devpass123'
        pubkey_text: 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC7vbqajDhA... dev@laptop'
```

#### Multiple users with different key methods:
```yaml
stage_1:
  ssh:
    enable: true
    host_port: 2222
    users:
      alice:
        password: 'alice123'
        pubkey_file: 'stage-1/system/ssh/keys/alice-key.pub'  # File-based public key
      bob:
        password: 'bob456'
        privkey_text: |  # Inline private key
          -----BEGIN OPENSSH PRIVATE KEY-----
          b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAABlwAAAAdzc2g...
          -----END OPENSSH PRIVATE KEY-----
      charlie:
        password: 'charlie789'
        pubkey_text: 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... charlie@workstation'  # Inline public key
```

_For more advanced SSH key configuration including environment variable integration and team development scenarios, see [Advanced Examples](advanced-examples.md)._

## GPU image with external storage

This example is based on [the basic ssh image](#basic-ssh), which demonstrates how to use GPU in the container. The image is based on `nvidia/cuda:11.8.0-runtime-ubuntu22.04` that makes use of the GPU. As such, the `device` section is added to the `stage_1` and `stage_2` sections, which are set to `gpu`.

```yaml
stage_1:
  # ...
  device:
    type: gpu

stage_2:
  device:
    type: gpu
  storage:
    app:
      type: host
      host_path: d:/code/PeiDocker/build/storage/app
    data:
      type: host
      host_path: d:/code/PeiDocker/build/storage/data
    workspace:
      type: host
      host_path: d:/code/PeiDocker/build/storage/workspace
```

### External storage with host directory

The `stage-2` image has three external storage directories: `app`, `data`, and `workspace` (note that the directory names are **NOT CUSTOMIZABLE**, for arbitrary mounts, see [the next section](#mount-additional-volumes)), where specified host directories are mounted. In the container, you can access these directories through `/soft/app`, `/soft/data`, and `/soft/workspace`, which are linked to  `/app`, `/data`, and `/workspace` under the `/hard/volume`. In this example, the host directories are `d:/code/PeiDocker/build/storage/app`, `d:/code/PeiDocker/build/storage/data`, and `d:/code/PeiDocker/build/storage/workspace` (Windows path).

[](){#mount-additional-volumes}
### Mounting arbitrary volumes or directories

You can mount additional volumes or directories to the container by adding them to the `mount` section. The following example demonstrates how to mount the `apt_cache` directory to the container's `/var/cache/apt` directory, in this way, the apt cache will be saved for future use. `home_me` is mounted to `/home/me` to save the home directory to a volume, so that it will not get lost after the container is deleted. 

**Warning:** Do not use `workspace`, `app`, or `data` as keys in the `mount` section. These are reserved keywords managed by the `storage` system. If used, PeiDocker will force their destination path to the internal storage location, ignoring your `dst_path`.

Unlike `app`, `data`, and `workspace`, the mounted volumes `apt_cache` and `home_me` will not be linked to `/soft`, they are mounted directly to the container and not managed by PeiDocker. You can also use `manual-volume` (with `volume_name` set) or `host` (with `host_path` set) in ther `type` though.

```yaml
# heavy duty cpp development, install a lot of things

stage_1:
  # ...
  # additional mounts
  mount:
    # save apt cache, to speed up future installations
    apt_cache:
      type: auto-volume
      dst_path: /var/cache/apt
    
stage_2:
  # ...
  # additional mounts
  mount:
    # save home directory to volume, so that it will not get lost after container deletion
    home_me:
      type: auto-volume
      dst_path: /home/me

    # mount will not be inherited from stage-1
    # you need to mount it again, or otherwise it will not be mounted
    apt_cache:
      type: auto-volume
      dst_path: /var/cache/apt
```

[](){#external-storage}
## Using docker volume as external storage

Using [docker volumes](https://docs.docker.com/storage/volumes/) is preferred if you run the image locally, because it is more efficient, and will not get lost when the container is removed. Docker volumes can be created automatically, or [manually](https://docs.docker.com/reference/cli/docker/volume/create/) with a given name that can be used to mount the volume to the container. 

This example is based on [the basic ssh image](#basic-ssh), which demonstrates how to use existing docker volumes as external storage. The `stage-2` image has three external storage directories: `app`, `data`, and `workspace` (note that the directory names are **NOT CUSTOMIZABLE**, they are **predefined**), where **docker volumes** are mounted. 


```yaml
# user_config.yml
stage_2:
  storage:
    app:
      type: auto-volume
    data:
      type: manual-volume
      volume_name: my_data
    workspace:
      type: manual-volume
      volume_name: my_workspace
```

`app` is an **auto-volume**, which means it will be created automatically. `data` and `workspace` are **manual-volumes**, which means they are created manually with the names `my_data` and `my_workspace` first, and then mounted to the container. To create these volumes, use the following commands

```bash
docker volume create my_data
docker volume create my_workspace
``` 

You can check the volumes with `docker volume ls`.

In the container, you can access these directories through `/soft/app`, `/soft/data`, and `/soft/workspace`, which are linked to  `/app`, `/data`, and `/workspace` under the `/hard/volume`.

[](){#install-pixi}
## Install pixi package manager

### Basic pixi installation

This example demonstrates how to automatically install pixi and set up a common development environment with essential Python packages. The installation uses the tuna mirror for faster downloads in China.

```yaml
# user_config.yml - basic pixi installation
stage_2:
  # custom scripts to install and configure pixi
  custom:
    on_first_run:
      - 'stage-2/system/pixi/install-pixi.bash'
      - 'stage-2/system/pixi/create-env-common.bash'

_For configuration options (mirrors, cache), see [Pixi Installation Guide](install-pixi-usage.md)._
```





### Verify Installation:

After the container starts, SSH in and verify pixi is working:

```bash
# SSH into the container
ssh -p 2222 developer@127.0.0.1

# Check pixi installation
pixi --version

# List global environments
pixi global list

# Verify packages are available
python -c "import scipy; print('scipy:', scipy.__version__)"
```



## Custom script parameters

PeiDocker supports passing parameters to custom scripts using shell-like syntax. This allows you to customize script behavior without modifying the script files themselves.

```yaml
stage_1:
  custom:
    on_build:
      - 'stage-1/custom/my-build.sh --verbose --config=/tmp/build.conf'
    # ...

stage_2:
  custom:
    on_build:
      - 'stage-2/custom/install-app.sh --enable-desktop --theme=dark'
    # ...
```

### How it works:

- Parameters are safely parsed using shell-like syntax
- Scripts receive parameters as standard command-line arguments
- Both single and double quotes are supported for complex parameters
- Works across all script lifecycle events (on_build, on_first_run, on_every_run, on_user_login)

### Example script implementation:

```bash
#!/bin/bash
# example script showing parameter handling

# Default values
VERBOSE=false
CONFIG_FILE=""

# Parse parameters
for arg in "$@"; do
    case $arg in
        --verbose)
            VERBOSE=true
            ;;
        --config=*)
            CONFIG_FILE="${arg#*=}"
            ;;
        *)
            echo "Unknown parameter: $arg"
            ;;
    esac
done

# Use parameters
if [ "$VERBOSE" = true ]; then
    echo "Verbose mode enabled"
fi
```

## Using proxy

You can use proxy when building the image. The following example demonstrates how to use proxy for `stage_1`. The proxy is set to `host.docker.internal:30080`, which refers to `127.0.0.1:30080` in host where the proxy is running. The `apt` is specified to use the proxy, and the proxy is kept after the build.

```yaml
stage_1:
  # ...
  proxy:
    address: host.docker.internal
    port: 30080
    enable_globally: false  # if set to true, the proxy is used during build and run
    remove_after_build: false # if set to true, the proxy is removed after build, not effecting run
    use_https: false  # false means http proxy is used where https proxy is requested
  apt:
    use_proxy: true   # use the proxy for apt
    keep_proxy_after_build: true  # keep apt using the proxy after build
```

### Using proxy during build and run

This example shows how to use proxy during build and run. The key is to set `enable_globally` to `true`. If you set `remove_after_build` to `true`, the proxy will be removed after the build, and not effecting the run, this is useful if you only want to use the proxy during build (e.g., pulling sources from github).

```yaml
stage_1:
  # ...
  proxy:
    address: host.docker.internal
    port: 30080
    enable_globally: true
    remove_after_build: false
    use_https: false
  apt:
    repo_source: tuna
```

### Using proxy across stages

This example shows how to use proxy across stages. The proxy is set in `stage-1`, but only used for `apt`. In `stage-2`, the proxy is enabled globally, affecting all commands after run. 

```yaml
stage_1:
  # ...
  proxy:
    address: host.docker.internal
    port: 30080
    enable_globally: false
    remove_after_build: false
    use_https: false
  apt:
    use_proxy: true
    keep_proxy_after_build: true
stage_2:
  proxy:
    enable_globally: true
```

You can verify the proxy is set by running `env` after ssh into the container.

```sh
# inside container
env | grep -i proxy

# you should see the proxy settings
# http_proxy=http://host.docker.internal:30080
# https_proxy=http://host.docker.internal:30080
```

### Using proxy manually

If you specify the proxy, but never use it, it will not automatically affect anything. But you can find the proxy setting in the container, using environment variables `PEI_HTTP_PROXY_1` and `PEI_HTTPS_PROXY_1` (replace `1` with the stage number). 

```yaml
stage_1:
  # ...
  proxy:
    address: host.docker.internal
    port: 30080
  apt:
    repo_source: tuna
```

After ssh into the container, you can find the proxy settings:

```sh
# inside container
env | grep -i proxy

# you should see the proxy settings
# PEI_HTTP_PROXY_1=http://host.docker.internal:30080
# PEI_HTTPS_PROXY_1=http://host.docker.internal:30080
```
