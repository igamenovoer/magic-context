# Advanced Examples

## Environment Variable Substitution

PeiDocker supports environment variable substitution in `user_config.yml` using Docker Compose syntax (`${VARIABLE_NAME:-default_value}`).

### Examples

#### Configurable SSH Port
```yaml
stage_1:
  ssh:
    enable: true
    port: 22
    host_port: "${SSH_HOST_PORT:-2222}"
```

**Usage:**
```bash
export SSH_HOST_PORT=3333
pixi run pei-docker-cli configure
```

#### Environment-Specific Base Images
```yaml
stage_1:
  image:
    base: "${BASE_IMAGE:-ubuntu:24.04}"
    output: "${OUTPUT_IMAGE:-my-app}:stage-1"
```

**Usage:**
```bash
export BASE_IMAGE='nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04'
pixi run pei-docker-cli configure
```

## Advanced SSH Key Configuration

You can use environment variables to inject SSH keys, keeping sensitive data out of your version-controlled `user_config.yml`. This is especially useful for teams where each developer needs access using their own key.

### Injecting Public Keys via Environment Variables

```yaml
stage_1:
  ssh:
    enable: true
    port: 22
    host_port: 2222
    users:
      developer:
        password: 'password123'
        # Inject public key from environment variable
        pubkey_text: "${SSH_PUBKEY}"
```

**Usage:**
```bash
# Export your public key to an environment variable
export SSH_PUBKEY=$(cat ~/.ssh/id_rsa.pub)
pixi run pei-docker-cli configure
```

### Injecting Private Keys (e.g., for Git Access)

You can also inject private keys, for example, to allow the container to access private Git repositories.

**Important:** PeiDocker copies the private key **as-is**. It does **not** attempt to decrypt it or prompt for passphrases during the build. If you provide an encrypted private key, it will remain encrypted inside the container, and you will need to provide the passphrase when using it.

**Note:** Handling multi-line private keys in environment variables can be tricky. Ensure newlines are preserved.

```yaml
stage_1:
  ssh:
    users:
      deploy_user:
        password: 'password123'
        # Inject private key from environment variable
        privkey_text: "${SSH_PRIVKEY}"
```

**Usage:**

```bash
# Export private key preserving newlines (quote the variable)
export SSH_PRIVKEY="$(cat ~/.ssh/id_rsa)"
pixi run pei-docker-cli configure
```

### Team Configuration Pattern

Use a default value that works for a shared "dev" key, but allow overriding:

```yaml
stage_1:
  ssh:
    users:
      me:
        password: '123456'
        # Default to a placeholder or shared key if not provided
        pubkey_text: "${MY_PUBKEY:-ssh-rsa AAAAB3... shared-key}"
```

## Hardware-accelerated OpenGL (Windows/WSL2)

To enable GPU acceleration for GUI apps on Windows via WSL2:

```yaml
stage_1:
  image:
    base: nvidia/cuda:12.3.2-base-ubuntu22.04
    output: pei-opengl:stage-1
  device:
    type: gpu
    
stage_2:
  image:
    output: pei-opengl:stage-2
  device:
    type: gpu
  environment:
    NVIDIA_VISIBLE_DEVICES: all
    NVIDIA_DRIVER_CAPABILITIES: all
  custom:
    on_build: 
      - 'stage-2/system/opengl/setup-opengl-win32.sh'
```

**Note:** You must also mount WSLg directories in the generated `docker-compose.yml` (PeiDocker does not currently automate these specific mounts, so you may need to add them manually to the `volumes` section of the generated file if not handled by your base image or other tools):
- `/tmp/.X11-unix:/tmp/.X11-unix`
- `/mnt/wslg:/mnt/wslg`
- `/usr/lib/wsl:/usr/lib/wsl`
- `/dev:/dev`