# Conan Configuration Guide

Conan's behavior is controlled by several configuration files and environment variables. The most important ones are `global.conf` for core settings and profiles for build configurations.

## Location

By default, Conan stores its configuration and cache in the "Conan Home" directory:
- **Linux/Mac**: `~/.conan2/`
- **Windows**: `%USERPROFILE%/.conan2/`

You can override this location by setting the environment variable `CONAN_HOME`.

## Core Configuration (`global.conf`)

The `global.conf` file (located in `<CONAN_HOME>/global.conf`) controls system-wide behavior.

### 1. Storage & Cache

To change where Conan stores packages (the local cache):
```ini
# global.conf
core.cache:storage_path = /abs/path/to/storage
```
*Note: Must be an absolute path.*

To enable a shared download cache (useful for CI to avoid re-downloading artifacts):
```ini
# global.conf
core.download:download_cache = /abs/path/to/download_cache
```

### 2. Network & Proxies

```ini
# global.conf
core.net.http:timeout = 60
core.net.http:no_proxy_match = *.internal.com

# Proxies are often better handled via standard env vars, 
# but can be explicit here:
core.net.http:proxies = {"http": "http://user:pass@proxy:8080", "https": "http://user:pass@proxy:8080"}
```

### 3. Build Behavior

Skip the tedious "missing binary" errors by defaulting to build from source:
```ini
# global.conf
# Equivalent to always passing --build=missing
tools.build:missing_binary_policy = build_missing
```

## Profiles (`profiles/`)

Profiles define the target environment (OS, compiler, arch) for your build. The `default` profile is auto-generated.

### Common Profile Settings

```ini
# profiles/default
[settings]
os=Linux
arch=x86_64
compiler=gcc
compiler.version=11
compiler.cppstd=20
build_type=Release

[conf]
# Pass flags to the compiler
tools.build:cxxflags=["-fPIC", "-O3"]
tools.build:cflags=["-fPIC"]

# Use a specific linker
tools.build:exelinkflags=["-fuse-ld=gold"]
```

## Managing Configuration (`conan config install`)

Instead of manually editing files on every machine, use `conan config install` to distribute configuration from a git repo or zip file.

```bash
# Install from a git repository
conan config install https://github.com/myorg/conan-config.git

# Install from a local folder
conan config install ./my-conan-config/
```

This effectively synchronizes `global.conf`, profiles, remotes, and settings across your team or CI agents.
