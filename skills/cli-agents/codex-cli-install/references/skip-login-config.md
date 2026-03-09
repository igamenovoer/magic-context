# Skip Login Config

Use this subskill when Codex is installed but the host should bypass the built-in login flow by switching to a custom model provider in `config.toml`.

## Goal

Configure `CODEX_HOME/config.toml` so Codex uses a custom provider with `requires_openai_auth = false` and reads the key from an environment variable.

## Environment-first workflow

1. Confirm `codex` is already installed.
2. Determine `CODEX_HOME`, defaulting to `~/.codex` on Linux/macOS or `%USERPROFILE%\.codex` on Windows.
3. Choose or create a provider id such as `codex-custom` or one that matches a wrapper alias.
4. Update `model_provider` and the matching `[model_providers.<provider-id>]` block.
5. Verify the resulting TOML includes `env_key` and `requires_openai_auth = false`.

## Linux and macOS

Inspect the current configuration:

```bash
command -v codex
codex_home="${CODEX_HOME:-$HOME/.codex}"
config_path="$codex_home/config.toml"
mkdir -p "$codex_home"
test -f "$config_path" && cat "$config_path" || true
```

Minimal manual configuration example:

```bash
codex_home="${CODEX_HOME:-$HOME/.codex}"
config_path="$codex_home/config.toml"
provider_id="codex-custom"
mkdir -p "$codex_home"
cat > "$config_path" <<EOF
model_provider = "$provider_id"

[model_providers]

[model_providers.$provider_id]
name = "Custom OpenAI-compatible endpoint"
base_url = "https://api.openai.com/v1"
env_key = "OPENAI_API_KEY"
env_key_instructions = "Set OPENAI_API_KEY in your environment before launching codex."
requires_openai_auth = false
wire_api = "responses"
EOF
```

Optional helper script:

```bash
sh scripts/config-skip-login.sh --provider-id codex-custom --env-key OPENAI_API_KEY
```

## Windows

Inspect the current configuration:

```powershell
Get-Command codex
$codexHome = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $env:USERPROFILE '.codex' }
$configPath = Join-Path $codexHome 'config.toml'
New-Item -ItemType Directory -Force -Path $codexHome | Out-Null
if (Test-Path $configPath) { Get-Content $configPath -Raw }
```

Minimal manual configuration example:

```powershell
$providerId = 'codex-custom'
$codexHome = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $env:USERPROFILE '.codex' }
$configPath = Join-Path $codexHome 'config.toml'
New-Item -ItemType Directory -Force -Path $codexHome | Out-Null
@"
model_provider = "$providerId"

[model_providers]

[model_providers.$providerId]
name = "Custom OpenAI-compatible endpoint"
base_url = "https://api.openai.com/v1"
env_key = "OPENAI_API_KEY"
env_key_instructions = "Set OPENAI_API_KEY in your environment before launching codex."
requires_openai_auth = false
wire_api = "responses"
"@ | Set-Content -Path $configPath -Encoding UTF8
```

Optional helper script:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/config-skip-login.ps1 -ProviderId codex-custom -EnvKey OPENAI_API_KEY
```

Or:

```bat
scripts\config-skip-login.bat
```

## Verification

Linux/macOS:

```bash
grep -n 'model_provider' "${CODEX_HOME:-$HOME/.codex}/config.toml"
grep -n 'requires_openai_auth = false' "${CODEX_HOME:-$HOME/.codex}/config.toml"
```

Windows:

```powershell
Get-Content (Join-Path $(if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $env:USERPROFILE '.codex' }) 'config.toml')
```

The resulting TOML should point `model_provider` to the chosen provider and include `requires_openai_auth = false`.