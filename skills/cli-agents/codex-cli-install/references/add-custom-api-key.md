# Add Custom API Key

Use this subskill when the user wants a custom alias, function, or launcher that runs Codex against an OpenAI-compatible endpoint with a supplied API key.

## Goal

Create a reusable local entrypoint such as `codex-openai-proxy` that sets environment variables and then runs `codex --search`.

## Environment-first workflow

1. Confirm `codex` is already installed.
2. Determine the shell family and where persistent user commands should live.
3. Before asking for any key, inspect the existing environment variable names for likely API-key candidates.
4. Prefer an environment-variable-backed key instead of storing the raw key in plain text.
5. On Linux and macOS, prefer adding an alias or shell function to the user's shell profile by default.
6. Only create a separate launcher script when the user explicitly asks for a separate launcher script, standalone wrapper, or similar behavior.
7. If no suitable known environment variable already exists, ask the user to choose one of the approved key-source options.
8. Ensure the Codex provider configuration is updated to match the alias or chosen provider id.
9. For custom providers, set `wire_api = "responses"` in `config.toml`.
10. Do not use legacy chat/completions wiring for Codex custom providers.
11. Verify that the generated launcher or function resolves and forwards arguments correctly.

## API wiring policy

Use `wire_api = "responses"` for Codex custom providers.

This matches the upstream Codex implementation and test configuration, where custom providers are consistently configured with `wire_api = "responses"`.

Do not use legacy values such as `wire_api = "chat"`. Upstream Codex rejects that configuration and explicitly tells users to switch to `responses`.

When writing or revising `config.toml`, treat `responses` as the default and recommended wire protocol unless the upstream Codex project documents a newer replacement.

## Key discovery policy

Always try to discover an existing environment variable by name before asking the user for new input.

Check variable names only. Do not dump all environment variable values into chat.

Prefer names that clearly match the target provider or a generic OpenAI-compatible key, such as:

- `OPENAI_API_KEY`
- `CODEX_API_KEY`
- `OPENAI_PROXY_KEY`
- other obvious provider-specific names ending with `_KEY` or `_API_KEY`

If a suitable environment variable name already exists, prefer using that variable in the generated alias or function.

If no suitable variable name exists, ask the user to choose one of these paths:

1. Point to a non-default environment variable name that already contains the key.
2. Point to a plain text file that contains the key.
3. Type the API key directly.

If the user chooses option 1, use that exact environment variable name in the generated alias or function.

If the user chooses option 2, read the file, extract the key, and use it without echoing the key value back into chat.

If the user chooses option 3, proceed with the typed key. No additional security measure is taken in that path.

## Linux and macOS

Detect the active shell first and choose the matching profile file.

Examples:

```bash
printf '%s\n' "$SHELL"
env | cut -d= -f1 | grep -Ei '(openai|codex|api).*key|key.*(openai|codex|api)' || true
```

Typical profile choices:

- bash: `~/.bashrc`
- zsh: `~/.zshrc`
- fish: use the fish config flow instead of POSIX shell syntax

Preferred default: add a shell function to the user's shell profile.

If the environment already contains a suitable variable such as `OPENAI_API_KEY`, reference that variable in the shell function.

If no suitable known variable exists yet, ask the user to choose one of these:

1. give the name of a non-default environment variable that already contains the key,
2. give a plain text file path that contains the key,
3. type the key directly.

If the user provides a file path, read the file and extract the key without repeating its contents in chat.

Example for bash or zsh:

```bash
profile_file="$HOME/.bashrc"
cat >> "$profile_file" <<'SH'

codex-openai-proxy() {
  export OPENAI_BASE_URL='https://api.example.com/v1'
  export OPENAI_API_KEY="${OPENAI_API_KEY:?OPENAI_API_KEY is required}"
  codex --search "$@"
}
SH
. "$profile_file"
```

Only if the user explicitly asks for a separate launcher script, create one in `~/.local/bin`:

```bash
mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/codex-openai-proxy" <<'SH'
#!/usr/bin/env sh
set -eu
export OPENAI_BASE_URL='https://api.example.com/v1'
export OPENAI_API_KEY="${OPENAI_API_KEY:?OPENAI_API_KEY is required}"
exec codex --search "$@"
SH
chmod +x "$HOME/.local/bin/codex-openai-proxy"
```

After creating the function or launcher, ensure `config.toml` points to a matching provider and uses `requires_openai_auth = false`.

Recommended matching provider block:

```toml
model_provider = "codex-openai-proxy"

[model_providers.codex-openai-proxy]
name = "Custom OpenAI-compatible endpoint"
base_url = "https://api.example.com/v1"
env_key = "OPENAI_API_KEY"
requires_openai_auth = false
wire_api = "responses"
```

Optional helper script:

```bash
sh scripts/config-custom-api-key.sh --alias-name codex-openai-proxy --base-url "https://api.example.com/v1" --api-key-env OPENAI_API_KEY
```

Note: the helper script updates `config.toml` and can create either a profile function or a separate launcher script.

If the user chooses to type the key directly instead of using an environment variable or file input, the shell function can embed a fixed value, but that should be treated as a less safe fallback.

## Windows

Preferred PowerShell profile function that reads from an environment variable.

First inspect environment variable names:

```powershell
Get-ChildItem Env: | Select-Object -ExpandProperty Name | Where-Object {
    $_ -match '(?i)(openai|codex|api).*key|key.*(openai|codex|api)'
}
```

If a suitable variable already exists, use it in the profile function.

If none exists, ask the user to choose one of these:

1. give the name of a non-default environment variable that already contains the key,
2. give a plain text file path that contains the key,
3. type the key directly.

If the user provides a file path, read the file and extract the key without repeating its contents in chat.

```powershell
$aliasName = 'codex-openai-proxy'
$profilePath = Join-Path $env:USERPROFILE 'Documents\PowerShell\Microsoft.PowerShell_profile.ps1'
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $profilePath) | Out-Null
if (-not (Test-Path $profilePath)) { New-Item -ItemType File -Force -Path $profilePath | Out-Null }

@'
function codex-openai-proxy {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [object[]]$ForwardArgs
    )
    if (-not $env:OPENAI_API_KEY) { throw 'OPENAI_API_KEY is required.' }
    $env:OPENAI_BASE_URL = 'https://api.example.com/v1'
    $env:OPENAI_API_KEY = $env:OPENAI_API_KEY
    codex --search @ForwardArgs
}
'@ | Add-Content -Path $profilePath -Encoding UTF8
```

If the user explicitly wants a plain-text key in profile content, the same function can assign a fixed string instead of `$env:...`, but that should be treated as a less safe option.

If the user typed the key directly or asked you to read it from a plain text file, embedding that resolved key into profile content is allowed when that is the user's chosen path.

Recommended matching provider block:

```toml
model_provider = "codex-openai-proxy"

[model_providers.codex-openai-proxy]
name = "Custom OpenAI-compatible endpoint"
base_url = "https://api.example.com/v1"
env_key = "OPENAI_API_KEY"
requires_openai_auth = false
wire_api = "responses"
```

Optional helper script:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/config-custom-api-key.ps1 -AliasName codex-openai-proxy -BaseUrl "https://api.example.com/v1" -ApiKeyEnv OPENAI_API_KEY
```

Or:

```bat
scripts\config-custom-api-key.bat
```

## Verification

Linux/macOS:

```bash
type codex-openai-proxy
```

If a shell-profile function was added, also verify the profile contains it:

```bash
grep -n 'codex-openai-proxy()' "$HOME/.bashrc" "$HOME/.zshrc" 2>/dev/null || true
```

If a separate launcher script was requested, verify that file instead:

```bash
command -v codex-openai-proxy
head -n 20 "$HOME/.local/bin/codex-openai-proxy"
```

Windows:

```powershell
. $PROFILE
Get-Command codex-openai-proxy
```

In both cases, the launcher or function should forward arbitrary extra arguments to `codex`.

Also verify that the corresponding provider block in `config.toml` uses `wire_api = "responses"`.