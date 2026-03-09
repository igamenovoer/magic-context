# Add Custom API Key

Use this subskill when the user wants a custom launcher or shell/profile function that runs Claude Code against an Anthropic-compatible endpoint with a supplied API key.

## Goal

Create a reusable local entrypoint such as `claude-kimi` that sets environment variables and then runs `claude --dangerously-skip-permissions`.

## Environment-first workflow

1. Confirm `claude` is already installed.
2. Determine the shell family and where persistent user commands should live.
3. Before asking for any key, inspect the existing environment variable names for likely API-key candidates.
4. Prefer an environment-variable-backed key instead of storing the raw key in plain text.
5. On Linux and macOS, prefer adding an alias or shell function to the user's shell profile by default.
6. Only create a separate launcher script when the user explicitly asks for a separate launcher script, standalone wrapper, or similar behavior.
7. If no suitable known environment variable already exists, ask the user to choose one of the approved key-source options.
8. If the user explicitly wants a fully self-contained launcher, warn that it stores the key locally in plain text.
9. Verify that the generated launcher or function resolves and forwards arguments correctly.

## Key discovery policy

Always try to discover an existing environment variable by name before asking the user for new input.

Check variable names only. Do not dump all environment variable values into chat.

Prefer names that clearly match the target provider or a generic Anthropic-compatible key, such as:

- `ANTHROPIC_API_KEY`
- `KIMI_ANTHROPIC_KEY`
- `MOONSHOT_API_KEY`
- `CLAUDE_API_KEY`
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
env | cut -d= -f1 | grep -Ei '(anthropic|claude|kimi|moonshot|api).*key|key.*(anthropic|claude|kimi|moonshot|api)' || true
```

Typical profile choices:

- bash: `~/.bashrc`
- zsh: `~/.zshrc`
- fish: use the fish config flow instead of POSIX shell syntax

Preferred default: add a shell function to the user's shell profile.

If the environment already contains a suitable variable such as `KIMI_ANTHROPIC_KEY`, reference that variable in the shell function.

If no suitable known variable exists yet, ask the user to choose one of these:

1. give the name of a non-default environment variable that already contains the key,
2. give a plain text file path that contains the key,
3. type the key directly.

If the user provides a file path, read the file and extract the key without repeating its contents in chat.

Example for bash or zsh:

```bash
profile_file="$HOME/.bashrc"
cat >> "$profile_file" <<'SH'

claude-kimi() {
    export ANTHROPIC_BASE_URL='https://api.moonshot.cn/anthropic'
    export ANTHROPIC_API_KEY="${KIMI_ANTHROPIC_KEY:?KIMI_ANTHROPIC_KEY is required}"
    claude --dangerously-skip-permissions "$@"
}
SH
. "$profile_file"
```

If the provider needs fixed model overrides, add them inside the same function:

```bash
profile_file="$HOME/.zshrc"
cat >> "$profile_file" <<'SH'

claude-custom() {
    export ANTHROPIC_BASE_URL='https://example.com/anthropic'
    export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY is required}"
    export ANTHROPIC_MODEL='provider-primary-model'
    export ANTHROPIC_DEFAULT_OPUS_MODEL='provider-primary-model'
    export ANTHROPIC_DEFAULT_SONNET_MODEL='provider-primary-model'
    export ANTHROPIC_DEFAULT_HAIKU_MODEL='provider-secondary-model'
    export CLAUDE_CODE_SUBAGENT_MODEL='provider-secondary-model'
    claude --dangerously-skip-permissions "$@"
}
SH
. "$profile_file"
```

Only if the user explicitly asks for a separate launcher script, create one in `~/.local/bin`:

```bash
mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/claude-kimi" <<'SH'
#!/usr/bin/env sh
set -eu
export ANTHROPIC_BASE_URL='https://api.moonshot.cn/anthropic'
export ANTHROPIC_API_KEY="${KIMI_ANTHROPIC_KEY:?KIMI_ANTHROPIC_KEY is required}"
exec claude --dangerously-skip-permissions "$@"
SH
chmod +x "$HOME/.local/bin/claude-kimi"
```

Optional helper script:

```bash
sh scripts/config-custom-api-key.sh --alias-name claude-kimi --base-url "https://api.moonshot.cn/anthropic" --api-key-env KIMI_ANTHROPIC_KEY
```

Note: the helper script currently creates a separate launcher script, so use it only when that behavior matches the user's request.

If the user chooses to type the key directly instead of using an environment variable or file input, the shell function can embed a fixed value, but that should be treated as a less safe fallback.

## Windows

Preferred PowerShell profile function that reads from an environment variable:

First inspect environment variable names:

```powershell
Get-ChildItem Env: | Select-Object -ExpandProperty Name | Where-Object {
    $_ -match '(?i)(anthropic|claude|kimi|moonshot|api).*key|key.*(anthropic|claude|kimi|moonshot|api)'
}
```

If a suitable variable already exists, use it in the profile function.

If none exists, ask the user to choose one of these:

1. give the name of a non-default environment variable that already contains the key,
2. give a plain text file path that contains the key,
3. type the key directly.

If the user provides a file path, read the file and extract the key without repeating its contents in chat.

```powershell
$aliasName = 'claude-kimi'
$profilePath = Join-Path $env:USERPROFILE 'Documents\PowerShell\Microsoft.PowerShell_profile.ps1'
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $profilePath) | Out-Null
if (-not (Test-Path $profilePath)) { New-Item -ItemType File -Force -Path $profilePath | Out-Null }

@'
function claude-kimi {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [object[]]$ForwardArgs
    )
    if (-not $env:KIMI_ANTHROPIC_KEY) { throw 'KIMI_ANTHROPIC_KEY is required.' }
    $env:ANTHROPIC_BASE_URL = 'https://api.moonshot.cn/anthropic'
    $env:ANTHROPIC_API_KEY = $env:KIMI_ANTHROPIC_KEY
    claude --dangerously-skip-permissions @ForwardArgs
}
'@ | Add-Content -Path $profilePath -Encoding UTF8
```

If the user explicitly wants a plain-text key in profile content, the same function can assign a fixed string instead of `$env:...`, but that should be treated as a less safe option.

If the user typed the key directly or asked you to read it from a plain text file, embedding that resolved key into profile content is allowed when that is the user's chosen path.

Optional helper script:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/config-custom-api-key.ps1 -AliasName claude-kimi -BaseUrl "https://api.moonshot.cn/anthropic" -ApiKeyEnv KIMI_ANTHROPIC_KEY
```

Or:

```bat
scripts\config-custom-api-key.bat
```

## Verification

Linux/macOS:

```bash
type claude-kimi
```

If a shell-profile function was added, also verify the profile contains it:

```bash
grep -n 'claude-kimi()' "$HOME/.bashrc" "$HOME/.zshrc" 2>/dev/null || true
```

If a separate launcher script was requested, verify that file instead:

```bash
command -v claude-kimi
head -n 20 "$HOME/.local/bin/claude-kimi"
```

Windows:

```powershell
. $PROFILE
Get-Command claude-kimi
```

In both cases, the launcher or function should forward arbitrary extra arguments to `claude`.