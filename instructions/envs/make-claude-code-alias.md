you are tasked to help the user create Claude CLI command aliases (wrappers) that pin provider settings and optionally pin a model + max reasoning effort

# Guidelines for Creating Claude CLI Aliases

## Safety and placeholders

- do NOT include real command names from the local machine in this template
- do NOT paste real API keys; use `<API_KEY_VALUE>` or reference an existing secret source the user already has
- do NOT commit secrets into the repo; aliases/wrappers should live in user shell profiles or user home directories

## Inputs (placeholders)

- `<BASE_CMD_NAME>`: provider-aware wrapper command name (non-model-pinned)
- `<MODEL_CMD_NAME>`: model-pinned wrapper command name (calls `<BASE_CMD_NAME>`)
- `<API_BASE_URL>`: Anthropic-compatible base URL (official or custom)
- `<API_KEY_VALUE>`: provider API key value used as `ANTHROPIC_AUTH_TOKEN` (or an env var / secret file the user already uses)
- `<MODEL_NAME>`: target model name (default: `opus`)
- `<OTHER_SUFFIX>`: purpose/role suffix (short kebab-case token)
- `<PROFILE_FILE>`: chosen shell profile path

## Example: Kimi (Moonshot) via Anthropic-compatible API

Moonshot (Kimi) provides an Anthropic Messages API compatible endpoint.

Example base URLs:

- `https://api.moonshot.ai/anthropic`
- `https://api.moonshot.cn/anthropic`

Example tokens:

- `<provider>`: `moonshot` (from `api.moonshot.ai`)
- `<MODEL_NAME>`: `kimi-k2-turbo-preview` (example; pick a supported model id)
- `<OTHER_SUFFIX>`: `coding`

Example command names:

- `<BASE_CMD_NAME>`: `claude-moonshot-coding`
- `<MODEL_CMD_NAME>`: `claude-moonshot-kimi-k2-turbo-preview-coding`

Optional (when you need to pin both the default model and the fast model in Claude Code):

- set `ANTHROPIC_MODEL=<MODEL_NAME>` and `ANTHROPIC_SMALL_FAST_MODEL=<MODEL_NAME>` in the model-pinned alias/wrapper


## Command naming convention

Model-pinned alias commands follow:

- `claude-<provider>-<model-name>-<other-suffix>`

Provider-aware (non-model-pinned) commands should follow:

- `claude-<provider>-<other-suffix>`

How to fill `<provider>`:

- derive it from the host of `<API_BASE_URL>` when the host is a custom domain (usually the registrable domain label, not `api`)
- if `<API_BASE_URL>` is the official Anthropic endpoint (for example host `api.anthropic.com`), omit `<provider>`

Normalization rules:

- provider/model/suffix tokens must be lowercase kebab-case
- if provider is omitted, use `claude-<model-name>-<other-suffix>` (avoid double hyphens)
- if provider is omitted for the provider-aware command, use `claude-<other-suffix>` (avoid double hyphens)
- do not override the upstream `claude` command name unless the user explicitly requests it

## Decide the setup mode (default preference)

1. prefer editing the user shell profile (functions/aliases) by default
2. only create standalone wrapper files + adjust PATH if the user explicitly asks for that pattern, or provides an example that implies file-based wrappers
3. if the user requests **"edit shell profile only"**, do exactly that (no wrapper files, no PATH edits)

## Pick the correct shell profile file

Choose based on current shell:

- `zsh` -> `~/.zshrc`
- `bash` -> `~/.bashrc` (fallback `~/.profile` if `.bashrc` is missing)
- `powershell` -> `$PROFILE`

If shell type is unclear, ask a single clarifying question or use the most likely profile file for the OS/shell family.

## Default implementation: add profile functions (idempotent)

Before adding anything:

- check if `<BASE_CMD_NAME>` or `<MODEL_CMD_NAME>` already exists in `<PROFILE_FILE>`
- if present, update in place or skip; do not duplicate entries

### POSIX (bash/zsh) profile functions

Use:

- `<BASE_CMD_NAME>`: a provider-aware wrapper
- `<MODEL_CMD_NAME>`: a model-pinned wrapper that forces max reasoning effort (`--effort high`)

Template:

```sh
<BASE_CMD_NAME>() {
  ANTHROPIC_BASE_URL='<API_BASE_URL>' \
  ANTHROPIC_AUTH_TOKEN='<API_KEY_VALUE>' \
  ANTHROPIC_API_KEY='' \
    claude --dangerously-skip-permissions "$@"
}

<MODEL_CMD_NAME>() {
  <BASE_CMD_NAME> "$@" --model <MODEL_NAME> --effort high
}
```

Notes:

- always forward args with `"$@"`
- `--dangerously-skip-permissions` is powerful; only use it in trusted environments

### PowerShell profile functions

Template:

```powershell
function <BASE_CMD_NAME> {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Args)
    $env:ANTHROPIC_BASE_URL = "<API_BASE_URL>"
    $env:ANTHROPIC_AUTH_TOKEN = "<API_KEY_VALUE>"
    $env:ANTHROPIC_API_KEY = ""
    & claude --dangerously-skip-permissions @Args
}

function <MODEL_CMD_NAME> {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Args)
    <BASE_CMD_NAME> @Args --model <MODEL_NAME> --effort high
}
```

## Optional implementation: wrapper files + PATH (only if requested)

Use this mode only when explicitly requested by the user.

Default wrapper directory placeholder:

- `<WRAPPER_DIR>`: `$HOME/.local/bin`

### Wrapper file templates (POSIX)

`<WRAPPER_DIR>/<BASE_CMD_NAME>`:

```sh
#!/usr/bin/env sh
set -eu
export ANTHROPIC_BASE_URL='<API_BASE_URL>'
export ANTHROPIC_AUTH_TOKEN='<API_KEY_VALUE>'
export ANTHROPIC_API_KEY=''
exec claude --dangerously-skip-permissions "$@"
```

`<WRAPPER_DIR>/<MODEL_CMD_NAME>`:

```sh
#!/usr/bin/env sh
set -eu
exec <BASE_CMD_NAME> "$@" --model <MODEL_NAME> --effort high
```

### PATH updates must be idempotent

Only add `<WRAPPER_DIR>` to PATH if it is missing, and do not duplicate existing logic.

POSIX profile snippet:

```sh
case ":$PATH:" in
  *":<WRAPPER_DIR>:"*) ;;
  *) export PATH="<WRAPPER_DIR>:$PATH" ;;
esac
```

PowerShell profile snippet:

```powershell
if (-not (($env:PATH -split ';') -contains "<WRAPPER_DIR>")) {
    $env:PATH = "<WRAPPER_DIR>;$env:PATH"
}
```

## Verification checklist

- reload the shell profile (or open a new terminal)
- `command -v <BASE_CMD_NAME>` / `command -v <MODEL_CMD_NAME>` (PowerShell: `Get-Command ...`)
- `<MODEL_CMD_NAME> -p "Reply exactly: OK" --output-format json`
- confirm the run uses `<MODEL_NAME>` and `--effort high` for that call
