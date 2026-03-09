# Installation

Use this subskill when Claude Code CLI is not yet installed or when the current install needs to be refreshed.

## Goal

Install the `claude` CLI in a way that matches the real host environment.

## Environment-first workflow

1. Detect the OS.
2. Detect the available package managers.
3. Check whether `claude` is already on `PATH`.
4. If a bundled helper script clearly matches the environment, it can be used.
5. If the helper script fails or is not a good fit, perform the installation manually.

## Linux and macOS

Check available tools:

```bash
uname -s
command -v claude || true
command -v bun || true
command -v node || true
command -v npm || true
```

Preferred manual install with Bun when available:

```bash
bun add -g @anthropic-ai/claude-code --registry https://registry.npmmirror.com
```

Manual install with npm:

```bash
npm install -g @anthropic-ai/claude-code --registry https://registry.npmmirror.com
```

If the mirror fails, retry with the official registry:

```bash
npm install -g @anthropic-ai/claude-code --registry https://registry.npmjs.org
```

If the global package bin path is not already visible, inspect it and add it to `PATH` for the current session before verifying:

```bash
npm bin -g
export PATH="$(npm bin -g):$PATH"
command -v claude
claude --version
```

Optional helper scripts:

```bash
sh scripts/install-comp.sh
sh scripts/install-comp.sh --from-official
```

## Windows

Check available tools:

```powershell
$PSVersionTable.PSVersion
Get-Command claude -ErrorAction SilentlyContinue
Get-Command bun -ErrorAction SilentlyContinue
Get-Command node -ErrorAction SilentlyContinue
Get-Command npm -ErrorAction SilentlyContinue
```

Preferred manual install with Bun when available:

```powershell
bun add -g @anthropic-ai/claude-code --registry https://registry.npmmirror.com
```

Manual install with npm:

```powershell
npm install -g @anthropic-ai/claude-code --registry https://registry.npmmirror.com
```

If the mirror fails, retry with the official registry:

```powershell
npm install -g @anthropic-ai/claude-code --registry https://registry.npmjs.org
```

If `claude` is not found after install, inspect the global npm prefix and restart the shell if needed:

```powershell
npm prefix -g
Get-Command claude
claude --version
```

Optional helper scripts:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install-comp.ps1
```

Or:

```bat
scripts\install-comp.bat
```

## Verification

- `claude` resolves on `PATH`
- `claude --version` succeeds
- the install command completed without package-manager errors