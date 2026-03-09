# Installation

Use this subskill when Codex CLI is not yet installed or when the current install needs to be refreshed.

## Goal

Install the `codex` CLI in a way that matches the real host environment.

## Environment-first workflow

1. Detect the OS.
2. Detect the available package managers.
3. Check whether `codex` is already on `PATH`.
4. If a bundled helper script clearly matches the environment, it can be used.
5. If the helper script fails or is not a good fit, perform the installation manually.

## Linux and macOS

Check available tools:

```bash
uname -s
command -v codex || true
command -v bun || true
command -v node || true
command -v npm || true
```

Preferred manual install with Bun when available:

```bash
bun add -g @openai/codex --registry https://registry.npmmirror.com
```

Manual install with npm:

```bash
npm install -g @openai/codex --registry https://registry.npmmirror.com
```

If the mirror fails, retry with the official registry:

```bash
npm install -g @openai/codex --registry https://registry.npmjs.org
```

If the global package bin path is not already visible, inspect it and add it to `PATH` for the current session before verifying:

```bash
npm bin -g
export PATH="$(npm bin -g):$PATH"
command -v codex
codex --version
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
Get-Command codex -ErrorAction SilentlyContinue
Get-Command bun -ErrorAction SilentlyContinue
Get-Command node -ErrorAction SilentlyContinue
Get-Command npm -ErrorAction SilentlyContinue
```

Preferred manual install with Bun when available:

```powershell
bun add -g @openai/codex --registry https://registry.npmmirror.com
```

Manual install with npm:

```powershell
npm install -g @openai/codex --registry https://registry.npmmirror.com
```

If the mirror fails, retry with the official registry:

```powershell
npm install -g @openai/codex --registry https://registry.npmjs.org
```

If `codex` is not found after install, inspect the global npm prefix and restart the shell if needed:

```powershell
npm prefix -g
Get-Command codex
codex --version
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

- `codex` resolves on `PATH`
- `codex --version` succeeds
- the install command completed without package-manager errors