# Skip Login Config

Use this subskill when Claude Code is installed but the host should bypass local first-run onboarding/login prompts.

## Goal

Set the local Claude configuration field `hasCompletedOnboarding` to `true` in the user config file.

## Environment-first workflow

1. Confirm `claude` is already installed.
2. Locate the per-user Claude config file for the current OS.
3. Update the JSON safely.
4. Verify the field is present and true.

## Linux and macOS

Check prerequisites:

```bash
command -v claude
command -v node || true
test -f "$HOME/.claude.json" && cat "$HOME/.claude.json" || true
```

Safe update with Node.js:

```bash
node - "$HOME/.claude.json" <<'NODE'
const fs = require('fs');
const target = process.argv[2];
let data = {};
try {
  if (fs.existsSync(target)) {
    const text = fs.readFileSync(target, 'utf8').trim();
    if (text) data = JSON.parse(text);
  }
} catch {
  data = {};
}
data.hasCompletedOnboarding = true;
fs.writeFileSync(target, JSON.stringify(data, null, 2) + '\n', 'utf8');
NODE
```

Optional helper script:

```bash
sh scripts/config-skip-login.sh
```

## Windows

Inspect the current file first:

```powershell
Get-Command claude
$configFile = Join-Path $env:USERPROFILE '.claude.json'
if (Test-Path $configFile) { Get-Content $configFile -Raw }
```

Safe update with PowerShell:

```powershell
$configFile = Join-Path $env:USERPROFILE '.claude.json'
$config = @{}
if (Test-Path $configFile) {
    try {
        $content = Get-Content $configFile -Raw
        if ($content.Trim().Length -gt 0) {
            $config = $content | ConvertFrom-Json -AsHashtable
        }
    } catch {
        $config = @{}
    }
}
$config.hasCompletedOnboarding = $true
$json = $config | ConvertTo-Json -Depth 10
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($configFile, $json, $utf8NoBom)
```

Optional helper script:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/config-skip-login.ps1
```

Or:

```bat
scripts\config-skip-login.bat
```

## Verification

Linux/macOS:

```bash
grep -n 'hasCompletedOnboarding' "$HOME/.claude.json"
```

Windows:

```powershell
Get-Content (Join-Path $env:USERPROFILE '.claude.json')
```

The resulting JSON should contain `"hasCompletedOnboarding": true`.