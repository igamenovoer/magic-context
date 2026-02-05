<#
.SYNOPSIS
  Install local (client-side) VS Code extensions from a directory of .vsix files.

.DESCRIPTION
  Installs all *.vsix in ExtensionsDir using:
    code --install-extension <vsix> --force

  This is safe to re-run for updates (it forces reinstall of the VSIX you provide).

.PARAMETER ExtensionsDir
  Directory containing .vsix files.

.PARAMETER Channel
  stable, insider, or auto. Default: auto.

.PARAMETER RequiredIds
  Extension IDs that must be present in ExtensionsDir by filename (substring match) OR already installed.
  Default includes ms-vscode-remote.remote-ssh.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ExtensionsDir,

    [Parameter(Mandatory = $false)]
    [ValidateSet("auto", "stable", "insider")]
    [string]$Channel = "auto",

    [Parameter(Mandatory = $false)]
    [string[]]$RequiredIds = @("ms-vscode-remote.remote-ssh")
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $ExtensionsDir)) {
    throw "ExtensionsDir not found: $ExtensionsDir"
}

function Resolve-CodeCommand {
    param([ValidateSet("auto", "stable", "insider")][string]$Channel)
    $stableCmd = Get-Command code -ErrorAction SilentlyContinue
    $insidersCmd = Get-Command code-insiders -ErrorAction SilentlyContinue

    if ($Channel -eq "stable") {
        if (-not $stableCmd) { throw "Requested channel=stable but 'code' is not on PATH." }
        return $stableCmd.Source
    }
    if ($Channel -eq "insider") {
        if (-not $insidersCmd) { throw "Requested channel=insider but 'code-insiders' is not on PATH." }
        return $insidersCmd.Source
    }

    if ($stableCmd) { return $stableCmd.Source }
    if ($insidersCmd) { return $insidersCmd.Source }
    throw "Neither 'code' nor 'code-insiders' found on PATH."
}

$codeCmd = Resolve-CodeCommand -Channel $Channel
$vsix = Get-ChildItem -LiteralPath $ExtensionsDir -Filter *.vsix -File -ErrorAction SilentlyContinue

if (-not $vsix) {
    throw "No .vsix files found in: $ExtensionsDir"
}

Write-Host "==> Installing $($vsix.Count) extension(s) from: $ExtensionsDir" -ForegroundColor Cyan
foreach ($f in $vsix) {
    Write-Host "  - $($f.Name)" -ForegroundColor Yellow
    & $codeCmd --install-extension $f.FullName --force | Out-Null
}

$installed = & $codeCmd --list-extensions --show-versions
Write-Host "==> Installed extensions (local):" -ForegroundColor Green
$installed | ForEach-Object { Write-Host "  $_" }

foreach ($rid in $RequiredIds) {
    if ([string]::IsNullOrWhiteSpace($rid)) { continue }
    if (-not ($installed | Select-String -SimpleMatch $rid)) {
        throw "Required extension not installed: $rid"
    }
}
