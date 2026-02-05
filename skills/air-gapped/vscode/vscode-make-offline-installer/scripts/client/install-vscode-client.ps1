<#
.SYNOPSIS
  Install VS Code on an air-gapped desktop client using pre-downloaded installers/archives.

.DESCRIPTION
  This script is intentionally conservative:
  - On Windows: launches the provided installer and waits (interactive by default).
  - On macOS/Linux: validates inputs and prints actionable next steps if automatic install is not supported.

  It supports re-running for updates: if VS Code is already installed, it reports the current version/commit
  and still allows launching the provided installer.

.PARAMETER InstallerPath
  Path to the offline VS Code installer/archive appropriate for the client OS.

.PARAMETER Channel
  stable, insider, or auto (used only for checking current install). Default: auto.

.PARAMETER Silent
  Attempt a silent install on Windows installers (best-effort). Default: off (interactive).
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$InstallerPath,

    [Parameter(Mandatory = $false)]
    [ValidateSet("auto", "stable", "insider")]
    [string]$Channel = "auto",

    [Parameter(Mandatory = $false)]
    [switch]$Silent
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $InstallerPath)) {
    throw "InstallerPath not found: $InstallerPath"
}

function Resolve-CodeCommand {
    param([ValidateSet("auto", "stable", "insider")][string]$Channel)
    $stableCmd = Get-Command code -ErrorAction SilentlyContinue
    $insidersCmd = Get-Command code-insiders -ErrorAction SilentlyContinue

    if ($Channel -eq "stable") { return $stableCmd }
    if ($Channel -eq "insider") { return $insidersCmd }
    if ($stableCmd) { return $stableCmd }
    if ($insidersCmd) { return $insidersCmd }
    return $null
}

$codeCmd = Resolve-CodeCommand -Channel $Channel
if ($codeCmd) {
    try {
        $lines = & $codeCmd.Source --version 2>$null
        if ($lines -and $lines.Count -ge 2) {
            Write-Host "==> Current VS Code:" -ForegroundColor Cyan
            Write-Host "    Version: $($lines[0])" -ForegroundColor Cyan
            Write-Host "    Commit : $($lines[1])" -ForegroundColor Cyan
        }
    }
    catch {
        # ignore
    }
}
else {
    Write-Host "==> VS Code not found on PATH (may not be installed yet)." -ForegroundColor Yellow
}

if ($IsWindows) {
    $ext = [IO.Path]::GetExtension($InstallerPath).ToLowerInvariant()
    if ($ext -ne ".exe" -and $ext -ne ".msi") {
        throw "On Windows, expected an .exe or .msi installer. Got: $InstallerPath"
    }

    Write-Host "==> Installing VS Code on Windows..." -ForegroundColor Yellow
    if ($Silent) {
        Write-Host "==> Attempting silent install (best-effort)..." -ForegroundColor Yellow
        if ($ext -eq ".msi") {
            Start-Process -FilePath "msiexec.exe" -ArgumentList @("/i", "`"$InstallerPath`"", "/qn", "/norestart") -Wait
        }
        else {
            # VSCodeUserSetup is typically an Inno Setup installer. These flags are common but not guaranteed.
            Start-Process -FilePath $InstallerPath -ArgumentList @("/VERYSILENT", "/NORESTART") -Wait
        }
    }
    else {
        Start-Process -FilePath $InstallerPath -Wait
    }

    Write-Host "==> Done (installer completed)." -ForegroundColor Green
    exit 0
}

Write-Host "==> This installer script currently supports automatic install on Windows only." -ForegroundColor Yellow
Write-Host "==> For macOS/Linux, use the appropriate offline package method for your artifact (zip/dmg/deb/rpm/tar.gz)." -ForegroundColor Yellow
Write-Host "    InstallerPath: $InstallerPath" -ForegroundColor Yellow
