<#
.SYNOPSIS
  Cleanup optional files on an air-gapped client after installation/update.

.DESCRIPTION
  This script deletes selected package artifacts from a package directory.
  Use it only after verifying VS Code + extensions are working.

.PARAMETER PackageDir
  Directory containing offline installers/archives and VSIX files.

.PARAMETER RemoveInstallers
  Remove common VS Code installer/archive files (*.exe, *.msi, *.zip, *.dmg, *.deb, *.rpm, *.tar.gz).

.PARAMETER RemoveVsix
  Remove *.vsix files.
#>

[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)]
    [string]$PackageDir,

    [Parameter(Mandatory = $false)]
    [switch]$RemoveInstallers,

    [Parameter(Mandatory = $false)]
    [switch]$RemoveVsix
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $PackageDir)) {
    throw "PackageDir not found: $PackageDir"
}

$patterns = @()
if ($RemoveInstallers) {
    $patterns += @("*.exe", "*.msi", "*.zip", "*.dmg", "*.deb", "*.rpm", "*.tar.gz")
}
if ($RemoveVsix) {
    $patterns += @("*.vsix")
}

if ($patterns.Count -eq 0) {
    throw "Nothing selected. Use -RemoveInstallers and/or -RemoveVsix."
}

foreach ($pat in $patterns) {
    Get-ChildItem -LiteralPath $PackageDir -Recurse -File -Filter $pat -ErrorAction SilentlyContinue | ForEach-Object {
        if ($PSCmdlet.ShouldProcess($_.FullName, "Remove")) {
            Remove-Item -LiteralPath $_.FullName -Force
        }
    }
}

Write-Host "==> Cleanup complete in: $PackageDir" -ForegroundColor Green

