<#
.SYNOPSIS
  Discover the VS Code release currently installed on this host (channel/version/commit) and export the local extension list.

.DESCRIPTION
  For air-gapped Remote-SSH, the VS Code Server must match the client COMMIT (line 2 of `code --version`).
  This script discovers:
    - channel: stable or insider
    - version
    - commit
    - arch (best-effort, from `code --version` line 3)
  And exports:
    - extensions.local.txt (id@version)
    - vscode.local.json (metadata)

.PARAMETER OutDir
  Output directory for exported files. Default: ./manifest

.PARAMETER Channel
  stable, insider, or auto. Default: auto (prefer `code` if present, else `code-insiders`)
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$OutDir = (Join-Path (Get-Location) "manifest"),

    [Parameter(Mandatory = $false)]
    [ValidateSet("auto", "stable", "insider")]
    [string]$Channel = "auto"
)

$ErrorActionPreference = "Stop"

function Resolve-CodeCommand {
    param(
        [ValidateSet("auto", "stable", "insider")]
        [string]$Channel
    )

    $stableCmd = Get-Command code -ErrorAction SilentlyContinue
    $insidersCmd = Get-Command code-insiders -ErrorAction SilentlyContinue

    if ($Channel -eq "stable") {
        if (-not $stableCmd) { throw "Requested channel=stable but 'code' is not on PATH." }
        return @{ Channel = "stable"; Command = $stableCmd.Source }
    }

    if ($Channel -eq "insider") {
        if (-not $insidersCmd) { throw "Requested channel=insider but 'code-insiders' is not on PATH." }
        return @{ Channel = "insider"; Command = $insidersCmd.Source }
    }

    if ($stableCmd) { return @{ Channel = "stable"; Command = $stableCmd.Source } }
    if ($insidersCmd) { return @{ Channel = "insider"; Command = $insidersCmd.Source } }

    throw "Neither 'code' nor 'code-insiders' found on PATH. Install VS Code or add it to PATH."
}

function Parse-CodeVersionOutput {
    param([string[]]$Lines)

    if (-not $Lines -or $Lines.Count -lt 2) {
        throw "Unexpected 'code --version' output."
    }

    $version = $Lines[0].Trim()
    $commit = $Lines[1].Trim()
    $arch = $null
    if ($Lines.Count -ge 3) { $arch = $Lines[2].Trim() }

    if ($commit -notmatch "^[0-9a-f]{40}$") {
        throw "Commit hash did not look like 40-hex: '$commit'"
    }

    return @{ Version = $version; Commit = $commit; Arch = $arch }
}

$resolved = Resolve-CodeCommand -Channel $Channel
$channelResolved = $resolved.Channel
$codeCmd = $resolved.Command

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
$OutDir = (Resolve-Path $OutDir).Path

Write-Host "==> Using VS Code command: $codeCmd" -ForegroundColor Cyan
Write-Host "==> Channel: $channelResolved" -ForegroundColor Cyan
Write-Host "==> Output dir: $OutDir" -ForegroundColor Cyan

$versionLines = & $codeCmd --version 2>$null
$parsed = Parse-CodeVersionOutput -Lines $versionLines

$extensionsPath = Join-Path $OutDir "extensions.local.txt"
$vscodeJsonPath = Join-Path $OutDir "vscode.local.json"

Write-Host "==> Exporting local extensions..." -ForegroundColor Yellow
$extensions = & $codeCmd --list-extensions --show-versions 2>$null
$extensions | Out-File -FilePath $extensionsPath -Encoding UTF8

$payload = [ordered]@{
    discovered_at = (Get-Date).ToString("s")
    channel       = $channelResolved
    version       = $parsed.Version
    commit        = $parsed.Commit
    arch          = $parsed.Arch
    command       = $codeCmd
    outputs       = @{
        extensions_local = (Split-Path -Leaf $extensionsPath)
    }
}

$payload | ConvertTo-Json -Depth 6 | Out-File -FilePath $vscodeJsonPath -Encoding UTF8

Write-Host "==> VERSION: $($parsed.Version)" -ForegroundColor Green
Write-Host "==> COMMIT : $($parsed.Commit)" -ForegroundColor Green
if ($parsed.Arch) { Write-Host "==> ARCH   : $($parsed.Arch)" -ForegroundColor Green }
Write-Host "==> Wrote  : $extensionsPath" -ForegroundColor Green
Write-Host "==> Wrote  : $vscodeJsonPath" -ForegroundColor Green
