<#
.SYNOPSIS
  Configure VS Code client settings for air-gapped Remote-SSH use.

.DESCRIPTION
  Writes/merges keys into the user's settings.json:
    "update.mode": "manual"
    "extensions.autoCheckUpdates": false
    "extensions.autoUpdate": false
    "remote.SSH.localServerDownload": "off"

  Safe to re-run after updates.

.PARAMETER Channel
  stable, insider, or auto. Default: auto (prefer `code`, else `code-insiders`)

.PARAMETER SettingsPath
  Optional override path to settings.json. If set, Channel is ignored.

.PARAMETER RemoteSshLocalServerDownload
  off|always. Default: off (recommended when server has pre-placed cache files).
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [ValidateSet("auto", "stable", "insider")]
    [string]$Channel = "auto",

    [Parameter(Mandatory = $false)]
    [string]$SettingsPath = $null,

    [Parameter(Mandatory = $false)]
    [ValidateSet("off", "always")]
    [string]$RemoteSshLocalServerDownload = "off"
)

$ErrorActionPreference = "Stop"

function Resolve-CodeChannel {
    param([ValidateSet("auto", "stable", "insider")][string]$Channel)

    $stableCmd = Get-Command code -ErrorAction SilentlyContinue
    $insidersCmd = Get-Command code-insiders -ErrorAction SilentlyContinue

    if ($Channel -eq "stable") {
        if (-not $stableCmd) { throw "Requested channel=stable but 'code' is not on PATH." }
        return "stable"
    }

    if ($Channel -eq "insider") {
        if (-not $insidersCmd) { throw "Requested channel=insider but 'code-insiders' is not on PATH." }
        return "insider"
    }

    if ($stableCmd) { return "stable" }
    if ($insidersCmd) { return "insider" }

    throw "Neither 'code' nor 'code-insiders' found on PATH."
}

function Get-DefaultSettingsPath {
    param([ValidateSet("stable", "insider")][string]$ChannelResolved)

    $home = $HOME
    if ($IsWindows) {
        $appData = $env:APPDATA
        if ([string]::IsNullOrWhiteSpace($appData)) { throw "APPDATA is not set; cannot locate VS Code settings.json." }
        if ($ChannelResolved -eq "stable") { return (Join-Path $appData "Code\\User\\settings.json") }
        return (Join-Path $appData "Code - Insiders\\User\\settings.json")
    }

    if ($IsMacOS) {
        if ($ChannelResolved -eq "stable") { return (Join-Path $home "Library/Application Support/Code/User/settings.json") }
        return (Join-Path $home "Library/Application Support/Code - Insiders/User/settings.json")
    }

    # Linux
    if ($ChannelResolved -eq "stable") { return (Join-Path $home ".config/Code/User/settings.json") }
    return (Join-Path $home ".config/Code - Insiders/User/settings.json")
}

function Read-JsonObjectOrEmpty {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) { return @{} }

    $raw = Get-Content -LiteralPath $Path -Raw -ErrorAction Stop
    if ([string]::IsNullOrWhiteSpace($raw)) { return @{} }

    try {
        $obj = $raw | ConvertFrom-Json -ErrorAction Stop
        if ($null -eq $obj) { return @{} }
        if ($obj -isnot [System.Collections.IDictionary]) {
            return ($obj | ConvertTo-Json -Depth 50 | ConvertFrom-Json -AsHashtable)
        }
        return $obj
    }
    catch {
        throw "Failed to parse JSON at $Path. Fix/rename it, then re-run. Error: $($_.Exception.Message)"
    }
}

$targetSettingsPath = $SettingsPath
if ([string]::IsNullOrWhiteSpace($targetSettingsPath)) {
    $resolvedChannel = Resolve-CodeChannel -Channel $Channel
    $targetSettingsPath = Get-DefaultSettingsPath -ChannelResolved $resolvedChannel
}

$dir = Split-Path -Parent $targetSettingsPath
New-Item -ItemType Directory -Path $dir -Force | Out-Null

$settings = Read-JsonObjectOrEmpty -Path $targetSettingsPath

$settings["update.mode"] = "manual"
$settings["extensions.autoCheckUpdates"] = $false
$settings["extensions.autoUpdate"] = $false
$settings["remote.SSH.localServerDownload"] = $RemoteSshLocalServerDownload

$json = $settings | ConvertTo-Json -Depth 50
$json | Out-File -FilePath $targetSettingsPath -Encoding UTF8

Write-Host "==> Updated: $targetSettingsPath" -ForegroundColor Green
Write-Host "    update.mode = manual" -ForegroundColor Green
Write-Host "    extensions.autoCheckUpdates = false" -ForegroundColor Green
Write-Host "    extensions.autoUpdate = false" -ForegroundColor Green
Write-Host "    remote.SSH.localServerDownload = $RemoteSshLocalServerDownload" -ForegroundColor Green

