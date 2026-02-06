<#
.SYNOPSIS
  Download commit-pinned VS Code client/server/cli artifacts for an air-gapped kit.

.DESCRIPTION
  Downloads from https://update.code.visualstudio.com/ using commit-pinned URLs:
    https://update.code.visualstudio.com/commit:<COMMIT>/<PLATFORM>/<CHANNEL>

  Writes a simple manifest JSON with URLs and SHA256 checksums.

.PARAMETER Commit
  VS Code commit hash (40 hex chars). Must match the client commit used for Remote-SSH.

.PARAMETER Channel
  stable or insider.

.PARAMETER OutDir
  Output directory root for the kit (creates clients/, server/, manifest/).

.PARAMETER ClientPlatforms
  Array of client platform strings (e.g. win32-x64-user, darwin-universal, linux-deb-x64).
  If omitted, no client artifacts are downloaded.

.PARAMETER ServerArch
  Array of Linux server arches to download (x64, arm64). Default: x64.

.PARAMETER Force
  Re-download and overwrite existing files.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Commit,

    [Parameter(Mandatory = $false)]
    [ValidateSet("stable", "insider")]
    [string]$Channel = "stable",

    [Parameter(Mandatory = $true)]
    [string]$OutDir,

    [Parameter(Mandatory = $false)]
    [string[]]$ClientPlatforms = @(),

    [Parameter(Mandatory = $false)]
    [ValidateSet("x64", "arm64")]
    [string[]]$ServerArch = @("x64"),

    [Parameter(Mandatory = $false)]
    [switch]$Force
)

$ErrorActionPreference = "Stop"

if ($Commit -notmatch "^[0-9a-f]{40}$") {
    throw "Commit must be 40 hex chars: $Commit"
}

function Ensure-Dir {
    param([Parameter(Mandatory = $true)][string]$Path)
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Download-UrlToFile {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$OutDir,
        [Parameter(Mandatory = $true)][string]$FallbackFileName,
        [Parameter(Mandatory = $true)][switch]$Force
    )

    Ensure-Dir $OutDir
    $tmp = Join-Path $OutDir "$FallbackFileName.tmp"
    if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Force }

    $resp = Invoke-WebRequest -Uri $Url -OutFile $tmp -UseBasicParsing -ErrorAction Stop

    $contentDisposition = $null
    try { $contentDisposition = $resp.Headers["Content-Disposition"] } catch { $contentDisposition = $null }

    $fileName = $null
    if ($contentDisposition) {
        if ($contentDisposition -match 'filename\*=UTF-8''''([^;]+)') {
            $fileName = [System.Uri]::UnescapeDataString($Matches[1])
        }
        elseif ($contentDisposition -match 'filename="?([^";]+)"?') {
            $fileName = $Matches[1]
        }
    }

    if ([string]::IsNullOrWhiteSpace($fileName)) {
        $fileName = $FallbackFileName
    }

    $outFile = Join-Path $OutDir $fileName

    if ((-not $Force) -and (Test-Path -LiteralPath $outFile)) {
        Remove-Item -LiteralPath $tmp -Force
        return @{ Downloaded = $false; Path = $outFile; Url = $Url }
    }

    Move-Item -LiteralPath $tmp -Destination $outFile -Force
    return @{ Downloaded = $true; Path = $outFile; Url = $Url }
}

Ensure-Dir $OutDir
$OutDir = (Resolve-Path $OutDir).Path

$clientsDir = Join-Path $OutDir "clients"
$serverDir = Join-Path $OutDir "server"
$manifestDir = Join-Path $OutDir "manifest"

Ensure-Dir $clientsDir
Ensure-Dir $serverDir
Ensure-Dir $manifestDir

$manifest = [ordered]@{
    created_at = (Get-Date).ToString("s")
    channel    = $Channel
    commit     = $Commit
    clients    = @()
    server     = @()
}

Write-Host "==> OutDir  : $OutDir" -ForegroundColor Cyan
Write-Host "==> Channel : $Channel" -ForegroundColor Cyan
Write-Host "==> Commit  : $Commit" -ForegroundColor Cyan

foreach ($platform in $ClientPlatforms) {
    if ([string]::IsNullOrWhiteSpace($platform)) { continue }
    $plat = $platform.Trim()
    $url = "https://update.code.visualstudio.com/commit:$Commit/$plat/$Channel"

    $osFolder = "misc"
    if ($plat.StartsWith("win32-")) { $osFolder = "windows" }
    elseif ($plat.StartsWith("darwin-")) { $osFolder = "macos" }
    elseif ($plat.StartsWith("linux-")) { $osFolder = "linux" }

    $outSub = Join-Path $clientsDir $osFolder
    Ensure-Dir $outSub

    Write-Host "==> Download client: $plat" -ForegroundColor Yellow
    $dl = Download-UrlToFile -Url $url -OutDir $outSub -FallbackFileName "vscode-$plat-$Commit.bin" -Force:$Force

    $manifest.clients += [ordered]@{
        platform = $plat
        url      = $url
        file     = (Resolve-Path $dl.Path).Path
        sha256   = Get-Sha256 -Path $dl.Path
    }
}

foreach ($arch in $ServerArch) {
    $srvUrl = "https://update.code.visualstudio.com/commit:$Commit/server-linux-$arch/$Channel"
    $cliUrl = "https://update.code.visualstudio.com/commit:$Commit/cli-alpine-$arch/$Channel"

    $srvOutDir = Join-Path $serverDir "linux-$arch"
    $cliOutDir = Join-Path $serverDir "cli"
    Ensure-Dir $srvOutDir
    Ensure-Dir $cliOutDir

    $srvOut = Join-Path $srvOutDir "vscode-server-linux-$arch-$Commit.tar.gz"
    $cliOut = Join-Path $cliOutDir "vscode-cli-alpine-$arch-$Commit.tar.gz"

    Write-Host "==> Download server: linux-$arch" -ForegroundColor Yellow
    $dlSrv = Download-UrlToFile -Url $srvUrl -OutDir $srvOutDir -FallbackFileName (Split-Path -Leaf $srvOut) -Force:$Force
    $manifest.server += [ordered]@{
        platform = "server-linux-$arch"
        url      = $srvUrl
        file     = (Resolve-Path $dlSrv.Path).Path
        sha256   = Get-Sha256 -Path $dlSrv.Path
    }

    Write-Host "==> Download cli   : alpine-$arch" -ForegroundColor Yellow
    $dlCli = Download-UrlToFile -Url $cliUrl -OutDir $cliOutDir -FallbackFileName (Split-Path -Leaf $cliOut) -Force:$Force
    $manifest.server += [ordered]@{
        platform = "cli-alpine-$arch"
        url      = $cliUrl
        file     = (Resolve-Path $dlCli.Path).Path
        sha256   = Get-Sha256 -Path $dlCli.Path
    }
}

$manifestPath = Join-Path $manifestDir "vscode.json"
$manifest | ConvertTo-Json -Depth 10 | Out-File -FilePath $manifestPath -Encoding UTF8
Write-Host "==> Manifest: $manifestPath" -ForegroundColor Green
