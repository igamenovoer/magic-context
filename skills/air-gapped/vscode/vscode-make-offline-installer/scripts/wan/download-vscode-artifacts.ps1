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
  If omitted, no client artifacts are downloaded. Artifacts are saved under clients/<os>-<arch>/ by default.

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

    $aria2 = Get-Command aria2c -ErrorAction SilentlyContinue

    $fileName = $FallbackFileName
    $contentLength = $null

    try {
        $head = Invoke-WebRequest -Uri $Url -Method Head -MaximumRedirection 10 -ErrorAction Stop
        $contentDisposition = $null
        try { $contentDisposition = $head.Headers["Content-Disposition"] } catch { $contentDisposition = $null }

        if ($contentDisposition) {
            if ($contentDisposition -match 'filename\*=UTF-8''''([^;]+)') {
                $fileName = [System.Uri]::UnescapeDataString($Matches[1])
            }
            elseif ($contentDisposition -match 'filename="?([^";]+)"?') {
                $fileName = $Matches[1]
            }
        }

        try {
            $len = $head.Headers["Content-Length"]
            if (-not [string]::IsNullOrWhiteSpace($len)) {
                $contentLength = [int64]$len
            }
        }
        catch {
            $contentLength = $null
        }
    }
    catch {
        # ignore HEAD failures; fall back to the provided filename
    }

    if ([string]::IsNullOrWhiteSpace($fileName)) {
        $fileName = $FallbackFileName
    }

    $outFile = Join-Path $OutDir $fileName

    if ((-not $Force) -and (Test-Path -LiteralPath $outFile)) {
        if ($contentLength) {
            $localLen = (Get-Item -LiteralPath $outFile).Length
            if ($localLen -eq $contentLength) {
                return @{ Downloaded = $false; Path = $outFile; Url = $Url }
            }
        }
    }

    if (-not $aria2) {
        $tmp = Join-Path $OutDir "$fileName.tmp"
        if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue }
        Invoke-WebRequest -Uri $Url -OutFile $tmp -ErrorAction Stop | Out-Null
        Move-Item -LiteralPath $tmp -Destination $outFile -Force
        return @{ Downloaded = $true; Path = $outFile; Url = $Url }
    }

    $partFile = "$outFile.part"
    if ((Test-Path -LiteralPath $outFile) -and (-not (Test-Path -LiteralPath $partFile))) {
        try { Move-Item -LiteralPath $outFile -Destination $partFile -Force } catch { }
    }

    $outName = [System.IO.Path]::GetFileName($partFile)
    $dirName = [System.IO.Path]::GetDirectoryName($partFile)

    $args = @(
        "--continue=true",
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "--check-integrity=true",
        "--file-allocation=none",
        "--max-tries=10",
        "--retry-wait=5",
        "--timeout=60",
        "--max-connection-per-server=4",
        "--split=4",
        "--min-split-size=1M",
        "--summary-interval=0",
        "--dir=$dirName",
        "--out=$outName",
        $Url
    )

    & $aria2.Source @args | Out-Null

    if (-not (Test-Path -LiteralPath $partFile)) {
        throw "Download failed (file missing): $partFile"
    }

    if ($contentLength) {
        $localLen = (Get-Item -LiteralPath $partFile).Length
        if ($localLen -ne $contentLength) {
            throw "Download size mismatch for $fileName (expected $contentLength bytes, got $localLen)"
        }
    }

    if (Test-Path -LiteralPath $outFile) {
        Remove-Item -LiteralPath $outFile -Force -ErrorAction SilentlyContinue
    }

    Move-Item -LiteralPath $partFile -Destination $outFile -Force
    if (Test-Path -LiteralPath "$partFile.aria2") {
        Remove-Item -LiteralPath "$partFile.aria2" -Force -ErrorAction SilentlyContinue
    }

    return @{ Downloaded = $true; Path = $outFile; Url = $Url }
}

function Get-ClientBinaryDirName {
    param([Parameter(Mandatory = $true)][string]$Platform)
    $p = $Platform.Trim().ToLowerInvariant()
    if ([string]::IsNullOrWhiteSpace($p)) { return $null }

    $parts = $p.Split('-', [System.StringSplitOptions]::RemoveEmptyEntries)
    if (-not $parts -or $parts.Count -lt 2) { return $p }

    $os = $parts[0]

    if ($os -eq "win32") {
        $arch = $parts[1]
        if ($arch -in @("x64", "arm64")) { return "win32-$arch" }
        return $p
    }

    if ($os -eq "darwin") {
        $arch = $parts[1]
        if ($arch -in @("universal", "arm64", "x64")) { return "darwin-$arch" }
        return $p
    }

    if ($os -eq "linux") {
        $arch = $parts[$parts.Count - 1]
        if ($arch -in @("x64", "arm64")) { return "linux-$arch" }
        return $p
    }

    return $p
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

    $outSubName = Get-ClientBinaryDirName -Platform $plat
    if ([string]::IsNullOrWhiteSpace($outSubName)) { $outSubName = $plat }
    $outSub = Join-Path $clientsDir $outSubName
    Ensure-Dir $outSub

    Write-Host "==> Download client: $plat -> clients\\$outSubName" -ForegroundColor Yellow
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
    $cliOutDir = Join-Path $serverDir "alpine-$arch"
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
