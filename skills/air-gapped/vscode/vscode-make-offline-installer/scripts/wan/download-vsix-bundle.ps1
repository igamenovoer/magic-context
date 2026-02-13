<#
.SYNOPSIS
  Download pinned VS Code extensions as .vsix files for offline use.

.DESCRIPTION
  Given a file containing lines like:
    publisher.extension@1.2.3
  this script downloads each VSIX using:
    1) Open VSX (via ovsx CLI if available, else Open VSX HTTP)
    2) Marketplace vspackage (version-pinned)
    3) Skip if not found

  It writes a download report JSON and keeps going on failures.

.PARAMETER InputList
  Text file with one extension per line: publisher.extension@version

.PARAMETER OutDir
  Output directory for downloaded .vsix files and report.json

.PARAMETER ReportName
  Report JSON filename. Default: report.json

.PARAMETER TargetPlatform
  Optional VSIX target platform string to prefer when downloading platform-specific variants
  (for example: linux-x64, linux-arm64, win32-x64, darwin-arm64).

.PARAMETER LocalCacheDir
  Optional list of VS Code CachedExtensionVSIXs directories to use as offline sources.

.PARAMETER RequiredIds
  Extension IDs (publisher.name) that must be downloadable. If any required ID cannot be downloaded
  (or results in a bad file), the script exits non-zero after writing the report.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$InputList,

    [Parameter(Mandatory = $true)]
    [string]$OutDir,

    [Parameter(Mandatory = $false)]
    [string]$ReportName = "report.json",

    [Parameter(Mandatory = $false)]
    [ValidatePattern("^[a-z0-9-]+$")]
    [string]$TargetPlatform = "",

    [Parameter(Mandatory = $false)]
    [string[]]$LocalCacheDir = @(),

    [Parameter(Mandatory = $false)]
    [string[]]$RequiredIds = @("ms-vscode-remote.remote-ssh")
)

$ErrorActionPreference = "Stop"

function Test-ZipHeader {
    param([Parameter(Mandatory = $true)][string]$Path)
    try {
        Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction SilentlyContinue | Out-Null
        $zip = [System.IO.Compression.ZipFile]::OpenRead($Path)
        try {
            # VSIX typically contains "extension/package.json" and/or "extension.vsixmanifest"
            $hasPackageJson = $zip.Entries | Where-Object { $_.FullName -eq "extension/package.json" } | Select-Object -First 1
            $hasManifest = $zip.Entries | Where-Object { $_.FullName -eq "extension.vsixmanifest" } | Select-Object -First 1
            return ($null -ne $hasPackageJson -or $null -ne $hasManifest)
        }
        finally {
            $zip.Dispose()
        }
    }
    catch {
        return $false
    }
}

function Get-VsixTargetPlatform {
    param([Parameter(Mandatory = $true)][string]$Path)
    try {
        Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction SilentlyContinue | Out-Null
        $zip = [System.IO.Compression.ZipFile]::OpenRead($Path)
        try {
            $entry = $zip.Entries | Where-Object { $_.FullName -eq "extension.vsixmanifest" } | Select-Object -First 1
            if (-not $entry) { return "" }

            $sr = New-Object System.IO.StreamReader($entry.Open())
            try {
                $raw = $sr.ReadToEnd()
            }
            finally {
                $sr.Dispose()
            }

            if ([string]::IsNullOrWhiteSpace($raw)) { return "" }
            [xml]$xml = $raw

            $identity = $xml.SelectSingleNode("//*[local-name()='Identity']")
            if (-not $identity) { return "" }
            $tp = $identity.GetAttribute("TargetPlatform")
            if ([string]::IsNullOrWhiteSpace($tp)) { return "" }
            return $tp.Trim().ToLowerInvariant()
        }
        finally {
            $zip.Dispose()
        }
    }
    catch {
        return ""
    }
}

function Test-VsixCompatibleWithTarget {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $false)][string]$TargetPlatform
    )

    if ([string]::IsNullOrWhiteSpace($TargetPlatform)) { return $true }
    $tp = Get-VsixTargetPlatform -Path $Path
    if ([string]::IsNullOrWhiteSpace($tp)) { return $true } # universal or unknown => assume compatible
    return ($tp -ieq $TargetPlatform)
}

function Invoke-Download {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$OutFile
    )

    $aria2 = Get-Command aria2c -ErrorAction SilentlyContinue
    if ($aria2) {
        try {
            $dirName = [System.IO.Path]::GetDirectoryName($OutFile)
            $fileName = [System.IO.Path]::GetFileName($OutFile)
            $partFile = "$OutFile.part"

            if ((Test-Path -LiteralPath $OutFile) -and (-not (Test-Path -LiteralPath $partFile))) {
                try { Move-Item -LiteralPath $OutFile -Destination $partFile -Force } catch { }
            }

            New-Item -ItemType Directory -Path $dirName -Force | Out-Null
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
                "--out=$($fileName).part",
                $Url
            )

            & $aria2.Source @args | Out-Null

            if (-not (Test-Path -LiteralPath $partFile)) { return $false }

            if (Test-Path -LiteralPath $OutFile) {
                Remove-Item -LiteralPath $OutFile -Force -ErrorAction SilentlyContinue
            }

            Move-Item -LiteralPath $partFile -Destination $OutFile -Force
            if (Test-Path -LiteralPath "$partFile.aria2") {
                Remove-Item -LiteralPath "$partFile.aria2" -Force -ErrorAction SilentlyContinue
            }

            return $true
        }
        catch {
            return $false
        }
    }

    try {
        Invoke-WebRequest -Uri $Url -OutFile $OutFile -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Parse-ExtensionSpec {
    param([Parameter(Mandatory = $true)][string]$Spec)

    $s = $Spec.Trim().Trim([char]0xFEFF)
    if ([string]::IsNullOrWhiteSpace($s)) { return $null }
    if ($s.StartsWith("#")) { return $null }

    $parts = $s.Split("@", 2)
    if ($parts.Count -ne 2) { throw "Invalid extension spec (expected id@version): $Spec" }

    $id = ($parts[0].Trim() -replace "\p{C}", "")
    $ver = ($parts[1].Trim() -replace "\p{C}", "")
    $dotIndex = $id.IndexOf(".")
    if ($dotIndex -le 0 -or $dotIndex -ge ($id.Length - 1)) { throw "Invalid extension id (expected publisher.name): $id" }
    if ([string]::IsNullOrWhiteSpace($ver)) { throw "Missing version for: $id" }

    $publisher = $id.Substring(0, $dotIndex)
    $name = $id.Substring($dotIndex + 1)

    return [pscustomobject]@{
        Id        = $id
        Publisher = $publisher
        Name      = $name
        Version   = $ver
    }
}

if (-not (Test-Path -LiteralPath $InputList)) {
    throw "InputList not found: $InputList"
}

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
$OutDir = (Resolve-Path $OutDir).Path

$cacheDirs = @()
if ($LocalCacheDir -and $LocalCacheDir.Count -gt 0) {
    foreach ($d in $LocalCacheDir) {
        if (-not [string]::IsNullOrWhiteSpace($d) -and (Test-Path -LiteralPath $d)) {
            $cacheDirs += (Resolve-Path $d).Path
        }
    }
}
else {
    $candidates = @(
        (Join-Path $env:APPDATA "Code\\CachedExtensionVSIXs"),
        (Join-Path $env:APPDATA "Code - Insiders\\CachedExtensionVSIXs")
    )
    foreach ($d in $candidates) {
        if (-not [string]::IsNullOrWhiteSpace($d) -and (Test-Path -LiteralPath $d)) {
            $cacheDirs += (Resolve-Path $d).Path
        }
    }
}

function Try-GetCacheKey {
    param([Parameter(Mandatory = $true)][string]$ZipPath)
    try {
        Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction SilentlyContinue | Out-Null
        $zip = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
        try {
            $entry = $zip.Entries | Where-Object { $_.FullName -eq "extension/package.json" } | Select-Object -First 1
            if (-not $entry) { return $null }

            $sr = New-Object System.IO.StreamReader($entry.Open())
            try {
                $json = $sr.ReadToEnd() | ConvertFrom-Json
            }
            finally {
                $sr.Dispose()
            }

            if (-not $json.publisher -or -not $json.name -or -not $json.version) { return $null }
            $id = "$($json.publisher).$($json.name)"
            $ver = "$($json.version)"
            if ([string]::IsNullOrWhiteSpace($id) -or [string]::IsNullOrWhiteSpace($ver)) { return $null }
            return ("$id@$ver").ToLowerInvariant()
        }
        finally {
            $zip.Dispose()
        }
    }
    catch {
        return $null
    }
}

$cacheIndex = @{}
foreach ($dir in $cacheDirs) {
    try {
        $files = Get-ChildItem -LiteralPath $dir -File -ErrorAction Stop
        foreach ($f in $files) {
            $key = Try-GetCacheKey -ZipPath $f.FullName
            if (-not $key) { continue }
            if (-not $cacheIndex.ContainsKey($key)) {
                $cacheIndex[$key] = $f.FullName
            }
        }
    }
    catch {
        # ignore cache scanning failures
    }
}

$installedExtRoots = @(
    (Join-Path $env:USERPROFILE ".vscode\\extensions"),
    (Join-Path $env:USERPROFILE ".vscode-insiders\\extensions")
)

function Try-ExportFromInstalledFolder {
    param(
        [Parameter(Mandatory = $true)][pscustomobject]$Spec,
        [Parameter(Mandatory = $true)][string]$OutFile,
        [Parameter(Mandatory = $true)][string[]]$Roots
    )

    $folderPrefix = "$($Spec.Id)-$($Spec.Version)"
    $sourceDir = $null

    foreach ($root in $Roots) {
        if ([string]::IsNullOrWhiteSpace($root)) { continue }
        if (-not (Test-Path -LiteralPath $root)) { continue }

        $exact = Join-Path $root $folderPrefix
        if (Test-Path -LiteralPath $exact) {
            $sourceDir = $exact
            break
        }

        try {
            $match = Get-ChildItem -LiteralPath $root -Directory -Filter "$folderPrefix*" -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($match) {
                $sourceDir = $match.FullName
                break
            }
        }
        catch {
            # ignore
        }
    }

    if (-not $sourceDir) { return $null }

    try {
        if (Test-Path -LiteralPath $OutFile) {
            Remove-Item -LiteralPath $OutFile -Force -ErrorAction SilentlyContinue
        }

        Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction SilentlyContinue | Out-Null

        $outFs = [System.IO.File]::Open($OutFile, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
        try {
            $zip = New-Object System.IO.Compression.ZipArchive($outFs, [System.IO.Compression.ZipArchiveMode]::Create, $true)
            try {
                $root = (Resolve-Path -LiteralPath $sourceDir).Path.TrimEnd("\", "/")
                $files = Get-ChildItem -LiteralPath $root -Recurse -File -ErrorAction Stop
                foreach ($f in $files) {
                    $full = $f.FullName
                    $rel = $full.Substring($root.Length).TrimStart("\", "/")
                    if ([string]::IsNullOrWhiteSpace($rel)) { continue }

                    $zipPath = ("extension/" + ($rel -replace "\\", "/"))
                    $entry = $zip.CreateEntry($zipPath, [System.IO.Compression.CompressionLevel]::Optimal)

                    $inStream = [System.IO.File]::Open($full, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read)
                    try {
                        $outStream = $entry.Open()
                        try {
                            $inStream.CopyTo($outStream)
                        }
                        finally {
                            $outStream.Dispose()
                        }
                    }
                    finally {
                        $inStream.Dispose()
                    }
                }
            }
            finally {
                $zip.Dispose()
            }
        }
        finally {
            $outFs.Dispose()
        }

        if (Test-ZipHeader -Path $OutFile) {
            return $sourceDir
        }

        try { Remove-Item -LiteralPath $OutFile -Force -ErrorAction SilentlyContinue } catch { }
        return $null
    }
    catch {
        try { Remove-Item -LiteralPath $OutFile -Force -ErrorAction SilentlyContinue } catch { }
        return $null
    }
}

$ovsxCmd = Get-Command ovsx -ErrorAction SilentlyContinue

$report = [ordered]@{
    started_at = (Get-Date).ToString("s")
    input_list = (Resolve-Path $InputList).Path
    out_dir    = $OutDir
    target_platform = $TargetPlatform
    local_cache_dir = $cacheDirs
    required   = $RequiredIds
    results    = @()
}

$requiredSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
foreach ($rid in $RequiredIds) {
    if (-not [string]::IsNullOrWhiteSpace($rid)) { $null = $requiredSet.Add($rid.Trim()) }
}

$hadRequiredFailure = $false

$lines = Get-Content -LiteralPath $InputList
foreach ($line in $lines) {
    $spec = $null
    try {
        $spec = Parse-ExtensionSpec -Spec $line
    }
    catch {
        $report.results += [ordered]@{
            spec    = $line
            status  = "invalid"
            message = $_.Exception.Message
        }
        continue
    }

    if (-not $spec) { continue }

    $suffix = ""
    if (-not [string]::IsNullOrWhiteSpace($TargetPlatform)) {
        $suffix = "@$TargetPlatform"
    }
    $baseName = "$($spec.Id)-$($spec.Version)$suffix.vsix"
    $outFile = Join-Path $OutDir $baseName

    $item = [ordered]@{
        id        = $spec.Id
        version   = $spec.Version
        target_platform = $TargetPlatform
        filename  = $baseName
        status    = "skipped"
        source    = $null
        url       = $null
        zip_valid = $false
        message   = $null
    }

    if ([string]::IsNullOrWhiteSpace($TargetPlatform)) {
        Write-Host "==> $($spec.Id)@$($spec.Version)" -ForegroundColor Cyan
    }
    else {
        Write-Host "==> $($spec.Id)@$($spec.Version) (prefer target: $TargetPlatform)" -ForegroundColor Cyan
    }

    if (Test-Path -LiteralPath $outFile) {
        if ((Test-ZipHeader -Path $outFile) -and (Test-VsixCompatibleWithTarget -Path $outFile -TargetPlatform $TargetPlatform)) {
            $item.status = "already_present"
            $item.source = "local"
            $item.url = $null
            $item.zip_valid = $true
            $report.results += $item
            Write-Host "    OK: $baseName (already present)" -ForegroundColor Green
            continue
        }

        try { Remove-Item -LiteralPath $outFile -Force -ErrorAction SilentlyContinue } catch { }
    }

    $cacheKey = ("$($spec.Id)@$($spec.Version)").ToLowerInvariant()
    if ($cacheIndex.ContainsKey($cacheKey)) {
        $cachePath = $cacheIndex[$cacheKey]
        try {
            Copy-Item -LiteralPath $cachePath -Destination $outFile -Force
            if ((Test-ZipHeader -Path $outFile) -and (Test-VsixCompatibleWithTarget -Path $outFile -TargetPlatform $TargetPlatform)) {
                $item.status = "downloaded"
                $item.source = "local_cache"
                $item.url = $cachePath
                $item.zip_valid = $true
                $report.results += $item
                Write-Host "    OK: $baseName (local cache)" -ForegroundColor Green
                continue
            }
            else {
                try { Remove-Item -LiteralPath $outFile -Force -ErrorAction SilentlyContinue } catch { }
            }
        }
        catch {
            try { Remove-Item -LiteralPath $outFile -Force -ErrorAction SilentlyContinue } catch { }
        }
    }

    $installedSource = Try-ExportFromInstalledFolder -Spec $spec -OutFile $outFile -Roots $installedExtRoots
    if ($installedSource) {
        if (Test-VsixCompatibleWithTarget -Path $outFile -TargetPlatform $TargetPlatform) {
            $item.status = "downloaded"
            $item.source = "local_install"
            $item.url = $installedSource
            $item.zip_valid = $true
            $report.results += $item
            Write-Host "    OK: $baseName (local install)" -ForegroundColor Green
            continue
        }
        else {
            try { Remove-Item -LiteralPath $outFile -Force -ErrorAction SilentlyContinue } catch { }
        }
    }

    $downloaded = $false

    # 1) Open VSX (ovsx CLI) if available
    if ($ovsxCmd) {
        try {
            if (-not [string]::IsNullOrWhiteSpace($TargetPlatform)) {
                & $ovsxCmd.Source get "$($spec.Id)@$($spec.Version)" -o $outFile --targetPlatform $TargetPlatform 2>$null | Out-Null
            }
            else {
                & $ovsxCmd.Source get "$($spec.Id)@$($spec.Version)" -o $outFile 2>$null | Out-Null
            }
            if ((Test-Path -LiteralPath $outFile) -and (Test-ZipHeader -Path $outFile)) {
                if (Test-VsixCompatibleWithTarget -Path $outFile -TargetPlatform $TargetPlatform) {
                    $downloaded = $true
                    $item.status = "downloaded"
                    $item.source = "openvsx"
                    $item.url = "ovsx get"
                }
                else {
                    try { Remove-Item -LiteralPath $outFile -Force -ErrorAction SilentlyContinue } catch { }
                }
            }
            elseif (Test-Path -LiteralPath $outFile) {
                try { Remove-Item -LiteralPath $outFile -Force -ErrorAction SilentlyContinue } catch { }
            }
        }
        catch {
            # fall through
        }
    }

    # 1b) Open VSX HTTP fallback
    if (-not $downloaded) {
        $openVsxUrls = @()
        if (-not [string]::IsNullOrWhiteSpace($TargetPlatform)) {
            $openVsxUrls += "https://open-vsx.org/api/$($spec.Publisher)/$($spec.Name)/$($spec.Version)/file/$($spec.Id)-$($spec.Version)@$TargetPlatform.vsix"
        }
        $openVsxUrls += "https://open-vsx.org/api/$($spec.Publisher)/$($spec.Name)/$($spec.Version)/file/$($spec.Id)-$($spec.Version).vsix"

        foreach ($openVsxUrl in $openVsxUrls) {
            if ($downloaded) { break }
            if (Invoke-Download -Url $openVsxUrl -OutFile $outFile) {
                if ((Test-ZipHeader -Path $outFile) -and (Test-VsixCompatibleWithTarget -Path $outFile -TargetPlatform $TargetPlatform)) {
                    $downloaded = $true
                    $item.status = "downloaded"
                    $item.source = "openvsx"
                    $item.url = $openVsxUrl
                }
                else {
                    try { Remove-Item -LiteralPath $outFile -Force -ErrorAction SilentlyContinue } catch { }
                }
            }
        }
    }

    # 2) Marketplace fallback (version-pinned)
    if (-not $downloaded) {
        $marketUrls = @()
        if (-not [string]::IsNullOrWhiteSpace($TargetPlatform)) {
            $tp = [System.Uri]::EscapeDataString($TargetPlatform)
            $marketUrls += "https://marketplace.visualstudio.com/_apis/public/gallery/publishers/$($spec.Publisher)/vsextensions/$($spec.Name)/$($spec.Version)/vspackage?targetPlatform=$tp"
        }
        $marketUrls += "https://marketplace.visualstudio.com/_apis/public/gallery/publishers/$($spec.Publisher)/vsextensions/$($spec.Name)/$($spec.Version)/vspackage"

        foreach ($marketUrl in $marketUrls) {
            if ($downloaded) { break }
            if (Invoke-Download -Url $marketUrl -OutFile $outFile) {
                if ((Test-ZipHeader -Path $outFile) -and (Test-VsixCompatibleWithTarget -Path $outFile -TargetPlatform $TargetPlatform)) {
                    $downloaded = $true
                    $item.status = "downloaded"
                    $item.source = "marketplace"
                    $item.url = $marketUrl
                }
                else {
                    try { Remove-Item -LiteralPath $outFile -Force -ErrorAction SilentlyContinue } catch { }
                }
            }
        }
    }

    if ($downloaded) {
        $item.zip_valid = Test-ZipHeader -Path $outFile
        if (-not $item.zip_valid) {
            $item.status = "bad_file"
            $item.message = "Downloaded file does not look like a VSIX (ZIP header missing)."
            if ($requiredSet.Contains($spec.Id)) {
                $hadRequiredFailure = $true
                $item.status = "required_bad_file"
            }
        }
        else {
            Write-Host "    OK: $baseName ($($item.source))" -ForegroundColor Green
        }
    }
    else {
        if ($requiredSet.Contains($spec.Id)) {
            $hadRequiredFailure = $true
            $item.status = "required_missing"
            $item.message = "Required extension not found on Open VSX or Marketplace."
            Write-Host "    ERROR: required extension not found" -ForegroundColor Red
        }
        else {
            $item.status = "not_found"
            $item.message = "Not found on Open VSX or Marketplace; skipped."
            Write-Host "    SKIP: not found" -ForegroundColor Yellow
        }
    }

    $report.results += $item
}

$report.finished_at = (Get-Date).ToString("s")
$report.had_required_failure = $hadRequiredFailure
$reportPath = Join-Path $OutDir $ReportName
$report | ConvertTo-Json -Depth 8 | Out-File -FilePath $reportPath -Encoding UTF8
Write-Host "==> Report: $reportPath" -ForegroundColor Green

if ($hadRequiredFailure) {
    throw "One or more required extensions failed to download. See report: $reportPath"
}
