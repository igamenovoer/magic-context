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

function Invoke-Download {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$OutFile
    )
    try {
        Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing -ErrorAction Stop | Out-Null
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

$ovsxCmd = Get-Command ovsx -ErrorAction SilentlyContinue

$report = [ordered]@{
    started_at = (Get-Date).ToString("s")
    input_list = (Resolve-Path $InputList).Path
    out_dir    = $OutDir
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

    $baseName = "$($spec.Id)-$($spec.Version).vsix"
    $outFile = Join-Path $OutDir $baseName

    $item = [ordered]@{
        id        = $spec.Id
        version   = $spec.Version
        filename  = $baseName
        status    = "skipped"
        source    = $null
        url       = $null
        zip_valid = $false
        message   = $null
    }

    Write-Host "==> $($spec.Id)@$($spec.Version)" -ForegroundColor Cyan

    $downloaded = $false

    # 1) Open VSX (ovsx CLI) if available
    if ($ovsxCmd) {
        try {
            & $ovsxCmd.Source get "$($spec.Id)@$($spec.Version)" -o $outFile 2>$null | Out-Null
            if (Test-Path -LiteralPath $outFile) {
                $downloaded = $true
                $item.status = "downloaded"
                $item.source = "openvsx"
                $item.url = "ovsx get"
            }
        }
        catch {
            # fall through
        }
    }

    # 1b) Open VSX HTTP fallback
    if (-not $downloaded) {
        $openVsxUrl = "https://open-vsx.org/api/$($spec.Publisher)/$($spec.Name)/$($spec.Version)/file/$($spec.Id)-$($spec.Version).vsix"
        if (Invoke-Download -Url $openVsxUrl -OutFile $outFile) {
            $downloaded = $true
            $item.status = "downloaded"
            $item.source = "openvsx"
            $item.url = $openVsxUrl
        }
    }

    # 2) Marketplace fallback (version-pinned)
    if (-not $downloaded) {
        $marketUrl = "https://marketplace.visualstudio.com/_apis/public/gallery/publishers/$($spec.Publisher)/vsextensions/$($spec.Name)/$($spec.Version)/vspackage"
        if (Invoke-Download -Url $marketUrl -OutFile $outFile) {
            $downloaded = $true
            $item.status = "downloaded"
            $item.source = "marketplace"
            $item.url = $marketUrl
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
