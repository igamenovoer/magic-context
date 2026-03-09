<#
.SYNOPSIS
Configures a PowerShell alias/function for Codex with a custom endpoint and API key, and updates config.toml to disable built-in login.
#>

[CmdletBinding()]
param(
    [switch]$NoExit,
    [string]$AliasName,
    [string]$ProviderId,
    [string]$BaseUrl,
    [string]$ApiKey,
    [string]$ApiKeyEnv,
    [switch]$DryRun
)

function Exit-WithWait {
    param([int]$Code = 0)
    if ($NoExit) {
        Write-Host "Press any key to exit..."
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    }
    exit $Code
}

$ErrorActionPreference = "Stop"

try {
    if (-not (Get-Command codex -ErrorAction SilentlyContinue)) {
        throw "Codex CLI ('codex') is not on PATH. Install it first."
    }
    if (-not $AliasName) {
        throw "AliasName is required."
    }
    if ($AliasName -notmatch '^[A-Za-z0-9_-]+$') {
        throw "AliasName contains invalid characters."
    }
    if (-not $ProviderId) {
        $ProviderId = $AliasName
    }
    if ($ProviderId -notmatch '^[A-Za-z0-9_-]+$') {
        throw "ProviderId contains invalid characters."
    }
    if ($ProviderId -ieq 'openai') {
        throw "ProviderId 'openai' is reserved."
    }
    if ($BaseUrl -and $BaseUrl -notmatch '^https?://') {
        throw "BaseUrl must start with http:// or https:// when provided."
    }
    if ($ApiKey -and $ApiKeyEnv) {
        throw "Use either ApiKey or ApiKeyEnv, not both."
    }
    if (-not $ApiKey -and -not $ApiKeyEnv) {
        throw "One of ApiKey or ApiKeyEnv is required."
    }
    if ($ApiKeyEnv -and $ApiKeyEnv -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') {
        throw "ApiKeyEnv must be a valid environment variable name."
    }

    $userHome = $env:USERPROFILE
    $profileTargets = @(
        @{ Name = 'Windows PowerShell 5.1'; Path = (Join-Path $userHome 'Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1') },
        @{ Name = 'PowerShell 7+'; Path = (Join-Path $userHome 'Documents\PowerShell\Microsoft.PowerShell_profile.ps1') }
    )

    $beginMarker = "# BEGIN: Codex custom endpoint ($AliasName)"
    $endMarker = "# END: Codex custom endpoint ($AliasName)"

    $functionLines = @()
    $functionLines += ""
    $functionLines += $beginMarker
    $functionLines += "function $AliasName {"
    $functionLines += "    param("
    $functionLines += "        [Parameter(ValueFromRemainingArguments = `$true)]"
    $functionLines += "        [object[]]`$ForwardArgs"
    $functionLines += "    )"
    if ($BaseUrl) {
        $functionLines += "    `$env:OPENAI_BASE_URL = '$BaseUrl'"
    }
    if ($ApiKeyEnv) {
        $functionLines += "    if (-not `$env:$ApiKeyEnv) { throw 'Environment variable $ApiKeyEnv is required.' }"
        $functionLines += "    `$env:OPENAI_API_KEY = `$env:$ApiKeyEnv"
    } else {
        $functionLines += "    `$env:OPENAI_API_KEY = '$ApiKey'"
    }
    $functionLines += "    codex --search @ForwardArgs"
    $functionLines += "}"
    $functionLines += $endMarker
    $functionLines += ""

    if ($DryRun) {
        foreach ($target in $profileTargets) {
            Write-Host "Dry-run: would update profile $($target.Path)"
        }
        Exit-WithWait 0
    }

    foreach ($target in $profileTargets) {
        $profilePath = $target.Path
        $profileDir = Split-Path -Parent $profilePath
        if (-not (Test-Path $profileDir)) { New-Item -ItemType Directory -Path $profileDir -Force | Out-Null }
        if (-not (Test-Path $profilePath)) { New-Item -ItemType File -Path $profilePath -Force | Out-Null }
        $content = Get-Content -Path $profilePath -Raw -ErrorAction SilentlyContinue
        if ($null -eq $content) { $content = "" }
        $pattern = [regex]::Escape($beginMarker) + '.*?' + [regex]::Escape($endMarker)
        if ($content -match $pattern) {
            $content = [regex]::Replace($content, $pattern, '', 'Singleline')
            $content = $content.TrimEnd() + [Environment]::NewLine
        }
        $escapedAlias = [regex]::Escape($AliasName)
        $funcPattern = "(?msi)^\s*function\s+$escapedAlias\s*\{.*?^\}"
        if ($content -match $funcPattern) {
            $content = [regex]::Replace($content, $funcPattern, '', 'Singleline')
            $content = $content.TrimEnd() + [Environment]::NewLine
        }
        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($profilePath, $content, $utf8NoBom)
        Add-Content -Path $profilePath -Value $functionLines -Encoding UTF8
    }

    $skipScript = Join-Path $PSScriptRoot 'config-skip-login.ps1'
    $effectiveBaseUrl = if ($BaseUrl) { $BaseUrl } else { 'https://api.openai.com/v1' }
    & powershell -NoProfile -ExecutionPolicy Bypass -File $skipScript -ProviderId $ProviderId -BaseUrl $effectiveBaseUrl -EnvKey OPENAI_API_KEY
    if ($LASTEXITCODE -ne 0) {
        throw "config-skip-login.ps1 failed while updating Codex provider config."
    }

    Write-Host "Configured custom Codex alias: $AliasName"
    if ($ApiKeyEnv) {
        Write-Host "The generated alias reads its API key from environment variable: $ApiKeyEnv"
    } else {
        Write-Warning "The generated alias stores the API key in your PowerShell profile as plain text."
    }
    Exit-WithWait 0
}
catch {
    Write-Error $_.Exception.Message
    Exit-WithWait 1
}