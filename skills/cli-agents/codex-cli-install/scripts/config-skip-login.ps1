<#
.SYNOPSIS
Configures Codex to use a custom provider in config.toml and disables built-in login.
#>

[CmdletBinding()]
param(
    [switch]$NoExit,
    [string]$ProviderId = "codex-custom",
    [string]$BaseUrl = "https://api.openai.com/v1",
    [string]$EnvKey = "OPENAI_API_KEY",
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
    if ($ProviderId -notmatch '^[A-Za-z0-9_-]+$') {
        throw "ProviderId contains invalid characters."
    }
    if ($EnvKey -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') {
        throw "EnvKey must be a valid environment variable name."
    }
    if ($BaseUrl -and $BaseUrl -notmatch '^https?://') {
        throw "BaseUrl must start with http:// or https://"
    }

    $codexHome = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $env:USERPROFILE '.codex' }
    $configPath = Join-Path $codexHome 'config.toml'

    if ($DryRun) {
        Write-Host "Dry-run: would configure provider $ProviderId in $configPath"
        Exit-WithWait 0
    }

    New-Item -ItemType Directory -Force -Path $codexHome | Out-Null
    if (Test-Path $configPath) {
        $configContent = Get-Content $configPath -Raw -ErrorAction SilentlyContinue
        if ($null -eq $configContent) { $configContent = "" }
    } else {
        $configContent = ""
    }

    $configContent = $configContent -replace "`r`n", "`n"

    if ($configContent -match '(?m)^\s*model_provider\s*=') {
        $configContent = [regex]::Replace($configContent, '(?m)^\s*model_provider\s*=.*$', "model_provider = `"$ProviderId`"")
    } else {
        $configContent = "model_provider = `"$ProviderId`"`n" + $configContent
    }

    if ($configContent -notmatch '(?m)^\[model_providers\]') {
        if ($configContent.Length -gt 0 -and -not $configContent.EndsWith("`n")) {
            $configContent += "`n"
        }
        $configContent += "`n[model_providers]`n"
    }

    $escapedProviderId = [regex]::Escape($ProviderId)
    $providerBlockPattern = "(?ms)^\[model_providers\.$escapedProviderId\].*?(?=^\[|\z)"
    if ($configContent -match $providerBlockPattern) {
        $configContent = [regex]::Replace($configContent, $providerBlockPattern, "")
        $configContent = $configContent.TrimEnd() + "`n"
    }

    $providerLines = @()
    $providerLines += ""
    $providerLines += "[model_providers.$ProviderId]"
    $providerLines += 'name = "Custom OpenAI-compatible endpoint"'
    $providerLines += "base_url = `"$BaseUrl`""
    $providerLines += "env_key = `"$EnvKey`""
    $providerLines += "env_key_instructions = `"Set $EnvKey in your environment before launching codex.`""
    $providerLines += 'requires_openai_auth = false'
    $providerLines += 'wire_api = "responses"'
    $providerLines += ""

    if ($configContent.Length -gt 0 -and -not $configContent.EndsWith("`n")) {
        $configContent += "`n"
    }
    $configContent += ($providerLines -join "`n") + "`n"

    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($configPath, $configContent, $utf8NoBom)
    Write-Host "Updated $configPath for provider $ProviderId"
    Exit-WithWait 0
}
catch {
    Write-Error $_.Exception.Message
    Exit-WithWait 1
}