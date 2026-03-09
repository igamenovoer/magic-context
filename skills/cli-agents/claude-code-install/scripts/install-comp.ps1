<#
.SYNOPSIS
Installs the Claude Code CLI and optionally configures it to skip onboarding.

.DESCRIPTION
Installs `@anthropic-ai/claude-code` globally via bun or npm, preferring a
China mirror by default and falling back to the official npm registry when
needed. Optionally calls `config-skip-login.ps1` after installation.
#>

[CmdletBinding()]
param(
    [switch]$NoExit,
    [string]$Proxy,
    [switch]$FromOfficial,
    [switch]$Force,
    [switch]$SkipOnboarding,
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

function Test-ToolOnPath {
    param([string]$CommandName)
    return [bool](Get-Command $CommandName -ErrorAction SilentlyContinue)
}

$ErrorActionPreference = "Stop"

try {
    $packageName = "@anthropic-ai/claude-code"
    $mirrorRegistry = "https://registry.npmmirror.com"
    $officialRegistry = "https://registry.npmjs.org"
    $registry = if ($FromOfficial) { $officialRegistry } else { $mirrorRegistry }
    $runner = $null

    if ($Proxy) {
        $env:HTTP_PROXY = $Proxy
        $env:HTTPS_PROXY = $Proxy
        $env:http_proxy = $Proxy
        $env:https_proxy = $Proxy
    }

    if ((Test-ToolOnPath "claude") -and -not $Force) {
        Write-Host "claude is already available on PATH. Use -Force to reinstall."
        Exit-WithWait 0
    }

    if (Test-ToolOnPath "bun") {
        $runner = "bun"
    } elseif (Test-ToolOnPath "npm") {
        $runner = "npm"
    } else {
        throw "Neither bun nor npm is available. Install Bun or Node.js first."
    }

    if ($DryRun) {
        Write-Host "Dry-run: would install $packageName using $runner from $registry"
        if ($SkipOnboarding) {
            Write-Host "Dry-run: would run config-skip-login.ps1 after install"
        }
        Exit-WithWait 0
    }

    if ($runner -eq "bun") {
        Write-Host "Installing $packageName with bun from $registry"
        & bun add -g $packageName --registry $registry
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0 -and -not $FromOfficial) {
            Write-Host "Mirror install failed. Retrying with $officialRegistry"
            & bun add -g $packageName --registry $officialRegistry
            $exitCode = $LASTEXITCODE
        }
        if ($exitCode -ne 0) {
            throw "bun failed to install $packageName"
        }
    } else {
        Write-Host "Installing $packageName with npm from $registry"
        & npm install -g $packageName --registry $registry
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0 -and -not $FromOfficial) {
            Write-Host "Mirror install failed. Retrying with $officialRegistry"
            & npm install -g $packageName --registry $officialRegistry
            $exitCode = $LASTEXITCODE
        }
        if ($exitCode -ne 0) {
            throw "npm failed to install $packageName"
        }
    }

    if (-not (Test-ToolOnPath "claude")) {
        Write-Warning "Installation finished, but 'claude' is still not on PATH. Check your global package bin directory."
    } else {
        Write-Host "Claude Code CLI installed successfully."
        try {
            claude --version
        } catch {
        }
    }

    if ($SkipOnboarding) {
        $skipScript = Join-Path $PSScriptRoot "config-skip-login.ps1"
        & powershell -NoProfile -ExecutionPolicy Bypass -File $skipScript
        if ($LASTEXITCODE -ne 0) {
            throw "config-skip-login.ps1 failed after installation."
        }
    }

    Exit-WithWait 0
}
catch {
    Write-Error $_.Exception.Message
    Exit-WithWait 1
}