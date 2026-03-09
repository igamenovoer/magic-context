<#
.SYNOPSIS
Installs the Codex CLI.
#>

[CmdletBinding()]
param(
    [switch]$NoExit,
    [string]$Proxy,
    [switch]$FromOfficial,
    [switch]$Force,
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
    $packageName = "@openai/codex"
    $mirrorRegistry = "https://registry.npmmirror.com"
    $officialRegistry = "https://registry.npmjs.org"
    $registry = if ($FromOfficial) { $officialRegistry } else { $mirrorRegistry }

    if ($Proxy) {
        $env:HTTP_PROXY = $Proxy
        $env:HTTPS_PROXY = $Proxy
        $env:http_proxy = $Proxy
        $env:https_proxy = $Proxy
    }

    if ((Test-ToolOnPath "codex") -and -not $Force) {
        Write-Host "codex is already available on PATH. Use -Force to reinstall."
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
        Exit-WithWait 0
    }

    if ($runner -eq "bun") {
        & bun add -g $packageName --registry $registry
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0 -and -not $FromOfficial) {
            & bun add -g $packageName --registry $officialRegistry
            $exitCode = $LASTEXITCODE
        }
    } else {
        & npm install -g $packageName --registry $registry
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0 -and -not $FromOfficial) {
            & npm install -g $packageName --registry $officialRegistry
            $exitCode = $LASTEXITCODE
        }
    }

    if ($exitCode -ne 0) {
        throw "Failed to install $packageName"
    }

    if (Test-ToolOnPath "codex") {
        try { codex --version } catch {}
    } else {
        Write-Warning "Installation finished, but 'codex' is still not on PATH. Check your global package bin directory."
    }

    Exit-WithWait 0
}
catch {
    Write-Error $_.Exception.Message
    Exit-WithWait 1
}