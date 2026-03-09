<#
.SYNOPSIS
Marks Claude Code CLI onboarding as completed so it skips the login/onboarding flow.
#>

[CmdletBinding()]
param(
    [switch]$NoExit,
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
    if (-not (Get-Command claude -ErrorAction SilentlyContinue)) {
        throw "Claude Code CLI ('claude') is not on PATH. Install it first."
    }

    $configFile = Join-Path $env:USERPROFILE ".claude.json"

    if ($DryRun) {
        Write-Host "Dry-run: would set hasCompletedOnboarding=true in $configFile"
        Exit-WithWait 0
    }

    $config = @{}
    if (Test-Path $configFile) {
        try {
            $content = Get-Content $configFile -Raw -ErrorAction Stop
            if ($content.Trim().Length -gt 0) {
                $config = $content | ConvertFrom-Json -AsHashtable -ErrorAction Stop
            }
        } catch {
            $config = @{}
        }
    }

    $config.hasCompletedOnboarding = $true
    $json = $config | ConvertTo-Json -Depth 10
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($configFile, $json, $utf8NoBom)

    Write-Host "Updated $configFile with hasCompletedOnboarding=true"
    Exit-WithWait 0
}
catch {
    Write-Error $_.Exception.Message
    Exit-WithWait 1
}