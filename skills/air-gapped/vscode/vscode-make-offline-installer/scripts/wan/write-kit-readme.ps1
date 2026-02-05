<#
.SYNOPSIS
  Write an installation README.md into a VS Code air-gapped kit output directory.

.DESCRIPTION
  This script is meant to run on the WAN-connected prep host after building the kit.
  It generates a README.md that can be copied along with the kit to air-gapped client(s) and server(s).

  It also (optionally) stages the required install scripts into the kit under ./scripts/.

.PARAMETER KitDir
  Root directory of the kit (contains clients/, server/, extensions/, manifest/).

.PARAMETER ReadmeName
  README filename to write at the kit root. Default: README.md

.PARAMETER NoStageScripts
  Do not copy the skill's client/server scripts into <KitDir>/scripts/.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$KitDir,

    [Parameter(Mandatory = $false)]
    [string]$ReadmeName = "README.md",

    [Parameter(Mandatory = $false)]
    [switch]$NoStageScripts
)

$ErrorActionPreference = "Stop"

function Ensure-Dir {
    param([Parameter(Mandatory = $true)][string]$Path)
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
}

function Read-JsonOrNull {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    $raw = Get-Content -LiteralPath $Path -Raw
    if ([string]::IsNullOrWhiteSpace($raw)) { return $null }
    try { return ($raw | ConvertFrom-Json -ErrorAction Stop) } catch { return $null }
}

function Get-RelativePath {
    param(
        [Parameter(Mandatory = $true)][string]$BaseDir,
        [Parameter(Mandatory = $true)][string]$Path
    )
    try {
        $base = (Resolve-Path -LiteralPath $BaseDir).Path
        $full = (Resolve-Path -LiteralPath $Path).Path
        return [System.IO.Path]::GetRelativePath($base, $full)
    }
    catch {
        return $Path
    }
}

function Find-Files {
    param(
        [Parameter(Mandatory = $true)][string]$Dir,
        [Parameter(Mandatory = $true)][string[]]$Patterns
    )
    if (-not (Test-Path -LiteralPath $Dir)) { return @() }
    $files = @()
    foreach ($pat in $Patterns) {
        $files += Get-ChildItem -LiteralPath $Dir -File -Filter $pat -ErrorAction SilentlyContinue
    }
    return @($files | Sort-Object -Property Name -Unique)
}

$KitDir = (Resolve-Path -LiteralPath $KitDir).Path

$manifestDir = Join-Path $KitDir "manifest"
$vscodeJson = Join-Path $manifestDir "vscode.json"
$vscodeLocalJson = Join-Path $manifestDir "vscode.local.json"

$manifest = Read-JsonOrNull -Path $vscodeJson
$local = Read-JsonOrNull -Path $vscodeLocalJson

$channel = $null
$commit = $null
$version = $null

if ($manifest) {
    $channel = $manifest.channel
    $commit = $manifest.commit
}
if ($local) {
    if (-not $channel) { $channel = $local.channel }
    if (-not $commit) { $commit = $local.commit }
    $version = $local.version
}

if (-not $channel) { $channel = "<stable|insider>" }
if (-not $commit) { $commit = "<COMMIT>" }
if (-not $version) { $version = "<VERSION>" }

if ($commit -is [string] -and $commit -ne "<COMMIT>" -and $commit -notmatch "^[0-9a-f]{40}$") {
    $commit = "<COMMIT>"
}

$clientsWindowsDir = Join-Path $KitDir "clients\\windows"
$clientsMacDir = Join-Path $KitDir "clients\\macos"
$clientsLinuxDir = Join-Path $KitDir "clients\\linux"

$serverLinuxX64Dir = Join-Path $KitDir "server\\linux-x64"
$serverLinuxArm64Dir = Join-Path $KitDir "server\\linux-arm64"
$serverCliDir = Join-Path $KitDir "server\\cli"

$extLocalDir = Join-Path $KitDir "extensions\\local"
$extRemoteDir = Join-Path $KitDir "extensions\\remote"

$winInstallers = @(Find-Files -Dir $clientsWindowsDir -Patterns @("*.exe", "*.msi", "*.zip"))
$macPackages = @(Find-Files -Dir $clientsMacDir -Patterns @("*.zip", "*.dmg"))
$linuxDeb = @(Find-Files -Dir $clientsLinuxDir -Patterns @("*.deb"))
$linuxRpm = @(Find-Files -Dir $clientsLinuxDir -Patterns @("*.rpm"))
$linuxTarGz = @(Find-Files -Dir $clientsLinuxDir -Patterns @("*.tar.gz"))
$linuxPackages = @($linuxDeb + $linuxRpm + $linuxTarGz)

$srvX64 = @(Find-Files -Dir $serverLinuxX64Dir -Patterns @("*.tar.gz"))
$srvArm64 = @(Find-Files -Dir $serverLinuxArm64Dir -Patterns @("*.tar.gz"))
$cliTars = @(Find-Files -Dir $serverCliDir -Patterns @("*.tar.gz"))

function Maybe-StageScripts {
    param([Parameter(Mandatory = $true)][string]$KitRoot)

    $kitScriptsDir = Join-Path $KitRoot "scripts"
    $kitClientDir = Join-Path $kitScriptsDir "client"
    $kitServerDir = Join-Path $kitScriptsDir "server"

    $skillScriptsDir = Resolve-Path (Join-Path $PSScriptRoot "..\\..\\scripts")
    $srcClient = Join-Path $skillScriptsDir "client"
    $srcServer = Join-Path $skillScriptsDir "server"

    if (-not (Test-Path -LiteralPath $srcClient)) { throw "Source scripts missing: $srcClient" }
    if (-not (Test-Path -LiteralPath $srcServer)) { throw "Source scripts missing: $srcServer" }

    Ensure-Dir $kitClientDir
    Ensure-Dir $kitServerDir

    Copy-Item -Path (Join-Path $srcClient "*") -Destination $kitClientDir -Recurse -Force
    Copy-Item -Path (Join-Path $srcServer "*") -Destination $kitServerDir -Recurse -Force
}

if (-not $NoStageScripts) {
    Maybe-StageScripts -KitRoot $KitDir
}

$readmePath = Join-Path $KitDir $ReadmeName

function Format-List {
    param(
        [Parameter(Mandatory = $false)][System.IO.FileInfo[]]$Files = @(),
        [Parameter(Mandatory = $true)][string]$KitRoot
    )
    if (-not $Files -or $Files.Count -eq 0) { return "  (none found)" }
    return ($Files | ForEach-Object { "  - ./{0}" -f (Get-RelativePath -BaseDir $KitRoot -Path $_.FullName).Replace('\', '/') }) -join "`n"
}

$winList = Format-List -Files $winInstallers -KitRoot $KitDir
$macList = Format-List -Files $macPackages -KitRoot $KitDir
$linuxList = Format-List -Files $linuxPackages -KitRoot $KitDir

$srvX64List = Format-List -Files $srvX64 -KitRoot $KitDir
$srvArmList = Format-List -Files $srvArm64 -KitRoot $KitDir
$cliList = Format-List -Files $cliTars -KitRoot $KitDir

$hasWindows = $winInstallers.Count -gt 0
$hasMac = $macPackages.Count -gt 0
$hasLinuxClient = $linuxPackages.Count -gt 0
$hasLinuxDeb = $linuxDeb.Count -gt 0
$hasLinuxRpm = $linuxRpm.Count -gt 0
$hasLinuxTarGz = $linuxTarGz.Count -gt 0
$hasServerX64 = $srvX64.Count -gt 0
$hasServerArm = $srvArm64.Count -gt 0
$hasCli = $cliTars.Count -gt 0

$clientInventory = @()
if ($hasWindows) { $clientInventory += "Windows:`n$winList" }
if ($hasMac) { $clientInventory += "macOS:`n$macList" }
if ($hasLinuxClient) { $clientInventory += "Linux:`n$linuxList" }
if ($clientInventory.Count -eq 0) { $clientInventory = @("(no client installers included)") }

$serverInventory = @()
if ($hasServerX64) { $serverInventory += "linux-x64:`n$srvX64List" }
if ($hasServerArm) { $serverInventory += "linux-arm64:`n$srvArmList" }
if ($serverInventory.Count -eq 0) { $serverInventory = @("(no server tarballs included)") }

$cliInventory = @()
if ($hasCli) { $cliInventory = @($cliList) } else { $cliInventory = @("  (none found)") }

$clientInstallSections = @()
if ($hasWindows) {
    $clientInstallSections += (@(
            '#### Windows'
            ''
            '1. Pick the installer under `./clients/windows/`'
            '2. Run the installer (interactive):'
            '   - `scripts\client\install-vscode-client.bat -InstallerPath .\clients\windows\<FILE>.exe`'
        ) -join "`n")
}
if ($hasLinuxClient) {
    $linuxInstallLines = @(
        '#### Linux desktop (air-gapped client)'
        ''
        'Pick a package under `./clients/linux/` and install it with your distro tooling.'
        ''
    )

    if ($hasLinuxDeb) {
        $debRel = ('./{0}' -f (Get-RelativePath -BaseDir $KitDir -Path $linuxDeb[0].FullName).Replace('\', '/'))
        $linuxInstallLines += @(
            'Ubuntu Desktop / Debian:'
            ('- Install: `sudo dpkg -i {0}`' -f $debRel)
            ('- Or helper: `bash scripts/client/install-vscode-client.sh --installer-path {0}`' -f $debRel)
            ''
            'If `dpkg` reports missing dependencies, you must stage and install those dependency `.deb` files offline too.'
            ''
        )
    }

    if ($hasLinuxRpm) {
        $rpmRel = ('./{0}' -f (Get-RelativePath -BaseDir $KitDir -Path $linuxRpm[0].FullName).Replace('\', '/'))
        $linuxInstallLines += @(
            'Fedora / RHEL-like:'
            ('- Install: `sudo rpm -Uvh {0}`' -f $rpmRel)
            ('- Or helper: `bash scripts/client/install-vscode-client.sh --installer-path {0}`' -f $rpmRel)
            ''
        )
    }

    if ($hasLinuxTarGz -and (-not ($hasLinuxDeb -or $hasLinuxRpm))) {
        $tgzRel = ('./{0}' -f (Get-RelativePath -BaseDir $KitDir -Path $linuxTarGz[0].FullName).Replace('\', '/'))
        $linuxInstallLines += @(
            'Tarball-based install:'
            ('- Extract the archive and follow your preferred offline deployment method: `{0}`' -f $tgzRel)
            ''
        )
    }

    $clientInstallSections += ($linuxInstallLines -join "`n")
}
if ($hasMac) {
    $clientInstallSections += (@(
            '#### macOS'
            ''
            '1. Pick the `.zip` or `.dmg` under `./clients/macos/`'
            '2. Install using your standard offline method (for `.zip`, extract and move the app into `/Applications`).'
        ) -join "`n")
}
if ($clientInstallSections.Count -eq 0) {
    $clientInstallSections += "(No client installers were included in this kit.)"
}

$serverInstallNote = "(No server artifacts were included in this kit.)"
if ($hasServerX64 -or $hasServerArm) {
    $arch = if ($hasServerX64) { "x64" } else { "arm64" }
    $serverTarFile = if ($hasServerX64) { $srvX64[0] } else { $srvArm64[0] }
    $serverTarRel = ('./{0}' -f (Get-RelativePath -BaseDir $KitDir -Path $serverTarFile.FullName).Replace('\', '/'))

    $cliCandidate = $null
    foreach ($f in $cliTars) {
        if ($f.Name -match [regex]::Escape("-$arch-")) { $cliCandidate = $f; break }
    }
    if (-not $cliCandidate -and $cliTars.Count -gt 0) { $cliCandidate = $cliTars[0] }

    $cliTarRel = if ($cliCandidate) { ('./{0}' -f (Get-RelativePath -BaseDir $KitDir -Path $cliCandidate.FullName).Replace('\', '/')) } else { "./server/cli/<CLI_TARBALL>.tar.gz" }

    $serverInstallNote = (@(
            'Pick the matching server tarball for your server architecture (x64 vs arm64) and a CLI tarball.'
            ''
            ('Example ({0}):' -f $arch)
            ''
            '```bash'
            'sudo bash scripts/server/install-vscode-server-cache.sh \'
            ('  --commit "{0}" --user "<USERNAME>" \' -f $commit)
            ('  --server-tar "{0}" \' -f $serverTarRel)
            ('  --cli-tar "{0}"' -f $cliTarRel)
            '```'
        ) -join "`n")
}

$configureHelpers = @()
if ($hasWindows) {
    $configureHelpers += '- Windows helper: `scripts\client\configure-vscode-client.bat -Channel auto`'
}
if ($hasLinuxClient -or $hasMac) {
    $configureHelpers += '- Linux/macOS helper: `bash scripts/client/configure-vscode-client.sh --channel auto`'
}
if ($configureHelpers.Count -eq 0) {
    $configureHelpers += '- Helper scripts: `./scripts/client/` (if included)'
}

$localExtHelpers = @()
if ($hasWindows) {
    $localExtHelpers += '- Windows helper: `scripts\client\install-vscode-client-extensions.bat -ExtensionsDir .\extensions\local -Channel auto`'
}
if ($hasLinuxClient -or $hasMac) {
    $localExtHelpers += '- Linux/macOS helper: `bash scripts/client/install-vscode-client-extensions.sh --extensions-dir ./extensions/local --channel auto`'
}
if ($localExtHelpers.Count -eq 0) {
    $localExtHelpers += '- Helper scripts: `./scripts/client/` (if included)'
}

$serverHasEnough = ($hasServerX64 -or $hasServerArm) -and $hasCli

$readme = @"
# VS Code Air-gapped Offline Kit (Remote-SSH)

This folder is an offline kit for Microsoft VS Code Remote-SSH.

## Release pin

- Channel: `$channel`
- Version: `$version` (if known)
- Commit: `$commit` (must match line 2 of `code --version` on the client)

## What is in this kit

### Client installers/archives

$(($clientInventory -join "`n`n"))

### Server artifacts (headless Linux)

VS Code Server tarballs:

$(($serverInventory -join "`n`n"))

VS Code CLI tarballs:
$($cliInventory -join "`n")

### Extensions (VSIX)

- Local (client/UI side): `./extensions/local/`
- Remote (server extension host): `./extensions/remote/`

## Install on an air-gapped desktop client

### 0) Copy the kit

Copy this entire folder to the client machine (USB/NAS).

### 1) Install VS Code (client)

$(($clientInstallSections -join "`n"))

### 2) Configure VS Code for air-gapped stability

Set these in user settings (disables auto-updates and prevents Remote-SSH downloads):

- `"update.mode": "manual"`
- `"extensions.autoCheckUpdates": false`
- `"extensions.autoUpdate": false`
- `"remote.SSH.localServerDownload": "off"`

$(($configureHelpers -join "`n"))

### 3) Install local (client-side) extensions from VSIX

Remote-SSH must be installed locally (`ms-vscode-remote.remote-ssh`).

$(($localExtHelpers -join "`n"))

## Install on an air-gapped headless Linux server (Remote-SSH target)

These steps must be run on the target Linux server (or via SSH to it). They pre-place the cache files so Remote-SSH does not try to download anything.

### 1) Install the server cache + extract the server

$serverInstallNote

### 2) Configure server state (settings + readiness marker)

```bash
sudo bash scripts/server/configure-vscode-server.sh --commit "$commit" --user "<USERNAME>"
```

### 3) Install remote-side extensions from VSIX (optional but common)

```bash
sudo bash scripts/server/install-vscode-server-extensions.sh \
  --commit "$commit" --user "<USERNAME>" \
  --extensions-dir "./extensions/remote"
```

### 4) Verify on the server

```bash
test -f ~/.vscode-server/vscode-cli-$commit.tar.gz.done
test -f ~/.vscode-server/vscode-server.tar.gz
test -x ~/.vscode-server/cli/servers/Stable-$commit/server/bin/code-server
~/.vscode-server/cli/servers/Stable-$commit/server/bin/code-server --list-extensions --show-versions --extensions-dir ~/.vscode-server/extensions
```

## Connect (desktop client)

In VS Code: run “Remote-SSH: Connect to Host...”. The Remote-SSH log should indicate it found an existing server installation and should not attempt downloads.
"@

if (-not $serverHasEnough) {
    $readme += (@(
            ""
            "---"
            ""
            "Note: this kit does not include both server tarballs and CLI tarballs for Linux. Remote-SSH offline typically requires both, plus the extracted server cache."
            ""
        ) -join "`n")
}

$readme | Out-File -FilePath $readmePath -Encoding UTF8
Write-Host "==> Wrote: $readmePath" -ForegroundColor Green
