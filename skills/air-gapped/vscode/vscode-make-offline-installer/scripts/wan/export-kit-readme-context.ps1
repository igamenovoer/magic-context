<#
.SYNOPSIS
  Export kit README fill context for a VS Code air-gapped kit output directory (does NOT generate the final README.md).

.DESCRIPTION
  This script is meant to run on the WAN-connected prep host after building the kit.
  It exports a JSON context file that the agent can use to fill the README template manually.

  It also (optionally) stages the required install scripts into the kit under ./scripts/.

.PARAMETER KitDir
  Root directory of the kit (contains clients/, server/, extensions/, manifest/).

.PARAMETER ReadmeName
  Deprecated (kept for compatibility). This script does not generate README.md.

.PARAMETER ContextName
  Context JSON filename to write under <KitDir>/manifest/. Default: readme.context.json

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
    [string]$ContextName = "readme.context.json",

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

$clientsRoot = Join-Path $KitDir "clients"

function Find-ClientPlatformDirs {
    param([Parameter(Mandatory = $true)][string]$ClientsRoot)

    if (-not (Test-Path -LiteralPath $ClientsRoot)) { return @() }

    $dirs = Get-ChildItem -LiteralPath $ClientsRoot -Directory -ErrorAction SilentlyContinue
    if (-not $dirs) { return @() }

    return @($dirs | Select-Object -ExpandProperty FullName)
}

function Find-ClientFilesByPrefix {
    param(
        [Parameter(Mandatory = $true)][string]$ClientsRoot,
        [Parameter(Mandatory = $true)][string[]]$Prefixes,
        [Parameter(Mandatory = $true)][string[]]$Patterns
    )

    $files = @()

    # Legacy OS folders (windows/macos/linux)
    foreach ($legacy in @("windows", "macos", "linux")) {
        $p = Join-Path $ClientsRoot $legacy
        if (Test-Path -LiteralPath $p) {
            $files += Find-Files -Dir $p -Patterns $Patterns
        }
    }

    # Platform-reflecting folders (e.g. win32-x64-user, linux-deb-x64, darwin-universal)
    $platDirs = Find-ClientPlatformDirs -ClientsRoot $ClientsRoot
    foreach ($d in $platDirs) {
        $name = Split-Path -Leaf $d
        $match = $false
        foreach ($pref in $Prefixes) {
            if ($name.StartsWith($pref)) { $match = $true; break }
        }
        if (-not $match) { continue }
        $files += Find-Files -Dir $d -Patterns $Patterns
    }

    return @($files | Sort-Object -Property FullName -Unique)
}

$winInstallers = @()
$macPackages = @()
$linuxPackages = @()

if (Test-Path -LiteralPath $clientsRoot) {
    $winInstallers = @(Find-ClientFilesByPrefix -ClientsRoot $clientsRoot -Prefixes @("win32-") -Patterns @("*.exe", "*.msi", "*.zip"))
    $macPackages = @(Find-ClientFilesByPrefix -ClientsRoot $clientsRoot -Prefixes @("darwin-") -Patterns @("*.zip", "*.dmg"))
    $linuxPackages = @(Find-ClientFilesByPrefix -ClientsRoot $clientsRoot -Prefixes @("linux-") -Patterns @("*.deb", "*.rpm", "*.tar.gz"))
}

$serverLinuxX64Dir = Join-Path $KitDir "server\\linux-x64"
$serverLinuxArm64Dir = Join-Path $KitDir "server\\linux-arm64"
$serverCliDir = Join-Path $KitDir "server\\cli"

$extLocalDir = Join-Path $KitDir "extensions\\local"
$extRemoteDir = Join-Path $KitDir "extensions\\remote"

$linuxDeb = @($linuxPackages | Where-Object { $_.Extension -eq ".deb" })
$linuxRpm = @($linuxPackages | Where-Object { $_.Extension -eq ".rpm" })
$linuxTarGz = @($linuxPackages | Where-Object { $_.Name -like "*.tar.gz" })

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
            '1. Pick an installer under `./clients/win32-*/` (or legacy `./clients/windows/`)'
            '2. Run the installer (interactive):'
            '   - Recommended (auto-detects from `./clients/`): `scripts\client\install-vscode-client.bat`'
            '   - Or specify an exact file: `scripts\client\install-vscode-client.bat -InstallerPath .\clients\<PLATFORM>\<FILE>.exe`'
        ) -join "`n")
}
if ($hasLinuxClient) {
    $linuxInstallLines = @(
        '#### Linux desktop (air-gapped client)'
        ''
        'Pick a package under `./clients/linux-*/` (or legacy `./clients/linux/`) and install it with your distro tooling.'
        ''
    )

    if ($hasLinuxDeb) {
        $debRel = ('./{0}' -f (Get-RelativePath -BaseDir $KitDir -Path $linuxDeb[0].FullName).Replace('\', '/'))
        $linuxInstallLines += @(
            'Ubuntu Desktop / Debian:'
            ('- Install: `sudo dpkg -i {0}`' -f $debRel)
            '- Or helper (auto-detects from `./clients/`): `bash scripts/client/install-vscode-client.sh`'
            ('- Or helper (explicit): `bash scripts/client/install-vscode-client.sh --installer-path {0}`' -f $debRel)
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
            '- Or helper (auto-detects from `./clients/`): `bash scripts/client/install-vscode-client.sh`'
            ('- Or helper (explicit): `bash scripts/client/install-vscode-client.sh --installer-path {0}`' -f $rpmRel)
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
            'Recommended: run the helper with no tarball paths; it auto-detects `--commit` from `./manifest/` and tarballs from `./server/` relative to the script location.'
            ''
            '```bash'
            'sudo bash scripts/server/install-vscode-server-cache.sh --user "<USERNAME>"'
            '```'
            ''
            'If you need to override (for example multiple commits/artifacts in one kit), pass explicit args:'
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
    $localExtHelpers += '- Windows helper (auto-detects from `./extensions/local/`): `scripts\client\install-vscode-client-extensions.bat -Channel auto`'
    $localExtHelpers += '- Windows helper (explicit): `scripts\client\install-vscode-client-extensions.bat -ExtensionsDir .\extensions\local -Channel auto`'
}
if ($hasLinuxClient -or $hasMac) {
    $localExtHelpers += '- Linux/macOS helper (auto-detects from `./extensions/local/`): `bash scripts/client/install-vscode-client-extensions.sh --channel auto`'
    $localExtHelpers += '- Linux/macOS helper (explicit): `bash scripts/client/install-vscode-client-extensions.sh --extensions-dir ./extensions/local --channel auto`'
}
if ($localExtHelpers.Count -eq 0) {
    $localExtHelpers += '- Helper scripts: `./scripts/client/` (if included)'
}

$serverHasEnough = ($hasServerX64 -or $hasServerArm) -and $hasCli

Ensure-Dir $manifestDir
$contextPath = Join-Path $manifestDir $ContextName

function RelFiles {
    param(
        [Parameter(Mandatory = $false)]
        [AllowEmptyCollection()]
        [System.IO.FileInfo[]]$Files = @(),

        [Parameter(Mandatory = $true)]
        [string]$KitRoot
    )
    if (-not $Files -or $Files.Count -eq 0) { return ,@() }
    return ,@($Files | ForEach-Object { ('./{0}' -f (Get-RelativePath -BaseDir $KitRoot -Path $_.FullName).Replace('\', '/')) })
}

$extLocalVsix = @(Find-Files -Dir $extLocalDir -Patterns @("*.vsix"))
$extRemoteVsix = @(Find-Files -Dir $extRemoteDir -Patterns @("*.vsix"))

$payload = [ordered]@{
    generated_at = (Get-Date).ToString("s")
    kit_dir      = $KitDir
    release      = [ordered]@{
        channel = $channel
        version = $version
        commit  = $commit
    }
    inventories  = [ordered]@{
        clients    = [ordered]@{
            windows = RelFiles -Files $winInstallers -KitRoot $KitDir
            macos   = RelFiles -Files $macPackages -KitRoot $KitDir
            linux   = RelFiles -Files $linuxPackages -KitRoot $KitDir
        }
        server     = [ordered]@{
            linux_x64  = RelFiles -Files $srvX64 -KitRoot $KitDir
            linux_arm64 = RelFiles -Files $srvArm64 -KitRoot $KitDir
            cli        = RelFiles -Files $cliTars -KitRoot $KitDir
        }
        extensions = [ordered]@{
            local_vsix  = RelFiles -Files $extLocalVsix -KitRoot $KitDir
            remote_vsix = RelFiles -Files $extRemoteVsix -KitRoot $KitDir
        }
    }
    helpers      = [ordered]@{
        server_has_enough_artifacts = $serverHasEnough
        client_install_sections_md  = ($clientInstallSections -join "`n")
        server_install_note_md      = $serverInstallNote
        configure_helpers_md        = ($configureHelpers -join "`n")
        local_extension_helpers_md  = ($localExtHelpers -join "`n")
    }
    notes        = @(
        "This script does NOT generate README.md. Fill the README template manually.",
        "Template source (in this skill pack): references/kit-readme.template.md",
        "Recommended: copy the template into the kit root as README.md and replace {{...}} placeholders using this context plus the actual kit contents."
    )
}

$payload | ConvertTo-Json -Depth 10 | Out-File -FilePath $contextPath -Encoding UTF8

Write-Host "==> Wrote context: $contextPath" -ForegroundColor Green
Write-Host "==> NOTE: This script does NOT generate README.md (ReadmeName '$ReadmeName' is deprecated)." -ForegroundColor Yellow
