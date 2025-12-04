# Checking PowerShell (`.ps1`) Scripts for Syntax and Linting (Without Running Them)

This guide summarizes common ways to validate PowerShell scripts **without executing** them:

- Syntax-only parsing to catch language errors.
- Static analysis / linting to catch style, best-practice, and some correctness issues.

The examples use generic paths and work in many environments; adapt them to your own workspace and tooling.

## 1. Syntax Checking via the PowerShell Parser

PowerShell exposes its parser as a .NET API. You can call it from the command line to validate a script file without running it.

### Using `Parser.ParseFile` (Recommended)

The `[System.Management.Automation.Language.Parser]` type can parse a script file and return syntax errors:

```powershell
pwsh -NoLogo -NoProfile -Command "
  $null = $tokens = $null
  $null = $errors = $null
  [void][System.Management.Automation.Language.Parser]::ParseFile(
    'path\to\script.ps1',
    [ref]$tokens,
    [ref]$errors
  )
  if ($errors.Count -gt 0) {
    $errors | Format-List *
    exit 1
  }
"
```

Notes:
- Works for both Windows PowerShell (`powershell`) and PowerShell 7+ (`pwsh`).
- Parses the script only; it does **not** execute any code.
- Non-zero exit code can be used in CI to fail on syntax errors.

### Using `Parser.ParseInput` for Inline Code

To validate dynamically generated or inline PowerShell code:

```powershell
pwsh -NoLogo -NoProfile -Command "
  $code   = Get-Content 'path\to\script.ps1' -Raw
  $null   = $tokens = $null
  $null   = $errors = $null
  [void][System.Management.Automation.Language.Parser]::ParseInput(
    $code,
    [ref]$tokens,
    [ref]$errors
  )
  if ($errors.Count -gt 0) {
    $errors | Format-List *
    exit 1
  }
"
```

This pattern is useful when the script content is not saved to disk yet (for example, editor or pipeline scenarios).

## 2. Static Analysis and Linting with PSScriptAnalyzer

[`PSScriptAnalyzer`](https://learn.microsoft.com/powershell/utility-modules/psscriptanalyzer/overview) is the de facto linter for PowerShell:

- Static code checker (no execution).
- Ships with rules for style, security, performance, and best practices.
- Extensible via custom rules and configurable settings.

### Installation

Install from the PowerShell Gallery:

```powershell
Install-Module PSScriptAnalyzer -Scope CurrentUser
```

> In locked-down environments, installation may need to happen via an internal repository or pre-bundled module.

### Basic Usage

Run the analyzer on a single script:

```powershell
Invoke-ScriptAnalyzer -Path 'path\to\script.ps1'
```

Analyze all scripts in a directory:

```powershell
Invoke-ScriptAnalyzer -Path 'path\to\scripts' -Recurse
```

Typical output includes rule name, severity, line, and a short description. The command does not run the script; it only inspects the source.

### Controlling Rules and Severity

Use built-in presets or a custom settings file:

```powershell
Invoke-ScriptAnalyzer -Path 'path\to\script.ps1' -Settings 'Recommended'
```

Or point to a configuration file:

```powershell
Invoke-ScriptAnalyzer -Path 'path\to\script.ps1' -Settings 'path\to\ScriptAnalyzerSettings.psd1'
```

In CI, you can treat warnings or errors as build failures by checking the returned objects or the command’s exit behavior.

## 3. Integrating Syntax Check and Linting in Automation

Common patterns for automated validation (local or CI/CD):

- **Syntax gate only**
  - Run a parser-based check (for example, `Parser.ParseFile`) and fail the job on any parse errors.
- **Syntax + lint**
  - First run the parser-based check.
  - Then run `Invoke-ScriptAnalyzer` and treat selected severities (e.g., errors) as failures.
- **Editor integration**
  - Many PowerShell editor extensions (for example, PowerShell extension for VS Code) use PSScriptAnalyzer under the hood.
  - Enable those integrations to get real-time squiggles for both syntax and rule violations.

These approaches validate scripts **without actually executing** them, which is essential when evaluating untrusted or partially written code or when scripting against sensitive environments.

