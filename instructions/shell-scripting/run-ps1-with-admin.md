# Running a PowerShell Script with Administrative Rights via Batch Wrapper

This pattern lets you run a PowerShell script (`.ps1`) with administrative privileges using a small batch (`.bat`) wrapper. It is suitable for one‑off helpers and temporary tools that need elevation (for example, Hyper‑V operations, system configuration, or editing protected locations).

The approach also supports writing output to a log file so that the batch script can display it back to the user.

## High‑Level Pattern

- Put your main logic in a PowerShell script (`tool.ps1`).
- Create a batch wrapper (`tool.bat`) that:
  - Locates the `.ps1` file.
  - Ensures any temporary/script directory exists.
  - Checks for administrative privileges.
  - If not admin, relaunches PowerShell elevated (`RunAs`) and waits.
  - Optionally captures output to a log file and prints it.

## Choosing Where to Store Temporary Scripts

When you need to generate or place temporary helper scripts:

- If the current workspace has a `tmp` directory:
  - Prefer a subdirectory under it, such as:  
    `<workspace>\tmp\<subdir>`
  - Example convention: `tmp\hyperv`, `tmp\dev-tools`, etc.
- If there is no workspace `tmp` directory:
  - Use the system temporary location (for example, `%TEMP%` on Windows).

Temporary scripts and logs should be treated as disposable artifacts.

## PowerShell Script Pattern (`.ps1`)

Key ideas for the PowerShell script:

- Accept optional `-CaptureLogFile` to send all output to a file.
- Use `$ErrorActionPreference = "Stop"` so failures surface clearly.
- Build output in a string array and write it once at the end.
- Use `Out-File -Encoding Default` when writing logs that a batch file will print.

Example structure (simplified):

```powershell
param(
    [string]$CaptureLogFile
)

$ErrorActionPreference = "Stop"

$lines = @()
$lines += ""
$lines += "=== Tool Header ==="
$lines += ""

try {
    # Main logic goes here
    # ...
    $lines += "Operation completed successfully."
} catch {
    $lines += "Error: $($_.Exception.Message)"
    if ($CaptureLogFile) {
        $lines -join "`r`n" | Out-File -FilePath $CaptureLogFile -Encoding Default -Force
    } else {
        $lines | ForEach-Object { Write-Host $_ }
    }
    exit 1
}

if ($CaptureLogFile) {
    $lines -join "`r`n" | Out-File -FilePath $CaptureLogFile -Encoding Default -Force
} else {
    $lines | ForEach-Object { Write-Host $_ }
}
```

## Batch Wrapper Pattern (`.bat`)

Key ideas for the batch wrapper:

- Check for admin via `net session`.
- If not admin, call `powershell` with `Start-Process -Verb RunAs`.
- Generate a unique log file (for example using a GUID).
- After the PowerShell script finishes, print and delete the log file.

Example structure (simplified):

```bat
@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PS1_FILE=%SCRIPT_DIR%tool.ps1"

rem Generate a unique log file in %TEMP%
for /f "usebackq delims=" %%G in (`
  powershell -NoProfile -Command "[guid]::NewGuid().ToString()"
`) do set "LOG_ID=%%G"
set "OUT_FILE=%TEMP%\tool-%LOG_ID%.log"

rem Check that the PowerShell script exists
if not exist "%PS1_FILE%" (
    echo Error: PowerShell script not found:
    echo   "%PS1_FILE%"
    echo.
    endlocal
    exit /b 1
)

rem Check for administrative privileges
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting administrative privileges...

    powershell -NoLogo -NoProfile -Command ^
        "$ps1 = '%PS1_FILE%'; $out = '%OUT_FILE%';" ^
        "Start-Process -FilePath 'powershell.exe' -Verb RunAs -Wait " ^
        "-ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',$ps1,'-CaptureLogFile',$out)"

    if exist "%OUT_FILE%" (
        type "%OUT_FILE%"
        del "%OUT_FILE%" >nul 2>&1
    ) else (
        echo No output received from elevated PowerShell process.
    )

    endlocal
    exit /b
)

rem Already running as admin: call PowerShell directly
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%PS1_FILE%" -CaptureLogFile "%OUT_FILE%"
set "EXITCODE=%ERRORLEVEL%"

if exist "%OUT_FILE%" (
    type "%OUT_FILE%"
    del "%OUT_FILE%" >nul 2>&1
)

endlocal & exit /b %EXITCODE%
```

## Usage Workflow

1) Place your main logic in a `.ps1` script following the logging pattern.  
2) Create a `.bat` wrapper in the same folder that:
   - Locates the `.ps1`,
   - Requests elevation if needed,
   - Writes output to a log file,
   - Prints the log and cleans it up.  
3) For temporary helpers:
   - Store the scripts under `<workspace>\tmp\<subdir>` if available,  
   - Otherwise, place them in the system temp directory.  
4) Run the batch file from a normal shell; it will handle elevation and output routing automatically.

