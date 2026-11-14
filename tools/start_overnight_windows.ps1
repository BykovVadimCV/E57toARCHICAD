param(
    [string]$Output = 'D:\remote_e57\dataset_native_hd_v3',
    [int]$Count = 1000,
    [int]$Workers = 0,
    [int]$Seed = 31111
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (!(Test-Path $Python)) { throw 'Create the environment first: python -m venv .venv; then install the project.' }
New-Item -ItemType Directory -Force "$Root\logs" | Out-Null
$Log = Join-Path $Root 'logs\overnight_hd.log'
if (!(Test-Path $Log)) { New-Item -ItemType File -Path $Log | Out-Null }
$Runner = Join-Path $Root 'logs\run_overnight_hd.cmd'
$Body = "@echo off`r`nchcp 65001 >nul`r`nset PYTHONUTF8=1`r`nset PYTHONUNBUFFERED=1`r`ncd /d `"$Root`"`r`n`"$Python`" -u tools\run_overnight_dataset.py --out `"$Output`" --count $Count --workers $Workers --seed $Seed >> `"$Log`" 2>&1`r`necho Exit code: %ERRORLEVEL% >> `"$Log`"`r`n"
[System.IO.File]::WriteAllText($Runner, $Body, (New-Object System.Text.UTF8Encoding($false)))
# WMI owns the child process rather than the SSH session; local disk paths only.
$Result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = "cmd.exe /c `"`"$Runner`"`""; CurrentDirectory = $Root}
if ($Result.ReturnValue -ne 0) { throw "Windows could not start the detached job (code $($Result.ReturnValue))." }
Write-Host "Started process $($Result.ProcessId). Log: $Log"
