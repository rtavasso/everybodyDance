# everybodyDance live sandbox -- one-shot setup + run on Windows.
#
#   Right-click > Run with PowerShell, or in a terminal:
#       powershell -ExecutionPolicy Bypass -File scripts\sandbox_windows.ps1
#
# Creates a local venv, installs the real-time stack (mediapipe / opencv /
# sounddevice -- pose weights auto-download on first run), then launches the
# webcam sandbox. Re-running skips install if the venv already exists.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment (.venv)..." -ForegroundColor Cyan
    python -m venv .venv
    & .\.venv\Scripts\python.exe -m pip install --upgrade pip
    Write-Host "Installing the real-time stack (this can take a few minutes)..." -ForegroundColor Cyan
    & .\.venv\Scripts\python.exe -m pip install -r requirements-realtime.txt
} else {
    Write-Host "Reusing existing .venv (delete it to reinstall)." -ForegroundColor DarkGray
}

Write-Host "`nLaunching live sandbox. Keys: q quit | r restart song | c re-calibrate | SPACE pause`n" -ForegroundColor Green
& .\.venv\Scripts\python.exe -m tools.live @args
