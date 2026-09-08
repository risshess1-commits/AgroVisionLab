$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\backend"

$venvPython = ".\.venv\Scripts\python.exe"
$recreateVenv = -not (Test-Path $venvPython)
$needInstall = $recreateVenv

if (-not $recreateVenv -and (Test-Path ".\.venv\pyvenv.cfg")) {
  $executable = (Select-String -Path ".\.venv\pyvenv.cfg" -Pattern "^executable = " -ErrorAction SilentlyContinue).Line
  if ($executable) {
    $executablePath = $executable -replace "^executable = ", ""
    if (-not (Test-Path $executablePath)) {
      $recreateVenv = $true
    }
  }
}

if (-not $recreateVenv) {
  $previousErrorActionPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  & $venvPython --version > $null 2>&1
  $venvCheckExitCode = $LASTEXITCODE
  $ErrorActionPreference = $previousErrorActionPreference
  if ($venvCheckExitCode -ne 0) {
    $recreateVenv = $true
  }
}

if ($recreateVenv) {
  if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python не найден. Установите Python 3.13+ или передайте проект вместе с рабочим .venv."
    exit 1
  }
  if (Test-Path ".venv") {
    Remove-Item -LiteralPath ".venv" -Recurse -Force
  }
  python -m venv .venv
}

if (-not $recreateVenv) {
  $checkScript = "import importlib.util, sys; mods=['fastapi','uvicorn','multipart','cv2','imageio_ffmpeg','ultralytics','numpy','pydantic']; sys.exit(0 if all(importlib.util.find_spec(m) for m in mods) else 1)"
  & $venvPython -c $checkScript > $null 2>&1
  if ($LASTEXITCODE -ne 0) {
    $needInstall = $true
  }
}

if ($needInstall) {
  & $venvPython -m pip install -r requirements.txt
}

& $venvPython -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
