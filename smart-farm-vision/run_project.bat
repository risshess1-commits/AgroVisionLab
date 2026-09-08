@echo off
cd /d "%~dp0"
if exist "%~dp0frontend\dist\index.html" (
  start "AgroVision Backend" powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_backend.ps1"
  timeout /t 6 /nobreak >nul
  start "" "http://127.0.0.1:8000"
  goto :eof
)
start "AgroVision Backend" powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_backend.ps1"
timeout /t 5 /nobreak >nul
start "AgroVision Frontend" powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_frontend.ps1"
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:5173"
