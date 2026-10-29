@echo off
title AeroSoil AI - Live UAV Soil Dashboard
echo ======================================================================
echo   AeroSoil AI: Live UAV Hyperspectral Soil Dashboard Launcher
echo ======================================================================
echo.

cd /d "%~dp0"

echo [*] Checking for lingering processes on port 8501...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8501') do (
    if not "%%a"=="" (
        echo [*] Terminating lingering process PID %%a on port 8501...
        taskkill /F /PID %%a >nul 2>&1
    )
)

echo [*] Launching optimized zero-CDN dashboard server...
echo [*] URL: http://127.0.0.1:8501
echo.
"%~dp0python311\python.exe" dashboard.py

pause
