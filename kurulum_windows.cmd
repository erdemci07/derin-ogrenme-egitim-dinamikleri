@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0kurulum_windows.ps1"
if errorlevel 1 (
    echo.
    echo Kurulum sirasinda hata olustu.
    pause
    exit /b 1
)
echo.
pause
