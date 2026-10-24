@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0BUILD_WINDOWS.ps1" -InstallOnly
if errorlevel 1 (
 echo Error. Instala Python 3.12 x64 o indica su ruta mediante BUILD_WINDOWS.ps1 -Python.
 pause
 exit /b 1
)
pause
