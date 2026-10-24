@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Ejecuta INSTALAR_DEPENDENCIAS.cmd primero.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" run_local.py --self-check
set "SIGNLENS_RESULT=%ERRORLEVEL%"
pause
exit /b %SIGNLENS_RESULT%
