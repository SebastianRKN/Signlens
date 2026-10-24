@echo off
setlocal
cd /d "%~dp0"
set "SIGNLENS_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%SIGNLENS_PYTHON%" (
  echo Falta el entorno local. Ejecuta INSTALAR_DEPENDENCIAS.cmd primero.
  pause
  exit /b 1
)
"%SIGNLENS_PYTHON%" "%~dp0run_local.py" %*
set "SIGNLENS_RESULT=%ERRORLEVEL%"
if not "%SIGNLENS_RESULT%"=="0" (
  echo SignLens no pudo iniciar. Revisa reports\AAAA-MM-DD\application_*.log y ejecuta VERIFICAR_APP.cmd.
  pause
)
exit /b %SIGNLENS_RESULT%
