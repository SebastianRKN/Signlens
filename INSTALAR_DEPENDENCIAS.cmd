@echo off
setlocal
cd /d "%~dp0"
py -3.12 -m venv .venv
if errorlevel 1 goto failed
"%~dp0.venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
"%~dp0.venv\Scripts\python.exe" -m pip install -r requirements-tested.txt
if errorlevel 1 goto failed
"%~dp0.venv\Scripts\python.exe" -m pip install --no-deps -e .
if errorlevel 1 goto failed
echo Dependencias instaladas. Abre ABRIR_APP.cmd.
pause
exit /b 0
:failed
echo No se pudo instalar. Se requiere Python 3.12 y conexion a internet.
pause
exit /b 1
