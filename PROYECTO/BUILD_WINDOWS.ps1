param([string]$Python, [switch]$InstallOnly)
$ErrorActionPreference = 'Stop'
if (-not $Python) {
    $candidates = @('python','python3') | ForEach-Object { Get-Command $_ -ErrorAction SilentlyContinue } | Select-Object -ExpandProperty Source -Unique
    foreach ($candidate in $candidates) {
        & $candidate -c "import sys,struct;sys.exit(0 if sys.version_info[:2]==(3,12) and struct.calcsize('P')==8 else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) { $Python = $candidate; break }
    }
}
if (-not $Python -or -not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'No se encontro Python 3.12 x64. Ejecuta BUILD_WINDOWS.ps1 -Python "ruta completa a python.exe". No se requiere el lanzador py.'
}
$buildArguments = @((Join-Path $PSScriptRoot 'build_windows.py'))
if ($InstallOnly) { $buildArguments += '--install-only' }
& $Python @buildArguments
if ($LASTEXITCODE -ne 0) { throw 'La compilacion/instalacion fallo. Revisa reports. No se continua con otras etapas.' }
