"""Run with an actual Python 3.12 x64 interpreter on Windows; no py launcher assumed."""
import argparse
import datetime
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import uuid
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--install-only', action='store_true')
    args = ap.parse_args()
    if os.name != 'nt' or sys.version_info[:2] != (3, 12) or struct.calcsize('P') != 8:
        raise SystemExit('Se requiere Python 3.12 de 64 bits en Windows. Selecciona ese python.exe; no se requiere py -3.12.')
    sys.path.insert(0, str(ROOT / 'src'))
    from signlens.reporting import report_path
    log_path = report_path('build', '.log', ROOT)
    work = Path(tempfile.mkdtemp(prefix='SignLens-build-'))
    venv = ROOT / '.venv' if args.install_only else work / 'venv'
    with log_path.open('w', encoding='utf-8', buffering=1) as log:
        def run(command, cwd=ROOT, env=None):
            log.write('\nCOMMAND: ' + repr(list(map(str,command))) + '\n')
            proc = subprocess.Popen(list(map(str,command)), cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
            for line in proc.stdout:
                log.write(line)
                print(line, end='', flush=True)
            if proc.wait() != 0:
                raise RuntimeError(f'Etapa fallida ({proc.returncode}). No se crea entrega. Registro: {log_path}')
        run([sys.executable, '-m', 'venv', venv])
        python = venv / 'Scripts/python.exe'
        if not python.is_file(): raise RuntimeError('El entorno virtual no contiene python.exe.')
        run([python, '-m', 'pip', 'install', '-r', ROOT/'requirements-tested.txt', '-r', ROOT/'requirements-build.txt'])
        if args.install_only:
            print('Entorno listo. Ejecuta ABRIR_APP.cmd.'); return
        env = os.environ.copy(); env['PYTHONPATH'] = str(ROOT/'src')
        run([python,'-m','pytest','-q',f'--junitxml={report_path("tests", ".xml", ROOT)}'],env=env)
        run([python,ROOT/'run_local.py','--self-check'])
        run([python,'-m','PyInstaller','--noconfirm','--clean','--workpath',work/'build','--distpath',work/'dist',ROOT/'SignLens.spec'])
        dist = work/'dist/SignLens'; exe = dist/'SignLens.exe'
        if not exe.is_file() or exe.read_bytes()[:2] != b'MZ': raise RuntimeError('No se genero un ejecutable Windows valido.')
        clean = os.environ.copy()
        clean.pop('PYTHONPATH',None); clean.pop('PYTHONHOME',None); clean.pop('SIGNLENS_HOME',None)
        clean['PATH'] = str(Path(os.environ['SystemRoot'])/'System32')
        run([exe,'--self-check'],cwd=work,env=clean)
        run([exe,'--tracker-check'],cwd=work,env=clean)
        stage = work/'SignLens_CONCURSO_COMPLETO'; stage.mkdir()
        shutil.copytree(ROOT,stage/'PROYECTO',ignore=shutil.ignore_patterns('.venv','__pycache__','.pytest_cache','*.pyc','.git'))
        shutil.copytree(dist,stage/'EJECUTABLE_WINDOWS')
        instructions = ROOT.parent/'instrucciones.md'
        if not instructions.is_file(): raise RuntimeError('Falta instrucciones.md junto a PROYECTO.')
        shutil.copy2(instructions,stage/instructions.name)
        target=ROOT.parent.parent/('SignLens_CONCURSO_COMPLETO_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.zip')
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for f in stage.rglob('*'):
                if f.is_file():z.write(f,f.relative_to(stage))
        with zipfile.ZipFile(target) as z:
            if z.testzip() or 'EJECUTABLE_WINDOWS/SignLens.exe' not in z.namelist():raise RuntimeError('ZIP invalido.')
        print(f'Entrega compilada y comprobada: {target}')
        print(f'Entorno y archivos de compilacion conservados en: {work}')


if __name__ == '__main__':
    main()
