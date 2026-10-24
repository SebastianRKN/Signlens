"""Read-only bundled seed resources and writable portable/user storage."""
from pathlib import Path
import os
import shutil
import sys
import uuid


def resource_root():
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS) / 'seed'
    return Path(__file__).resolve().parents[2]


def writable(path):
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ('.write-' + uuid.uuid4().hex)
        with probe.open('x') as f:
            f.write('ok')
        probe.unlink()
        return True
    except OSError:
        return False


def prepare_storage():
    seed = resource_root()
    default = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else seed
    root = Path(os.environ.get('SIGNLENS_HOME', default)).expanduser().resolve()
    fallback = False
    def usable(folder):
        return all(writable(folder / part) for part in ('', 'models', 'data', 'reports', 'config'))
    if not usable(root):
        fallback = True
        root = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local')) / 'SignLens'
        if not usable(root):
            raise OSError('No hay una carpeta con permiso de escritura para SignLens.')
    if root != seed:
        for folder in ('models', 'data'):
            for src in (seed / folder).rglob('*'):
                if not src.is_file() or src.name.startswith('camera_'):
                    continue
                dest = root / folder / src.relative_to(seed / folder)
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    temporary = dest.with_name('.seed-' + uuid.uuid4().hex)
                    try:
                        shutil.copyfile(src, temporary)
                        # Windows rename does not overwrite an existing destination.
                        if os.name == 'nt':
                            temporary.rename(dest)
                        elif not dest.exists():
                            os.link(temporary, dest)
                    except FileExistsError:
                        pass
                    finally:
                        temporary.unlink(missing_ok=True)
    for folder in ('models', 'data', 'reports', 'config'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    os.environ['SIGNLENS_HOME'] = str(root)
    os.environ['SIGNLENS_STORAGE_FALLBACK'] = '1' if fallback else '0'
    return root
