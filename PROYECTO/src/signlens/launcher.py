"""One entry point for source, GUI and frozen training subprocesses."""
import os
import sys
import traceback
from pathlib import Path
from .paths import prepare_storage
from .reporting import report_path


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    root = prepare_storage()
    log_path = report_path('application', '.log', root)
    cli = any(a in sys.argv for a in ('--train', '--evaluate', '--extract', '--self-check', '--tracker-check', '--benchmark', '--probe-camera', '--help'))
    # Keep console/pipe support for training. Hide only a console owned by this GUI process.
    if getattr(sys, 'frozen', False) and os.name == 'nt' and not cli:
        import ctypes
        owners = (ctypes.c_ulong * 2)()
        if ctypes.windll.kernel32.GetConsoleProcessList(owners, 2) == 1:
            ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
    with log_path.open('w', encoding='utf-8', buffering=1) as log:
        class Tee:
            def __init__(self, original): self.original = original
            def write(self, value):
                log.write(value)
                if self.original: self.original.write(value)
                return len(value)
            def flush(self):
                log.flush()
                if self.original: self.original.flush()
            def isatty(self): return False
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(stdout), Tee(stderr)
        try:
            print(f'SignLens: datos y reportes en {root}', flush=True)
            if '--self-check' in sys.argv:
                from .diagnostics import check
                check(root)
            elif '--tracker-check' in sys.argv:
                import numpy as np
                from .tracker import Tracker
                from .config import Config
                from .reporting import write_report
                for pose in ('lite', 'full'):
                    tracker = Tracker(Config(pose_model=pose))
                    try: tracker.process(np.zeros((480, 640, 3), dtype=np.uint8), 1000)
                    finally: tracker.close()
                write_report('native_tracker', {'status': 'PASS', 'pose_models': ['lite','full'], 'camera': 'NOT RUN'}, root)
            else:
                dispatch = {'--train': 'train', '--evaluate': 'validation', '--extract': 'dataset'}
                mode = next((a for a in dispatch if a in sys.argv), None)
                if mode:
                    import importlib
                    sys.argv.remove(mode)
                    importlib.import_module('signlens.' + dispatch[mode]).main()
                else:
                    from .app import main as app_main
                    app_main()
        except Exception:
            traceback.print_exc()
            if not cli and os.name == 'nt':
                import ctypes
                ctypes.windll.user32.MessageBoxW(None, f'No se pudo iniciar SignLens.\nDetalle: {log_path}', 'SignLens', 16)
            raise SystemExit(1)
        finally:
            sys.stdout, sys.stderr = stdout, stderr
