"""Portable source and Windows executable entry point."""
import os, sys, multiprocessing
from pathlib import Path
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "2")
if __name__ == "__main__":
    multiprocessing.freeze_support()
    from signlens.launcher import main
    main()
