"""Launch with explicit project-local data and models, independent of working directory."""
import sys
import os
import traceback
from pathlib import Path
from functools import partial
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
# Avoid CPU oversubscription while the camera and training process run together.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "2")

if __name__ == "__main__":
    try:
        if "--self-check" in sys.argv:
            from signlens.diagnostics import check
            check(ROOT)
        else:
            import signlens.app as app
            app.Config = partial(app.Config, models_dir=ROOT / "models", data_dir=ROOT / "data")
            app.main()
    except Exception:
        detail = traceback.format_exc()
        (ROOT / "app_error.log").write_text(detail, encoding="utf-8")
        print(detail, file=sys.stderr)
        sys.exit(1)
