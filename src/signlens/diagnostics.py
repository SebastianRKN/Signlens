"""Local installation/model checks. Camera trials are a separate verification."""
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

from signlens.classifier import SignClassifier
from signlens.training_job import active_model, file_hash
from signlens.validation import dataset_inventory


def check(root: Path):
    root = Path(root)
    models = root / "models"
    required = [models / name for name in ("hand_landmarker.task", "pose_landmarker_lite.task", "landmarks.npz")]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Faltan archivos: " + ", ".join(missing))
    path = active_model(models)
    clf = SignClassifier.load(str(path))
    with np.load(models / "landmarks.npz", allow_pickle=True) as d:
        clf.predict(d["world"][0])
    result = {"status": "PASS", "python": sys.version.split()[0], "executable": sys.executable,
              "model": str(path.relative_to(root)), "model_sha256": file_hash(path),
              "model_date": clf.meta.get("trained_at"), "classes": list(map(str, clf.classes)),
              "packages": {name: importlib.metadata.version(name) for name in
                           ("numpy", "mediapipe", "opencv-python", "scikit-learn", "joblib")},
              "training_data": dataset_inventory(root / "data/static"),
              "reserved_data": dataset_inventory(root / "data/validation"),
              "camera_check": "NOT RUN"}
    if result["training_data"]["errors"] or result["reserved_data"]["errors"]:
        result["status"] = "FAIL"
    folder = root / "reports"
    folder.mkdir(exist_ok=True)
    (folder / "installation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise ValueError("Hay capturas ilegibles; revisa reports/installation.json.")
    return result
