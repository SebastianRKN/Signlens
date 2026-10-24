"""Downloads and caches MediaPipe model files on first run."""
from __future__ import annotations

import urllib.request
from pathlib import Path

_BASE = "https://storage.googleapis.com/mediapipe-models"
MODELS = {
    "hand": f"{_BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_lite": f"{_BASE}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_full": f"{_BASE}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
}


def ensure_model(name: str, models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / Path(MODELS[name]).name
    if not path.exists():
        print(f"[signlens] downloading {name} model -> {path}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODELS[name], tmp)
        tmp.rename(path)
    return path
