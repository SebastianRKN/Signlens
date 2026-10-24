"""Loads the local MediaPipe models shipped with this distribution."""
from __future__ import annotations


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
    if not path.is_file():
        raise FileNotFoundError(f"Falta el modelo local {path.name}. Extrae de nuevo la distribucion completa; no se descargan modelos al iniciar.")
    return path
