"""Records labelled feature sequences -> data/<label>/<timestamp>.npz with x (T, FEATURE_DIM) and feature_version.
This is the dataset format for training the sign classifier in the next phase."""
from __future__ import annotations

import time
import re
import uuid
from pathlib import Path

import numpy as np

from signlens.features import FEATURE_VERSION


def safe_id(value: str) -> str:
    """Identifiers are directory components, never paths."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", value) or value.upper() in {
        "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        raise ValueError("Usa 1-48 letras sin tildes, numeros, guion o guion bajo.")
    return value


def new_session(signer: str) -> str:
    return f"{safe_id(signer)}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


class SequenceRecorder:
    def __init__(self, data_dir: Path, label: str):
        self.dir = data_dir / label
        self._buf: list[np.ndarray] = []
        self.active = False

    def toggle(self) -> Path | None:
        if not self.active:
            self._buf.clear()
            self.active = True
            return None
        self.active = False
        if len(self._buf) < 5:  # discard accidental taps
            return None
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}.npz"
        np.savez_compressed(path, x=np.stack(self._buf), feature_version=FEATURE_VERSION)
        return path

    def add(self, features: np.ndarray) -> None:
        if self.active:
            self._buf.append(features)


class StaticSampleRecorder:
    """Records single-frame hand shapes of YOUR signing for training/evaluation.
    Output: <root>/<signer>/<label>_<session>_<n>.npz, directly consumable by `train --extra`.

    Frames of one burst are near-duplicates, so they share a `group`; one app run is one `session`.
    Record every letter in several separate sittings (other day, light, background) - sessions are
    what make the evaluation honest."""

    def __init__(self, root: Path, label: str, signer: str, every: int = 3, session: str | None = None):
        self.root, self.label, self.signer, self.every = Path(root), safe_id(label), safe_id(signer), max(every, 1)
        self.session = session or new_session(signer)
        self.active, self._blocks, self._i = False, 0, 0
        self._world: list[np.ndarray] = []
        self._pts: list[np.ndarray] = []
        self._score: list[float] = []
        self._times: list[float] = []

    def toggle(self) -> Path | None:
        if not self.active:
            self.active, self._i = True, 0
            self._world, self._pts, self._score, self._times = [], [], [], []
            return None
        if len(self._world) < 5:
            self.active = False
            return None
        self._blocks += 1
        n = len(self._world)
        (self.root / self.signer).mkdir(parents=True, exist_ok=True)
        # A recorder is recreated for each click: a per-object counter alone overwrote prior takes.
        block = uuid.uuid4().hex
        path = self.root / self.signer / f"{self.label}_{block}.npz"
        pending = path.with_suffix(".pending")
        with pending.open("xb") as stream:
            np.savez_compressed(
                stream, world=np.stack(self._world).astype(np.float32), pts=np.stack(self._pts).astype(np.float32),
                score=np.array(self._score, dtype=np.float32), y=np.array([self.label] * n),
                group=np.array([block] * n), captured_at=np.array(self._times),
                session=np.array([self.session] * n), signer=np.array([self.signer] * n),
                domain=np.array(["webcam"] * n), path=np.array([str(path)] * n))
        pending.replace(path)
        self.active = False
        return path

    def add(self, hand) -> None:
        """hand: a fresh (non-stale) Hand, or None."""
        if not self.active or hand is None or hand.stale:
            return
        if (hand.world.shape != (21, 3) or hand.pts.shape != (21, 3)
                or not np.isfinite(hand.world).all() or not np.isfinite(hand.pts).all()):
            return
        self._i += 1
        if self._i % self.every == 0:
            self._world.append(hand.world.copy())
            self._pts.append(hand.pts.copy())
            self._score.append(hand.score)
            self._times.append(time.monotonic())

    @property
    def count(self) -> int:
        return len(self._world)
