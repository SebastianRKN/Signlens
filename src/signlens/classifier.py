"""Inference: single-hand letter prediction + temporal stabilisation for live use."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import time

import numpy as np

from signlens.features import static_features

DEMO_LABELS = tuple("ABDEGKLOST")
DEMO_THRESHOLD = 0.8


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


class SignClassifier:
    def __init__(self, bundle: dict):
        self.model, self.mode, self.classes = bundle["model"], bundle["mode"], bundle["classes"]
        self.threshold = float(bundle.get("threshold", 0.8))
        self.meta = bundle.get("meta", {})

    @classmethod
    def load(cls, path: str) -> "SignClassifier":
        import joblib

        return cls(joblib.load(path))

    def predict(self, world: np.ndarray, threshold: float | None = None,
                allowed: tuple[str, ...] | None = None) -> Prediction | None:
        if world.shape != (21, 3) or not np.isfinite(world).all():
            return None
        if np.linalg.norm(world[9] - world[0]) < 1e-6:
            return None
        proba = self.model.predict_proba(static_features(world[None], self.mode))[0]
        i = int(proba.argmax())
        label = str(self.classes[int(self.model.classes_[i])])
        # Reject outside vocabulary after predicting all classes; never force a demo letter.
        if proba[i] < (self.threshold if threshold is None else threshold) or (allowed and label not in allowed):
            return None
        return Prediction(label, float(proba[i]))


class Stabilizer:
    """Sliding-window vote. A letter is emitted only if enough recent frames agree with enough
    confidence, which removes single-frame flicker between similar signs."""

    def __init__(self, window: int = 7, min_share: float = 0.6, min_conf: float = 0.5,
                 hold_seconds: float = 0.0, max_gap: float = 0.35):
        self.window, self.min_share, self.min_conf = window, min_share, min_conf
        self._buf: deque[Prediction | None] = deque(maxlen=window)
        self.hold_seconds, self.max_gap = hold_seconds, max_gap
        self._candidate, self._since, self._last = None, 0.0, None

    def reset(self) -> None:
        self._buf.clear()
        self._candidate, self._since, self._last = None, 0.0, None

    def update(self, pred: Prediction | None, now: float | None = None) -> Prediction | None:
        now = time.monotonic() if now is None else now
        if self._last is not None and (now - self._last > self.max_gap or now < self._last):
            self.reset()
        if pred is None or pred.confidence < self.min_conf:
            self.reset()
            return None
        self._last = now
        if pred.label != self._candidate:
            self._candidate, self._since = pred.label, now
        self._buf.append(pred)
        votes: dict[str, list[float]] = defaultdict(list)
        for p in self._buf:
            if p is not None and p.confidence >= self.min_conf:
                votes[p.label].append(p.confidence)
        if not votes:
            return None
        label, confs = max(votes.items(), key=lambda kv: len(kv[1]))
        if (label != pred.label or len(confs) / self.window < self.min_share
                or now - self._since < self.hold_seconds):
            return None
        return Prediction(label, float(np.mean(confs)))
