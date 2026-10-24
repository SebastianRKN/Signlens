"""Leakage-aware evaluation splits.

Why this module exists: the dataset has ~3 augmented copies per photo AND photos taken seconds apart
by the same person in the same place. Splitting by photo (or by Roboflow's tiny train/valid/test
folders) lets near-identical samples sit on both sides, so accuracy looks great and means little.

  assign_sessions      photo names -> capture sessions (timestamp gaps / consecutive IMG numbers)
  class_session_folds  each class is tested only on sessions its training never saw"""
from __future__ import annotations

import datetime as dt
import re

import numpy as np

_IMG_TS = re.compile(r"IMG_(\d{8})_(\d{6})")
_WHATSAPP = re.compile(r"WhatsApp-Image-(\d{4})-(\d{2})-(\d{2})-at-(\d+)-(\d+)-(\d+)")
_SCREENSHOT = re.compile(r"Captura-de-pantalla-(\d{4})-(\d{2})-(\d{2})-(\d{2})(\d{2})(\d{2})")
_IMG_NUM = re.compile(r"IMG_(\d+)_(PNG|JPG|JPEG|HEIC)", re.I)


def _timestamp(name: str) -> dt.datetime | None:
    if m := _IMG_TS.match(name):
        try:
            return dt.datetime.strptime(m[1] + m[2], "%Y%m%d%H%M%S")
        except ValueError:
            return None
    if m := (_WHATSAPP.match(name) or _SCREENSHOT.match(name)):
        try:
            return dt.datetime(*map(int, m.groups()))
        except ValueError:
            return None
    return None


def assign_sessions(names, gap_s: float = 300.0, seq_gap: int = 5) -> np.ndarray:
    """Cluster photos into capture sessions: a new session starts after a `gap_s` pause in the
    timestamp in the file name, or after a jump of more than `seq_gap` in a camera's IMG_#### counter.
    Files with no recoverable timestamp each become their own session (never merged by guess)."""
    names = [str(n) for n in names]
    out: dict[str, str] = {}
    stamped, numbered = [], []
    for n in dict.fromkeys(names):
        if (t := _timestamp(n)) is not None:
            stamped.append((t, n))
        elif m := _IMG_NUM.match(n):
            numbered.append((int(m[1]), n))
        else:
            out[n] = f"single:{n}"
    prev, k = None, 0
    for t, n in sorted(stamped):
        k += prev is None or (t - prev).total_seconds() > gap_s
        out[n], prev = f"t{k}", t
    prev, k = None, 0
    for v, n in sorted(numbered):
        k += prev is None or v - prev > seq_gap
        out[n], prev = f"n{k}", v
    return np.array([out[n] for n in names])


def class_session_folds(labels, sessions, k: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Fold id per sample plus a mask of evaluable samples.

    Each class's sessions are dealt round-robin into k folds, so in fold f the test samples of a
    class come from sessions that class never shows in training. Classes seen in only one session
    cannot be tested this way; they stay in training and are excluded from the metrics."""
    labels, sessions = np.asarray(labels), np.asarray(sessions)
    rng = np.random.default_rng(seed)
    fold = np.full(len(labels), -1, dtype=int)
    evaluable = np.zeros(len(labels), dtype=bool)
    for c in np.unique(labels):
        in_c = labels == c
        ss = sorted(set(sessions[in_c]))
        if len(ss) < 2:
            continue  # Always in training, including fold zero.
        rng.shuffle(ss)
        for i, s in enumerate(ss):
            fold[in_c & (sessions == s)] = i % k
        evaluable[in_c] = len(ss) >= 2
    return fold, evaluable
