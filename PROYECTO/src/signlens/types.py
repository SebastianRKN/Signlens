"""Plain data containers passed between pipeline stages (no logic, no mediapipe types)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HANDS = ("Left", "Right")  # from the USER's point of view


@dataclass(frozen=True)
class Hand:
    pts: np.ndarray  # (21, 3) image-normalized landmarks (x, y in 0..1)
    world: np.ndarray  # (21, 3) metric 3D landmarks in metres: true hand shape, scale-aware
    score: float = 1.0  # handedness confidence
    stale: bool = False  # True when held over from a previous frame


@dataclass(frozen=True)
class FrameResult:
    timestamp_ms: int
    aspect: float  # width / height, needed to make x/y isotropic
    hands: dict[str, Hand | None] = field(default_factory=dict)  # (21, 3) normalized
    pose: np.ndarray | None = None  # (33, 4): x, y, z, visibility
    mirrored: bool = True
