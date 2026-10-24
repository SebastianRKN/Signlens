"""One Euro filter (Casiez et al., CHI 2012): adaptive low-pass that kills jitter
at rest while staying responsive during fast motion - ideal for signing."""
from __future__ import annotations

import math

import numpy as np


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    @staticmethod
    def _alpha(cutoff, dt: float):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x = self._dx = self._t = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self._x is None or self._t is None or t <= self._t:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return x
        dt = t - self._t
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        self._t = t
        return self._x
