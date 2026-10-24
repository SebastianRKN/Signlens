"""Hand identity, outlier rejection, dropout hold-over and smoothing.
Pure numpy (no mediapipe), so every rule is unit-tested without a camera.

Why: MediaPipe's per-frame Left/Right label flips when hands cross or overlap, and a single bad
frame (blur, partial occlusion) produces a garbage hand. For sign classification, one glitchy
frame in a 30-frame window is noise the model must learn to ignore - better to fix it upstream."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from signlens.config import Config
from signlens.skeleton import arm_chains
from signlens.smoothing import OneEuroFilter
from signlens.types import HANDS, Hand

_UNKNOWN_COST = 0.25  # distance charged when a side has no anchor (no arm seen, no previous track)


@dataclass(frozen=True)
class Detection:
    pts: np.ndarray  # (21, 3)
    world: np.ndarray  # (21, 3)
    score: float
    label: str  # user-side label already corrected for mirroring


def _xy(p: np.ndarray, aspect: float) -> np.ndarray:
    return np.array([p[0] * aspect, p[1]])  # isotropic image coordinates


def _palm(pts: np.ndarray, aspect: float) -> float:
    return float(np.linalg.norm(_xy(pts[9], aspect) - _xy(pts[0], aspect)))


class _Track:
    def __init__(self, cfg: Config):
        self.f_img = OneEuroFilter(cfg.hand_min_cutoff, cfg.hand_beta)
        self.f_world = OneEuroFilter(cfg.hand_min_cutoff, cfg.hand_beta)
        self.hand: Hand | None = None
        self.t = 0.0
        self.missed = 0
        self.vel = np.zeros(2)  # wrist velocity (isotropic image units / s)

    def drop(self) -> None:
        self.f_img.reset()
        self.f_world.reset()
        self.hand, self.missed = None, 0
        self.vel = np.zeros(2)


class HandResolver:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._tracks = {s: _Track(cfg) for s in HANDS}
        self.stats: Counter = Counter()

    def has_tracks(self) -> bool:
        return any(tr.hand is not None for tr in self._tracks.values())

    def update(self, dets: list[Detection], pose: np.ndarray | None, aspect: float, t: float) -> dict[str, Hand | None]:
        self.stats["frames"] += 1
        assigned = self._assign(dets, pose, aspect, t)
        out: dict[str, Hand | None] = {}
        for side in HANDS:
            tr, det = self._tracks[side], assigned.get(side)
            if det is not None and self._plausible(tr, det, aspect, t):
                out[side] = self._accept(side, tr, det, t, aspect)
            else:
                if det is not None:
                    self.stats["outliers"] += 1
                out[side] = self._hold(side, tr)
        return out

    # -- identity -------------------------------------------------------------------------
    def _assign(self, dets, pose, aspect, t) -> dict[str, Detection]:
        cfg = self.cfg
        chains = arm_chains(pose, cfg.mirror) if pose is not None else None
        anchors: dict[str, list[np.ndarray]] = {}
        for side in HANDS:
            pts = []
            if chains is not None:
                wrist = chains[side][2]
                if pose[wrist, 3] >= cfg.min_pose_visibility:
                    pts.append(_xy(pose[wrist], aspect))  # the arm's wrist: where this hand must be
            tr = self._tracks[side]
            if tr.hand is not None:
                # continuity: where this hand should be now, extrapolating its recent velocity
                # (keeps identities straight when the two hands cross)
                pts.append(_xy(tr.hand.pts[0], aspect) + tr.vel * min(max(t - tr.t, 0.0), 0.15))
            anchors[side] = pts

        def cost(d: Detection, side: str) -> float:
            w = _xy(d.pts[0], aspect)
            # A clear anatomical wrist match must override a previously swapped
            # track: taking min(pose, history) otherwise preserves the wrong side.
            if chains is not None:
                wrists = {s: pose[chains[s][2]] for s in HANDS}
                if all(p[3] >= .75 for p in wrists.values()):
                    distances = {s: float(np.linalg.norm(w - _xy(p, aspect))) for s, p in wrists.items()}
                    nearest = min(distances, key=distances.get)
                    other = "Right" if nearest == "Left" else "Left"
                    if distances[nearest] < .15 and distances[other] - distances[nearest] > .12:
                        return distances[side]
            dist = min((float(np.linalg.norm(w - a)) for a in anchors[side]), default=_UNKNOWN_COST)
            return dist + (0.0 if d.label == side else cfg.label_penalty)

        dets = dets[:2]
        if not dets:
            return {}
        if len(dets) == 1:
            result = {min(HANDS, key=lambda s: cost(dets[0], s)): dets[0]}
        else:
            a, b = dets
            straight = cost(a, "Left") + cost(b, "Right")
            swapped = cost(a, "Right") + cost(b, "Left")
            result = {"Left": a, "Right": b} if straight <= swapped else {"Left": b, "Right": a}
        self.stats["relabeled"] += sum(d.label != side for side, d in result.items())
        return result

    # -- validation, smoothing, hold ------------------------------------------------------
    def _plausible(self, tr: _Track, det: Detection, aspect: float, t: float) -> bool:
        if tr.hand is None:
            return True
        cfg, prev = self.cfg, tr.hand.pts
        dt = max(t - tr.t, 1e-3)
        jump = float(np.linalg.norm(_xy(det.pts[0], aspect) - _xy(prev[0], aspect)))
        if jump > max(cfg.min_jump, cfg.max_speed * dt):
            return False
        ratio = (_palm(det.pts, aspect) + 1e-6) / (_palm(prev, aspect) + 1e-6)
        return 1 / cfg.max_size_ratio <= ratio <= cfg.max_size_ratio

    def _accept(self, side: str, tr: _Track, det: Detection, t: float, aspect: float = 1.0) -> Hand:
        pts, world = det.pts, det.world
        if tr.hand is not None:
            v = (_xy(det.pts[0], aspect) - _xy(tr.hand.pts[0], aspect)) / max(t - tr.t, 1e-3)
            cap = self.cfg.max_speed
            tr.vel = np.clip(0.5 * tr.vel + 0.5 * v, -cap, cap)
        if self.cfg.smoothing:
            pts, world = tr.f_img(pts, t), tr.f_world(world, t)
        tr.hand = Hand(pts.copy(), world.copy(), det.score, stale=False)
        tr.t, tr.missed = t, 0
        self.stats[f"{side}_fresh"] += 1
        return tr.hand

    def _hold(self, side: str, tr: _Track) -> Hand | None:
        if tr.hand is None:
            return None
        tr.missed += 1
        if tr.missed <= self.cfg.hold_frames:
            self.stats[f"{side}_held"] += 1
            return replace(tr.hand, stale=True)
        self.stats[f"{side}_lost"] += 1
        tr.drop()
        return None
