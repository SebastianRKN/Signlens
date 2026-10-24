"""Rendering only. No state, no mediapipe."""
from __future__ import annotations

import cv2
import numpy as np

from signlens.skeleton import arm_chains
from signlens.types import FrameResult

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
              (18, 19), (19, 20), (0, 17)]
COLORS = {"Left": (255, 160, 0), "Right": (0, 200, 255)}  # BGR, per the user's real side


def _px(p: np.ndarray, w: int, h: int) -> tuple[int, int]:
    return int(p[0] * w), int(p[1] * h)


def draw(frame: np.ndarray, res: FrameResult, fps: float, recording: bool, min_vis: float = 0.5, info: str = "", caption: str = "") -> np.ndarray:
    h, w = frame.shape[:2]
    if res.pose is not None:
        for side, idx in arm_chains(res.pose, res.mirrored).items():
            color = COLORS[side]
            pts = res.pose[list(idx)]
            if pts[:, 3].min() < min_vis:
                continue
            xy = [_px(p, w, h) for p in pts]
            cv2.polylines(frame, [np.array(xy)], False, color, 4, cv2.LINE_AA)
            for q in xy:
                cv2.circle(frame, q, 6, color, -1, cv2.LINE_AA)
    for side, hand in res.hands.items():
        if hand is None:
            continue
        color = (140, 140, 140) if hand.stale else COLORS[side]  # grey = held through a dropout
        xy = [_px(p, w, h) for p in hand.pts]
        for a, b in HAND_EDGES:
            cv2.line(frame, xy[a], xy[b], color, 2, cv2.LINE_AA)
        for q in xy:
            cv2.circle(frame, q, 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, side, (xy[0][0] - 20, xy[0][1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f"{fps:4.1f} FPS", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if caption:
        cv2.putText(frame, caption, (16, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 8, cv2.LINE_AA)
        cv2.putText(frame, caption, (16, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 3, cv2.LINE_AA)
    for i, line in enumerate(info.split("\n") if info else []):
        cv2.putText(frame, line, (12, 54 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    if recording:
        cv2.circle(frame, (w - 28, 24), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 90, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame
