"""MediaPipe wrapper. Hands run in the caller's thread; pose runs in a background worker so the
slower pose model never blocks the hand loop (arms move slowly, hands need the high frame rate)."""
from __future__ import annotations

import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from signlens.config import Config
from signlens.hands import Detection, HandResolver
from signlens.models import ensure_model
from signlens.smoothing import OneEuroFilter
from signlens.types import FrameResult


def _to_array(landmarks, with_visibility: bool = False) -> np.ndarray:
    if with_visibility:
        return np.array([[p.x, p.y, p.z, p.visibility] for p in landmarks], dtype=np.float32)
    return np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)


def _ema(prev: float, new: float) -> float:
    return new if prev == 0 else 0.9 * prev + 0.1 * new


class _PoseWorker(threading.Thread):
    """Always processes the most recent submitted frame; older pending frames are dropped."""

    def __init__(self, landmarker, cfg: Config):
        super().__init__(daemon=True)
        self._lm, self._cfg = landmarker, cfg
        self._cv = threading.Condition()
        self._job: tuple[mp.Image, int] | None = None
        self._closed = False
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._latest_ts = -10**9
        self._filter = OneEuroFilter(cfg.pose_min_cutoff, cfg.pose_beta)
        self.ms = 0.0
        self.start()

    def submit(self, image: mp.Image, ts_ms: int) -> None:
        with self._cv:
            self._job = (image, ts_ms)
            self._cv.notify()

    def latest(self, now_ms: int, max_age_ms: int) -> np.ndarray | None:
        """Newest pose, or None if it is older than max_age_ms (stalled worker, person left)."""
        with self._lock:
            return self._latest if now_ms - self._latest_ts <= max_age_ms else None

    def run(self) -> None:
        while True:
            with self._cv:
                while self._job is None and not self._closed:
                    self._cv.wait()
                if self._closed:
                    return
                image, ts = self._job
                self._job = None
            t0 = time.perf_counter()
            res = self._lm.detect_for_video(image, ts)
            pose = None
            if res.pose_landmarks:
                pose = _to_array(res.pose_landmarks[0], with_visibility=True)
                if self._cfg.smoothing:
                    pose[:, :3] = self._filter(pose[:, :3], ts / 1000.0)
            else:
                self._filter.reset()
            self.ms = _ema(self.ms, (time.perf_counter() - t0) * 1000)
            with self._lock:
                self._latest, self._latest_ts = pose, ts

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify()
        self.join(timeout=2.0)


class Tracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        base = lambda n: mp_python.BaseOptions(model_asset_path=str(ensure_model(n, cfg.models_dir)))
        mode = vision.RunningMode.VIDEO
        self._hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=base("hand"), running_mode=mode, num_hands=cfg.num_hands,
                min_hand_detection_confidence=cfg.min_detection_conf,
                min_hand_presence_confidence=cfg.min_presence_conf,
                min_tracking_confidence=cfg.min_tracking_conf))
        pose_lm = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=base(f"pose_{cfg.pose_model}"), running_mode=mode,
                min_pose_detection_confidence=cfg.min_detection_conf,
                min_pose_presence_confidence=cfg.min_presence_conf,
                min_tracking_confidence=cfg.min_tracking_conf))
        self._pose_lm = pose_lm
        self._pose = _PoseWorker(pose_lm, cfg)
        self._resolver = HandResolver(cfg)
        self.hand_ms = 0.0
        self._last_timestamp_ms = -1
        self._idle = 0

    @property
    def stats(self):
        return self._resolver.stats

    @property
    def pose_ms(self) -> float:
        return self._pose.ms

    def process(self, bgr: np.ndarray, timestamp_ms: int) -> FrameResult:
        # Buffered camera frames can arrive within the same millisecond. MediaPipe
        # VIDEO requires strictly increasing timestamps, including the pose worker.
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        h, w = bgr.shape[:2]
        small = bgr
        if w > self.cfg.infer_width:
            small = cv2.resize(bgr, (self.cfg.infer_width, int(h * self.cfg.infer_width / w)),
                               interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mk = lambda: mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._pose.submit(mk(), timestamp_ms)  # async, non-blocking

        t0 = time.perf_counter()
        dets = []
        active = self._resolver.has_tracks()
        if active or self._idle % max(self.cfg.idle_stride, 1) == 0:
            hand_res = self._hands.detect_for_video(mk(), timestamp_ms)
            for lms, world, cats in zip(hand_res.hand_landmarks, hand_res.hand_world_landmarks, hand_res.handedness):
                label = cats[0].category_name  # MediaPipe assumes a mirrored image
                if not self.cfg.mirror:
                    label = "Right" if label == "Left" else "Left"
                dets.append(Detection(_to_array(lms), _to_array(world), float(cats[0].score), label))
            self.hand_ms = _ema(self.hand_ms, (time.perf_counter() - t0) * 1000)
        self._idle = 0 if active else self._idle + 1
        pose = self._pose.latest(timestamp_ms, self.cfg.pose_max_age_ms)
        hands = self._resolver.update(dets, pose, w / h, timestamp_ms / 1000.0)
        return FrameResult(timestamp_ms, w / h, hands, pose, self.cfg.mirror)

    def close(self) -> None:
        self._pose.close()
        self._hands.close()
        self._pose_lm.close()
