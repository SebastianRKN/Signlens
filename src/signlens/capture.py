"""Threaded camera that measures the FPS it actually gets.

Why probing: drivers report what they were asked for, not what they deliver. On Windows the default
OpenCV backend (Media Foundation) often ignores MJPG/60 FPS and silently runs 30 FPS YUY2, while
DirectShow honours the request. We try backend/mode combinations, time real frame delivery, keep the
fastest and cache the winner so later launches are instant."""
from __future__ import annotations

import json
from collections import deque
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from signlens.config import Config


@dataclass
class Probe:
    backend: str
    size: tuple[int, int]
    requested_fps: int
    reported_fps: float
    measured_fps: float


def backends() -> list[tuple[str, int]]:
    if sys.platform.startswith("win"):
        return [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]
    if sys.platform == "darwin":
        return [("AVFOUNDATION", cv2.CAP_AVFOUNDATION)]
    return [("V4L2", cv2.CAP_V4L2), ("ANY", cv2.CAP_ANY)]


def candidate_modes(cfg: Config) -> list[tuple[int, int, int]]:
    modes = [(cfg.width, cfg.height, cfg.fps), (640, 360, 60), (1280, 720, 60), (640, 480, 30)]
    return list(dict.fromkeys(modes))


def configure(cap: cv2.VideoCapture, w: int, h: int, fps: int) -> None:
    # Order matters: the codec first (DirectShow/V4L2 only offer high FPS in MJPG), then size, then rate.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def measure_fps(cap: cv2.VideoCapture, seconds: float = 0.7, warmup: int = 8) -> float:
    """Frames actually delivered per second (read() blocks until a new frame arrives)."""
    if seconds <= 0:
        return 0.0
    for _ in range(warmup):  # flush buffered frames so they don't inflate the rate
        cap.read()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        n += bool(cap.read()[0])
    dt = time.perf_counter() - t0
    return n / dt if dt > 0 else 0.0


def _try(index: int, name: str, api: int, mode: tuple[int, int, int], seconds: float):
    cap = cv2.VideoCapture(index, api)
    if not cap.isOpened():
        cap.release()
        return None, None
    w, h, fps = mode
    configure(cap, w, h, fps)
    measured = measure_fps(cap, seconds)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    return cap, Probe(name, size, fps, float(cap.get(cv2.CAP_PROP_FPS)), measured)


def probe_all(cfg: Config, seconds: float = 0.7) -> list[Probe]:
    """Every backend x mode, for the --probe-camera report. Releases everything it opens."""
    out = []
    for name, api in backends():
        for mode in candidate_modes(cfg):
            cap, pr = _try(cfg.camera_index, name, api, mode, seconds)
            if cap is None:
                break  # this backend cannot open the camera at all
            cap.release()
            out.append(pr)
    return out


def _cache_path(cfg: Config) -> Path:
    return Path(cfg.models_dir) / f"camera_{cfg.camera_index}.json"


def open_camera(cfg: Config, reprobe: bool = False, accept_ratio: float = 0.85):
    """Return (cap, Probe). Fast path: replay the cached winner if it still performs."""
    cache = _cache_path(cfg)
    if not reprobe and cache.exists():
        try:
            c = json.loads(cache.read_text())
            names = dict(backends())
            cap, pr = _try(cfg.camera_index, c["backend"], names[c["backend"]], (c["w"], c["h"], c["fps"]), 0.5)
            if cap is not None and pr.measured_fps > 0 and pr.measured_fps >= accept_ratio * c["measured"]:
                return cap, pr
            if cap is not None:
                cap.release()
        except (OSError, ValueError, KeyError):
            pass
    # Probe with exclusive ownership: never open a second handle to the same device.
    candidates = []
    best_cap, best = None, None
    for name, api in backends():
        for mode in candidate_modes(cfg):
            cap, pr = _try(cfg.camera_index, name, api, mode, 0.7)
            if cap is None:
                break
            if pr.measured_fps >= accept_ratio * cfg.fps and pr.measured_fps > 0:
                best_cap, best = cap, pr
                break
            cap.release()
            if pr.measured_fps > 0:
                candidates.append((pr, mode))
        if best_cap is not None:
            break
    if best_cap is None:
        # Reopen the winner only after every competing probe is closed.
        fastest = max((pr.measured_fps for pr, _ in candidates), default=0)
        # Short probe differences below 5% are noise; prefer the requested resolution.
        candidates.sort(key=lambda item: (
            item[0].measured_fps < fastest * 0.95,
            item[0].size != (cfg.width, cfg.height),
            item[0].size[0] * item[0].size[1],
            -item[0].measured_fps,
        ))
        for candidate, mode in candidates:
            cap, pr = _try(cfg.camera_index, candidate.backend, dict(backends())[candidate.backend], mode, 0.7)
            if cap is not None and pr.measured_fps > 0:
                best_cap, best = cap, pr
                break
            if cap is not None:
                cap.release()
    if best_cap is None:
        raise RuntimeError(f"Camera {cfg.camera_index} did not deliver frames")

    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"backend": best.backend, "w": best.size[0], "h": best.size[1],
                                     "fps": best.requested_fps, "measured": best.measured_fps}))
    except OSError:
        pass
    return best_cap, best


class Camera:
    def __init__(self, cfg: Config, reprobe: bool = False):
        if cfg.camera_probe:
            self._cap, self.info = open_camera(cfg, reprobe)
        else:
            name, api = backends()[0]
            self._cap, self.info = _try(cfg.camera_index, name, api, (cfg.width, cfg.height, cfg.fps), 0.0)
            if self._cap is None:
                raise RuntimeError(f"Cannot open camera {cfg.camera_index}")
        self.size = self.info.size
        self.backend = self.info.backend
        self.captured = 0
        self.error: str | None = None
        self._fps = 0.0
        self._mirror = cfg.mirror
        self._frame: np.ndarray | None = None
        self._new = threading.Event()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def fps(self) -> float:
        """Frames per second the camera is delivering right now (smoothed)."""
        return self._fps

    def _loop(self) -> None:
        fails = 0
        arrivals = deque(maxlen=120)
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                fails += 1
                time.sleep(0.005)
                if fails > 400:  # ~2 s without a frame: unplugged, or taken by another app
                    self.error = "camera stopped delivering frames"
                    self._new.set()
                    return
                continue
            fails = 0
            now = time.perf_counter()
            arrivals.append(now)
            while len(arrivals) > 2 and now - arrivals[0] > 2.0:
                arrivals.popleft()
            if len(arrivals) > 1:
                self._fps = (len(arrivals) - 1) / max(now - arrivals[0], 1e-6)
            if self._mirror:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._frame = frame
                self.captured += 1
                self._new.set()

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        if not self._new.wait(timeout):
            return None
        if self.error:
            raise RuntimeError(self.error)
        with self._lock:
            self._new.clear()
            return self._frame

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._cap.release()


def format_probe(results: list[Probe], target: int) -> str:
    rows = [f"{'backend':13} {'mode':>10} {'asked':>6} {'driver says':>12} {'REAL fps':>9}"]
    for p in results:
        rows.append(f"{p.backend:13} {p.size[0]:>5}x{p.size[1]:<4} {p.requested_fps:>6} {p.reported_fps:>12.0f} {p.measured_fps:>9.1f}")
    best = max(results, key=lambda p: p.measured_fps, default=None)
    if best is None:
        return "No camera could be opened."
    verdict = ("OK" if best.measured_fps >= 0.85 * target else
               "The camera/driver cannot deliver this rate: low light makes webcams lengthen exposure and "
               "halve FPS; also try a USB 3 port, closing other apps that use the camera, or another camera.")
    return "\n".join(rows) + f"\n\nbest: {best.backend} {best.size[0]}x{best.size[1]} -> {best.measured_fps:.1f} FPS (target {target}). {verdict}"
