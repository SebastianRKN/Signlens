import sys
from pathlib import Path

import numpy as np
import pytest

import signlens.capture as cap
from signlens.config import Config
from signlens.hands import Detection, HandResolver


# ---------------------------------------------------------------- camera probing
class _FakeDriver:
    """Windows-like: DirectShow honours MJPG/60, Media Foundation silently runs 30 FPS."""
    opened: list = []
    dshow_works = True

    def __init__(self, index, api):
        self.api, self.rate = api, (60 if api == cap.cv2.CAP_DSHOW else 30)
        self._ok = api != cap.cv2.CAP_DSHOW or _FakeDriver.dshow_works
        _FakeDriver.opened.append(api)

    def isOpened(self): return self._ok
    def set(self, *a): return True
    def get(self, prop):
        return {cap.cv2.CAP_PROP_FPS: float(self.rate), cap.cv2.CAP_PROP_FRAME_WIDTH: 640.0,
                cap.cv2.CAP_PROP_FRAME_HEIGHT: 480.0}.get(prop, 0.0)
    def read(self): return True, np.zeros((480, 640, 3), np.uint8)
    def release(self): pass


def _patch(monkeypatch, dshow_works=True):
    _FakeDriver.opened, _FakeDriver.dshow_works = [], dshow_works
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(cap.cv2, "VideoCapture", _FakeDriver)
    monkeypatch.setattr(cap, "measure_fps", lambda c, seconds=0.7, warmup=8: float(c.rate))


def test_probe_picks_the_backend_that_really_delivers_60(monkeypatch, tmp_path):
    _patch(monkeypatch)
    _, pr = cap.open_camera(Config(models_dir=tmp_path))
    assert pr.backend == "DSHOW" and pr.measured_fps == 60
    assert _FakeDriver.opened == [cap.cv2.CAP_DSHOW]  # stopped at the first good mode


def test_probe_falls_back_when_best_backend_cannot_open(monkeypatch, tmp_path):
    _patch(monkeypatch, dshow_works=False)
    _, pr = cap.open_camera(Config(models_dir=tmp_path))
    assert pr.backend == "MSMF" and pr.measured_fps == 30


def test_cached_mode_skips_the_search(monkeypatch, tmp_path):
    _patch(monkeypatch)
    cfg = Config(models_dir=tmp_path)
    cap.open_camera(cfg)
    _FakeDriver.opened.clear()
    _, pr = cap.open_camera(cfg)
    assert pr.backend == "DSHOW" and len(_FakeDriver.opened) == 1
    cap.open_camera(cfg, reprobe=True)
    assert len(_FakeDriver.opened) == 2


def test_probe_report_explains_a_slow_camera():
    text = cap.format_probe([cap.Probe("MSMF", (640, 480), 60, 30.0, 29.8)], 60)
    assert "29.8" in text and "low light" in text


# ---------------------------------------------------------------- pose staleness
def test_stale_pose_is_discarded():
    from signlens.tracker import _PoseWorker

    class _Lm:
        def detect_for_video(self, *a): raise AssertionError("not used")

    w = _PoseWorker(_Lm(), Config())
    try:
        pose = np.ones((33, 4))
        w._latest, w._latest_ts = pose, 1000
        assert w.latest(1200, 250) is pose
        assert w.latest(1400, 250) is None
    finally:
        w.close()


# ---------------------------------------------------------------- hands crossing
def _det(x, y, label):
    pts = np.tile(np.array([x, y, 0.0], dtype=np.float32), (21, 1))
    pts[9, 1] -= 0.1
    return Detection(pts, pts.copy(), 0.9, label)


def test_identities_survive_crossing_even_when_mediapipe_labels_flip():
    r = HandResolver(Config())
    t, out = 0.0, None
    for i in range(24):  # left hand sweeps right, right hand sweeps left; they cross in the middle
        xl, xr = 0.30 + 0.015 * i, 0.70 - 0.015 * i
        crossed = xl > xr
        la, lb = ("Right", "Left") if crossed else ("Left", "Right")  # mediapipe swaps labels after crossing
        out = r.update([_det(xl, 0.50, la), _det(xr, 0.60, lb)], None, 1.0, t)
        t += 0.02
    assert out["Left"].pts[0, 0] > 0.55 and out["Right"].pts[0, 0] < 0.45
    assert out["Left"].pts[0, 1] < out["Right"].pts[0, 1]


def test_tracker_skips_palm_detection_on_idle_frames(tmp_path):
    models = next((p for p in (Path(__file__).resolve().parents[1] / "models", Path.home() / ".signlens" / "models") if (p / "hand_landmarker.task").exists()), Path("none"))
    if not (models / "hand_landmarker.task").exists() or not (models / "pose_landmarker_lite.task").exists():
        pytest.skip("MediaPipe model files not present (they download on first run)")
    from signlens.tracker import Tracker

    tr = Tracker(Config(models_dir=models, idle_stride=2))
    calls, real = [], tr._hands.detect_for_video
    tr._hands = type("W", (), {"detect_for_video": lambda s, img, ts: (calls.append(ts), real(img, ts))[1],
                               "close": lambda s: real.__self__.close()})()
    for i in range(10):
        tr.process(np.zeros((240, 320, 3), np.uint8), 1000 + i * 17)
    tr.close()
    assert len(calls) == 5
# Regression: a real webcam may not allow a second handle while the first is open.
def test_slow_camera_probe_uses_only_one_open_handle(monkeypatch, tmp_path):
    _patch(monkeypatch)
    class Exclusive(_FakeDriver):
        active=0
        def __init__(self,index,api):
            assert Exclusive.active==0, 'concurrent camera handles'
            super().__init__(index,api)
            self.rate=30
            Exclusive.active+=1
        def release(self):
            if self._ok:
                Exclusive.active-=1
                self._ok=False
    monkeypatch.setattr(cap.cv2,'VideoCapture',Exclusive)
    camera,pr=cap.open_camera(Config(models_dir=tmp_path))
    assert pr.measured_fps==30 and Exclusive.active==1
    camera.release()
    assert Exclusive.active==0

def test_zero_frame_modes_are_not_selected(monkeypatch,tmp_path):
    _patch(monkeypatch)
    monkeypatch.setattr(cap,'measure_fps',lambda *a,**k:0.)
    with pytest.raises(RuntimeError,match='did not deliver'):
        cap.open_camera(Config(models_dir=tmp_path))

def test_tracker_normalizes_duplicate_millisecond_timestamps():
    from signlens.tracker import Tracker
    models=Path(__file__).resolve().parents[1]/'models'
    if not (models/'hand_landmarker.task').exists():
        pytest.skip('model files unavailable')
    tr=Tracker(Config(models_dir=models,idle_stride=1))
    try:
        frame=np.zeros((240,320,3),np.uint8)
        results=[tr.process(frame,1000).timestamp_ms for _ in range(3)]
        assert results==[1000,1001,1002]
    finally:
        tr.close()
