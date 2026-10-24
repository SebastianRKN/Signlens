import numpy as np

from signlens.config import Config
from signlens.features import ARMS_AT, FEATURE_DIM, FLAGS_AT, HAND_DIM, HAND_SHAPE, extract
from signlens.hands import Detection, HandResolver
from signlens.skeleton import arm_chains
from signlens.smoothing import OneEuroFilter
from signlens.types import FrameResult, Hand


def _hand(seed=0, wrist=(0.5, 0.5)):
    r = np.random.default_rng(seed)
    pts = r.random((21, 3)).astype(np.float32) * 0.1 + np.array([*wrist, 0], dtype=np.float32)
    pts[0] = [*wrist, 0]
    return Hand(pts, r.random((21, 3)).astype(np.float32) * 0.1)


def _pose(left_x=0.6, right_x=0.4):
    p = np.zeros((33, 4), dtype=np.float32)
    p[:, 3] = 1.0
    for i, x in ((11, left_x), (12, right_x)):
        p[i], p[i + 2], p[i + 4] = [x, 0.4, 0, 1], [x, 0.6, 0, 1], [x, 0.8, 0, 1]
    return p


def _res(hands, pose=None):
    return FrameResult(0, 1.0, {"Left": None, "Right": None, **hands}, pose)


def _det(wrist, label="Left", palm=0.1):
    pts = np.tile(np.array([*wrist, 0], dtype=np.float32), (21, 1))
    pts[9, 1] -= palm
    return Detection(pts, pts.copy(), 0.9, label)


# ---------------- features ----------------
def test_shape_and_flags():
    v = extract(_res({"Left": _hand()}))
    assert v.shape == (FEATURE_DIM,)
    assert (v[FLAGS_AT], v[FLAGS_AT + 1], v[FLAGS_AT + 2]) == (1, 0, 0)


def test_hand_shape_invariant_to_scale_and_image_position():
    h = _hand()
    h2 = Hand(h.pts + 0.2, h.world * 0.5)
    np.testing.assert_allclose(extract(_res({"Left": h}))[:HAND_SHAPE],
                               extract(_res({"Left": h2}))[:HAND_SHAPE], atol=1e-4)


def test_empty_frame_is_zeros():
    assert not extract(_res({})).any()


def test_hand_location_relative_to_body():
    v = extract(_res({"Left": _hand(wrist=(0.4, 0.8))}, _pose()))
    # shoulder midpoint (0.5, 0.4), shoulder width 0.2 -> ((0.4-.5)/.2, (0.8-.4)/.2, 0)
    np.testing.assert_allclose(v[HAND_SHAPE:HAND_DIM], [-0.5, 2.0, 0.0], atol=1e-5)


def test_arm_sides_follow_screen_position_not_model_labels():
    assert arm_chains(_pose(0.6, 0.4))["Left"] == (12, 14, 16)
    assert arm_chains(_pose(0.4, 0.6))["Left"] == (11, 13, 15)
    assert arm_chains(_pose(0.4, 0.6), mirrored=False)["Left"] == (12, 14, 16)


def test_features_identical_when_model_labels_swap():
    p1 = _pose(0.6, 0.4)
    p2 = p1.copy()
    for a, b in ((11, 12), (13, 14), (15, 16)):
        p2[[a, b]] = p2[[b, a]]
    f1, f2 = extract(_res({}, p1)), extract(_res({}, p2))
    np.testing.assert_allclose(f1, f2)
    assert f1[FLAGS_AT + 2] == 1 and f1[ARMS_AT:FLAGS_AT].any()


# ---------------- hand resolver ----------------
def test_wrong_mediapipe_label_is_corrected_by_arm_position():
    r = HandResolver(Config())
    out = r.update([_det((0.6, 0.8), "Left")], _pose(), 1.0, 0.0)  # near the user's RIGHT wrist
    assert out["Right"] is not None and out["Left"] is None
    assert r.stats["relabeled"] == 1


def test_two_hands_keep_identity_when_both_labels_are_wrong():
    r = HandResolver(Config())
    out = r.update([_det((0.4, 0.8), "Right"), _det((0.6, 0.8), "Left")], _pose(), 1.0, 0.0)
    np.testing.assert_allclose(out["Left"].pts[0, :2], (0.4, 0.8), atol=1e-5)
    np.testing.assert_allclose(out["Right"].pts[0, :2], (0.6, 0.8), atol=1e-5)


def test_dropout_is_held_then_dropped():
    cfg = Config()
    r = HandResolver(cfg)
    r.update([_det((0.5, 0.5))], None, 1.0, 0.0)
    for i in range(1, cfg.hold_frames + 1):
        out = r.update([], None, 1.0, i * 0.02)
        assert out["Left"] is not None and out["Left"].stale
    assert r.update([], None, 1.0, 1.0)["Left"] is None
    assert r.stats["Left_lost"] == 1


def test_teleporting_and_resized_detections_are_rejected():
    r = HandResolver(Config())
    r.update([_det((0.3, 0.5))], None, 1.0, 0.0)
    out = r.update([_det((0.6, 0.5))], None, 1.0, 0.02)  # 0.3 jump in 20 ms
    assert out["Left"].stale
    np.testing.assert_allclose(out["Left"].pts[0, :2], (0.3, 0.5), atol=1e-5)
    out = r.update([_det((0.3, 0.5), palm=0.5)], None, 1.0, 0.04)  # palm 5x bigger
    assert out["Left"].stale and r.stats["outliers"] == 2


# ---------------- smoothing ----------------
def test_one_euro_reduces_jitter_at_rest():
    rng = np.random.default_rng(1)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    raw = rng.normal(0, 0.01, (200, 3))
    out = np.array([f(x, i / 30) for i, x in enumerate(raw)])
    assert out[20:].std() < raw[20:].std()


def test_one_euro_beta_removes_lag_in_fast_motion():
    def final_error(beta):
        f = OneEuroFilter(min_cutoff=1.2, beta=beta)
        for i in range(100):
            y = f(np.array([i / 50]), i / 50)  # 1 unit/s ramp at 50 FPS
        return 99 / 50 - float(y[0])

    assert final_error(10.0) < final_error(0.05) / 3
