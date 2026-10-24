import numpy as np
import pytest

from signlens.features import (FEATURE_MODES, augment_world, canonical_shape, engineered_shape, mirror_world,
                               oriented_shape, static_features)
from signlens.recorder import StaticSampleRecorder
from signlens.types import Hand

pytest.importorskip("sklearn")
from signlens.train import (MODELS, evaluate_class_session, evaluate_grouped, feature_bank, load_landmarks,  # noqa: E402
                            pick_threshold, risk_coverage, selective_accuracy)


def _world(seed=0):
    return np.random.default_rng(seed).normal(0, 0.05, (21, 3)).astype(np.float64)


def _rot(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q if np.linalg.det(q) > 0 else -q


def test_canonical_is_rotation_invariant_but_oriented_is_not():
    w = _world()
    r = w @ _rot(1).T
    np.testing.assert_allclose(canonical_shape(w), canonical_shape(r), atol=1e-4)
    assert not np.allclose(oriented_shape(w), oriented_shape(r), atol=1e-2)


def test_engineered_descriptors_ignore_rotation_scale_and_mirroring():
    w = _world(3)
    e = engineered_shape(w)
    assert e.shape == (42,)
    np.testing.assert_allclose(e, engineered_shape(w @ _rot(4).T * 1.7), atol=1e-4)
    np.testing.assert_allclose(e, engineered_shape(mirror_world(w)), atol=1e-4)
    assert FEATURE_MODES["oriented_eng"](w).shape == (105,)
    assert static_features(np.stack([w, w]), "oriented_eng").shape == (2, 105)


def test_augmentation_perturbs_without_destroying_shape():
    w = np.stack([_world(5)])
    copies = augment_world(w, np.random.default_rng(0), k=3)
    assert len(copies) == 3 and all(c.shape == w.shape for c in copies)
    base = engineered_shape(w[0])
    for c in copies:
        assert np.abs(engineered_shape(c[0]) - base).mean() < 0.35


def _toy(n_cls=4, n_ses=3, per=12, seed=0):
    rng = np.random.default_rng(seed)
    templates = [_world(10 + c) for c in range(n_cls)]
    world, y, ses, grp = [], [], [], []
    for c, t in enumerate(templates):
        for s in range(n_ses):
            tilt = _rot(100 * c + s)
            for i in range(per):
                world.append((t + rng.normal(0, 0.003, t.shape)) @ tilt.T)
                y.append(c); ses.append(f"s{s}"); grp.append(f"c{c}s{s}p{i // 3}")
    return np.array(world), np.array(y), np.array(ses), np.array(grp)


def test_pipeline_generalises_to_unseen_sessions_on_toy_data():
    world, yi, ses, grp = _toy()
    r = evaluate_class_session(world, yi, ses, MODELS["logreg"], k_aug=2, seeds=(0,))
    assert np.mean(r["acc"]) > 0.9
    assert evaluate_grouped(world, yi, grp, MODELS["logreg"], k_aug=2) > 0.9


def test_feature_bank_rows_stay_aligned():
    world, yi, _, _ = _toy(n_cls=2, n_ses=2, per=3)
    bank = feature_bank(world, "oriented_eng", 2, 0)
    assert len(bank) == 4 and all(b.shape == bank[0].shape for b in bank)
    np.testing.assert_allclose(bank[0], static_features(world, "oriented_eng"))


def test_abstention_threshold_and_selective_accuracy():
    rng = np.random.default_rng(0)
    conf = rng.random(2000)
    correct = rng.random(2000) < (0.3 + 0.7 * conf)
    y_true = np.zeros(2000, int)
    y_pred = np.where(correct, 0, 1)
    thr, reached = pick_threshold(y_true, y_pred, conf, target=0.85)
    assert reached and 0.4 <= thr <= 0.95
    assert selective_accuracy(y_true, y_pred, conf, 0.2) > selective_accuracy(y_true, y_pred, conf, 1.0)
    rows = risk_coverage(y_true, y_pred, conf)
    assert rows[0]["coverage"] == 1.0 and rows[-1]["coverage"] < rows[0]["coverage"]
    assert not pick_threshold(y_true, 1 - y_true, conf, target=0.9)[1]


def test_static_recorder_roundtrip_feeds_training_loader(tmp_path):
    rec = StaticSampleRecorder(tmp_path / "static", "A", "ana", every=1)
    rec.toggle()
    h = Hand(np.random.rand(21, 3).astype(np.float32), np.random.rand(21, 3).astype(np.float32), 0.9)
    for _ in range(8):
        rec.add(h)
    rec.add(Hand(h.pts, h.world, 0.9, stale=True))  # held hands must not be recorded
    assert rec.toggle() and rec.count == 8
    base = tmp_path / "base.npz"
    np.savez(base, world=np.zeros((2, 21, 3), np.float32), y=np.array(["B", "B"]), group=np.array(["g1", "g2"]))
    d = load_landmarks(base, [tmp_path / "static"])
    assert len(d["y"]) == 10 and set(d["signer"]) == {"dataset", "ana"} and (d["y"] == "A").sum() == 8
