import cv2
import numpy as np
import pytest

from signlens.classifier import Prediction, Stabilizer
from signlens.dataset import build, read_yolo, source_id
from signlens.features import canonical_shape, mirror_world, oriented_shape, static_features

pytest.importorskip("sklearn")


def _world(seed=0):
    return np.random.default_rng(seed).normal(0, 0.05, (21, 3)).astype(np.float64)


def _rot(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q if np.linalg.det(q) > 0 else -q  # proper rotation


def test_canonical_is_rotation_invariant_but_oriented_is_not():
    w = _world()
    r = w @ _rot(1).T
    np.testing.assert_allclose(canonical_shape(w), canonical_shape(r), atol=1e-4)
    assert not np.allclose(oriented_shape(w), oriented_shape(r), atol=1e-2)


def test_mirroring_negates_canonical_z_only():
    w = _world(2)
    a, b = canonical_shape(w), canonical_shape(mirror_world(w))
    np.testing.assert_allclose(a[:, :2], b[:, :2], atol=1e-5)
    np.testing.assert_allclose(a[:, 2], -b[:, 2], atol=1e-5)


def test_yolo_parsing_and_build_keeps_source_groups(tmp_path):
    (tmp_path / "train/images").mkdir(parents=True)
    (tmp_path / "train/labels").mkdir()
    (tmp_path / "data.yaml").write_text("nc: 2\nnames: ['A', 'CH']\n")
    for name, cls in (("A_IMG1_jpg.rf.aaa.jpg", 0), ("A_IMG1_jpg.rf.bbb.jpg", 0), ("CH_WIN_x_jpg.rf.ccc.jpg", 1)):
        cv2.imwrite(str(tmp_path / "train/images" / name), np.zeros((64, 64, 3), np.uint8))
        (tmp_path / "train/labels" / (name[:-4] + ".txt")).write_text(f"{cls} 0.5 0.5 0.4 0.4\n")
    names, samples = read_yolo(tmp_path)
    assert names == ["A", "CH"] and len(samples) == 3
    detector = lambda img, box: (np.zeros((21, 3)), _world(), 0.9) if box[0] else None
    data, failed = build(samples, detector, progress=0)
    assert len(data["y"]) == 3 and not failed
    assert len(set(data["group"])) == 2 and set(data["domain"]) == {"phone", "webcam"}
    assert source_id(samples[0].image) == "A_IMG1_jpg"


def test_stabilizer_filters_flicker_and_low_confidence():
    s = Stabilizer(window=5, min_share=0.6, min_conf=0.5)
    out = None
    for p in [Prediction("A", .9), Prediction("B", .9), Prediction("A", .8), Prediction("A", .9), Prediction("A", .9)]:
        out = s.update(p)
    assert out.label == "A"  # the single "B" is outvoted
    s2 = Stabilizer(window=5)
    for _ in range(5):
        out = s2.update(Prediction("A", 0.3))
    assert out is None
    for _ in range(5):
        out = s2.update(None)
    assert out is None
