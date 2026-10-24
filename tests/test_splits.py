import numpy as np

from signlens.splits import assign_sessions, class_session_folds


def test_sessions_follow_time_gaps_and_counters():
    names = ["IMG_20230731_173339_jpg", "IMG_20230731_173343_jpg", "IMG_20230731_190000_jpg",
             "WhatsApp-Image-2023-07-31-at-17-34-10_jpeg", "IMG_7572_PNG", "IMG_7574_PNG", "IMG_7700_JPG",
             "6f1c-uuid_jpg"]
    s = assign_sessions(names)
    assert s[0] == s[1] == s[3]           # same sitting (< 5 min apart), even across naming schemes
    assert s[2] != s[0]                   # 1h30 later
    assert s[4] == s[5] and s[6] != s[4]  # consecutive camera counter vs far jump
    assert s[7].startswith("single:")     # unknown names are never merged by guesswork


def test_class_session_folds_never_train_on_a_tested_class_session():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(5), 60)
    sessions = np.array([f"s{rng.integers(0, 4)}" for _ in labels])
    for seed in range(3):
        fold, ok = class_session_folds(labels, sessions, k=3, seed=seed)
        for f in range(3):
            te, tr = (fold == f) & ok, fold != f
            for c in np.unique(labels):
                assert not (set(sessions[te & (labels == c)]) & set(sessions[tr & (labels == c)]))


def test_single_session_classes_are_not_evaluable():
    labels = np.array(["A"] * 4 + ["B"] * 4)
    sessions = np.array(["s1"] * 4 + ["s1", "s1", "s2", "s2"])
    fold, ok = class_session_folds(labels, sessions)
    assert not ok[:4].any() and ok[4:].all()


def test_extractor_retries_with_upscaled_crop_for_small_hands():
    from collections import Counter

    from signlens.dataset import StaticHandExtractor

    class Fake(StaticHandExtractor):
        def __init__(self):
            self.stages = Counter()

        def _detect(self, rgb):  # a "hand" only appears once the crop has been upscaled to 256 px
            if min(rgb.shape[:2]) != 256:
                return []
            pts = np.full((21, 3), 0.5, dtype=np.float32)
            return [(pts, pts.copy(), 0.9)]

    ex = Fake()
    hit = ex(np.zeros((640, 640, 3), np.uint8), (0.5, 0.5, 0.2, 0.2))
    assert hit is not None and ex.stages["wide-crop-upscaled"] + ex.stages["tight-crop-upscaled"] == 1
    np.testing.assert_allclose(hit[0][:, :2].mean(0), (0.5, 0.5), atol=0.02)  # remapped to full-image coords
    assert Fake()(np.zeros((640, 640, 3), np.uint8), None) is None            # no box -> no retry cascade
