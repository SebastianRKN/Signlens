"""Regressions for observed data loss, stale predictions and model deployment failures."""
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from signlens.classifier import Prediction, SignClassifier, Stabilizer
from signlens.panel import ControlPanel
from signlens.recorder import StaticSampleRecorder
from signlens.runtime import Runtime
from signlens.splits import class_session_folds
from signlens.training_job import TrainingJob, active_model, file_hash, publish_model
from signlens.types import Hand
from signlens.validation import evaluate_reserved


def hand(seed=1):
    r = np.random.default_rng(seed)
    return Hand(r.random((21, 3)), r.normal(0, .04, (21, 3)), .95)


def save_take(root, session="new", label="A", seed=1):
    rec = StaticSampleRecorder(root, label, "tester", 1, session)
    rec.toggle()
    for i in range(15):
        rec.add(hand(seed + i))
    rec._times = list(np.arange(15) * .1)
    return rec.toggle()


def fixture_model(models, sub=""):
    dest = models / sub
    dest.mkdir(parents=True, exist_ok=True)
    clf = DummyClassifier(strategy="constant", constant=0).fit(np.zeros((2, 105)), [0, 1])
    report = {"trained_at": "2026-10-20 12:00:00", "scope": "letters"}
    joblib.dump({"model": clf, "mode": "oriented_eng", "classes": ["A", "B"],
                 "threshold": .8, "meta": report}, dest / "sign_classifier.joblib")
    (dest / "evaluation.json").write_text(json.dumps(report), encoding="utf-8")
    np.savez(models / "landmarks.npz", world=np.stack([hand(900).world, hand(901).world]),
             y=np.array(["A", "B"]), group=np.array(["base-a", "base-b"]),
             session=np.array(["base-session"] * 2), signer=np.array(["dataset"] * 2))
    if sub:
        (dest / "training_data").mkdir()
        (dest / "training_data" / "base.npz").write_bytes((models / "landmarks.npz").read_bytes())
    return dest


def test_repeated_takes_same_letter_and_session_never_overwrite(tmp_path):
    first = save_take(tmp_path)
    original = first.read_bytes()
    second = save_take(tmp_path, seed=200)
    assert first != second
    assert len(list(tmp_path.rglob("*.npz"))) == 2
    assert first.read_bytes() == original
    with np.load(first) as a, np.load(second) as b:
        assert a["session"][0] == b["session"][0]
        assert a["group"][0] != b["group"][0]


def test_capture_copies_arrays_and_rejects_invalid_or_stale_hand(tmp_path):
    rec = StaticSampleRecorder(tmp_path, "A", "tester", 1)
    rec.toggle()
    h = hand()
    expected = h.world.copy()
    rec.add(h)
    h.world[:] = 99
    np.testing.assert_array_equal(rec._world[0], expected)
    rec.add(Hand(h.pts, h.world, stale=True))
    rec.add(Hand(h.pts, np.full((21, 3), np.nan)))
    assert rec.count == 1


@pytest.mark.parametrize("bad", ["../escape", "a/b", "C:\\escape", "", "NUL", "a" * 49])
def test_capture_identifiers_cannot_escape_storage(tmp_path, bad):
    with pytest.raises(ValueError):
        StaticSampleRecorder(tmp_path, "A", bad)


def test_stabilizer_clears_old_letter_on_transition_and_long_gap():
    s = Stabilizer(hold_seconds=.45)
    for i in range(12):
        out = s.update(Prediction("A", .99), i * .05)
    assert out.label == "A"
    assert s.update(Prediction("B", .99), .6) is None
    assert s.update(Prediction("A", .99), 5) is None
    assert s.update(None, 5.1) is None


def test_single_session_classes_are_always_training():
    folds, valid = class_session_folds(["A", "B", "B"], ["single", "one", "two"])
    assert folds[0] == -1 and not valid[0]
    assert all(folds[0] != f for f in range(3))


def test_model_publication_is_versioned_and_invalid_candidate_preserves_active(tmp_path):
    fixture_model(tmp_path)
    old = (tmp_path / "sign_classifier.joblib").read_bytes()
    valid = fixture_model(tmp_path, "runs/good")
    deployed = publish_model(tmp_path, valid)
    assert active_model(tmp_path) == deployed
    assert (tmp_path / "sign_classifier.joblib").read_bytes() == old
    pointer = (tmp_path / "active_model.json").read_bytes()
    invalid = tmp_path / "runs/broken"
    invalid.mkdir()
    (invalid / "evaluation.json").write_text("{}", encoding="utf-8")
    with pytest.raises(Exception):
        publish_model(tmp_path, invalid)
    assert (tmp_path / "active_model.json").read_bytes() == pointer
    deployed.write_bytes(b"broken")
    with pytest.raises(ValueError, match="incompleto"):
        active_model(tmp_path)


def test_training_log_streams_and_cancellation_preserves_model(tmp_path, monkeypatch):
    models, data = tmp_path / "models", tmp_path / "data"
    fixture_model(models)
    save_take(data / "static")
    before = file_hash(models / "sign_classifier.joblib")
    real_popen = subprocess.Popen

    def sleeping_process(_command, **kwargs):
        return real_popen([sys.executable, "-u", "-c", "import time; print('progress-visible'); time.sleep(30)"], **kwargs)
    monkeypatch.setattr("signlens.training_job.subprocess.Popen", sleeping_process)
    job = TrainingJob(models, data)
    assert job.start()
    assert not job.start()
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if job.log_path.exists() and "progress-visible" in job.log_path.read_text(encoding="utf-8"):
            break
        time.sleep(.03)
    else:
        pytest.fail("Progress did not stream to disk")
    job.close()
    assert not job.running
    assert job.process.poll() is not None
    assert before == file_hash(models / "sign_classifier.joblib")
    assert not (models / "active_model.json").exists()
    assert not (models / ".training.lock").exists()


def test_reserved_capture_and_hand_switch_use_real_controller(tmp_path):
    fixture_model(tmp_path / "models")
    app = Runtime(tmp_path / "models", tmp_path / "data", every=1)
    app.handle("reserved")
    app.handle("capture")
    app.capture_at = 0
    old_hand = app.panel.hand
    app.handle("hand")
    assert app.panel.hand == old_hand  # Cannot change hand during a labeled take.
    for i in range(8):
        app.process_hand(hand(i + 1), now=100 + i * .1)
    app.handle("capture")
    assert len(list((tmp_path / "data/validation").rglob("*.npz"))) == 1
    assert not list((tmp_path / "data/static").rglob("*.npz"))
    app.panel.confirmed = "A"
    app.handle("hand")
    assert app.panel.confirmed == "" and app.panel.hand != old_hand
    app.close()


def test_keyboard_data_controls_are_disabled_when_busy():
    p = ControlPanel(["A", "B"])
    p.view = "data"
    p.draw(np.zeros((480, 640, 3), dtype=np.uint8))
    assert p._buttons == []
    p.capturing = True
    p.on_mouse(cv2.EVENT_LBUTTONUP, 10, 10, 0, None)
    p.on_key(ord("l"))
    assert p.pop_actions() == []
    p.on_key(ord("c"))
    assert p.pop_actions() == ["capture"]
    p.training = True
    p.on_key(ord("c"))
    assert p.pop_actions() == []


def test_evaluation_rejects_training_session_leakage_and_small_evidence(tmp_path):
    models, data = tmp_path / "models", tmp_path / "data"
    fixture_model(models)
    save_take(data / "static", session="same", seed=1)
    save_take(data / "validation", session="same", seed=50)
    with pytest.raises(ValueError, match="comparte sesion"):
        evaluate_reserved(models, data)
    # Independent directories retain the rejected input for diagnosis.
    clean = tmp_path / "clean"
    save_take(clean / "validation", session="held-out", seed=100)
    report = evaluate_reserved(models, clean)
    assert report["known_takes"] == 1
    assert report["accuracy_on_accepted_takes"] == 1.0
    assert report["demo_gate"] == "PENDING"
    assert not report["minimum_evidence_met"]


class ModerateConfidenceModel:
    """A valid F prediction that the accidentally enabled demo rejected twice."""
    classes_ = np.array([0, 1])

    def predict_proba(self, x):
        return np.tile([.65, .35], (len(x), 1))


def moderate_runtime(tmp_path):
    fixture_model(tmp_path / "models")
    app = Runtime(tmp_path / "models", tmp_path / "data", every=1)
    app.classifier = SignClassifier({"model": ModerateConfidenceModel(), "mode": "oriented_eng",
                                    "classes": ["F", "A"], "threshold": .56})
    return app


def confirm(app, start=10):
    for i in range(7):
        app.process_hand(hand(), now=start + i * .03)


def test_default_preserves_full_vocabulary_original_threshold_and_fast_confirmation(tmp_path):
    app = moderate_runtime(tmp_path)
    assert not app.panel.demo
    confirm(app)
    assert app.panel.provisional == app.panel.confirmed == "F"


def test_limited_demo_requires_explicit_selection_and_can_be_undone(tmp_path):
    app = moderate_runtime(tmp_path)
    app.handle("mode")
    confirm(app)
    assert app.panel.confirmed == app.panel.provisional == ""
    assert app.stabilizer.hold_seconds == .45
    app.handle("mode")
    confirm(app, start=20)
    assert app.panel.confirmed == "F"
    assert app.stabilizer.hold_seconds == 0


def test_reserved_destination_does_not_hide_recognition_outside_actual_take(tmp_path):
    app = moderate_runtime(tmp_path)
    app.handle("reserved")
    confirm(app)
    assert app.panel.confirmed == "F"
    app.handle("capture")
    app.capture_at = 0
    confirm(app, start=20)
    assert app.panel.confirmed == app.panel.provisional == ""
    app.handle("capture")
    confirm(app, start=30)
    assert app.panel.confirmed == "F"


def test_unknown_capture_label_does_not_hide_normal_recognition(tmp_path):
    app = moderate_runtime(tmp_path)
    app.panel.label_index = app.panel.labels.index("NINGUNA")
    confirm(app)
    assert app.panel.confirmed == "F"


class IndependentHandsClassifier:
    threshold = .56
    meta = {}

    def predict(self, world, threshold=None, allowed=None):
        label = {1: "A", 2: "B", 3: "F"}.get(int(world[0, 0]))
        if label is None or (allowed and label not in allowed):
            return None
        return Prediction(label, .95)


def marked_hand(marker, stale=False):
    h = hand()
    world = h.world.copy()
    world[0, 0] = marker
    return Hand(h.pts, world, h.score, stale=stale)


def two_hand_runtime(tmp_path):
    fixture_model(tmp_path / "models")
    app = Runtime(tmp_path / "models", tmp_path / "data", every=1)
    app.classifier = IndependentHandsClassifier()
    return app


def feed_pair(app, left=1, right=2, start=10):
    for i in range(7):
        app.process_hands({"Left": marked_hand(left), "Right": marked_hand(right)}, start + i * .03)


def test_two_different_letters_confirm_independently(tmp_path):
    app = two_hand_runtime(tmp_path)
    feed_pair(app, left=1, right=3)
    assert app.panel.results["Left"].confirmed == "A"
    assert app.panel.results["Right"].confirmed == "F"
    assert app.stabilizers["Left"] is not app.stabilizers["Right"]


@pytest.mark.parametrize("lost", ["Left", "Right"])
def test_missing_or_stale_hand_clears_only_its_result(tmp_path, lost):
    app = two_hand_runtime(tmp_path)
    feed_pair(app)
    other = "Right" if lost == "Left" else "Left"
    pair = {"Left": marked_hand(1), "Right": marked_hand(2)}
    pair[lost] = marked_hand(1 if lost == "Left" else 2, stale=True)
    app.process_hands(pair, 10.22)
    assert app.panel.results[lost].confirmed == ""
    expected = "B" if other == "Right" else "A"
    assert app.panel.results[other].confirmed == expected
    del pair[lost]
    app.process_hands(pair, 10.25)
    assert app.panel.results[other].confirmed == expected


def test_change_on_one_side_does_not_reset_the_other(tmp_path):
    app = two_hand_runtime(tmp_path)
    feed_pair(app)
    app.process_hands({"Left": marked_hand(3), "Right": marked_hand(2)}, 10.22)
    assert app.panel.results["Left"].confirmed == ""
    assert app.panel.results["Right"].confirmed == "B"
    feed_pair(app, left=3, right=2, start=10.25)
    assert app.panel.results["Left"].confirmed == "F"
    assert app.panel.results["Right"].confirmed == "B"


def test_capture_hand_selector_does_not_interrupt_two_hand_recognition(tmp_path):
    app = two_hand_runtime(tmp_path)
    feed_pair(app)
    app.handle("hand")
    assert app.panel.hand == "Left"
    assert app.panel.results["Left"].confirmed == "A"
    assert app.panel.results["Right"].confirmed == "B"


def test_two_hands_record_only_selected_label_and_hide_only_reserved_side(tmp_path):
    app = two_hand_runtime(tmp_path)
    app.handle("reserved")
    app.handle("capture")
    app.capture_at = 0
    feed_pair(app)
    assert app.panel.results["Left"].confirmed == "A"
    assert app.panel.results["Right"].confirmed == ""
    app.handle("capture")
    files = list((tmp_path / "data/validation").rglob("*.npz"))
    assert len(files) == 1
    with np.load(files[0]) as saved:
        assert np.all(saved["world"][:, 0, 0] == 2)
    feed_pair(app, start=10.3)
    assert app.panel.results["Right"].confirmed == "B"


def test_camera_loss_and_mode_change_clear_both_histories(tmp_path):
    app = two_hand_runtime(tmp_path)
    feed_pair(app)
    app.clear_prediction(message="Sin imagen de camara")
    assert all(not state.confirmed and not state.provisional for state in app.panel.results.values())
    assert all(state.recognition == "Sin imagen de camara" for state in app.panel.results.values())
    feed_pair(app, start=20)
    app.handle("mode")
    assert all(s.hold_seconds == .45 for s in app.stabilizers.values())
    assert all(not state.confirmed for state in app.panel.results.values())
