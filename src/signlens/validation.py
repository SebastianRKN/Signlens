"""Evaluate reserved camera takes. Never uses these files for training or tuning."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from signlens.classifier import DEMO_LABELS, DEMO_THRESHOLD, Prediction, SignClassifier, Stabilizer
from signlens.features import static_features
from signlens.training_job import active_model, file_hash

UNKNOWN = "NINGUNA"


def dataset_inventory(root: Path) -> dict:
    counts, sessions, files, errors = Counter(), {}, 0, []
    for path in sorted(root.rglob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as d:
                counts.update(map(str, d["y"]))
                for label, session in zip(d["y"], d["session"]):
                    sessions.setdefault(str(label), set()).add(str(session))
                files += 1
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{path.name}: {exc}")
    return {"files": files, "samples": sum(counts.values()), "per_label": dict(counts),
            "sessions_per_label": {k: len(v) for k, v in sessions.items()}, "errors": errors}


def _row_hashes(world) -> set[str]:
    return {hashlib.sha256(np.asarray(w, dtype=np.float32).tobytes()).hexdigest() for w in world}


def evaluate_reserved(models: Path, data: Path, model_path: Path | None = None,
                      demo: bool = True) -> dict:
    path = model_path or active_model(models)
    clf = SignClassifier.load(str(path))
    validation_files = sorted((data / "validation").rglob("*.npz"))
    if not validation_files:
        raise ValueError("No hay tomas reservadas. Elige Guardar: Evaluar y captura una sesion nueva.")
    snapshot = path.parent / "training_data"
    train_files = ([snapshot / "base.npz"] + sorted((snapshot / "captures").glob("*.npz"))
                   if snapshot.exists() else [models / "landmarks.npz"] + sorted((data / "static").rglob("*.npz")))
    train_sessions, train_signers, train_hashes = set(), set(), set()
    for train in train_files:
        with np.load(train, allow_pickle=True) as d:
            train_sessions.update(map(str, d["session"])) if "session" in d else None
            train_signers.update(map(str, d["signer"])) if "signer" in d else None
            train_hashes.update(_row_hashes(d["world"]))
    allowed = DEMO_LABELS if demo else tuple(map(str, clf.classes))
    threshold = max(DEMO_THRESHOLD, clf.threshold) if demo else clf.threshold
    takes, inputs, eval_signers = [], [], set()
    for f in validation_files:
        with np.load(f, allow_pickle=False) as archive:
            d = dict(archive)
        labels = set(map(str, d["y"]))
        if len(labels) != 1 or d["world"].shape != (len(d["y"]), 21, 3):
            raise ValueError(f"Toma invalida: {f.name}")
        if not np.isfinite(d["world"]).all():
            raise ValueError(f"Landmarks no finitos: {f.name}")
        if train_sessions.intersection(map(str, d["session"])) or train_hashes.intersection(_row_hashes(d["world"])):
            raise ValueError(f"Evaluacion bloqueada: {f.name} comparte sesion o muestras con entrenamiento. Usa otra sesion real.")
        label = next(iter(labels))
        if label != UNKNOWN and label not in allowed:
            continue
        eval_signers.update(map(str, d["signer"]))
        prob = clf.model.predict_proba(static_features(d["world"], clf.mode))
        index = prob.argmax(axis=1)
        predictions = np.asarray(clf.classes)[clf.model.classes_[index].astype(int)]
        confidence = prob.max(axis=1)
        timestamps = d.get("captured_at")
        stable = Stabilizer(hold_seconds=0.45)
        accepted, raw_accepted = [], []
        for i, (prediction, conf) in enumerate(zip(predictions, confidence)):
            pred = Prediction(str(prediction), float(conf)) if conf >= threshold and prediction in allowed else None
            raw_accepted.append(pred.label if pred else None)
            # Older captures have no timing: do not fabricate a stability result.
            out = stable.update(pred, now=float(timestamps[i])) if timestamps is not None else pred
            accepted.append(out.label if out else None)
        votes = Counter(x for x in accepted if x is not None)
        winner, votes_n = votes.most_common(1)[0] if votes else (None, 0)
        emitted = sum(votes.values())
        decision = winner if votes_n >= max(3, len(accepted) * 0.5) else None
        takes.append({"file": f.relative_to(data).as_posix(), "expected": label, "predicted": decision,
                      "frames": len(accepted), "raw_accepted": sum(x is not None for x in raw_accepted),
                      "confirmed_frames": emitted, "any_false_accept": label == UNKNOWN and emitted > 0,
                      "timed_stability": timestamps is not None,
                      "session": str(d["session"][0]), "signer": str(d["signer"][0])})
        inputs.append({"file": str(f.relative_to(data)), "sha256": file_hash(f)})
    known = [x for x in takes if x["expected"] != UNKNOWN]
    accepted = [x for x in known if x["predicted"] is not None]
    negative = [x for x in takes if x["expected"] == UNKNOWN]
    precision = sum(x["predicted"] == x["expected"] for x in accepted) / len(accepted) if accepted else None
    coverage = len(accepted) / len(known) if known else None
    far = sum(x["any_false_accept"] for x in negative) / len(negative) if negative else None
    by_label = {label: sum(x["expected"] == label for x in known) for label in allowed}
    sufficient = all(n >= 5 for n in by_label.values()) and len(negative) >= 20 and len({x["session"] for x in takes}) >= 2
    meets = sufficient and precision is not None and precision >= .9 and coverage is not None and coverage >= .8 and far is not None and far <= .05
    report = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "model": str(path),
              "model_sha256": file_hash(path), "mode": "demo" if demo else "explore",
              "labels": list(allowed), "threshold": threshold, "unit": "capture_take",
              "known_takes": len(known), "unknown_takes": len(negative), "takes_per_label": by_label,
              "accuracy_on_accepted_takes": precision, "coverage": coverage, "unknown_false_accept_rate": far,
              "minimum_evidence_met": sufficient, "demo_gate": "PASS" if meets else "PENDING",
              "webcam_signers_unseen": bool(eval_signers) and not bool(eval_signers & train_signers),
              "limitations": ["Dataset source has no real signer IDs; signer independence cannot be established.",
                              "Captures in one session are correlated; take-level results are descriptive.",
                              "Test data must not be used to choose labels or confidence thresholds."],
              "inputs": inputs, "takes": takes}
    from signlens.reporting import report_path
    dest = report_path("validation", root=data.parent)
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return {**report, "report_path": str(dest)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--explore", action="store_true")
    args = ap.parse_args()
    report = evaluate_reserved(args.models, args.data, demo=not args.explore)
    print(json.dumps({k: v for k, v in report.items() if k not in {"takes", "inputs"}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
