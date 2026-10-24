"""Train and honestly evaluate the static-sign classifier.

    python -m signlens.train models/landmarks.npz --out models [--scope letters|all] [--extra data/static]

Evaluation (what each number means)
  main       class-session holdout: every class is tested on capture sessions it never saw in
             training. This is the number to quote and the one models are selected on.
  optimistic grouped by source photo only. Neighbouring photos from one session end up on both sides.
  strict     grouped by session. Pessimistic here because many classes exist in only 2-4 sessions.
  transfer   (with --extra) train on the dataset, test on YOUR own webcam captures.
Augmentation (mirror, small 3-D rotations, landmark noise) is applied to training folds only."""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from signlens.features import augment_world, mirror_world, static_features
from signlens.splits import assign_sessions, class_session_folds

MODE = "oriented_eng"
BASELINE_MODE = "oriented"
_svm = lambda C=3: make_pipeline(StandardScaler(), SVC(C=C, probability=True, random_state=0))
_mlp = lambda: make_pipeline(StandardScaler(), MLPClassifier((256, 128), alpha=1e-2, max_iter=400, random_state=0))
_lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))
MODELS = {
    "svm": _svm,
    "mlp": _mlp,
    "logreg": _lr,
    "ensemble": lambda: VotingClassifier([("svm", _svm()), ("mlp", _mlp()), ("lr", _lr())], voting="soft"),
}
SCOPES = {
    "letters": lambda y: y.isalpha() and (len(y) == 1 or y in {"CH", "LL", "RR"}),
    "all": lambda y: True,
}


# ------------------------------------------------------------------ data
def load_landmarks(base: Path, extras: list[Path] | None = None) -> dict[str, np.ndarray]:
    """Dataset landmarks (+ optional own captures). Every sample gets photo-group, session, signer."""
    with np.load(base, allow_pickle=True) as archive:
        d = dict(archive)
    d["session"] = d["session"] if "session" in d else assign_sessions(d["group"])
    d["signer"] = d["signer"] if "signer" in d else np.array(["dataset"] * len(d["y"]))
    parts = [d]
    for p in extras or []:
        p = Path(p)
        for f in sorted(p.rglob("*.npz")) if p.is_dir() else [p]:
            with np.load(f, allow_pickle=True) as archive:
                e = dict(archive)
            if {"world", "y", "group", "session", "signer"} <= set(e):
                parts.append(e)
            else:
                raise ValueError(f"Captura incompleta: {f}")
    keys = ["world", "y", "group", "session", "signer"]
    for part in parts:
        n = len(part["y"])
        if part["world"].shape != (n, 21, 3) or not np.isfinite(part["world"]).all():
            raise ValueError("Captura con landmarks invalidos.")
        if any(len(part[k]) != n for k in keys):
            raise ValueError("Metadatos desalineados en la captura.")
    return {k: np.concatenate([np.asarray(p[k]) for p in parts]) for k in keys}


def select_scope(d: dict, scope: str) -> dict:
    keep = np.array([SCOPES[scope](str(y)) for y in d["y"]])
    return {k: v[keep] for k, v in d.items()}


# ------------------------------------------------------------------ training matrix
def feature_bank(world: np.ndarray, mode: str, k_aug: int, seed: int) -> list[np.ndarray]:
    """[original, mirrored, aug_1..aug_k] feature matrices, each (N, D). Row i of every matrix comes
    from sample i, so folds just index rows and no augmented copy can cross a split."""
    rng = np.random.default_rng(seed)
    worlds = [world, mirror_world(world)] + (augment_world(world, rng, k_aug) if k_aug else [])
    return [static_features(w, mode) for w in worlds]


def _stack(bank: list[np.ndarray], yi: np.ndarray, idx: np.ndarray):
    return np.vstack([b[idx] for b in bank]), np.tile(yi[idx], len(bank))


def _predict(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return model.classes_[p.argmax(1)], p.max(1)
    return model.predict(X), np.ones(len(X))


# ------------------------------------------------------------------ evaluation
def evaluate_class_session(world, yi, sessions, factory, mode=MODE, k_aug=4, seeds=(0, 1, 2), k=3) -> dict:
    out = {"pred": [], "conf": [], "valid": [], "acc": [], "f1": []}
    for sd in seeds:
        bank = feature_bank(world, mode, k_aug, sd)
        fold, ok = class_session_folds(yi, sessions, k, sd)
        pred, conf = np.full(len(yi), -1), np.zeros(len(yi))
        for f in range(k):
            te, tr = (fold == f) & ok, np.where(fold != f)[0]
            if not te.any():
                continue
            X, y = _stack(bank, yi, tr)
            pred[te], conf[te] = _predict(factory().fit(X, y), bank[0][te])
        v = pred >= 0
        out["pred"].append(pred); out["conf"].append(conf); out["valid"].append(v)
        out["acc"].append(accuracy_score(yi[v], pred[v])); out["f1"].append(f1_score(yi[v], pred[v], average="macro"))
    return out


def evaluate_grouped(world, yi, groups, factory, mode=MODE, k_aug=4, n=5, seed=0) -> float:
    bank = feature_bank(world, mode, k_aug, seed)
    pred = np.empty(len(yi), dtype=int)
    for tr, te in StratifiedGroupKFold(n, shuffle=True, random_state=seed).split(bank[0], yi, groups):
        X, y = _stack(bank, yi, tr)
        pred[te] = factory().fit(X, y).predict(bank[0][te])
    return float(accuracy_score(yi, pred))


def risk_coverage(y_true, y_pred, conf, thresholds=(0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) -> list[dict]:
    rows = []
    for t in thresholds:
        sel = conf >= t
        rows.append({"threshold": t, "coverage": float(sel.mean()),
                     "accuracy_on_accepted": float((y_true[sel] == y_pred[sel]).mean()) if sel.any() else None})
    return rows


def selective_accuracy(y_true, y_pred, conf, coverage=0.5) -> float:
    """Accuracy on the `coverage` fraction of predictions the model is MOST confident about.
    High value = confidence really tracks correctness, which is what makes abstaining useful."""
    keep = np.argsort(-conf, kind="stable")[: max(int(len(conf) * coverage), 1)]
    return float((y_true[keep] == y_pred[keep]).mean())


def pick_threshold(y_true, y_pred, conf, target=0.75, lo=0.4, hi=0.95, fallback_coverage=0.4) -> tuple[float, bool]:
    """Smallest out-of-fold confidence threshold whose accepted predictions reach `target` accuracy.
    If the target is unreachable, fall back to accepting the most confident `fallback_coverage`.
    Returns (threshold, target_reached)."""
    for t in np.arange(lo, hi + 1e-9, 0.01):
        sel = conf >= t
        if sel.sum() >= 30 and (y_true[sel] == y_pred[sel]).mean() >= target:
            return float(round(t, 2)), True
    return float(round(np.quantile(conf, 1 - fallback_coverage), 3)), False


def fit_final(world, yi, factory, mode=MODE, k_aug=4, seed=0):
    bank = feature_bank(world, mode, k_aug, seed)
    X, y = _stack(bank, yi, np.arange(len(yi)))
    return factory().fit(X, y)


# ------------------------------------------------------------------ CLI
def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("landmarks", type=Path)
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--scope", choices=SCOPES, default="letters")
    ap.add_argument("--extra", type=Path, nargs="*", help="folder(s)/files of your own captures (.npz)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--target-precision", type=float, default=0.75, help="accuracy wanted on accepted predictions")
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS), help="candidates to compare")
    args = ap.parse_args()
    if args.seeds < 1 or not 0 < args.target_precision <= 1:
        ap.error("--seeds debe ser positivo y --target-precision estar entre 0 y 1.")

    d = select_scope(load_landmarks(args.landmarks, args.extra), args.scope)
    names, yi = np.unique(d["y"], return_inverse=True)
    if len(names) < 2 or not class_session_folds(yi, d["session"])[1].any():
        raise ValueError("Se requieren al menos dos letras y varias sesiones para evaluar.")
    world = d["world"].astype(np.float64)
    seeds = tuple(range(args.seeds))
    print(f"scope={args.scope}: {len(yi)} samples | {len(names)} classes | {len(set(d['group']))} photos | "
          f"{len(set(d['session']))} sessions | signers {sorted(map(str, set(d['signer'])))}\n")

    print(f"{'candidate':34} {'accuracy':>16} {'macro-F1':>9} {'acc@50%cov':>11}   (class-session holdout, {len(seeds)} seeds)")
    prev_model = lambda: make_pipeline(StandardScaler(), SVC(C=10, probability=True, random_state=0))
    base = evaluate_class_session(world, yi, d["session"], prev_model, BASELINE_MODE, k_aug=0, seeds=seeds)
    print(f"{'previous: oriented + SVM C=10':34} {np.mean(base['acc']):9.3f} ±{np.std(base['acc']):.3f} {np.mean(base['f1']):9.3f}")
    results = {}
    for name in args.models:
        t0 = time.time()
        r = evaluate_class_session(world, yi, d["session"], MODELS[name], seeds=seeds)
        v = np.concatenate(r["valid"])
        r["flat"] = (np.concatenate([yi] * len(seeds))[v], np.concatenate(r["pred"])[v], np.concatenate(r["conf"])[v])
        r["sel50"] = selective_accuracy(*r["flat"])
        results[name] = r
        print(f"{MODE + ' + ' + name:34} {np.mean(r['acc']):9.3f} ±{np.std(r['acc']):.3f} {np.mean(r['f1']):9.3f} {r['sel50']:11.3f}   ({time.time() - t0:.0f}s)", flush=True)

    # Select: macro-F1 decides, but candidates within 0.01 of the best are ranked by how informative
    # their confidence is (accuracy on the 50% most confident predictions).
    top_f1 = max(np.mean(x["f1"]) for x in results.values())
    near = [n for n, x in results.items() if np.mean(x["f1"]) >= top_f1 - 0.01]
    best = max(near, key=lambda n: results[n]["sel50"])
    r = results[best]
    y_true, y_pred, conf = r["flat"]
    print(f"\nselected: {best}")
    optimistic = evaluate_grouped(world, yi, d["group"], MODELS[best])
    strict = evaluate_grouped(world, yi, d["session"], MODELS[best])
    print(f"  optimistic (photo-grouped): {optimistic:.3f}   <- inflated by same-session neighbours")
    print(f"  strict (session-grouped)  : {strict:.3f}   <- pessimistic: classes live in only 2-4 sessions")

    table = risk_coverage(y_true, y_pred, conf)
    thr, reached = pick_threshold(y_true, y_pred, conf, args.target_precision)
    sel = conf >= thr
    print(f"\nabstention threshold {thr:.2f} ({'target ' + format(args.target_precision, '.0%') + ' reached' if reached else 'target unreachable -> top-40% fallback'}): "
          f"accepts {sel.mean():.0%} of frames, {(y_true[sel] == y_pred[sel]).mean():.0%} correct among accepted")
    per_class = {str(c): float((y_pred[y_true == i] == i).mean()) for i, c in enumerate(names) if (y_true == i).any()}
    worst = sorted(per_class.items(), key=lambda kv: kv[1])[:6]
    cm = confusion_matrix(y_true, y_pred, labels=range(len(names)))
    off = cm.copy(); np.fill_diagonal(off, 0)
    top = [(str(names[a]), str(names[b]), int(off[a, b])) for a, b in zip(*np.unravel_index(np.argsort(off, axis=None)[::-1][:6], off.shape)) if off[a, b]]
    print("weakest classes:", [(c, round(a, 2)) for c, a in worst])
    print("most confused (true -> predicted):", top)

    transfer = None
    own = d["signer"] != "dataset"
    if own.any() and (~own).any():
        seen = np.isin(yi[own], yi[~own])
        m = fit_final(world[~own], yi[~own], MODELS[best])
        pred, _ = _predict(m, static_features(world[own], MODE))
        transfer = float(accuracy_score(yi[own][seen], pred[seen]))
        print(f"\ntransfer: dataset-only model on YOUR captures ({own.sum()} samples): {transfer:.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    final = fit_final(world, yi, MODELS[best])
    meta = {"scope": args.scope, "model": best, "feature_mode": MODE, "samples": int(len(yi)), "sessions": int(len(set(d["session"]))),
            "cv_class_session_accuracy": float(np.mean(r["acc"])), "cv_class_session_macro_f1": float(np.mean(r["f1"])),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset_license": "CC BY 4.0 (Roboflow Universe)"}
    out_name = "sign_classifier.joblib" if args.scope == "letters" else f"sign_classifier_{args.scope}.joblib"
    joblib.dump({"model": final, "mode": MODE, "classes": names.tolist(), "threshold": thr, "experimental": True, "meta": meta},
                args.out / out_name)
    report = {**meta, "previous_baseline": {"accuracy": float(np.mean(base["acc"])), "macro_f1": float(np.mean(base["f1"]))},
              "candidates": {n: {"accuracy": float(np.mean(x["acc"])), "accuracy_std": float(np.std(x["acc"])),
                                 "macro_f1": float(np.mean(x["f1"])), "accuracy_at_50pct_coverage": x["sel50"]} for n, x in results.items()},
              "optimistic_photo_grouped": optimistic, "strict_session_grouped": strict, "threshold": thr,
              "threshold_target_reached": reached, "risk_coverage": table, "per_class_accuracy": per_class, "top_confusions": top,
              "transfer_to_own_captures": transfer, "classes": names.tolist(), "confusion_matrix": cm.tolist(),
              "limitations": ["No signer IDs in the dataset: sessions are a proxy, not a signer-independent test.",
                              "Single-frame appearance only: no motion, words or phrases.",
                              "Only successfully extracted hands are evaluated.",
                              "No non-sign/unknown class: the threshold is an out-of-fold heuristic."]}
    rep_name = "evaluation.json" if args.scope == "letters" else f"evaluation_{args.scope}.json"
    (args.out / rep_name).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(args.out / "out_of_fold.npz", true=names[y_true], predicted=names[y_pred], confidence=conf)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        used = sorted(set(y_true) | set(y_pred))  # classes seen in a single session are never tested
        fig, ax = plt.subplots(figsize=(11, 10))
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=used, display_labels=[str(names[i]) for i in used],
                                                ax=ax, cmap="Blues", colorbar=False, normalize="true", values_format=".1f")
        ax.set_title(f"Out-of-fold confusion matrix ({MODE} + {best}, class-session holdout)")
        fig.tight_layout(); fig.savefig(args.out / f"confusion_matrix_{args.scope}.png", dpi=130)
    except ImportError:
        pass
    print(f"\nsaved {args.out / out_name} and {rep_name}")


if __name__ == "__main__":
    main()
