"""YOLO-format image dataset -> hand landmarks (.npz). Run locally (needs MediaPipe models):
    python -m signlens.dataset path/to/dataset -o landmarks.npz

Design notes
- The dataset's boxes are only used to pick the right hand and to retry on a crop when the full
  image fails. The classifier never sees them.
- `group` = source photo. Roboflow exports several augmented copies of each photo; they MUST stay in
  the same CV fold or the evaluation leaks."""
from __future__ import annotations

import argparse
import ast
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import yaml

from signlens.splits import assign_sessions

# detector(bgr, bbox_cxcywh | None) -> (pts (21,3), world (21,3), score) | None
Detector = Callable[[np.ndarray, tuple | None], tuple | None]


@dataclass(frozen=True)
class Sample:
    image: Path
    label: str
    bbox: tuple[float, float, float, float]  # normalized cx, cy, w, h


def source_id(path: Path) -> str:
    return path.name.split(".rf.")[0]


def domain_of(src: str) -> str:
    return "webcam" if "_WIN_" in src else "phone"


def parse_names(yaml_path: Path) -> list[str]:
    names = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["names"]
    if isinstance(names, dict):
        names = [names[i] if i in names else names[str(i)] for i in range(len(names))]
    return [str(n) for n in names]


def read_yolo(root: Path) -> tuple[list[str], list[Sample]]:
    names, samples = parse_names(root / "data.yaml"), []
    for split in ("train", "valid", "test"):
        img_dir = root / split / "images"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            lab = root / split / "labels" / f"{img.stem}.txt"
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"} or not lab.exists():
                continue
            for line in lab.read_text().splitlines():
                tok = line.split()
                if not tok:
                    continue
                vals = list(map(float, tok[1:]))
                if len(vals) == 4:
                    box = tuple(vals)
                elif len(vals) >= 6 and len(vals) % 2 == 0:
                    xs, ys = vals[0::2], vals[1::2]
                    box = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys))
                else:
                    raise ValueError(f"Invalid annotation: {lab}: {line}")
                samples.append(Sample(img, names[int(tok[0])], box))
    return names, samples


class StaticHandExtractor:
    """MediaPipe HandLandmarker in IMAGE mode (independent detection per photo)."""

    def __init__(self, models_dir: Path | None = None, min_conf: float = 0.3):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        from signlens.config import Config
        from signlens.models import ensure_model

        self._mp = mp
        path = ensure_model("hand", models_dir or Config().models_dir)
        self._lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=min_conf, min_hand_presence_confidence=min_conf))

    def _detect(self, rgb: np.ndarray) -> list[tuple]:
        from signlens.tracker import _to_array

        res = self._lm.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        return [(_to_array(p), _to_array(w), float(c[0].score))
                for p, w, c in zip(res.hand_landmarks, res.hand_world_landmarks, res.handedness)]

    @staticmethod
    def _pick(cands: list[tuple], bbox: tuple | None, slack: float = 1.0):
        """Best candidate hand. With a box: only hands whose centre lies inside it (a bystander's
        hand is rejected), nearest to the box centre first."""
        if not cands:
            return None
        if bbox is None:
            return max(cands, key=lambda c: c[2])
        cx, cy, bw, bh = bbox
        cands = [c for c in cands if abs(float(c[0][:, 0].mean()) - cx) <= slack * bw / 2
                 and abs(float(c[0][:, 1].mean()) - cy) <= slack * bh / 2]
        return min(cands, key=lambda c: float(np.hypot(*(c[0][:, :2].mean(0) - bbox[:2]))), default=None)

    def _crop_detect(self, rgb: np.ndarray, bbox: tuple, margin: float, min_side: int):
        h, w = rgb.shape[:2]
        cx, cy, bw, bh = bbox
        m = margin * max(bw, bh)
        x0, x1 = int(max(cx - bw / 2 - m, 0) * w), int(min(cx + bw / 2 + m, 1) * w)
        y0, y1 = int(max(cy - bh / 2 - m, 0) * h), int(min(cy + bh / 2 + m, 1) * h)
        if x1 - x0 < 24 or y1 - y0 < 24:
            return None
        crop = rgb[y0:y1, x0:x1]
        if min(crop.shape[:2]) < min_side:  # small hands: upscale so the palm detector has pixels to work with
            f = min_side / min(crop.shape[:2])
            crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        cands = []
        for pts, world, score in self._detect(crop):
            pts = pts.copy()
            pts[:, 0] = (x0 + pts[:, 0] * (x1 - x0)) / w  # back to full-image coordinates
            pts[:, 1] = (y0 + pts[:, 1] * (y1 - y0)) / h
            cands.append((pts, world, score))
        return self._pick(cands, bbox, slack=1.25)

    # (name, margin around the annotated box, minimum crop side in px)
    STAGES = (("crop", 0.25, 0), ("wide-crop-upscaled", 0.6, 256), ("tight-crop-upscaled", 0.1, 256))

    def __call__(self, bgr: np.ndarray, bbox: tuple | None = None):
        if not hasattr(self, "stages"):
            self.stages = Counter()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        hit = self._pick(self._detect(rgb), bbox)
        if hit is not None:
            self.stages["full image"] += 1
            return hit
        if bbox is not None:
            for name, margin, min_side in self.STAGES:
                hit = self._crop_detect(rgb, bbox, margin, min_side)
                if hit is not None:
                    self.stages[name] += 1
                    return hit
        self.stages["FAILED"] += 1
        return None


def build(samples: list[Sample], detector: Detector, progress: int = 100) -> tuple[dict, list[Sample]]:
    keep, pts, world, score, failed = [], [], [], [], []
    for i, s in enumerate(samples, 1):
        img = cv2.imread(str(s.image))
        hit = detector(img, s.bbox) if img is not None else None
        if hit is None:
            failed.append(s)
        else:
            keep.append(s); pts.append(hit[0]); world.append(hit[1]); score.append(hit[2])
        if progress and i % progress == 0:
            print(f"  {i}/{len(samples)}  detected {len(keep)}")
    srcs = [source_id(s.image) for s in keep]
    data = dict(
        world=np.array(world, dtype=np.float32), pts=np.array(pts, dtype=np.float32),
        y=np.array([s.label for s in keep]), group=np.array(srcs),
        domain=np.array([domain_of(x) for x in srcs]), score=np.array(score, dtype=np.float32),
        session=assign_sessions(srcs) if srcs else np.array([]), signer=np.array(["dataset"] * len(srcs)),
        path=np.array([str(s.image) for s in keep]))
    return data, failed


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract hand landmarks from a YOLO-format image dataset")
    ap.add_argument("root", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("landmarks.npz"))
    args = ap.parse_args()
    names, samples = read_yolo(args.root)
    print(f"{len(samples)} images, {len(names)} classes, {len({source_id(s.image) for s in samples})} source photos")
    extractor = StaticHandExtractor()
    data, failed = build(samples, extractor)
    np.savez_compressed(args.out, **data)
    print("extraction stage that succeeded:", dict(extractor.stages))
    print(f"\nsaved {args.out}: {len(data['y'])} samples ({len(failed)} without a detectable hand)")
    total = {n: sum(s.label == n for s in samples) for n in names}
    for n in names:
        ok = int((data["y"] == n).sum())
        print(f"  {n:3} detected {ok:3}/{total[n]:3}  ({100 * ok / max(total[n], 1):.0f}%)")
    (args.out.parent / "failed.txt").write_text("\n".join(f"{s.label}\t{s.image}" for s in failed))


if __name__ == "__main__":
    main()
