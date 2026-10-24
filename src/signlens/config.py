"""Central, immutable configuration. Every tunable lives here."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # Capture (640x480 MJPG @ 60 is what most webcams can actually sustain)
    camera_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 60
    mirror: bool = True  # selfie view
    camera_probe: bool = True  # measure real FPS per backend/mode and keep the fastest (cached)

    # Inference
    infer_width: int = 640  # frames wider than this are downscaled before inference
    pose_model: str = "lite"  # "lite" (fast) | "full" (more accurate)
    num_hands: int = 2
    min_detection_conf: float = 0.5
    min_presence_conf: float = 0.5
    min_tracking_conf: float = 0.5
    min_pose_visibility: float = 0.5

    # Smoothing (One Euro). Lower min_cutoff = steadier at rest; higher beta = less lag in motion.
    # Coordinates are normalized (0..1), so beta must be large for the filter to open up when moving.
    smoothing: bool = True
    hand_min_cutoff: float = 1.2
    hand_beta: float = 10.0
    pose_min_cutoff: float = 1.0
    pose_beta: float = 4.0

    # Hand robustness
    hold_frames: int = 4  # keep the last good hand this many frames through a dropout
    label_penalty: float = 0.1  # soft prior for MediaPipe's Left/Right label (image-width units)
    max_speed: float = 12.0  # max plausible wrist speed, image-widths/second
    min_jump: float = 0.12  # always-allowed per-frame wrist movement
    max_size_ratio: float = 2.2  # max frame-to-frame palm size change
    idle_stride: int = 2  # with no hand tracked, look for one every N frames (palm detection is the costly part)
    pose_max_age_ms: int = 250  # async pose older than this is ignored instead of reused

    # Storage
    models_dir: Path = Path.home() / ".signlens" / "models"
    data_dir: Path = Path("data")
