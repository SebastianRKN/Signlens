"""FrameResult -> fixed-size feature vector. This is the exact input of the sign classifier.

Layout (FEATURE_DIM = 147, FEATURE_VERSION = 2):
  per hand (x2, Left then Right), 66 values each:
    [0:63]  hand SHAPE: metric world landmarks, wrist-centred, divided by wrist->middle-MCP length.
            Invariant to position, camera distance and hand size; keeps palm orientation.
    [63:66] hand LOCATION: wrist position relative to the shoulder midpoint, in shoulder-widths
            (x, y from the hand model; z = depth of the arm's wrist in front of the body).
  [132:144] left elbow, right elbow, left wrist, right wrist (4x3), shoulder-centred/scaled
  [144:147] presence flags: left hand, right hand, pose
Missing parts are zeros with flag 0, so a model can tell "absent" from "at the origin"."""
from __future__ import annotations

import numpy as np

from signlens.skeleton import arm_chains
from signlens.types import FrameResult, Hand

FEATURE_VERSION = 2
HAND_SHAPE = 63
HAND_DIM = HAND_SHAPE + 3
ARMS_AT = 2 * HAND_DIM
FLAGS_AT = ARMS_AT + 12
FEATURE_DIM = FLAGS_AT + 3
_EPS = 1e-6


def oriented_shape(world: np.ndarray) -> np.ndarray:
    """(21,3) world landmarks -> wrist-centred, hand-size-normalised. Keeps palm orientation."""
    w = world - world[0]
    return (w / (np.linalg.norm(w[9]) + _EPS)).astype(np.float32)


def canonical_shape(world: np.ndarray) -> np.ndarray:
    """Like oriented_shape but expressed in the hand's own frame (y: wrist->middle MCP, x: across the
    knuckles, z: palm normal). Invariant to 3-D rotation; mirroring the hand == negating z."""
    w = world - world[0]
    y = w[9] / (np.linalg.norm(w[9]) + _EPS)
    x = w[5] - w[17]
    x = x - (x @ y) * y
    x /= np.linalg.norm(x) + _EPS
    frame = np.stack([x, y, np.cross(x, y)])
    return ((frame @ w.T).T / (np.linalg.norm(w[9]) + _EPS)).astype(np.float32)


_FINGERS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
_TIPS = (4, 8, 12, 16, 20)


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + _EPS), -1.0, 1.0)))


def engineered_shape(world: np.ndarray) -> np.ndarray:
    """42 rotation-invariant descriptors of finger configuration: joint flexion angles (15),
    fingertip-fingertip (10) and fingertip-wrist (5) distances, thumb-to-knuckle distances (8) and
    the spread between neighbouring fingers (4). They separate look-alike letters far better than raw
    coordinates because they ignore camera angle and hand tilt."""
    w = world - world[0]
    w = w / (np.linalg.norm(w[9]) + _EPS)
    out = []
    for a, b, c, d in _FINGERS:
        out += [_angle(w[0] - w[a], w[b] - w[a]), _angle(w[a] - w[b], w[c] - w[b]), _angle(w[b] - w[c], w[d] - w[c])]
    tips = w[list(_TIPS)]
    out += [float(np.linalg.norm(tips[i] - tips[j])) for i in range(5) for j in range(i + 1, 5)]
    out += [float(np.linalg.norm(t)) for t in tips]
    out += [float(np.linalg.norm(w[4] - w[k])) for k in (5, 9, 13, 17, 6, 10, 14, 18)]
    dirs = [(w[d] - w[a]) / (np.linalg.norm(w[d] - w[a]) + _EPS) for a, _, _, d in _FINGERS]
    out += [_angle(dirs[i], dirs[i + 1]) for i in range(4)]
    return np.asarray(out, dtype=np.float32)


def oriented_eng_shape(world: np.ndarray) -> np.ndarray:
    """Default for static signs: oriented coordinates (63) + engineered descriptors (42)."""
    return np.concatenate([oriented_shape(world).ravel(), engineered_shape(world)])


FEATURE_MODES = {"oriented": oriented_shape, "canonical": canonical_shape, "oriented_eng": oriented_eng_shape}


def static_features(world: np.ndarray, mode: str) -> np.ndarray:
    """(N,21,3) world landmarks -> (N,D) features for single-frame (static sign) classifiers."""
    fn = FEATURE_MODES[mode]
    return np.stack([np.asarray(fn(w)).ravel() for w in world])


def mirror_world(world: np.ndarray) -> np.ndarray:
    """Left<->right hand. Label-preserving for letters; used as augmentation."""
    return world * np.array([-1, 1, 1], dtype=world.dtype)


def hand_shape(hand: Hand) -> np.ndarray:
    return oriented_shape(hand.world).ravel()


def _body_frame(pose: np.ndarray, aspect: float, min_vis: float, mirrored: bool):
    chains = arm_chains(pose, mirrored)
    ls, rs = chains["Left"][0], chains["Right"][0]
    if pose[[ls, rs], 3].min() < min_vis:
        return None
    p = pose[:, :3].copy()
    p[:, 0] *= aspect
    center = (p[ls] + p[rs]) / 2
    width = float(np.linalg.norm(p[ls, :2] - p[rs, :2])) + _EPS
    return p, chains, center, width


def extract(result: FrameResult, min_pose_vis: float = 0.5) -> np.ndarray:
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    body = None
    if result.pose is not None:
        body = _body_frame(result.pose, result.aspect, min_pose_vis, result.mirrored)
        if body is not None:
            p, chains, center, width = body
            idx = [chains["Left"][1], chains["Right"][1], chains["Left"][2], chains["Right"][2]]
            vec[ARMS_AT:FLAGS_AT] = ((p[idx] - center) / width).astype(np.float32).ravel()
            vec[FLAGS_AT + 2] = 1.0

    for i, side in enumerate(("Left", "Right")):
        hand = result.hands.get(side)
        if hand is None or hand.stale:
            continue
        base = i * HAND_DIM
        vec[base:base + HAND_SHAPE] = hand_shape(hand)
        if body is not None:
            p, chains, center, width = body
            wrist = np.array([hand.pts[0, 0] * result.aspect, hand.pts[0, 1], p[chains[side][2], 2]])
            vec[base + HAND_SHAPE:base + HAND_DIM] = (wrist - center) / width
        vec[FLAGS_AT + i] = 1.0
    return vec


def _random_rotation(rng: np.random.Generator, sigma_deg: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + _EPS
    th = np.deg2rad(rng.normal(0.0, sigma_deg))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * (k @ k)


def augment_world(world: np.ndarray, rng: np.random.Generator, k: int = 4,
                  sigma_deg: float = 12.0, noise: float = 0.02, mirror_p: float = 0.5) -> list[np.ndarray]:
    """k augmented copies of (N,21,3) world landmarks: small random 3-D rotations (camera/hand tilt),
    landmark noise (2% of palm length, mimicking tracker jitter) and random left/right mirroring."""
    copies = []
    for _ in range(k):
        out = np.empty_like(world)
        for i, w in enumerate(world):
            rot = _random_rotation(rng, sigma_deg)
            out[i] = w @ rot.T + rng.normal(0.0, noise * np.linalg.norm(w[9] - w[0]), w.shape)
        copies.append(mirror_world(out) if rng.random() < mirror_p else out)
    return copies
