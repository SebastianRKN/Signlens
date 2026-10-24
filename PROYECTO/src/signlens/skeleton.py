"""Arm-side assignment. Single source of truth shared by features and viz."""
from __future__ import annotations

import numpy as np

# MediaPipe pose chains (shoulder, elbow, wrist) for indices 11/13/15 and 12/14/16.
_CHAIN_A = (11, 13, 15)
_CHAIN_B = (12, 14, 16)


def arm_chains(pose: np.ndarray, mirrored: bool = True) -> dict[str, tuple[int, int, int]]:
    """Map the USER's real Left/Right to a pose chain using on-screen position, so the result
    does not depend on how the model labels anatomy in a flipped image.
    Mirrored (selfie) view: the user's left arm is on the left side of the image."""
    a_is_image_left = pose[_CHAIN_A[0], 0] <= pose[_CHAIN_B[0], 0]
    left_chain, right_chain = (_CHAIN_A, _CHAIN_B) if a_is_image_left else (_CHAIN_B, _CHAIN_A)
    if not mirrored:  # raw camera view: user's left appears on the image's right
        left_chain, right_chain = right_chain, left_chain
    return {"Left": left_chain, "Right": right_chain}
