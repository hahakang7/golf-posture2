"""Candidate QC rules that look beyond a single frame's visibility score.

The row-level QC (qc_tracked_features) misses most wrong arms: a pose that sits on another person or limb
still reports high visibility. These rules use what a correct pose must satisfy over time and across the body:
  - joint_jump / arm_jump : a joint should not leave the straight path between its neighbouring frames
  - bone_deviation        : a bone's 3D length is constant over a few frames
  - bone_asymmetry        : left and right arm bones have similar 3D length
  - elbow_angle_floor     : a golf elbow is never folded below a floor angle
Each returns a score (larger = more suspicious) or None when it cannot be measured; thresholds are chosen
separately against hand labels (qc_evaluate), not here.
"""

from __future__ import annotations

import math
import statistics
from typing import Mapping, Sequence

Joint = Sequence[float]  # (px, py, visibility, wx, wy, wz)
Frame = Mapping[str, Joint]

# Joint-jump threshold in torso lengths. 0.2 was the leave-one-video-out pick on the 6 user videos (caught 8/12
# wrong arms, 2 false alarms on 65 correct ones) but did not carry over to hand-labelled pro frames, where it
# flagged 31% of correct arms (fast swings move joints a lot between frames). 0.3 flags 15.5% of correct pro
# arms and catches 5/12 on the user videos. Chosen as the compromise; see qc_evaluate for the numbers.
JUMP_THRESHOLD = 0.3
# Elbow-angle floor in degrees, proposed before looking at any label results. On hand labels: no correct arm
# below it among 123 (user 65 + pro 58); it caught 3 extra wrong pro arms. The collapsed/blurred poses it
# targets measure 5-35 degrees, while the smallest correct elbow seen was ~50.
ELBOW_ANGLE_FLOOR_DEG = 40.0
MIN_NEIGHBOURS = 2
TORSO_JOINTS = ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP")


def _mid(a: Joint, b: Joint) -> tuple[float, float]:
    return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2


def torso_length(frame: Frame | None) -> float | None:
    """2D distance between the shoulder midpoint and the hip midpoint (pixels)."""
    if frame is None or any(name not in frame for name in TORSO_JOINTS):
        return None
    shoulders = _mid(frame["LEFT_SHOULDER"], frame["RIGHT_SHOULDER"])
    hips = _mid(frame["LEFT_HIP"], frame["RIGHT_HIP"])
    length = math.dist(shoulders, hips)
    return length if length > 0 else None


def joint_jump(frames: Sequence[Frame | None], index: int, joint: str) -> float | None:
    """Distance of a joint from the midpoint of its previous and next frame, in torso lengths."""
    if index < 1 or index + 1 >= len(frames):
        return None
    before, here, after = frames[index - 1], frames[index], frames[index + 1]
    if any(frame is None or joint not in frame for frame in (before, here, after)):
        return None
    scale = torso_length(here)
    if scale is None:
        return None
    expected = _mid(before[joint], after[joint])
    return math.dist(here[joint][0:2], expected) / scale


def arm_jump(frames: Sequence[Frame | None], index: int, joints: Sequence[str]) -> float | None:
    jumps = [value for value in (joint_jump(frames, index, name) for name in joints) if value is not None]
    return max(jumps) if jumps else None


def _bone_3d(frame: Frame | None, a: str, b: str) -> float | None:
    if frame is None or a not in frame or b not in frame:
        return None
    return math.dist(frame[a][3:6], frame[b][3:6])


def bone_deviation(frames: Sequence[Frame | None], index: int, a: str, b: str, radius: int) -> float | None:
    """Relative difference of a bone's 3D length from its median length in the surrounding frames."""
    if not 0 <= index < len(frames):
        return None
    here = _bone_3d(frames[index], a, b)
    lo, hi = max(0, index - radius), min(len(frames), index + radius + 1)
    others = [
        length for i in range(lo, hi) if i != index
        if (length := _bone_3d(frames[i], a, b)) is not None
    ]
    if here is None or len(others) < MIN_NEIGHBOURS:
        return None
    reference = statistics.median(others)
    return abs(here - reference) / reference if reference > 0 else None


def bone_asymmetry(frame: Frame | None, arm_a: Sequence[str], arm_b: Sequence[str]) -> float | None:
    """Largest relative left/right difference of the upper-arm and forearm 3D lengths."""
    if frame is None:
        return None
    differences = []
    for (a1, a2), (b1, b2) in zip(zip(arm_a, arm_a[1:]), zip(arm_b, arm_b[1:])):
        first, second = _bone_3d(frame, a1, a2), _bone_3d(frame, b1, b2)
        if first is None or second is None:
            return None
        mean = (first + second) / 2
        if mean <= 0:
            return None
        differences.append(abs(first - second) / mean)
    return max(differences) if differences else None


def elbow_angle_floor(angle: float | None, floor: float) -> bool:
    return angle is not None and angle < floor
