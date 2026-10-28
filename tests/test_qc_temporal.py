from __future__ import annotations

import pytest

from pipeline.qc_temporal import (
    arm_jump,
    bone_asymmetry,
    bone_deviation,
    elbow_angle_floor,
    joint_jump,
    torso_length,
)


def _frame(elbow=(30.0, 40.0), wrist=(50.0, 40.0), shoulder=(30.0, 20.0), upper3d=0.25, fore3d=0.22):
    """Right arm plus a torso 100 px tall (shoulder line y=20, hip line y=120)."""
    return {
        "LEFT_SHOULDER": (70.0, 20.0, 0.9, -0.1, 0.0, 0.0),
        "RIGHT_SHOULDER": (shoulder[0], shoulder[1], 0.9, 0.1, 0.0, 0.0),
        "LEFT_HIP": (65.0, 120.0, 0.9, 0.0, 0.5, 0.0),
        "RIGHT_HIP": (35.0, 120.0, 0.9, 0.0, 0.5, 0.0),
        "RIGHT_ELBOW": (elbow[0], elbow[1], 0.9, 0.1, upper3d, 0.0),
        "RIGHT_WRIST": (wrist[0], wrist[1], 0.9, 0.1 + fore3d, upper3d, 0.0),
        "LEFT_ELBOW": (70.0, 40.0, 0.9, -0.1, upper3d, 0.0),
        "LEFT_WRIST": (70.0, 60.0, 0.9, -0.1 - fore3d, upper3d, 0.0),
    }


RIGHT_ARM = ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST")
LEFT_ARM = ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST")


class TestTorsoLength:
    def test_distance_between_shoulder_and_hip_midpoints(self):
        assert torso_length(_frame()) == pytest.approx(100.0)

    def test_none_when_a_torso_joint_is_missing(self):
        frame = _frame()
        del frame["LEFT_HIP"]

        assert torso_length(frame) is None

    def test_none_for_missing_frame(self):
        assert torso_length(None) is None


class TestJointJump:
    def test_smooth_motion_has_no_jump(self):
        frames = [_frame(wrist=(50.0 + 5 * i, 40.0)) for i in range(3)]

        assert joint_jump(frames, 1, "RIGHT_WRIST") == pytest.approx(0.0)

    def test_single_frame_glitch_is_a_large_jump(self):
        frames = [_frame(wrist=(50, 40)), _frame(wrist=(50, 140)), _frame(wrist=(50, 40))]

        assert joint_jump(frames, 1, "RIGHT_WRIST") == pytest.approx(1.0)  # 100 px off, torso 100 px

    def test_jump_is_normalised_by_torso_length(self):
        frames = [_frame(wrist=(50, 40)), _frame(wrist=(50, 90)), _frame(wrist=(50, 40))]

        assert joint_jump(frames, 1, "RIGHT_WRIST") == pytest.approx(0.5)

    def test_none_without_both_neighbours(self):
        frames = [_frame(), _frame(), None]

        assert joint_jump(frames, 1, "RIGHT_WRIST") is None
        assert joint_jump(frames, 0, "RIGHT_WRIST") is None

    def test_arm_jump_is_the_largest_joint_jump(self):
        frames = [_frame(), _frame(elbow=(30, 90)), _frame()]

        assert arm_jump(frames, 1, RIGHT_ARM) == pytest.approx(0.5)

    def test_arm_jump_none_when_nothing_measurable(self):
        assert arm_jump([None, None, None], 1, RIGHT_ARM) is None


class TestBoneDeviation:
    def test_constant_bone_length_has_no_deviation(self):
        frames = [_frame() for _ in range(7)]

        assert bone_deviation(frames, 3, "RIGHT_SHOULDER", "RIGHT_ELBOW", radius=3) == pytest.approx(0.0)

    def test_stretched_bone_is_relative_to_the_neighbour_median(self):
        frames = [_frame() for _ in range(7)]
        frames[3] = _frame(upper3d=0.5)

        assert bone_deviation(frames, 3, "RIGHT_SHOULDER", "RIGHT_ELBOW", radius=3) == pytest.approx(1.0)

    def test_none_with_too_few_neighbours(self):
        frames = [None, None, _frame(), None, None]

        assert bone_deviation(frames, 2, "RIGHT_SHOULDER", "RIGHT_ELBOW", radius=2) is None


class TestBoneAsymmetry:
    def test_symmetric_arms_have_no_asymmetry(self):
        assert bone_asymmetry(_frame(), RIGHT_ARM, LEFT_ARM) == pytest.approx(0.0, abs=1e-6)

    def test_one_overlong_forearm_shows_up(self):
        frame = _frame(fore3d=0.22)
        frame["LEFT_WRIST"] = (70.0, 60.0, 0.9, -0.1 - 0.66, 0.25, 0.0)

        assert bone_asymmetry(frame, RIGHT_ARM, LEFT_ARM) > 0.5

    def test_none_when_a_joint_is_missing(self):
        frame = _frame()
        del frame["LEFT_WRIST"]

        assert bone_asymmetry(frame, RIGHT_ARM, LEFT_ARM) is None


class TestElbowAngleFloor:
    def test_flags_angles_below_the_floor(self):
        assert elbow_angle_floor(30.0, 40.0) is True

    def test_passes_angles_at_or_above_the_floor(self):
        assert elbow_angle_floor(40.0, 40.0) is False
        assert elbow_angle_floor(120.0, 40.0) is False

    def test_missing_angle_is_not_flagged(self):
        assert elbow_angle_floor(None, 40.0) is False
