from __future__ import annotations

import pytest

from pipeline.qc_evaluate import (
    arm_error_fraction,
    choose_threshold,
    confusion,
    leave_one_group_out,
)

HUMAN = [(0.0, 0.0), (100.0, 0.0), (200.0, 0.0)]  # arm length 200


class TestArmErrorFraction:
    def test_identical_points_have_zero_error(self):
        assert arm_error_fraction(HUMAN, HUMAN) == pytest.approx(0.0)

    def test_error_is_the_worst_joint_over_arm_length(self):
        model = [(0.0, 0.0), (100.0, 50.0), (200.0, 0.0)]

        assert arm_error_fraction(HUMAN, model) == pytest.approx(0.25)

    def test_none_when_either_side_is_missing_a_joint(self):
        assert arm_error_fraction(HUMAN, [(0.0, 0.0), None, (200.0, 0.0)]) is None
        assert arm_error_fraction([(0.0, 0.0), None, (200.0, 0.0)], HUMAN) is None

    def test_none_for_degenerate_human_arm(self):
        assert arm_error_fraction([(1.0, 1.0)] * 3, HUMAN) is None


class TestConfusion:
    def test_counts_catches_false_alarms_and_misses(self):
        flags = [True, True, False, False]
        bad = [True, False, True, False]

        assert confusion(flags, bad) == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}


class TestChooseThreshold:
    def test_picks_the_threshold_that_separates_bad_from_fine(self):
        scores = [0.1, 0.2, 0.3, 0.9, 1.0]
        bad = [False, False, False, True, True]

        assert choose_threshold(scores, bad, [0.05, 0.5, 0.95], "high") == 0.5

    def test_low_direction_flags_scores_below_the_threshold(self):
        scores = [150, 160, 170, 20, 30]
        bad = [False, False, False, True, True]

        assert choose_threshold(scores, bad, [40, 100, 160], "low") == 100

    def test_unmeasurable_scores_are_never_flagged(self):
        scores = [None, None, 1.0]
        bad = [True, True, False]

        assert choose_threshold(scores, bad, [0.5], "high") == 0.5

    def test_false_alarms_are_weighted_against_catches(self):
        scores = [0.9, 0.9, 0.9, 0.9, 0.95]
        bad = [False, False, False, False, True]

        assert choose_threshold(scores, bad, [0.5, 0.93], "high", false_alarm_cost=2.0) == 0.93


class TestLeaveOneGroupOut:
    def test_threshold_is_chosen_without_the_held_out_group(self):
        rows = (
            [{"group": "a", "score": 0.9, "bad": True}, {"group": "a", "score": 0.1, "bad": False}]
            + [{"group": "b", "score": 0.8, "bad": True}, {"group": "b", "score": 0.2, "bad": False}]
            + [{"group": "c", "score": 0.7, "bad": True}, {"group": "c", "score": 0.3, "bad": False}]
        )

        result = leave_one_group_out(rows, "score", [0.5], "high")

        assert result == {"tp": 3, "fp": 0, "fn": 0, "tn": 3}

    def test_rows_without_a_score_count_as_not_flagged(self):
        rows = [{"group": "a", "score": None, "bad": True}, {"group": "b", "score": 0.9, "bad": True},
                {"group": "b", "score": 0.1, "bad": False}]

        result = leave_one_group_out(rows, "score", [0.5], "high")

        assert result["fn"] == 1 and result["tp"] == 1
