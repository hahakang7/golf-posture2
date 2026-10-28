from __future__ import annotations

import pytest

from pipeline.pro_label_sample import hardness_scores, select_pro_frames


def _tracked(video_id, event=0, trail=150.0, lead=150.0, frame=10):
    return {
        "video_id": video_id, "event_index": event, "event_name": f"E{event}", "frame_index": frame,
        "pose_detected": True, "trail_elbow_angle_2d": trail, "lead_elbow_angle_2d": lead,
    }


def _static(video_id, event=0, trail=150.0, lead=150.0):
    return {"video_id": video_id, "event_index": event, "trail_elbow_angle": trail, "lead_elbow_angle": lead}


def _labels(count, slow_ids=()):
    return {v: {"youtube_id": f"yt{v}", "slow": 1 if v in slow_ids else 0} for v in range(count)}


class TestHardnessScores:
    def test_is_the_largest_static_tracked_disagreement_of_the_two_arms(self):
        scores = hardness_scores([_tracked(1, trail=150, lead=100)], [_static(1, trail=140, lead=160)])

        assert scores[(1, 0)] == pytest.approx(60.0)

    def test_value_recovered_by_tracking_counts_as_a_fixed_medium_score(self):
        scores = hardness_scores([_tracked(1, trail=150)], [_static(1, trail=None)])

        assert scores[(1, 0)] == pytest.approx(90.0)

    def test_frame_without_any_comparable_angle_has_no_score(self):
        scores = hardness_scores([_tracked(1, trail=None, lead=None)], [_static(1, trail=None, lead=None)])

        assert (1, 0) not in scores

    def test_missing_static_row_is_skipped(self):
        assert hardness_scores([_tracked(1)], []) == {}


class TestSelectProFrames:
    def _data(self, n=30):
        tracked = [_tracked(v, event=e, trail=100 + v) for v in range(n) for e in (0, 3)]
        static = [_static(v, event=e, trail=100) for v in range(n) for e in (0, 3)]
        return tracked, static, _labels(n)

    def test_returns_the_requested_number_of_frames_by_kind(self):
        tracked, static, labels = self._data()

        picks = select_pro_frames(tracked, static, labels, n_hard=5, n_random=4, seed=1)

        assert sum(p["kind"] == "hard" for p in picks) == 5
        assert sum(p["kind"] == "random" for p in picks) == 4

    def test_hard_picks_are_the_largest_disagreements(self):
        tracked, static, labels = self._data()

        picks = select_pro_frames(tracked, static, labels, n_hard=3, n_random=0, seed=1)

        assert {p["video_id"] for p in picks} == {27, 28, 29}

    def test_each_video_and_youtube_id_appears_once(self):
        tracked, static, labels = self._data()
        labels[5]["youtube_id"] = labels[6]["youtube_id"] = "shared"

        picks = select_pro_frames(tracked, static, labels, n_hard=10, n_random=10, seed=2)

        assert len({p["video_id"] for p in picks}) == len(picks)
        assert len({labels[p["video_id"]]["youtube_id"] for p in picks}) == len(picks)

    def test_slow_motion_videos_are_never_picked(self):
        tracked, static, _ = self._data()
        labels = _labels(30, slow_ids={27, 28, 29})

        picks = select_pro_frames(tracked, static, labels, n_hard=5, n_random=5, seed=3)

        assert not {p["video_id"] for p in picks} & {27, 28, 29}

    def test_same_seed_gives_the_same_picks(self):
        tracked, static, labels = self._data()

        first = select_pro_frames(tracked, static, labels, n_hard=3, n_random=5, seed=7)
        second = select_pro_frames(tracked, static, labels, n_hard=3, n_random=5, seed=7)

        assert first == second

    def test_frames_without_a_detected_pose_are_not_candidates(self):
        tracked, static, labels = self._data()
        for row in tracked:
            if row["video_id"] == 29:
                row["pose_detected"] = False

        picks = select_pro_frames(tracked, static, labels, n_hard=3, n_random=0, seed=1)

        assert 29 not in {p["video_id"] for p in picks}

    def test_fewer_candidates_than_requested_returns_what_exists(self):
        tracked, static, labels = self._data(n=3)

        picks = select_pro_frames(tracked, static, labels, n_hard=5, n_random=5, seed=1)

        assert len(picks) == 3
