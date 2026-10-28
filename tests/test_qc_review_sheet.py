from __future__ import annotations

import pytest

from pipeline.qc_review_sheet import arm_crop_box, build_review_items

FRAME_SIZE = (200, 100)
LANDMARKS = [
    "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST",
    "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
]


def _joint(px, py, vis=0.9, wx=0.0, wy=0.0, wz=0.0):
    return (px, py, vis, wx, wy, wz)


def _frame(vis: float = 0.9):
    frame = {name: _joint(100, 50, vis) for name in LANDMARKS}
    frame.update({
        "RIGHT_SHOULDER": _joint(60, 20, vis, 0.1, 0.0, 0.0),
        "RIGHT_ELBOW": _joint(60, 40, vis, 0.1, 0.25, 0.0),
        "RIGHT_WRIST": _joint(80, 40, vis, 0.32, 0.25, 0.0),
        "LEFT_SHOULDER": _joint(140, 20, vis, -0.1, 0.0, 0.0),
        "LEFT_ELBOW": _joint(140, 40, vis, -0.1, 0.25, 0.0),
        "LEFT_WRIST": _joint(140, 60, vis, -0.1, 0.5, 0.0),
    })
    return frame


class TestArmCropBox:
    def test_box_contains_all_points_with_margin(self):
        x0, y0, x1, y1 = arm_crop_box([(60, 20), (60, 40), (80, 40)], FRAME_SIZE, margin=0.5)

        assert x0 <= 60 and x1 >= 80 and y0 <= 20 and y1 >= 40

    def test_box_is_clamped_to_the_frame(self):
        x0, y0, x1, y1 = arm_crop_box([(2, 2), (10, 10)], FRAME_SIZE, margin=5.0)

        assert (x0, y0) == (0, 0)
        assert x1 <= FRAME_SIZE[0] and y1 <= FRAME_SIZE[1]

    def test_box_is_never_degenerate(self):
        x0, y0, x1, y1 = arm_crop_box([(50, 50), (50, 50)], FRAME_SIZE, margin=0.3)

        assert x1 > x0 and y1 > y0

    def test_points_outside_the_frame_are_still_clamped(self):
        x0, y0, x1, y1 = arm_crop_box([(-30, 20), (60, 140)], FRAME_SIZE, margin=0.2)

        assert x0 >= 0 and y1 <= FRAME_SIZE[1]


class TestBuildReviewItems:
    def test_one_item_per_event_and_arm(self):
        frames = [_frame(), _frame()]

        items = build_review_items("v", frames, [0, 1], FRAME_SIZE, "right")

        assert len(items) == 4
        assert {item["arm"] for item in items} == {"trail", "lead"}

    def test_item_carries_the_scored_value_and_qc_flags(self):
        items = build_review_items("v", [_frame()], [0], FRAME_SIZE, "right")
        trail = next(item for item in items if item["arm"] == "trail")

        assert trail["video"] == "v"
        assert trail["event_index"] == 0
        assert trail["elbow_angle"] == pytest.approx(90.0)
        assert trail["qc_flagged"] is False
        assert trail["points"] == [(60, 20), (60, 40), (80, 40)]

    def test_low_visibility_arm_is_flagged(self):
        frame = _frame()
        frame["RIGHT_WRIST"] = _joint(80, 40, 0.1, 0.32, 0.25, 0.0)

        items = build_review_items("v", [frame], [0], FRAME_SIZE, "right")
        trail = next(item for item in items if item["arm"] == "trail")

        assert trail["qc_flagged"] is True
        assert "low_vis" in trail["qc_flags"]

    def test_missing_pose_gives_item_without_points(self):
        items = build_review_items("v", [None], [0], FRAME_SIZE, "right")

        assert all(item["points"] is None for item in items)
        assert all(item["elbow_angle"] is None for item in items)

    def test_left_handed_swaps_which_arm_is_trail(self):
        frame = _frame()

        right = {i["arm"]: i["points"] for i in build_review_items("v", [frame], [0], FRAME_SIZE, "right")}
        left = {i["arm"]: i["points"] for i in build_review_items("v", [frame], [0], FRAME_SIZE, "left")}

        assert right["trail"] == left["lead"]
        assert right["lead"] == left["trail"]
