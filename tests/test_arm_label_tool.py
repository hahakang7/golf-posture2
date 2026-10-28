from __future__ import annotations

import json

import pytest

from pipeline.arm_label_tool import label_task, render_html, scale_for, shuffled_order

TEMPLATE = "<script>const TASKS = /*__TASKS__*/[];</script>"


class TestScaleFor:
    def test_large_frame_is_scaled_down_to_max_side(self):
        assert scale_for((2000, 1000), 1000) == pytest.approx(0.5)

    def test_small_frame_is_never_upscaled(self):
        assert scale_for((400, 300), 1000) == 1.0

    def test_uses_the_longer_side(self):
        assert scale_for((1000, 2000), 1000) == pytest.approx(0.5)


class TestShuffledOrder:
    def test_is_a_permutation(self):
        assert sorted(shuffled_order(10, seed=1)) == list(range(10))

    def test_same_seed_gives_same_order(self):
        assert shuffled_order(20, seed=7) == shuffled_order(20, seed=7)

    def test_different_seed_gives_different_order(self):
        assert shuffled_order(20, seed=1) != shuffled_order(20, seed=2)


class TestLabelTask:
    def test_task_records_original_frame_geometry_and_scale(self):
        task = label_task("test2", 3, "Top", 179, (1080, 1920), 0.5, "data:image/jpeg;base64,AAA")

        assert task["id"] == "test2_3"
        assert task["video"] == "test2"
        assert task["event_index"] == 3
        assert task["frame_index"] == 179
        assert (task["width"], task["height"]) == (1080, 1920)
        assert task["scale"] == 0.5
        assert task["image"].startswith("data:image/jpeg")

    def test_task_does_not_carry_any_model_output(self):
        task = label_task("v", 0, "Address", 1, (10, 10), 1.0, "data:,")

        assert set(task) == {"id", "video", "event_index", "event_name", "frame_index", "width", "height", "scale", "image"}


class TestRenderHtml:
    def test_tasks_are_embedded_as_parseable_json(self):
        tasks = [label_task("v", 0, "Address", 1, (10, 10), 1.0, "data:,")]

        html = render_html(TEMPLATE, tasks)
        payload = html.split("const TASKS = ")[1].split(";</script>")[0]

        assert json.loads(payload)[0]["id"] == "v_0"

    def test_closing_script_tag_in_data_cannot_break_out(self):
        tasks = [label_task("</script><b>", 0, "Address", 1, (10, 10), 1.0, "data:,")]

        html = render_html(TEMPLATE, tasks)

        assert html.count("</script>") == 1

    def test_missing_placeholder_is_an_error(self):
        with pytest.raises(ValueError):
            render_html("<script>no placeholder</script>", [])
