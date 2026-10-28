"""Build a single self-contained HTML page for hand-labelling arm joints on event frames.

Open the generated file in any browser (no install, no server): zoom and pan the frame, click shoulder, elbow
and wrist of each arm, mark hidden joints, then export the labels as JSON. The page shows the bare frame only,
never a model's joints, so the labels stay independent of the pose model they will be used to evaluate.
Coordinates are exported in original video-frame pixels.
"""

from __future__ import annotations

import argparse
import base64
import json
import random
from pathlib import Path
from typing import Any, Sequence

from pipeline.extract_pro_features_tracked import video_size
from pipeline.qc_review_sheet import event_frames_from_result
from pipeline.run_analysis import EVENT_NAMES

PLACEHOLDER = "/*__TASKS__*/[]"
DEFAULT_TEMPLATE = Path(__file__).with_name("arm_label_template.html")
FRAME_MAX_SIDE = 1000
JPEG_QUALITY = 82
DEFAULT_SEED = 20261014
DEFAULT_VIDEOS = ("longform", "shorts", "shorts2", "test1", "test2", "test3")


def scale_for(size: tuple[int, int], max_side: int) -> float:
    """Factor that fits the longer side within max_side; frames are never upscaled."""
    return min(1.0, max_side / max(size))


def shuffled_order(count: int, seed: int) -> list[int]:
    order = list(range(count))
    random.Random(seed).shuffle(order)
    return order


def label_task(
    video: str,
    event_index: int,
    event_name: str,
    frame_index: int,
    size: tuple[int, int],
    scale: float,
    image: str,
) -> dict[str, Any]:
    """One frame to label. Holds geometry only; no model output."""
    return {
        "id": f"{video}_{event_index}",
        "video": video,
        "event_index": event_index,
        "event_name": event_name,
        "frame_index": frame_index,
        "width": size[0],
        "height": size[1],
        "scale": scale,
        "image": image,
    }


def render_html(template: str, tasks: Sequence[dict[str, Any]]) -> str:
    if PLACEHOLDER not in template:
        raise ValueError(f"template has no {PLACEHOLDER} placeholder")
    payload = json.dumps(list(tasks), ensure_ascii=False).replace("</", "<\\/")
    return template.replace(PLACEHOLDER, payload)


# ------------------------------------------------------------------------- IO
def _frame_data_url(video_path: Path, frame_index: int, max_side: int) -> tuple[str, float] | None:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr = capture.read()
    finally:
        capture.release()
    if not ok:
        return None
    scale = scale_for((bgr.shape[1], bgr.shape[0]), max_side)
    if scale < 1.0:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1]
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"), scale


def build_tasks_from_items(
    items: Sequence[tuple[str, Path, int, str, int]], max_side: int
) -> list[dict[str, Any]]:
    """Tasks for (name, video path, event index, event name, frame index) items."""
    tasks = []
    for name, video_path, event_index, event_name, frame_index in items:
        rendered = _frame_data_url(video_path, frame_index, max_side)
        if rendered is None:
            raise RuntimeError(f"could not read frame {frame_index} of {video_path}")
        image, scale = rendered
        tasks.append(label_task(name, event_index, event_name, frame_index, video_size(video_path), scale, image))
    return tasks


def build_tasks(videos: Sequence[str], videos_dir: Path, results_dir: Path, max_side: int) -> list[dict[str, Any]]:
    items = []
    for name in videos:
        frames = event_frames_from_result(results_dir / name / f"{name}_result.json")
        items += [
            (name, videos_dir / f"{name}.mp4", event_index, EVENT_NAMES[event_index], frame_index)
            for event_index, frame_index in enumerate(frames)
        ]
    return build_tasks_from_items(items, max_side)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", nargs="+", default=list(DEFAULT_VIDEOS))
    parser.add_argument("--videos-dir", type=Path, default=Path("raw_videos"))
    parser.add_argument("--results-dir", type=Path, default=Path("documents/baseline_eval/2026-10-13"))
    parser.add_argument("--output", type=Path, default=Path("outputs/arm_label_tool.html"))
    parser.add_argument("--max-side", type=int, default=FRAME_MAX_SIDE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = build_tasks(args.videos, args.videos_dir, args.results_dir, args.max_side)
    tasks = [tasks[i] for i in shuffled_order(len(tasks), args.seed)]
    html = render_html(DEFAULT_TEMPLATE.read_text(encoding="utf-8"), tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"{len(tasks)} frames -> {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
