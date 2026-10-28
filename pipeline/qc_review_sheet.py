"""Build arm-review sheets: cropped event frames with the tracked arm drawn, plus a verdict CSV to fill in.

The tracked extraction scores an elbow only if it passes QC (qc_tracked_features). Nothing measures how often
a pose that passes QC is actually wrong, or how often a correct pose is discarded. This module produces the
material to label that: for every event and arm, a crop of the frame with shoulder/elbow/wrist drawn and the
scored value, and a CSV with an empty `verdict` column (ok / wrong / unsure) plus a note.

Verdicts are a reviewer's judgement of whether the drawn arm sits on the real arm, not pixel-accurate ground
truth: they can expose gross errors (wrong person, wrong limb) but not a few degrees of angle error.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

from pipeline.extract_pro_features_tracked import arm_geometry, in_frame, video_size
from pipeline.pose_tracking import FrameJoints, angle_at, arm_by_feature, extract_tracked_frames
from pipeline.qc_tracked_features import arm_flags
from pipeline.run_analysis import EVENT_NAMES, video_frame_count
from pipeline.user_tracking import tracking_window

DEFAULT_MARGIN = 0.6
MIN_BOX_SIDE = 40
TARGET_CROP_SIDE = 360
LABEL_FIELDS = [
    "video", "event_index", "event_name", "arm", "frame_index", "elbow_angle", "qc_flagged", "qc_flags",
    "min_vis", "image", "verdict", "note",
]


def arm_crop_box(
    points: Sequence[Sequence[float]], frame_size: tuple[int, int], margin: float = DEFAULT_MARGIN
) -> tuple[int, int, int, int]:
    """Crop box around the arm points, padded by `margin` of its size, clamped to the frame, never empty."""
    width, height = frame_size
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    side = max(max(xs) - min(xs), max(ys) - min(ys), MIN_BOX_SIDE)
    pad = side * margin
    x0 = int(max(0, min(xs) - pad))
    y0 = int(max(0, min(ys) - pad))
    x1 = int(min(width, max(xs) + pad))
    y1 = int(min(height, max(ys) + pad))
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def _qc_row(frame: FrameJoints | None, arm_names: Sequence[str], frame_size: tuple[int, int]) -> dict[str, Any]:
    present = frame is not None and all(name in frame for name in arm_names)
    row: dict[str, Any] = {
        "min_vis": min(frame[name][2] for name in arm_names) if present else None,
        "in_frame": all(in_frame(frame[name], frame_size) for name in arm_names) if present else False,
    }
    row.update(arm_geometry(frame, arm_names))
    return row


def build_review_items(
    video: str,
    frames: Sequence[FrameJoints | None],
    frame_indices: Sequence[int],
    frame_size: tuple[int, int],
    handedness: str,
) -> list[dict[str, Any]]:
    """One item per event and arm: drawn points, the single-frame 2D elbow angle and the row-level QC result."""
    arms = arm_by_feature(handedness)
    arm_for = {"trail": arms["trail_elbow_angle"], "lead": arms["lead_elbow_angle"]}
    items = []
    for event_index, frame_index in enumerate(frame_indices):
        frame = frames[frame_index] if 0 <= frame_index < len(frames) else None
        for arm, names in arm_for.items():
            row = _qc_row(frame, names, frame_size)
            present = frame is not None and all(name in frame for name in names)
            points = [(frame[name][0], frame[name][1]) for name in names] if present else None
            flags = arm_flags({f"{arm}_{key}": value for key, value in row.items()}, arm) if present else {}
            items.append({
                "video": video,
                "event_index": event_index,
                "event_name": EVENT_NAMES[event_index],
                "arm": arm,
                "frame_index": frame_index,
                "points": points,
                "elbow_angle": angle_at(points) if points else None,
                "min_vis": row["min_vis"],
                "qc_flags": [name for name, hit in flags.items() if hit],
                "qc_flagged": any(flags.values()),
            })
    return items


# ----------------------------------------------------------------- rendering / CLI
def render_item(image: Any, item: dict[str, Any]) -> Any:
    """Crop `image` around the arm and draw shoulder-elbow-wrist; returns a new image."""
    import cv2

    points = item["points"]
    if not points:
        return image.copy()
    height, width = image.shape[:2]
    x0, y0, x1, y1 = arm_crop_box(points, (width, height))
    crop = image[y0:y1, x0:x1].copy()
    pts = [(int(x - x0), int(y - y0)) for x, y in points]
    cv2.line(crop, pts[0], pts[1], (0, 255, 255), 2)
    cv2.line(crop, pts[1], pts[2], (0, 255, 255), 2)
    for point, color in zip(pts, [(255, 0, 0), (0, 0, 255), (0, 200, 0)]):
        cv2.circle(crop, point, 5, color, -1)
    scale = max(1.0, TARGET_CROP_SIDE / max(crop.shape[:2]))
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def event_frames_from_result(result_json: Path) -> list[int]:
    result = json.loads(result_json.read_text(encoding="utf-8"))
    return [int(result["events"][name]["frame_index"]) for name in EVENT_NAMES]


def review_video(video_path: Path, event_frames: Sequence[int], handedness: str, out_dir: Path) -> list[dict[str, Any]]:
    import cv2

    start, last = tracking_window(event_frames, video_frame_count(video_path))
    frames = extract_tracked_frames(video_path, last, start_frame=start)
    indices = [frame - start for frame in event_frames]
    items = build_review_items(video_path.stem, frames, indices, video_size(video_path), handedness)
    capture = cv2.VideoCapture(str(video_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        for item in items:
            item["frame_index"] = event_frames[item["event_index"]]
            item["image"] = ""
            if not item["points"]:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, item["frame_index"])
            ok, bgr = capture.read()
            if ok:
                name = f"{item['video']}_{item['event_index']}_{item['arm']}.jpg"
                cv2.imwrite(str(out_dir / name), render_item(bgr, item))
                item["image"] = name
    finally:
        capture.release()
    return items


def write_label_sheet(items: Sequence[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for item in items:
            writer.writerow({
                **item,
                "elbow_angle": "" if item["elbow_angle"] is None else round(item["elbow_angle"], 1),
                "min_vis": "" if item["min_vis"] is None else round(item["min_vis"], 2),
                "qc_flags": "|".join(item["qc_flags"]),
                "verdict": "",
                "note": "",
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path, help="analysis result JSON holding the event frames")
    parser.add_argument("--handedness", default="right", choices=["right", "left"])
    parser.add_argument("--output-dir", default="outputs/qc_review_labels", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir / args.video.stem
    items = review_video(args.video, event_frames_from_result(args.result_json), args.handedness, out_dir)
    write_label_sheet(items, out_dir / "labels.csv")
    print(f"{len(items)} items -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
