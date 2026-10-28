"""Pick pro-video event frames for hand-labelling and build the labelling page for them.

Extends the 48 user-video labels so the QC rules can be checked on more varied footage. The sample mixes
  - 'hard' frames: where the per-frame (static) and video-tracked elbow angles disagree most, which are
    the frames most likely to contain a wrong arm, and
  - 'random' frames: an unbiased draw, so false-alarm rates are not distorted by the hard picks.
Only normal-speed videos and one frame per video and per youtube_id are used, because slow-motion copies
are the same swing. Which kind a frame is stays out of the labelling page (it is saved beside it) so the
labeller is not primed.

Run: pixi run python -m pipeline.pro_label_sample
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipeline.arm_label_tool import FRAME_MAX_SIDE, build_tasks_from_items, render_html, shuffled_order

RECOVERED_SCORE = 90.0
DEFAULT_SEED = 20261028
ARMS = ("trail", "lead")
Row = Mapping[str, Any]


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    return None if math.isnan(number) else number


def hardness_scores(tracked_rows: Sequence[Row], static_rows: Sequence[Row]) -> dict[tuple[int, int], float]:
    """Per (video, event): the largest static-vs-tracked elbow-angle disagreement of the two arms.

    A value that only the tracked extraction produced gets a fixed medium score (it is unverified, not
    necessarily wrong). Frames with nothing to compare are left out.
    """
    static_by_key = {(int(r["video_id"]), int(r["event_index"])): r for r in static_rows}
    scores: dict[tuple[int, int], float] = {}
    for row in tracked_rows:
        key = (int(row["video_id"]), int(row["event_index"]))
        static = static_by_key.get(key)
        if static is None:
            continue
        values = []
        for arm in ARMS:
            tracked_angle = _number(row.get(f"{arm}_elbow_angle_2d"))
            static_angle = _number(static.get(f"{arm}_elbow_angle"))
            if tracked_angle is None:
                continue
            values.append(RECOVERED_SCORE if static_angle is None else abs(tracked_angle - static_angle))
        if values:
            scores[key] = max(values)
    return scores


def _row_key(row: Row) -> tuple[int, int]:
    return int(row["video_id"]), int(row["event_index"])


def select_pro_frames(
    tracked_rows: Sequence[Row],
    static_rows: Sequence[Row],
    labels: Mapping[int, Mapping[str, Any]],
    n_hard: int,
    n_random: int,
    seed: int,
) -> list[dict[str, Any]]:
    scores = hardness_scores(tracked_rows, static_rows)
    candidates = [
        row for row in tracked_rows
        if str(row.get("pose_detected")) in ("True", "true", "1")
        and int(row["video_id"]) in labels
        and int(labels[int(row["video_id"])]["slow"]) == 0
    ]
    used_videos: set[int] = set()
    used_ids: set[str] = set()
    picks: list[dict[str, Any]] = []

    def take(row: Row, kind: str) -> bool:
        video = int(row["video_id"])
        youtube = labels[video]["youtube_id"]
        if video in used_videos or youtube in used_ids:
            return False
        used_videos.add(video)
        used_ids.add(youtube)
        picks.append({
            "video_id": video,
            "event_index": int(row["event_index"]),
            "event_name": row["event_name"],
            "frame_index": int(row["frame_index"]),
            "kind": kind,
            "hardness": scores.get(_row_key(row)),
        })
        return True

    ranked = sorted(
        (row for row in candidates if _row_key(row) in scores),
        key=lambda row: (-scores[_row_key(row)], *_row_key(row)),
    )
    hard = 0
    for row in ranked:
        if hard >= n_hard:
            break
        hard += take(row, "hard")

    pool = sorted(candidates, key=_row_key)
    random.Random(seed).shuffle(pool)
    drawn = 0
    for row in pool:
        if drawn >= n_random:
            break
        drawn += take(row, "random")
    return picks


# ------------------------------------------------------------------------- IO
def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_labels(path: Path) -> dict[int, dict[str, Any]]:
    return {int(r["id"]): {"youtube_id": r["youtube_id"], "slow": int(r["slow"])} for r in _read_csv(path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracked", type=Path, default=Path("data/pro_feature_raw_tracked.csv"))
    parser.add_argument("--static", type=Path, default=Path("data/pro_feature_raw.csv"))
    parser.add_argument("--labels", type=Path, default=Path("event_detection/data/golfDB_labels.csv"))
    parser.add_argument("--videos-dir", type=Path, default=Path("learning_video/golfdb_720p_slice"))
    parser.add_argument("--n-hard", type=int, default=24)
    parser.add_argument("--n-random", type=int, default=24)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=Path("outputs/arm_label_tool_pro.html"))
    parser.add_argument("--sample-out", type=Path, default=Path("tests/fixtures/pro_label_sample.json"))
    parser.add_argument("--template", type=Path, default=Path(__file__).with_name("arm_label_template.html"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    picks = select_pro_frames(
        _read_csv(args.tracked), _read_csv(args.static), load_labels(args.labels),
        args.n_hard, args.n_random, args.seed,
    )
    items = [
        (f"pro{p['video_id']}", args.videos_dir / f"{p['video_id']}.mp4", p["event_index"], p["event_name"], p["frame_index"])
        for p in picks
    ]
    tasks = build_tasks_from_items(items, FRAME_MAX_SIDE)
    tasks = [tasks[i] for i in shuffled_order(len(tasks), args.seed)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(args.template.read_text(encoding="utf-8"), tasks), encoding="utf-8")
    args.sample_out.parent.mkdir(parents=True, exist_ok=True)
    args.sample_out.write_text(json.dumps(picks, indent=1), encoding="utf-8")
    print(f"{len(picks)} frames ({sum(p['kind'] == 'hard' for p in picks)} hard) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
