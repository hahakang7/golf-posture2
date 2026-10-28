"""Evaluate candidate QC rules against hand-labelled arm joints.

Ground truth is tests/fixtures/human_arm_labels.json (shoulder/elbow/wrist clicked by hand on 48 event frames).
A model arm is 'bad' when its worst joint is more than BAD_FRACTION of the human arm length away from the
human position. Each rule yields a score per arm; a rule flags an arm when the score is beyond a threshold.
Thresholds are chosen leave-one-video-out so a rule is never scored on data its threshold was fitted on.

Run: pixi run python -m pipeline.qc_evaluate
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipeline.pose_tracking import angle_at, arm_by_feature, extract_tracked_frames
from pipeline.qc_review_sheet import event_frames_from_result
from pipeline.qc_temporal import arm_jump, bone_asymmetry, bone_deviation
from pipeline.run_analysis import EVENT_NAMES, video_frame_count
from pipeline.user_tracking import tracking_window

BAD_FRACTION = 0.35
SENSITIVITY_FRACTIONS = (0.25, 0.35, 0.5)
WINDOW_RADIUS = 8
BONE_RADIUS = 6
FALSE_ALARM_COST = 2.0
# Frames whose left/right labels may be swapped (flagged to the labeller, not yet re-checked): excluded.
EXCLUDED_FRAMES = frozenset({"shorts_3", "shorts2_7", "longform_2"})
RULES = {
    "angle": ("low", [20, 30, 40, 50, 60]),
    "jump": ("high", [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3]),
    "bone": ("high", [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]),
    "asym": ("high", [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]),
}
Point = Sequence[float] | None


# ----------------------------------------------------------------- pure helpers
def arm_error_fraction(human: Sequence[Point], model: Sequence[Point]) -> float | None:
    """Worst joint distance between model and human, as a fraction of the human arm length."""
    if any(p is None for p in human) or any(p is None for p in model):
        return None
    length = math.dist(human[0][:2], human[1][:2]) + math.dist(human[1][:2], human[2][:2])
    if length <= 0:
        return None
    return max(math.dist(h[:2], m[:2]) for h, m in zip(human, model)) / length


def confusion(flags: Sequence[bool], bad: Sequence[bool]) -> dict[str, int]:
    pairs = list(zip(flags, bad))
    return {
        "tp": sum(f and b for f, b in pairs),
        "fp": sum(f and not b for f, b in pairs),
        "fn": sum(b and not f for f, b in pairs),
        "tn": sum(not f and not b for f, b in pairs),
    }


def _flags(scores: Sequence[float | None], threshold: float, direction: str) -> list[bool]:
    if direction == "low":
        return [s is not None and s < threshold for s in scores]
    return [s is not None and s > threshold for s in scores]


def choose_threshold(
    scores: Sequence[float | None],
    bad: Sequence[bool],
    candidates: Sequence[float],
    direction: str,
    false_alarm_cost: float = FALSE_ALARM_COST,
) -> float:
    """Candidate maximising catches minus weighted false alarms; ties go to the more inclusive threshold."""
    best, best_value = None, None
    for threshold in candidates:
        counts = confusion(_flags(scores, threshold, direction), bad)
        value = counts["tp"] - false_alarm_cost * counts["fp"]
        better = best_value is None or value > best_value
        tied = best_value is not None and value == best_value
        more_inclusive = best is not None and ((threshold > best) if direction == "low" else (threshold < best))
        if better or (tied and more_inclusive):
            best, best_value = threshold, value
    return best


def leave_one_group_out(
    rows: Sequence[Mapping[str, Any]],
    score_key: str,
    candidates: Sequence[float],
    direction: str,
    false_alarm_cost: float = FALSE_ALARM_COST,
) -> dict[str, int]:
    """Pooled confusion where each group is judged with a threshold chosen on all the other groups."""
    flags: list[bool] = []
    bad: list[bool] = []
    for group in sorted({row["group"] for row in rows}):
        train = [row for row in rows if row["group"] != group]
        test = [row for row in rows if row["group"] == group]
        threshold = choose_threshold(
            [row[score_key] for row in train], [row["bad"] for row in train], candidates, direction, false_alarm_cost
        )
        flags += _flags([row[score_key] for row in test], threshold, direction)
        bad += [row["bad"] for row in test]
    return confusion(flags, bad)


def arm_rule_scores(
    window: Sequence[Mapping[str, Sequence[float]] | None],
    radius: int,
    arm: Sequence[str],
    other_arm: Sequence[str],
) -> dict[str, float | None]:
    """Rule scores for one arm at the centre frame of a window (index `radius`)."""
    centre = window[radius]
    present = centre is not None and all(name in centre for name in arm)
    deviations = [bone_deviation(window, radius, a, b, BONE_RADIUS) for a, b in zip(arm, arm[1:])]
    deviations = [d for d in deviations if d is not None]
    return {
        "angle": angle_at([centre[name][:2] for name in arm]) if present else None,
        "jump": arm_jump(window, radius, arm),
        "bone": max(deviations) if deviations else None,
        "asym": bone_asymmetry(centre, arm, other_arm),
    }


# ----------------------------------------------------------------- data collection
def collect_windows(video_path: Path, event_frames: Sequence[int], radius: int) -> dict[int, list[dict | None]]:
    """Tracked joints for frames f-radius..f+radius around every event frame f (None where unavailable)."""
    start, last = tracking_window(event_frames, video_frame_count(video_path))
    frames = extract_tracked_frames(video_path, last, start_frame=start)

    def at(frame: int) -> dict | None:
        i = frame - start
        joints = frames[i] if 0 <= i < len(frames) else None
        return {name: list(joint) for name, joint in joints.items()} if joints else None

    return {f: [at(f + d) for d in range(-radius, radius + 1)] for f in event_frames}


def load_windows(cache: Path, videos: Sequence[str], videos_dir: Path, results_dir: Path) -> dict[str, Any]:
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    data = {}
    for name in videos:
        frames = event_frames_from_result(results_dir / name / f"{name}_result.json")
        windows = collect_windows(videos_dir / f"{name}.mp4", frames, WINDOW_RADIUS)
        data[name] = {"event_frames": frames, "windows": {str(f): w for f, w in windows.items()}}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data), encoding="utf-8")
    return data


def build_rows(labels: Path, windows: Mapping[str, Any], flagged: Mapping[tuple, bool]) -> list[dict[str, Any]]:
    rows = []
    arms = arm_by_feature("right")
    for task in json.loads(labels.read_text(encoding="utf-8"))["tasks"]:
        video, event = task["video"], task["event_index"]
        frame = windows[video]["event_frames"][event]
        window = windows[video]["windows"][str(frame)]
        for arm, key, other_key in (
            ("trail", "trail_elbow_angle", "lead_elbow_angle"),
            ("lead", "lead_elbow_angle", "trail_elbow_angle"),
        ):
            names, other = arms[key], arms[other_key]
            human = [task["joints"][name] for name in names]
            centre = window[WINDOW_RADIUS]
            model = [centre[name][:2] if centre and name in centre else None for name in names]
            rows.append({
                "id": task["id"], "group": video, "event": EVENT_NAMES[event], "arm": arm,
                "error": arm_error_fraction(human, model),
                "existing_flag": flagged.get((video, event, arm), False),
                **arm_rule_scores(window, WINDOW_RADIUS, names, other),
            })
    return rows


def load_existing_flags(verdicts: Path) -> dict[tuple, bool]:
    with verdicts.open(encoding="utf-8-sig") as handle:
        return {(r["video"], int(r["event_index"]), r["arm"]): r["qc_flagged"] == "True" for r in csv.DictReader(handle)}


# ------------------------------------------------------------------------ report
def _rate(count: int, total: int) -> str:
    return f"{count}/{total} ({100 * count / total:.0f}%)" if total else "0/0"


def _usable(rows: Sequence[dict[str, Any]], fraction: float) -> list[dict[str, Any]]:
    usable = [dict(r) for r in rows if r["error"] is not None and r["id"] not in EXCLUDED_FRAMES]
    for row in usable:
        row["bad"] = row["error"] > fraction
    return usable


def report(rows: Sequence[dict[str, Any]], fraction: float) -> str:
    usable = _usable(rows, fraction)
    bad_total = sum(r["bad"] for r in usable)
    lines = [f"arms evaluated: {len(usable)}  bad (any joint > {fraction:.0%} of arm length off): {bad_total}"]
    existing = confusion([r["existing_flag"] for r in usable], [r["bad"] for r in usable])
    lines.append(f"existing QC          catch {_rate(existing['tp'], bad_total)}  false alarms {existing['fp']}")
    for name, (direction, candidates) in RULES.items():
        unmeasurable = sum(r[name] is None for r in usable)
        pooled = leave_one_group_out(usable, name, candidates, direction)
        lines.append(
            f"{name:6s} (LOVO)        catch {_rate(pooled['tp'], bad_total)}  false alarms {pooled['fp']}"
            f"  (unmeasurable arms: {unmeasurable})"
        )
    return "\n".join(lines)


def grid(rows: Sequence[dict[str, Any]], fraction: float) -> str:
    usable = _usable(rows, fraction)
    bad = [r["bad"] for r in usable]
    lines = ["threshold grid on all data (descriptive only, thresholds not validated):"]
    for name, (direction, candidates) in RULES.items():
        for threshold in candidates:
            c = confusion(_flags([r[name] for r in usable], threshold, direction), bad)
            sign = "<" if direction == "low" else ">"
            lines.append(f"  {name:6s} {sign} {threshold:<5}  catch {c['tp']:2d}/{sum(bad)}  false alarms {c['fp']:2d}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("tests/fixtures/human_arm_labels.json"))
    parser.add_argument("--verdicts", type=Path, default=Path("tests/fixtures/qc_arm_verdicts.csv"))
    parser.add_argument("--videos-dir", type=Path, default=Path("raw_videos"))
    parser.add_argument("--results-dir", type=Path, default=Path("documents/baseline_eval/2026-10-13"))
    parser.add_argument("--cache", type=Path, default=Path("outputs/qc_eval_windows.json"))
    parser.add_argument("--grid", action="store_true", help="also print the full threshold grid")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = sorted({t["video"] for t in json.loads(args.labels.read_text(encoding="utf-8"))["tasks"]})
    windows = load_windows(args.cache, videos, args.videos_dir, args.results_dir)
    rows = build_rows(args.labels, windows, load_existing_flags(args.verdicts))
    for fraction in SENSITIVITY_FRACTIONS:
        print(report(rows, fraction), "\n")
    if args.grid:
        print(grid(rows, BAD_FRACTION))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
