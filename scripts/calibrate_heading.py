"""Measure the heading drift while walking on the old straight hallway trials.

    ../me193-step-counter/.venv/bin/python scripts/calibrate_heading.py

Replays each trial through the board's own heading code (board/me193_nav/
python/nav.py) with the walking drift correction off, and measures the drift
rate while walking (deg/s, + = right) = net heading change / walking time.
The trials are straight walks, so the net heading should be ~0.

The drift is fitted on sessions A/B (median over trials: a few A trials
include real turns, which a median ignores) and tested on session C.
Paste the printed value into WALK_YAW_DRIFT_DPS in nav.py.
Writes models/heading_results.md.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "board" / "me193_nav" / "python"))
import nav  # noqa: E402


def replay(csv, drift):
    nav.WALK_YAW_DRIFT_DPS = drift
    n = nav.Navigator(REPO / "models" / "step_detector_imu.npz")
    df = pd.read_csv(csv)
    walking = 0
    for r in df.itertuples(index=False):
        n.add({"t": r.t_ms, "ax": r.ax, "ay": r.ay, "az": r.az, "gx": r.gx, "gy": r.gy, "gz": r.gz})
        walking += n.walk_left > 0
    return n.heading, walking / nav.FS, n.turns


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--old-repo", type=Path, default=REPO.parent / "me193-step-counter")
    args = ap.parse_args()
    raw = args.old_repo.resolve() / "data" / "raw"
    rows = []
    for meta_path in sorted(raw.glob("[ABC]_*.json")):
        meta = json.loads(meta_path.read_text())
        if not meta.get("counted_steps"):
            continue                        # standing trials
        net, walk_s, _ = replay(meta_path.with_suffix(".csv"), 0.0)
        rows.append({"trial": meta_path.stem, "session": meta_path.stem[0], "walk_s": round(walk_s, 1),
                     "net_deg (no correction)": round(net, 1), "drift_dps": round(net / walk_s, 2)})
    df = pd.DataFrame(rows)
    ab = df[df.session.isin(["A", "B"])]
    drift = round(float(ab.drift_dps.median()), 2)
    df["net_deg (corrected)"] = [round(replay(raw / f"{t}.csv", drift)[0], 1) for t in df.trial]
    c = df[df.session == "C"]
    print(df.to_string(index=False))
    summary = (f"Walking drift fitted on A/B (median of {len(ab)} trials): **{drift:+.2f} deg/s** (+ = right).\n\n"
               f"Session C (held out), mean |net heading| over {len(c)} straight walks: "
               f"{c['net_deg (no correction)'].abs().mean():.1f} deg without the correction, "
               f"{c['net_deg (corrected)'].abs().mean():.1f} deg with it.")
    print("\n" + summary.replace("**", ""))
    md = "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
                   + ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])
    (REPO / "models" / "heading_results.md").write_text(
        "# Heading drift while walking\n\nMade by `python scripts/calibrate_heading.py`: the old straight "
        "hallway trials replayed through `board/me193_nav/python/nav.py` (complementary-filter gravity, "
        f"tau {nav.UP_TAU_S} s).\n\n{summary}\n\n{md}\n")


if __name__ == "__main__":
    main()
