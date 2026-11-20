"""Replay the old recordings through the board's navigation code and check it.

    ../me193-step-counter/.venv/bin/python scripts/replay_nav.py

Feeds every old trial (sessions A/B/C with counted steps, plus bench_05, the
small room with turns) sample by sample through board/me193_nav/python/nav.py,
calling process() every 0.25 s like the board, then 4 s of standing still so
the last steps are decided. Compares:
  - live CNN count vs the offline pipeline (old repo's counting.count_steps on
    the same IMU-only model): checks the numpy port and the live windowing;
  - live gyro count vs the offline gyro stride detector (strides.step_times);
  - counted steps (nav.STEP_SOURCE) vs your hand count, distance vs the tape-measured distance,
    turns detected, net heading, time with the gyro detector active.
Writes models/nav_replay.md.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "board" / "me193_nav" / "python"))
import nav  # noqa: E402

MODEL = REPO / "models" / "step_detector_imu.npz"


def replay(df):
    n = nav.Navigator(MODEL)
    n.start_session(0)
    turning = 0
    rows = list(df.itertuples(index=False))
    last = rows[-1]
    rows += [last._replace(t_ms=last.t_ms + 20 * (k + 1), gx=0.0, gy=0.0, gz=0.0) for k in range(200)]
    for k, r in enumerate(rows):
        n.add({"t": r.t_ms, "ax": r.ax, "ay": r.ay, "az": r.az, "gx": r.gx, "gy": r.gy, "gz": r.gz})
        turning += n.turning
        if k % 25 in (12, 24):              # every 12-13 samples = 0.25 s
            n.process()
    n.process()
    return n, turning / nav.FS


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--old-repo", type=Path, default=REPO.parent / "me193-step-counter")
    args = ap.parse_args()
    old = args.old_repo.resolve()
    sys.path.insert(0, str(old / "training" / "step_detector"))
    import data  # noqa: E402
    from counting import count_steps  # noqa: E402
    from model import StepCNN  # noqa: E402
    from strides import step_times  # noqa: E402

    ck = torch.load(REPO / "models" / "step_detector" / "imu" / "model.pt", map_location="cpu")
    m = StepCNN(in_channels=6)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    norm = json.loads((REPO / "models" / "step_detector" / "imu" / "norm.json").read_text())
    mean, std = np.array(norm["mean"]), np.array(norm["std"])

    rows = []
    for meta_path in sorted((old / "data" / "raw").glob("*.json")):
        name = meta_path.stem
        meta = json.loads(meta_path.read_text())
        if meta.get("counted_steps") is None or (name.startswith("bench") and name != "bench_05"):
            continue
        df = pd.read_csv(meta_path.with_suffix(".csv"))
        x, _ = data.preprocess(df, int(meta.get("us_lag_samples", 1)))
        with torch.no_grad():
            y = m(torch.from_numpy(((x[:, :6] - mean) / std).T[None].astype(np.float32)))[0].numpy()
        gy = x[:, data.CHANNELS.index("gy")]
        n, turning_s = replay(df)
        dist = meta.get("distance_m")
        rows.append({
            "trial": name, "speed": meta.get("speed"), "counted": meta["counted_steps"],
            "CNN live": n.cnn_steps, "CNN offline": len(count_steps(y, gy)),
            "gyro live": n.gyro_steps, "gyro offline": len(step_times(gy)),
            "steps": n.steps, "steps err": n.steps - meta["counted_steps"],
            "distance_m": dist if dist else "", "est_m": round(n.distance_m, 2),
            "dist err %": round(100 * (n.distance_m - dist) / dist, 1) if dist else "",
            "turns": n.turns, "net heading": round(n.heading - n.heading0),
            "gyro active s": round(turning_s, 1), "length s": round(len(df) / nav.FS, 1)})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))

    walks = out[(out.counted > 0) & ~out.trial.str.startswith("bench")]
    dists = out[out["dist err %"] != ""]
    lines = [
        f"CNN live = offline on {int((out['CNN live'] == out['CNN offline']).sum())} of {len(out)} trials; "
        f"gyro live = offline on {int((out['gyro live'] == out['gyro offline']).sum())} of {len(out)}.",
        f"Step source {nav.STEP_SOURCE!r}, K {nav.WEINBERG_K}. Straight walks (A/B/C, {len(walks)} trials): "
        f"counted steps MAE {walks['steps err'].abs().mean():.2f} steps (CNN live {(walks['CNN live'] - walks.counted).abs().mean():.2f}, "
        f"gyro live {(walks['gyro live'] - walks.counted).abs().mean():.2f}); turns detected in "
        f"{int((walks.turns > 0).sum())} of them.",
        f"Distance ({len(dists)} trials with a tape-measured distance): mean |error| "
        f"{dists['dist err %'].astype(float).abs().mean():.1f}%, bias {dists['dist err %'].astype(float).mean():+.1f}%.",
    ]
    for _, r in out[out.trial == "bench_05"].iterrows():
        lines.append(f"bench_05 (room with turns, {r.counted} steps): counted {r.steps}, CNN {r['CNN live']}, "
                     f"gyro {r['gyro live']}; {r.turns} turns detected; gyro active {r['gyro active s']} of "
                     f"{r['length s']} s.")
    print("\n" + "\n".join(lines))
    md = "\n".join(["| " + " | ".join(out.columns) + " |", "|" + "---|" * len(out.columns)]
                   + ["| " + " | ".join(str(v) for v in r) + " |" for r in out.itertuples(index=False)])
    (REPO / "models" / "nav_replay.md").write_text(
        "# Old recordings replayed through board/me193_nav\n\nMade by `python scripts/replay_nav.py`.\n\n"
        + "\n".join(f"- {line}" for line in lines) + "\n\n" + md + "\n")


if __name__ == "__main__":
    main()
