"""Calibrate the Weinberg step length, K * (|a| max - min within the step)^0.25, on the old hallway trials.

    ../me193-step-counter/.venv/bin/python scripts/calibrate_weinberg.py               # gyro steps (the board's)
    ../me193-step-counter/.venv/bin/python scripts/calibrate_weinberg.py --steps cnn   # the IMU-only CNN's

--steps gyro (default, what the board counts since STEP_SOURCE = "gyro"): each
trial is replayed through the board's own code (board/me193_nav/python/nav.py)
with K = 1, so every step and accel-range window is exactly the board's: the
live gyro stride detector's steps over the whole trial, each step's range taken
from the previous step to this one.

--steps cnn: steps from the IMU-only CNN (models/step_detector/imu, counted
with the board's rule counting.count_steps) over the whole trial. Step k's
accel range is taken from its sample to step k+1's (the last step reuses the
previous step's length of time), as in the old features.py. (K = 0.7419.)

K minimizes the summed squared relative distance error over trials (closed
form, the old step_length/train.py fit_scale). Reported:
  - leave-one-trial-out over the B distance trials,
  - fitted on B, tested on C (held out),
and the board's K is then fitted on all 14 hallway distance trials (B + C).
Writes models/weinberg.json (the K for each step source) and
models/weinberg_<source>_results.md.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
EXP = 0.25


def acc_ranges(x, steps):
    acc = np.linalg.norm(x[:, :3], axis=1)
    out = []
    for k, s in enumerate(steps):
        nxt = steps[k + 1] if k + 1 < len(steps) else s + (s - steps[k - 1] if k else 25)
        seg = acc[s:max(min(nxt, len(x)), s + 1)]
        out.append(seg.max() - seg.min())
    return np.array(out)


def board_replay(csv):
    """(steps, sum of per-step Weinberg terms) from the board's code, with K = 1 and gyro steps."""
    sys.path.insert(0, str(REPO / "board" / "me193_nav" / "python"))
    import nav
    nav.WEINBERG_K, nav.STEP_SOURCE = 1.0, "gyro"
    n = nav.Navigator(REPO / "models" / "step_detector_imu.npz")
    n.start_session(0)
    rows = list(pd.read_csv(csv).itertuples(index=False))
    last = rows[-1]
    rows += [last._replace(t_ms=last.t_ms + 20 * (k + 1), gx=0.0, gy=0.0, gz=0.0) for k in range(200)]
    for k, r in enumerate(rows):         # 4 s standing at the end, so the last steps are decided
        n.add({"t": r.t_ms, "ax": r.ax, "ay": r.ay, "az": r.az, "gx": r.gx, "gy": r.gy, "gz": r.gz})
        if k % 25 in (12, 24):
            n.process()
    n.process()
    return n.steps, n.distance_m


def fit_k(terms, dists):
    r = np.array([t.sum() for t in terms]) / np.asarray(dists)
    return float(r.sum() / (r ** 2).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--old-repo", type=Path, default=REPO.parent / "me193-step-counter")
    ap.add_argument("--steps", choices=("gyro", "cnn"), default="gyro")
    args = ap.parse_args()
    old = args.old_repo.resolve()
    sys.path.insert(0, str(old / "training" / "step_detector"))
    import data  # noqa: E402
    from counting import count_steps  # noqa: E402
    from model import StepCNN  # noqa: E402

    ck = torch.load(REPO / "models" / "step_detector" / "imu" / "model.pt", map_location="cpu")
    m = StepCNN(in_channels=6)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    nrm = json.loads((REPO / "models" / "step_detector" / "imu" / "norm.json").read_text())
    mean, std = np.array(nrm["mean"]), np.array(nrm["std"])

    trials, _ = data.load_trials(old / "data" / "raw")
    items = []
    for tr in trials:
        if tr.meta.get("type") != "distance" or not tr.meta.get("distance_m"):
            continue
        if args.steps == "gyro":
            n_steps, term_sum = board_replay(old / "data" / "raw" / f"{tr.name}.csv")
            term = np.full(n_steps, term_sum / max(n_steps, 1))   # only the sum and count are used
        else:
            z = ((tr.x[:, :6] - mean) / std).T[None].astype(np.float32)
            with torch.no_grad():
                y = m(torch.from_numpy(z))[0].numpy()
            steps = count_steps(y, tr.gy)
            n_steps, term = len(steps), acc_ranges(tr.x, steps) ** EXP
        items.append({"trial": tr.name, "session": tr.session, "speed": tr.meta.get("speed"),
                      "dist": float(tr.meta["distance_m"]), "counted": tr.counted, "steps": n_steps,
                      "term": term})
    B = [i for i in items if i["session"] == "B"]
    C = [i for i in items if i["session"] == "C"]

    rows = []
    for it in B:
        k = fit_k([o["term"] for o in B if o is not it], [o["dist"] for o in B if o is not it])
        rows.append((it, "B, leave-one-out", k))
    k_b = fit_k([o["term"] for o in B], [o["dist"] for o in B])
    rows += [(it, "C, K fitted on B", k_b) for it in C]
    k_all = fit_k([o["term"] for o in items], [o["dist"] for o in items])

    df = pd.DataFrame([{"eval": ev, "trial": it["trial"], "speed": it["speed"], "distance_m": it["dist"],
                        "counted": it["counted"], f"{args.steps} steps": it["steps"], "K": round(k, 4),
                        "predicted_m": round(k * it["term"].sum(), 2),
                        "err%": round(100 * (k * it["term"].sum() - it["dist"]) / it["dist"], 1),
                        "err% if count were exact": round(100 * (k * it["term"].mean() * it["counted"] - it["dist"])
                                                          / it["dist"], 1)}
                       for it, ev, k in rows])
    summ = df.groupby("eval", sort=False).agg(trials=("trial", "size"),
                                              mean_abs_err=("err%", lambda e: round(np.abs(e).mean(), 1)),
                                              bias=("err%", lambda e: round(e.mean(), 1))).reset_index()
    print(df.to_string(index=False))
    print("\n" + summ.to_string(index=False))
    print(f"\nK fitted on B: {k_b:.4f}; on all {len(items)} hallway trials (board): {k_all:.4f}")

    out = REPO / "models" / "weinberg.json"
    doc = json.loads(out.read_text()) if out.exists() else {}
    doc = doc if "by_steps" in doc else {"by_steps": {"cnn": {k: doc[k] for k in doc}}} if doc else {"by_steps": {}}
    doc["by_steps"][args.steps] = {
        "K": round(k_all, 4), "K_fitted_on_B_only": round(k_b, 4), "exponent": EXP,
        "steps_from": ("board/me193_nav nav.py replay, gyro stride detector, window = previous step to this step"
                       if args.steps == "gyro" else
                       "models/step_detector_imu.npz with counting.count_steps, window = this step to next step"),
        "fitted_on": [i["trial"] for i in items]}
    doc["formula"] = "step_m = K * (max|a| - min|a| over one step)^0.25, |a| in g, 50 Hz"
    doc["board_uses"] = "gyro"
    out.write_text(json.dumps(doc, indent=1) + "\n")

    def md(d):
        return "\n".join(["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
                         + ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)])
    (REPO / "models" / f"weinberg_{args.steps}_results.md").write_text(
        f"# Weinberg step length on the {args.steps} detector's steps\n\n"
        f"Made by `python scripts/calibrate_weinberg.py --steps {args.steps}`. "
        f"Board K = {k_all:.4f} (all {len(items)} hallway distance trials); K on B only = {k_b:.4f}.\n\n"
        + md(summ) + "\n\n" + md(df) + "\n")


if __name__ == "__main__":
    main()
