"""Retrain the step detector on the 6 IMU channels only (no ultrasonic) and export it for the board.

    ../me193-step-counter/.venv/bin/python scripts/train_detector_imu.py

Uses the old repo's training code and data unchanged (../me193-step-counter,
or --old-repo): same loading/despiking, same split (sessions A/B, val_frac 0.2,
split seed 0), same warm start + count-loss fine-tune, same settings as the
IMU-only ablation in analysis/evaluate.py (retrain_detector), seeds 0, 1, 2.
The one difference from that ablation: the ablation kept 8 input channels and
held the ultrasonic at "no echo"; here the model has 6 input channels, so the
board needs no ultrasonic at all.

The exported seed is chosen on the VALIDATION trials (counting-rule MAE, ties
-> lowest seed), never on session C. Everything is scored against the old
committed IMU + ultrasonic model (models/step_detector/real) with the same
counting rule (counting.count_steps), on validation, session C (the held-out
test set) and bench_05 (small room with turns, never trained on).

Writes models/step_detector/imu/ (model.pt, norm.json, results.md/.csv,
train_log.txt) and models/step_detector_imu.npz (numpy, for the board).
"""

import argparse
import json
import math
import random
import sys
from argparse import Namespace
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
SEEDS = (0, 1, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--old-repo", type=Path, default=REPO.parent / "me193-step-counter")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    old = args.old_repo.resolve()
    sys.path.insert(0, str(old / "training" / "step_detector"))
    import data  # noqa: E402
    import train as det  # noqa: E402
    from counting import count_steps, peak_steps, sum_count  # noqa: E402
    from model import StepCNN  # noqa: E402
    from strides import step_times  # noqa: E402

    AXES = data.AXES
    det.CHANNELS = AXES          # train.batches() sizes its input from this
    lines = []

    def log(m=""):
        print(m, flush=True)
        lines.append(m)

    def imu(tr):
        t = copy(tr)
        t.x = np.ascontiguousarray(tr.x[:, :len(AXES)])
        return t

    trials, _ = data.load_trials(old / "data" / "raw")
    train_set, val, test, _ = det.split(trials, 0.2, 0)
    bench = old / "data" / "raw" / "bench_05.csv"
    meta = json.loads(bench.with_suffix(".json").read_text())
    bx, _ = data.preprocess(pd.read_csv(bench), int(meta.get("us_lag_samples", 1)))
    bench_tr = data.Trial("bench_05", "bench", meta, bx, np.arange(len(bx)))
    log(f"data: {old / 'data/raw'}; train {len(train_set)}, val {len(val)}, test C {len(test)}, + bench_05 (turns)")
    log(f"train: {', '.join(t.name for t in train_set)}")
    log(f"val:   {', '.join(t.name for t in val)}")

    def run(model, mean, std, trs, nch):
        rows = []
        model = model.eval().cpu()
        for tr in trs:
            x = ((tr.x[:, :nch] - mean) / std).T[None].astype(np.float32)
            with torch.no_grad():
                y = model(torch.from_numpy(x))[0].numpy()
            rows.append({"trial": tr.name, "speed": tr.meta.get("speed"), "counted": tr.counted,
                         "steps": len(count_steps(y, tr.gy)), "peaks": len(peak_steps(y)),
                         "sum": round(sum_count(y), 2), "gyro": len(step_times(tr.gy))})
        return pd.DataFrame(rows)

    # ---- old committed model: IMU + ultrasonic, 8 channels ----
    ck = torch.load(old / "models" / "step_detector" / "real" / "model.pt", map_location="cpu")
    old_m = StepCNN(in_channels=8)
    old_m.load_state_dict(ck["state_dict"])
    on = json.loads((old / "models" / "step_detector" / "real" / "norm.json").read_text())
    old_mean, old_std = np.array(on["mean"]), np.array(on["std"])
    sets = {"val": val, "C": test, "bench_05": [bench_tr]}
    old_res = {k: run(old_m, old_mean, old_std, v, 8) for k, v in sets.items()}

    # ---- IMU-only, 6 channels, 3 seeds, exactly the ablation's procedure ----
    tr6, va6 = [imu(t) for t in train_set], [imu(t) for t in val]
    device = det.pick_device(args.device)
    per_epoch = -(-len(tr6) // 8)
    runs = {}
    for seed in SEEDS:
        a = Namespace(lr=3e-3, ft_lr=3e-4, batch=8, seed=seed, log_every=10 ** 9,
                      pretrain_epochs=-(-90 // per_epoch), epochs=-(-2400 // per_epoch),
                      patience=-(-600 // per_epoch))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mean, std = det.norm_stats(tr6)
        m = StepCNN(in_channels=len(AXES)).to(device)
        log(f"\nseed {seed}: training on {device}...")
        hist = det.train(m, tr6, va6, mean, std, device, a, lambda s: None)
        m = m.eval().cpu()
        res = {k: run(m, mean, std, v, 6) for k, v in sets.items()}
        mae = {k: float((r["steps"] - r["counted"]).abs().mean()) for k, r in res.items()}
        log(f"seed {seed}: kept epoch {hist['best_epoch']}; MAE val {mae['val']:.2f}, C {mae['C']:.2f}, "
            f"bench_05 {res['bench_05']['steps'][0]} of {bench_tr.counted}")
        runs[seed] = dict(model=m, mean=mean, std=std, res=res, mae=mae, epoch=hist["best_epoch"])

    pick = min(SEEDS, key=lambda s: (runs[s]["mae"]["val"], s))
    log(f"\nexported seed {pick} (lowest validation MAE; session C not used to choose)")
    best = runs[pick]

    # ---- tables ----
    def score(df, col):
        e = (df[col] - df["counted"]).to_numpy(float)
        tot = 100 * (df[col].sum() - df["counted"].sum()) / max(df["counted"].sum(), 1)
        return f"{np.abs(e).mean():.2f}", f"{e.mean():+.2f}", f"{tot:+.1f}%"

    summ = []
    for k in ("val", "C"):
        for label, df, col in (("old CNN, IMU + ultrasonic (committed)", old_res[k], "steps"),
                               (f"new CNN, IMU only (seed {pick}, exported)", best["res"][k], "steps"),
                               ("gyro stride detector", old_res[k], "gyro")):
            mae, bias, tot = score(df, col)
            summ.append({"set": k, "method": label, "trials": len(df), "steps counted": int(df["counted"].sum()),
                         "MAE (steps/trial)": mae, "bias": bias, "total": tot})
        maes = [runs[s]["mae"][k] for s in SEEDS]
        summ.append({"set": k, "method": "new CNN, IMU only, seeds 0/1/2", "trials": len(sets[k]),
                     "steps counted": int(sum(t.counted for t in sets[k])),
                     "MAE (steps/trial)": " / ".join(f"{v:.2f}" for v in maes) + f" (mean {np.mean(maes):.2f})",
                     "bias": "", "total": ""})
    summ = pd.DataFrame(summ)

    per = []
    for k in ("val", "C", "bench_05"):
        o, n = old_res[k], best["res"][k]
        for i in range(len(o)):
            per.append({"set": k, "trial": o["trial"][i], "speed": o["speed"][i], "counted": o["counted"][i],
                        "old CNN": o["steps"][i], "new IMU-only CNN": n["steps"][i],
                        **{f"IMU seed {s}": runs[s]["res"][k]["steps"][i] for s in SEEDS},
                        "gyro": o["gyro"][i]})
    per = pd.DataFrame(per)
    log("\n" + summ.to_string(index=False) + "\n\n" + per.to_string(index=False))

    # ---- save ----
    out = REPO / "models" / "step_detector" / "imu"
    out.mkdir(parents=True, exist_ok=True)
    m, mean, std = best["model"], best["mean"], best["std"]
    torch.save({"state_dict": m.state_dict(), "channels": AXES, "seed": pick,
                "config": {"width": m.head.in_channels}}, out / "model.pt")
    (out / "norm.json").write_text(json.dumps({
        "channels": list(AXES), "mean": np.round(mean, 6).tolist(), "std": np.round(std, 6).tolist(),
        "seed": pick, "from": f"{len(tr6)} training trials ({old.name}/data/raw, sessions A/B)"}, indent=1) + "\n")
    per.to_csv(out / "results.csv", index=False)

    def md(df):
        return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
                         + ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])
    (out / "results.md").write_text(
        "# Step detector, IMU only (6 channels)\n\n"
        "Made by `python scripts/train_detector_imu.py`. Training code and data: `../me193-step-counter`, "
        "same split and settings as its IMU-only ablation, seeds 0/1/2; the exported seed was chosen on validation. "
        "All counts use the board's counting rule (`counting.count_steps`). Session C is the held-out test set; "
        "bench_05 is a small room with turns (never trained on).\n\n" + md(summ) + "\n\n## Per trial\n\n" + md(per) + "\n")
    (out / "train_log.txt").write_text("\n".join(lines) + "\n")

    # board export: same layout as the old scripts/export_models.py, but 6 channels and no ultrasonic
    sd = {k: v.numpy() for k, v in m.state_dict().items()}
    convs = sorted({k.split(".")[1] for k in sd if k.startswith("body.") and k.endswith(".weight")}, key=int)
    arrays = {"channels": np.array(AXES), "mean": np.asarray(mean, np.float32), "std": np.asarray(std, np.float32),
              "dilations": np.array([1, 2, 4, 8]), "head_w": sd["head.weight"][:, :, 0], "head_b": sd["head.bias"]}
    for i, k in enumerate(convs):
        arrays[f"conv{i}_w"], arrays[f"conv{i}_b"] = sd[f"body.{k}.weight"], sd[f"body.{k}.bias"]
    npz = REPO / "models" / "step_detector_imu.npz"
    with open(npz, "wb") as f:
        np.savez(f, **arrays)
    log(f"\nsaved {out.relative_to(REPO)}/ and {npz.relative_to(REPO)} ({npz.stat().st_size // 1024} KB)")
    (out / "train_log.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
