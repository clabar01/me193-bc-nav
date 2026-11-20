"""Train the behavior-cloning navigation policy on the cleaned walks; leave-one-walk-out; export.

    ../me193-step-counter/.venv/bin/python scripts/train_policy.py

Data: every session with a cleaned events file (data/clean/<name>_events.csv,
from scripts/clean_labels.py) and its 10 Hz log in data/raw/: Record-mode
demos and DAgger sessions alike. Who walked comes from data/people.json (or the
session JSON's "person"); each walk's distances are rescaled to that person's
K (board/me193_nav/python/people.json), so they match what the board computes.

Actions (from the cleaned events; "unmatched" presses are left out):
    LEFT / RIGHT   matched presses and second presses of one long rotation,
                   plus auto turns nobody pressed for (status unlabeled_turn:
                   real turns whose press came late)
    STAIRS         STAIRS_START
    ARRIVED        ARRIVED
    STRAIGHT       DAgger "STRAIGHT, ignore that" presses
Labels per 10 Hz row: the next action if it is <= HORIZON_M meters ahead
(distance at the action minus distance now), STRAIGHT otherwise; rows after
ARRIVED are ARRIVED. A STRAIGHT action makes the last HORIZON_M before it
STRAIGHT (the policy was wrong there).
Features: policy.FEATURES (since_turn_m, turns, distance_m, sin/cos heading,
in_stairs from the STAIRS windows), with and without in_stairs.

Models: MLP (32-32, classes balanced by resampling) and kNN (k 7, 1/d
weights), z-scored. Leave-one-walk-out: row accuracy, balanced accuracy
(mean recall over the classes present) and, with the board's 1 s majority
smoothing, how many meters before each action the held-out walk's policy
announces it (or a miss), plus false announcements. The model with the best
mean of balanced accuracy and action hit rate is refitted on all walks and
exported to models/policy.npz (+ the board copy); scores within TIE are a tie,
broken by fewer features, then fewer false announcements. Report:
models/policy_results.md.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "board" / "me193_nav" / "python"))
import policy  # noqa: E402

CLASSES = ("STRAIGHT", "LEFT", "RIGHT", "STAIRS", "ARRIVED")
HORIZON_M = 3.0
EVAL_BEFORE_M = 10.0        # an announcement counts if it starts at most this far before the action
EVAL_AFTER_S = 1.0          # ... or at most this long after it (late)
KNN_K = 7
REG_K = 7
MLP_HIDDEN = (32, 32)
TIE = 0.02                  # scores closer than this are a tie on 6 walks


def load_walks():
    people = json.loads((REPO / "data" / "people.json").read_text())["sessions"]
    ks = json.loads((REPO / "board" / "me193_nav" / "python" / "people.json").read_text())
    walks = []
    for evf in sorted((REPO / "data" / "clean").glob("rec_*_events.csv")):
        name = evf.name.removesuffix("_events.csv")
        log, meta_path = REPO / "data" / "raw" / f"{name}.csv", REPO / "data" / "raw" / f"{name}.json"
        if not log.exists():
            continue                              # discarded since it was cleaned
        meta = json.loads(meta_path.read_text())
        person = meta.get("person") or people.get(name, {}).get("person", "?")
        scale = ks.get(person, meta.get("weinberg_K")) / meta.get("weinberg_K", ks.get(person))
        df = pd.read_csv(log)
        df[["distance_m", "since_turn_m"]] *= scale
        ev = pd.read_csv(evf)
        walks.append(label(name, person, df, ev))
    return walks


def label(name, person, df, ev):
    t, d = df.t_s.to_numpy(), df.distance_m.to_numpy()
    acts = []
    for r in ev.itertuples():
        a = {"TURN_L": "LEFT", "TURN_R": "RIGHT", "STAIRS_START": "STAIRS"}.get(r.event, r.event)
        if r.status == "unmatched" or a not in CLASSES:
            continue
        acts.append((r.t_s, a))
    acts.sort()
    windows = [(a, b) for a, b in zip(ev[ev.event == "STAIRS_START"].t_s, ev[ev.event == "STAIRS_END"].t_s)]
    in_stairs = np.array([any(a <= ti <= b for a, b in windows) for ti in t], float)
    dist_at = lambda te: float(np.interp(te, t, d))  # noqa: E731
    y, to_go = [], []
    arrived = next((te for te, a in acts if a == "ARRIVED"), None)
    for ti, di in zip(t, d):
        nxt = next(((te, a) for te, a in acts if te > ti), None)
        if arrived is not None and ti >= arrived:
            y.append("ARRIVED")
        elif nxt and dist_at(nxt[0]) - di <= HORIZON_M:
            y.append(nxt[1])
        else:
            y.append("STRAIGHT")
        real = next(((te, a) for te, a in acts if te > ti and a != "STRAIGHT"), None)
        to_go.append(max(dist_at(real[0]) - di, 0.0) if real else 0.0)
    X = np.array([policy.features(r._asdict(), s) for r, s in zip(df.itertuples(index=False), in_stairs)])
    return {"name": name, "person": person, "t": t, "d": d, "X": X, "y": np.array(y), "to_go": np.array(to_go),
            "acts": [(te, a, dist_at(te)) for te, a in acts if a != "STRAIGHT"]}


def fit(kind, X, y, seed=0):
    mean, std = X.mean(0), X.std(0)
    std = np.where(std < 1e-9, 1.0, std)
    Z = (X - mean) / std
    if kind == "knn":
        return KNeighborsClassifier(KNN_K, weights="distance").fit(Z, y), mean, std
    rng = np.random.default_rng(seed)
    top = max((y == c).sum() for c in np.unique(y))
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), top, replace=True) for c in np.unique(y)])
    return MLPClassifier(MLP_HIDDEN, alpha=1e-3, max_iter=3000, random_state=seed).fit(Z[idx], y[idx]), mean, std


def smooth(raw):
    """The board's smoother (policy.Policy.step), on a whole walk."""
    out, hist, cur = [], [], "STRAIGHT"
    for r in raw:
        hist = (hist + [r])[-policy.SMOOTH:]
        top = max(set(hist), key=hist.count)
        if hist.count(top) > hist.count(cur):
            cur = top
        out.append(cur)
    return np.array(out)


def announcements(w, sm):
    """Per action: meters before it the smoothed policy starts announcing it (None = missed)."""
    out, prev_t = [], -1e9
    for te, a, de in w["acts"]:
        ok = (w["d"] >= de - EVAL_BEFORE_M) & (w["t"] <= te + EVAL_AFTER_S) & (w["t"] > prev_t) & (sm == a)
        idx = np.flatnonzero(ok)
        if not len(idx):
            out.append((te, a, None, None))
        else:
            run_end = idx[-1]
            start = run_end
            while start - 1 >= 0 and sm[start - 1] == a:
                start -= 1
            out.append((te, a, round(de - w["d"][start], 1), round(w["t"][start] - te, 1)))
        prev_t = te
    covered = np.zeros(len(sm), bool)
    for te, a, de in w["acts"]:
        covered |= (w["d"] >= de - EVAL_BEFORE_M) & (w["t"] <= te + EVAL_AFTER_S) & (sm == a)
    covered |= w["t"] >= max((te for te, a, _ in w["acts"] if a == "ARRIVED"), default=1e9)
    runs = np.flatnonzero((sm != "STRAIGHT") & ~covered)
    false = 0 if not len(runs) else 1 + int((np.diff(runs) > 1).sum())
    return out, false


def lowo(walks, kind, cols, zero_stairs_at_test=False):
    rows, events, falses = [], [], 0
    for i, w in enumerate(walks):
        train = [v for j, v in enumerate(walks) if j != i]
        X = np.concatenate([v["X"][:, cols] for v in train])
        y = np.concatenate([v["y"] for v in train])
        m, mean, std = fit(kind, X, y)
        Xt = w["X"].copy()
        if zero_stairs_at_test:                 # what the board sees: nothing measures in_stairs live
            Xt[:, policy.FEATURES.index("in_stairs")] = 0.0
        raw = m.predict((Xt[:, cols] - mean) / std)
        sm = smooth(raw)
        ann, false = announcements(w, sm)
        falses += false
        rows.append({"walk": w["name"][-6:], "who": w["person"], "acc": (sm == w["y"]).mean(),
                     "bal_acc": balanced_accuracy_score(w["y"], sm), "false": false})
        events += [(w["name"][-6:], *e) for e in ann]
    return pd.DataFrame(rows), events, falses


def export(walks, kind, cols, names):
    X = np.concatenate([w["X"][:, cols] for w in walks])
    y = np.concatenate([w["y"] for w in walks])
    to_go = np.concatenate([w["to_go"] for w in walks])
    m, mean, std = fit(kind, X, y)
    classes = list(m.classes_)
    arrays = {"kind": np.array(kind), "classes": np.array(classes), "features": np.array(names),
              "mean": mean, "std": std, "reg_X": (X - mean) / std, "reg_m": to_go, "reg_k": np.array(REG_K)}
    if kind == "mlp":
        for i, (W, b) in enumerate(zip(m.coefs_, m.intercepts_)):
            arrays[f"w{i}"], arrays[f"b{i}"] = W.T, b
    else:
        arrays.update(knn_X=(X - mean) / std, knn_y=np.array([classes.index(c) for c in y]), knn_k=np.array(KNN_K))
    for out in (REPO / "models" / "policy.npz", REPO / "board" / "me193_nav" / "python" / "models" / "policy.npz"):
        with open(out, "wb") as f:
            np.savez(f, **arrays)
    # check: the board's numpy Policy agrees with sklearn on every training row
    p = policy.Policy(REPO / "models" / "policy.npz")
    agree = np.mean([p.classes[int(np.argmax(p.probs(x)))] == c for x, c in zip(X, m.predict((X - mean) / std))])
    return agree, len(X)


def main():
    walks = load_walks()
    print(f"{len(walks)} walks: " + ", ".join(f"{w['name'][-6:]} ({w['person']}, {len(w['y'])} rows)" for w in walks))
    counts = pd.Series(np.concatenate([w["y"] for w in walks])).value_counts()
    print("labels:", counts.to_dict())
    all_names = list(policy.FEATURES)
    full = list(range(len(all_names)))
    no_stairs = [i for i, n in enumerate(all_names) if n != "in_stairs"]
    # (name, columns, in_stairs zeroed on the held-out walk, can run live on the board)
    variants = [("with in_stairs, true at test (NOT available live)", full, False, False),
                ("with in_stairs, 0 at test (as live)", full, True, True),
                ("without in_stairs", no_stairs, False, True)]
    results, lines = [], []
    for kind in ("mlp", "knn"):
        for vname, cols, zero, live in variants:
            df, events, falses = lowo(walks, kind, cols, zero)
            hits = [e for e in events if e[3] is not None]
            turns = [e for e in events if e[2] in ("LEFT", "RIGHT")]
            turn_hits = [e for e in turns if e[3] is not None]
            res = {"model": kind.upper(), "features": vname, "acc": df.acc.mean(), "bal_acc": df.bal_acc.mean(),
                   "actions hit": f"{len(hits)}/{len(events)}", "turns hit": f"{len(turn_hits)}/{len(turns)}",
                   "median m ahead (turns)": float(np.median([e[3] for e in turn_hits])) if turn_hits else np.nan,
                   "false announcements": falses,
                   "score": (df.bal_acc.mean() + len(hits) / len(events)) / 2 if live else -1}
            results.append((res, df, events, kind, cols))
            print(f"{kind} {vname}: acc {res['acc']:.3f}, balanced {res['bal_acc']:.3f}, actions {res['actions hit']}, "
                  f"turns {res['turns hit']}, false {falses}")
    # Only variants that run live; within TIE of the best score, prefer fewer features (no input that is
    # always 0 on the board), then fewer false announcements, then the MLP.
    top = max(r[0]["score"] for r in results)
    best = min((r for r in results if r[0]["score"] >= top - TIE),
               key=lambda r: (len(r[4]), r[0]["false announcements"], r[3] != "mlp"))
    res, df, events, kind, cols = best
    names = [all_names[i] for i in cols]
    agree, n = export(walks, kind, cols, names)
    print(f"\nexported {kind.upper()} ({res['features']}); board numpy == sklearn on {100 * agree:.1f}% of {n} rows")

    def md(d):
        return "\n".join(["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
                         + ["| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |"
                            for r in d.itertuples(index=False)])
    summary = pd.DataFrame([r[0] for r in results]).drop(columns="score")
    ev_tables = []
    for r in results:
        if r[3] in ("mlp", "knn") and r[0]["features"] == res["features"]:
            e = pd.DataFrame(r[2], columns=["walk", "t_s", "action", "m ahead", "s from action"])
            e["m ahead"] = e["m ahead"].map(lambda v: "MISSED" if v is None or v != v else v)
            ev_tables.append(f"### {r[0]['model']} ({r[0]['features']}), per action\n\n" + md(e))
    perwalk = df.assign(acc=df.acc.round(3), bal_acc=df.bal_acc.round(3))
    (REPO / "models" / "policy_results.md").write_text(
        "# Navigation policy (behavior cloning)\n\nMade by `python scripts/train_policy.py`. "
        f"{len(walks)} walks: " + ", ".join(f"{w['name']} ({w['person']})" for w in walks)
        + f". Labels: {counts.to_dict()}. Leave-one-walk-out; predictions smoothed like the board "
          f"({policy.SMOOTH}-row majority). An action is announced if the smoothed output shows it from at most "
          f"{EVAL_BEFORE_M:.0f} m before to {EVAL_AFTER_S:.0f} s after it; 'm ahead' is where that announcement starts.\n\n"
        + md(summary) + f"\n\n**Exported: {kind.upper()}, {res['features']}** (within {TIE} of the best mean of balanced "
          f"accuracy and actions hit among the variants that can run live - the board does not measure in_stairs - "
          f"ties going to fewer features, then fewer false announcements).\n\n## Exported model, per walk\n\n" + md(perwalk) + "\n\n" + "\n\n".join(ev_tables) + "\n")
    print((REPO / "models" / "policy_results.md").read_text())


if __name__ == "__main__":
    main()
