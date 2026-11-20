"""Clean the button labels of me193_nav sessions: snap presses to the auto-detected turns, flag the rest.

    python scripts/clean_labels.py data/raw/rec_A.csv data/raw/rec_B.csv ...
    python scripts/clean_labels.py --all                 # every session in data/raw/

For each session writes data/clean/<name>_events.csv and <name>_review.png,
and for the whole run data/clean/summary.md (per walk, plus anything that
looks inconsistent between walks).

Rules (nothing is dropped: every press stays in the events file with a status)
  LEFT / RIGHT
    matched     paired with an auto turn of the same direction (TURN_L / TURN_R)
                within +-SNAP_S; closest pairs first, one turn per press. The
                clean time is the turn's time.
    rotation    not paired, but the heading turned >= ROT_DEG that way within
                +-SNAP_S: a second press during one long rotation (e.g. two
                LEFTs for a 180), which the turn detector counts as one turn.
                Kept at its own time (it was pressed during the turn).
    unmatched   neither: no turn near the press. Kept with the raw time;
                exclude it from training labels (status != "unmatched").
  auto turns without a press -> "unlabeled_turn" (with whether it is inside a
    STAIRS window).
  STAIRS: consecutive presses pair into start/end windows (STAIRS_START /
    STAIRS_END); an odd one out is "unpaired", several in one 0.1 s row are
    flagged.
  ARRIVED, and STRAIGHT (DAgger: "the policy is wrong here"): kept as is. A
    DAgger session without an ARRIVED press gets one "inferred" at its last
    counted step - ARRIVED_LAG_S (the demos' median).
  STAIRS rule: presses alternate - first = START (top step), second = END
    (bottom step); only STAIRS_START is an action, never the END.
  data/label_fixes.json: reviewed corrections (event moved, status
    "corrected", raw time kept), e.g. late STAIRS_START presses.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_session  # noqa: E402

SNAP_S = 3.0
ROT_DEG = 60.0
DIST_TOL = 0.15             # walks more than this far from the median distance / steps are flagged
OUT = REPO / "data" / "clean"
ARRIVED_LAG_S = 1.35        # demos: the ARRIVED press is a median 1.35 s before the last counted step
FIXES_PATH = REPO / "data" / "label_fixes.json"
FIXES = json.loads(FIXES_PATH.read_text()) if FIXES_PATH.exists() else {}
FLAG = "#e34948"            # status: problem (always with a text label)
OK = "#52514e"


def heading_change(df, t0, t1):
    h = df.set_index("t_s")["heading_deg"]
    a, b = h[h.index <= t0], h[h.index <= t1]
    return float(b.iloc[-1] - (a.iloc[-1] if len(a) else h.iloc[0]))


def clean(csv):
    df = pd.read_csv(csv)
    events = plot_session.events_of(df)
    presses = [(t, e) for t, e in events if e in ("LEFT", "RIGHT")]
    turns = [(t, e) for t, e in events if e in ("TURN_L", "TURN_R")]
    rows = []

    # LEFT / RIGHT -> auto turns
    pairs = sorted((abs(tt - tp), i, k) for i, (tp, ep) in enumerate(presses) for k, (tt, et) in enumerate(turns)
                   if et == f"TURN_{ep[0]}" and abs(tt - tp) <= SNAP_S)
    press_turn, used = {}, set()
    for _, i, k in pairs:
        if i not in press_turn and k not in used:
            press_turn[i] = k
            used.add(k)
    for i, (tp, ep) in enumerate(presses):
        sign = -1 if ep == "LEFT" else 1
        if i in press_turn:
            tt = turns[press_turn[i]][0]
            rows.append({"t_s": tt, "event": ep, "raw_t_s": tp, "status": "matched",
                         "note": f"{turns[press_turn[i]][1]} {tt - tp:+.1f} s"})
            continue
        dh = heading_change(df, tp - SNAP_S, tp + SNAP_S)
        if sign * dh >= ROT_DEG:
            rows.append({"t_s": tp, "event": ep, "raw_t_s": tp, "status": "rotation",
                         "note": f"heading {dh:+.0f} deg within +-{SNAP_S:.0f} s, part of one long turn"})
            continue
        free = [(tt, et) for k, (tt, et) in enumerate(turns) if k not in used and et == f"TURN_{ep[0]}"]
        near = min(free, key=lambda x: abs(x[0] - tp), default=None)
        hint = f"; nearest unlabeled {near[1]} {near[0] - tp:+.1f} s" if near else ""
        rows.append({"t_s": tp, "event": ep, "raw_t_s": tp, "status": "unmatched",
                     "note": f"no turn: heading {dh:+.0f} deg within +-{SNAP_S:.0f} s{hint}"})

    # STAIRS windows
    stairs = [t for t, e in events if e == "STAIRS"]
    windows = []
    for j in range(0, len(stairs) - 1, 2):
        windows.append((stairs[j], stairs[j + 1]))
        same_row = stairs[j + 1] - stairs[j] < 0.15
        for t, ev in ((stairs[j], "STAIRS_START"), (stairs[j + 1], "STAIRS_END")):
            rows.append({"t_s": t, "event": ev, "raw_t_s": t, "status": "flagged" if same_row else "ok",
                         "note": "start and end in the same 0.1 s row" if same_row else ""})
    if len(stairs) % 2:                  # presses alternate top step / bottom step: an odd last one is a start
        rows.append({"t_s": stairs[-1], "event": "STAIRS_START", "raw_t_s": stairs[-1], "status": "unpaired",
                     "note": "odd number of STAIRS presses: a start without its end"})

    for k, (tt, et) in enumerate(turns):
        if k not in used:
            inside = any(a <= tt <= b for a, b in windows)
            rows.append({"t_s": tt, "event": et, "raw_t_s": tt, "status": "unlabeled_turn",
                         "note": f"heading {heading_change(df, tt - 2, tt + 3):+.0f} deg; "
                                 f"{'inside' if inside else 'outside'} STAIRS window"})
    for t, e in events:
        if e in ("ARRIVED", "STRAIGHT"):
            rows.append({"t_s": t, "event": e, "raw_t_s": t, "status": "ok", "note": ""})
    meta = json.loads(csv.with_suffix(".json").read_text()) if csv.with_suffix(".json").exists() else {}
    if meta.get("mode") == "dagger" and not any(e == "ARRIVED" for _, e in events) and (df.steps.diff() > 0).any():
        last = float(df.t_s[df.steps.diff() > 0].iloc[-1])
        rows.append({"t_s": round(last - ARRIVED_LAG_S, 2), "event": "ARRIVED", "raw_t_s": None, "status": "inferred",
                     "note": f"no ARRIVED press (DAgger): last counted step {last:.1f} s - {ARRIVED_LAG_S} s"})

    # reviewed label fixes (data/label_fixes.json): move an event, keep its raw time
    for fix in FIXES.get("sessions", {}).get(csv.stem, []):
        hit = [r for r in rows if r["event"] == fix["event"] and abs(r["raw_t_s"] - fix["raw_t_s"]) < 0.06]
        if len(hit) != 1:
            raise SystemExit(f"{csv.stem}: label fix {fix} matches {len(hit)} events")
        hit[0].update(t_s=fix["t_s"], status="corrected", note=fix["why"])
        if fix["event"] == "STAIRS_START":   # the STAIRS window moves with it
            windows[:] = [(fix["t_s"] if abs(a - fix["raw_t_s"]) < 0.06 else a, b) for a, b in windows]

    ev = pd.DataFrame(rows, columns=["t_s", "event", "raw_t_s", "status", "note"]).sort_values("t_s")
    ev[["t_s", "raw_t_s"]] = ev[["t_s", "raw_t_s"]].round(2)
    return df, ev, windows, turns


def review_plot(csv, df, ev, windows, out):
    st = ev.status.value_counts()
    title = (f"{csv.stem} review: {st.get('matched', 0) + st.get('rotation', 0)} presses matched, "
             f"{st.get('unmatched', 0)} unmatched, {st.get('unlabeled_turn', 0)} "
             f"turn{'' if st.get('unlabeled_turn', 0) == 1 else 's'} without a press")
    fig, (a1, a2, a3), *_ = plot_session.draw(csv, title)
    for a, b in windows:
        for ax in (a1, a2, a3):
            ax.axvspan(a, b, color="#ecebe7", linewidth=0, zorder=0)
        a1.annotate("stairs", ((a + b) / 2, 1), xycoords=("data", "axes fraction"), xytext=(0, -12),
                    textcoords="offset points", ha="center", fontsize=8, color=OK)
    for r in ev.itertuples():
        if r.status in ("matched", "rotation"):
            a3.plot([r.raw_t_s, r.t_s], [1, 0], color=OK, linewidth=1, zorder=2)
        elif r.status == "unmatched":
            a3.scatter(r.raw_t_s, 1, s=320, facecolors="none", edgecolors=FLAG, linewidths=2, zorder=4)
            a3.annotate("unmatched", (r.raw_t_s, 1), xytext=(0, -22), textcoords="offset points",
                        ha="center", fontsize=8, color=FLAG, fontweight="bold")
        elif r.status == "unlabeled_turn":
            a3.scatter(r.t_s, 0, s=320, facecolors="none", edgecolors=FLAG, linewidths=2, zorder=4)
            a3.annotate("no press", (r.t_s, 0), xytext=(0, 14), textcoords="offset points",
                        ha="center", fontsize=8, color=FLAG, fontweight="bold")
    fig.savefig(out, dpi=130)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true", help="every session in data/raw/")
    args = ap.parse_args()
    csvs = args.csv or (sorted(p for p in (REPO / "data" / "raw").glob("rec_*.csv") if not p.stem.endswith("_imu"))
                        if args.all else [])
    if not csvs:
        ap.error("give session CSVs or --all")
    OUT.mkdir(parents=True, exist_ok=True)

    summary = []
    for csv in csvs:
        df, ev, windows, turns = clean(csv)
        ev.to_csv(OUT / f"{csv.stem}_events.csv", index=False)
        review_plot(csv, df, ev, windows, OUT / f"{csv.stem}_review.png")
        meta = json.loads(csv.with_suffix(".json").read_text()) if csv.with_suffix(".json").exists() else {}
        last, st = df.iloc[-1], ev.status.value_counts()
        n_press = int(ev.event.isin(["LEFT", "RIGHT"]).sum())
        summary.append({
            "walk": csv.stem, "who": meta.get("person") or "?", "s": round(float(last.t_s), 1),
            "presses L/R": n_press, "matched": int(st.get("matched", 0)), "rotation": int(st.get("rotation", 0)),
            "unmatched": int(st.get("unmatched", 0)), "turns w/o press": int(st.get("unlabeled_turn", 0)),
            "turns": int(last.turns), "steps": int(last.steps), "distance_m": round(float(last.distance_m), 2),
            "end heading": int(round(float(last.heading_deg))), "stairs windows": len(windows),
            "stairs s": round(sum(b - a for a, b in windows), 1)})
        print(f"\n{csv.stem}")
        print(ev.to_string(index=False))

    s = pd.DataFrame(summary)
    flags = []
    if s.turns.nunique() > 1:
        flags.append("turn counts differ: " + ", ".join(f"{w[-6:]} {t}" for w, t in zip(s.walk, s.turns)))
    for col in ("distance_m", "steps"):
        med = s[col].median()
        for w, v in zip(s.walk, s[col]):
            if abs(v - med) > DIST_TOL * med:
                flags.append(f"{w}: {col} {v} is {100 * (v - med) / med:+.0f}% from the median {med}")
        flags.append(f"{col} spread: {s[col].min()}-{s[col].max()} "
                     f"({100 * (s[col].max() - s[col].min()) / s[col].min():.0f}% max/min)")
    if s["presses L/R"].nunique() > 1:
        flags.append("number of LEFT/RIGHT presses differs: "
                     + ", ".join(f"{w[-6:]} {n}" for w, n in zip(s.walk, s["presses L/R"])))
    if s["stairs s"].max() > 2 * max(s["stairs s"].min(), 0.1):
        flags.append("STAIRS windows are very different lengths: "
                     + ", ".join(f"{w[-6:]} {d} s" for w, d in zip(s.walk, s["stairs s"])))
    if (s["stairs windows"] != s["stairs windows"].iloc[0]).any():
        flags.append("stairs windows differ: " + ", ".join(f"{w[-6:]} {n}" for w, n in zip(s.walk, s["stairs windows"])))
    print("\n" + s.to_string(index=False) + "\n\n" + "\n".join(flags))
    md = "\n".join(["| " + " | ".join(s.columns) + " |", "|" + "---|" * len(s.columns)]
                   + ["| " + " | ".join(str(v) for v in r) + " |" for r in s.itertuples(index=False)])
    (OUT / "summary.md").write_text("# Label cleaning summary\n\nMade by `python scripts/clean_labels.py "
                                    + " ".join(str(c.resolve().relative_to(REPO)) for c in csvs) + "`.\n\n" + md
                                    + "\n\n## Inconsistencies\n\n" + "\n".join(f"- {f}" for f in flags) + "\n")


if __name__ == "__main__":
    main()
