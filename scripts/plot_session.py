"""Plot one me193_nav session: heading, step counts (fused / CNN / gyro) and events on one time axis.

    python scripts/plot_session.py                       # newest session in data/raw/
    python scripts/plot_session.py path/to/rec_<...>.csv [--out fig.png]

Three panels share the time axis (s since Start):
  1. heading (deg, + = right), shaded where the detector column is gyro (all of a session since
     STEP_SOURCE = "gyro"; only the turns in older "fused" sessions)
  2. running step counts: counted (steps: what distance uses), CNN, gyro stride detector
  3. events: your button presses (LEFT / RIGHT / STAIRS / ARRIVED) and the
     auto-detected turns (TURN_L / TURN_R); dashed lines carry button presses
     through the panels above.
Prints the totals at the end of the session (the last row).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
INK, MUTED, GRID, SHADE = "#0b0b0b", "#52514e", "#e4e3df", "#ecebe7"
SERIES = {"steps": "#2a78d6", "cnn_steps": "#eb6834", "gyro_steps": "#1baf7a"}   # categorical slots 1-3
LABELS = {"steps": "counted (steps)", "cnn_steps": "CNN", "gyro_steps": "gyro stride detector"}
BUTTON_MARK = {"LEFT": "<", "RIGHT": ">", "STAIRS": "s", "ARRIVED": "*"}
TURN_MARK = {"TURN_L": "<", "TURN_R": ">"}


def style(ax, ylabel):
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def spans(t, mask):
    """(start, end) times of the runs where mask is True."""
    out, start = [], None
    for ti, m in zip(t, mask):
        if m and start is None:
            start = ti
        elif not m and start is not None:
            out.append((start, ti))
            start = None
    if start is not None:
        out.append((start, t.iloc[-1]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="?", type=Path, help="session CSV (default: newest rec_*.csv in data/raw/)")
    ap.add_argument("--out", type=Path, help="PNG to write (default: next to the CSV)")
    args = ap.parse_args()
    csv = args.csv or max((p for p in (REPO / "data" / "raw").glob("rec_*.csv") if not p.stem.endswith("_imu")),
                          key=lambda p: p.stem)
    df = pd.read_csv(csv)
    df["event"] = df["event"].fillna("")
    t = df["t_s"]
    events = [(ti, e) for ti, ev in zip(t, df["event"]) for e in ev.split(";") if e]
    buttons = [(ti, e) for ti, e in events if e in BUTTON_MARK]
    turns = [(ti, e) for ti, e in events if e in TURN_MARK]

    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(12, 8.2), sharex=True, layout="constrained",
                                     gridspec_kw={"height_ratios": [3, 3, 1.3]})
    gyro_on = spans(t, df["detector"] == "gyro")
    for ax in (a1, a2):
        for s, e in gyro_on:
            ax.axvspan(s, e, color=SHADE, linewidth=0, zorder=0)
        for ti, _ in buttons:
            ax.axvline(ti, color=MUTED, linewidth=0.9, linestyle=(0, (3, 3)), zorder=1)

    a1.plot(t, df["heading_deg"], color=SERIES["steps"], linewidth=2)
    style(a1, "heading (deg, + = right)")
    a1.set_title(f"{csv.stem}: heading, step counts and events   (shaded = detector column 'gyro')",
                 loc="left", color=INK, fontsize=11)

    end = t.iloc[-1]
    for col in ("steps", "cnn_steps", "gyro_steps"):
        a2.step(t, df[col], where="post", color=SERIES[col], linewidth=2, label=LABELS[col])
        a2.annotate(f"{LABELS[col]} {df[col].iloc[-1]}", (end, df[col].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    style(a2, "steps (running count)")
    a2.legend(frameon=False, labelcolor=MUTED, loc="upper left", fontsize=9)

    rows = {"buttons": 1, "auto turns": 0}
    for ti, e in buttons:
        a3.scatter(ti, rows["buttons"], marker=BUTTON_MARK[e], s=90, color=INK, zorder=3)
        a3.annotate(e, (ti, rows["buttons"]), xytext=(0, 9), textcoords="offset points", ha="center",
                    fontsize=8, color=INK)
    for ti, e in turns:
        a3.scatter(ti, rows["auto turns"], marker=TURN_MARK[e], s=90, color=MUTED, zorder=3)
        a3.annotate(e[-1], (ti, rows["auto turns"]), xytext=(0, -15), textcoords="offset points", ha="center",
                    fontsize=8, color=MUTED)
    a3.set_yticks([0, 1], ["auto turns", "your buttons"])
    a3.set_ylim(-0.8, 1.8)
    style(a3, "")
    a3.grid(axis="y", visible=False)
    a3.set_xlabel("time since Start (s)", color=MUTED)
    a3.set_xlim(t.iloc[0], end * 1.09)       # room for the end labels

    out = args.out or csv.with_suffix(".png")
    fig.savefig(out, dpi=130)
    last = df.iloc[-1]
    print(f"{csv.name}: {end:.1f} s, {len(df)} rows")
    print(f"total steps {int(last['steps'])} (CNN {int(last['cnn_steps'])}, gyro {int(last['gyro_steps'])}), "
          f"distance {last['distance_m']:.2f} m, turns {int(last['turns'])}")
    print(f"buttons: {', '.join(f'{e} @ {ti:.1f}s' for ti, e in buttons) or 'none'}")
    print(f"auto turns: {', '.join(f'{e} @ {ti:.1f}s' for ti, e in turns) or 'none'}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
