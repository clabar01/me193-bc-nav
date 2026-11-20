"""Checks for the step fusion at a detector switch, heading, turns and the session log.

    python board/me193_nav/test_nav.py         (numpy only; run from the repo root)

The fusion tests fake both detectors' output and the turning flag, so they
test nav.Navigator.process() alone.
"""

import csv
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "python"))
import nav  # noqa: E402
import session  # noqa: E402

MODEL = HERE / "python" / "models" / "step_detector_imu.npz"
SWITCH = 1000               # the turning flag switches from CNN to gyro at this sample


def fake_nav(cnn, gyro, switch_back=None, source="fused"):
    """A Navigator whose detectors report exactly these step indices and whose
    turning flag is on from SWITCH (to switch_back). The fusion tests use the
    "fused" step source; the board runs "gyro"."""
    nav.STEP_SOURCE = source
    n = nav.Navigator(MODEL)
    n.start_session(0)
    end = 3000
    n.mode.clear()
    n.mode.extend(SWITCH <= i < (switch_back or end) for i in range(end))
    n.mode_base, n.mode_upto, n.n = end - len(n.mode), end - 1, end
    n.switches.extend([SWITCH] + ([switch_back] if switch_back else []))
    n.acc.extend([1.0, 1.5] * (end // 2))
    n.acc_base = end - len(n.acc)
    n.cnn.process = lambda: [(j, 0) for j in cnn]
    n.gyro.process = lambda: [(j, 0) for j in gyro]
    n.process()
    return n


def test_same_step_both_sides_counts_once():
    # CNN puts the step just before the switch, gyro just after: one step
    n = fake_nav(cnn=[SWITCH - 3], gyro=[SWITCH + 2])
    assert n.steps == 1, n.steps
    assert (n.cnn_steps, n.gyro_steps) == (1, 1)


def test_step_on_wrong_sides_not_dropped():
    # each detector puts the step on the OTHER detector's side: still one step, not zero
    n = fake_nav(cnn=[SWITCH + 3], gyro=[SWITCH - 3])
    assert n.steps == 1, n.steps


def test_only_active_detector_away_from_switch():
    # far from the switch: CNN steps count before it, gyro steps after it, nothing else
    cnn = [700, 728, 756, 1300, 1328]            # the last two are in the turn: ignored
    gyro = [701, 729, 1301, 1329, 1357]          # the first two are on the straight: ignored
    n = fake_nav(cnn, gyro)
    assert n.steps == 6, n.steps
    assert (n.cnn_steps, n.gyro_steps) == (5, 5)


def test_walk_through_a_turn_no_double_no_gap():
    # steps every 28 samples through a turn from 1000 to 1200; detectors jitter by up to 4 samples
    true = list(range(860, 1400, 28))
    rng = np.random.default_rng(0)
    cnn = [s + int(rng.integers(-4, 5)) for s in true]
    gyro = [s + int(rng.integers(-4, 5)) for s in true]
    n = fake_nav(cnn, gyro, switch_back=1200)
    assert n.steps == len(true), (n.steps, len(true))


def test_gyro_source_counts_every_gyro_step():
    # the board's setting: every gyro step counts, turning or not; CNN steps are only logged
    cnn = [700, 728, 756, 1300, 1328]
    gyro = [701, 729, 757, 1301, 1329, 1357]
    n = fake_nav(cnn, gyro, source="gyro")
    assert n.steps == 6, n.steps
    assert (n.cnn_steps, n.gyro_steps) == (5, 6)
    assert n.snapshot()["detector"] == "gyro"
    assert n.distance_m > 0


def test_late_step_dedupes():
    # an edge step reported in a later process() call, for a step already counted
    n = fake_nav(cnn=[600, 628], gyro=[])
    n.cnn.process = lambda: [(627, 0)]
    n.gyro.process = lambda: []
    n.process()
    assert n.steps == 2


def feed(n, seconds, yaw_dps=0.0, walk=False, t0=0):
    """Thigh vertical (x up); rotate about it at yaw_dps (+ = right = negative gx)."""
    for k in range(int(seconds * nav.FS)):
        gy = 100 * np.sin(2 * np.pi * k / 55) if walk else 0.0
        n.add({"t": t0 + 20 * k, "ax": 1.0, "ay": 0.0, "az": 0.0, "gx": -yaw_dps, "gy": gy, "gz": 0.0})
    return t0 + 20 * int(seconds * nav.FS)


def test_heading_and_turn():
    n = nav.Navigator(MODEL)
    t = feed(n, 2)
    n.start_session(None)
    t = feed(n, 1.5, yaw_dps=60, t0=t)          # 90 deg right in 1.5 s
    t = feed(n, 2, t0=t)
    s = n.snapshot()
    assert abs(s["heading_deg"] - 90) < 3, s
    assert s["turns"] == 1 and s["last_turn"] == "R", s
    t = feed(n, 1.0, yaw_dps=-70, t0=t)          # 70 deg left in 1 s
    feed(n, 2, t0=t)
    s = n.snapshot()
    assert s["turns"] == 2 and s["last_turn"] == "L", s
    assert [e for e, _ in n.take_events()] == ["TURN_R", "TURN_L"]


def test_u_turn_counts_once_and_slow_drift_none():
    n = nav.Navigator(MODEL)
    t = feed(n, 2)
    n.start_session(None)
    t = feed(n, 3.0, yaw_dps=60, t0=t)           # 180 deg in one sweep
    t = feed(n, 2, t0=t)
    assert n.turns == 1, n.turns
    feed(n, 20, yaw_dps=5, t0=t)                 # 100 deg over 20 s: a curve, not a turn
    assert n.turns == 1, n.turns


def test_flag_hysteresis():
    n = nav.Navigator(MODEL)
    t = feed(n, 2)
    t = feed(n, 2, yaw_dps=40, t0=t)             # above enter: turning
    assert n.turning
    t = feed(n, 2, yaw_dps=20, t0=t)             # between exit and enter: stays turning
    assert n.turning
    feed(n, 1.5, yaw_dps=5, t0=t)                # below exit for > 0.5 s: off
    assert not n.turning
    assert len(n.switches) == 2


def test_session_log():
    with tempfile.TemporaryDirectory() as d:
        rec = session.Recorder(Path(d))
        n = nav.Navigator(MODEL)
        feed(n, 1)
        name = rec.start(n.snapshot(), wall=time.time())
        rec.add_sample({"t": 1000, "ax": 1, "ay": 0, "az": 0, "gx": 0, "gy": 0, "gz": 0})
        rec.event("LEFT")
        rec.event("STAIRS")
        rec.row(n.snapshot(), [])
        rec.row(n.snapshot(), [("TURN_R", 5)])
        rec.flush()
        rec.stop(n.snapshot())
        rows = list(csv.DictReader(open(Path(d) / f"{name}.csv")))
        assert rows[0]["event"] == "START"
        assert rows[1]["event"] == "LEFT;STAIRS"
        assert rows[2]["event"] == "TURN_R"
        assert rows[-1]["event"] == "STOP"
        assert set(session.FIELDS) == set(rows[0])
        assert (Path(d) / f"{name}_imu.csv").exists() and (Path(d) / f"{name}.json").exists()
        assert [s["name"] for s in rec.list()] == [name]
        assert rec.path_for(f"{name}.csv") and rec.path_for("../etc/passwd") is None


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    board_source = nav.STEP_SOURCE
    for name, fn in tests:
        nav.STEP_SOURCE = board_source
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
