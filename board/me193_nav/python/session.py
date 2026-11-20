"""Record-mode session log: a 10 Hz CSV of the navigation state plus button events.

Pure Python (no Arduino imports). main.py calls start() / stop() and event()
from the web server, add_sample() from the Bridge handler (cheap: appends to a
list), and row() + flush() from the 10 Hz logging thread.

Each session <name> = rec_<YYYYmmdd-HHMMSS> writes three files to the folder:
  <name>.csv       10 Hz, FIELDS below. One row every 0.1 s while recording,
                   plus a START row and a STOP row.
  <name>_imu.csv   every IMU sample (50 Hz): t_ms, ax, ay, az, gx, gy, gz, so
                   the session can be replayed through other detectors later.
  <name>.json      metadata, written at stop (mode: record / navigate / dagger,
                   person and their K, start/stop times, totals, constants).
Record and DAgger sessions are demos: clean them (scripts/clean_labels.py) and
scripts/train_policy.py trains on them.

CSV columns:
  unix_time          board clock, s
  t_s                s since the session started (sketch clock)
  distance_m         total distance (Weinberg on the counted steps)
  since_turn_m       distance since the last detected turn
  heading_deg        + = right, 0 at the session start, unwrapped
  turns              turns detected (>= 60 deg within 2 s)
  steps              counted steps: the step source in the JSON (nav.STEP_SOURCE;
                     "gyro" since 2026-11-20, earlier sessions "fused")
  cnn_steps          the CNN's own running count
  gyro_steps         the gyro stride detector's own running count
  detector           which detector steps is taking steps from now: always gyro
                     with STEP_SOURCE "gyro"; cnn (straight) / gyro (turning) with "fused"
  event              button presses since the last row (LEFT, RIGHT, STAIRS,
                     ARRIVED; in DAgger mode also STRAIGHT = "the policy is
                     wrong here"), detected turns (TURN_L, TURN_R), START,
                     STOP; several are joined with ";"
  advice             the policy's smoothed action (STRAIGHT, LEFT, RIGHT,
                     STAIRS, ARRIVED; 1 s majority; SENSOR_LOST while no IMU
                     samples arrive), what Navigate / DAgger
                     show and speak; logged in every mode (empty if no policy)
  advice_raw         the policy's unsmoothed action
  advice_m           meters to the next action (the "in ~X m")
Steps (and so distance) appear ~1.5 s after they happen: the detectors need
data after each step before deciding it.
"""

import datetime
import json
import re
import threading
import time
from pathlib import Path

FIELDS = ("unix_time", "t_s", "distance_m", "since_turn_m", "heading_deg", "turns", "steps",
          "cnn_steps", "gyro_steps", "detector", "event", "advice", "advice_raw", "advice_m")
IMU_FIELDS = ("t_ms", "ax", "ay", "az", "gx", "gy", "gz")
IMU_FORMATS = ("{}", "{:.4f}", "{:.4f}", "{:.4f}", "{:.2f}", "{:.2f}", "{:.2f}")
BUTTONS = ("LEFT", "RIGHT", "STAIRS", "ARRIVED", "STRAIGHT")
MODES = ("record", "navigate", "dagger")
MODE_BUTTONS = {"record": BUTTONS[:4], "navigate": (), "dagger": BUTTONS}
NAME_RE = re.compile(r"^rec_\d{8}-\d{6}(-\d+)?(_imu)?\.(csv|json)$")


class SessionError(ValueError):
    pass


class Recorder:
    def __init__(self, folder, meta=None):
        self.folder = Path(folder)
        self.meta_extra = meta or {}
        self.lock = threading.Lock()
        self.name = None
        self.files = None
        self.imu_rows = []
        self.pending = []                # button events not yet in a row
        self.rows = 0
        self.t0 = None
        self.started_wall = None
        self.last_events = []            # (unix_time, name) of recent presses, for the page
        self.mode, self.person, self.k = "record", None, None

    @property
    def recording(self):
        return self.name is not None

    # ---------- commands ----------
    def start(self, snap, wall=None, mode="record", person=None, k=None):
        if mode not in MODES:
            raise SessionError(f"mode must be one of {', '.join(MODES)}")
        with self.lock:
            if self.name:
                raise SessionError("already recording")
            self.mode, self.person, self.k = mode, person, k
            self.folder.mkdir(parents=True, exist_ok=True)
            wall = wall or time.time()
            base = "rec_" + datetime.datetime.fromtimestamp(wall).strftime("%Y%m%d-%H%M%S")
            name, k = base, 1
            while (self.folder / f"{name}.csv").exists():
                k += 1
                name = f"{base}-{k}"
            self.files = ((self.folder / f"{name}.csv").open("w", newline=""),
                          (self.folder / f"{name}_imu.csv").open("w", newline=""))
            self.files[0].write(",".join(FIELDS) + "\n")
            self.files[1].write(",".join(IMU_FIELDS) + "\n")
            self.name, self.t0, self.started_wall = name, snap["t"], wall
            self.rows, self.imu_rows, self.pending, self.last_events = 0, [], [], []
            self._write(snap, ["START"], wall)
            return name

    def stop(self, snap):
        with self.lock:
            if not self.name:
                raise SessionError("not recording")
            name = self.name
            self._write(snap, self.pending + ["STOP"], time.time())
            self.pending = []
            self._flush()
            for f in self.files:
                f.close()
            doc = {"name": name, "csv": f"{name}.csv", "imu_csv": f"{name}_imu.csv",
                   "started_at": _iso(self.started_wall), "stopped_at": _iso(time.time()),
                   "rows": self.rows,
                   "seconds": round((snap["t"] - self.t0) / 1000, 2) if None not in (snap["t"], self.t0) else None,
                   "totals": snap, **self.meta_extra, "mode": self.mode, "person": self.person,
                   "weinberg_K": self.k if self.k is not None else self.meta_extra.get("weinberg_K")}
            (self.folder / f"{name}.json").write_text(json.dumps(doc, indent=2) + "\n")
            self.name, self.files = None, None
            return name, self.rows

    def event(self, button):
        if button not in BUTTONS:
            raise SessionError(f"unknown button {button!r}")
        with self.lock:
            if not self.name:
                raise SessionError("not recording: press Start first")
            if button not in MODE_BUTTONS[self.mode]:
                raise SessionError(f"{button} isn't a label in {self.mode} mode")
            self.pending.append(button)
            self.last_events = (self.last_events + [(round(time.time(), 2), button)])[-5:]

    # ---------- data ----------
    def add_sample(self, s):
        if self.name is None:            # unlocked read: a sample at the edge may go either way
            return
        with self.lock:
            if self.name is not None:
                self.imu_rows.append(tuple(s[k] for k in ("t", "ax", "ay", "az", "gx", "gy", "gz")))

    def row(self, snap, nav_events, advice=None):
        """One 10 Hz row: the snapshot plus button presses and detected turns since the last row."""
        with self.lock:
            if not self.name:
                return
            self._write(snap, self.pending + [e for e, _ in nav_events], time.time(), advice)
            self.pending = []

    def _write(self, snap, events, wall, advice=None):
        if self.t0 is None and snap["t"] is not None:
            self.t0 = snap["t"]           # started before any sample arrived (sensor down): time from the first
        t_s = (snap["t"] - self.t0) / 1000 if snap["t"] is not None and self.t0 is not None else 0.0
        vals = (f"{wall:.2f}", f"{t_s:.2f}", f"{snap['distance_m']:.2f}", f"{snap['since_turn_m']:.2f}",
                f"{snap['heading_deg']:.1f}", snap["turns"], snap["steps"], snap["cnn_steps"],
                snap["gyro_steps"], snap["detector"], ";".join(events),
                *((advice["advice"], advice["advice_raw"], f"{advice['advice_m']:.1f}") if advice else ("", "", "")))
        self.files[0].write(",".join(str(v) for v in vals) + "\n")
        self.rows += 1

    def flush(self):
        with self.lock:
            self._flush()

    def _flush(self):
        if not self.files:
            return
        rows, self.imu_rows = self.imu_rows, []
        self.files[1].write("".join(",".join(f.format(v) for f, v in zip(IMU_FORMATS, r)) + "\n"
                                    for r in rows))
        for f in self.files:
            f.flush()

    # ---------- for the page ----------
    def info(self):
        with self.lock:
            return {"recording": self.name is not None, "name": self.name, "rows": self.rows,
                    "mode": self.mode if self.name else None, "person": self.person if self.name else None,
                    "started": self.started_wall, "events": list(self.last_events)}

    def list(self):
        """Sessions in the folder, newest first."""
        out = []
        for p in sorted(self.folder.glob("rec_*.csv"), reverse=True):
            if p.name.endswith("_imu.csv"):
                continue
            name = p.stem
            imu = self.folder / f"{name}_imu.csv"
            out.append({"name": name, "csv": p.name, "imu_csv": imu.name if imu.exists() else None,
                        "kb": round(p.stat().st_size / 1024, 1), "recording": name == self.name})
        return out

    def path_for(self, filename):
        """The file to download, or None if the name isn't one of our session files."""
        if not NAME_RE.match(filename or ""):
            return None
        p = self.folder / filename
        return p if p.is_file() else None


def _iso(wall):
    return datetime.datetime.fromtimestamp(wall).astimezone().isoformat(timespec="seconds")
