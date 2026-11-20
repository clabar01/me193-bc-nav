"""Dead reckoning from the thigh IMU: heading, turns, steps (two detectors, fused) and distance.

Pure numpy (no Arduino imports), so the laptop replays recordings through
exactly what the board runs. main.py calls add() from the Bridge handler for
every sample (cheap: a few float ops), and process() every 0.25 s from a worker
thread, under one lock.

Heading (per sample)
    The thigh's yaw rate is the gyro projected onto the gravity direction:
    rate = omega . up. "up" is gravity low-passed with a complementary filter:
    the previous estimate is rotated by the gyro, then pulled toward the
    normalized accelerometer with time constant UP_TAU_S. (A plain low-pass of
    the accelerometer lags the thigh's +-25 deg swing; on the old hallway trials
    that added a false +2-3 deg/s while walking, vs -1.5 with this filter.)
    While walking (|gy| > 20 deg/s within the last 1 s) WALK_YAW_DRIFT_DPS is
    subtracted: a steady drift to the right measured on the old straight
    hallway trials (scripts/calibrate_heading.py). Heading is in degrees, POSITIVE = RIGHT
    (clockwise seen from above), 0 at the start of a recording, unwrapped (a
    full circle to the right reads 360).

Turning flag (per sample, with hysteresis)
    The yaw rate averaged over 1 s CENTRED on the sample, so the flag lines up
    with the turn itself (it is known TURN_DELAY samples later; every step is
    decided later than that anyway). Turning starts when |rate| >= TURN_ENTER_DPS
    and ends after |rate| < TURN_EXIT_DPS for TURN_EXIT_HOLD samples in a row.
    Between the two thresholds it keeps its last value, so it can't flicker.

Turn detector
    A turn is counted when the heading has changed by >= TURN_DEG within the
    last TURN_WINDOW_S (left or right). It then waits until the rotation is over
    (|1 s mean rate| < TURN_EXIT_DPS for TURN_EXIT_HOLD samples) before it can
    count another, so a single U-turn counts once.

Steps: CNN (straight walking) and gyro stride detector (turns)
    Both run all the time and keep their own running counts (cnn_steps,
    gyro_steps). The fused count (steps) takes each detector's steps only where
    that detector is active, judged at the STEP's sample (not when it is
    reported, 1.5 s later): CNN where the turning flag is off, gyro where it is
    on. Near a switch the two detectors can place the same step on either side
    of it, so within SWITCH_MARGIN samples of a switch steps from both are
    accepted; any step within DEDUPE samples of a step already counted is the
    same step and is dropped. That way a step at the boundary is counted once,
    whichever detector sees it, and not lost when each detector puts it on the
    other's side.

Distance (Weinberg)
    Each fused step's length is K * (max|a| - min|a|)^0.25 over the interval
    from the previous fused step (if within 1.5 s; else the last typical step
    interval) to this step. K = 0.7419 (scripts/calibrate_weinberg.py, all 14
    old hallway distance trials). distance_since_turn adds only steps at or
    after the sample where the last turn was detected.
"""

import collections

import numpy as np

from detectors import FS, CNNSteps, GyroSteps

DT = 1 / FS

UP_TAU_S = 3.0
WALK_YAW_DRIFT_DPS = 1.73    # deg/s, + = right; scripts/calibrate_heading.py (median, old A/B hallway trials)
WALK_GY_DPS = 20
WALK_HOLD = FS               # walking = |gy| > WALK_GY_DPS within the last 1 s

RATE_WIN = FS                # turning flag: 1 s centred mean of the yaw rate
TURN_DELAY = RATE_WIN // 2   # ... so it is known this many samples after the sample
TURN_ENTER_DPS = 30.0
TURN_EXIT_DPS = 15.0
TURN_EXIT_HOLD = FS // 2     # 0.5 s below TURN_EXIT_DPS ends a turn

TURN_DEG = 60.0
TURN_WINDOW_S = 2.0

SWITCH_MARGIN = 15           # 0.3 s: steps this close to a detector switch are taken from both
DEDUPE = 13                  # 0.26 s: closer than this to a counted step = the same step

WEINBERG_K = 0.7419
WEINBERG_EXP = 0.25
MAX_STEP_GAP = int(1.5 * FS) # previous step further back than this: use a typical interval instead
DEFAULT_GAP = 28             # typical step interval (0.56 s) before any steps are known

KEEP = 2000                  # per-sample history kept (40 s); steps are decided within ~4 s


class Navigator:
    def __init__(self, model_path):
        self.cnn = CNNSteps(model_path)
        self.gyro = GyroSteps()
        self.n = 0                       # samples added since the app started
        self.t = None                    # newest sketch time, ms
        self.last_t = None
        # heading
        self.up = None
        self.heading = 0.0               # since the app started, deg, + = right
        self.walk_left = 0               # samples of "walking" left after the last |gy| > WALK_GY_DPS
        self.rates = collections.deque(maxlen=RATE_WIN)    # recent yaw rates (deg/s, + = right)
        self.rate_sum = 0.0
        # turning flag: mode[i] for every sample i whose flag is known (i <= mode_upto)
        self.mode = collections.deque(maxlen=KEEP)         # True = turning (gyro active)
        self.mode_base = 0                                  # index of mode[0]
        self.mode_upto = -1
        self.turning = False
        self.calm = 0
        self.switches = collections.deque(maxlen=64)       # sample indices where the flag changed
        self.rate_smooth = 0.0
        # turn detector
        self.head_hist = collections.deque(maxlen=int(TURN_WINDOW_S * FS) + 1)   # (index, heading)
        self.turn_armed = True
        self.turn_calm = 0
        # |a| per sample for Weinberg
        self.acc = collections.deque(maxlen=KEEP)
        self.acc_base = 0
        # fused steps
        self.fused = []                  # recent (index, length_m), sorted
        self.waiting = []                # steps reported before their sample's flag was known
        self.start_session(None)

    # ---------- session ----------
    def start_session(self, index):
        """Zero the totals from sample `index` on (None = from the next sample)."""
        self.start = self.n if index is None else index
        self.heading0 = self.heading
        self.steps = self.cnn_steps = self.gyro_steps = 0
        self.turns = 0
        self.turn_dir = None             # "L" / "R" of the last turn
        self.last_turn_idx = self.start
        self.distance_m = 0.0
        self.since_turn_m = 0.0
        self.events = []                 # ("TURN_L" | "TURN_R", index) since the last take_events()
        if hasattr(self, "head_hist"):
            self.head_hist.clear()

    # ---------- every sample ----------
    def add(self, s):
        """s: {"t", "ax", "ay", "az", "gx", "gy", "gz"} (g, deg/s, body frame)."""
        i = self.n
        a = np.array((s["ax"], s["ay"], s["az"]), dtype=float)
        w = np.array((s["gx"], s["gy"], s["gz"]), dtype=float)
        dt = DT if self.last_t is None else min(max((s["t"] - self.last_t) / 1000, DT), 0.1)
        self.last_t = self.t = s["t"]
        imu6 = (s["ax"], s["ay"], s["az"], s["gx"], s["gy"], s["gz"])
        self.cnn.add(i, s["t"], imu6)
        self.gyro.add(i, s["t"], imu6)
        anorm = float(np.linalg.norm(a))
        self.acc.append(anorm)
        if len(self.acc) == self.acc.maxlen:
            self.acc_base = i - len(self.acc) + 1

        # gravity direction: rotate with the gyro, pull toward the accelerometer
        if self.up is None:
            self.up = a / anorm if anorm > 0.2 else np.array([1.0, 0.0, 0.0])
        self.up = self.up - np.cross(np.radians(w), self.up) * dt
        if anorm > 0.2:
            self.up += (dt / (UP_TAU_S + dt)) * (a / anorm - self.up)
        self.up /= np.linalg.norm(self.up)

        self.walk_left = WALK_HOLD if abs(s["gy"]) > WALK_GY_DPS else max(self.walk_left - 1, 0)
        rate = -float(w @ self.up)                     # + = right (clockwise from above)
        if self.walk_left:
            rate -= WALK_YAW_DRIFT_DPS
        self.heading += rate * dt
        self._turning_flag(rate)
        self._turn_detector(i)
        self.n += 1

    def _turning_flag(self, rate):
        if len(self.rates) == self.rates.maxlen:
            self.rate_sum -= self.rates[0]
        self.rates.append(rate)
        self.rate_sum += rate
        if len(self.rates) < RATE_WIN:
            return
        self.rate_smooth = self.rate_sum / RATE_WIN    # centred on sample n - TURN_DELAY
        j = self.n - TURN_DELAY
        r = abs(self.rate_smooth)
        if not self.turning and r >= TURN_ENTER_DPS:
            self.turning, self.calm = True, 0
            self.switches.append(j)
        elif self.turning:
            self.calm = self.calm + 1 if r < TURN_EXIT_DPS else 0
            if self.calm >= TURN_EXIT_HOLD:
                self.turning = False
                self.switches.append(j)
        while self.mode_upto < j:                      # the first TURN_DELAY samples: not turning
            self.mode_upto += 1
            self.mode.append(self.turning if self.mode_upto == j else False)
            if len(self.mode) == self.mode.maxlen:
                self.mode_base = self.mode_upto - len(self.mode) + 1

    def _turn_detector(self, i):
        self.head_hist.append((i, self.heading))
        if not self.turn_armed:
            self.turn_calm = self.turn_calm + 1 if abs(self.rate_smooth) < TURN_EXIT_DPS else 0
            if self.turn_calm >= TURN_EXIT_HOLD:
                self.turn_armed = True
                self.head_hist.clear()
                self.head_hist.append((i, self.heading))
            return
        if i < self.start:
            return
        hs = [h for _, h in self.head_hist]
        d_right, d_left = self.heading - min(hs), max(hs) - self.heading
        if max(d_right, d_left) >= TURN_DEG:
            self.turns += 1
            self.turn_dir = "R" if d_right >= d_left else "L"
            self.last_turn_idx = i
            self.since_turn_m = sum(m for j, m in self.fused if j >= i)
            self.events.append((f"TURN_{self.turn_dir}", i))
            self.turn_armed, self.turn_calm = False, 0

    # ---------- every 0.25 s ----------
    def process(self):
        """Run both detectors and fuse their new steps. Returns the fused steps added: [(index, length_m)]."""
        new = [("cnn", j) for j, _ in self.cnn.process()] + [("gyro", j) for j, _ in self.gyro.process()]
        for src, j in new:
            if j >= self.start:
                if src == "cnn":
                    self.cnn_steps += 1
                else:
                    self.gyro_steps += 1
        added = []
        pending = self.waiting + sorted(new, key=lambda p: p[1])
        self.waiting = []
        for src, j in pending:
            if j + SWITCH_MARGIN > self.mode_upto:     # flag not known yet (doesn't happen with these lags)
                self.waiting.append((src, j))
                continue
            if j < self.start:
                continue
            active = "gyro" if self._mode(j) else "cnn"
            near_switch = any(abs(j - s) <= SWITCH_MARGIN for s in self.switches)
            if src != active and not near_switch:
                continue
            if any(abs(j - k) < DEDUPE for k, _ in self.fused):
                continue
            length = self._length(j)
            self.fused.append((j, length))
            self.fused.sort()
            self.fused = self.fused[-40:]
            self.steps += 1
            self.distance_m += length
            if j >= self.last_turn_idx:
                self.since_turn_m += length
            added.append((j, length))
        return added

    def _mode(self, j):
        k = j - self.mode_base
        return self.mode[k] if 0 <= k < len(self.mode) else False

    def _length(self, j):
        prev = [k for k, _ in self.fused if k < j]
        if prev and j - prev[-1] <= MAX_STEP_GAP:
            a = prev[-1]
        else:
            gaps = np.diff([k for k, _ in self.fused])
            gaps = gaps[(gaps >= DEDUPE) & (gaps <= MAX_STEP_GAP)]
            a = j - (int(np.median(gaps)) if len(gaps) else DEFAULT_GAP)
        lo, hi = max(a - self.acc_base, 0), min(j - self.acc_base, len(self.acc))
        if hi - lo < 2:
            return 0.0
        seg = np.fromiter((self.acc[k] for k in range(lo, hi)), float, hi - lo)
        return WEINBERG_K * float(seg.max() - seg.min()) ** WEINBERG_EXP

    # ---------- for the log and the page ----------
    def take_events(self):
        ev, self.events = self.events, []
        return ev

    def snapshot(self):
        h = self.heading - self.heading0
        return {
            "t": self.t,
            "distance_m": round(self.distance_m, 2),
            "since_turn_m": round(self.since_turn_m, 2),
            "heading_deg": round(h, 1),
            "heading_rate_dps": round(self.rate_smooth, 1),
            "turns": self.turns,
            "last_turn": self.turn_dir,
            "steps": self.steps,
            "cnn_steps": self.cnn_steps,
            "gyro_steps": self.gyro_steps,
            "detector": "gyro" if self.turning else "cnn",
        }
