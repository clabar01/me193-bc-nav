"""Live step detectors on the board, numpy only: the IMU-only CNN and the gyro stride detector.

Both take one sample at a time (add) and, every 0.25 s, report the steps they
have decided since the last call (process). A step is a global sample index
(0 = first sample since the app started) plus its sketch time t. Each detector
decides each sample once, so overlapping windows never report a step twice.
Neither computes step length or knows about turning: nav.py fuses them.

CNNSteps: the me193-step-counter live pipeline (board/python/live_steps.py) for
the 6-channel model (models/step_detector_imu.npz, scripts/train_detector_imu.py):
    1. despike the IMU, z-score, run the CNN over the last 6 s
    2. a sample's step mass is final once the CNN has seen 1.2 s after it
    3. a final sample is a step if its mass >= 0.5, it is the largest within
       +-0.25 s, and the last step was >= 0.25 s earlier
    4. edge steps (counting.count_steps): at a walking bout's start (once its
       2nd step is known) and end (1.5 s without a step), the gyro stride
       detector runs on a window of gy around the bout's first / last step and
       adds a step 0.5-1.5 step intervals beyond it if the CNN missed one
    Steps are reported ~1.3-1.5 s after they happen (end-of-walk edge steps ~3 s).

GyroSteps: strides.step_times (two steps per forward swing of the thigh) run on
the last 8 s of gy every call; a step is final once it is GYRO_LAG samples old.
    Steps are reported ~1.5-1.75 s after they happen.
"""

import collections

import numpy as np

FS = 50
BUFFER = 300                 # CNN window, 6 s
RF_HALF = 60                 # CNN receptive field: 121 samples = +-60
MASS_K = 5                   # step mass = 100 ms moving sum
PEAK_HALF = 6                # a step is the largest mass within +-6 samples (0.25 s window)
PEAK_HEIGHT = 0.5
MIN_GAP = 13                 # samples between steps (0.25 s)
FINAL_LAG = RF_HALF + 2 + MASS_K // 2 + 1   # newest sample index - this = newest final mass
# Edge steps: the same constants as training/step_detector/counting.py
EDGE_GAP = 75                # samples (1.5 s) without a step end a walking bout
START_WINDOW = (-150, 70)    # gy window around a bout's first / last step, samples
END_WINDOW = (-150, 100)
END_GAP_STEPS = 6
LONE_GAP = 30
HISTORY = 500                # IMU samples kept for the edge check (10 s)

GYRO_WINDOW = 400            # gyro detector window, 8 s
GYRO_LAG = 75                # a gyro step is final once this many newer samples exist (1.5 s)

# Same glitch filter as training/step_detector/data.py
SPIKE = np.array([6.0, 6.0, 6.0, 500.0, 500.0, 500.0])
AGREE = SPIKE / 2


def despike(x):
    x = np.array(x, dtype=float)
    if len(x) < 3:
        return x
    prev, cur, nxt = x[:-2], x[1:-1], x[2:]
    far = (np.abs(cur - prev) > SPIKE) & (np.abs(cur - nxt) > SPIKE)
    agree = np.abs(prev - nxt) <= AGREE
    bad = np.flatnonzero((far & agree).any(axis=1)) + 1
    x[bad] = (x[bad - 1] + x[bad + 1]) / 2
    return x


# ---- gyro stride detector: numpy copy of training/step_detector/strides.py ----
SWING_MIN_DPS = 60
MOVING_DPS = 20
PERIOD_RANGE_S = (0.5, 2.5)


def find_peaks(x, height, distance):
    """scipy.signal.find_peaks(x, height=height, distance=distance)[0], in numpy:
    local maxima (a flat top counts once, at its middle), >= height, then the
    tallest first, dropping lower peaks closer than distance."""
    x = np.asarray(x, dtype=float)
    peaks, i, i_max = [], 1, len(x) - 1
    while i < i_max:
        if x[i - 1] < x[i]:
            ahead = i + 1
            while ahead < i_max and x[ahead] == x[i]:
                ahead += 1
            if x[ahead] < x[i]:
                peaks.append((i + ahead - 1) // 2)
                i = ahead
        i += 1
    peaks = np.array(peaks, dtype=int)
    peaks = peaks[x[peaks] >= height] if len(peaks) else peaks
    if len(peaks) < 2 or distance <= 1:
        return peaks
    keep = np.ones(len(peaks), dtype=bool)
    for j in np.argsort(x[peaks])[::-1]:
        if not keep[j]:
            continue
        k = j - 1
        while k >= 0 and peaks[j] - peaks[k] < distance:
            keep[k] = False
            k -= 1
        k = j + 1
        while k < len(peaks) and peaks[k] - peaks[j] < distance:
            keep[k] = False
            k += 1
    return peaks[keep]


def stride_period(gy):
    moving = np.abs(gy) > MOVING_DPS
    if moving.sum() < 2 * FS:
        return None
    w = np.where(moving, gy, 0.0)
    w = w - w[moving].mean() * moving
    ac = np.correlate(w, w, "full")[len(w) - 1:]
    lo, hi = int(PERIOD_RANGE_S[0] * FS), min(int(PERIOD_RANGE_S[1] * FS), len(ac) - 1)
    if hi <= lo:
        return None
    return lo + int(np.argmax(ac[lo:hi]))


def gyro_step_times(gy):
    """strides.step_times: the steps the gyro detector implies (2 per stride)."""
    gy = np.asarray(gy, dtype=float)
    period = stride_period(gy)
    peaks = find_peaks(gy, SWING_MIN_DPS, int(0.6 * period) if period else int(0.4 * FS))
    if len(peaks):
        peaks = peaks[gy[peaks] >= 0.5 * np.median(gy[peaks])]
    steps = []
    for i, p in enumerate(peaks):
        local = (peaks[i + 1] - p) if i + 1 < len(peaks) else (p - peaks[i - 1]) if i else (period or FS)
        after = gy[p:p + int(0.6 * local)]
        down = np.flatnonzero((after[:-1] > 0) & (after[1:] <= 0))
        hs = p + (down[0] + 1 if len(down) else int(0.23 * local))
        steps += [hs, hs + local // 2]
    return np.array([s for s in steps if s < len(gy)], dtype=int)


class StepCNN:
    """numpy forward pass of training/step_detector/model.py (dilated Conv1d, same padding)."""

    def __init__(self, path):
        d = np.load(path)
        self.channels = [str(c) for c in d["channels"]]
        if self.channels != ["ax", "ay", "az", "gx", "gy", "gz"]:
            raise ValueError(f"{path}: expected the 6-channel IMU-only model, got {self.channels}")
        self.mean, self.std = d["mean"].astype(float), d["std"].astype(float)
        self.layers = [(d[f"conv{i}_w"], d[f"conv{i}_b"], int(dil)) for i, dil in enumerate(d["dilations"])]
        self.head_w, self.head_b = d["head_w"], d["head_b"]

    def __call__(self, x):
        """x: (6, T) z-scored -> (T,) step output in 0..1."""
        h = x.astype(np.float32)
        for w, b, dil in self.layers:
            out_ch, in_ch, k = w.shape
            pad = dil * (k - 1) // 2
            hp = np.pad(h, ((0, 0), (pad, pad)))
            t = h.shape[1]
            cols = np.stack([hp[:, j * dil:j * dil + t] for j in range(k)], axis=1)     # (in, k, T)
            h = np.maximum(w.reshape(out_ch, -1) @ cols.reshape(in_ch * k, t) + b[:, None], 0)
        z = self.head_w @ h + self.head_b[:, None]
        return 1 / (1 + np.exp(-np.clip(z[0], -60, 60)))


class CNNSteps:
    def __init__(self, model_path):
        self.cnn = StepCNN(model_path)
        self.buf = collections.deque(maxlen=BUFFER)    # (index, t, ax..gz)
        self.imu = collections.deque(maxlen=HISTORY)   # same, longer, for the edge check
        self.n = 0
        self.mass = {}                                 # finalized index -> step mass (recent only)
        self.final_upto = -1
        self.decided_upto = -1                         # every sample <= this has been decided
        self.step_idx = []                             # recent step indices
        self.bout = []                                 # CNN steps of the current walking bout
        self.bout_open = False

    def add(self, index, t, imu6):
        row = (index, t, *imu6)
        self.buf.append(row)
        self.imu.append(row)
        self.n = index + 1

    def process(self):
        """Steps decided since the last call: [(index, t)], in the order found
        (an edge step at a bout's start can come after later steps)."""
        out = []
        if len(self.buf) < 2 * FINAL_LAG + 1:
            return out
        arr = np.array(self.buf, dtype=float)
        first = int(arr[0, 0])
        x = despike(arr[:, 2:8])
        y = self.cnn(((x - self.cnn.mean) / self.cnn.std).T)
        mass = np.convolve(y, np.ones(MASS_K), "same")

        lo = max(self.final_upto + 1, first + FINAL_LAG)
        hi = first + len(arr) - 1 - FINAL_LAG
        for i in range(lo, hi + 1):
            self.mass[i] = float(mass[i - first])
        if hi >= lo:
            self.final_upto = hi

        # decide each sample once its +-PEAK_HALF neighbours are final
        for j in range(self.decided_upto + 1, self.final_upto - PEAK_HALF + 1):
            m = self.mass.get(j, 0.0)
            if m >= PEAK_HEIGHT and (not self.step_idx or j - self._last_cnn() >= MIN_GAP):
                left = max((self.mass.get(i, 0.0) for i in range(j - PEAK_HALF, j)), default=0.0)
                right = max(self.mass.get(i, 0.0) for i in range(j + 1, j + PEAK_HALF + 1))
                if m > left and m >= right:
                    self._on_cnn_step(j, arr, first, out)
            self.decided_upto = j
            if self.bout_open and j >= self.bout[-1] + EDGE_GAP - 1:
                self._close_bout(out)
        for i in [i for i in self.mass if i < self.decided_upto - 2 * PEAK_HALF]:
            del self.mass[i]
        return out

    def _last_cnn(self):
        return self.bout[-1] if self.bout else -10 ** 9

    def _on_cnn_step(self, j, arr, first, out):
        if self.bout_open and j - self.bout[-1] < EDGE_GAP:
            self.bout.append(j)
        else:
            self.bout, self.bout_open = [j], True
        if len(self.bout) == 2:
            self._edge(self.bout[0], self.bout[1] - self.bout[0], "start", out)
        self._count(j, int(arr[j - first, 1]), out)

    def _close_bout(self, out):
        b = self.bout
        if len(b) == 1:
            self._edge(b[0], LONE_GAP, "start", out)
        gap = float(np.median(np.diff(b[-(END_GAP_STEPS + 1):]))) if len(b) > 1 else LONE_GAP
        self._edge(b[-1], gap, "end", out)
        self.bout_open = False

    def _edge(self, step, gap, side, out):
        """Add the gyro detector's step 0.5-1.5 intervals beyond `step`, if there is one."""
        lo, hi = START_WINDOW if side == "start" else END_WINDOW
        rows = np.array(self.imu, dtype=float)
        start = int(rows[0, 0])
        a, b = max(0, step + lo), min(self.n, step + hi)
        a0, b0 = max(a - 1, start), min(b + 1, start + len(rows))        # one more each side for despike
        if b - a < 3 or a0 > a:
            return
        gy = despike(rows[a0 - start:b0 - start, 2:8])[a - a0:a - a0 + (b - a), 4]
        sign = -1 if side == "start" else 1
        cands = [int(g) + a for g in gyro_step_times(gy) if 0.5 * gap <= sign * (int(g) + a - step) <= 1.5 * gap]
        if not cands:
            return
        e = min(cands, key=lambda g: abs(sign * (g - step) - gap))
        if any(abs(e - k) < MIN_GAP for k in self.step_idx):        # already counted (counting.edge_steps)
            return
        self._count(e, int(rows[e - start, 1]), out)

    def _count(self, j, t, out):
        self.step_idx = sorted(self.step_idx + [j])[-20:]
        out.append((j, t))


class GyroSteps:
    def __init__(self):
        self.buf = collections.deque(maxlen=GYRO_WINDOW)   # (index, t, ax..gz)
        self.decided_upto = -1
        self.recent = collections.deque(maxlen=20)         # counted step indices

    def add(self, index, t, imu6):
        self.buf.append((index, t, *imu6))

    def process(self):
        """Steps decided since the last call: [(index, t)] in time order."""
        if len(self.buf) < 3:
            return []
        arr = np.array(self.buf, dtype=float)
        first, newest = int(arr[0, 0]), int(arr[-1, 0])
        final = newest - GYRO_LAG
        if final <= self.decided_upto:
            return []
        gy = despike(arr[:, 2:8])[:, 4]
        out = []
        # The window slides, so a step can move by a sample or two between calls:
        # look back MIN_GAP past the last decision and skip steps already counted.
        for s in gyro_step_times(gy):
            j = int(s) + first
            if self.decided_upto - MIN_GAP < j <= final and all(abs(j - k) >= MIN_GAP for k in self.recent):
                self.recent.append(j)
                out.append((j, int(arr[s, 1])))
        self.decided_upto = final
        return sorted(out)
