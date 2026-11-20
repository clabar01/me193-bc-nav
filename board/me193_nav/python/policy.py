"""The navigation policy on the board (numpy only): features -> action, smoothed, plus meters to go.

Trained and exported by scripts/train_policy.py to models/policy.npz (copied
to python/models/). The training script imports features() from here, so the
board and training compute exactly the same inputs.

Features (FEATURES, from one nav.Navigator.snapshot() at 10 Hz):
    since_turn_m, turns, distance_m, sin(heading), cos(heading)[, in_stairs]
in_stairs is only used if the exported model has it (see the training
report): it is a label (STAIRS presses) in the demos and is not measured on
the board, so the board passes the in_stairs it is given (default 0).

Actions: STRAIGHT, LEFT, RIGHT, STAIRS, ARRIVED. The classifier is either
    "mlp"  z-score -> Linear -> ReLU -> ... -> softmax
    "knn"  z-scored Euclidean distance, k nearest training rows, 1/d weighted vote
and "meters to go" (the "in ~X m") is a kNN regression on the same features
of the training rows' distance to their next action.

Smoother: majority vote over the last SMOOTH predictions (10 = 1 s at 10 Hz);
a tie keeps the current output.
"""

import collections
import math

import numpy as np

SMOOTH = 10


def features(snap, in_stairs=0.0, names=None):
    h = math.radians(snap["heading_deg"])
    allf = {"since_turn_m": snap["since_turn_m"], "turns": snap["turns"], "distance_m": snap["distance_m"],
            "sin_heading": math.sin(h), "cos_heading": math.cos(h), "in_stairs": float(in_stairs)}
    return np.array([allf[n] for n in (names or FEATURES)], dtype=float)


FEATURES = ("since_turn_m", "turns", "distance_m", "sin_heading", "cos_heading", "in_stairs")


class Policy:
    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.kind = str(d["kind"])
        self.classes = [str(c) for c in d["classes"]]
        self.features = [str(f) for f in d["features"]]
        self.mean, self.std = d["mean"].astype(float), d["std"].astype(float)
        if self.kind == "mlp":
            n = len([k for k in d.files if k.startswith("w")])
            self.layers = [(d[f"w{i}"], d[f"b{i}"]) for i in range(n)]
        else:
            self.X, self.y, self.k = d["knn_X"], d["knn_y"].astype(int), int(d["knn_k"])
        self.reg_X, self.reg_m, self.reg_k = d["reg_X"], d["reg_m"].astype(float), int(d["reg_k"])
        self.hist = collections.deque(maxlen=SMOOTH)
        self.smoothed = "STRAIGHT"

    def reset(self):
        self.hist.clear()
        self.smoothed = "STRAIGHT"

    def _z(self, f):
        return (np.asarray(f, float) - self.mean) / self.std

    def probs(self, f):
        z = self._z(f)
        if self.kind == "mlp":
            h = z
            for i, (w, b) in enumerate(self.layers):
                h = w @ h + b
                if i < len(self.layers) - 1:
                    h = np.maximum(h, 0)
            e = np.exp(h - h.max())
            return e / e.sum()
        d = np.sqrt(((self.X - z) ** 2).sum(axis=1))
        nn = np.argsort(d)[:self.k]
        p = np.zeros(len(self.classes))
        np.add.at(p, self.y[nn], 1 / (d[nn] + 1e-6))
        return p / p.sum()

    def meters_to_go(self, f):
        d = np.sqrt(((self.reg_X - self._z(f)) ** 2).sum(axis=1))
        nn = np.argsort(d)[:self.reg_k]
        return float(np.average(self.reg_m[nn], weights=1 / (d[nn] + 1e-6)))

    def step(self, snap, in_stairs=0.0):
        """One 10 Hz step: (raw action, smoothed action, meters to go)."""
        f = features(snap, in_stairs, self.features)
        raw = self.classes[int(np.argmax(self.probs(f)))]
        self.hist.append(raw)
        counts = collections.Counter(self.hist)
        top, n = counts.most_common(1)[0]
        if n > counts.get(self.smoothed, 0):
            self.smoothed = top
        return raw, self.smoothed, round(self.meters_to_go(f), 1)
