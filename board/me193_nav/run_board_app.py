"""Run the board's Python app (python/main.py) on the laptop, fed from a recording.

It stands in for the board runtime (arduino.app_utils: App, Bridge, Logger) and
replays an IMU recording into the app's imu_sample Bridge handler in real time
(or faster with --speed), so the phone page, logging and step detection can be
tested without the board. LED icon requests are printed as ASCII icons read
from sketch/sketch.ino.

Recordings: an old me193-step-counter trial (t_ms, ax..gz[, us_mm, us_hit]) or
a me193_nav session's <name>_imu.csv. Sessions go to --record-dir (default:
the scratch folder data/sim/nav, never data/raw/).

    python board/me193_nav/run_board_app.py --replay ../me193-step-counter/data/raw/bench_05.csv --loop
    then open http://localhost:7000/ (or http://<laptop IP>:7000/ on a phone on the same Wi-Fi)

    python board/me193_nav/run_board_app.py --icons      # just print the LED icons
"""

import argparse
import csv
import os
import re
import runpy
import sys
import threading
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
APP_DIR = HERE / "python"
SKETCH = HERE / "sketch" / "sketch.ino"
FIELDS = ("t_ms", "ax", "ay", "az", "gx", "gy", "gz")


def read_icons():
    """{label: [8 row strings]} from the ICON_* arrays in the sketch."""
    text = SKETCH.read_text()
    icons = {}
    for name, body in re.findall(r"ICON_(\w+)\[8\]\s*=\s*\{(.*?)\};", text, re.S):
        icons[name.lower()] = re.findall(r'"([.#]{13})"', body)
    return icons


def draw(label, icons):
    rows = icons.get(label)
    if rows is None:
        return f"    (no icon for {label!r}: matrix blank)"
    return "\n".join("    " + "".join("██" if c == "#" else "· " for c in r) for r in rows)


def make_runtime(icons, deadline):
    """A stand-in arduino.app_utils module."""
    handlers = {}

    class Logger:
        def __init__(self, name):
            self.name = name

        def _log(self, level, msg):
            print(f"{time.strftime('%H:%M:%S')} {level:7s} {self.name}: {msg}", flush=True)

        def info(self, msg):
            self._log("INFO", msg)

        def warning(self, msg):
            self._log("WARNING", msg)

        def error(self, msg):
            self._log("ERROR", msg)

    shown = [None]

    class Bridge:
        @staticmethod
        def provide(name, fn):
            handlers[name] = fn

        @staticmethod
        def call(name, *args):
            if name == "show_icon" and args[0] != shown[0]:
                shown[0] = args[0]
                print(f"LED matrix -> {args[0]}\n{draw(args[0], icons)}", flush=True)
            return None

        @staticmethod
        def notify(name, *args):
            pass

    class App:
        @staticmethod
        def run(user_loop=None):
            while deadline is None or time.monotonic() < deadline:
                if user_loop:
                    user_loop()
                else:
                    time.sleep(0.5)

    module = types.ModuleType("arduino.app_utils")
    module.App, module.Bridge, module.Logger = App, Bridge, Logger
    package = types.ModuleType("arduino")
    package.app_utils = module
    sys.modules["arduino"], sys.modules["arduino.app_utils"] = package, module
    return handlers


def feed(path, handlers, speed, loop):
    """Replay the recording into the imu_sample handler at speed x real time."""
    with open(path, newline="") as f:
        rows = [tuple(float(r[k]) for k in FIELDS) for r in csv.DictReader(f)]
    while "imu_sample" not in handlers:
        time.sleep(0.05)
    offset = 0
    while True:
        start, t0 = time.monotonic(), rows[0][0]
        for r in rows:
            due = start + (r[0] - t0) / 1000 / speed
            time.sleep(max(0.0, due - time.monotonic()))
            handlers["imu_sample"](int(r[0] - t0) + offset, *r[1:])
        offset += int(rows[-1][0] - t0) + 20
        print(f"replayed {path.name} ({len(rows)} samples)", flush=True)
        if not loop:
            return


def main():
    ap = argparse.ArgumentParser(description="Run the me193_nav board app on the laptop from a recording.")
    ap.add_argument("--replay", type=Path, help="IMU CSV to replay (t_ms, ax..gz)")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed (1 = real time)")
    ap.add_argument("--loop", action="store_true", help="replay the recording over and over")
    ap.add_argument("--port", type=int, default=7000)
    ap.add_argument("--record-dir", type=Path, default=REPO / "data" / "sim" / "nav")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this long")
    ap.add_argument("--icons", action="store_true", help="print the LED icons from the sketch and exit")
    args = ap.parse_args()

    icons = read_icons()
    if args.icons:
        for label in icons:
            print(f"{label}\n{draw(label, icons)}\n")
        return
    if not args.replay:
        ap.error("--replay is required")
    os.environ["NAV_RECORD_DIR"] = str(args.record_dir.resolve())
    os.environ["NAV_PORT"] = str(args.port)
    deadline = time.monotonic() + args.seconds if args.seconds else None
    handlers = make_runtime(icons, deadline)
    threading.Thread(target=feed, args=(args.replay, handlers, args.speed, args.loop), daemon=True).start()
    print(f"board app on the laptop: replaying {args.replay} at {args.speed}x; http://localhost:{args.port}/",
          flush=True)

    sys.path.insert(0, str(APP_DIR))
    os.chdir(APP_DIR)
    try:
        runpy.run_path(str(APP_DIR / "main.py"), run_name="__main__")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
