"""me193_nav: thigh-IMU dead reckoning with a phone page (Record mode).

The sketch reads the MPU-6050 at 50 Hz and calls
Bridge.notify("imu_sample", t, ax, ay, az, gx, gy, gz) for every sample (no
ultrasonic any more). This side:

  every sample   nav.Navigator.add(): heading, turning flag, turn detector
                 (cheap); session.Recorder.add_sample() while recording
  every 0.25 s   nav.Navigator.process(): the CNN and gyro step detectors,
                 fused, and Weinberg distance
  every 0.1 s    while recording, one CSV row (session.py) with the button
                 presses since the last row
  HTTP :PORT     the phone page (web/index.html) and its API:
                   GET  /                  the page
                   GET  /api/state         live numbers + recording state
                   GET  /api/sessions      recorded sessions
                   GET  /files/<name>      download a session CSV / JSON
                   POST /api/start         start recording (zeroes distance, heading, counts)
                   POST /api/stop
                   POST /api/event         {"button": "LEFT" | "RIGHT" | "STAIRS" | "ARRIVED"}
  LED matrix     blank while idle (the sketch clears it at boot, so nothing from
                 a previous app stays on); blinking dot while recording.

Nothing here waits on the sensor: the Bridge handler only does arithmetic and
appends to lists; files are written from the logging thread.

No internet needed: the phone talks to the board directly on the same Wi-Fi
(http://<board IP>:PORT). There is no login, so anyone on that network can
start/stop a recording or download the logs.

Environment (only set by run_board_app.py when this runs on the laptop):
    NAV_RECORD_DIR   where sessions go instead of <app>/recordings
    NAV_PORT         HTTP port instead of 7000
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from arduino.app_utils import App, Bridge, Logger

import nav
from session import FIELDS, Recorder, SessionError

logger = Logger("me193_nav")

HERE = Path(__file__).resolve().parent
MODEL = HERE / "models" / "step_detector_imu.npz"
PAGE = HERE / "web" / "index.html"
RECORD_DIR = Path(os.environ.get("NAV_RECORD_DIR", HERE.parent / "recordings"))
PORT = int(os.environ.get("NAV_PORT", "7000"))

PROCESS_PERIOD_S = 0.25
ROW_PERIOD_S = 0.1
FLUSH_EVERY = 5             # rows between file flushes (0.5 s)
STATUS_PERIOD_S = 5.0
LED_REFRESH_S = 5.0
STALE_S = 1.0               # no sample for this long: the page shows "no sensor data"

lock = threading.Lock()     # the navigator and the recorder's start/stop
navigator = nav.Navigator(MODEL)
recorder = Recorder(RECORD_DIR, meta={
    "weinberg_K": nav.WEINBERG_K, "walk_yaw_drift_dps": nav.WALK_YAW_DRIFT_DPS,
    "turn": {"deg": nav.TURN_DEG, "window_s": nav.TURN_WINDOW_S},
    "turning_flag_dps": {"enter": nav.TURN_ENTER_DPS, "exit": nav.TURN_EXIT_DPS},
    "model": MODEL.name, "columns": list(FIELDS)})

received = 0
last_sample_wall = 0.0
last_mcu_status = ""
proc_ms = 0.0


# ---------- sketch -> Python ----------
def imu_sample(t, ax, ay, az, gx, gy, gz):
    """Bridge handler, once per sample. Must stay fast: no I/O here."""
    global received, last_sample_wall
    s = {"t": int(t), "ax": float(ax), "ay": float(ay), "az": float(az),
         "gx": float(gx), "gy": float(gy), "gz": float(gz)}
    with lock:
        navigator.add(s)
    recorder.add_sample(s)
    received += 1
    last_sample_wall = time.time()


def imu_status(text):
    global last_mcu_status
    last_mcu_status = text
    logger.info(f"sketch: {text}")


# ---------- worker threads ----------
def process_loop():
    global proc_ms
    next_run = time.monotonic()
    while True:
        next_run += PROCESS_PERIOD_S
        time.sleep(max(0.0, next_run - time.monotonic()))
        start = time.perf_counter()
        try:
            with lock:
                navigator.process()
        except Exception as e:          # never let a bad window stop the loop
            logger.error(f"step detection failed: {e}")
        proc_ms = (time.perf_counter() - start) * 1000


def row_loop():
    next_run, k = time.monotonic(), 0
    while True:
        next_run += ROW_PERIOD_S
        time.sleep(max(0.0, next_run - time.monotonic()))
        if not recorder.recording:
            with lock:
                navigator.take_events()          # turns while idle aren't logged
            continue
        try:
            with lock:
                snap, events = navigator.snapshot(), navigator.take_events()
            recorder.row(snap, events)
            k += 1
            if k % FLUSH_EVERY == 0:
                recorder.flush()
        except OSError as e:
            logger.error(f"writing the log failed: {e}")


def led_loop():
    last_icon, last_sent = None, 0.0
    while True:
        time.sleep(0.25)
        icon = "rec" if recorder.recording else "off"      # "off": no such icon -> blank
        now = time.monotonic()
        if icon != last_icon or now - last_sent >= LED_REFRESH_S:
            last_sent = now
            try:
                Bridge.call("show_icon", icon)
                last_icon = icon
            except Exception as e:
                logger.warning(f"show_icon failed: {e}")


def status_loop():
    """App loop: a log line every STATUS_PERIOD_S."""
    global received
    time.sleep(STATUS_PERIOD_S)
    rate, received = received / STATUS_PERIOD_S, 0
    with lock:
        s = navigator.snapshot()
    rec = recorder.info()
    logger.info(f"rate {rate:.1f} Hz, steps {s['steps']} (cnn {s['cnn_steps']}, gyro {s['gyro_steps']}), "
                f"{s['distance_m']:.1f} m, heading {s['heading_deg']:.0f}, turns {s['turns']}, "
                f"detector {s['detector']}, process {proc_ms:.0f} ms, "
                f"{'recording ' + rec['name'] if rec['recording'] else 'idle'}")


# ---------- HTTP ----------
def state():
    with lock:
        s = navigator.snapshot()
    rec = recorder.info()
    age = time.time() - last_sample_wall if last_sample_wall else None
    return {**s, "recording": rec["recording"], "session": rec["name"], "rows": rec["rows"],
            "elapsed_s": round(time.time() - rec["started"], 1) if rec["recording"] else None,
            "events": rec["events"], "sensor_ok": age is not None and age < STALE_S,
            "mcu": last_mcu_status, "process_ms": round(proc_ms, 1)}


class Handler(BaseHTTPRequestHandler):
    server_version = "me193_nav"

    def log_message(self, fmt, *args):      # quiet: the page polls 4x a second
        pass

    def _send(self, code, body, ctype="application/json", extra=None):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/state":
            self._send(200, state())
        elif path == "/api/sessions":
            self._send(200, recorder.list())
        elif path.startswith("/files/"):
            name = unquote(path[len("/files/"):])
            p = recorder.path_for(name)
            if p is None:
                self._send(404, {"error": "no such file"})
            else:
                ctype = "text/csv; charset=utf-8" if name.endswith(".csv") else "application/json"
                self._send(200, p.read_bytes(), ctype, {"Content-Disposition": f'attachment; filename="{name}"'})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(min(n, 4096)) or b"{}") if n else {}
        except ValueError:
            return self._send(400, {"error": "bad JSON"})
        try:
            if path == "/api/start":
                with lock:
                    navigator.start_session(None)
                    navigator.take_events()
                    name = recorder.start(navigator.snapshot())
                logger.info(f"recording {name}")
                self._send(200, {"ok": True, "session": name})
            elif path == "/api/stop":
                with lock:
                    snap = navigator.snapshot()
                name, rows = recorder.stop(snap)
                logger.info(f"stopped {name}: {rows} rows")
                self._send(200, {"ok": True, "session": name, "rows": rows})
            elif path == "/api/event":
                button = str(body.get("button", "")).upper()
                recorder.event(button)
                self._send(200, {"ok": True, "button": button})
            else:
                self._send(404, {"error": "not found"})
        except SessionError as e:
            self._send(409, {"ok": False, "error": str(e)})
        except OSError as e:
            logger.error(f"{path} failed: {e}")
            self._send(500, {"ok": False, "error": f"file error: {e}"})


def serve():
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    httpd.daemon_threads = True
    logger.info(f"phone page on http://<board IP>:{PORT}/ ; sessions in {RECORD_DIR}")
    httpd.serve_forever()


Bridge.provide("imu_sample", imu_sample)
Bridge.provide("imu_status", imu_status)
for target in (process_loop, row_loop, led_loop, serve):
    threading.Thread(target=target, daemon=True).start()

App.run(user_loop=status_loop)
