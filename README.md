# ME193 Behavior Cloning Navigation

Dead reckoning from one thigh IMU, logged with navigation labels from a phone,
as data for behavior cloning. Built on the step counter from
`../me193-step-counter` (training code and data).

## Hardware

| Part | Notes |
| --- | --- |
| Arduino Uno Q "AirFour" | 10.247.137.47 on Tufts_Robot. Linux side runs Python, MCU side runs the sketch (Arduino_RouterBridge). |
| MPU-6050 | I2C 0x68 on Wire, low on the thigh just above the knee. Axis remap `BODY_R` in the sketch: x up the thigh, y right, z forward. |

No ultrasonic sensor any more (the HC-SR04 of `me193_steps` is gone).

## Models

```bash
../me193-step-counter/.venv/bin/python scripts/train_detector_imu.py   # 6-channel CNN -> models/step_detector_imu.npz
../me193-step-counter/.venv/bin/python scripts/calibrate_weinberg.py   # K for the gyro steps -> models/weinberg.json
../me193-step-counter/.venv/bin/python scripts/calibrate_heading.py    # walking yaw drift -> nav.py
../me193-step-counter/.venv/bin/python scripts/replay_nav.py           # old trials through the board code
```

| | Result |
| --- | --- |
| Step detector, IMU only | Session C (held out): 1.67 steps/trial vs 3.00 for the old IMU + ultrasonic model. Worse in turns (bench_05: 6 of 20). [results](models/step_detector/imu/results.md) |
| Weinberg K = 0.7312 (gyro steps, used on the board) | 3.4% leave-one-out on B, 4.3% on C when fitted on B. [results](models/weinberg_gyro_results.md); on the CNN's steps K = 0.7419 ([results](models/weinberg_cnn_results.md)) |
| Heading | Walking drift +1.73 deg/s removed; C straight walks: 47 deg to 13 deg mean net error. [results](models/heading_results.md) |
| Navigation policy (MLP) | 6 demo walks (3 Ceci, 3 Ryan, start-Nolop), leave-one-walk-out: 24/24 turns announced, median 2.7 m ahead; 35/36 actions; 8 false announcements. [results](models/policy_results.md) |
| Board code on old trials | live CNN = offline on 28/28; gyro steps: 0.75 steps/trial on straight walks, bench_05 22 of 20; distance 3.5% mean error on 14 trials. [results](models/nav_replay.md) |

## Board app `board/me193_nav`

```bash
board/me193_nav/deploy_board.sh --sketch      # then power-cycle the board
# phone on Tufts_Robot: http://10.247.137.47:7000/
python board/me193_nav/pull_recordings.py     # finished sessions -> data/raw/
python scripts/plot_session.py                # newest session: heading, step counts, events + totals
python scripts/clean_labels.py data/raw/rec_<...>.csv ...   # snap presses to turns -> data/clean/
python scripts/fit_person_k.py                # per-person K (data/people.json) -> python/people.json
../me193-step-counter/.venv/bin/python scripts/train_policy.py   # BC policy on data/clean/ -> models/policy.npz
python board/me193_nav/test_nav.py            # fusion / heading / turn / log checks
python board/me193_nav/run_board_app.py --replay ../me193-step-counter/data/raw/bench_05.csv --loop
                                              # the whole app on the laptop: http://localhost:7000/
```

- **Heading**: gyro projected onto gravity (complementary filter: gyro-rotated,
  pulled toward the accelerometer, tau 3 s), integrated; + = right, 0 at Start.
- **Turns**: >= 60 deg within 2 s; one per rotation (a U-turn counts once).
- **Steps**: the gyro stride detector's (`STEP_SOURCE = "gyro"` in `nav.py`).
  The CNN still runs and is logged as `cnn_steps`: on the first room walk on
  the board (rec_20261120-183930) it found 0 steps to the gyro detector's 80.
  `STEP_SOURCE = "fused"` restores the turn-based switch (gyro while turning,
  CNN otherwise, deduped at the switch). Steps appear ~1.5 s after they happen.
- **Distance**: Weinberg, K * (|a| range over the step)^0.25, K = 0.7312.
- **Phone page**: pick the mode and who's walking (per-person K from
  `python/people.json`), Start/Stop, live distance and heading, downloads.
  No login: anyone on the network can use it.
  - **Record**: LEFT / RIGHT / STAIRS / ARRIVED presses are the labels (a demo).
  - **Navigate**: the policy runs at 10 Hz on the board (1 s majority); the page
    shows a big arrow and "in ~X m" and speaks each change (tapping Start
    unlocks speech on iPhone). No labels.
  - **DAgger**: Navigate plus labels, including STRAIGHT ("ignore that"); clean
    the session and it trains like a demo.

Each session `rec_<date>-<time>` is `.csv` (10 Hz: `unix_time, t_s,
distance_m, since_turn_m, heading_deg, turns, steps, cnn_steps, gyro_steps,
detector, event`), `_imu.csv` (50 Hz raw IMU) and `.json` (metadata), in
`~/ArduinoApps/me193_nav/recordings/` on the board. Columns are described in
`board/me193_nav/python/session.py`.

`board/me193_steps` is the old step-counter app, kept for reference.
