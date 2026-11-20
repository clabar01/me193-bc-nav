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
../me193-step-counter/.venv/bin/python scripts/calibrate_weinberg.py   # K -> models/weinberg.json
../me193-step-counter/.venv/bin/python scripts/calibrate_heading.py    # walking yaw drift -> nav.py
../me193-step-counter/.venv/bin/python scripts/replay_nav.py           # old trials through the board code
```

| | Result |
| --- | --- |
| Step detector, IMU only | Session C (held out): 1.67 steps/trial vs 3.00 for the old IMU + ultrasonic model. Worse in turns (bench_05: 6 of 20). [results](models/step_detector/imu/results.md) |
| Weinberg K = 0.7419 | 1.9% leave-one-out on B, 5.0% on C when fitted on B. [results](models/weinberg_results.md) |
| Heading | Walking drift +1.73 deg/s removed; C straight walks: 47 deg to 13 deg mean net error. [results](models/heading_results.md) |
| Board code on old trials | live CNN = offline on 28/28; bench_05 fused 17 of 20; distance 3.0% mean error on 14 trials. [results](models/nav_replay.md) |

## Board app `board/me193_nav`

```bash
board/me193_nav/deploy_board.sh --sketch      # then power-cycle the board
# phone on Tufts_Robot: http://10.247.137.47:7000/
python board/me193_nav/pull_recordings.py     # finished sessions -> data/raw/
python board/me193_nav/test_nav.py            # fusion / heading / turn / log checks
python board/me193_nav/run_board_app.py --replay ../me193-step-counter/data/raw/bench_05.csv --loop
                                              # the whole app on the laptop: http://localhost:7000/
```

- **Heading**: gyro projected onto gravity (complementary filter: gyro-rotated,
  pulled toward the accelerometer, tau 3 s), integrated; + = right, 0 at Start.
- **Turns**: >= 60 deg within 2 s; one per rotation (a U-turn counts once).
- **Steps**: the CNN and the gyro stride detector both run. The fused count
  takes the gyro detector's steps while turning (1 s mean yaw rate >= 30 deg/s
  on, < 15 deg/s for 0.5 s off) and the CNN's otherwise, judged at each step's
  own sample; within 0.3 s of a switch both are accepted and duplicates within
  0.26 s dropped. Steps appear ~1.5 s after they happen.
- **Distance**: Weinberg, K * (|a| range over the step)^0.25.
- **Phone page (Record mode)**: Start/Stop, LEFT / RIGHT / STAIRS / ARRIVED,
  live distance and heading, downloads. No login: anyone on the network can use it.

Each session `rec_<date>-<time>` is `.csv` (10 Hz: `unix_time, t_s,
distance_m, since_turn_m, heading_deg, turns, steps, cnn_steps, gyro_steps,
detector, event`), `_imu.csv` (50 Hz raw IMU) and `.json` (metadata), in
`~/ArduinoApps/me193_nav/recordings/` on the board. Columns are described in
`board/me193_nav/python/session.py`.

`board/me193_steps` is the old step-counter app, kept for reference.
