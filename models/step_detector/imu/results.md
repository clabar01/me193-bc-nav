# Step detector, IMU only (6 channels)

Made by `python scripts/train_detector_imu.py`. Training code and data: `../me193-step-counter`, same split and settings as its IMU-only ablation, seeds 0/1/2; the exported seed was chosen on validation. All counts use the board's counting rule (`counting.count_steps`). Session C is the held-out test set; bench_05 is a small room with turns (never trained on).

| set | method | trials | steps counted | MAE (steps/trial) | bias | total |
|---|---|---|---|---|---|---|
| val | old CNN, IMU + ultrasonic (committed) | 7 | 180 | 0.29 | +0.29 | +1.1% |
| val | new CNN, IMU only (seed 2, exported) | 7 | 180 | 0.71 | +0.71 | +2.8% |
| val | gyro stride detector | 7 | 180 | 0.29 | +0.29 | +1.1% |
| val | new CNN, IMU only, seeds 0/1/2 | 7 | 180 | 0.86 / 0.86 / 0.71 (mean 0.81) |  |  |
| C | old CNN, IMU + ultrasonic (committed) | 6 | 173 | 3.00 | -2.67 | -9.2% |
| C | new CNN, IMU only (seed 2, exported) | 6 | 173 | 1.67 | -1.00 | -3.5% |
| C | gyro stride detector | 6 | 173 | 1.17 | +0.17 | +0.6% |
| C | new CNN, IMU only, seeds 0/1/2 | 6 | 173 | 1.50 / 1.17 / 1.67 (mean 1.44) |  |  |

## Per trial

| set | trial | speed | counted | old CNN | new IMU-only CNN | IMU seed 0 | IMU seed 1 | IMU seed 2 | gyro |
|---|---|---|---|---|---|---|---|---|---|
| val | A_04 | fast | 30 | 30 | 31 | 31 | 31 | 31 | 30 |
| val | A_09 | normal | 30 | 30 | 31 | 31 | 31 | 31 | 30 |
| val | A_02 | slow | 30 | 30 | 31 | 31 | 31 | 31 | 30 |
| val | A_11 | normal | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| val | B_08 | fast | 12 | 13 | 13 | 13 | 13 | 13 | 12 |
| val | B_03 | normal | 37 | 37 | 38 | 38 | 38 | 38 | 38 |
| val | B_06 | slow | 41 | 42 | 41 | 42 | 42 | 41 | 42 |
| C | C_01 | normal | 30 | 29 | 29 | 29 | 29 | 29 | 28 |
| C | C_02 | slow | 30 | 29 | 31 | 31 | 31 | 31 | 32 |
| C | C_03 | fast | 28 | 25 | 26 | 29 | 28 | 26 | 28 |
| C | C_04 | slow | 29 | 24 | 29 | 28 | 29 | 29 | 28 |
| C | C_05 | normal | 29 | 30 | 30 | 30 | 30 | 30 | 30 |
| C | C_06 | fast | 27 | 20 | 22 | 23 | 23 | 22 | 28 |
| bench_05 | bench_05 | fast | 20 | 15 | 6 | 0 | 12 | 6 | 22 |
