# Weinberg step length on the gyro detector's steps

Made by `python scripts/calibrate_weinberg.py --steps gyro`. Board K = 0.7312 (all 14 hallway distance trials); K on B only = 0.7364.

| eval | trials | mean_abs_err | bias |
|---|---|---|---|
| B, leave-one-out | 8 | 3.4 | -0.1 |
| C, K fitted on B | 6 | 4.3 | 1.4 |

| eval | trial | speed | distance_m | counted | gyro steps | K | predicted_m | err% | err% if count were exact |
|---|---|---|---|---|---|---|---|---|---|
| B, leave-one-out | B_01 | normal | 19.0 | 24 | 24 | 0.7349 | 18.7 | -1.6 | -1.6 |
| B, leave-one-out | B_02 | normal | 19.0 | 24 | 24 | 0.7361 | 18.94 | -0.3 | -0.3 |
| B, leave-one-out | B_03 | normal | 30.0 | 37 | 38 | 0.7355 | 29.7 | -1.0 | -3.6 |
| B, leave-one-out | B_04 | normal | 19.0 | 25 | 26 | 0.7411 | 19.94 | 5.0 | 0.9 |
| B, leave-one-out | B_05 | slow | 19.0 | 25 | 28 | 0.7417 | 20.05 | 5.6 | -5.8 |
| B, leave-one-out | B_06 | slow | 30.0 | 41 | 42 | 0.739 | 30.83 | 2.8 | 0.3 |
| B, leave-one-out | B_07 | fast | 19.0 | 23 | 22 | 0.7321 | 18.07 | -4.9 | -0.6 |
| B, leave-one-out | B_08 | fast | 11.0 | 12 | 12 | 0.7308 | 10.3 | -6.4 | -6.4 |
| C, K fitted on B | C_01 | normal | 23.0 | 30 | 28 | 0.7364 | 22.37 | -2.7 | 4.2 |
| C, K fitted on B | C_02 | slow | 23.0 | 30 | 32 | 0.7364 | 23.7 | 3.0 | -3.4 |
| C, K fitted on B | C_03 | fast | 23.0 | 28 | 28 | 0.7364 | 23.87 | 3.8 | 3.8 |
| C, K fitted on B | C_04 | slow | 23.0 | 29 | 28 | 0.7364 | 21.66 | -5.8 | -2.5 |
| C, K fitted on B | C_05 | normal | 23.0 | 29 | 30 | 0.7364 | 23.97 | 4.2 | 0.7 |
| C, K fitted on B | C_06 | fast | 23.0 | 27 | 28 | 0.7364 | 24.42 | 6.2 | 2.4 |
