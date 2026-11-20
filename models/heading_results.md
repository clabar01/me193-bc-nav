# Heading drift while walking

Made by `python scripts/calibrate_heading.py`: the old straight hallway trials replayed through `board/me193_nav/python/nav.py` (complementary-filter gravity, tau 3.0 s).

Walking drift fitted on A/B (median of 18 trials): **+1.73 deg/s** (+ = right).

Session C (held out), mean |net heading| over 6 straight walks: 46.8 deg without the correction, 12.7 deg with it.

| trial | session | walk_s | net_deg (no correction) | drift_dps | net_deg (corrected) |
|---|---|---|---|---|---|
| A_01 | A | 19.4 | 37.8 | 1.95 | 4.3 |
| A_02 | A | 26.6 | 61.3 | 2.3 | 15.2 |
| A_03 | A | 15.1 | 18.6 | 1.23 | -7.6 |
| A_04 | A | 17.5 | 12.5 | 0.72 | -17.8 |
| A_05 | A | 19.3 | 30.0 | 1.55 | -3.5 |
| A_06 | A | 12.5 | 24.9 | 1.99 | 3.3 |
| A_07 | A | 20.6 | 38.1 | 1.85 | 2.4 |
| A_08 | A | 18.8 | -75.0 | -4.0 | -107.5 |
| A_09 | A | 20.5 | 105.4 | 5.15 | 70.0 |
| A_10 | A | 20.8 | -229.6 | -11.02 | -265.7 |
| B_01 | B | 17.0 | 41.5 | 2.44 | 12.0 |
| B_02 | B | 16.4 | 30.3 | 1.84 | 1.9 |
| B_03 | B | 26.2 | 43.8 | 1.67 | -1.5 |
| B_04 | B | 16.7 | 29.9 | 1.79 | 1.0 |
| B_05 | B | 20.5 | 16.0 | 0.78 | -19.4 |
| B_06 | B | 31.9 | 58.7 | 1.84 | 3.5 |
| B_07 | B | 14.3 | 13.4 | 0.94 | -11.4 |
| B_08 | B | 8.4 | 2.4 | 0.28 | -12.2 |
| C_01 | C | 27.9 | 43.1 | 1.55 | -5.1 |
| C_02 | C | 25.4 | 55.0 | 2.16 | 11.0 |
| C_03 | C | 16.8 | 48.4 | 2.87 | 19.3 |
| C_04 | C | 22.1 | 50.8 | 2.3 | 12.6 |
| C_05 | C | 17.5 | 42.4 | 2.42 | 12.1 |
| C_06 | C | 14.7 | 41.3 | 2.82 | 15.9 |
