# Navigation policy (behavior cloning)

Made by `python scripts/train_policy.py`. 8 walks: rec_20261120-203227 (Ceci), rec_20261120-203759 (Ryan), rec_20261120-204001 (Ceci), rec_20261120-204400 (Ryan), rec_20261120-205439 (Ceci), rec_20261120-205635 (Ryan), rec_20261120-214933 (Ryan), rec_20261120-215247 (Ceci). Labels: {'STRAIGHT': 3371, 'LEFT': 610, 'ARRIVED': 470, 'RIGHT': 412, 'STAIRS': 253}. Leave-one-walk-out; predictions smoothed like the board (10-row majority). An action is announced if the smoothed output shows it from at most 10 m before to 1 s after it; 'm ahead' is where that announcement starts.

| model | features | acc | bal_acc | actions hit | turns hit | median m ahead (turns) | false announcements |
|---|---|---|---|---|---|---|---|
| MLP | with in_stairs, true at test (NOT available live) | 0.905 | 0.902 | 46/47 | 31/31 | 2.800 | 3 |
| MLP | with in_stairs, 0 at test (as live) | 0.740 | 0.866 | 47/47 | 31/31 | 3.300 | 11 |
| MLP | without in_stairs | 0.900 | 0.897 | 46/47 | 31/31 | 3.100 | 7 |
| KNN | with in_stairs, true at test (NOT available live) | 0.892 | 0.837 | 45/47 | 30/31 | 1.900 | 1 |
| KNN | with in_stairs, 0 at test (as live) | 0.709 | 0.806 | 47/47 | 31/31 | 2.400 | 9 |
| KNN | without in_stairs | 0.897 | 0.833 | 46/47 | 31/31 | 1.900 | 2 |

**Exported: MLP, without in_stairs** (within 0.02 of the best mean of balanced accuracy and actions hit among the variants that can run live - the board does not measure in_stairs - ties going to fewer features, then fewer false announcements).

## Exported model, per walk

| walk | who | acc | bal_acc | false |
|---|---|---|---|---|
| 203227 | Ceci | 0.893 | 0.882 | 1 |
| 203759 | Ryan | 0.911 | 0.903 | 2 |
| 204001 | Ceci | 0.918 | 0.977 | 1 |
| 204400 | Ryan | 0.918 | 0.937 | 1 |
| 205439 | Ceci | 0.907 | 0.939 | 1 |
| 205635 | Ryan | 0.906 | 0.760 | 0 |
| 214933 | Ryan | 0.894 | 0.903 | 0 |
| 215247 | Ceci | 0.855 | 0.875 | 1 |

### MLP (without in_stairs), per action

| walk | t_s | action | m ahead | s from action |
|---|---|---|---|---|
| 203227 | 6.420 | RIGHT | 2.300 | -6.400 |
| 203227 | 28.000 | LEFT | 1.600 | -1.200 |
| 203227 | 33.600 | STAIRS | 2.800 | -3.900 |
| 203227 | 49.880 | LEFT | 4.200 | -4.500 |
| 203227 | 51.080 | LEFT | 5.400 | -5.700 |
| 203227 | 60.260 | ARRIVED | 0.900 | -0.300 |
| 203759 | 4.020 | RIGHT | 0.000 | -4.000 |
| 203759 | 25.700 | LEFT | 0.700 | -1.100 |
| 203759 | 29.400 | STAIRS | 2.800 | -2.500 |
| 203759 | 47.080 | LEFT | 3.400 | -3.300 |
| 203759 | 48.580 | LEFT | 4.700 | -4.800 |
| 203759 | 59.180 | ARRIVED | 2.300 | -1.900 |
| 204001 | 3.900 | RIGHT | 1.400 | -3.900 |
| 204001 | 26.180 | LEFT | 3.300 | -2.200 |
| 204001 | 29.360 | STAIRS | 3.100 | -2.700 |
| 204001 | 45.860 | LEFT | 3.200 | -2.300 |
| 204001 | 47.360 | LEFT | 4.600 | -3.800 |
| 204001 | 57.140 | ARRIVED | 3.200 | -2.400 |
| 204400 | 5.820 | RIGHT | 1.900 | -5.800 |
| 204400 | 28.100 | LEFT | 2.700 | -2.300 |
| 204400 | 31.900 | STAIRS | 2.600 | -3.300 |
| 204400 | 51.780 | LEFT | 2.100 | -6.000 |
| 204400 | 54.780 | LEFT | 4.600 | -9.000 |
| 204400 | 64.760 | ARRIVED | 4.700 | -3.000 |
| 205439 | 4.680 | RIGHT | 0.000 | -4.700 |
| 205439 | 26.760 | LEFT | 3.200 | -2.200 |
| 205439 | 31.560 | STAIRS | 3.200 | -4.000 |
| 205439 | 48.140 | LEFT | 3.500 | -3.700 |
| 205439 | 51.220 | LEFT | 5.500 | -6.800 |
| 205439 | 58.800 | ARRIVED | 0.900 | -1.000 |
| 205635 | 4.920 | RIGHT | 0.600 | -4.900 |
| 205635 | 26.980 | LEFT | 1.600 | -0.700 |
| 205635 | 31.480 | STAIRS | MISSED | nan |
| 205635 | 48.360 | LEFT | 4.100 | -5.500 |
| 205635 | 51.660 | LEFT | 6.700 | -8.800 |
| 205635 | 60.540 | ARRIVED | 2.300 | -2.100 |
| 214933 | 7.780 | RIGHT | 2.500 | -7.800 |
| 214933 | 28.960 | LEFT | 2.300 | -2.000 |
| 214933 | 33.460 | STAIRS | 4.200 | -4.000 |
| 214933 | 52.440 | LEFT | 5.500 | -5.200 |
| 214933 | 54.840 | LEFT | 7.600 | -7.600 |
| 214933 | 64.070 | ARRIVED | 0.700 | -0.800 |
| 215247 | 3.140 | RIGHT | 0.000 | -3.100 |
| 215247 | 25.120 | LEFT | 3.100 | -2.800 |
| 215247 | 30.520 | STAIRS | 4.100 | -4.300 |
| 215247 | 45.200 | LEFT | 0.800 | -1.200 |
| 215247 | 56.450 | ARRIVED | 3.800 | -2.700 |

### KNN (without in_stairs), per action

| walk | t_s | action | m ahead | s from action |
|---|---|---|---|---|
| 203227 | 6.420 | RIGHT | 2.300 | -6.400 |
| 203227 | 28.000 | LEFT | 0.000 | -0.100 |
| 203227 | 33.600 | STAIRS | 2.800 | -3.700 |
| 203227 | 49.880 | LEFT | 0.600 | -0.900 |
| 203227 | 51.080 | LEFT | 1.800 | -2.100 |
| 203227 | 60.260 | ARRIVED | 0.000 | -0.200 |
| 203759 | 4.020 | RIGHT | 0.000 | -4.000 |
| 203759 | 25.700 | LEFT | 0.700 | -1.100 |
| 203759 | 29.400 | STAIRS | 2.100 | -1.700 |
| 203759 | 47.080 | LEFT | 1.900 | -2.200 |
| 203759 | 48.580 | LEFT | 3.300 | -3.700 |
| 203759 | 59.180 | ARRIVED | 2.300 | -1.800 |
| 204001 | 3.900 | RIGHT | 1.400 | -3.900 |
| 204001 | 26.180 | LEFT | 2.400 | -1.600 |
| 204001 | 29.360 | STAIRS | 2.200 | -2.400 |
| 204001 | 45.860 | LEFT | 1.600 | -1.000 |
| 204001 | 47.360 | LEFT | 3.000 | -2.500 |
| 204001 | 57.140 | ARRIVED | 2.300 | -1.800 |
| 204400 | 5.820 | RIGHT | 1.900 | -5.800 |
| 204400 | 28.100 | LEFT | 2.700 | -2.300 |
| 204400 | 31.900 | STAIRS | 0.700 | -0.800 |
| 204400 | 51.780 | LEFT | 1.400 | -5.600 |
| 204400 | 54.780 | LEFT | 3.800 | -8.600 |
| 204400 | 64.760 | ARRIVED | 3.100 | -2.400 |
| 205439 | 4.680 | RIGHT | 0.000 | -4.700 |
| 205439 | 26.760 | LEFT | 3.200 | -2.200 |
| 205439 | 31.560 | STAIRS | 1.900 | -1.400 |
| 205439 | 48.140 | LEFT | 3.500 | -3.400 |
| 205439 | 51.220 | LEFT | 5.500 | -6.500 |
| 205439 | 58.800 | ARRIVED | 1.700 | -1.500 |
| 205635 | 4.920 | RIGHT | 0.600 | -4.900 |
| 205635 | 26.980 | LEFT | 0.900 | -0.500 |
| 205635 | 31.480 | STAIRS | MISSED | nan |
| 205635 | 48.360 | LEFT | 2.600 | -4.900 |
| 205635 | 51.660 | LEFT | 1.400 | -1.200 |
| 205635 | 60.540 | ARRIVED | 2.300 | -1.700 |
| 214933 | 7.780 | RIGHT | 2.500 | -7.800 |
| 214933 | 28.960 | LEFT | 2.300 | -1.900 |
| 214933 | 33.460 | STAIRS | 4.200 | -4.000 |
| 214933 | 52.440 | LEFT | 5.500 | -4.900 |
| 214933 | 54.840 | LEFT | 7.600 | -7.300 |
| 214933 | 64.070 | ARRIVED | 0.000 | -0.300 |
| 215247 | 3.140 | RIGHT | 0.000 | -3.100 |
| 215247 | 25.120 | LEFT | 3.100 | -2.800 |
| 215247 | 30.520 | STAIRS | 3.400 | -4.000 |
| 215247 | 45.200 | LEFT | 0.800 | -0.900 |
| 215247 | 56.450 | ARRIVED | 2.900 | -2.200 |

## Before and after DAgger (MLP, without in_stairs)

Leave-one-walk-out on all 8 walks with the corrected labels as truth. A is how the deployed pre-DAgger model was trained; B adds only the STAIRS label fixes (data/label_fixes.json); C adds the DAgger walks. 'live' is what the deployed model actually announced on the DAgger walks.

| model | walks | acc | bal_acc | actions hit | turns hit | STAIRS hit | STAIRS m ahead (median) | STAIRS m ahead (min-max) | ARRIVED hit | false announcements |
|---|---|---|---|---|---|---|---|---|---|---|
| A: pre-DAgger (demos, original labels) | 8 | 0.883 | 0.859 | 46/47 | 31/31 | 7/8 | 3.200 | 1.6-3.5 | 8/8 | 8 |
| B: demos, fixed STAIRS labels | 8 | 0.912 | 0.906 | 46/47 | 31/31 | 7/8 | 3.200 | 2.6-4.2 | 8/8 | 3 |
| C: B + DAgger walks (exported) | 8 | 0.900 | 0.897 | 46/47 | 31/31 | 7/8 | 3.100 | 2.6-4.2 | 8/8 | 7 |
| deployed pre-DAgger model, live on the DAgger walks (what it actually said) | 2 | 0.868 | 0.867 | 11/11 | 7/7 | 2/2 | 3.450 | 3.4-3.5 | 2/2 | 2 |

### STAIRS, meters ahead per walk

| walk | A | B | C | live |
|---|---|---|---|---|
| 203227 | 2.8 | 3.6 | 2.8 | n/a (no live run) |
| 203759 | 3.5 | 2.8 | 2.8 | n/a (no live run) |
| 204001 | 1.6 | 3.1 | 3.1 | n/a (no live run) |
| 204400 | 2.1 | 2.6 | 2.6 | n/a (no live run) |
| 205439 | 3.2 | 3.2 | 3.2 | n/a (no live run) |
| 205635 | MISSED | MISSED | MISSED | n/a (no live run) |
| 214933 | 3.5 | 4.2 | 4.2 | 3.5 |
| 215247 | 3.4 | 4.1 | 4.1 | 3.4 |
