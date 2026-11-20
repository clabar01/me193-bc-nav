# Navigation policy (behavior cloning)

Made by `python scripts/train_policy.py`. 6 walks: rec_20261120-203227 (Ceci), rec_20261120-203759 (Ryan), rec_20261120-204001 (Ceci), rec_20261120-204400 (Ryan), rec_20261120-205439 (Ceci), rec_20261120-205635 (Ryan). Labels: {'STRAIGHT': 2557, 'LEFT': 484, 'ARRIVED': 367, 'RIGHT': 302, 'STAIRS': 144}. Leave-one-walk-out; predictions smoothed like the board (10-row majority). An action is announced if the smoothed output shows it from at most 10 m before to 1 s after it; 'm ahead' is where that announcement starts.

| model | features | acc | bal_acc | actions hit | turns hit | median m ahead (turns) | false announcements |
|---|---|---|---|---|---|---|---|
| MLP | with in_stairs, true at test (NOT available live) | 0.876 | 0.820 | 34/36 | 24/24 | 2.450 | 5 |
| MLP | with in_stairs, 0 at test (as live) | 0.833 | 0.807 | 35/36 | 24/24 | 2.450 | 9 |
| MLP | without in_stairs | 0.857 | 0.786 | 35/36 | 24/24 | 2.650 | 8 |
| KNN | with in_stairs, true at test (NOT available live) | 0.855 | 0.757 | 33/36 | 23/24 | 2.400 | 7 |
| KNN | with in_stairs, 0 at test (as live) | 0.830 | 0.778 | 35/36 | 24/24 | 2.950 | 10 |
| KNN | without in_stairs | 0.876 | 0.759 | 33/36 | 24/24 | 2.500 | 10 |

**Exported: MLP, without in_stairs** (within 0.02 of the best mean of balanced accuracy and actions hit among the variants that can run live - the board does not measure in_stairs - ties going to fewer features, then fewer false announcements).

## Exported model, per walk

| walk | who | acc | bal_acc | false |
|---|---|---|---|---|
| 203227 | Ceci | 0.829 | 0.723 | 2 |
| 203759 | Ryan | 0.840 | 0.724 | 2 |
| 204001 | Ceci | 0.865 | 0.822 | 1 |
| 204400 | Ryan | 0.887 | 0.812 | 1 |
| 205439 | Ceci | 0.804 | 0.736 | 2 |
| 205635 | Ryan | 0.920 | 0.897 | 0 |

### MLP (without in_stairs), per action

| walk | t_s | action | m ahead | s from action |
|---|---|---|---|---|
| 203227 | 6.420 | RIGHT | 2.300 | -6.400 |
| 203227 | 28.000 | LEFT | 1.600 | -1.200 |
| 203227 | 43.280 | STAIRS | MISSED | nan |
| 203227 | 49.880 | LEFT | 3.400 | -4.000 |
| 203227 | 51.080 | LEFT | 4.600 | -5.200 |
| 203227 | 60.260 | ARRIVED | 1.700 | -1.400 |
| 203759 | 4.020 | RIGHT | 0.000 | -4.000 |
| 203759 | 25.700 | LEFT | 1.600 | -1.300 |
| 203759 | 38.300 | STAIRS | -0.800 | 0.400 |
| 203759 | 47.080 | LEFT | 3.400 | -3.600 |
| 203759 | 48.580 | LEFT | 4.700 | -5.100 |
| 203759 | 59.180 | ARRIVED | 2.300 | -1.800 |
| 204001 | 3.900 | RIGHT | 1.400 | -3.900 |
| 204001 | 26.180 | LEFT | 3.300 | -2.100 |
| 204001 | 29.360 | STAIRS | 1.600 | -1.600 |
| 204001 | 45.860 | LEFT | 3.200 | -2.400 |
| 204001 | 47.360 | LEFT | 4.600 | -3.900 |
| 204001 | 57.140 | ARRIVED | 2.300 | -2.000 |
| 204400 | 5.820 | RIGHT | 1.900 | -5.800 |
| 204400 | 28.100 | LEFT | 2.700 | -2.300 |
| 204400 | 31.900 | STAIRS | 2.100 | -2.100 |
| 204400 | 51.780 | LEFT | 2.100 | -6.000 |
| 204400 | 54.780 | LEFT | 4.600 | -9.000 |
| 204400 | 64.760 | ARRIVED | 4.700 | -3.000 |
| 205439 | 4.680 | RIGHT | 0.000 | -4.700 |
| 205439 | 26.760 | LEFT | 3.200 | -2.200 |
| 205439 | 43.260 | STAIRS | 0.800 | -0.600 |
| 205439 | 48.140 | LEFT | 2.000 | -1.900 |
| 205439 | 51.220 | LEFT | 3.900 | -5.000 |
| 205439 | 58.800 | ARRIVED | 0.900 | -1.000 |
| 205635 | 4.920 | RIGHT | 0.600 | -4.900 |
| 205635 | 26.980 | LEFT | 1.600 | -0.700 |
| 205635 | 44.260 | STAIRS | 5.300 | -3.600 |
| 205635 | 48.360 | LEFT | 2.600 | -4.600 |
| 205635 | 51.660 | LEFT | 5.100 | -7.900 |
| 205635 | 60.540 | ARRIVED | 2.300 | -2.100 |

### KNN (without in_stairs), per action

| walk | t_s | action | m ahead | s from action |
|---|---|---|---|---|
| 203227 | 6.420 | RIGHT | 2.300 | -6.400 |
| 203227 | 28.000 | LEFT | 0.000 | -0.100 |
| 203227 | 43.280 | STAIRS | MISSED | nan |
| 203227 | 49.880 | LEFT | 1.800 | -3.100 |
| 203227 | 51.080 | LEFT | 3.000 | -4.300 |
| 203227 | 60.260 | ARRIVED | 0.900 | -0.300 |
| 203759 | 4.020 | RIGHT | 0.000 | -4.000 |
| 203759 | 25.700 | LEFT | 0.700 | -1.100 |
| 203759 | 38.300 | STAIRS | MISSED | nan |
| 203759 | 47.080 | LEFT | 2.700 | -3.100 |
| 203759 | 48.580 | LEFT | 4.000 | -4.600 |
| 203759 | 59.180 | ARRIVED | 2.300 | -1.800 |
| 204001 | 3.900 | RIGHT | 1.400 | -3.900 |
| 204001 | 26.180 | LEFT | 3.300 | -2.200 |
| 204001 | 29.360 | STAIRS | 1.600 | -1.600 |
| 204001 | 45.860 | LEFT | 2.400 | -2.100 |
| 204001 | 47.360 | LEFT | 3.800 | -3.600 |
| 204001 | 57.140 | ARRIVED | 2.300 | -1.800 |
| 204400 | 5.820 | RIGHT | 1.900 | -5.800 |
| 204400 | 28.100 | LEFT | 2.700 | -2.300 |
| 204400 | 31.900 | STAIRS | MISSED | nan |
| 204400 | 51.780 | LEFT | 1.400 | -5.600 |
| 204400 | 54.780 | LEFT | 3.800 | -8.600 |
| 204400 | 64.760 | ARRIVED | 6.200 | -4.400 |
| 205439 | 4.680 | RIGHT | 0.000 | -4.700 |
| 205439 | 26.760 | LEFT | 3.200 | -2.200 |
| 205439 | 43.260 | STAIRS | 0.000 | 0.100 |
| 205439 | 48.140 | LEFT | 2.800 | -3.200 |
| 205439 | 51.220 | LEFT | 4.700 | -6.300 |
| 205439 | 58.800 | ARRIVED | 1.700 | -1.500 |
| 205635 | 4.920 | RIGHT | 0.600 | -4.900 |
| 205635 | 26.980 | LEFT | 0.900 | -0.500 |
| 205635 | 44.260 | STAIRS | 6.200 | -3.900 |
| 205635 | 48.360 | LEFT | 2.600 | -4.900 |
| 205635 | 51.660 | LEFT | 5.100 | -8.200 |
| 205635 | 60.540 | ARRIVED | 2.300 | -1.700 |
