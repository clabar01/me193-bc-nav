# Label cleaning summary

Made by `python scripts/clean_labels.py data/raw/rec_20261120-203227.csv data/raw/rec_20261120-203759.csv data/raw/rec_20261120-204001.csv data/raw/rec_20261120-204400.csv data/raw/rec_20261120-205439.csv data/raw/rec_20261120-205635.csv data/raw/rec_20261120-214933.csv data/raw/rec_20261120-215247.csv`.

| walk | who | s | presses L/R | matched | rotation | unmatched | turns w/o press | turns | steps | distance_m | end heading | stairs windows | stairs s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rec_20261120-203227 | ? | 63.9 | 4 | 2 | 1 | 1 | 1 | 3 | 92 | 71.07 | -267 | 1 | 12.4 |
| rec_20261120-203759 | ? | 62.7 | 4 | 2 | 1 | 1 | 1 | 3 | 95 | 75.95 | -283 | 1 | 14.5 |
| rec_20261120-204001 | ? | 60.4 | 4 | 3 | 1 | 0 | 0 | 3 | 92 | 72.99 | -268 | 1 | 14.0 |
| rec_20261120-204400 | ? | 69.0 | 4 | 4 | 0 | 0 | 0 | 4 | 97 | 76.05 | -271 | 1 | 15.2 |
| rec_20261120-205439 | ? | 62.0 | 4 | 3 | 0 | 1 | 1 | 4 | 92 | 71.79 | -226 | 1 | 12.9 |
| rec_20261120-205635 | ? | 65.6 | 4 | 4 | 0 | 0 | 0 | 4 | 98 | 76.06 | -224 | 1 | 19.7 |
| rec_20261120-214933 | Ryan | 67.5 | 0 | 0 | 0 | 0 | 4 | 4 | 101 | 76.07 | -273 | 1 | 15.8 |
| rec_20261120-215247 | Ceci | 58.2 | 1 | 1 | 0 | 0 | 2 | 3 | 91 | 73.52 | -257 | 1 | 13.0 |

## Inconsistencies

- turn counts differ: 203227 3, 203759 3, 204001 3, 204400 4, 205439 4, 205635 4, 214933 4, 215247 3
- distance_m spread: 71.07-76.07 (7% max/min)
- steps spread: 91-101 (11% max/min)
- number of LEFT/RIGHT presses differs: 203227 4, 203759 4, 204001 4, 204400 4, 205439 4, 205635 4, 214933 0, 215247 1
