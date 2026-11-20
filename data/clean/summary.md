# Label cleaning summary

Made by `python scripts/clean_labels.py data/raw/rec_20261120-203227.csv data/raw/rec_20261120-203759.csv data/raw/rec_20261120-204001.csv data/raw/rec_20261120-204400.csv data/raw/rec_20261120-205439.csv data/raw/rec_20261120-205635.csv`.

| walk | who | s | presses L/R | matched | rotation | unmatched | turns w/o press | turns | steps | distance_m | end heading | stairs windows | stairs s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rec_20261120-203227 | ? | 63.9 | 4 | 2 | 1 | 1 | 1 | 3 | 92 | 71.07 | -267 | 1 | 2.7 |
| rec_20261120-203759 | ? | 62.7 | 4 | 2 | 1 | 1 | 1 | 3 | 95 | 75.95 | -283 | 1 | 5.6 |
| rec_20261120-204001 | ? | 60.4 | 4 | 3 | 1 | 0 | 0 | 3 | 92 | 72.99 | -268 | 1 | 14.0 |
| rec_20261120-204400 | ? | 69.0 | 4 | 4 | 0 | 0 | 0 | 4 | 97 | 76.05 | -271 | 1 | 15.2 |
| rec_20261120-205439 | ? | 62.0 | 4 | 3 | 0 | 1 | 1 | 4 | 92 | 71.79 | -226 | 1 | 1.2 |
| rec_20261120-205635 | ? | 65.6 | 4 | 4 | 0 | 0 | 0 | 4 | 98 | 76.06 | -224 | 1 | 6.9 |

## Inconsistencies

- turn counts differ: 203227 3, 203759 3, 204001 3, 204400 4, 205439 4, 205635 4
- distance_m spread: 71.07-76.06 (7% max/min)
- steps spread: 92-98 (7% max/min)
- STAIRS windows are very different lengths: 203227 2.7 s, 203759 5.6 s, 204001 14.0 s, 204400 15.2 s, 205439 1.2 s, 205635 6.9 s
