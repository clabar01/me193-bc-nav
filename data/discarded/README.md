# Discarded sessions

Kept for the record, never used for training or evaluation. Discarded on
2026-11-20 by the user ("we messed those up").

| Session | Note |
| --- | --- |
| rec_20261120-204631 | Demo route attempt. IMU stream stopped after 2.4 s (123 samples) while the 10 Hz log kept running (~154 s, frozen values, 0 steps); only RIGHT and LEFT pressed; never stopped, so there is no .json (copied straight from the board). |
| rec_20261120-205204 | Demo route, 91.5 s (~30 s longer than the other walks); three STAIRS presses in one 0.1 s row at 67 s. |

`board/me193_nav/pull_recordings.py` skips any session that has files here.
