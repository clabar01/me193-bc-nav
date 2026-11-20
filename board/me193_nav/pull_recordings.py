"""Copy finished me193_nav sessions off the board into data/raw/.

A session is finished when its <name>.json exists (written at Stop), so one
still recording is never copied. Each session is three files: <name>.csv
(10 Hz log), <name>_imu.csv (50 Hz IMU) and <name>.json. Sessions with files
in data/discarded/ are skipped. Files already in data/raw/ are never
overwritten; a same-named file with a different size is reported instead.

    python board/me193_nav/pull_recordings.py
    BOARD=arduino@<ip> python board/me193_nav/pull_recordings.py
    python board/me193_nav/pull_recordings.py --dry-run
    python board/me193_nav/pull_recordings.py --src /some/folder --dest /tmp/pulled   # local copy (testing)
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REMOTE_DIR = "ArduinoApps/me193_nav/recordings"   # relative to the board user's home


def remote_listing(board, remote_dir):
    cmd = f"cd {shlex.quote(remote_dir)} 2>/dev/null && find . -maxdepth 1 -type f -printf '%s %f\\n'"
    out = subprocess.run(["ssh", "-o", "ConnectTimeout=10", board, cmd], capture_output=True, text=True)
    if out.returncode not in (0, 1):
        sys.exit(f"ssh {board} failed: {out.stderr.strip() or out.returncode}")
    files = {}
    for line in out.stdout.splitlines():
        size, _, name = line.partition(" ")
        if name:
            files[name] = int(size)
    return files


def main():
    ap = argparse.ArgumentParser(description="Copy finished sessions from the board into data/raw/.")
    ap.add_argument("--board", default=os.environ.get("BOARD", "arduino@10.247.137.47"))
    ap.add_argument("--remote-dir", default=REMOTE_DIR)
    ap.add_argument("--src", type=Path, help="copy from this local folder instead of the board")
    ap.add_argument("--dest", type=Path, default=REPO / "data" / "raw")
    ap.add_argument("--discarded", type=Path, default=REPO / "data" / "discarded",
                    help="sessions with files here are never pulled again")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = ({p.name: p.stat().st_size for p in args.src.iterdir() if p.is_file()} if args.src
             else remote_listing(args.board, args.remote_dir))
    where = args.src if args.src else f"{args.board}:~/{args.remote_dir}"
    done = sorted(n[:-5] for n in files if n.startswith("rec_") and n.endswith(".json"))
    open_ = sorted(n[:-4] for n in files if n.startswith("rec_") and n.endswith(".csv")
                   and not n.endswith("_imu.csv") and f"{n[:-4]}.json" not in files)

    discarded = {p.name.split(".")[0].removesuffix("_imu") for p in args.discarded.glob("rec_*")} \
        if args.discarded.is_dir() else set()
    skipped = sorted(n for n in done if n in discarded)
    args.dest.mkdir(parents=True, exist_ok=True)
    wanted, conflicts = [], []
    for name in done:
        if name in discarded:
            continue
        for fname in (f"{name}.csv", f"{name}_imu.csv", f"{name}.json"):
            if fname not in files:
                continue
            local = args.dest / fname
            if not local.exists():
                wanted.append(fname)
            elif local.stat().st_size != files[fname]:
                conflicts.append(fname)

    print(f"{where}: {len(done)} finished sessions, {len(open_)} still recording"
          f"{f', {len(skipped)} discarded (skipped)' if skipped else ''}")
    for fname in conflicts:
        print(f"  CONFLICT, not copied: {fname} is already in {args.dest} with a different size")
    if not wanted:
        print("nothing new")
        return
    print(f"copying {len(wanted)} files to {args.dest}{' (dry run)' if args.dry_run else ''}:")
    for fname in wanted:
        print(f"  {fname}")
    if args.dry_run:
        return
    if args.src:
        for fname in wanted:
            shutil.copy2(args.src / fname, args.dest / fname)
    else:
        remote = [f"{args.board}:{args.remote_dir}/{fname}" for fname in wanted]
        if subprocess.run(["scp", "-q", "-p", *remote, str(args.dest)]).returncode != 0:
            sys.exit("scp failed")
    print("done")


if __name__ == "__main__":
    main()
