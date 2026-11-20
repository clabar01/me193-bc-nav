#!/usr/bin/env bash
# Deploy the me193_nav app to the Uno Q (AirFour) and restart it.
#
#   board/me193_nav/deploy_board.sh             # Python side + model
#   board/me193_nav/deploy_board.sh --sketch    # also copies the sketch
#   BOARD=arduino@<ip> board/me193_nav/deploy_board.sh   # another address
#
# Either way 'app start' recompiles and reflashes the MCU, so power-cycle the
# board afterwards.
#
# Checks that the model shipped in python/models/ still matches models/ at the
# repo root (scripts/train_detector_imu.py writes it there).
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-}" in
  "" | --sketch) ;;
  -h | --help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "unknown option: $1 (use --sketch or --help)" >&2; exit 2 ;;
esac

BOARD="${BOARD:-arduino@10.247.137.47}"     # AirFour on Tufts_Robot
APP_NAME="me193_nav"
APP="~/ArduinoApps/$APP_NAME"

for app_copy in python/models/*.npz; do
  f="../../models/$(basename "$app_copy")"
  if [[ ! -f "$f" ]] || ! cmp -s "$f" "$app_copy"; then
    echo "$app_copy differs from models/$(basename "$app_copy"): copy the exported model again" >&2
    exit 1
  fi
done

files=(app.yaml python)
if [[ "${1:-}" == "--sketch" ]]; then
  files+=(sketch)
fi

ssh "$BOARD" "mkdir -p ArduinoApps/$APP_NAME"
echo "copying ${files[*]} to $BOARD:ArduinoApps/$APP_NAME/"
scp -q -r "${files[@]}" "$BOARD:ArduinoApps/$APP_NAME/"
echo "restarting $APP_NAME (stop + start; 'app restart' can fail with 'app is running')"
ssh "$BOARD" "timeout -k 5 120 arduino-app-cli app stop $APP; timeout -k 5 420 arduino-app-cli app start $APP"
# 'app start' reflashes the sketch even when it hasn't changed (seen 2026-10-26)
echo
echo ">>> The MCU was reflashed: POWER-CYCLE THE BOARD before trusting it (unplug ~5 s, plug back in)."
echo ">>> Phone page: http://${BOARD#*@}:7000/"
