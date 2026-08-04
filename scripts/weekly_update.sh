#!/bin/bash
# Weekly prospective-validation cycle: new auction days -> new forecasts ->
# score matured ones. Installed as a launchd agent
# (~/Library/LaunchAgents/com.samihabbal.cardamom-weekly.plist);
# logs to ~/Library/Logs/cardamom-weekly.log.
# If the repo moves (plan item 6), update REPO here AND in the plist.
set -euo pipefail

REPO="/Users/samihabbal/dev/cardamom-quant"
PY="$REPO/.venv/bin/python"

echo ""
echo "=== cardamom weekly update · $(date '+%Y-%m-%d %H:%M %Z') ==="
cd "$REPO"

# launchd's calendar trigger can fire right as the machine wakes from sleep,
# before wifi has reassociated and DNS is up (this killed the 2026-08-02
# run outright). Retry the network-dependent step a few times before giving up.
attempt=1
max_attempts=5
until "$PY" scripts/refresh_spot.py; do
  if [ "$attempt" -ge "$max_attempts" ]; then
    echo "=== refresh_spot.py failed after $attempt attempts, giving up ==="
    exit 1
  fi
  echo "=== refresh_spot.py failed (attempt $attempt/$max_attempts), retrying in 60s ==="
  attempt=$((attempt + 1))
  sleep 60
done

"$PY" scripts/forecast.py

if ! git diff --quiet -- data/live data/processed data/raw/spices_board || \
   ! git diff --cached --quiet -- data/live data/processed data/raw/spices_board; then
  git add data/live data/processed data/raw/spices_board
  git commit -m "Weekly automation: $(date '+%Y-%m-%d') run

Auto-committed by weekly_update.sh." >/dev/null
  echo "=== committed ledger update ($(git rev-parse --short HEAD)) ==="
else
  echo "=== no data changes to commit ==="
fi

echo "=== done ==="
