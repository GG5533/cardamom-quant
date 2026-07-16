#!/bin/bash
# Weekly prospective-validation cycle: new auction days -> new forecasts ->
# score matured ones. Installed as a launchd agent
# (~/Library/LaunchAgents/com.samihabbal.cardamom-weekly.plist);
# logs to ~/Library/Logs/cardamom-weekly.log.
# If the repo moves (plan item 6), update REPO here AND in the plist.
set -euo pipefail

REPO="/Users/samihabbal/Documents/Claude/Projects/My personal Profile and who i am/cardamom-quant"
PY="$REPO/.venv/bin/python"

echo ""
echo "=== cardamom weekly update · $(date '+%Y-%m-%d %H:%M %Z') ==="
cd "$REPO"
"$PY" scripts/refresh_spot.py
"$PY" scripts/forecast.py
echo "=== done ==="
