---
name: verify
description: Verify cardamom-quant end-to-end — tests, real-data scorecard reproduction, dashboard boot, and the honesty anchors that must never drift silently.
---

# Verifying cardamom-quant

Run from the repo root with the project venv (`.venv`, Python 3.12 — system
3.14 breaks numba/shap).

## 1. Test suite (fast, always first)

```bash
.venv/bin/python -m pytest tests/ -q
```

Expect **all green** (64 tests as of edge-hunt round 2). If pytest collects
FEWER tests than expected with no errors: stale `__pycache__` synced through
iCloud is masking modules — `rm -rf tests/__pycache__ .pytest_cache` and
rerun. (This actually happened; it silently hid 9 tests.)

## 2. Scorecard reproduction (the numbers are the product)

```bash
.venv/bin/python run.py            # [REAL] tag, not [SYNTHETIC]
```

Anchors (sklearn-version wiggle ±0.05 Sharpe / ±0.5pt hit is known and OK):
- seasonal_baseline Sharpe ≈ +0.30
- Since round 3, `market.parquet` carries real rain, so run.py's "core"
  includes rain_anom_30/90 — expect T6-like numbers (AUC ≈ 0.57, hit
  ≈ +6pts). For the PRE-rain core/gbm anchor (+3.6…+4.2pts, AUC ≈ 0.55),
  drop the three rain columns first (see the champion re-verification
  snippet in git history).
- If anything moved more than the wiggle: STOP and find out why before
  trusting any new result produced in the same session.

## 3. Experiment scripts regenerate their published tables

```bash
.venv/bin/python scripts/analyze.py             # ablation/DSR/calibration
.venv/bin/python scripts/horizon_experiment.py  # grid + tranched books
.venv/bin/python scripts/edge_hunt.py           # round-2 trials
```

Check against RESULTS_REAL.md. The tranched calibrated book ≈ +0.57 and
physics-gbm ≈ +0.79 are the headline anchors.

## 4. Dashboard boots with the REAL banner

```bash
.venv/bin/streamlit run app.py --server.headless true --server.port 8601
# curl -s localhost:8601 → HTTP 200, no tracebacks in the log, then kill it
```

## 5. Data lineage

`market.parquet` is rebuilt by `scripts/refresh_spot.py` (crawl → exact-
overlap verification against the repaired history → append-only → rebuild
with ONI + rain). Canonical mutable files: `sessions_canonical.csv` /
`spot_daily_canonical.csv`. The 07-Jul-2026 browser dump
(`sessions_full_repaired.csv` / `spot_daily_full.csv`) is the immutable
provenance seed — macOS quarantine-locks it against rewriting; never try.
If the refresh ABORTs on overlap mismatch, that is the comma-corruption
tripwire doing its job — investigate, don't override.

## 6. Honesty checklist (read RESULTS_REAL.md ledger line)

- Every model/config ever evaluated appears in the DSR trial ledger
  (29 trials after round 2). If you evaluated anything new — even a failed
  idea — the ledger grew.
- Every new feature has a "mutate today, assert today unchanged" test.
- No feature crosses the 2021–2025 MCX suspension gap; ONI/Comtrade keep
  their 2m/3m publication lags.
