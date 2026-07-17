# LinkedIn post — Cardamom Quant

**Fill the [REAL: …] placeholders from `python run.py` / `scripts/analyze.py` output
after the backfill. Do not post with synthetic numbers — the honesty is the pitch.**

---

## Main post

I built a price-direction model for Indian cardamom. The most valuable thing
it produced wasn't the Sharpe ratio — it was what I found before writing a
single feature.

**The trap.** The plan was standard: pull MCX cardamom futures history, build
weather + seasonality features, backtest a decade. Except when I went to wire
the data, the futures contract had only existed for 11 months — SEBI-era
suspension in 2021, relaunched July 2025 as a redesigned compulsory-delivery
contract. Any backtest splicing the old and new regimes into one series would
have been fiction with confidence intervals.

**The pivot.** The tradable instrument was young, but the underlying cash
market wasn't: the Spices Board publishes every e-auction session — price AND
physical arrivals — back roughly a decade, on the same ex-Idukki pricing basis
as the future. So the model trains on the cash market and treats the young
future as what it is: a basis signal and a tradability check, not a history.

**The signals.** Beyond price: lagged monsoon rainfall anomalies over the
Idukki growing belt (IMD gridded data), ENSO/IOD indices as *forecastable*
weather, auction microstructure (sell-through tension, arrival surprises),
Guatemala export shocks (the world's #1 exporter lost ~40–50% of its crop in
2024-25), and demand calendars that rotate against the solar year — Ramadan
moves 11 days earlier annually, so Gulf stocking demand is invisible to
day-of-year seasonality. Every non-calendar feature carries an explicit
publication lag, each enforced by a unit test.

**The discipline.** Purged walk-forward CV (labels overlap; training windows
are cut back accordingly). A deliberately embarrassing seasonal baseline the
ML has to beat. Costs charged on turnover, execution lagged a day. And the
part I'd want to see if I were hiring: a block-bootstrap CI on the Sharpe and
a Deflated Sharpe Ratio that haircuts for every model I tried — because a
good number selected from six attempts is often just the luckiest of six.

**The result.** On 3,148 real auction days (2014–2026): the gradient
booster found genuine predictive signal — +4.2pts hit rate over base, AUC
0.553, out-of-sample — and still lost to a one-line seasonal rule after
15bps costs at daily rebalancing (Sharpe +0.12, CI straddling zero).
Adding my carefully-built alt-data features made the ML *worse* — on
3,000 noisy samples, complexity isn't free, it's negative. The autopsy
said turnover, not skill, was the killer: 16.6× annual turnover trading a
5-day signal daily. So I matched the trading to the label — isotonic-
calibrated probabilities, rebalanced at the 5-day horizon, implemented as
five staggered tranches so no lucky anchor day flatters the number.
Then I mined the columns nobody models — the auction's own physics:
the max/average bid spread (competition intensity), crop-year inventory
overhang, a rolling Hurst regime dial. That configuration printed Sharpe
+0.79. And here is the part I'm most proud of: I then attacked my own
number twice. Re-slicing the walk-forward folds after a data refresh
moved it to +0.69; averaging over six equally-defensible fold layouts —
because the layout every table used turned out to be the luckiest of
six — settled it at **+0.43, 90% CI [−0.01, +0.90]**, against a
34-configuration deflated-Sharpe ledger where every failed idea stays
counted. The honest label is *a maybe* — positive with ~94% confidence,
shallow drawdowns, real classification skill (AUC 0.55) — and since
July 2026 the model forecasts LIVE into an append-only, hash-chained
ledger, scored as outcomes mature, because a backtest has degrees of
freedom and a time-stamped forecast has none. A backtest that can say
"no" is the only kind whose "maybe" means anything.

Capacity caveat up front: this is a niche market (~₹25 crore/day through the
auctions). The point was never scale — it was demonstrating desk-grade
process on a market nobody else models.

Repo: [link]. Stack: Python, pandas, scikit-learn, Streamlit. 77 tests,
no notebook heroics.

---

## Short variant (for comments / reposts)

Spent the last stretch building an ML price model for MCX cardamom. Biggest
lesson: the futures contract I planned to model had only 11 months of history
(suspended 2021, relaunched 2025) — so the real work was rebuilding the
problem around 12 years of cash-market e-auction data, with the future as a
tradability overlay. Purged walk-forward, honest seasonal baseline, bootstrap
CIs, deflated Sharpe. Result: real predictive signal (AUC 0.55 OOS) that dies
at daily rebalancing, revives at its own 5-day horizon, peaks at +0.79 with
auction-microstructure features — and settles at +0.43 once I averaged away
my own fold-slicing luck. 34 configurations tried, every failure still in the
ledger, and the model now forecasts live into a hash-chained ledger so the
next number can't be argued with. Reporting the haircuts IS the portfolio
piece. Repo in comments.

---

## Attachments (in order of impact)

1. Full spot history chart, suspension/relaunch window shaded — *the* visual
   of the data-regime story (generate after backfill).
2. The horizon grid + tranched table from `scripts/horizon_experiment.py` —
   the turnover-was-the-killer story in one screenshot.
3. Ablation × robustness table from `scripts/analyze.py` (core vs core+alt,
   with p(Sharpe≤0) column).
3. Dashboard screenshot (equity tab, REAL banner visible).
4. Calibration table or SHAP bar chart — pick whichever tells the cleaner story.

## Posting notes

- Lead with the trap/pivot, not the model — hiring managers read the first
  two lines only.
- Keep the [SYNTHETIC] appendix out of the post entirely; link the README
  section instead if asked.
- Expect the question "why cardamom?" — answer: clean thesis (weather →
  yield → price with a lag), public spot venue, and nobody else models it,
  so every result is attributable to process rather than crowding.
- Tag: quantitative finance, commodities, machine learning. No more than 3–4
  hashtags; it reads junior otherwise.
