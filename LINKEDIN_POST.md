# LinkedIn post — Cardamom Quant

**Every number here is filled from real output and current as of the 2026-08-11
weekly run (3,178 auction days; live ledger 12/19, Brier 0.2476). Re-run
`scripts/weekly_update.sh` on the morning you post and re-check the live-ledger
sentence and the figures — the scorecard moves ~5 forecasts a week. Never post
with synthetic numbers; the honesty is the pitch.**

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

**The result.** On 3,178 real auction days (2014–2026): the gradient
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
37-configuration deflated-Sharpe ledger where every failed idea stays
counted. The honest label is *a maybe* — positive with ~94% confidence,
shallow drawdowns, real classification skill (AUC 0.55) — and since
July 2026 the model forecasts LIVE into an append-only, hash-chained
ledger, scored as outcomes mature, because a backtest has degrees of
freedom and a time-stamped forecast has none. That live ledger is at 19
scored forecasts, 12 of them correct, Brier 0.2476 against 0.25
climatology. Watch it move and you learn exactly what n=19 is worth: it
was 9 of 13, then 10 of 17, now 12 of 19 — the Brier crossed to the
wrong side of climatology and back inside three weeks, and none of those
swings was the model getting better or worse.

**Why 19 is the number I'm proudest of.** Before this I spent months on
the impressive version: a market platform, 37,004 lines across 81
modules, an LLM debate engine, prediction markets, four background
daemons. It has a calibration engine — reliability curves, Brier, ECE —
and a database table with exactly the right columns to grade its own
calls: predicted probability, actual result, correct, simulated P&L.

I checked that table this week. It made **8,438 predictions. It graded
zero of them.** Not because the code was wrong — the schema is right, the
resolution daemon exists. I stopped running it before it could ever tell
me whether it worked.

The two projects aren't strangers: the seasonal Kalman filter in this
cardamom model is a direct port of that platform's world-model engine.
Same author, same maths, same calibration apparatus. The difference is
that the small one closed the loop and the big one never did. Eight
thousand ungraded predictions are worth less than nineteen graded ones,
and I only know that because I built both.

So: a backtest that can say "no" is the only kind whose "maybe" means
anything — and a forecast log you can't quietly edit is the only kind
that ever finds out.

**The finding that outlived the model.** Five separate times I added a
carefully-built feature block to a model that already worked. Five times
it got worse. The whole alt-data layer — rain, ENSO/IOD, rotating demand
calendars, auction microstructure, Guatemala supply — took the gradient
booster from Sharpe +0.25 down to +0.03. Stacking a Kalman anomaly filter
onto the physics champion: +0.79 → +0.71. Stacking rainfall: → +0.55.
Stacking Guatemala export volume: → +0.58. Stacking as-issued GEFS
forecast rain — the most theoretically motivated of the lot, and the one
I most wanted to work — lost to a same-window realized-rain baseline by
0.27 Sharpe. Every one of those blocks was individually defensible.
Several cleared zero on their own; rainfall posted the best
classification numbers in the entire project (AUC 0.572, +6.3 hit-rate
points). None survived contact with a model that already had signal.

On roughly 500 effectively independent 5-day bets, one more column in the
tree costs more than it pays. That's the result I'd actually defend in an
interview — not the Sharpe. On small samples, feature addition is a tax,
and the only way to learn that about your own work is to pre-register
each attempt and leave the failures in the ledger, where they go on
haircutting your headline forever.

Capacity caveat up front: this is a niche market (~₹25 crore/day through the
auctions). The point was never scale — it was demonstrating desk-grade
process on a market nobody else models.

Repo: github.com/GG5533/cardamom-quant. Stack: Python, pandas,
scikit-learn, Streamlit. 82 tests, no notebook heroics.

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
my own fold-slicing luck. 37 configurations tried, every failure still in the
ledger, and the model now forecasts live into a hash-chained ledger so the
next number can't be argued with. Reporting the haircuts IS the portfolio
piece. Repo in comments.

---

## Attachments (in order of impact)

All four are built and current. Regenerate the first three with
`python scripts/make_flagship_chart.py` + `python scripts/make_post_figures.py`
(both recompute from the real dataset; `make_post_figures.py` refuses to run on
synthetic data). The dashboard shot is captured headless via Playwright — set
the model to **gbm** with alt-data OFF first; the app defaults to `logistic`,
which is the worst variant in the project (Sharpe −0.20).

1. `figures/spot_history.png` — 12 years of spot, suspension/relaunch window
   shaded, 2018-flood peak marked. *The* visual of the data-regime story.
2. `figures/turnover_story.png` — paired dumbbell, every configuration traded
   daily vs at its own 5-day horizon: 2.5× the turnover for a mean Sharpe
   change of −0.02. The turnover-was-the-killer story, stated honestly.
3. `figures/ablation.png` — core vs core+alt across three models. This is
   dilution #1 of the five, and it pairs directly with "the finding that
   outlived the model" paragraph.
4. `figures/dashboard.png` — equity tab, REAL banner visible, all three models.
5. **The interactive one — lead with this if the platform allows a link.**
   The showcase page's "predictions made vs predictions graded" section:
   8,438 hollow marks for the big platform against 24 for this model, 19 of
   them filled. Counts come from each project's own store (the platform's
   SQLite: 7,647 Kalshi + 791 Polymarket recommendations, `kalshi_outcomes`
   empty; and this repo's `data/live/*.csv`). Hovering any mark on the right
   shows its date, verdict and chain hash. The asymmetry is the argument, and
   it lands faster than any paragraph about it does.

Deliberately cut: the raw horizon grid and the SHAP bar chart. Sixteen rows of
numbers don't survive a phone screen, and SHAP invites a feature-importance
argument that distracts from the actual finding.

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
