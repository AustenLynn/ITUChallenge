# Handoff

Read this first, then `README.md` for the measured results table.

## Situation

Zindi "A Step Ahead of Drought": forecast global Total Water Storage one month
ahead. **Closes 2026-09-13.** Scored 50% on leaderboard RMSE; the other 50% is a
Phase Two trustworthiness write-up that **only the top 10 may submit**.

| | score |
|---|---|
| competition benchmark | 0.899886 |
| v2 | 0.774957 |
| v4 (recency + drift) | 0.767074 |
| **v5 (anchor bracketing) — current** | **0.746183** |
| leader (rank 1) | 0.559588 |

Rank was 259 of 476 active at 0.775; the rank at 0.746 is unknown — **ask the
user for it**, and for the rank-10 score, since that determines whether Phase
Two is even reachable.

## Start here: the transfer problem

**Offline gains keep failing to reach the board, and the cause is unresolved.**

- v2 → v4: offline −0.0180, board −0.0079. ~44% transferred.
- v4 → v5: offline looked enormous (persistence 0.7627 → model 0.6252), board
  moved only −0.0209.

I claimed v5's config-keyed validation was "closely calibrated to the board"
because its persistence baseline (0.7627) sat near v4's board score (0.767).
**That reasoning was wrong** — comparing a persistence *baseline* against a
*model's* board score is not a calibration, and the v5 result contradicts it.

**First task, before any new modelling:** score the v4 causal feature set on the
v5 config-keyed validation. That yields the first clean offline delta for v5 and
a real transfer ratio. `model_causal_reference.py` holds the v4 features;
`model.py` holds the config-keyed harness; `ablate.py` shows the pattern for
running two feature sets against one fixed validation set.

Two hypotheses worth separating while you're in there:

1. **Public-subset effect.** The public board scores "only a portion of the
   total test set". If that portion is drawn by month rather than at random, the
   gain on bracketed rows may be under-represented and the private score could
   be much better than the public one. Cannot be resolved directly, but it
   changes how much weight to put on public feedback.
2. **Validation optimism.** Bracketed validation rows are drawn from the dense
   contiguous train panel; test anchors span the GRACE → GRACE-FO mission change
   (months 198 and 202 are GRACE-FO). Per-month means show no obvious offset
   (−0.21 to +0.09 across all observed test months), so this looks weak, but the
   `dR=12..18` configs — 39% of test, all anchored on month 198 — are where it
   would bite.

## Problem structure — the non-obvious facts

- **Train has zero missing `TWS_t`; test is 66.5% missing, masked by whole
  months.** Anything leaning on `TWS_t` directly looks fine in CV and dies on
  the board.
- **88.8% of test rows are bracketed**: Test.csv carries six fully-observed
  months (164, 168, 173, 179, 198, 202) and every masked block sits between two
  of them. Those rows are interpolation, not forecasting. This is the single
  largest structural fact and v5 is built on it.
- **Train months are themselves non-contiguous** — 22 GRACE outages clustered
  2011–2014 — so training windows have holes that test windows never have.
- **`target(m) == TWS_t(m+1)` exactly** (verified, zero max abs diff). The panel
  in `build_panel` relies on this.
- Month index convention throughout: `m = (year-2002)*12 + (month-1)`. Train
  spans 4–163, test targets 165–204.
- **The target is non-stationary**: annual mean falls from +0.226 (2002) to
  −0.136 (2015). Uniform training over 13 years biases predictions high for a
  2015–2019 test period.

## What worked, with measured gains

| change | gain |
|---|---|
| horizon framing + per-cell climatology | baseline |
| anchor-delta covariates | −0.0072 |
| spatial neighbour smoothing | −0.0009 |
| exponential recency weighting (half-life 15 mo) | −0.017 |
| per-cell drift features | −0.004 |
| anchor bracketing / interpolation (v5) | large offline, +0.021 board |

## What was measured and REJECTED — do not re-propose

Each looked principled and lost. Numbers are config- or test-weighted RMSE.

- Dropping features whose train/test availability differs (`last_delta` is 100%
  absent for masked test rows): **−0.009 worse**. Keep them; LightGBM routes
  NaNs fine and they still pay on the rows where they exist.
- Masking those features during training to match test availability: a further
  **−0.008 worse**.
- Residual learning against a seasonal-persistence base: **−0.014 worse**;
  against plain persistence, **−0.009 worse**. The seasonal base is itself worse
  than raw persistence (0.862 vs 0.838).
- Hyperparameter tuning (255→63 leaves, lr 0.05→0.02): **−0.001**. Model is
  feature- and distribution-limited, not capacity-limited.
- Requiring complete covariate windows in training: **−0.004 worse**.

## Ranked next steps

1. **The transfer diagnostic above.** Everything else is guesswork until offline
   movement can be trusted.
2. **External Copernicus covariates — the biggest untouched legitimate lever.**
   The rules *explicitly permit* data from the Climate Data Store and the
   European/Global Drought Observatories. We have only ever used the five
   columns in the CSV. ERA5 precipitation, evapotranspiration, temperature,
   runoff and snow water equivalent are all fair game and are the most likely
   explanation for a leader at 0.5596. Days of work, not an afternoon.
3. **Ensembling** across seeds and configs. Reduces variance rather than bias,
   so it should transfer closer to 1:1 than feature work has. Modest but
   dependable.
4. **Phase Two artifacts** — only if the rank makes top 10 plausible.
   `emissions.csv` (CodeCarbon, per run) and `valid_preds.parquet` (held-out
   predictions with lat/lon/config) already exist for the bias and
   sustainability sections.

## The rules decision

The competition permits Copernicus covariates *"provided that these variables
are available at the stated prediction time and do not directly or indirectly
include future GRACE/TWS information."*

v5 uses observed TWS at *later* months as a right anchor. Read strictly, that is
future TWS information. Read narrowly, that sentence governs external covariates
a participant brings in, not values the organisers themselves placed in
Test.csv. **The user was shown the measurement and the ambiguity and chose to
build it.** Do not silently revert it; equally, do not extend it further without
checking in.

`model_causal_reference.py` is the strictly forward-looking version (board
0.767) and still runs. If a moderator rules on this, a compliant submission is
one run away.

## Practical constraints

- **7 GB RAM total; the v5 run peaked at 6 GB and finished with 68 MB free.**
  There is no headroom. Raising `N_PER_CFG` (currently 300k × 17 configs) will
  OOM. Reduce it or move feature frames to disk before scaling anything up.
- 12 cores. A full `model.py` run takes ~15 min.
- Environment is `uv`-managed; `python3 -m venv` fails (no `ensurepip`):
  `uv venv .venv && uv pip install -p .venv/bin/python pandas numpy lightgbm scikit-learn pyarrow codecarbon`
- **Redirected Python output is buffered — always run with `python -u`** or logs
  appear empty and progress is invisible.
- `data/` is gitignored; re-download from Zindi if the checkout is fresh.

## A caution on reading importance tables

`nb_last_tws` tops every gain ranking, with `interp_nb` second. These are
near-duplicates of `last_tws` and `interp`; gain redistributes between collinear
features without skill moving. Do not report that ordering as a skill ranking —
in the Phase Two transparency section or anywhere else. Use SHAP for that claim.
