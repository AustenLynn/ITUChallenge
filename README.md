# A Step Ahead of Drought — Global Water Storage Forecasting

Zindi challenge: forecast Total Water Storage (TWS) one month ahead on a global
1° grid. Scored by RMSE (50% of final score); the remaining 50% is a Phase Two
trustworthiness write-up available only to the top 10.

## Problem shape

The framing that matters is not "predict TWS from TWS_t". `TWS_t` is masked for
66.5% of test rows, **by whole months**, reproducing the real GRACE / GRACE-FO
data gap. Every row is instead posed as:

> predict TWS at month `m`, given the last month `L` where TWS was actually
> observed, the horizon `h = m - L` (1..7), and covariates over `[L, m-1]`.

Training rows are replayed at every horizon so the model learns how skill decays
with the gap. Test horizon mix: `{1: 94048, 2: 62576, 3: 46777, 4: 31076,
5: 15560, 6: 15479, 7: 15445}` — validation is reweighted to match it, otherwise
an unweighted average misreports where you stand.

Two structural facts that are easy to miss:

- **Train has zero missing `TWS_t`; test is 66.5% missing.** Anything that leans
  on `TWS_t` directly looks fine in CV and collapses on the leaderboard.
- **Train months are themselves non-contiguous** — 22 missing months (GRACE
  outages, clustered 2011–2014), so a training window can have holes where a
  test window never does.

## What worked

| change | measured gain (test-weighted RMSE) |
|---|---|
| horizon framing + per-cell climatology | baseline |
| anchor-delta covariates (change since month `L`) | −0.0072 |
| spatial neighbour smoothing of the anchor | −0.0009 |
| **exponential recency weighting** (half-life 15 mo) | **−0.017** |
| **per-cell drift features** (trailing mean/trend vs training mean) | **−0.004** |

Persistence scores 0.818 and climatology 1.011 on the same weighted metric; the
current model scores **0.681**.

The dominant effect is **non-stationarity**, not missing features. Mean target
declines steadily across the record (+0.226 in 2002 → −0.136 in 2015), so a
model weighting 2002–2010 equally with 2014–2015 is structurally biased high for
a 2015–2019 test period. Soft recency decay beats every hard cutoff, and the
gain *grows* under a gapped holdout that mimics the longer lead we are scored at.

## What was measured and rejected

Recorded so they are not re-proposed. Each looked principled and lost:

- **Dropping features whose train/test availability differs** (`last_delta` is
  100% absent for masked test rows, present in training): −0.009 *worse*.
  Keeping them and letting LightGBM route the NaNs wins — they still pay off on
  the h=1 rows where they exist.
- **Masking those features during training** to match test availability: a
  further −0.008 worse.
- **Residual learning** against a seasonal-persistence base: −0.014 worse;
  against plain persistence, −0.009. The seasonal base is itself worse than raw
  persistence (0.862 vs 0.838), so it is a worse anchor, not a better one.
- **Hyperparameter tuning** (255→63 leaves, lr 0.05→0.02): −0.001. The model is
  feature- and distribution-limited, not capacity-limited.
- **Requiring complete covariate windows** in training: −0.004 worse.

## Layout

    model.py                 end-to-end: features, training, submission
    explore.py               schema / null-rate inspection
    ablate.py, ablate2.py    feature-parity and residual-target ablations
    diag.py, diag2.py, diag3.py   non-stationarity diagnostics (recency, drift)
    model_v2_reference.py    superseded version kept for comparison
    run_summary.json         selected model + scores from the last run
    emissions.csv            CodeCarbon log, appended per run

## Running

    uv venv .venv && uv pip install -p .venv/bin/python \
        pandas numpy lightgbm scikit-learn pyarrow codecarbon
    # download Train.csv, Test.csv, SampleSubmission.csv into data/
    .venv/bin/python model.py     # writes submission.csv

## Leaderboard calibration

| version | test-weighted validation | public leaderboard | commit |
|---|---|---|---|
| v2 (anchor deltas + neighbours) | 0.6992 | 0.774957 | pre-repo |
| v4 (recency weighting + drift)  | 0.6812 | 0.767074 | 6a20efd |

Benchmark 0.899886; 476 active participants. The public leaderboard scores only
a portion of the test set, so the final standing will move.

**Roughly 44% of validation gains transfer** (-0.0180 offline bought -0.0079 on
the board). Discount offline improvements by ~2.3x when predicting leaderboard
movement. The absolute gap also widened slightly (0.076 -> 0.086), so some of
the recency gain is specific to a validation window that sits immediately after
the training data, where the real test period runs up to 40 months out.

## Caveats

- `nb_last_tws` dominates gain-based importance while `last_tws` sits third.
  This is collinearity redistributing gain, not skill — the two are near
  duplicates. Do not read the importance table as a skill ranking; use SHAP.
- Validation (2013–2015) has consistently scored better than the leaderboard
  (2015–2019). The recency correction targets that gap directly but has not been
  confirmed to close it.
