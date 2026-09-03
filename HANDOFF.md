# Handoff

Read this first, then `README.md` for the results tables.

## Situation

Zindi "A Step Ahead of Drought": forecast global Total Water Storage one month
ahead. **Closes 2026-09-13.** Scored 50% on leaderboard RMSE; the other 50% is a
Phase Two trustworthiness write-up that **only the top 10 may submit**.

| | score |
|---|---|
| competition benchmark | 0.899886 |
| v4 (recency + drift, strictly causal) | 0.767074 |
| v5 (anchor bracketing) | 0.746183 |
| v6 (leak fix + larger refit) | 0.736748 |
| **v7 (this session) — `submission.csv`, NOT YET SUBMITTED** | ? |
| top-10 cutoff | 0.655935 |
| leader (rank 1) | 0.559588 |

**The first thing to do is submit `submission.csv` (v7) and record the score
in the calibration table in `README.md`.** Everything below is the reasoning
for why it should move the board a lot more than v5/v6 did, and how to tell if
it did not.

## The transfer problem is solved (diagnosed, and the fix is in v7)

The open question from the previous handoff was why offline gains reached the
board at a fraction of their size. Two diagnostics answered it this session.

**1. `replay.py` — the honest offline metric.** It reproduces the test set's
exact layout on the training panel: the same 18 row months, the same six
anchor months, the panel masked identically, covariates present only where the
test has them, training strictly before the window. Four folds (base months 44,
56, 68, 123; the first three share the test's calendar, 123 is the closest
late-regime fold).

On it, the v6 model beat linear interpolation by **2%** (0.5938 vs 0.6063
pooled; fold 44 was *worse* than interpolation), against the **10%** the old
24-month validation claimed (0.6298 vs 0.7027). The board agrees with the
replay: v6 at 0.737 against an estimated test-period interpolation baseline of
~0.75 (section below) is a ratio of ~0.98.

The old validation was optimistic for three reasons, all structural:
- A right anchor 12–18 months out only fits inside a 24-month window for the
  three winter months right after the training cutoff, so 39% of the
  test-weighted metric was scored on the easiest, shortest-lead rows available
  (validation interp RMSE 0.52 for `(1,18)` vs 0.64 for `(1,3)` — physically
  backwards).
- Every feature was computed from a dense panel; the test panel has whole-month
  holes, so `drift36`/`nobs36`/`last_delta`/`_d1` saw a different world.
- Training rows carried recency weights and cell-identity features
  (`clim_m`, `cell_mean`, `cell_std`, lat/lon), so the model memorised each
  cell's recent anomaly. A validation window adjacent to training rewards that;
  a test window 2–40 months out does not. This is why early stopping on the
  replay halts at ~100 rounds for v6: the transferable signal is learned
  immediately and everything after is memorisation.

**2. `pseudotest.py` — real test-period labels.** The six observed test months
(164, 168, 173, 179, 198, 202) can each be predicted from their neighbouring
anchors and scored against truth. That is the only 2015–2018 ground truth we
have. Results (pooled RMSE, model / interp / ratio):

| feature set | 2013–15 window | **real 2015–18 anchors** |
|---|---|---|
| v6-style (reduced) | 0.698 / 0.754 / 0.925 | 0.793 / 0.829 / **0.957** |
| + target-month covariates | 0.660 / 0.754 / 0.874 | 0.715 / 0.829 / **0.862** |
| + drop climatology group | 0.652 / 0.754 / 0.864 | 0.704 / 0.829 / **0.850** |

So the v6 recipe kept only ~4% over interpolation in the real period, while
target-month covariates without climatology keep ~15%. That gain is measured
on the actual test years, not on a replay. (The pseudo-test configs use longer
`dL` than the test's and have target-month covariates for 100% of rows where
the test has them for 67%, so expect the board gain to be smaller than this.)

**What the leaderboard's interpolation baseline probably is.** Predicting each
observed test anchor from its neighbours gives errors 15–40% above the same
configuration and calendar month in training years (e.g. interp 173|179|198:
1.067 vs a 2002–2014 mean of 0.769). 2015 itself has persistence error 0.87
against ~0.53 in every earlier year, concentrated in deserts (Sahara, Arabia,
Tibet) where real storage barely moves — late-mission GRACE noise. The
season-honest all-years interp baseline for the test layout is 0.604; scaled by
~1.25 for period difficulty gives ~0.75. Part of the test error is therefore
unpredictable measurement noise, and the top-10 cutoff at 0.656 is a ratio of
~0.87 to that baseline.

## What v7 changed, with replay numbers

Pooled RMSE over the four replay folds, global model, model / interp:

| change | pooled | ratio | fold 123 (late) |
|---|---|---|---|
| v6 feature set, recency weights | 0.5938 / 0.6063 | 0.979 | 0.6835 |
| − climatology & cell-level features | 0.5619 | 0.927 | 0.6773 |
| + target-month covariates (`_n`, `_nres`, `_rmean`) | 0.5534 | 0.913 | 0.6839 |
| − drift features, uniform weights | 0.5474 | 0.903 | 0.6832 |
| + smoothed residuals, per-cell slope, `_nL`/`_nR` | **0.5401** | **0.891** | **0.6760** |

Note the late fold moves far less than the early ones (0.6835 → 0.6760). The
pseudo-test is the reason to expect more on the board than fold 123 suggests,
but be ready for the smaller number.

Measured flat or worse on the replay (do not re-propose without a new reason):
hyperparameters (leaves 15–255, `min_data_in_leaf` 200–10k, L2, feature
fraction: all within 0.002); residual target `y − interp` (worse by 0.003 with
the new features); per-dL blend (−0.001, dropped — v7 is the global model
only); recency half-life 15 or 40 vs uniform (uniform best by 0.002); data
volume per config 5k vs 100k rows (fold 68: 0.5028 vs 0.5026 — the model is
signal-limited, not data-limited).

**The previous handoff's "measured and REJECTED" list was measured on the old
metric and should be treated as unmeasured.** In particular residual learning
and feature dropping were rejected there for reasons that no longer hold.
`model_v6_reference.py` is the v6 code; `ablate*.py`, `diag*.py` and
`pseudotest.py` import it.

## Ranked next steps

1. **Submit v7. Then decide by the number.** If the board moves to ≲0.70, the
   replay+pseudo-test protocol is validated and the remaining ideas below are
   worth the runs. If it barely moves, the public subset is doing something the
   anchors cannot see, and the cause is again unknown.
2. **Covariate availability at masked target months.** Test.csv has no rows at
   all for months 165–167, 171–172, 177–178, 186–197, 199–201, so a third of
   rows get NaN target-month covariates. SPEI is computed from precipitation
   and PET which the rules explicitly permit from the Climate Data Store; ERA5
   precipitation/temperature at every month would fill that gap and extend the
   covariate path across the whole 18-month masked block. This is the biggest
   legitimate lever left, and it is days of work with a CDS account.
3. **More from the target-month covariates we have.** Each v2 subgroup (nL/nR,
   smoothed, per-cell slope) is worth ~0.002 on its own and they are largely
   redundant with each other; the per-cell slope on the *smoothed* residual is
   the top non-anchor feature. Try slopes fitted on residuals at the test's
   spans rather than month-to-month diffs, and 7×7 smoothing.
4. **Seed/fold ensembling** of the final refit. Cheap, variance-only.
5. **Phase Two artifacts** — `emissions.csv` (CodeCarbon) and
   `valid_preds.parquet` (now replay-fold predictions with lat/lon) exist.

## Problem structure — the non-obvious facts

- **Test has only 18 row months**, not 40: 164, 168–170, 173–176, 179–185,
  198, 202–203. Six are fully observed anchors; the other twelve are masked by
  whole month (≈0.2% stray observed cells). 88.8% of rows are bracketed.
- **Covariates exist only at those 18 months** in Test.csv. The 12–18-month
  masked block (rows 180–185, 39% of test) sees covariates for 179–185 and 198
  only.
- **Train has zero missing `TWS_t`**; train months themselves have 22 outages
  (2011–2014), and `target(m) == TWS_t(m+1)` exactly, so the panel `G` knows
  more months than `time` lists.
- **The field is extremely smooth spatially** (month-to-month change correlates
  0.999 with E–W neighbours): 15.7k cells are a few hundred independent
  samples, which is why data volume does nothing.
- **Late-mission GRACE (2015+) is noisier**, especially in deserts, and the
  test anchors (179 above all) carry that noise into every interpolation.
- Month index convention: `m = (year-2002)*12 + (month-1)`. Train spans 4–163,
  test targets 165–204.

## The rules decision

The competition permits Copernicus covariates *"provided that these variables
are available at the stated prediction time and do not directly or indirectly
include future GRACE/TWS information."* v5+ uses observed TWS at a later month
as a right anchor; v7 additionally uses SPEI/soil-moisture at the target month
(the row for that month in Test.csv). Both are values the organisers placed in
Test.csv; neither is external data. The user chose to build the right anchor
with the ambiguity in view; the target-month covariates are the same category
of choice and less strong. `model_causal_reference.py` (board 0.767) is the
strictly forward-looking version if a moderator rules otherwise.

## Practical constraints

- 7 GB RAM, 12 cores. `model.py` builds two replay folds (200k rows/config)
  then refits on 400k rows/config (5.9M rows × 77 features, 220 rounds); peak ~5 GB. Last run: fold 68 0.5022, fold 123 0.6767 (interp 0.5736 / 0.7249).
  `SMOKE=1 .venv/bin/python -u model.py` runs the whole pipeline on 5k
  rows/config in ~3 min — use it after any edit.
- `replay.py` caches feature frames per fold as `cache_replay_v7_b*_n*.parquet`
  (gitignored); an ablation on cached folds takes ~1 min per fold.
- Always run Python with `-u`; redirected output is otherwise buffered.
- Environment is `uv`-managed: `.venv/bin/python`.
- `data/` is gitignored; download from Zindi on a fresh checkout.
- Background waits longer than ~10 min get killed by the harness; poll the log
  instead.

## A caution on reading importance tables

`last_tws`, `nb_last_tws`, `interp`, `interp_nb` top the gain ranking and are
near-duplicates of each other; gain redistributes between collinear features
without skill moving. Among the covariate features `beta_sm_nres_s3` is first.
Use SHAP, not gain, for any skill claim in the Phase Two write-up.
