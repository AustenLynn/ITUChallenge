# Handoff

Read this first, then `README.md` for the results tables.

## Situation

Zindi "A Step Ahead of Drought": forecast global Total Water Storage one month
ahead. **Closes 2026-09-13.** Scored 50% on leaderboard RMSE (public board is
~30% of test rows, 5 submissions a day); the other 50% is a Phase Two
trustworthiness write-up that **only the top 10 may submit**.

| | score |
|---|---|
| competition benchmark | 0.899886 |
| v4 (recency + drift, strictly causal) | 0.767074 |
| v5 (anchor bracketing) | 0.746183 |
| v6 (leak fix + larger refit) | 0.736748 |
| v7 (transferable features, replay selection) | 0.738568 |
| v8 (half-life 24, lag anchors, wide neighbour means) | **0.730218** |
| linear interpolation (calibration) | 0.802237 |
| hybrid: v7 on long block + extrapolation, interp elsewhere | 0.753301 |
| top-10 cutoff | 0.655935 |
| leader (rank 1) | 0.559588 |

v7 was predicted to move the board a lot (replay ratio 0.891 vs v6's 0.979)
and moved it by −0.002. This session's work was to find out why and to leave
the next session with checks that do predict the board.

## Board calibration (done 2026-09-03)

Interpolation scores **0.802237**; v7 0.738568 is a ratio of 0.921, between
fold 123 (0.934) and the 2015 late check (0.906). The hybrid
(`submission_hybrid_long.csv`, v7 on `dR ≥ 12` and `dR = 0`, interpolation
elsewhere) scored 0.753301, so in public-board MSE v7 removes 0.098 of
interpolation's 0.644: 0.076 on the long block + extrapolation half and 0.022
on the short bracketed half. The short-span gain is therefore 3–4%, which is
what the non-junk 2015 months of the late check show (the late check's
pooled 0.906 is inflated by the junk-anchor months 159/163/164, which the
test does not have). The late-regime checks predict the board; the pooled
replay does not. `calib_subs.py` rebuilds the three files.

To reach the top-10 cutoff (0.656, MSE 0.430) another 0.115 MSE has to go —
21% of what is left. The leader (0.560, MSE 0.313) removes half of
interpolation's MSE; nothing measured here comes close to that legitimately.

## Why v7 did not move the board

Measured this session with `diag5_regime.py`, `diag6_shrink.py`,
`diag7_covariate_regime.py`; details in README.md "What the data is".

1. **The covariate coupling halved after 2011.** Correlation of one-month TWS
   change with same-month soil-moisture (and SPEI) change is 0.23–0.29 in
   every year 2004–2011 and 0.09–0.21 from 2012 on; the soil-moisture product
   itself did not change. The test anchors show the weak-regime coupling.
   Replay folds 44/56/68 sit entirely in the strong regime and carried the
   whole v7 selection; fold 123 (2012–2015) gave the v7 changes ~1%, the
   board ~0%. The pooled replay is not a metric for anything involving
   covariates.
2. **The test regime is noisy in a way the early folds are not.** One-step RMS
   change is 0.5–0.6 through 2014 and 0.83 in 2015, with junk months
   (157–159, 162–163) at 1.0–1.2 and 4× the normal east–west roughness.
   Test anchor-to-anchor differences run 11–39% above the same calendar
   pairs in training years, concentrated in the low-autocorrelation cells
   (Sahara, Arabia, Tibet). Targets 180–186 (39% of rows) are the
   accelerometer-off end of GRACE. Part of the score is irreducible.
3. **TWS is deseasonalised** (seasonal cycle 4% of cell variance), which is
   why dropping climatology features helped everywhere and why seasonal
   interpolation loses to linear on every configuration.

What still holds: v7 beats interpolation on every late-regime check —
fold 123 0.934, the 2015 short blocks 0.906 (`latecheck.py`), the six
observed test months with real labels 0.823 (`pseudotest7.py`). The gains are
largest where the anchor is a junk month (the neighbour-mean features average
out the striping) and on long spans.

## Measured on the late checks this session

Model RMSE and ratio to interpolation. Late check: 2015 short blocks,
test-weighted, interp 0.7848. Pseudo-test: six observed test months, real
labels, interp 0.8290. Logs: `logs/latecheck_ablate.log`,
`logs/pseudotest7_variants.log`.

| variant | late check (short spans) | pseudo-test (long spans) |
|---|---|---|
| v7 recipe | 0.7111 / 0.906 | 0.6820 / 0.823 |
| − target-month covariates (`next,nLR,smooth,beta`) | 0.7153 / 0.911 | 0.7850 / 0.947 |
| − all covariate features (`+path,dyn,delta`) | 0.7273 / 0.927 | |
| − all covariates and lat/lon | 0.7275 / 0.927 | |
| train on target months ≥ 108 (2011+) only | 0.7052 / 0.899 | 0.6880 / 0.830 |
| train on target months ≥ 120 (2012-05+) only | 0.7024 / 0.895 | |
| recency half-life 12 months | 0.7017 / 0.894 | |
| recency half-life 24 months | 0.7020 / 0.894 | 0.6832 / 0.824 |
| exclude 2015 junk months from training | n/a (already excluded) | 0.6837 / 0.825 |

Reading: the target-month covariates are worth 0.5% on short spans in the
late regime (4% on the early folds) but **12% on the long spans with real
labels** — keep them; the long block is 39% of the test. **A recency
half-life of 24 months is +1.3% on the short spans and neutral on the long
spans** — v7 had dropped recency weighting because the pooled replay
(early folds) said uniform was best by 0.002; that was the early-fold
artefact again. The junk months in training are harmless.

### v8 (this session's second half): half-life 24 + lag anchors + wide neighbour means

New in `model.py`: `lag4/5/6_delta` (TWS at L minus TWS at L−k; the test
has k=6 for the 179 block, 5 for 173, 4 for 168 and 202 — training rows
have every lag masked at random with p=0.5 so NaN is a learned state),
`nb5_/nb7_last_tws`, `nb5_/nb7_right_tws`, `interp_nb5`, `last_minus_nb`,
`right_minus_nb` (5×5 / 7×7 neighbour means and the local residual), and
recency weights on the training target month (`HALFLIFE = 24`). Caches were
rebuilt as `cache_late_v8_*` and `cache_replay_v8_b123_*`. All with
half-life 24 unless stated; logs `logs/replay_v8_ablate.log`,
`logs/v8_checks.log`, `logs/latecheck_v8_hl24.log`.

| variant | late check | fold 123 | pseudo-test |
|---|---|---|---|
| v7 features, uniform (reference on the new caches) | 0.7121 / 0.907 | 0.6783 / 0.936 | 0.6820 / 0.823 |
| v7 features + half-life 24 | 0.7033 / 0.896 | 0.6796 / 0.937 | 0.6844 / 0.825 |
| v8 without lag anchors | 0.7019 / 0.894 | 0.6770 / 0.934 | 0.6828 / 0.824 |
| v8 without wide neighbour means | 0.7007 / 0.893 | 0.6776 / 0.935 | 0.6820 / 0.823 |
| **v8 (all)** | **0.6996 / 0.891** | **0.6748 / 0.931** | **0.6818 / 0.822** |

v8 is −1.75% on the short blocks, −0.5% on fold 123 (the long block and
extrapolation configs improve: (5,14) 0.527→0.519, (7,12) 0.687→0.672,
(2,0) 0.731→0.712), and neutral on real labels. Nothing is worse anywhere.
The late-check gain is again concentrated on junk-anchor months (159, 163,
164: 0.68–0.74), so expect ~0.5–1% on the board, not 1.75%.

## Ranked next steps

1. ~~Submit the calibration files~~ done; see above.
2. ~~Submit `submission_v8.csv`~~ done: **0.730218** (−1.1% on v7; the checks
   had bracketed 0.5–1.75%). The selection protocol predicts the board.
3. **Keep selecting on the late checks only** (`latecheck.py`, fold 123,
   `pseudotest7.py`); treat the pooled replay as unmeasured. Excluding the
   2015 junk months is measured harmless — not a lever.
4. ~~Where the remaining headroom is~~ **measured 2026-09-04/06: nothing left
   in this direction.** Four candidate feature groups and two post-processing
   levers all came out inside the noise (section below). The model is
   signal-limited; see "Levers measured flat".

5. **Covariate availability at masked target months** (ERA5 precipitation /
   temperature from the Climate Data Store, permitted by the rules) — days of
   work, and the weak post-2011 coupling caps what it can return.
6. Seed/fold ensembling of the final refit. Cheap, variance-only.
7. Phase Two artifacts — `emissions.csv` (CodeCarbon) and `valid_preds.parquet`
   (replay-fold predictions with lat/lon) exist.

## Levers measured flat (2026-09-04 / 09-06)

Everything here was measured on the late-regime checks with the v8 model as
the reference, and every result sits inside the ±0.001 the checks reproduce
run to run. **The model is signal-limited: nothing about the estimator is
holding the score back.** Do not re-propose these without new evidence.

**v9 candidate features** (logs `logs/v9_seq.log`, `logs/v9_a.log`; code
removed after the measurement, the diff is in this commit's parent):

| variant | late check | fold 123 | pseudo-test |
|---|---|---|---|
| v8 reference | 0.6987 | 0.6751 | 0.6826 |
| all four groups | 0.6990 | 0.6760 | 0.6816 |
| cumulative SPEI_01 over the gap | 0.6985 | 0.6756 | |
| second right anchor (`rlag4_delta`) | 0.6993 | 0.6759 | |
| 7×7-smoothed target-month residuals | 0.6983 | | |
| per-cell slopes fitted on the last 36 months | 0.6983 | | |

The sign flips between checks; the whole spread is 0.0010.

**Seed averaging of the final refit** (`latecheck.py --fixed-rounds 380
--seeds N`, log `logs/seeds.log`): 1 seed 0.6996, 3 seeds 0.6997, 5 seeds
0.6995. Individual seeds span 0.7045–0.7056 on the unweighted eval, so there
*is* seed variance — averaging just does not convert it into pooled score.

**Spatially smoothing the prediction's departure from interpolation**
(`interp + (1−α)·dev + α·smooth(dev)`, log `logs/smoothpred.log`): best
0.6987 at α=0.5–0.75 with one 3×3 pass, against 0.6996 unsmoothed; two passes
are worse (0.7009). The 0.0009 is the same size as the noise above. The model
already reads 3×3/5×5/7×7 neighbour means, so its output is smooth already.

Earlier, on the (unreliable) pooled replay: hyperparameters, residual target,
per-dL blend, and training rows per config were all flat too. Five
independent kinds of lever now measure flat.

## The kriging attempt (2026-09-06) — a good baseline that the model already beats

The one structurally different idea tried after v8, and the only one that
moved a number outside the noise. Kept in `krige.py`; the estimator functions
`ar1_noise` and `blup_weights` live in `model.py`.

**The reasoning.** Every estimator here carries an *observed* anchor forward,
but the scored target is a noisy GRACE observation whose own noise is
unpredictable, so the best achievable prediction is the *signal* at the target
month. That predictor shrinks the anchors by each cell's noise fraction. Write
`y_t = s_t + e_t` with `s` an AR(1): the observed autocovariance is
`c_k = vs·phi^k` (k ≥ 1) and `c_0 = vs + ve`, so `phi = c2/c1`, `vs = c1/phi`,
`ve = c0 − vs` — three parameters per cell from lags 1–3, all dynamical
properties rather than levels. The kriging weights on the two anchors follow
in closed form; linear interpolation is the special case `ve = 0, phi = 1`.

**The fit is physically sensible.** Median `phi` 0.88; noise fraction 0.00 in
the Amazon, 0.13 in Siberia, 0.29 in the Sahara/Arabia.

**The striking result.** As a *pure baseline with no training at all*, the
BLUP scores **0.7280** on the late check against linear interpolation's
0.7848 — a 7.2% gain, where the whole trained model reaches 0.6996 (10.9%).
It improves every configuration, the extrapolation ones `(1,0)` and `(2,0)` by
10–13%. **Three parameters per cell recover two thirds of the model's skill
over interpolation.** That belongs in the Phase Two write-up: it is the
interpretable core of what the model is doing.

**But it does not improve the model,** because the model already beats it
everywhere (0.6996 < 0.7280). Late check, test-weighted:

| variant | pooled |
|---|---|
| v8 reference | 0.6996 |
| + all six kriging features | 0.7030 |
| + estimates only (no `phi`/`nfrac`) | 0.7030 |
| + `phi`/`nfrac` only | 0.6991 |
| + `blup`/`blup_dev` only | 0.7035 |
| learn the residual from the BLUP instead | 0.7063 |

The per-cell parameters are harmless; the **BLUP estimate itself is what
hurts**, and the reason is collinearity — it is a cell-weighted blend of
`last_tws` and `right_tws`, the two features that already top the importance
table, so it duplicates `interp` and splits the trees' capacity. Using it as a
residual baseline instead of a feature is worse again.

**Do not re-propose** kriging, shrinkage or per-cell noise features as model
inputs. Do use the BLUP as the interpretable baseline in the write-up.

## Problem structure — the non-obvious facts

- **Test has only 18 row months**, not 40: 164, 168–170, 173–176, 179–185,
  198, 202–203. Six are fully observed anchors; the other twelve are masked by
  whole month (417 stray observed cells, 0.15%). 88.8% of rows are bracketed.
- **Covariates exist only at those 18 months** in Test.csv. The 12–18-month
  masked block (rows 180–185, 39% of test) sees covariates for 179–185 and 198
  only.
- **Train has zero missing `TWS_t`**; the 22 missing train row months are the
  11 GRACE outages (labelled one month after the GRACE convention), and
  `target(m) == TWS_t(m+1)` exactly, so the panel `G` knows more months than
  `time` lists.
- **The field is extremely smooth spatially** (month-to-month change correlates
  0.999 with E–W neighbours): 15.7k cells are a few hundred independent
  samples, which is why data volume does nothing. GRACE noise lives at the
  mascon scale and above, not at 1°, except in the junk months (striping).
- Month index convention: `m = (year-2002)*12 + (month-1)`. Train spans 4–163,
  test targets 165–204.

## The rules decision

The competition page says *"For predictions at month t, only information
available at or before t may be used; future observed values must not be
used"* and permits Copernicus covariates *"provided that these variables are
available at the stated prediction time and do not directly or indirectly
include future GRACE/TWS information."* v5+ uses observed TWS at a later month
as a right anchor; v7 additionally uses SPEI/soil-moisture at the target month
(the row for that month in Test.csv). Both are values the organisers placed in
Test.csv; neither is external data. The user chose to build the right anchor
with the ambiguity in view; the target-month covariates are the same category
of choice and less strong. `model_causal_reference.py` (board 0.767) is the
strictly forward-looking version if a moderator rules otherwise. A top-10
finish would put this choice in front of the Phase Two reviewers.

## Practical constraints

- 7 GB RAM, 12 cores. `model.py` builds two replay folds (200k rows/config)
  then refits on 400k rows/config (5.9M rows × 87 features, 223 rounds); peak
  ~5 GB. Last run (v8): fold 68 0.5042, fold 123 0.6766 (interp 0.5736 /
  0.7249). `SMOKE=1 .venv/bin/python -u model.py` runs the whole pipeline on 5k
  rows/config in ~3 min — use it after any edit.
- `replay.py` caches feature frames per fold as `cache_replay_v7_b*_n*.parquet`;
  `latecheck.py` caches as `cache_late_c150_n100000_*.parquet` (gitignored,
  ~6 min to rebuild). An ablation on cached frames takes 1–3 min.
  `pseudotest7.py` has no cache (~4 min per variant).
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
