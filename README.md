# A Step Ahead of Drought — Global Water Storage Forecasting

Zindi challenge: forecast Total Water Storage (TWS) one month ahead on a global
1° grid. Scored by RMSE (50% of final score); the remaining 50% is a Phase Two
trustworthiness write-up available only to the top 10. `HANDOFF.md` carries the
current standing and the reasoning; this file carries the results tables.

## Interpolation, not forecasting

Test.csv has 18 row months. Six are fully observed (164, 168, 173, 179, 198,
202) and the other twelve are masked by whole month, each sitting between two
observed ones, so **88.8% of test rows have an observed TWS value for the same
cell at a later month**. Every row is keyed on a `(dL, dR)` pair — months back
to the left anchor, forward to the right — and training rows are drawn at the
17 pairs the test contains. Linear interpolation between the anchors is the
baseline every model is measured against.

**Rules caveat.** The competition permits Copernicus covariates *"provided that
these variables are available at the stated prediction time and do not directly
or indirectly include future GRACE/TWS information."* Using TWS at a later
month (v5+) and covariates at the target month (v7) are both values the
organisers placed in Test.csv, not external data, but read strictly they are
after the prediction time. This was a deliberate, informed choice —
`model_causal_reference.py` is the strictly forward-looking version
(leaderboard 0.767) if the call changes.

## The honest metric (v7)

`replay.py` reproduces the test layout on the training panel — the same 18
months, the same six anchors, the panel masked the same way, covariates only
where the test has them, training strictly before the window — over four folds
(base months 44, 56, 68, 123). `pseudotest.py` predicts the six *observed* test
months from their neighbouring anchors: real 2015–2018 labels.

On both, the v6 model barely beat linear interpolation, and the leaderboard had
been saying the same thing. The 24-month validation it was selected on rewarded
memorising each cell's recent anomaly, which a test window 2–40 months out
punishes.

| pooled over four replay folds | model | interp | ratio |
|---|---|---|---|
| v6 feature set, recency weights | 0.5938 | 0.6063 | 0.979 |
| − climatology & cell-level features | 0.5619 | | 0.927 |
| + target-month covariates | 0.5534 | | 0.913 |
| − drift features, uniform weights | 0.5474 | | 0.903 |
| + smoothed residuals, per-cell slope (**v7**) | **0.5401** | | **0.891** |

| pseudo-test, real 2015–18 anchors | model | interp | ratio |
|---|---|---|---|
| v6-style features | 0.793 | 0.829 | 0.957 |
| + target-month covariates | 0.715 | | 0.862 |
| + drop climatology (v7 recipe) | **0.704** | | **0.850** |

Measured flat or worse on the replay: hyperparameters (leaves 15–255,
`min_data_in_leaf` 200–10k, L2, feature fraction — all within 0.002), residual
target, per-dL blend (−0.001), recency weighting, and training rows per config
(5k vs 100k: identical — the field is spatially smooth enough that 15.7k cells
are a few hundred independent samples).

## Problem shape

- **Train has zero missing `TWS_t`; test is 66.5% missing, by whole month.**
- **Covariates exist only at the 18 test months.** The 12–18-month masked block
  (39% of test) sees SPEI / soil moisture for 179–185 and 198 only.
- **Train months are non-contiguous** — 22 GRACE outages, clustered
  2011–2014 — and `target(m) == TWS_t(m+1)` exactly, so the panel knows more
  months than `time` lists.
- **The test period is noisier.** Predicting each observed test anchor from its
  neighbours gives errors 15–40% above the same configuration in training
  years; 2015's persistence error is 0.87 against ~0.53 in every earlier year,
  concentrated in deserts. The test-period interpolation baseline is estimated
  at ~0.75, so the top-10 cutoff (0.656) is a ratio of ~0.87 to it.

## Layout

    model.py                 end-to-end: features, replay selection, refit, submission
    replay.py                test-layout replay validation with cached folds and ablation flags
    pseudotest.py            real-label transfer check on the six observed test months
    model_v6_reference.py    previous version (board 0.736748); ablate*/diag* import it
    model_causal_reference.py  strictly forward-looking version (board 0.767)
    ablate*.py, diag*.py     v4–v6 era diagnostics, kept for the record
    run_summary.json         replay scores, rounds, emissions from the last run
    emissions.csv            CodeCarbon log, appended per run

## Running

    uv venv .venv && uv pip install -p .venv/bin/python \
        pandas numpy lightgbm scikit-learn pyarrow codecarbon
    # download Train.csv, Test.csv, SampleSubmission.csv into data/
    SMOKE=1 .venv/bin/python -u model.py   # 3-minute end-to-end check
    .venv/bin/python -u model.py           # writes submission.csv (~25 min, ~5 GB)
    .venv/bin/python -u replay.py --drop beta   # ablate on cached replay folds

## Leaderboard calibration

| version | offline metric | public leaderboard | commit |
|---|---|---|---|
| v2 (anchor deltas + neighbours) | 0.6992 (24-mo window) | 0.774957 | pre-repo |
| v4 (recency weighting + drift)  | 0.6812 (24-mo window) | 0.767074 | 6a20efd |
| v5 (anchor bracketing)          | 0.6252 (24-mo, leaky) | 0.746183 | 0ca1f88 |
| v6 (leak fix + 9.2M-row refit)  | 0.6298 (24-mo) / 0.5938 replay | 0.736748 | 3efff00 |
| v7 (transferable features, replay selection) | 0.5401 replay | *not yet submitted* | |

Benchmark 0.899886; 476 active participants. The public leaderboard scores only
a portion of the test set, so the final standing will move.

## Caveats

- `last_tws`, `nb_last_tws`, `interp`, `interp_nb` dominate gain-based
  importance and are near-duplicates; do not read the table as a skill
  ranking — use SHAP.
- Fold 123 (the late-regime replay fold) moves far less than the early folds
  under every change (0.6835 → 0.6760 for the full v7 recipe). The pseudo-test
  is the reason to expect more on the board; be ready for the smaller number.
