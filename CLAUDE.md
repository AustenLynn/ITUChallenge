# ITUChallenge — Zindi global water storage forecasting

**Read `HANDOFF.md` before starting work.** It carries the current standing, the
transfer diagnosis, the approaches already measured on the honest metrics, and
the rules decision behind the current model. `README.md` has the results tables.

## Operating notes

- Run Python with `-u`; redirected output is otherwise buffered and logs look empty.
- Environment is `uv`-managed: `.venv/bin/python`. `python3 -m venv` fails here.
- 7 GB RAM total; the full `model.py` run peaks ~5 GB. `SMOKE=1` runs the whole
  pipeline on tiny samples in ~3 min — use it after any edit, before a full run.
- `.venv/bin/python -u model.py` writes `submission.csv` (~25 min).
- `data/` is gitignored; download Train.csv, Test.csv and SampleSubmission.csv
  from Zindi into it.
- Background shell waits longer than ~10 min get killed; poll the log file.

## Before proposing a modelling idea

Measure it on the **late-regime** checks — `latecheck.py` (2015 short blocks,
cached), `replay.py --folds 123`, and `pseudotest7.py` (real test-period
labels) — not on the pooled replay and not on an adjacent-window validation.
The pooled replay is dominated by folds 44/56/68, which sit in a 2005–2011
regime where covariates couple to TWS twice as strongly as they do in the test
years; v7 was selected on it and did not move the board. `HANDOFF.md` lists
what has already been measured on each check.

TWS is deseasonalised: never add climatology features, and do not propose
seasonal interpolation.
