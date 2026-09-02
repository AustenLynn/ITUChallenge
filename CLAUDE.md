# ITUChallenge — Zindi global water storage forecasting

**Read `HANDOFF.md` before starting work.** It carries the current standing, the
open transfer problem, the approaches already measured and rejected, and the
rules decision behind the current model. `README.md` has the results tables.

## Operating notes

- Run Python with `-u`; redirected output is otherwise buffered and logs look empty.
- Environment is `uv`-managed: `.venv/bin/python`. `python3 -m venv` fails here.
- 7 GB RAM total and the main run peaks at 6 GB — check memory before scaling
  `N_PER_CFG` or `N_PER_H`.
- `.venv/bin/python model.py` writes `submission.csv` (~15 min).
- `data/` is gitignored; download Train.csv, Test.csv and SampleSubmission.csv
  from Zindi into it.

## Before proposing a modelling idea

Check the "measured and REJECTED" section of `HANDOFF.md` first. Five
plausible-sounding approaches have already been tested and lost, with numbers.
