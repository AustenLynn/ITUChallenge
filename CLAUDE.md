# ITUChallenge — Zindi global water storage forecasting

**Read `HANDOFF.md` before starting work.** It carries the current standing, the
transfer diagnosis, the approaches already measured on the honest metric, and
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

Measure it on `replay.py` (test-layout replay, cached folds, `--drop` /
`--halflife` / `--perh` flags), not on an adjacent-window validation. The
24-month validation used through v6 rewarded memorisation and was the reason
offline gains failed to reach the leaderboard. `HANDOFF.md` lists what has
already been measured flat or worse on the replay.
