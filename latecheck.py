"""Late-regime check: the test's short-span block structure replayed inside 2015.

Why. The replay folds (base months 44/56/68/123) put the 41-month test layout on
the training panel, but only fold 123 touches the years in which (a) GRACE noise
is elevated and (b) the coupling between covariate changes and TWS changes is
half what it was in 2003-2011.  Three of the four folds are in the strong
regime, and they carried the v7 gains that the board did not show.  2015 (months
152-164) is the last year with labels and is in the test regime, but the full
layout does not fit in it.  The short blocks do:

    type A  rows b         right anchor b+4   (test block 164 / 198)   -> (1,3)
    type B  rows b..b+2    right anchor b+5   (test block 168)         -> (1,4) (2,3) (3,2)
    type C  rows b..b+3    right anchor b+6   (test block 173)         -> (1,5) (2,4) (3,3) (4,2)
    type E  rows b, b+1    no right anchor    (test block 202)         -> (1,0) (2,0)

Those ten configurations are 61% of the test rows.  For every base month b in
2015 the panel is masked exactly as the test would be: TWS known up to b-3, at
b and at the right anchor; covariates known up to b-3, at the block's row
months, and at the right anchor's block.  Training uses rows with targets at
or before --cut (default 150, i.e. through 2014) built on the dense panel, as
model.py does for the final refit.

    .venv/bin/python -u latecheck.py                    # v7 features
    .venv/bin/python -u latecheck.py --drop next,nLR,smooth,beta
    .venv/bin/python -u latecheck.py --train-from 108   # train on 2011+ only
"""
import argparse
import os

import numpy as np
import pandas as pd
import lightgbm as lgb

import model as M
from replay import GROUPS

BLOCKS = {"A": (1, 4), "B": (3, 5), "C": (4, 6), "E": (2, None)}   # n rows, right-anchor offset
GAP = 2           # months masked before each anchor (test: 2-3)
FIRST, LAST = 152, 164


def build(cut, n_per_cfg, rng):
    tr, te = M.load()
    cells = pd.Index(sorted(set(zip(tr.lat, tr.lon))))
    tr["cell_idx"] = cells.get_indexer(pd.MultiIndex.from_arrays([tr.lat, tr.lon]))
    te["cell_idx"] = cells.get_indexer(pd.MultiIndex.from_arrays([te.lat, te.lon]))
    G_full = M.build_panel(tr, te, cells)
    C_full = M.build_covariates(tr, te, len(cells), G_full.shape[1])
    n_m = G_full.shape[1]
    src = tr[tr.m < cut]

    # ---- training frame: dense panel restricted to months <= cut ----
    G_tr = G_full.copy(); G_tr[:, cut + 1:] = np.nan
    C_tr = {c: v.copy() for c, v in C_full.items()}
    for c in C_tr:
        C_tr[c][:, cut + 1:] = np.nan
    P_tr = M.Panels(cells, src, G_tr, C_tr, beta_upto=cut)
    cfgs = sorted({(d + 1, (s - d - 1) if s else 0) for n, s in BLOCKS.values() for d in range(n)})
    TRX, TRY, TRM = [], [], []
    for (dl, dr) in cfgs:
        ok = src[src.m + 1 - dl >= src.m.min()]
        if len(ok) > n_per_cfg:
            ok = ok.iloc[rng.choice(len(ok), n_per_cfg, replace=False)]
        X, g = M.make_features(ok, dl, dr or None, P_tr)
        TRX.append(M.mask_lags(X[g].reset_index(drop=True), rng)); TRY.append(ok.target.to_numpy()[g]); TRM.append(ok.m.to_numpy()[g] + 1)
    TRX = pd.concat(TRX, ignore_index=True)
    TRX["_y"], TRX["_m"] = np.concatenate(TRY), np.concatenate(TRM)   # _m = target month
    del P_tr

    # ---- evaluation blocks in 2015, each on its own masked panel ----
    VA = []
    for name, (n, s) in BLOCKS.items():
        last_b = LAST - (s if s else n)
        for b in range(FIRST, last_b + 1):
            R = b + s if s else None
            keep = np.zeros(n_m, bool); keep[:b - GAP] = True; keep[b] = True
            if R is not None: keep[R] = True
            G = G_full.copy(); G[:, ~keep] = np.nan
            ckeep = np.zeros(n_m, bool); ckeep[:b - GAP] = True; ckeep[b:b + n] = True
            if R is not None: ckeep[R:min(R + 4, LAST + 1)] = True
            C = {c: v.copy() for c, v in C_full.items()}
            for c in C:
                C[c][:, ~ckeep] = np.nan
            P = M.Panels(cells, src, G, C, beta_upto=b)
            for d in range(n):
                m = b + d
                rows = tr[tr.m == m]
                if len(rows) == 0 or not np.isfinite(G_full[rows["cell_idx"].to_numpy(), m + 1]).any():
                    continue
                dl, dr = d + 1, (R - (m + 1)) if R else None
                X, g = M.make_features(rows, dl, dr, P)
                X = X[g].copy(); X["_y"] = rows.target.to_numpy()[g]
                X["_cfg"] = f"{dl},{dr or 0}"; X["_tgt"] = m + 1; X["_block"] = f"{name}{b}"
                VA.append(X)
            del P
    VAX = pd.concat(VA, ignore_index=True)
    VAX["_interp"] = M.interp_base(VAX)
    return TRX, VAX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", default="")
    ap.add_argument("--cut", type=int, default=150)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--rounds", type=int, default=4000)
    ap.add_argument("--train-from", type=int, default=0, help="use only training rows with target month >= this")
    ap.add_argument("--halflife", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, default=1, help="average this many seeds (needs --fixed-rounds)")
    ap.add_argument("--fixed-rounds", type=int, default=0,
                    help="train exactly this many rounds with no early stopping; required for a clean --seeds comparison")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    cache = f"cache_late_v8_c{a.cut}_n{a.n}"
    if os.path.exists(cache + "_tr.parquet"):
        TRX, VAX = pd.read_parquet(cache + "_tr.parquet"), pd.read_parquet(cache + "_va.parquet")
    else:
        TRX, VAX = build(a.cut, a.n, np.random.default_rng(M.SEED))
        TRX.to_parquet(cache + "_tr.parquet"); VAX.to_parquet(cache + "_va.parquet")
    drop = set()
    for d in filter(None, a.drop.split(",")):
        drop |= set(GROUPS.get(d, [d]))
    feats = [f for f in M.FEATS if f not in drop]
    k = TRX["_m"].to_numpy() >= a.train_from
    W = 0.5 ** ((TRX["_m"].to_numpy()[k].max() - TRX["_m"].to_numpy()[k]) / a.halflife) if a.halflife > 0 else None
    print(f"train {k.sum():,} rows (from month {a.train_from})  eval {len(VAX):,} rows  features {len(feats)}  drop={sorted(drop)}  halflife={a.halflife}")
    y, ip = VAX["_y"].to_numpy(), VAX["_interp"].to_numpy()
    rm = lambda p, s=slice(None): float(np.sqrt(np.mean((y[s] - p[s]) ** 2)))
    ds = lgb.Dataset(TRX.loc[k, feats], TRX["_y"].to_numpy()[k], weight=W, free_raw_data=False)
    if a.fixed_rounds:
        # No early stopping: every seed sees the same budget, so an averaged
        # prediction is not also picking its stopping point off the eval set.
        preds, best = [], a.fixed_rounds
        for s in range(a.seeds):
            m = lgb.train(dict(M.PARAMS, seed=M.SEED + s), ds, num_boost_round=a.fixed_rounds)
            preds.append(m.predict(VAX[feats]))
            print(f"    seed {s}: {rm(preds[-1]):.4f}")
        p = np.mean(preds, axis=0)
    else:
        mdl = lgb.train(M.PARAMS, ds, num_boost_round=a.rounds,
                        valid_sets=[lgb.Dataset(VAX[feats], y)],
                        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        p, best = mdl.predict(VAX[feats]), mdl.best_iteration
    # test-weighted pooled score: weight each config by its share of test rows
    TEST_N = {"1,3": 31105, "1,4": 15683, "1,5": 15586, "2,3": 15582, "2,4": 15559, "3,2": 15547, "3,3": 15483,
              "4,2": 15409, "1,0": 15715, "2,0": 15624}
    cfg = VAX["_cfg"].to_numpy()
    wm = wi = wn = 0.0
    print(f"  {'cfg':>5} {'n':>7} {'model':>7} {'interp':>7} {'ratio':>6}")
    for c in sorted(TEST_N):
        s = cfg == c
        if not s.any(): continue
        wm += TEST_N[c] * np.mean((y[s] - p[s]) ** 2); wi += TEST_N[c] * np.mean((y[s] - ip[s]) ** 2); wn += TEST_N[c]
        print(f"  {c:>5} {s.sum():7d} {rm(p, s):7.4f} {rm(ip, s):7.4f} {rm(p, s) / rm(ip, s):6.3f}")
    print(f"  pooled (test-weighted): model {np.sqrt(wm / wn):.4f}  interp {np.sqrt(wi / wn):.4f}  ratio {np.sqrt(wm / wi):.3f}  rounds {best}  seeds {a.seeds}")
    if not a.quiet:
        print("  by target month:")
        for t in sorted(set(VAX["_tgt"])):
            s = (VAX["_tgt"] == t).to_numpy()
            print(f"    {t} {2002 + t // 12}-{t % 12 + 1:02d}: n={s.sum():6d}  model {rm(p, s):.4f}  interp {rm(ip, s):.4f}  ratio {rm(p, s) / rm(ip, s):.3f}")


if __name__ == "__main__":
    main()
