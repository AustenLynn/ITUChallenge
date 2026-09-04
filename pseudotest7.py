"""Pseudo-test on the v7 pipeline: the six observed test months, predicted from
their neighbouring anchors with model.py's own features and refit recipe, and
scored on the real 2015-2018 labels.  The only test-period ground truth there
is.  Spans are longer than most test configurations ((1,4) (4,5) (5,6) (6,19)
(19,4) (4,0)), so read it as evidence about the long block and extrapolation.

The month before each target is not a test row (except 163 for 164), so the
row-month covariate features are dropped, as pseudotest.py does; anchor and
target-month covariates are what the test provides and are kept.

    .venv/bin/python -u pseudotest7.py                       # v7 recipe
    .venv/bin/python -u pseudotest7.py --exclude-junk        # drop 2015 junk months from training
    .venv/bin/python -u pseudotest7.py --drop next,nLR,smooth,beta
    .venv/bin/python -u pseudotest7.py --train-from 108
"""
import argparse

import numpy as np
import pandas as pd
import lightgbm as lgb

import model as M
from replay import GROUPS

TARGETS = [164, 168, 173, 179, 198, 202]
JUNK = {157, 158, 159, 162, 163}          # 2015 months with 1-step RMS change 0.9-1.2 (normal: 0.5)
ROWMONTH = set(M.DYN + [f"{c}_d" for c in M.DYN] + [f"{c}_gmean" for c in M.DYN]
               + [f"{c}_gmin" for c in M.EXTREME] + [f"{c}_gmax" for c in M.EXTREME]
               + [f"{c}_d1" for c in M.LAG1] + ["SPEI_06_t_dR", "SOIL_MOISTURE_t_dR"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", default="")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--rounds", type=int, default=250)
    ap.add_argument("--train-from", type=int, default=0)
    ap.add_argument("--exclude-junk", action="store_true")
    ap.add_argument("--halflife", type=float, default=0.0)
    a = ap.parse_args()
    rng = np.random.default_rng(M.SEED)
    tr, te = M.load()
    cells = pd.Index(sorted(set(zip(tr.lat, tr.lon))))
    tr["cell_idx"] = cells.get_indexer(pd.MultiIndex.from_arrays([tr.lat, tr.lon]))
    te["cell_idx"] = cells.get_indexer(pd.MultiIndex.from_arrays([te.lat, te.lon]))
    G = M.build_panel(tr, te, cells)
    C = M.build_covariates(tr, te, len(cells), G.shape[1])
    P = M.Panels(cells, tr, G, C, beta_upto=164)
    drop = set(ROWMONTH)
    for d in filter(None, a.drop.split(",")):
        drop |= set(GROUPS.get(d, [d]))
    feats = [f for f in M.FEATS if f not in drop]

    known = sorted(set(tr.m.unique()) | set(tr.m.unique() + 1) | set(TARGETS))
    cfgs = {}
    for t in TARGETS:
        L = max(k for k in known if k < t); R = [k for k in known if k > t]
        cfgs[t] = (int(t - L), int(R[0] - t) if R else 0)
    print("configs:", cfgs, " features:", len(feats), " drop:", sorted(drop - ROWMONTH))

    src = tr[tr.m + 1 >= a.train_from]
    if a.exclude_junk:
        src = src[~(src.m + 1).isin(JUNK)]
    TRX, TRY, TRM = [], [], []
    for t, (dl, dr) in cfgs.items():
        ok = src[src.m + 1 - dl >= src.m.min()]
        if a.exclude_junk:      # anchors on junk months are as bad as junk targets
            ok = ok[~(ok.m + 1 - dl).isin(JUNK) & ~(ok.m + 1 + dr).isin(JUNK)]
        if len(ok) > a.n:
            ok = ok.iloc[rng.choice(len(ok), a.n, replace=False)]
        X, g = M.make_features(ok, dl, dr or None, P)
        TRX.append(M.mask_lags(X[g].reset_index(drop=True), rng)[feats]); TRY.append(ok.target.to_numpy()[g]); TRM.append(ok.m.to_numpy()[g] + 1)
    TRX = pd.concat(TRX, ignore_index=True); TRY = np.concatenate(TRY); TRM = np.concatenate(TRM)
    W = 0.5 ** ((TRM.max() - TRM) / a.halflife) if a.halflife > 0 else None
    print(f"train {TRX.shape}  target months {TRM.min()}..{TRM.max()}")
    mdl = lgb.train(M.PARAMS, lgb.Dataset(TRX, TRY, weight=W), num_boost_round=a.rounds)

    obs = te[te.TWS_t.notna()]
    rm = lambda e: float(np.sqrt(np.mean(e ** 2)))
    se_m = se_i = 0.0; n = 0
    for t, (dl, dr) in cfgs.items():
        o = obs[obs.m == t][["cell_idx", "lat", "lon"]].copy(); o["m"] = t - 1
        srcm = tr[tr.m == t - 1].set_index("cell_idx") if (t - 1) in set(tr.m) else None
        for c in M.DYN:
            o[c] = srcm[c].reindex(o["cell_idx"]).to_numpy() if srcm is not None else np.nan
        X, g = M.make_features(o, dl, dr or None, P)
        X = X[g]; y = G[o["cell_idx"].to_numpy(), t][g]
        p = mdl.predict(X[feats]); ip = M.interp_base(X)
        print(f"  t={t} ({2002 + t // 12}-{t % 12 + 1:02d}) cfg {(dl, dr)} n={len(y):,}: model {rm(y - p):.4f}  interp {rm(y - ip):.4f}  ratio {rm(y - p) / rm(y - ip):.3f}")
        se_m += np.sum((y - p) ** 2); se_i += np.sum((y - ip) ** 2); n += len(y)
    print(f"  pooled: model {np.sqrt(se_m / n):.4f}  interp {np.sqrt(se_i / n):.4f}  ratio {np.sqrt(se_m / se_i):.3f}")


if __name__ == "__main__":
    main()
