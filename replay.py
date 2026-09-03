"""
Replay validation: reproduce the test set's exact layout on the training panel.

Why this exists. The 24-month config-keyed validation that v5/v6 used drew rows
for every (dL, dR) pair from the last 24 train months, but a right anchor
12..18 months out only fits inside that window for three winter months right
after the training cutoff, so 39% of the test-weighted metric was scored on
the easiest, shortest-lead rows available. It also computed every feature from
a dense panel, while the test panel has whole-month holes. On that metric the
v6 model beat linear interpolation by 10%; on this one, by 2% -- and the
leaderboard agreed with this one.

A fold is a base month b. The 41-month window [b, b+40] is masked to the six
anchor offsets the test has (0, 4, 9, 15, 34, 38); covariates are kept only at
the 18 row offsets the test has; training uses months < b only; the replay
rows are the train rows at those 18 months, scored on their true targets. Each
month contributes ~15.6k rows, so plain RMSE over the fold already carries the
test's configuration mix.

Folds must have every anchor known and every row month present. b = 44, 56,
68 are fully clean (dense 2003-2010 panel) and share the test's calendar
(b == 164 mod 12); b = 123 is the closest late-regime fold but loses rows at
GRACE outages. Feature frames are cached per fold so ablations are cheap.

    .venv/bin/python -u replay.py                         # current model.FEATS
    .venv/bin/python -u replay.py --drop beta,smooth      # ablate groups/features
    .venv/bin/python -u replay.py --halflife 15 --perh    # old weighting, per-dL blend

Results so far (pooled RMSE over the four folds, model / interp):
    v6 feature set                       0.5938 / 0.6063  (ratio 0.979)
    - climatology & cell-level features  0.5619           (0.927)
    + target-month covariates            0.5534           (0.913)
    - drift features, uniform weights    0.5474           (0.903)
    + smoothed residuals, per-cell slope 0.5401           (0.891)   <- v7
    hyperparameters (leaves 15..255, min_data 200..10k, l2, ff): flat
    residual target (y - interp): worse; per-dL blend: -0.001, not worth it
"""
import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import lightgbm as lgb

import model as M

CLIM = ["clim_m", "clim_L", "clim_delta", "cell_mean", "cell_std"]
GROUPS = {
    "delta": ["last_delta"] + [f"{c}_d1" for c in M.LAG1],
    "path": [f"{c}_d" for c in M.DYN] + [f"{c}_gmean" for c in M.DYN]
            + [f"{c}_gmin" for c in M.EXTREME] + [f"{c}_gmax" for c in M.EXTREME],
    "dyn": list(M.DYN),
    "anom": ["last_anom", "nb_last_anom", "right_anom"],
    "nb": ["nb_last_tws", "nb_last_anom", "nb_right_tws", "interp_nb"],
    "latlon": ["lat", "lon"],
    "next": [f"{c}_n" for c in M.DYN] + [f"{c}_nres" for c in M.DYN]
            + [f"{c}_rmean" for c in ("SPEI_03_t", "SOIL_MOISTURE_t")],
    "nLR": [f"{c}_nL" for c in M.DYN] + [f"{c}_nR" for c in M.DYN],
    "smooth": [f"{c}_nres_s3" for c in M.SMOOTHED] + [f"{c}_nres_s5" for c in M.SMOOTHED],
    "beta": ["beta_sm", "beta_sm_nres", "beta_sm_nres_s3", "beta_spei3", "beta_spei3_nres"],
}


def build_fold(b, tr, cells, G_full, C_full, cfgs, n_per_cfg, rng):
    n_m = G_full.shape[1]
    G = G_full.copy()
    keep = np.zeros(n_m, bool); keep[:b] = True
    keep[[b + o for o in M.OFFS_ANCH]] = True
    G[:, ~keep] = np.nan
    C = {c: v.copy() for c, v in C_full.items()}
    ckeep = np.zeros(n_m, bool); ckeep[:b] = True
    ckeep[[b + o for o in M.OFFS_ROWS]] = True
    for c in C:
        C[c][:, ~ckeep] = np.nan
    src = tr[tr.m < b]
    P = M.Panels(cells, src, G, C, beta_upto=b)

    rep = tr[tr.m.isin([b + o for o in M.OFFS_ROWS])]
    rep = rep[np.isfinite(G_full[rep["cell_idx"].to_numpy(), rep.m.to_numpy() + 1])]
    rdL, rdR = M.anchors_for(np.isfinite(G), rep["cell_idx"].to_numpy(), rep.m.to_numpy() + 1, n_m)
    cnt = Counter(zip(rdL.tolist(), rdR.tolist()))
    present = sorted(c for c, n in cnt.items() if n >= 4000 and c in set(cfgs))
    print(f"fold b={b}: {len(rep):,} candidate rows over {rep.m.nunique()} months; configs {present}")
    TRX, TRY, TRM, VAX, VAY, VACFG = [], [], [], [], [], []
    for (dl, dr) in present:
        ok = src[src.m + 1 - dl >= src.m.min()]
        if len(ok) > n_per_cfg:
            ok = ok.iloc[rng.choice(len(ok), n_per_cfg, replace=False)]
        X, g = M.make_features(ok, dl, dr or None, P)
        TRX.append(X[g]); TRY.append(ok.target.to_numpy()[g]); TRM.append(ok.m.to_numpy()[g])
        k = (rdL == dl) & (rdR == dr)
        X, g = M.make_features(rep[k], dl, dr or None, P)
        VAX.append(X[g]); VAY.append(rep[k].target.to_numpy()[g])
        VACFG.append(np.full(int(g.sum()), f"{dl},{dr}"))
    TRX = pd.concat(TRX, ignore_index=True); VAX = pd.concat(VAX, ignore_index=True)
    TRX["_y"], TRX["_m"] = np.concatenate(TRY), np.concatenate(TRM)
    VAX["_y"], VAX["_cfg"] = np.concatenate(VAY), np.concatenate(VACFG)
    VAX["_interp"] = M.interp_base(VAX)
    print(f"  train {TRX.shape}  replay {VAX.shape}")
    return TRX, VAX


def evaluate(TRX, VAX, feats, params, rounds, tag, verbose=True, halflife=0.0, resid=False, perh=False):
    VAY, VAI = VAX["_y"].to_numpy(), VAX["_interp"].to_numpy()
    rm = lambda p: float(np.sqrt(np.mean((VAY - p) ** 2)))
    TM = TRX["_m"].to_numpy()
    W = 0.5 ** ((TM.max() - TM) / halflife) if halflife > 0 else np.ones(len(TM))
    ytr, off_va = TRX["_y"].to_numpy(), 0.0
    if resid:
        ytr, off_va = ytr - M.interp_base(TRX), VAI
    mdl = lgb.train(params, lgb.Dataset(TRX[feats], ytr, weight=W), num_boost_round=rounds,
                    valid_sets=[lgb.Dataset(VAX[feats], VAY - off_va)],
                    callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    p = mdl.predict(VAX[feats]) + off_va
    if perh:
        hp = np.empty(len(VAX)); htr, hva = TRX["h"].to_numpy(), VAX["h"].to_numpy()
        for h in np.unique(hva):
            ktr, kva = htr == h, hva == h
            hm = lgb.train(params, lgb.Dataset(TRX[feats][ktr], ytr[ktr], weight=W[ktr]), num_boost_round=rounds,
                           valid_sets=[lgb.Dataset(VAX[feats][kva], (VAY - off_va)[kva])],
                           callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
            hp[kva] = hm.predict(VAX[feats][kva]) + (off_va[kva] if resid else 0.0)
        print(f"  [{tag}] per-h {rm(hp):.4f}  blend {rm(0.5 * (p + hp)):.4f}  (global {rm(p):.4f})")
        p = 0.5 * (p + hp)
    print(f"  [{tag}] model {rm(p):.4f}  interp {rm(VAI):.4f}  ratio {rm(p) / rm(VAI):.3f}  best_iter {mdl.best_iteration}")
    if verbose:
        for c in sorted(set(VAX["_cfg"])):
            k = (VAX["_cfg"] == c).to_numpy()
            print(f"    cfg {c:>6}: n={k.sum():>6}  model {np.sqrt(np.mean((VAY[k]-p[k])**2)):.4f}"
                  f"  interp {np.sqrt(np.mean((VAY[k]-VAI[k])**2)):.4f}")
    return rm(p), rm(VAI), len(VAY), float(np.sum((VAY - p) ** 2)), float(np.sum((VAY - VAI) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", default="", help="comma-separated feature names or group names to drop")
    ap.add_argument("--folds", default="44,56,68,123")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--rounds", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--leaves", type=int, default=None)
    ap.add_argument("--mdl", type=int, default=None, help="min_data_in_leaf")
    ap.add_argument("--ff", type=float, default=None, help="feature_fraction")
    ap.add_argument("--l2", type=float, default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--halflife", type=float, default=0.0, help="recency half-life in months; 0 = uniform")
    ap.add_argument("--resid", action="store_true", help="learn target - interp")
    ap.add_argument("--perh", action="store_true", help="also fit per-dL models and blend 50/50")
    a = ap.parse_args()
    drop = set()
    for d in filter(None, a.drop.split(",")):
        drop |= set(GROUPS.get(d, [d]))
    feats = [f for f in M.FEATS if f not in drop]
    params = dict(M.PARAMS)
    for k, v in (("learning_rate", a.lr), ("num_leaves", a.leaves), ("min_data_in_leaf", a.mdl),
                 ("feature_fraction", a.ff), ("lambda_l2", a.l2)):
        if v is not None:
            params[k] = v
    print(f"features: {len(feats)}  drop={sorted(drop)}  halflife={a.halflife}  resid={a.resid}")
    print("params:", {k: params[k] for k in ("learning_rate", "num_leaves", "min_data_in_leaf", "feature_fraction", "lambda_l2")})
    tot_m = tot_i = 0.0; tot_n = 0; res = []
    ctx = None
    for b in [int(x) for x in a.folds.split(",")]:
        cache = f"cache_replay_v7_b{b}_n{a.n}"
        if os.path.exists(cache + "_tr.parquet"):
            TRX, VAX = pd.read_parquet(cache + "_tr.parquet"), pd.read_parquet(cache + "_va.parquet")
            print(f"fold b={b}: loaded cache  train {TRX.shape}  replay {VAX.shape}")
        else:
            if ctx is None:
                rng = np.random.default_rng(M.SEED)
                tr, te = M.load()
                cells = pd.Index(sorted(set(zip(tr.lat, tr.lon))))
                tr["cell_idx"] = cells.get_indexer(pd.MultiIndex.from_arrays([tr.lat, tr.lon]))
                te["cell_idx"] = cells.get_indexer(pd.MultiIndex.from_arrays([te.lat, te.lon]))
                G_full = M.build_panel(tr, te, cells)
                C_full = M.build_covariates(tr, te, len(cells), G_full.shape[1])
                dL, dR = M.anchors_for(np.isfinite(G_full), te["cell_idx"].to_numpy(),
                                       te["m"].to_numpy() + 1, G_full.shape[1])
                cfgs = sorted(c for c, n in Counter(zip(dL.tolist(), dR.tolist())).items() if n >= 4000)
                ctx = True
            TRX, VAX = build_fold(b, tr, cells, G_full, C_full, cfgs, a.n, rng)
            TRX.to_parquet(cache + "_tr.parquet"); VAX.to_parquet(cache + "_va.parquet")
        m, i, n, sm, si = evaluate(TRX, VAX, feats, params, a.rounds, f"b={b}", verbose=not a.quiet,
                                   halflife=a.halflife, resid=a.resid, perh=a.perh)
        res.append((b, m, i)); tot_m += sm; tot_i += si; tot_n += n
    print("\nsummary (fold: model / interp / ratio):")
    for b, m, i in res:
        print(f"  b={b:3d}: {m:.4f} / {i:.4f} / {m / i:.3f}")
    print(f"  pooled: {np.sqrt(tot_m / tot_n):.4f} / {np.sqrt(tot_i / tot_n):.4f} / "
          f"{np.sqrt(tot_m / tot_i):.3f}")


if __name__ == "__main__":
    main()
