"""
Pseudo-test: the only real test-period labels we have.

Test.csv's six fully observed months (164, 168, 173, 179, 198, 202) can each be
predicted from their neighbouring anchors exactly the way the masked months
are, and scored against the truth. That measures how much of the model's gain
over linear interpolation survives in 2015-2018 -- the transfer ratio the
leaderboard has been hiding.

The month before each of those targets is not in Test.csv (except for 164), so
covariates at m = tgt-1 and the covariate path over the gap are unavailable.
This script therefore uses a reduced feature set that only reads covariates at
the anchors and at the target month itself, which both train and pseudo-test
rows have. It reports the same reduced model on the standard 2013-2015
validation window, so the two ratios are comparable.

    .venv/bin/python -u pseudotest.py
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

import model_v6_reference as M   # v6 feature API; diagnostic only

TARGETS = [164, 168, 173, 179, 198, 202]
DROP = set(M.DYN + [f"{c}_d" for c in M.DYN] + [f"{c}_gmean" for c in M.DYN]
           + [f"{c}_gmin" for c in M.EXTREME] + [f"{c}_gmax" for c in M.EXTREME]
           + [f"{c}_d1" for c in M.LAG1] + ["SPEI_06_t_dR", "SOIL_MOISTURE_t_dR"])
BASE = [f for f in M.FEATS if f not in DROP]
ANCH = ([f"{c}_L" for c in M.DYN] + [f"{c}_n" for c in M.DYN] + [f"{c}_R" for c in M.DYN]
        + [f"{c}_nres" for c in M.DYN])


def anchor_cov(rows, h, dR, C):
    ci, m = rows["cell_idx"].to_numpy(), rows["m"].to_numpy()
    tgt, L = m + 1, m + 1 - h
    out = {}
    for c in M.DYN:
        vL, vn = C[c][ci, L], C[c][ci, tgt]
        out[f"{c}_L"], out[f"{c}_n"] = vL, vn
        if dR is None:
            out[f"{c}_R"] = np.full(len(rows), np.nan, np.float32)
            out[f"{c}_nres"] = vn - vL
        else:
            w = h / (h + dR)
            vR = C[c][ci, tgt + dR]
            out[f"{c}_R"] = vR
            out[f"{c}_nres"] = vn - ((1 - w) * vL + w * vR)
    return pd.DataFrame(out).astype(np.float32)


def feats_for(rows, dl, dr, P, C, with_anchor):
    X, bs, _ = M.make_features(rows, dl, dr, *P)
    if with_anchor:
        X = pd.concat([X, anchor_cov(rows, dl, dr, C)], axis=1)
    g = np.isfinite(bs)
    if dr:
        g &= np.isfinite(X["right_tws"].to_numpy())
    return X, g


def main():
    rng = np.random.default_rng(M.SEED)
    tr, te = M.load()
    cells = pd.Index(sorted(set(zip(tr.lat, tr.lon))))
    for df in (tr, te):
        df["cell_idx"] = cells.get_indexer(pd.MultiIndex.from_arrays([df.lat, df.lon]))
    G = M.build_panel(tr, te, cells)
    C = M.build_covariates(tr, te, len(cells), G.shape[1])
    DM, DS, DN = M.drift_panels(G)
    val_start = tr.m.max() - M.VALID_MONTHS + 1

    def panels(src):
        clim, cmean, cstd = M.climatology(src, cells)
        NB, NBA = M.neighbour_panels(cells, G, clim)
        return (G, clim, cmean, cstd, C, NB, NBA, DM, DS, DN)
    P_sel, P_full = panels(tr[tr.m < val_start]), panels(tr)

    known = sorted(set(tr.m.unique()) | set(tr.m.unique() + 1) | set(TARGETS))
    cfgs = {}
    for t in TARGETS:
        L = max(k for k in known if k < t); R = [k for k in known if k > t]
        cfgs[t] = (t - L, (R[0] - t) if R else 0)
    print("pseudo-test configs:", cfgs)

    # pseudo-test rows: one per cell, m = t-1; covariates at m only exist for t=164
    obs = te[te.TWS_t.notna()]
    pt = []
    for t in TARGETS:
        o = obs[obs.m == t][["cell_idx", "lat", "lon"]].copy()
        o["m"] = t - 1
        o["target"] = G[o["cell_idx"].to_numpy(), t]
        src = tr[tr.m == t - 1].set_index("cell_idx") if (t - 1) in set(tr.m) else None
        for c in M.DYN:
            o[c] = src[c].reindex(o["cell_idx"]).to_numpy() if src is not None else np.nan
        o["t"] = t
        pt.append(o)
    pt = pd.concat(pt, ignore_index=True)

    CLIM = {"clim_m", "clim_L", "clim_delta", "cell_mean", "cell_std"}
    VARIANTS = [(False, False, "reduced"), (True, False, "reduced+anchor-cov"),
                (True, True, "reduced+anchor-cov, no clim")]
    for with_anchor, no_clim, name in VARIANTS:
        feats = [f for f in BASE + (ANCH if with_anchor else []) if not (no_clim and f in CLIM)]
        print(f"\n=== {name}: {len(feats)} features ===")
        TRX, TRY, TRM, VAX, VAY, VAI, VAT = [], [], [], [], [], [], []
        for t, (dl, dr) in cfgs.items():
            ok = tr[(tr.m + 1 - dl >= tr.m.min()) & (tr.m + 1 + dr <= tr.m.max() + 1)]
            trn = ok[ok.m < val_start]; val = ok[ok.m >= val_start]
            if len(trn) > M.N_PER_CFG:
                trn = trn.iloc[rng.choice(len(trn), M.N_PER_CFG, replace=False)]
            X, g = feats_for(trn, dl, dr or None, P_sel, C, with_anchor)
            TRX.append(X[g]); TRY.append(trn.target.to_numpy()[g]); TRM.append(trn.m.to_numpy()[g])
            X, g = feats_for(val, dl, dr or None, P_sel, C, with_anchor)
            VAX.append(X[g]); VAY.append(val.target.to_numpy()[g]); VAT.append(np.full(int(g.sum()), t))
        TRX = pd.concat(TRX, ignore_index=True); TRY = np.concatenate(TRY); TRM = np.concatenate(TRM)
        VAX = pd.concat(VAX, ignore_index=True); VAY = np.concatenate(VAY); VAT = np.concatenate(VAT)
        ip = VAX["interp"].to_numpy().copy(); bad = ~np.isfinite(ip); ip[bad] = VAX["last_tws"].to_numpy()[bad]
        mdl = lgb.train(M.PARAMS, lgb.Dataset(TRX[feats], TRY, weight=M.recency_weights(TRM)),
                        num_boost_round=M.N_ROUNDS, valid_sets=[lgb.Dataset(VAX[feats], VAY)],
                        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        pv = mdl.predict(VAX[feats])
        rm = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))
        print(f"validation 2013-15 (equal-weighted over the 6 configs): model {rm(VAY, pv):.4f}"
              f"  interp {rm(VAY, ip):.4f}  ratio {rm(VAY, pv)/rm(VAY, ip):.3f}  best_iter {mdl.best_iteration}")
        for t, (dl, dr) in cfgs.items():
            k = VAT == t
            print(f"   cfg {(dl, dr)}: model {rm(VAY[k], pv[k]):.4f}  interp {rm(VAY[k], ip[k]):.4f}"
                  f"  ratio {rm(VAY[k], pv[k])/rm(VAY[k], ip[k]):.3f}")

        # refit on all train with the selected rounds, exactly as model.py does
        FX, FY, FM = [], [], []
        for t, (dl, dr) in cfgs.items():
            ok = tr[(tr.m + 1 - dl >= tr.m.min()) & (tr.m + 1 + dr <= tr.m.max() + 1)]
            if len(ok) > M.N_PER_CFG * 2:
                ok = ok.iloc[rng.choice(len(ok), M.N_PER_CFG * 2, replace=False)]
            X, g = feats_for(ok, dl, dr or None, P_full, C, with_anchor)
            FX.append(X[g]); FY.append(ok.target.to_numpy()[g]); FM.append(ok.m.to_numpy()[g])
        FX = pd.concat(FX, ignore_index=True); FY = np.concatenate(FY); FM = np.concatenate(FM)
        fin = lgb.train(M.PARAMS, lgb.Dataset(FX[feats], FY, weight=M.recency_weights(FM)),
                        num_boost_round=mdl.best_iteration)
        del FX
        print("pseudo-test (real 2015-2018 labels):")
        se_m = se_i = 0.0; n = 0
        for t, (dl, dr) in cfgs.items():
            rows = pt[pt.t == t]
            X, g = feats_for(rows, dl, dr or None, P_full, C, with_anchor)
            X = X[g]; y = rows.target.to_numpy()[g]
            p = fin.predict(X[feats])
            ip = X["interp"].to_numpy().copy(); bad = ~np.isfinite(ip); ip[bad] = X["last_tws"].to_numpy()[bad]
            print(f"   t={t} ({2002 + t // 12}-{t % 12 + 1:02d}) cfg {(dl, dr)} n={len(y):,}: model {rm(y, p):.4f}"
                  f"  interp {rm(y, ip):.4f}  ratio {rm(y, p)/rm(y, ip):.3f}")
            se_m += np.sum((y - p) ** 2); se_i += np.sum((y - ip) ** 2); n += len(y)
        print(f"   pooled: model {np.sqrt(se_m/n):.4f}  interp {np.sqrt(se_i/n):.4f}  ratio {np.sqrt(se_m/se_i):.3f}")


if __name__ == "__main__":
    main()
