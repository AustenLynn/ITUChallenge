"""Is the validation-to-leaderboard gap partly self-inflicted?

`climatology()` aggregates over every train row, so a validation row's clim_m /
cell_mean / cell_std include that row's own target. A test row's do not -- they
come from train only. Validation has therefore been scored with a feature that
is better informed than the one test receives.

This rebuilds validation with those statistics computed ONLY from months before
the validation window, which is the honest analogue of what test gets, and
measures how much of the offline score was coming from the leak.
"""
import numpy as np, pandas as pd, lightgbm as lgb
from collections import Counter
from sklearn.metrics import mean_squared_error
import model_v6_reference as M   # written against the v6 feature API

N_PER_CFG = 150_000


def main():
    rng = np.random.default_rng(0)
    tr, te = M.load()
    cells = pd.Index(sorted(set(zip(tr.lat, tr.lon))))
    tr["cell_idx"] = cells.get_indexer(pd.MultiIndex.from_arrays([tr.lat, tr.lon]))
    te["cell_idx"] = cells.get_indexer(pd.MultiIndex.from_arrays([te.lat, te.lon]))
    G = M.build_panel(tr, te, cells)
    C = M.build_covariates(tr, te, len(cells), G.shape[1])
    val_start = tr.m.max() - M.VALID_MONTHS + 1

    def panels(src):
        clim, cmean, cstd = M.climatology(src, cells)
        NB, NBA = M.neighbour_panels(cells, G, clim)
        DM, DS, DN = M.drift_panels(G)
        return (G, clim, cmean, cstd, C, NB, NBA, DM, DS, DN)

    P_leak = panels(tr)                          # what we have been doing
    P_hon = panels(tr[tr.m < val_start])         # what test actually gets

    known = ~np.isnan(G); known[:, tr.m.max() + 1:] &= False
    obs = te[te.TWS_t.notna()]
    known[obs["cell_idx"].to_numpy(), obs.m.to_numpy()] = True
    tci, ttg = te["cell_idx"].to_numpy(), te["m"].to_numpy() + 1
    n_m = G.shape[1]
    dL = np.ones(len(te), np.int64); dR = np.zeros(len(te), np.int64)
    for i in range(len(te)):
        d = 1
        while d <= 24 and not known[tci[i], ttg[i] - d]:
            d += 1
        dL[i] = d
        d = 1
        while d <= 24 and ttg[i] + d < n_m and not known[tci[i], ttg[i] + d]:
            d += 1
        dR[i] = d if (d <= 24 and ttg[i] + d < n_m and known[tci[i], ttg[i] + d]) else 0
    cfg = Counter(zip(dL.tolist(), dR.tolist()))
    MAJOR = sorted(c for c, n in cfg.items() if n >= 4000)
    W = {i: cfg[c] for i, c in enumerate(MAJOR)}

    def build(P):
        TRX, TRY, TRM, VAX, VAY, VAC = [], [], [], [], [], []
        for i, (dl, dr) in enumerate(MAJOR):
            ok = tr[(tr.m + 1 - dl >= tr.m.min()) & (tr.m + 1 + dr <= tr.m.max() + 1)]
            trn = ok[ok.m < val_start]
            if len(trn) > N_PER_CFG:
                trn = trn.iloc[rng.choice(len(trn), N_PER_CFG, replace=False)]
            val = ok[ok.m >= val_start]
            Xtr, btr, _ = M.make_features(trn, dl, dr or None, *P)
            Xva, bva, _ = M.make_features(val, dl, dr or None, *P)
            gtr, gva = np.isfinite(btr), np.isfinite(bva)
            if dr:
                gtr &= np.isfinite(Xtr["right_tws"].to_numpy())
                gva &= np.isfinite(Xva["right_tws"].to_numpy())
            TRX.append(Xtr[gtr]); TRY.append(trn.target.to_numpy()[gtr])
            TRM.append(trn.m.to_numpy()[gtr])
            VAX.append(Xva[gva]); VAY.append(val.target.to_numpy()[gva])
            VAC.append(np.full(int(gva.sum()), i))
        return (pd.concat(TRX, ignore_index=True), np.concatenate(TRY),
                np.concatenate(TRM), pd.concat(VAX, ignore_index=True),
                np.concatenate(VAY), np.concatenate(VAC))

    for name, P in (("leaky  (clim from ALL train)", P_leak),
                    ("honest (clim from pre-validation months only)", P_hon)):
        TRX, TRY, TRM, VAX, VAY, VAC = build(P)
        w = M.recency_weights(TRM)
        mdl = lgb.train(M.PARAMS, lgb.Dataset(TRX[M.FEATS], TRY, weight=w),
                        num_boost_round=4000,
                        valid_sets=[lgb.Dataset(VAX[M.FEATS], VAY)],
                        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        p = mdl.predict(VAX[M.FEATS])
        print(f"{name:46s} iter={mdl.best_iteration:>4}  "
              f"config-weighted RMSE={M.wrmse(VAY, p, VAC, W):.4f}")
        del TRX, TRY, VAX, VAY, mdl


if __name__ == "__main__":
    main()
