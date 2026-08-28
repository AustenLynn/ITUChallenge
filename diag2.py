"""Where exactly is the recency optimum, and does soft weighting beat a hard cut?

A hard cutoff throws away rows; exponential recency weights keep every row but
lean on recent ones. Both are swept here against the same validation set.
"""
import numpy as np, pandas as pd, lightgbm as lgb
import model as M
from diag import drift_panels, DRIFT_FEATS, N_PER_H

SEED = 0
FE = M.FEATS + DRIFT_FEATS


def main():
    rng = np.random.default_rng(SEED)
    tr, te = M.load()
    cells = pd.Index(sorted(set(zip(tr.lat, tr.lon))))
    tr["cell_idx"] = cells.get_indexer(pd.MultiIndex.from_arrays([tr.lat, tr.lon]))
    te["cell_idx"] = cells.get_indexer(pd.MultiIndex.from_arrays([te.lat, te.lon]))
    G = M.build_panel(tr, te, cells)
    C = M.build_covariates(tr, te, len(cells), G.shape[1])
    clim, cmean, cstd = M.climatology(tr, cells)
    NB, NBA = M.neighbour_panels(cells, G, clim)
    P = (G, clim, cmean, cstd, C, NB, NBA)
    DM, DS, DN = drift_panels(G)

    known = ~np.isnan(G); known[:, tr.m.max() + 1:] &= False
    obs = te[te.TWS_t.notna()]
    known[obs["cell_idx"].to_numpy(), obs.m.to_numpy()] = True
    tci, ttg = te["cell_idx"].to_numpy(), te["m"].to_numpy() + 1
    te_h = np.ones(len(te), np.int64)
    for i in range(len(te)):
        h = 1
        while h <= 24 and not known[tci[i], ttg[i] - h]:
            h += 1
        te_h[i] = h
    W = {h: int(c) for h, c in enumerate(np.bincount(te_h, minlength=8)[1:], 1)}
    val_start = tr.m.max() - M.VALID_MONTHS + 1

    def featurise(rows, h):
        X, base, _ = M.make_features(rows, h, *P)
        ci, L = rows["cell_idx"].to_numpy(), rows["m"].to_numpy() + 1 - h
        X["drift36"] = DM[ci, L] - cmean[ci]
        X["trend36"] = DS[ci, L]
        X["nobs36"] = DN[ci, L]
        X["m_row"] = rows["m"].to_numpy()
        return X, base

    def build(month_min):
        Xs, ys = [], []
        for h in range(1, M.H_MAX + 1):
            ok = tr[((tr.m + 1 - h) >= tr.m.min()) & (tr.m < val_start) & (tr.m >= month_min)]
            if len(ok) > N_PER_H:
                ok = ok.iloc[rng.choice(len(ok), N_PER_H, replace=False)]
            X, base = featurise(ok, h)
            g = np.isfinite(base)
            Xs.append(X[g]); ys.append(ok.target.to_numpy()[g])
        return pd.concat(Xs, ignore_index=True), np.concatenate(ys)

    VX, VY, VH = [], [], []
    for h in range(1, M.H_MAX + 1):
        val = tr[((tr.m + 1 - h) >= tr.m.min()) & (tr.m >= val_start)]
        X, base = featurise(val, h)
        g = np.isfinite(base)
        VX.append(X[g]); VY.append(val.target.to_numpy()[g]); VH.append(np.full(int(g.sum()), h))
    VX = pd.concat(VX, ignore_index=True)
    VY, VH = np.concatenate(VY), np.concatenate(VH)

    ALL, ALLY = build(4)
    def run(name, X, y, w=None):
        mdl = lgb.train(M.PARAMS, lgb.Dataset(X[FE], y, weight=w), num_boost_round=4000,
                        valid_sets=[lgb.Dataset(VX[FE], VY)],
                        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        p = mdl.predict(VX[FE])
        print(f"{name:34s} rows={len(X):>9,} iter={mdl.best_iteration:>4}"
              f"  weighted RMSE={M.wrmse(VY, p, VH, W):.4f}")

    for hl in (18, 36, 72):
        w = 0.5 ** ((val_start - ALL.m_row.to_numpy()) / hl)
        run(f"weights, half-life {hl} months", ALL, ALLY, w)
    del ALL, ALLY

    for mmin, lbl in ((76, "2008-05"), (112, "2011-05"), (124, "2012-05")):
        X, y = build(mmin)
        run(f"hard cut from {lbl} (m>={mmin})", X, y)
        del X, y


if __name__ == "__main__":
    main()
