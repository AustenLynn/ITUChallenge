"""Diagnostic: is the validation-to-leaderboard gap caused by non-stationarity?

Two independent probes against one fixed validation set:

  RECENCY  -- train on progressively more recent slices of history at a fixed
              row count. If the 2002-2015 relationship still held in 2015-2019,
              older data is harmless and these should tie. If recency wins, the
              relationship is drifting and the training period is a real knob.

  DRIFT    -- give the model an explicit handle on how far each cell has moved
              from its training-period mean: the trailing-36-month mean of
              observed TWS at the anchor (minus the cell mean), and the trailing
              trend. Both use only months <= the anchor, so nothing is read from
              the future.
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import mean_squared_error
import model_v6_reference as M   # written against the v6 feature API

N_PER_H, SEED = 300_000, 0
WIN = 36
DRIFT_FEATS = ["drift36", "trend36", "nobs36"]


def drift_panels(G, win=WIN):
    """Trailing-window mean, OLS slope and observation count of observed TWS,
    for every (cell, month), computed from the observed panel only."""
    V, Msk = np.nan_to_num(G), np.isfinite(G).astype(np.float64)
    t = np.arange(G.shape[1], dtype=np.float64)[None, :]
    def cum(A):
        return np.concatenate([np.zeros((A.shape[0], 1)), np.cumsum(A, axis=1)], axis=1)
    CN, CY = cum(Msk), cum(V)
    CT, CTT, CTY = cum(Msk * t), cum(Msk * t * t), cum(V * t)
    idx = np.arange(G.shape[1])
    lo, hi = np.maximum(idx - win + 1, 0), idx + 1
    n = CN[:, hi] - CN[:, lo]
    Sy, St = CY[:, hi] - CY[:, lo], CT[:, hi] - CT[:, lo]
    Stt, Sty = CTT[:, hi] - CTT[:, lo], CTY[:, hi] - CTY[:, lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, Sy / np.maximum(n, 1), np.nan)
        den = n * Stt - St ** 2
        slope = np.where(n >= 3, (n * Sty - St * Sy) / np.where(den == 0, np.nan, den), np.nan)
    return mean.astype(np.float32), slope.astype(np.float32), n.astype(np.float32)


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

    # test-side availability of the new features, before trusting any of them
    tX, _ = featurise(te[te_h == 3], 3)
    print("drift feature availability (h=3):")
    for f in DRIFT_FEATS:
        print(f"  {f:10s} test nan {tX[f].isna().mean():.1%}   "
              f"test mean nobs {tX['nobs36'].mean():.1f}" if f == "nobs36"
              else f"  {f:10s} test nan {tX[f].isna().mean():.1%}")
    print(f"  trailing observations: train median {np.median(VX.nobs36):.0f}"
          f"  vs test median {np.median(tX.nobs36):.0f}\n")

    runs = [("F  all history (m>=4)", 4, M.FEATS),
            ("H1 from 2006-05 (m>=52)", 52, M.FEATS),
            ("H2 from 2010-05 (m>=100)", 100, M.FEATS),
            ("G  all history + drift feats", 4, M.FEATS + DRIFT_FEATS),
            ("GH from 2010-05 + drift feats", 100, M.FEATS + DRIFT_FEATS)]
    for name, mmin, feats in runs:
        X, y = build(mmin)
        mdl = lgb.train(M.PARAMS, lgb.Dataset(X[feats], y), num_boost_round=4000,
                        valid_sets=[lgb.Dataset(VX[feats], VY)],
                        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        p = mdl.predict(VX[feats])
        print(f"{name:32s} rows={len(X):>9,} iter={mdl.best_iteration:>4} "
              f" weighted RMSE={M.wrmse(VY, p, VH, W):.4f}")
        del X, y, mdl


if __name__ == "__main__":
    main()
