"""Controlled ablation on one fixed validation set.

Isolates the two changes made in v3 -- removing the availability-mismatched
features, and switching to residual learning -- and calibrates the offline
validation against the known leaderboard score of the v2 feature set.
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import mean_squared_error
import model as M

N_PER_H, SEED = 300_000, 0
LEAKY = ["last_delta", "SPEI_06_t_d1", "SOIL_MOISTURE_t_d1"]


def add_leaky(X, rows, h, G, C):
    """The v2 features whose availability does not survive into test."""
    ci, m = rows["cell_idx"].to_numpy(), rows["m"].to_numpy()
    L = m + 1 - h
    X = X.copy()
    X["last_delta"] = G[ci, L] - G[ci, np.maximum(L - 1, 0)]
    for c in ["SPEI_06_t", "SOIL_MOISTURE_t"]:
        X[f"{c}_d1"] = rows[c].to_numpy(np.float32) - C[c][ci, np.maximum(m - 1, 0)]
    return X


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

    known = ~np.isnan(G); known[:, tr.m.max() + 1:] &= False
    obs = te[te.TWS_t.notna()]
    known[obs["cell_idx"].to_numpy(), obs.m.to_numpy()] = True
    te_ci, te_tgt = te["cell_idx"].to_numpy(), te["m"].to_numpy() + 1
    te_h = np.ones(len(te), np.int64)
    for i in range(len(te)):
        h = 1
        while h <= 24 and not known[te_ci[i], te_tgt[i] - h]:
            h += 1
        te_h[i] = h
    W = {h: int(c) for h, c in enumerate(np.bincount(te_h, minlength=8)[1:], 1)}

    val_start = tr.m.max() - M.VALID_MONTHS + 1
    TRX, TRY, TRB, VAX, VAY, VAB, VAH = [], [], [], [], [], [], []
    for h in range(1, M.H_MAX + 1):
        ok = tr[(tr.m + 1 - h) >= tr.m.min()]
        trn = ok[ok.m < val_start]
        trn = trn.iloc[rng.choice(len(trn), min(len(trn), N_PER_H * 2), replace=False)]
        val = ok[ok.m >= val_start]
        Xtr, btr, gtr = M.make_features(trn, h, *P)
        Xva, bva, gva = M.make_features(val, h, *P)
        Xtr, Xva = add_leaky(Xtr, trn, h, G, C), add_leaky(Xva, val, h, G, C)
        TRX.append(Xtr[gtr]); TRY.append(trn.target.to_numpy()[gtr]); TRB.append(btr[gtr])
        VAX.append(Xva[gva]); VAY.append(val.target.to_numpy()[gva]); VAB.append(bva[gva])
        VAH.append(np.full(int(gva.sum()), h))
    TRX = pd.concat(TRX, ignore_index=True); VAX = pd.concat(VAX, ignore_index=True)
    TRY, TRB = np.concatenate(TRY), np.concatenate(TRB)
    VAY, VAB, VAH = np.concatenate(VAY), np.concatenate(VAB), np.concatenate(VAH)
    TRX.to_parquet("cache_trx.parquet"); VAX.to_parquet("cache_vax.parquet")
    np.savez("cache_y.npz", TRY=TRY, TRB=TRB, VAY=VAY, VAB=VAB, VAH=VAH,
             W=np.array([W[h] for h in range(1, 8)]))
    print(f"train {TRX.shape}  valid {VAX.shape}\n")
    print(f"baselines: persistence={M.wrmse(VAY, VAX.last_tws.to_numpy(), VAH, W):.4f}"
          f"  seasonal-persistence={M.wrmse(VAY, VAB, VAH, W):.4f}"
          f"  climatology={M.wrmse(VAY, VAX.clim_m.to_numpy(), VAH, W):.4f}\n")

    # A' -- variant A scored honestly. In validation the mismatched features are
    # present (train is contiguous); in test they are 100% absent for h>=2 and
    # 83.5% absent for h=1. Masking validation to those exact rates is the only
    # way to see what variant A would actually score on the leaderboard.
    VAX_HON = VAX.copy()
    rs = np.random.default_rng(1)
    for f in LEAKY:
        col = VAX_HON[f].to_numpy().copy()
        col[VAH >= 2] = np.nan if f == "last_delta" else col[VAH >= 2]
        m1 = (VAH == 1) & (rs.random(len(VAH)) < 0.835)
        col[m1] = np.nan
        VAX_HON[f] = col

    variants = {
        "A  v2-style, scored on validation":      (M.FEATS + LEAKY, None, VAX),
        "A' v2-style, masked to test availability": (M.FEATS + LEAKY, None, VAX_HON),
        "B  parity-clean, raw target":            (M.FEATS, None, VAX),
        "C  parity-clean, residual vs seasonal":  (M.FEATS, "seasonal", VAX),
        "D  parity-clean, residual vs persistence": (M.FEATS, "persistence", VAX),
    }
    for name, (feats, resid, vax) in variants.items():
        if resid == "seasonal":
            btr, bva = TRB, VAB
        elif resid == "persistence":
            btr, bva = TRX.last_tws.to_numpy(), VAX.last_tws.to_numpy()
        else:
            btr = bva = None
        ytr = TRY - btr if resid else TRY
        yva = VAY - bva if resid else VAY
        mdl = lgb.train(M.PARAMS, lgb.Dataset(TRX[feats], ytr), num_boost_round=4000,
                        valid_sets=[lgb.Dataset(vax[feats], yva)],
                        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        pred = mdl.predict(vax[feats]) + (bva if resid else 0)
        print(f"{name:40s} iter={mdl.best_iteration:>4}  "
              f"weighted RMSE={M.wrmse(VAY, pred, VAH, W):.4f}")
        by_h = "  ".join(f"h{h}:{mean_squared_error(VAY[VAH==h], pred[VAH==h])**0.5:.3f}"
                         for h in range(1, 8))
        print(f"{'':40s}  {by_h}")


if __name__ == "__main__":
    main()
