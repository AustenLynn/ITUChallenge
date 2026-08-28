"""Settle two remaining design questions against one fixed honest validation set:
   - does masking the mismatched features during TRAINING beat leaving them intact?
   - does the completeness filter on training rows help or hurt?
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import mean_squared_error
import model as M
from ablate import add_leaky, LEAKY

N_PER_H, SEED = 300_000, 0
FE = M.FEATS + LEAKY


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
    te_h = np.ones(len(te), np.int64)
    tci, ttg = te["cell_idx"].to_numpy(), te["m"].to_numpy() + 1
    for i in range(len(te)):
        h = 1
        while h <= 24 and not known[tci[i], ttg[i] - h]:
            h += 1
        te_h[i] = h
    W = {h: int(c) for h, c in enumerate(np.bincount(te_h, minlength=8)[1:], 1)}

    val_start = tr.m.max() - M.VALID_MONTHS + 1
    Ff, Yf, Fa, Ya, VX, VY, VH = [], [], [], [], [], [], []
    for h in range(1, M.H_MAX + 1):
        ok = tr[(tr.m + 1 - h) >= tr.m.min()]
        trn = ok[ok.m < val_start]
        trn = trn.iloc[rng.choice(len(trn), min(len(trn), N_PER_H * 2), replace=False)]
        val = ok[ok.m >= val_start]
        Xtr, btr, gtr = M.make_features(trn, h, *P)
        Xva, bva, gva = M.make_features(val, h, *P)
        Xtr, Xva = add_leaky(Xtr, trn, h, G, C), add_leaky(Xva, val, h, G, C)
        y = trn.target.to_numpy()
        Ff.append(Xtr[gtr]); Yf.append(y[gtr])                 # completeness-filtered
        keep = np.isfinite(btr)
        Fa.append(Xtr[keep]); Ya.append(y[keep])               # all rows with an anchor
        VX.append(Xva[gva]); VY.append(val.target.to_numpy()[gva])
        VH.append(np.full(int(gva.sum()), h))
    Ff, Fa = pd.concat(Ff, ignore_index=True), pd.concat(Fa, ignore_index=True)
    Yf, Ya = np.concatenate(Yf), np.concatenate(Ya)
    VX = pd.concat(VX, ignore_index=True)
    VY, VH = np.concatenate(VY), np.concatenate(VH)

    # validation masked to the real test availability of the mismatched features
    rs = np.random.default_rng(1)
    VXH = VX.copy()
    for f in LEAKY:
        col = VXH[f].to_numpy().copy()
        if f == "last_delta":
            col[VH >= 2] = np.nan
        col[(VH == 1) & (rs.random(len(VH)) < 0.835)] = np.nan
        VXH[f] = col

    def mask_train(X, hv):
        """Apply the same availability pattern to training rows."""
        X = X.copy()
        r = np.random.default_rng(2)
        for f in LEAKY:
            col = X[f].to_numpy().copy()
            if f == "last_delta":
                col[hv >= 2] = np.nan
            col[(hv == 1) & (r.random(len(hv)) < 0.835)] = np.nan
            X[f] = col
        return X

    runs = {
        "A' filtered rows, features intact":    (Ff, Yf),
        "A'' filtered rows, TRAIN-masked":      (mask_train(Ff, Ff.h.to_numpy()), Yf),
        "E  all anchored rows, features intact": (Fa, Ya),
        "E'' all anchored rows, TRAIN-masked":  (mask_train(Fa, Fa.h.to_numpy()), Ya),
    }
    print(f"filtered train {Ff.shape}  all-anchored train {Fa.shape}  valid {VX.shape}\n")
    for name, (X, y) in runs.items():
        mdl = lgb.train(M.PARAMS, lgb.Dataset(X[FE], y), num_boost_round=4000,
                        valid_sets=[lgb.Dataset(VXH[FE], VY)],
                        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        p = mdl.predict(VXH[FE])
        print(f"{name:38s} iter={mdl.best_iteration:>4}  weighted RMSE={M.wrmse(VY, p, VH, W):.4f}")
        print("   " + "  ".join(f"h{h}:{mean_squared_error(VY[VH==h], p[VH==h])**0.5:.3f}"
                                for h in range(1, 8)))


if __name__ == "__main__":
    main()
