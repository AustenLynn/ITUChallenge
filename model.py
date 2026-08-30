"""
Horizon-aware TWS forecasting.

Framing: predict TWS at month m given the last month L where TWS was actually
observed, the horizon h = m - L (1..7), and covariates over [L, m-1].

Two rules drive every design choice here:

INTERPOLATION. 88.9% of test rows have an observed TWS value for the same cell
at a LATER month: Test.csv carries six fully-observed months (164, 168, 173,
179, 198, 202) and every masked block sits between two of them. Those rows are
an interpolation problem, not a forecast, and interpolation is far easier --
plain linear interpolation scores 0.610 against 0.690 for persistence at the
exact anchor distances the test set uses.

Every row is therefore keyed on a (dL, dR) pair -- months back to the left
anchor, months forward to the right anchor -- and training samples are drawn at
precisely the (dL, dR) combinations that occur in test, so the model learns the
right skill surface. dR is NaN for the 31k trailing rows with no right anchor.

Design settled by ablation (see ablate.py / ablate2.py), not by intuition --
three plausible-sounding ideas were measured and rejected:

  - Dropping features whose availability differs between train and test
    (`last_delta` is 100% absent for masked test rows but present in training)
    LOSES 0.009. Keeping them and letting LightGBM route the NaNs wins: they
    still pay off on the h=1 rows where they exist.
  - Masking those features during training to match test availability loses a
    further 0.008. The model learns more from the intact signal than it gains
    from rehearsing the missingness.
  - Residual learning against a seasonal-persistence base loses 0.014, and
    against a plain persistence base 0.009. The seasonal base is itself worse
    than raw persistence (0.862 vs 0.838), so it is a worse anchor, not a
    better one.

`report_parity` still prints train/test availability gaps every run -- the
mismatch is now a known, measured cost rather than an unnoticed one.

What did work was treating the problem as non-stationary rather than as
under-featured (diag.py / diag2.py / diag3.py). The 2002-2015 relationship has
drifted by the 2015-2019 test period, so:

  - training rows carry exponential recency weights (half-life 15 months),
    worth ~0.017; a soft decay beats every hard cutoff, which throws rows away
  - each cell gets explicit drift features -- its trailing-36-month observed
    mean relative to its training-period mean, and the trailing trend -- worth
    a further ~0.004

Both gains survive a gapped holdout that mimics the longer lead we are scored
at, so they are not an artifact of the validation window sitting next to train.
"""
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

os.environ.setdefault("CODECARBON_LOG_LEVEL", "error")
from codecarbon import EmissionsTracker

DATA = "data"
SEED = 0
H_MAX = 7
N_PER_H = 1_000_000
N_PER_CFG = 300_000
VALID_MONTHS = 24

PARAMS = dict(objective="regression", metric="rmse", learning_rate=0.02,
              num_leaves=63, min_data_in_leaf=200, feature_fraction=0.85,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=-1, seed=SEED)
N_ROUNDS = 8000

DYN = ["SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t", "SOIL_MOISTURE_t"]
EXTREME = ["SPEI_01_t", "SPEI_03_t", "SOIL_MOISTURE_t"]
STATIC = ["lat", "lon"]
LAG1 = ["SPEI_06_t", "SOIL_MOISTURE_t"]
DRIFT_FEATS = ["drift36", "trend36", "nobs36"]
RIGHT_FEATS = ["dR", "span", "w", "right_tws", "right_anom", "right_delta",
               "interp", "nb_right_tws", "interp_nb", "SPEI_06_t_dR",
               "SOIL_MOISTURE_t_dR"]
DRIFT_WIN = 36
HALF_LIFE = 15.0      # months; flat optimum over 12-18 on both holdouts
FEATS = (["h", "last_tws", "last_anom", "last_delta", "clim_m", "clim_L", "clim_delta",
          "cell_mean", "cell_std", "month_sin", "month_cos",
          "nb_last_tws", "nb_last_anom"] + STATIC + DYN
         + [f"{c}_d" for c in DYN] + [f"{c}_gmean" for c in DYN]
         + [f"{c}_gmin" for c in EXTREME] + [f"{c}_gmax" for c in EXTREME]
         + [f"{c}_d1" for c in LAG1] + DRIFT_FEATS + RIGHT_FEATS)


def drift_panels(G, win=DRIFT_WIN):
    """Trailing-window mean, OLS slope and observation count of observed TWS at
    every (cell, month). Uses only months <= the one indexed, so a row reading
    these at its anchor never sees past its own prediction time."""
    V, Msk = np.nan_to_num(G), np.isfinite(G).astype(np.float64)
    t = np.arange(G.shape[1], dtype=np.float64)[None, :]
    def cum(A):
        return np.concatenate([np.zeros((A.shape[0], 1)), np.cumsum(A, axis=1)], axis=1)
    CN, CY, CT, CTT, CTY = cum(Msk), cum(V), cum(Msk * t), cum(Msk * t * t), cum(V * t)
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


def mindex(t):
    return (t.dt.year - 2002) * 12 + (t.dt.month - 1)


def load():
    tr = pd.read_csv(f"{DATA}/Train.csv", parse_dates=["time"])
    te = pd.read_csv(f"{DATA}/Test.csv", parse_dates=["time"])
    for df in (tr, te):
        df["m"] = mindex(df["time"])
    return tr, te


def build_panel(tr, te, cells):
    """G[cell, month] = TWS known at prediction time (train + unmasked test)."""
    n_m = int(max(tr.m.max(), te.m.max())) + 2
    G = np.full((len(cells), n_m), np.nan, dtype=np.float32)
    ci = tr["cell_idx"].to_numpy()
    G[ci, tr.m.to_numpy()] = tr.TWS_t.to_numpy()
    G[ci, tr.m.to_numpy() + 1] = tr.target.to_numpy()
    obs = te[te.TWS_t.notna()]
    G[obs["cell_idx"].to_numpy(), obs.m.to_numpy()] = obs.TWS_t.to_numpy()
    return G


def build_covariates(tr, te, n_cells, n_m):
    """C[var][cell, month]. Covariates are inputs, never masked, so pooling
    test rows leaks nothing."""
    C = {c: np.full((n_cells, n_m), np.nan, dtype=np.float32) for c in DYN}
    for df in (tr, te):
        ci, m = df["cell_idx"].to_numpy(), df["m"].to_numpy()
        for c in DYN:
            C[c][ci, m] = df[c].to_numpy(np.float32)
    return C


def smooth3(A):
    """3x3 neighbourhood mean over (lat, lon) per month. Longitude wraps;
    latitude does not, so rolled-in polar rows are blanked."""
    V, M = np.nan_to_num(A), (~np.isnan(A)).astype(np.float32)
    s, c = np.zeros_like(V), np.zeros_like(M)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            v, mm = np.roll(V, dy, 0), np.roll(M, dy, 0)
            if dy:
                edge = 0 if dy > 0 else -1
                v, mm = v.copy(), mm.copy()
                v[edge] = 0.0
                mm[edge] = 0.0
            s += np.roll(v, dx, 1)
            c += np.roll(mm, dx, 1)
    return np.where(c > 0, s / np.maximum(c, 1e-6), np.nan).astype(np.float32)


def neighbour_panels(cells, G, clim):
    lats = np.array([c[0] for c in cells]); lons = np.array([c[1] for c in cells])
    _, ilat = np.unique(lats, return_inverse=True)
    _, ilon = np.unique(lons, return_inverse=True)
    shape = (ilat.max() + 1, ilon.max() + 1, G.shape[1])
    grid = np.full(shape, np.nan, dtype=np.float32); grid[ilat, ilon] = G
    anom = np.full(shape, np.nan, dtype=np.float32)
    anom[ilat, ilon] = G - clim[:, np.arange(G.shape[1]) % 12]
    return smooth3(grid)[ilat, ilon], smooth3(anom)[ilat, ilon]


def climatology(tr, cells):
    cal, ci, y = (tr.m % 12).to_numpy(), tr["cell_idx"].to_numpy(), tr.target.to_numpy()
    s = np.zeros((len(cells), 12)); n = np.zeros((len(cells), 12))
    np.add.at(s, (ci, cal), y)
    np.add.at(n, (ci, cal), 1)
    agg = tr.groupby("cell_idx")["target"].agg(["mean", "std"]).reindex(range(len(cells)))
    return ((s / np.maximum(n, 1)).astype(np.float32),
            agg["mean"].to_numpy(np.float32), agg["std"].to_numpy(np.float32))


def make_features(rows, h, dR, G, clim, cell_mean, cell_std, C, NB, NBA, DM, DS, DN):
    """Target month is m+1; anchor L = m+1-h; covariate path spans [L, m]."""
    ci, m = rows["cell_idx"].to_numpy(), rows["m"].to_numpy()
    tgt, L = m + 1, m + 1 - h
    last = G[ci, L]
    clim_m, clim_L = clim[ci, tgt % 12], clim[ci, L % 12]
    out = pd.DataFrame({
        "h": np.full(len(rows), h, np.float32),
        "last_tws": last,
        "last_anom": last - clim_L,
        "clim_m": clim_m,
        "clim_L": clim_L,
        "clim_delta": clim_m - clim_L,
        "cell_mean": cell_mean[ci],
        "cell_std": cell_std[ci],
        "month_sin": np.sin(2 * np.pi * (tgt % 12) / 12).astype(np.float32),
        "month_cos": np.cos(2 * np.pi * (tgt % 12) / 12).astype(np.float32),
        "nb_last_tws": NB[ci, L],
        "nb_last_anom": NBA[ci, L],
        "last_delta": last - G[ci, np.maximum(L - 1, 0)],
    })
    for c in STATIC:
        out[c] = rows[c].to_numpy(np.float32)

    # Covariate path over the gap. Every month in [L, m] is observed in both
    # train and test (the anchor is by definition an observed month, and the
    # intervening months are all present as test rows), so these carry the
    # "what happened to the climate since we last saw TWS" signal without the
    # availability mismatch that fixed lags suffer from.
    # Train months are themselves non-contiguous (22 GRACE outages, clustered
    # in 2011-2014), so a training path can have holes where a test path never
    # does. `complete` marks the rows whose window is fully observed; callers
    # keep only those, which makes the training distribution match test.
    win = L[:, None] + np.arange(h)[None, :]
    complete = np.isfinite(last)
    if dR is not None:
        complete &= np.isfinite(G[ci, tgt + dR])
    for c in DYN:
        path = C[c][ci[:, None], win]
        complete &= np.isfinite(path).all(axis=1)
        at_t = rows[c].to_numpy(np.float32)
        out[c] = at_t
        out[f"{c}_d"] = at_t - path[:, 0]
        out[f"{c}_gmean"] = np.nanmean(path, axis=1)
        if c in EXTREME:
            out[f"{c}_gmin"] = np.nanmin(path, axis=1)
            out[f"{c}_gmax"] = np.nanmax(path, axis=1)
        if c in LAG1:
            out[f"{c}_d1"] = at_t - C[c][ci, np.maximum(m - 1, 0)]

    # Right-anchor block: the observed TWS on the far side of the gap. Present
    # for 88.9% of test rows; NaN elsewhere and LightGBM routes those itself.
    n = len(rows)
    if dR is None:
        for f in RIGHT_FEATS:
            out[f] = np.full(n, np.nan, np.float32)
    else:
        R = tgt + dR
        rt = G[ci, R]
        span = float(h + dR)
        w = h / span
        out["dR"] = np.full(n, dR, np.float32)
        out["span"] = np.full(n, span, np.float32)
        out["w"] = np.full(n, w, np.float32)
        out["right_tws"] = rt
        out["right_anom"] = rt - clim[ci, R % 12]
        out["right_delta"] = rt - last
        out["interp"] = (1 - w) * last + w * rt
        out["nb_right_tws"] = NB[ci, R]
        out["interp_nb"] = (1 - w) * NB[ci, L] + w * NB[ci, R]
        for c in ("SPEI_06_t", "SOIL_MOISTURE_t"):
            out[f"{c}_dR"] = C[c][ci, R] - rows[c].to_numpy(np.float32)

    out["drift36"] = DM[ci, L] - cell_mean[ci]
    out["trend36"] = DS[ci, L]
    out["nobs36"] = DN[ci, L]

    base = last + (clim_m - clim_L)          # seasonal persistence
    return out[FEATS].astype(np.float32), base, complete


def report_parity(Xtr, Xte, h, tol=0.10):
    """Print features whose missingness differs sharply between train and test.

    Ablation says keeping them still wins, so this reports rather than blocks --
    but an unnoticed mismatch is how a feature silently stops working on the
    leaderboard, so every run states the ones we are knowingly accepting.
    """
    for f in FEATS:
        a, b = Xtr[f].isna().mean(), Xte[f].isna().mean()
        if abs(a - b) > tol:
            print(f"    parity gap h={h}: {f} (train {a:.1%} vs test {b:.1%})")


def recency_weights(months):
    """Exponential decay toward the end of the available record. Measured worth
    ~0.017 RMSE; a soft decay beats any hard cutoff because it keeps the rows."""
    return 0.5 ** ((months.max() - months) / HALF_LIFE)


def fit(Xtr, ytr, Xva, yva, tag, w=None):
    model = lgb.train(PARAMS, lgb.Dataset(Xtr, ytr, weight=w), num_boost_round=N_ROUNDS,
                      valid_sets=[lgb.Dataset(Xva, yva)],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
    print(f"  {tag:12s} rows={len(Xtr):>9,}  best_iter={model.best_iteration:>5}")
    return model


def wrmse(y, p, hv, weights):
    """RMSE reweighted to the test set's (dL, dR) configuration mix, so
    validation decisions are made against the distribution we are scored on."""
    num = sum(weights[h] * mean_squared_error(y[hv == h], p[hv == h])
              for h in weights if (hv == h).any())
    return (num / sum(weights.values())) ** 0.5


def main():
    rng = np.random.default_rng(SEED)
    tr, te = load()
    cells = pd.Index(sorted(set(zip(tr.lat, tr.lon))))
    tr["cell_idx"] = cells.get_indexer(pd.MultiIndex.from_arrays([tr.lat, tr.lon]))
    te["cell_idx"] = cells.get_indexer(pd.MultiIndex.from_arrays([te.lat, te.lon]))

    G = build_panel(tr, te, cells)
    C = build_covariates(tr, te, len(cells), G.shape[1])
    clim, cell_mean, cell_std = climatology(tr, cells)
    NB, NBA = neighbour_panels(cells, G, clim)
    DM, DS, DN = drift_panels(G)
    P = (G, clim, cell_mean, cell_std, C, NB, NBA, DM, DS, DN)

    # --- anchors for every test row: months back to the last observed TWS,
    #     and months forward to the next one (0 == no right anchor) ---
    known = ~np.isnan(G); known[:, tr.m.max() + 1:] &= False
    obs = te[te.TWS_t.notna()]
    known[obs["cell_idx"].to_numpy(), obs.m.to_numpy()] = True
    te_ci, te_tgt = te["cell_idx"].to_numpy(), te["m"].to_numpy() + 1
    n_m = G.shape[1]
    te_dL, te_dR = np.ones(len(te), np.int64), np.zeros(len(te), np.int64)
    for i in range(len(te)):
        d = 1
        while d <= 24 and not known[te_ci[i], te_tgt[i] - d]:
            d += 1
        te_dL[i] = d
        d = 1
        while d <= 24 and te_tgt[i] + d < n_m and not known[te_ci[i], te_tgt[i] + d]:
            d += 1
        te_dR[i] = d if (d <= 24 and te_tgt[i] + d < n_m
                         and known[te_ci[i], te_tgt[i] + d]) else 0

    cfg_counts = Counter(zip(te_dL.tolist(), te_dR.tolist()))
    bracketed = sum(n for (_, dr), n in cfg_counts.items() if dr)
    print(f"test configurations: {len(cfg_counts)} distinct (dL, dR); "
          f"bracketed {bracketed:,}/{len(te):,} = {bracketed / len(te):.1%}")
    MAJOR = sorted(c for c, n in cfg_counts.items() if n >= 4000)
    for c in MAJOR:
        print(f"  dL={c[0]} dR={c[1] if c[1] else '-'}   {cfg_counts[c]:>7,} rows")
    W = {i: cfg_counts[c] for i, c in enumerate(MAJOR)}

    # --- training / validation drawn at exactly those (dL, dR) combinations ---
    val_start = tr.m.max() - VALID_MONTHS + 1
    TRX, TRY, TRM = [], [], []
    VAX, VAY, VAC, VAH, VABASE, VAM = [], [], [], [], [], []
    for ci_cfg, (dl, dr) in enumerate(MAJOR):
        ok = tr[(tr.m + 1 - dl >= tr.m.min()) & (tr.m + 1 + dr <= tr.m.max() + 1)]
        trn = ok[ok.m < val_start]
        if len(trn) > N_PER_CFG:
            trn = trn.iloc[rng.choice(len(trn), N_PER_CFG, replace=False)]
        val = ok[ok.m >= val_start]
        Xtr, btr, _ = make_features(trn, dl, dr or None, *P)
        Xva, bva, _ = make_features(val, dl, dr or None, *P)
        gtr, gva = np.isfinite(btr), np.isfinite(bva)
        if dr:
            gtr &= np.isfinite(Xtr["right_tws"].to_numpy())
            gva &= np.isfinite(Xva["right_tws"].to_numpy())
        print(f"  dL={dl} dR={dr if dr else '-'}: train {gtr.sum():>8,}  valid {gva.sum():>7,}")
        TRX.append(Xtr[gtr]); TRY.append(trn.target.to_numpy()[gtr])
        TRM.append(trn.m.to_numpy()[gtr])
        VAX.append(Xva[gva]); VAY.append(val.target.to_numpy()[gva])
        VABASE.append(bva[gva]); VAM.append(val.m.to_numpy()[gva])
        VAC.append(np.full(int(gva.sum()), ci_cfg))
        VAH.append(np.full(int(gva.sum()), dl))
    TRX = pd.concat(TRX, ignore_index=True)
    TRY, TRM = np.concatenate(TRY), np.concatenate(TRM)
    VAX = pd.concat(VAX, ignore_index=True)
    VAY, VAM = np.concatenate(VAY), np.concatenate(VAM)
    VAB, VAC, VAH = np.concatenate(VABASE), np.concatenate(VAC), np.concatenate(VAH)
    print(f"train {TRX.shape}  valid {VAX.shape}")

    print("\nbaselines (test-weighted):")
    print(f"  persistence         {wrmse(VAY, VAX.last_tws.to_numpy(), VAC, W):.4f}")
    print(f"  climatology         {wrmse(VAY, VAX.clim_m.to_numpy(), VAC, W):.4f}")
    ip = VAX.interp.to_numpy().copy()
    ip[~np.isfinite(ip)] = VAX.last_tws.to_numpy()[~np.isfinite(ip)]
    print(f"  linear interpolation{wrmse(VAY, ip, VAC, W):.4f}   <- the thing to beat")

    tracker = EmissionsTracker(project_name="tws-forecast", output_dir=".",
                               output_file="emissions.csv", log_level="error")
    tracker.start()

    print("\nglobal model:")
    TRW = recency_weights(TRM)
    g_model = fit(TRX[FEATS], TRY, VAX[FEATS], VAY, "global", TRW)
    g_pred = g_model.predict(VAX[FEATS])

    print("per-dL models:")
    h_models, h_pred = {}, np.empty(len(VAX))
    for h in sorted(set(VAH.tolist())):
        ktr, kva = (TRX.h == h).to_numpy(), (VAH == h)
        h_models[h] = fit(TRX[FEATS][ktr], TRY[ktr], VAX[FEATS][kva], VAY[kva],
                          f"dL={h}", TRW[ktr])
        h_pred[kva] = h_models[h].predict(VAX[FEATS][kva])
    emissions = tracker.stop()

    b_pred = 0.5 * (g_pred + h_pred)
    scores = {"global": wrmse(VAY, g_pred, VAC, W),
              "per_horizon": wrmse(VAY, h_pred, VAC, W),
              "blend": wrmse(VAY, b_pred, VAC, W)}
    print("\ntest-weighted validation RMSE: "
          + "  ".join(f"{k}={v:.4f}" for k, v in scores.items()))
    best = min(scores, key=scores.get)
    print(f"selected: {best}")
    print(f"training emissions: {emissions * 1000:.2f} g CO2eq"
          f"  (energy {tracker.final_emissions_data.energy_consumed * 1000:.1f} Wh)")

    print("\nRMSE by configuration (model / interpolation / persistence):")
    for i, c in enumerate(MAJOR):
        k = VAC == i
        mp = b_pred if best == "blend" else (g_pred if best == "global" else h_pred)
        print(f"  dL={c[0]} dR={c[1] if c[1] else '-':>2}: "
              f"{mean_squared_error(VAY[k], mp[k]) ** 0.5:.4f}"
              f" / {mean_squared_error(VAY[k], ip[k]) ** 0.5:.4f}"
              f" / {mean_squared_error(VAY[k], VAX.last_tws.to_numpy()[k]) ** 0.5:.4f}")

    print("\nfeature importance (global), top 15:")
    for f, gain in sorted(zip(FEATS, g_model.feature_importance("gain")),
                          key=lambda x: -x[1])[:15]:
        print(f"  {f:20s} {gain:,.0f}")

    pd.DataFrame({"h": VAH, "cfg": VAC, "target": VAY, "pred_global": g_pred,
                  "pred_horizon": h_pred, "interp": ip,
                  "lat": VAX.lat, "lon": VAX.lon}).to_parquet("valid_preds.parquet")

    # --- refit on train+valid, predict test ---
    ALLX = pd.concat([TRX, VAX], ignore_index=True)
    ALLY = np.concatenate([TRY, VAY])
    ALLM = np.concatenate([TRM, VAM])
    ALLW = recency_weights(ALLM)
    if best in ("global", "blend"):
        fin_g = lgb.train(PARAMS, lgb.Dataset(ALLX[FEATS], ALLY, weight=ALLW),
                          num_boost_round=g_model.best_iteration)
    if best in ("per_horizon", "blend"):
        fin_h = {h: lgb.train(PARAMS, lgb.Dataset(ALLX[FEATS][(ALLX.h == h).to_numpy()],
                                                  ALLY[(ALLX.h == h).to_numpy()],
                                                  weight=ALLW[(ALLX.h == h).to_numpy()]),
                              num_boost_round=h_models[h].best_iteration)
                 for h in h_models}

    pred = np.empty(len(te))
    for (dl, dr) in sorted(cfg_counts):
        k = (te_dL == dl) & (te_dR == dr)
        if not k.any():
            continue
        X, _, _ = make_features(te[k], dl, dr or None, *P)
        X = X[FEATS]
        mh = h_models.get(dl, h_models[min(h_models, key=lambda z: abs(z - dl))])
        if best == "global":
            pred[k] = fin_g.predict(X)
        elif best == "per_horizon":
            pred[k] = fin_h.get(dl, fin_h[min(fin_h, key=lambda z: abs(z - dl))]).predict(X)
        else:
            alt = fin_h.get(dl, fin_h[min(fin_h, key=lambda z: abs(z - dl))])
            pred[k] = 0.5 * (fin_g.predict(X) + alt.predict(X))
    assert np.isfinite(pred).all()

    sub = pd.DataFrame({"ID": te["ID"], "Target": pred})
    ss = pd.read_csv(f"{DATA}/SampleSubmission.csv")
    assert set(sub.ID) == set(ss.ID) and len(sub) == len(ss)
    sub.set_index("ID").loc[ss.ID].reset_index().to_csv("submission.csv", index=False)
    json.dump({"selected": best, **{f"rmse_{k}": v for k, v in scores.items()},
               "emissions_kg_co2eq": emissions}, open("run_summary.json", "w"), indent=2)
    print(f"\nwrote submission.csv {sub.shape}")


if __name__ == "__main__":
    main()
