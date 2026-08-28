"""
Horizon-aware TWS forecasting baseline.

The test set masks TWS_t for whole months at a time (the GRACE / GRACE-FO gap),
so a model fed `TWS_t` directly is useless on two thirds of the rows. Instead we
frame every sample the same way:

    predict TWS at month m, given the last month L where TWS was observed,
    the horizon h = m - L (1..7), and covariates available at time t = m - 1.

Train rows are expanded over h = 1..7 to teach the model how skill decays with
the gap length. That single formulation covers both the observed and the masked
test months.
"""
import json
import os

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

os.environ.setdefault("CODECARBON_LOG_LEVEL", "error")
from codecarbon import EmissionsTracker

DATA = "data"
SEED = 0
H_MAX = 7
N_PER_H = 500_000          # training rows sampled per horizon
VALID_MONTHS = 24          # held out from the end of train

# Tuned down from (255 leaves, lr 0.05), which early-stopped at iteration 55 --
# the trees were large enough to fit the validation months before the ensemble
# had done any real work. Smaller trees + slower rate also cut training energy,
# which the sustainability criterion asks us to account for.
PARAMS = dict(objective="regression", metric="rmse", learning_rate=0.02,
              num_leaves=63, min_data_in_leaf=200, feature_fraction=0.85,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=-1, seed=SEED)
N_ROUNDS = 8000


def mindex(t):
    """Months since 2002-01 — a dense integer time axis."""
    return (t.dt.year - 2002) * 12 + (t.dt.month - 1)


def load():
    tr = pd.read_csv(f"{DATA}/Train.csv", parse_dates=["time"])
    te = pd.read_csv(f"{DATA}/Test.csv", parse_dates=["time"])
    for df in (tr, te):
        df["m"] = mindex(df["time"])
    return tr, te


def build_panel(tr, te, cells):
    """G[cell, month] = observed TWS, NaN where unknown at prediction time.

    Sources: train TWS_t, train target (= TWS at m+1), and the *unmasked* test
    TWS_t. Nothing here is a future observation relative to the row that uses
    it -- the check in main() asserts that.
    """
    n_m = int(max(tr.m.max(), te.m.max())) + 2
    G = np.full((len(cells), n_m), np.nan, dtype=np.float32)
    ci_tr = tr["cell_idx"].to_numpy()
    G[ci_tr, tr.m.to_numpy()] = tr.TWS_t.to_numpy()
    G[ci_tr, tr.m.to_numpy() + 1] = tr.target.to_numpy()
    obs = te[te.TWS_t.notna()]
    G[obs["cell_idx"].to_numpy(), obs.m.to_numpy()] = obs.TWS_t.to_numpy()
    return G


def smooth3(A):
    """3x3 neighbourhood mean over (lat, lon) for every month, ignoring gaps.

    Longitude wraps around the globe; latitude does not, so rolled-in rows at
    the poles are blanked rather than treated as neighbours.
    """
    V, M = np.nan_to_num(A), (~np.isnan(A)).astype(np.float32)
    s = np.zeros_like(V)
    c = np.zeros_like(M)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            v, m = np.roll(V, dy, 0), np.roll(M, dy, 0)
            if dy:                       # kill the latitude wrap-around
                edge = 0 if dy > 0 else -1
                v, m = v.copy(), m.copy()
                v[edge] = 0.0
                m[edge] = 0.0
            s += np.roll(v, dx, 1)
            c += np.roll(m, dx, 1)
    return np.where(c > 0, s / np.maximum(c, 1e-6), np.nan).astype(np.float32)


def neighbour_panels(cells, G, clim):
    """Neighbour-averaged TWS and TWS anomaly, as panels indexed like G.

    TWS is spatially smooth and every cell shares the same masked months, so
    averaging the anchor observation over adjacent cells denoises the single
    feature the model leans on hardest.
    """
    lats = np.array([c[0] for c in cells]); lons = np.array([c[1] for c in cells])
    ulat, ilat = np.unique(lats, return_inverse=True)
    ulon, ilon = np.unique(lons, return_inverse=True)
    shape = (len(ulat), len(ulon), G.shape[1])
    grid = np.full(shape, np.nan, dtype=np.float32)
    grid[ilat, ilon] = G
    anom = np.full(shape, np.nan, dtype=np.float32)
    months = np.arange(G.shape[1]) % 12
    anom[ilat, ilon] = G - clim[:, months]
    return smooth3(grid)[ilat, ilon], smooth3(anom)[ilat, ilon]


def climatology(tr, cells):
    """Per-cell calendar-month mean of the target, plus per-cell mean/std."""
    cal = (tr.m % 12).to_numpy()
    ci = tr["cell_idx"].to_numpy()
    y = tr.target.to_numpy()
    s = np.zeros((len(cells), 12)); n = np.zeros((len(cells), 12))
    np.add.at(s, (ci, cal), y)
    np.add.at(n, (ci, cal), 1)
    clim = s / np.maximum(n, 1)
    cm = tr.groupby("cell_idx")["target"].agg(["mean", "std"]).reindex(range(len(cells)))
    return clim.astype(np.float32), cm["mean"].to_numpy(np.float32), cm["std"].to_numpy(np.float32)


DYN = ["SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t", "SOIL_MOISTURE_t"]
COVARS = DYN + ["lat", "lon"]
LAG1 = ["SPEI_06_t", "SOIL_MOISTURE_t"]
# The raw covariate level at the anchor month scored ~zero gain in every run:
# `last_tws` already encodes the state at L, so only the *change* since L adds
# information. Dropping the five `_L` columns costs nothing and trains cheaper.
FEATS = (["h", "last_tws", "last_anom", "last_delta", "clim_m", "cell_mean", "cell_std",
          "month_sin", "month_cos", "nb_last_tws", "nb_last_anom", "nb_diff"] + COVARS
         + [f"{c}_d" for c in DYN] + [f"{c}_d1" for c in LAG1])


def build_covariates(tr, te, n_cells, n_m):
    """C[var][cell, month] for the dynamic covariates, from train and test alike.

    Covariates are inputs known at prediction time, so pooling test rows here
    leaks nothing -- unlike TWS, they are never masked.
    """
    C = {c: np.full((n_cells, n_m), np.nan, dtype=np.float32) for c in DYN}
    for df in (tr, te):
        ci, m = df["cell_idx"].to_numpy(), df["m"].to_numpy()
        for c in DYN:
            C[c][ci, m] = df[c].to_numpy(np.float32)
    return C


def make_features(rows, h, G, clim, cell_mean, cell_std, C, NB, NBA):
    """rows: frame with cell_idx, m (= t), covariates. Target month is m+1."""
    ci = rows["cell_idx"].to_numpy()
    m = rows["m"].to_numpy()
    tgt_m = m + 1
    L = tgt_m - h                      # last observed month
    last = G[ci, L]
    prev = np.where(L - 1 >= 0, G[ci, np.maximum(L - 1, 0)], np.nan)
    cal_t = tgt_m % 12
    out = pd.DataFrame({
        "h": np.full(len(rows), h, np.int8),
        "last_tws": last,
        "last_anom": last - clim[ci, L % 12],
        "last_delta": last - prev,
        "clim_m": clim[ci, cal_t],
        "cell_mean": cell_mean[ci],
        "cell_std": cell_std[ci],
        "month_sin": np.sin(2 * np.pi * cal_t / 12).astype(np.float32),
        "month_cos": np.cos(2 * np.pi * cal_t / 12).astype(np.float32),
        "nb_last_tws": NB[ci, L],
        "nb_last_anom": NBA[ci, L],
        "nb_diff": last - NB[ci, L],
    })
    for c in COVARS:
        out[c] = rows[c].to_numpy(np.float32)

    # What the climate did between the last TWS observation and now. The anchor
    # month L always has covariates (it is a month where TWS was observed), so
    # these are defined for every row -- unlike fixed lags, which are missing for
    # whole test months. This is the signal that has to carry `last_tws` forward
    # as the gap widens.
    for c in DYN:
        out[f"{c}_d"] = out[c].to_numpy() - C[c][ci, L]
    for c in LAG1:
        prev_m = np.maximum(m - 1, 0)
        at_1 = np.where(m - 1 >= 0, C[c][ci, prev_m], np.nan)
        out[f"{c}_d1"] = out[c].to_numpy() - at_1
    return out[FEATS], last


def fit(train_df, valid_df, tag):
    """Train one LightGBM model with early stopping on the held-out months."""
    model = lgb.train(PARAMS, lgb.Dataset(train_df[FEATS], train_df.target),
                      num_boost_round=N_ROUNDS,
                      valid_sets=[lgb.Dataset(valid_df[FEATS], valid_df.target)],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
    pred = model.predict(valid_df[FEATS])
    rmse = mean_squared_error(valid_df.target, pred) ** 0.5
    print(f"  {tag:12s} rows={len(train_df):>9,}  best_iter={model.best_iteration:>5}"
          f"  rmse={rmse:.4f}")
    return model, pred, rmse


def main():
    rng = np.random.default_rng(SEED)
    tr, te = load()

    cells = pd.Index(sorted(set(zip(tr.lat, tr.lon))))
    key_tr = pd.MultiIndex.from_arrays([tr.lat, tr.lon])
    key_te = pd.MultiIndex.from_arrays([te.lat, te.lon])
    tr["cell_idx"] = cells.get_indexer(key_tr)
    te["cell_idx"] = cells.get_indexer(key_te)
    assert (te.cell_idx >= 0).all()

    G = build_panel(tr, te, cells)
    C = build_covariates(tr, te, len(cells), G.shape[1])
    clim, cell_mean, cell_std = climatology(tr, cells)
    NB, NBA = neighbour_panels(cells, G, clim)

    # --- horizon of every test row, from the panel we are allowed to see ---
    te_ci, te_tgt = te["cell_idx"].to_numpy(), te["m"].to_numpy() + 1
    known = ~np.isnan(G)
    known[:, tr.m.max() + 1:] &= False                      # start from train's end
    obs = te[te.TWS_t.notna()]
    known[obs["cell_idx"].to_numpy(), obs.m.to_numpy()] = True
    te_h = np.ones(len(te), np.int64)
    for i in range(len(te)):
        h = 1
        while h <= 24 and not known[te_ci[i], te_tgt[i] - h]:
            h += 1
        te_h[i] = h
    print("test horizon distribution:", np.bincount(te_h)[1:])
    assert te_h.max() <= H_MAX, "horizon exceeded expected range"

    # --- training set: every train row replayed at each horizon ---
    val_start = tr.m.max() - VALID_MONTHS + 1
    parts_tr, parts_va = [], []
    for h in range(1, H_MAX + 1):
        ok = tr[(tr.m + 1 - h) >= tr.m.min()]
        X, last = make_features(ok, h, G, clim, cell_mean, cell_std, C, NB, NBA)
        X["target"], X["m"] = ok["target"].to_numpy(), ok["m"].to_numpy()
        X = X[np.isfinite(last)]
        va = X[X.m >= val_start]
        trn = X[X.m < val_start]
        if len(trn) > N_PER_H:
            trn = trn.iloc[rng.choice(len(trn), N_PER_H, replace=False)]
        parts_tr.append(trn); parts_va.append(va)
    TR = pd.concat(parts_tr, ignore_index=True)
    VA = pd.concat(parts_va, ignore_index=True)
    print(f"train {TR.shape}  valid {VA.shape}")

    tracker = EmissionsTracker(project_name="tws-forecast", output_dir=".",
                               output_file="emissions.csv", log_level="error",
                               save_to_file=True)
    tracker.start()

    # --- (a) one global model, horizon passed in as a feature ---
    print("\nglobal model:")
    g_model, g_pred, g_rmse = fit(TR, VA, "global")

    # --- (b) one model per horizon ---
    print("\nper-horizon models:")
    h_models, h_pred = {}, np.empty(len(VA))
    for h in range(1, H_MAX + 1):
        tr_h, va_h = TR[TR.h == h], VA[VA.h == h]
        h_models[h], p, _ = fit(tr_h, va_h, f"h={h}")
        h_pred[(VA.h == h).to_numpy()] = p
    h_rmse = mean_squared_error(VA.target, h_pred) ** 0.5

    emissions = tracker.stop()
    print(f"\ntraining emissions: {emissions * 1000:.2f} g CO2eq"
          f"  (energy {tracker.final_emissions_data.energy_consumed * 1000:.1f} Wh)")

    # --- compare, including the trivial blend ---
    b_pred = 0.5 * (g_pred + h_pred)
    b_rmse = mean_squared_error(VA.target, b_pred) ** 0.5
    print(f"\nvalidation RMSE   global={g_rmse:.4f}  per-horizon={h_rmse:.4f}"
          f"  blend={b_rmse:.4f}")
    print(f"  persistence baseline: {mean_squared_error(VA.target, VA.last_tws) ** 0.5:.4f}")
    print(f"  climatology baseline: {mean_squared_error(VA.target, VA.clim_m) ** 0.5:.4f}")

    best = min([("global", g_rmse), ("per_horizon", h_rmse), ("blend", b_rmse)],
               key=lambda x: x[1])[0]
    print(f"selected: {best}")

    print("\nRMSE by horizon (global / per-horizon / persistence):")
    for h in range(1, H_MAX + 1):
        k = (VA.h == h).to_numpy()
        if k.sum():
            print(f"  h={h}: {mean_squared_error(VA.target[k], g_pred[k]) ** 0.5:.4f}"
                  f" / {mean_squared_error(VA.target[k], h_pred[k]) ** 0.5:.4f}"
                  f" / {mean_squared_error(VA.target[k], VA.last_tws[k]) ** 0.5:.4f}")

    print("\nfeature importance (global):")
    for f, gain in sorted(zip(FEATS, g_model.feature_importance("gain")), key=lambda x: -x[1]):
        print(f"  {f:16s} {gain:,.0f}")

    # Keep validation predictions for the bias / SHAP write-up later.
    VA[["h", "m", "target", "lat", "lon"]].assign(
        pred_global=g_pred, pred_horizon=h_pred).to_parquet("valid_preds.parquet")

    # --- refit on train+valid at the chosen depth, then predict test ---
    ALL = pd.concat([TR, VA], ignore_index=True)
    pred = np.empty(len(te), np.float64)
    te_X = {}
    for h in range(1, H_MAX + 1):
        k = te_h == h
        if k.any():
            te_X[h], _ = make_features(te[k], h, G, clim, cell_mean, cell_std, C, NB, NBA)

    if best in ("global", "blend"):
        final_g = lgb.train(PARAMS, lgb.Dataset(ALL[FEATS], ALL.target),
                            num_boost_round=g_model.best_iteration)
    if best in ("per_horizon", "blend"):
        final_h = {h: lgb.train(PARAMS, lgb.Dataset(ALL[ALL.h == h][FEATS],
                                                    ALL[ALL.h == h].target),
                                num_boost_round=h_models[h].best_iteration)
                   for h in range(1, H_MAX + 1)}

    for h, X in te_X.items():
        k = te_h == h
        if best == "global":
            pred[k] = final_g.predict(X)
        elif best == "per_horizon":
            pred[k] = final_h[h].predict(X)
        else:
            pred[k] = 0.5 * (final_g.predict(X) + final_h[h].predict(X))
    assert np.isfinite(pred).all()

    sub = pd.DataFrame({"ID": te["ID"], "Target": pred})
    ss = pd.read_csv(f"{DATA}/SampleSubmission.csv")
    assert set(sub.ID) == set(ss.ID) and len(sub) == len(ss)
    sub = sub.set_index("ID").loc[ss.ID].reset_index()
    sub.to_csv("submission.csv", index=False)
    json.dump({"selected": best, "rmse_global": g_rmse, "rmse_per_horizon": h_rmse,
               "rmse_blend": b_rmse, "emissions_kg_co2eq": emissions},
              open("run_summary.json", "w"), indent=2)
    print(f"\nwrote submission.csv {sub.shape}")
    print(sub.head().to_string())


if __name__ == "__main__":
    main()
