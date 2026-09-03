"""Calibration submissions for the leaderboard, plus the same baselines on the
replay folds so the board numbers can be read against known quantities.

  submission_interp.csv      linear interpolation between anchors (persistence
                             where there is no right anchor)
  submission_seasonal.csv    climatology at the target month + interpolated
                             anomaly (2002-2015 climatology)
  submission_hybrid_long.csv v7 on the configs where fold 123 shows a gain
                             (dR >= 12 or dR == 0), interpolation elsewhere
"""
import numpy as np, pandas as pd
import model as M

tr, te = M.load()
cells = pd.Index(sorted(set(zip(tr.lat, tr.lon))))
tr["cell_idx"] = cells.get_indexer(pd.MultiIndex.from_arrays([tr.lat, tr.lon]))
te["cell_idx"] = cells.get_indexer(pd.MultiIndex.from_arrays([te.lat, te.lon]))
G = M.build_panel(tr, te, cells)
n_m = G.shape[1]
known = np.isfinite(G)

def baselines(rows_ci, tgt, dL, dR, Gp, clim):
    L = tgt - dL; R = tgt + np.where(dR > 0, dR, 0)
    last = Gp[rows_ci, L]; right = np.where(dR > 0, Gp[rows_ci, R], np.nan)
    w = np.where(dR > 0, dL / (dL + dR), 0.0)
    lin = np.where(dR > 0, (1 - w) * last + w * right, last)
    cl_t, cl_L, cl_R = clim[rows_ci, tgt % 12], clim[rows_ci, L % 12], clim[rows_ci, R % 12]
    seas = cl_t + np.where(dR > 0, (1 - w) * (last - cl_L) + w * (right - cl_R), last - cl_L)
    return lin, seas

# ---- test ----
ci, tgt = te["cell_idx"].to_numpy(), te["m"].to_numpy() + 1
dL, dR = M.anchors_for(known, ci, tgt, n_m)
clim_all = M.climatology(tr, cells)
lin, seas = baselines(ci, tgt, dL, dR, G, clim_all)
assert np.isfinite(lin).all() and np.isfinite(seas).all()
v7 = pd.read_csv("submission.csv").set_index("ID").loc[te.ID].Target.to_numpy()
use_v7 = (dR >= 12) | (dR == 0)
hyb = np.where(use_v7, v7, lin)
ss = pd.read_csv(f"{M.DATA}/SampleSubmission.csv")
for name, p in [("submission_interp.csv", lin), ("submission_seasonal.csv", seas), ("submission_hybrid_long.csv", hyb)]:
    sub = pd.DataFrame({"ID": te["ID"], "Target": p}).set_index("ID").loc[ss.ID].reset_index()
    sub.to_csv(name, index=False)
    print(f"wrote {name}: rms vs v7 {np.sqrt(np.mean((p - v7) ** 2)):.4f}")
print(f"hybrid uses v7 on {use_v7.mean():.1%} of rows; rms(seasonal - interp) {np.sqrt(np.mean((seas - lin) ** 2)):.4f}")

# ---- replay folds: linear vs seasonal interp per config ----
def rm(a): return float(np.sqrt(np.mean(a ** 2)))
for b in (68, 123):
    keep = np.zeros(n_m, bool); keep[:b] = True; keep[[b + o for o in M.OFFS_ANCH]] = True
    Gm = G.copy(); Gm[:, ~keep] = np.nan
    clim_b = M.climatology(tr[tr.m < b], cells)
    rep = tr[tr.m.isin([b + o for o in M.OFFS_ROWS])]
    rep = rep[np.isfinite(G[rep["cell_idx"].to_numpy(), rep.m.to_numpy() + 1])]
    rci, rt = rep["cell_idx"].to_numpy(), rep.m.to_numpy() + 1
    rdL, rdR = M.anchors_for(np.isfinite(Gm), rci, rt, n_m)
    y = rep.target.to_numpy()
    l, s = baselines(rci, rt, rdL, rdR, Gm, clim_b)
    ok = np.isfinite(l) & np.isfinite(s)
    print(f"\nfold b={b}: n={ok.sum()}  linear {rm(y[ok] - l[ok]):.4f}  seasonal {rm(y[ok] - s[ok]):.4f}  persistence {rm(y[ok] - Gm[rci, rt - rdL][ok]):.4f}")
    df = pd.DataFrame({"dL": rdL, "dR": rdR, "e_lin": (y - l) ** 2, "e_seas": (y - s) ** 2})[ok]
    g = df.groupby(["dL", "dR"]).agg(n=("e_lin", "size"), lin=("e_lin", lambda x: np.sqrt(x.mean())), seas=("e_seas", lambda x: np.sqrt(x.mean())))
    print(g[g.n >= 4000].round(4).to_string())

# ---- 2015 per-month noise ----
print("\nper-month 1-step RMS change in 2014-2015 (dataset labels) and E-W roughness:")
ilat, ilon = M._grid_index(cells)
for m in range(144, 165):
    if np.isfinite(G[:, m]).mean() < 0.5 or np.isfinite(G[:, m - 1]).mean() < 0.5: continue
    grid = np.full((ilat.max() + 1, ilon.max() + 1), np.nan, np.float32); grid[ilat, ilon] = G[:, m]
    lap = grid - 0.5 * (np.roll(grid, 1, 1) + np.roll(grid, -1, 1))
    print(f"  {m} {2002 + m // 12}-{m % 12 + 1:02d}: d1rms {np.sqrt(np.nanmean((G[:, m] - G[:, m - 1]) ** 2)):.3f}   rough {np.sqrt(np.nanmean(lap ** 2)):.4f}   std {np.nanstd(G[:, m]):.3f}")
