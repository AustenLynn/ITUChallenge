"""Why did v7 not move the board?  Facts about the test regime that the replay
(2005-2012 panel) cannot contain: standardisation, anchor noise, and how far
the submission departs from interpolation per configuration."""
import numpy as np, pandas as pd
from collections import Counter
import model as M

tr, te = M.load()
cells = pd.Index(sorted(set(zip(tr.lat, tr.lon))))
tr["cell_idx"] = cells.get_indexer(pd.MultiIndex.from_arrays([tr.lat, tr.lon]))
te["cell_idx"] = cells.get_indexer(pd.MultiIndex.from_arrays([te.lat, te.lon]))
G = M.build_panel(tr, te, cells)
n_m = G.shape[1]

print("=== A. structure ===")
print("train rows", len(tr), "cells", len(cells), "months", tr.m.nunique(), "range", tr.m.min(), tr.m.max())
all_m = set(range(tr.m.min(), tr.m.max() + 1)); have = set(tr.m.unique())
miss = sorted(all_m - have)
print("train missing row months:", [(m, f"{2002 + m // 12}-{m % 12 + 1:02d}") for m in miss])
known_m = sorted({m for m in range(n_m) if np.isfinite(G[:, m]).mean() > 0.5})
gaps = sorted(set(range(4, 165)) - set(known_m))
print("panel months with TWS (>50% cells) missing inside train span:", [(m, f"{2002 + m // 12}-{m % 12 + 1:02d}") for m in gaps])

cm = tr.groupby("cell_idx").TWS_t.agg(["mean", "std", "count"])
print("per-cell TWS_t mean: quantiles", np.round(cm["mean"].quantile([0, .05, .5, .95, 1]).to_numpy(), 3))
print("per-cell TWS_t std : quantiles", np.round(cm["std"].quantile([0, .05, .5, .95, 1]).to_numpy(), 3))
print("global TWS_t mean/std", round(tr.TWS_t.mean(), 4), round(tr.TWS_t.std(), 4))
for c in M.DYN:
    print(f"  {c}: global mean {tr[c].mean():.3f} std {tr[c].std():.3f}   test mean {te[c].mean():.3f} std {te[c].std():.3f}  test nan {te[c].isna().mean():.3f}")

print("\ntest months:")
for m, g in te.groupby("m"):
    print(f"  m={m} {2002 + m // 12}-{m % 12 + 1:02d}: rows {len(g):>6}  TWS_t obs {g.TWS_t.notna().sum():>6}  masked_flag True {int(g.TWS_t_masked.sum()):>6}  SPEI_03 nan {g.SPEI_03_t.isna().sum()}")
stray = te[(te.TWS_t.notna()) & (~te.m.isin([164, 168, 173, 179, 198, 202]))]
print("stray observed cells in masked months:", len(stray), "months", sorted(stray.m.unique()))
if len(stray):
    print(stray[["m", "lat", "lon", "TWS_t", "TWS_t_masked"]].head(10).to_string())
    print("stray lat quantiles", np.round(stray.lat.quantile([0, .25, .5, .75, 1]).to_numpy(), 1))

# train target at 163 vs test TWS_t at 164
t163 = tr[tr.m == 163].set_index("cell_idx").target
o164 = te[(te.m == 164) & te.TWS_t.notna()].set_index("cell_idx").TWS_t
j = t163.to_frame("tr").join(o164.to_frame("te"), how="inner")
print(f"\ntrain target(163) vs test TWS_t(164): n={len(j)}  max|diff|={np.abs(j.tr - j.te).max():.2e}  corr={j.tr.corr(j.te):.5f}")

print("\n=== B. regime: field statistics per month ===")
ilat, ilon = M._grid_index(cells)
def roughness(v):
    """RMS of (cell - mean of E/W neighbours) over cells with both neighbours."""
    grid = np.full((ilat.max() + 1, ilon.max() + 1), np.nan, np.float32); grid[ilat, ilon] = v
    lap = grid - 0.5 * (np.roll(grid, 1, 1) + np.roll(grid, -1, 1))
    return float(np.sqrt(np.nanmean(lap ** 2)))
rows = []
for m in range(4, n_m):
    v = G[:, m]
    if np.isfinite(v).mean() < 0.5: continue
    rows.append((m, 2002 + m // 12, m % 12 + 1, float(np.nanstd(v)), float(np.nanmean(v)), roughness(v),
                 float(np.sqrt(np.nanmean((v - G[:, m - 1]) ** 2))) if np.isfinite(G[:, m - 1]).mean() > 0.5 else np.nan))
F = pd.DataFrame(rows, columns=["m", "year", "month", "std", "mean", "rough", "d1rms"])
print(F.groupby("year")[["std", "mean", "rough", "d1rms"]].mean().round(3).to_string())
print("test anchors:")
print(F[F.m >= 164].round(3).to_string())

print("\n=== C. per-cell signal-to-noise in train ===")
def lag1(ci_series):
    pass
# lag-1 autocorrelation of TWS per cell over train months (dense part <= 2010 to avoid outages)
Gd = G[:, 4:108]  # 2002-05 .. 2010-12
x = Gd - np.nanmean(Gd, 1, keepdims=True)
ac = np.nansum(x[:, 1:] * x[:, :-1], 1) / np.nansum(x ** 2, 1)
print("lag-1 autocorr quantiles", np.round(np.nanquantile(ac, [0, .05, .1, .25, .5, .75, .95]), 3))
lat = np.array([c[0] for c in cells]); lon = np.array([c[1] for c in cells])
low = ac < 0.5
print(f"cells with lag-1 autocorr < 0.5: {low.mean():.1%}; mean |lat| {np.abs(lat[low]).mean():.1f}")
# where are they?
for name, sel in [("Sahara/Arabia (15-35N, -15..60E)", (lat > 15) & (lat < 35) & (lon > -15) & (lon < 60)),
                  ("Tibet (28-40N, 75-105E)", (lat > 28) & (lat < 40) & (lon > 75) & (lon < 105)),
                  ("Amazon (-15..5, -75..-45)", (lat > -15) & (lat < 5) & (lon > -75) & (lon < -45)),
                  ("Australia interior", (lat > -30) & (lat < -20) & (lon > 120) & (lon < 140)),
                  ("Greenland", (lat > 60) & (lon > -60) & (lon < -20)),
                  ("Antarctica", lat < -60)]:
    print(f"  {name:36s} n={sel.sum():5d}  lag1 median {np.nanmedian(ac[sel]):.2f}  low-ac frac {low[sel].mean():.2f}")
np.save("cell_lag1.npy", ac)

print("\n=== D. anchor-to-anchor consistency in the test vs train ===")
# for each test anchor pair (L,R) with span s, RMS of (G[R]-G[L]) over cells, vs train pairs with same span & calendar months
pairs = [(164, 168), (168, 173), (173, 179), (179, 198), (198, 202)]
for L, R in pairs:
    d = G[:, R] - G[:, L]
    tv = [float(np.sqrt(np.nanmean((G[:, R - 12 * k] - G[:, L - 12 * k]) ** 2))) for k in range(1, 12)
          if L - 12 * k >= 4 and np.isfinite(G[:, R - 12 * k]).mean() > 0.5 and np.isfinite(G[:, L - 12 * k]).mean() > 0.5]
    print(f"  {L}->{R} (span {R - L:2d}): test rms diff {np.sqrt(np.nanmean(d ** 2)):.3f}   same calendar pair in train years: "
          f"mean {np.mean(tv):.3f}  min {np.min(tv):.3f} max {np.max(tv):.3f}  (n={len(tv)})")
    # low-ac cells vs the rest
    print(f"      low-ac cells rms {np.sqrt(np.nanmean(d[low] ** 2)):.3f}   high-ac cells rms {np.sqrt(np.nanmean(d[~low] ** 2)):.3f}")

print("\n=== E. submission vs interpolation per config ===")
sub = pd.read_csv("submission.csv").set_index("ID").loc[te.ID].Target.to_numpy()
known = np.isfinite(G)
dL, dR = M.anchors_for(known, te["cell_idx"].to_numpy(), te["m"].to_numpy() + 1, n_m)
ci, tgt = te["cell_idx"].to_numpy(), te["m"].to_numpy() + 1
last = G[ci, tgt - dL]
right = np.where(dR > 0, G[ci, np.minimum(tgt + dR, n_m - 1)], np.nan)
w = np.where(dR > 0, dL / (dL + dR), 0.0)
interp = np.where(dR > 0, (1 - w) * last + w * right, last)
dev = sub - interp
print(f"overall: rms(sub-interp) {np.sqrt(np.mean(dev ** 2)):.4f}  mean {dev.mean():+.4f}   rms(sub-last) {np.sqrt(np.mean((sub - last) ** 2)):.4f}")
cfg = pd.DataFrame({"dL": dL, "dR": dR, "dev": dev, "low": low[ci], "tgt": tgt})
g = cfg.groupby(["dL", "dR"]).agg(n=("dev", "size"), rms_dev=("dev", lambda x: np.sqrt(np.mean(x ** 2))), mean_dev=("dev", "mean"))
print(g[g.n >= 4000].round(4).to_string())
print("by low-ac vs rest: rms dev", np.sqrt(np.mean(dev[low[ci]] ** 2)).round(4), np.sqrt(np.mean(dev[~low[ci]] ** 2)).round(4))
vp = pd.read_parquet("valid_preds.parquet")
vp["dev"] = vp.pred - vp.interp
print("replay folds: rms(pred-interp) by fold:", vp.groupby("fold").dev.apply(lambda x: np.sqrt(np.mean(x ** 2))).round(4).to_dict())
print("replay folds: rmse interp / model by fold:", vp.groupby("fold").apply(lambda d: (round(float(np.sqrt(np.mean((d.target - d.interp) ** 2))), 4), round(float(np.sqrt(np.mean((d.target - d.pred) ** 2))), 4))).to_dict())
