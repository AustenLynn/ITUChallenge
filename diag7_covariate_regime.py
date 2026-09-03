"""Which covariate changed in 2012? Field statistics and cross-correlations of
soil moisture and SPEI by year, against TWS and against each other."""
import numpy as np, pandas as pd
import model as M

tr, te = M.load()
cells = pd.Index(sorted(set(zip(tr.lat, tr.lon))))
tr["cell_idx"] = cells.get_indexer(pd.MultiIndex.from_arrays([tr.lat, tr.lon]))
te["cell_idx"] = cells.get_indexer(pd.MultiIndex.from_arrays([te.lat, te.lon]))
G = M.build_panel(tr, te, cells)
C = M.build_covariates(tr, te, len(cells), G.shape[1])
n_m = G.shape[1]
clim = M.climatology(tr[tr.m < 152], cells)
A = G - clim[:, np.arange(n_m) % 12]
ac = np.load("cell_lag1.npy"); hi = ac >= 0.5
ilat, ilon = M._grid_index(cells)

def cc(a, b, sel=None):
    k = np.isfinite(a) & np.isfinite(b)
    if sel is not None: k &= sel
    return np.corrcoef(a[k], b[k])[0, 1] if k.sum() > 100 else np.nan
def rough(v):
    grid = np.full((ilat.max() + 1, ilon.max() + 1), np.nan, np.float32); grid[ilat, ilon] = v
    lap = grid - 0.5 * (np.roll(grid, 1, 1) + np.roll(grid, -1, 1))
    return float(np.sqrt(np.nanmean(lap ** 2)))
def okm(m): return 0 <= m < n_m and np.isfinite(G[:, m]).mean() > 0.5 and np.isfinite(C["SOIL_MOISTURE_t"][:, m]).mean() > 0.5

S, P1, P3, P12 = C["SOIL_MOISTURE_t"], C["SPEI_01_t"], C["SPEI_03_t"], C["SPEI_12_t"]
print("year | SM: std rough lag1 | corr(SM,SPEI01) corr(SM,SPEI03) | corr(A_tws,SM) corr(A_tws,SPEI12) | dTWS~dSM dTWS~dSPEI01 dTWS~dSPEI03 [high-ac cells]")
for y0 in list(range(2003, 2016)) + ["test"]:
    if y0 == "test":
        ms = [164, 168, 173, 179, 198, 202]
    else:
        ms = [m for m in range((y0 - 2002) * 12, (y0 - 2001) * 12) if okm(m)]
    sm = np.concatenate([S[:, m] for m in ms]); p1 = np.concatenate([P1[:, m] for m in ms]); p3 = np.concatenate([P3[:, m] for m in ms])
    p12 = np.concatenate([P12[:, m] for m in ms]); a = np.concatenate([A[:, m] for m in ms]); h = np.concatenate([hi for _ in ms])
    lag1 = np.nanmean([cc(S[:, m], S[:, m - 1]) for m in ms if okm(m - 1)])
    rg = np.nanmean([rough(S[:, m]) for m in ms])
    pairs = [m for m in ms if okm(m + 1)]
    if pairs:
        dG = np.concatenate([A[:, m + 1] - A[:, m] for m in pairs]); dS = np.concatenate([S[:, m + 1] - S[:, m] for m in pairs])
        d1 = np.concatenate([P1[:, m + 1] - P1[:, m] for m in pairs]); d3 = np.concatenate([P3[:, m + 1] - P3[:, m] for m in pairs])
        hh = np.concatenate([hi for _ in pairs])
        dd = f"{cc(dG, dS, hh):.3f} {cc(dG, d1, hh):.3f} {cc(dG, d3, hh):.3f}"
    else:
        dd = "  -     -     -"
    print(f"{str(y0):>5} | {np.nanstd(sm):.3f} {rg:.4f} {lag1:.3f} | {cc(sm, p1):.3f} {cc(sm, p3):.3f} | {cc(a, sm, h):.3f} {cc(a, p12, h):.3f} | {dd}")

print("\nper-month SM field std and SM~SPEI03 correlation around the break (2011-2013):")
for m in range(104, 140):
    if not okm(m): continue
    print(f"  {2002 + m // 12}-{m % 12 + 1:02d}: std {np.nanstd(S[:, m]):.3f}  rough {rough(S[:, m]):.4f}  corr(SM,SPEI03) {cc(S[:, m], P3[:, m]):.3f}  corr(dTWS,dSM) {cc(A[:, m + 1] - A[:, m], S[:, m + 1] - S[:, m], hi) if okm(m + 1) else float('nan'):.3f}")

print("\nregional dTWS~dSM by period (high-ac cells only):")
lat = np.array([c[0] for c in cells]); lon = np.array([c[1] for c in cells])
regions = {"N America": (lat > 25) & (lat < 60) & (lon > -130) & (lon < -60), "S America": (lat > -40) & (lat < 12) & (lon > -82) & (lon < -35),
           "Europe": (lat > 35) & (lat < 70) & (lon > -10) & (lon < 40), "Africa S of 15N": (lat > -35) & (lat < 15) & (lon > -20) & (lon < 52),
           "S/SE Asia": (lat > 5) & (lat < 35) & (lon > 60) & (lon < 120), "Siberia": (lat > 50) & (lon > 60) & (lon < 180), "Australia": (lat > -45) & (lat < -10) & (lon > 110) & (lon < 155)}
periods = {"2004-07": range(24, 72), "2008-11": range(72, 120), "2012-15": range(120, 164)}
print(f"{'region':16s}" + "".join(f"{p:>10s}" for p in periods))
for rn, rs in regions.items():
    vals = []
    for pn, pr in periods.items():
        pairs = [m for m in pr if okm(m) and okm(m + 1)]
        dG = np.concatenate([A[:, m + 1] - A[:, m] for m in pairs]); dS = np.concatenate([S[:, m + 1] - S[:, m] for m in pairs])
        sel = np.concatenate([rs & hi for _ in pairs])
        vals.append(cc(dG, dS, sel))
    print(f"{rn:16s}" + "".join(f"{v:10.3f}" for v in vals))
