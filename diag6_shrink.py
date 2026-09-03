"""Does the noisy late-mission regime change what the optimal predictor is?
Uses only real labels: the train panel by year (2015 = late regime) and the six
observed test months.  Measures, per period and per cell group:
  - the regression slope of the anomaly at R on the anomaly at L (attenuation
    from anchor noise shows up as a lower slope),
  - RMSE of persistence / shrunk persistence / neighbour-smoothed persistence,
  - how much of the TWS change the same-month soil-moisture change explains."""
import numpy as np, pandas as pd
import model as M

tr, te = M.load()
cells = pd.Index(sorted(set(zip(tr.lat, tr.lon))))
tr["cell_idx"] = cells.get_indexer(pd.MultiIndex.from_arrays([tr.lat, tr.lon]))
te["cell_idx"] = cells.get_indexer(pd.MultiIndex.from_arrays([te.lat, te.lon]))
G = M.build_panel(tr, te, cells)
C = M.build_covariates(tr, te, len(cells), G.shape[1])
n_m = G.shape[1]
clim = M.climatology(tr[tr.m < 152], cells)           # 2002-2014 climatology, excludes 2015
A = G - clim[:, np.arange(n_m) % 12]
ac = np.load("cell_lag1.npy"); low = ac < 0.5
NB = M.smooth_panel(cells, A, 1); NB5 = M.smooth_panel(cells, A, 2); NB7 = M.smooth_panel(cells, A, 3)

def ok_month(m): return 0 <= m < n_m and np.isfinite(G[:, m]).mean() > 0.5
def rm(x): return float(np.sqrt(np.nanmean(x ** 2)))

def report(pairs, label):
    """pairs: list of (L, R). Pool cells over pairs."""
    xs, ys, ns, n5, n7, grp = [], [], [], [], [], []
    for L, R in pairs:
        if not (ok_month(L) and ok_month(R)): continue
        xs.append(A[:, L]); ys.append(A[:, R]); ns.append(NB[:, L]); n5.append(NB5[:, L]); n7.append(NB7[:, L]); grp.append(low)
    if not xs: return
    x, y, nb, nb5, nb7, g = map(np.concatenate, (xs, ys, ns, n5, n7, grp))
    k = np.isfinite(x) & np.isfinite(y)
    x, y, nb, nb5, nb7, g = x[k], y[k], nb[k], nb5[k], nb7[k], g[k]
    def stats(sel, name):
        xx, yy = x[sel], y[sel]
        slope = float((xx * yy).sum() / (xx * xx).sum())
        nbb, nbb5, nbb7 = nb[sel], nb5[sel], nb7[sel]
        s3 = float((nbb * yy).sum() / (nbb * nbb).sum())
        print(f"    {name:8s} n={sel.sum():6d} rms(y) {rm(yy):.3f} rms(x) {rm(xx):.3f} corr {np.corrcoef(xx, yy)[0, 1]:.3f} slope {slope:.3f} | "
              f"rmse: persist {rm(yy - xx):.3f} zero {rm(yy):.3f} shrunk {rm(yy - slope * xx):.3f} nb3 {rm(yy - nbb):.3f} nb5 {rm(yy - nbb5):.3f} nb7 {rm(yy - nbb7):.3f} nb3*s {rm(yy - s3 * nbb):.3f}")
    print(f"  {label} ({len(xs)} pairs)")
    stats(np.ones(len(x), bool), "all"); stats(g, "low-ac"); stats(~g, "high-ac")

print("=== 1-month persistence (L -> L+1), by period ===")
for name, months in [("2005-2007", range(36, 72)), ("2008-2010", range(72, 108)), ("2012-2014", range(120, 156)),
                     ("2015 (train, late regime)", range(152, 164))]:
    report([(m, m + 1) for m in months], name)

print("\n=== 4-6 month spans: same calendar pairs as the test anchors, by period ===")
for (L, R) in [(164, 168), (168, 173), (173, 179), (198, 202)]:
    s = R - L
    early = [(L - 12 * k, R - 12 * k) for k in range(6, 13)]   # 2003-2009
    mid = [(L - 12 * k, R - 12 * k) for k in range(1, 6)]      # 2010-2015
    print(f"-- span {s} ({L % 12 + 1:02d} -> {R % 12 + 1:02d}) --")
    report(early, "train 2003-09"); report(mid, "train 2010-15"); report([(L, R)], f"TEST {L}->{R}")

print("\n=== 1-month change explained by same-month soil-moisture change, by year ===")
S = C["SOIL_MOISTURE_t"]
for y0 in range(2003, 2016):
    ms = [m for m in range((y0 - 2002) * 12, (y0 - 2001) * 12) if ok_month(m) and ok_month(m + 1)]
    dG = np.concatenate([A[:, m + 1] - A[:, m] for m in ms]); dS = np.concatenate([S[:, m + 1] - S[:, m] for m in ms])
    dSm1 = np.concatenate([S[:, m] - S[:, m - 1] for m in ms]); dSp1 = np.concatenate([S[:, m + 2] - S[:, m + 1] for m in ms])
    grp = np.concatenate([low for _ in ms])
    k = np.isfinite(dG) & np.isfinite(dS) & np.isfinite(dSm1) & np.isfinite(dSp1)
    c0, cm, cp = np.corrcoef(dG[k], dS[k])[0, 1], np.corrcoef(dG[k], dSm1[k])[0, 1], np.corrcoef(dG[k], dSp1[k])[0, 1]
    kh = k & ~grp
    print(f"  {y0}: rms dTWS {rm(dG[k]):.3f}  corr(dTWS, dSM lag -1/0/+1) {cm:.3f} / {c0:.3f} / {cp:.3f}   high-ac cells: {np.corrcoef(dG[kh], dS[kh])[0, 1]:.3f}  low-ac: {np.corrcoef(dG[k & grp], dS[k & grp])[0, 1]:.3f}")

print("\n=== test anchor pairs: change explained by soil-moisture change over the same span ===")
for (L, R) in [(164, 168), (168, 173), (173, 179), (179, 198), (198, 202)]:
    dG, dS = A[:, R] - A[:, L], S[:, R] - S[:, L]
    k = np.isfinite(dG) & np.isfinite(dS)
    ref = []
    for kk in range(1, 12):
        l, r = L - 12 * kk, R - 12 * kk
        if l >= 4 and ok_month(l) and ok_month(r):
            d1, d2 = A[:, r] - A[:, l], S[:, r] - S[:, l]; q = np.isfinite(d1) & np.isfinite(d2)
            ref.append(np.corrcoef(d1[q], d2[q])[0, 1])
    print(f"  {L}->{R}: corr(dTWS, dSM) test {np.corrcoef(dG[k], dS[k])[0, 1]:.3f}   same pair in train years mean {np.mean(ref):.3f} (min {np.min(ref):.3f} max {np.max(ref):.3f})")

print("\n=== stray observed cells: are they copies of a neighbouring month? ===")
stray = te[te.TWS_t.notna() & te.TWS_t_masked.eq(False) & ~te.m.isin([164, 168, 173, 179, 198, 202])]
ci, m = stray["cell_idx"].to_numpy(), stray.m.to_numpy()
prevL = np.array([max(a for a in [164, 168, 173, 179, 198, 202] if a < mm) for mm in m])
print("  n", len(stray), " |TWS - TWS at previous anchor| quantiles", np.round(np.nanquantile(np.abs(stray.TWS_t.to_numpy() - G[ci, prevL]), [0, .25, .5, .75, 1]), 3))
print("  |TWS| quantiles", np.round(np.quantile(np.abs(stray.TWS_t), [0, .25, .5, .75, 1]), 3), "  lag1 ac of those cells median", round(float(np.nanmedian(ac[ci])), 2))
