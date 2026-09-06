"""Per-cell AR(1)+noise decomposition, and the BLUP it implies.

The idea. Every estimator so far carries an observed anchor forward: `interp`
is a time-linear blend of TWS at the two bracketing anchors. But the anchors
are noisy GRACE observations, and the target is a noisy observation whose own
noise is unpredictable, so the best achievable prediction is the *signal* at
the target month. That predictor shrinks the anchors, by an amount set by each
cell's noise fraction -- which varies enormously (lag-1 autocorrelation is 0.81
in the Amazon and 0.3-0.4 in the Sahara, Arabia and Tibet).

The decomposition. Write the observation as signal plus white noise, with the
signal an AR(1):  y_t = s_t + e_t,  Cov(s_t, s_t+k) = vs * phi^|k|.  Then the
observed autocovariance is  c_k = vs * phi^k  for k >= 1  and  c_0 = vs + ve.
Two lags identify everything:

    phi = c2 / c1        vs = c1^2 / c2        ve = c0 - vs

Both are dynamical properties of a cell, not levels, so they transfer the way
`beta_sm` does and `cell_mean` does not.

The BLUP. With anchors at dL months back and dR forward, solve the 2x2
kriging system for the weights on (y_L, y_R). Linear interpolation is the
special case ve = 0, phi = 1 -- a random walk observed without error. Every
departure from that is shrinkage this model has never had.

    .venv/bin/python -u krige.py            # baseline comparison, no training
"""
import numpy as np
import pandas as pd

import model as M

def main():
    tr, te = M.load()
    cells = pd.Index(sorted(set(zip(tr.lat, tr.lon))))
    tr["cell_idx"] = cells.get_indexer(pd.MultiIndex.from_arrays([tr.lat, tr.lon]))
    te["cell_idx"] = cells.get_indexer(pd.MultiIndex.from_arrays([te.lat, te.lon]))
    G = M.build_panel(tr, te, cells)

    phi, vs, ve = M.ar1_noise(G, upto=150)
    lam = ve / np.maximum(vs + ve, 1e-9)
    print("per-cell fit on months < 150:")
    for name, v in (("phi", phi), ("noise fraction", lam), ("vs", vs)):
        print(f"  {name:15s} quantiles {np.round(np.nanquantile(v, [.05, .25, .5, .75, .95]), 3)}")
    lat = np.array([c[0] for c in cells]); lon = np.array([c[1] for c in cells])
    for nm, sel in (("Sahara/Arabia", (lat > 15) & (lat < 35) & (lon > -15) & (lon < 60)),
                    ("Amazon", (lat > -15) & (lat < 5) & (lon > -75) & (lon < -45)),
                    ("Siberia", (lat > 50) & (lon > 60) & (lon < 180))):
        print(f"  {nm:14s} phi {np.nanmedian(phi[sel]):.3f}  noise frac {np.nanmedian(lam[sel]):.3f}")

    # --- baseline comparison on the cached late-check eval rows ---
    VAX = pd.read_parquet("cache_late_v8_c150_n100000_va.parquet")
    ci = cells.get_indexer(pd.MultiIndex.from_arrays([VAX.lat, VAX.lon]))
    y = VAX["_y"].to_numpy(); ip = VAX["_interp"].to_numpy()
    yL = VAX["last_tws"].to_numpy(); yR = VAX["right_tws"].to_numpy()
    dLv = VAX["h"].to_numpy().astype(int); dRv = np.nan_to_num(VAX["dR"].to_numpy()).astype(int)
    pk = np.empty(len(VAX))
    for (dl, dr) in sorted(set(zip(dLv.tolist(), dRv.tolist()))):
        k = (dLv == dl) & (dRv == dr)
        wL, wR, _ = M.blup_weights(phi[ci[k]], vs[ci[k]], ve[ci[k]], dl, dr)
        r = np.nan_to_num(yR[k]) if dr else 0.0
        pk[k] = np.nan_to_num(wL, nan=1.0) * yL[k] + np.nan_to_num(wR, nan=0.0) * r
    TEST_N = {"1,3": 31105, "1,4": 15683, "1,5": 15586, "2,3": 15582, "2,4": 15559, "3,2": 15547,
              "3,3": 15483, "4,2": 15409, "1,0": 15715, "2,0": 15624}
    cfg = VAX["_cfg"].to_numpy()
    def pooled(p):
        wm = wn = 0.0
        for c, n in TEST_N.items():
            s = cfg == c
            if s.any():
                wm += n * np.mean((y[s] - p[s]) ** 2); wn += n
        return np.sqrt(wm / wn)
    print(f"\nlate-check eval rows, pure baselines (no model):")
    print(f"  linear interp {pooled(ip):.4f}")
    print(f"  BLUP          {pooled(pk):.4f}")
    for alpha in (0.25, 0.5, 0.75):
        print(f"  {1-alpha:.2f}*interp + {alpha:.2f}*BLUP  {pooled((1 - alpha) * ip + alpha * pk):.4f}")
    print("  per config (n, interp, BLUP):")
    for c in sorted(TEST_N):
        s = cfg == c
        if s.any():
            print(f"    {c:>5} {s.sum():7d}  {np.sqrt(np.mean((y[s]-ip[s])**2)):.4f}  {np.sqrt(np.mean((y[s]-pk[s])**2)):.4f}")
    np.savez("ar1_params.npz", phi=phi, vs=vs, ve=ve)
    print("\nwrote ar1_params.npz")


if __name__ == "__main__":
    main()
