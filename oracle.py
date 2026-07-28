"""Analog skill-ceiling (oracle) diagnostic.

Before investing in a learned distance metric we test whether the bottleneck is
the *metric* or the *predictability ceiling*. We compute four deseasonalized
(mesoscale) skill-vs-lead curves over the GLORYS library:

* persistence     — the current state carried forward.
* metric-analog   — analogs chosen by the front-MHD distance (the current pipeline).
* present-oracle  — analogs chosen by the best *current* full-field similarity
                    (the ceiling for any metric that only sees the current obs).
* future-oracle   — analogs chosen by the best *future* similarity (the absolute
                    ceiling of the analog library; uses the truth, so it cheats).

Reading it: if future-oracle ≈ persistence there is no headroom (predictability
ceiling) and a better metric cannot help. If present-oracle > metric-analog there
is room a smarter current-based (ML) metric could recover.
"""

import numpy as np

from analog import seasonal_climatology
from swot_analog import _weighted_corr


def _flatten(state, times, latitude):
    """Deseasonalized anomalies over ocean cells + latitude weights (flattened)."""
    A = np.asarray(state, dtype=np.float32) - \
        seasonal_climatology(state, times).astype(np.float32)
    ocean = np.isfinite(A[0])
    X = A[:, ocean]                                   # (n, m)
    w = np.broadcast_to(np.cos(np.deg2rad(latitude)).astype(np.float32)[:, None],
                        A.shape[1:])[ocean]           # (m,)
    return X, w


def similarity_matrix(X, w):
    """Weighted spatial correlation between every pair of states (n, n), diag = 1."""
    W = w.sum()
    mu = (X * w).sum(1, keepdims=True) / W
    Xc = (X - mu) * np.sqrt(w)
    norm = np.sqrt((Xc ** 2).sum(1, keepdims=True))
    Xhat = np.divide(Xc, norm, out=np.zeros_like(Xc), where=norm > 0)  # 0 if no variance
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return Xhat @ Xhat.T                          # float32 BLAS may warn spuriously


def _ensemble(X, sel, dist, lead, w):
    """Gaussian-weighted mean of the selected analogs advanced `lead` steps."""
    wt = np.exp(-(dist / dist.mean()) ** 2)
    return (wt[:, None] * X[sel + lead]).sum(0) / wt.sum()


def ceiling_curves(state, times, latitude, D, leads, k=10, exclude=30, stride=20):
    """Per-lead mean deseasonalized ACC for the four forecast strategies."""
    X, w = _flatten(state, times, latitude)
    S = similarity_matrix(X, w)                       # full-field current similarity
    n = X.shape[0]
    names = ["persistence", "metric-analog", "present-oracle", "future-oracle"]
    out = {nm: [] for nm in names}
    for L in leads:
        acc = {nm: [] for nm in names}
        for t0 in range(0, n, stride):
            if t0 + L >= n:
                continue
            cand = np.array([a for a in range(n)
                             if a + L < n and abs(a - t0) > exclude])
            if len(cand) < k:
                continue
            truth = X[t0 + L]
            acc["persistence"].append(float(S[t0, t0 + L]))

            sm = cand[np.argsort(D[t0, cand])][:k]                 # front-MHD
            fc = _ensemble(X, sm, D[t0, sm], L, w)
            acc["metric-analog"].append(_weighted_corr(truth, fc[None], w)[0])

            sp = cand[np.argsort(-S[t0, cand])][:k]                # best current sim
            fc = _ensemble(X, sp, 1.0 - S[t0, sp], L, w)
            acc["present-oracle"].append(_weighted_corr(truth, fc[None], w)[0])

            sf = cand[np.argsort(-S[t0 + L, cand + L])][:k]        # best future sim
            fc = _ensemble(X, sf, 1.0 - S[t0 + L, sf + L], L, w)
            acc["future-oracle"].append(_weighted_corr(truth, fc[None], w)[0])
        for nm in names:
            out[nm].append(np.nanmean(acc[nm]) if acc[nm] else np.nan)
    return list(leads), out
