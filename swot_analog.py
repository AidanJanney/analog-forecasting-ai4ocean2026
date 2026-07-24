"""SWOT-driven analog helpers: swath gridding, the correlation kernel, and the
dense-GLORYS evaluation used by the SWOT→GLORYS diagnostics.

The forecaster itself now lives in :mod:`analog` (:class:`analog.AnalogForecaster`,
built from a :class:`analog.ModelLibrary` + a :class:`distances.ObsDistance`). This
module keeps the pieces that are specific to *comparing a partial swath against the
model*:

1. **Coverage** — a SWOT day is only a few partial swaths. :func:`swath_to_grid`
   bins the swath onto the model grid, yielding a coverage mask and averaging SWOT
   (~2 km, noisy) down to the grid resolution (~8 km).
2. **Quantity** — SWOT ``ssha`` is an anomaly relative to a mean sea surface while
   GLORYS ``zos`` is absolute topography; :func:`_weighted_corr` compares *patterns*
   over the observed cells, invariant to the residual offset (this kernel backs
   ``distances.CorrelationDistance``).

:func:`aggregate_skill` (verified on future swaths) and :func:`analog_vs_glorys`
(verified on dense GLORYS) both take a unified :class:`analog.AnalogForecaster`.
"""

import numpy as np

from fronts import LEVEL, lon_scale, front_distance, loop_current_front as _lc_front


def _edges(centers):
    """Cell edges from monotonically-spaced cell centers."""
    c = np.asarray(centers, dtype=float)
    mid = (c[:-1] + c[1:]) / 2
    return np.r_[c[0] - (mid[0] - c[0]), mid, c[-1] + (c[-1] - mid[-1])]


def swath_to_grid(swaths, lon, lat):
    """Bin SWOT swath points onto the model grid.

    Parameters
    ----------
    swaths : list of (lon, lat, ssha) 1-D arrays  (see swot_data.load_swaths)
    lon, lat : 1-D grid coordinate arrays (cell centers), lengths nlon / nlat.

    Returns
    -------
    obs_grid : (nlat, nlon) float array, mean SSHA per cell, NaN where unobserved.
    mask     : (nlat, nlon) bool array, True where at least one swath point fell.
    """
    lon_e, lat_e = _edges(lon), _edges(lat)
    if swaths:
        L = np.concatenate([s[0] for s in swaths])
        A = np.concatenate([s[1] for s in swaths])
        V = np.concatenate([s[2] for s in swaths])
    else:
        L = A = V = np.empty(0)
    ssum, _, _ = np.histogram2d(L, A, bins=[lon_e, lat_e], weights=V)
    count, _, _ = np.histogram2d(L, A, bins=[lon_e, lat_e])       # (nlon, nlat)

    obs_grid = np.full((lat.size, lon.size), np.nan)
    mask = np.zeros((lat.size, lon.size), dtype=bool)
    hit = count > 0
    obs_grid.T[hit] = ssum[hit] / count[hit]     # .T view maps (nlon,nlat)->grid
    mask.T[hit] = True
    return obs_grid, mask


def _weighted_corr(o, X, w):
    """Weighted Pearson correlation between vector `o` (m,) and each row of `X` (n, m)."""
    W = w.sum()
    mo = (w * o).sum() / W
    mx = (X * w).sum(axis=1) / W
    oc = o - mo
    Xc = X - mx[:, None]
    cov = (w * oc * Xc).sum(axis=1) / W
    vo = (w * oc * oc).sum() / W
    vx = (w * Xc * Xc).sum(axis=1) / W
    denom = np.sqrt(vo * vx)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(denom > 0, cov / denom, 0.0)
    return corr


def loop_current_front(af, field, ref_mean, level=LEVEL, max_points=200,
                       main_only=True):
    """Back-compat wrapper: Loop Current front of `field` using `af`'s grid/ocean.

    Delegates to :func:`fronts.loop_current_front`; kept so callers holding a
    forecaster can extract a front without threading grid arrays through.
    """
    return _lc_front(field, af.lon, af.lat, af.ocean, ref_mean, level=level,
                     max_points=max_points, main_only=main_only)


def aggregate_skill(af, obs_days, leads, swaths_for, k=15, min_cells=500,
                    deseasonalize=True):
    """Mean ACC and RMSE per lead over every available (obs-day, lead) pair,
    verified against the *future SWOT swath* (partial coverage).

    Parameters
    ----------
    af : analog.AnalogForecaster (built with a CorrelationDistance over the library).
    obs_days : list of 'YYYY-MM-DD' strings that have SWOT swaths on disk.
    leads : iterable of lead times (days).
    swaths_for : callable, ``swaths_for(['YYYY-MM-DD', ...]) -> list of swaths``.
    k, min_cells : analogs kept; minimum observed cells to use an obs/verify day.
    deseasonalize : subtract the day-of-year seasonal climatology from both the
        forecast and the SWOT truth before scoring, so skill reflects the mesoscale
        residual and the climatology baseline collapses to ~0 ACC.

    Returns (leads, acc, rmse, counts) — `acc`/`rmse` dicts keyed by
    {"analog","persistence","climatology"} of per-lead mean arrays.
    """
    def _swaths(x):
        return x[0] if isinstance(x, tuple) else x

    names = ["analog", "persistence", "climatology"]
    leads = list(leads)
    acc = {n: {L: [] for L in leads} for n in names}
    rmse = {n: {L: [] for L in leads} for n in names}
    have = set(obs_days)
    for d0 in obs_days:
        t0 = np.datetime64(d0)
        obs_grid, mask = swath_to_grid(_swaths(swaths_for([d0])), af.lon, af.lat)
        if mask.sum() < min_cells:
            continue
        persist = af.anom_of(af.forecast_obs(obs_grid, mask, 0, k=k)[0])   # analog nowcast
        for L in leads:
            vday = t0 + np.timedelta64(int(L), "D")
            if str(vday) not in have:
                continue
            obs2, mask2 = swath_to_grid(_swaths(swaths_for([str(vday)])), af.lon, af.lat)
            if mask2.sum() < min_cells:
                continue
            fc = af.forecast_obs(obs_grid, mask, int(L), k=k)[0]
            fields = {"analog": af.anom_of(fc), "persistence": persist,
                      "climatology": af.clim_anomaly(vday)}
            truth = obs2
            if deseasonalize:
                cav = af.clim_anomaly(vday)
                fields = {n: fa - cav for n, fa in fields.items()}
                truth = obs2 - cav
            for n, fa in fields.items():
                acc[n][L].append(af.score(fa, truth, mask2))
                rmse[n][L].append(af.score_rmse(fa, truth, mask2))

    def means(dct):
        return {n: np.array([np.nanmean(dct[n][L]) if dct[n][L] else np.nan
                             for L in leads]) for n in names}

    counts = [sum(np.isfinite(acc["analog"][L])) for L in leads]
    return leads, means(acc), means(rmse), counts


def analog_vs_glorys(af, obs_grid, mask, truth_surf, vday, persist_surf=None,
                     lead=14, k=10, min_sep=14, lc_level=LEVEL):
    """Score each analog's `lead`-day forecast against dense GLORYS.

    The K analogs (de-clustered `min_sep` days apart) are selected from the
    observation (`obs_grid`, `mask`); each analog's forecast ``state[a_i + lead]``
    is scored separately alongside the Gaussian-weighted ensemble mean and (if
    `persist_surf` is given) a dense persistence baseline. All fields are
    deseasonalized at `vday`. Skill: full-field ACC/RMSE and the Loop Current front
    MHD (km). Returns a dict of indices/distances, per-analog forecast anomalies,
    the truth, the truth initial condition (``init``: `persist_surf` deseasonalized at
    the obs day), ACC/RMSE/MHD per analog + ensemble + persistence, and the fronts.
    """
    sel, dist = af.analogs_obs(obs_grid, mask, lead, k, min_sep=min_sep)
    truth = af.deseasonalize(af.anom_of(truth_surf), vday)
    ocean = af.ocean

    fc_anoms = [af.deseasonalize(af.anom_of(af.state[i + lead]), vday) for i in sel]
    acc = np.array([af.score(f, truth, ocean) for f in fc_anoms])
    rmse = np.array([af.score_rmse(f, truth, ocean) for f in fc_anoms])

    w = np.exp(-(dist / dist.mean()) ** 2)
    ens = np.tensordot(w, af.state[sel + lead], axes=1) / w.sum()
    ens_anom = af.deseasonalize(af.anom_of(ens), vday)
    acc_ens = af.score(ens_anom, truth, ocean)
    rmse_ens = af.score_rmse(ens_anom, truth, ocean)

    acc_persist = rmse_persist = np.nan
    init_anom = None
    if persist_surf is not None:
        p = af.deseasonalize(af.anom_of(persist_surf), vday)
        acc_persist = af.score(p, truth, ocean)
        rmse_persist = af.score_rmse(p, truth, ocean)
        # Same field deseasonalized at its own (obs) date: the truth initial condition.
        oday = np.datetime64(vday) - np.timedelta64(int(lead), "D")
        init_anom = af.deseasonalize(af.anom_of(persist_surf), oday)

    # --- Loop Current skill: MHD (km) between forecast and truth fronts --- #
    scale = lon_scale(af.lat)
    ref_mean = float(np.nanmean(af.mean_surf[ocean]))
    deg_km = 111.195
    truth_front = loop_current_front(af, truth_surf, ref_mean, level=lc_level)

    def _lc(field):
        f = loop_current_front(af, field, ref_mean, level=lc_level)
        if len(f) == 0 or len(truth_front) == 0:
            return np.nan, f
        return front_distance(f, truth_front, scale) * deg_km, f

    lc = [_lc(af.state[i + lead]) for i in sel]
    lc_mhd = np.array([d for d, _ in lc])
    fc_fronts = [f for _, f in lc]
    lc_mhd_ens, ens_front = _lc(ens)
    lc_mhd_persist, persist_front = (_lc(persist_surf) if persist_surf is not None
                                     else (np.nan, np.empty((0, 2))))

    return dict(sel=sel, dist=dist, fc_anoms=fc_anoms, truth=truth, init=init_anom,
                acc=acc, rmse=rmse, acc_ens=acc_ens, rmse_ens=rmse_ens,
                acc_persist=acc_persist, rmse_persist=rmse_persist,
                lc_mhd=lc_mhd, lc_mhd_ens=lc_mhd_ens, lc_mhd_persist=lc_mhd_persist,
                truth_front=truth_front, fc_fronts=fc_fronts, ens_front=ens_front,
                persist_front=persist_front, lc_level=lc_level)
