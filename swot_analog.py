"""SWOT-driven analog helpers: swath gridding, the correlation kernel, and the
dense-GLORYS scoring used by the SWOT→GLORYS diagnostics.

Selection and forecasting live in :mod:`analog` (:class:`analog.AnalogSelector`,
:class:`analog.Forecast`); this module keeps only the SWOT-specific pieces:

* :func:`swath_to_grid` — bin a partial swath onto the model grid (+ coverage mask);
* :func:`_weighted_corr` — the offset-invariant pattern-correlation kernel (backs
  ``distances.CorrelationDistance`` and the ACC/RMSE error metrics);
* :func:`aggregate_skill` — mean ACC/RMSE per lead verified on *future SWOT swaths*;
* :func:`evaluate` — score an already-selected, already-forecast analog set against
  a *dense* GLORYS field (per-analog + ensemble + persistence ACC/RMSE + Loop Current
  front MHD). It neither selects nor combines — the caller does
  ``sel = selector.select(...)`` then ``ens = forecaster.forecast(sel, lead)``.
"""

import numpy as np

from fronts import LEVEL, lon_scale, front_distance, loop_current_front
from metrics import SpatialCorrelation, DemeanedRMSE


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


def front_coverage(front, mask, lon, lat):
    """Fraction of a front's grid cells that fall inside the swath `mask`.

    Rasterizes the front's (lon, lat) vertices onto the model grid and returns the
    fraction of the cells it passes through that the swath observed — a measure of
    whether a SWOT pass actually *sampled the Loop Current* on this day (vs. merely
    covering enough ocean somewhere in the box). Returns 0 for an empty front.
    """
    if len(front) == 0:
        return 0.0
    jx = np.round((front[:, 0] - lon[0]) / (lon[1] - lon[0])).astype(int)
    iy = np.round((front[:, 1] - lat[0]) / (lat[1] - lat[0])).astype(int)
    ok = (jx >= 0) & (jx < lon.size) & (iy >= 0) & (iy < lat.size)
    cells = set(zip(iy[ok].tolist(), jx[ok].tolist()))     # unique grid cells on the front
    if not cells:
        return 0.0
    return sum(bool(mask[i, j]) for i, j in cells) / len(cells)


def aggregate_skill(selector, forecaster, obs_days, leads, swaths_for, k=15,
                    min_cells=500, deseasonalize=True):
    """Mean ACC and RMSE per lead over every (obs-day, lead) pair, verified against
    the *future SWOT swath* (partial coverage).

    Selection and forecasting are the decoupled ``analog`` pieces: for each obs day
    the analogs are ``selector.select(obs_grid, mask, lead, k)`` and the forecast is
    ``forecaster.forecast(sel, lead)``. Persistence is the analog nowcast (lead-0
    forecast). `deseasonalize` subtracts the day-of-year seasonal climatology from
    forecast and SWOT truth so skill is mesoscale and climatology collapses to ~0.

    Returns (leads, acc, rmse, counts) — `acc`/`rmse` dicts keyed by
    {"analog","persistence","climatology"} of per-lead mean arrays.
    """
    def _swaths(x):
        return x[0] if isinstance(x, tuple) else x

    lib = selector.lib
    acc_m, rmse_m = SpatialCorrelation(lib.lat), DemeanedRMSE(lib.lat)
    names = ["analog", "persistence", "climatology"]
    leads = list(leads)
    acc = {n: {L: [] for L in leads} for n in names}
    rmse = {n: {L: [] for L in leads} for n in names}
    have = set(obs_days)
    for d0 in obs_days:
        t0 = np.datetime64(d0)
        obs_grid, mask = swath_to_grid(_swaths(swaths_for([d0])), lib.lon, lib.lat)
        if mask.sum() < min_cells:
            continue
        persist = lib.anom_of(forecaster.forecast(selector.select(obs_grid, mask, 0, k), 0))
        for L in leads:
            vday = t0 + np.timedelta64(int(L), "D")
            if str(vday) not in have:
                continue
            obs2, mask2 = swath_to_grid(_swaths(swaths_for([str(vday)])), lib.lon, lib.lat)
            if mask2.sum() < min_cells:
                continue
            fc = forecaster.forecast(selector.select(obs_grid, mask, int(L), k), int(L))
            fields = {"analog": lib.anom_of(fc), "persistence": persist,
                      "climatology": lib.clim_anomaly(vday)}
            truth = obs2
            if deseasonalize:
                cav = lib.clim_anomaly(vday)
                fields = {n: fa - cav for n, fa in fields.items()}
                truth = obs2 - cav
            for n, fa in fields.items():
                acc[n][L].append(acc_m(fa, truth, mask=mask2))
                rmse[n][L].append(rmse_m(fa, truth, mask=mask2))

    def means(dct):
        return {n: np.array([np.nanmean(dct[n][L]) if dct[n][L] else np.nan
                             for L in leads]) for n in names}

    counts = [sum(np.isfinite(acc["analog"][L])) for L in leads]
    return leads, means(acc), means(rmse), counts


def evaluate(library, analogs, ens_forecast, truth_surf, vday, persist_surf=None,
             lead=14, lc_level=LEVEL):
    """Score an analog selection's forecast against a *dense* GLORYS field.

    Pure scoring — it does **not** select or combine. Pass ``analogs`` (an
    :class:`analog.AnalogSet` from ``AnalogSelector.select``) and ``ens_forecast``
    (the ensemble field from ``Forecast.forecast``). Each analog's own future
    (``state[i + lead]``) is scored separately alongside the ensemble and, if
    `persist_surf` is given, a dense persistence baseline. All fields are
    deseasonalized at `vday`. Skill: full-field ACC/RMSE and the Loop Current front
    MHD (km). Returns the dict the SWOT→GLORYS figures consume — including the truth
    initial condition (``init``: `persist_surf` deseasonalized at the obs day) and
    the extracted fronts.
    """
    lib = library
    sel, dist = analogs.indices, analogs.distances
    acc_m, rmse_m = SpatialCorrelation(lib.lat), DemeanedRMSE(lib.lat)
    ocean = lib.ocean

    def dz(field, when=vday):
        return lib.deseasonalize(lib.anom_of(field), when)

    truth = dz(truth_surf)
    fc_anoms = [dz(lib.state[i + lead]) for i in sel]
    acc = np.array([acc_m(f, truth, mask=ocean) for f in fc_anoms])
    rmse = np.array([rmse_m(f, truth, mask=ocean) for f in fc_anoms])

    ens_anom = dz(ens_forecast)
    acc_ens = acc_m(ens_anom, truth, mask=ocean)
    rmse_ens = rmse_m(ens_anom, truth, mask=ocean)

    acc_persist = rmse_persist = np.nan
    init_anom = None
    if persist_surf is not None:
        p = dz(persist_surf)
        acc_persist = acc_m(p, truth, mask=ocean)
        rmse_persist = rmse_m(p, truth, mask=ocean)
        oday = np.datetime64(vday) - np.timedelta64(int(lead), "D")
        init_anom = dz(persist_surf, oday)          # initial condition (obs day)

    # --- Loop Current skill: MHD (km) between forecast and truth fronts --- #
    scale = lon_scale(lib.lat)
    ref_mean = float(np.nanmean(lib.mean_surf[ocean]))
    deg_km = 111.195

    def front(field):
        return loop_current_front(field, lib.lon, lib.lat, ocean, ref_mean, level=lc_level)

    truth_front = front(truth_surf)

    def _lc(field):
        f = front(field)
        if len(f) == 0 or len(truth_front) == 0:
            return np.nan, f
        return front_distance(f, truth_front, scale) * deg_km, f

    lc = [_lc(lib.state[i + lead]) for i in sel]
    lc_mhd = np.array([d for d, _ in lc])
    fc_fronts = [f for _, f in lc]
    lc_mhd_ens, ens_front = _lc(ens_forecast)
    lc_mhd_persist, persist_front = (_lc(persist_surf) if persist_surf is not None
                                     else (np.nan, np.empty((0, 2))))

    return dict(sel=sel, dist=dist, fc_anoms=fc_anoms, truth=truth, init=init_anom,
                acc=acc, rmse=rmse, acc_ens=acc_ens, rmse_ens=rmse_ens,
                acc_persist=acc_persist, rmse_persist=rmse_persist,
                lc_mhd=lc_mhd, lc_mhd_ens=lc_mhd_ens, lc_mhd_persist=lc_mhd_persist,
                truth_front=truth_front, fc_fronts=fc_fronts, ens_front=ens_front,
                persist_front=persist_front, lc_level=lc_level)
