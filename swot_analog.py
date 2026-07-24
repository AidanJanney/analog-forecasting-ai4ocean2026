"""SWOT-driven model-analog forecasting against the GLORYS library.

Analogs are selected from a GLORYS library using a *real SWOT swath* as the
observation. The two datasets are heterogeneous, so the distance is measured in
observation space:

1. **Coverage** — a SWOT day is only a few partial swaths. We bin the swath onto
   the GLORYS grid (:func:`swath_to_grid`), which yields a coverage mask and, at
   the same time, averages SWOT (~2 km, noisy) down to the GLORYS resolution
   (~8 km).
2. **Quantity** — GLORYS ``zos`` is absolute dynamic topography while SWOT
   ``ssha_karin`` is an anomaly relative to a different mean surface. We compare
   *anomalies* with a **spatial correlation** over the observed cells, which is
   invariant to the residual offset/scale difference between the two references.
3. Distance ``= 1 - weighted_corr`` (0 = identical pattern, up to 2).

Because the SWOT era (2023+) and the GLORYS library (2004–2013) are disjoint,
every analog is temporally independent — there is no leakage/exclusion problem.
"""

import warnings

import numpy as np

from analog import climatology_of, _day_of_year


def _edges(centers):
    """Cell edges from monotonically-spaced cell centers."""
    c = np.asarray(centers, dtype=float)
    mid = (c[:-1] + c[1:]) / 2
    return np.r_[c[0] - (mid[0] - c[0]), mid, c[-1] + (c[-1] - mid[-1])]


def swath_to_grid(swaths, lon, lat):
    """Bin SWOT swath points onto the GLORYS grid.

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


class SwotAnalogForecaster:
    """Select GLORYS analogs of a SWOT observation and advance them into a forecast."""

    def __init__(self, ds, state, times, anomaly=None):
        """
        ds     : GLORYS xr.Dataset (provides longitude/latitude grid).
        state  : (time, *field) library advanced to make forecasts (e.g. zos).
        times  : datetime64 array aligned with `state`.
        anomaly: (time, nlat, nlon) surface field used for matching; defaults to
                 the surface `state` minus its library mean (offset removed; the
                 seasonal cycle is kept to match SWOT ssha, and correlation makes
                 the residual reference difference harmless).
        """
        self.lon = ds["longitude"].values
        self.lat = ds["latitude"].values
        self.state = np.asarray(state)
        self.times = np.asarray(times)
        self.n = self.state.shape[0]

        surf = self.state if self.state.ndim == 3 else self.state[:, 0]
        self.mean_surf = climatology_of(surf)               # library mean surface
        if anomaly is None:
            anomaly = surf - self.mean_surf
        self.anom = np.asarray(anomaly, dtype=np.float32)   # (n, nlat, nlon)
        self.ocean = np.isfinite(self.anom[0])              # constant land mask
        wlat = np.cos(np.deg2rad(self.lat))[:, None]
        self.wlat2d = np.broadcast_to(wlat, self.anom.shape[1:])

    def anom_of(self, field):
        """Surface anomaly of a forecast/state field (same reference as self.anom)."""
        surf = field if field.ndim == 2 else field[0]
        return surf - self.mean_surf

    def score(self, field_anom, obs_grid, mask):
        """Weighted spatial correlation of a field anomaly to a SWOT obs over `mask`.

        This is the verification analogue of the selection distance: higher is
        better (perfect pattern match = 1). Returns NaN if too few common cells.
        """
        valid = mask & np.isfinite(obs_grid) & self.ocean & np.isfinite(field_anom)
        if valid.sum() < 10:
            return np.nan
        o = obs_grid[valid].astype(np.float32)
        f = field_anom[valid].astype(np.float32)
        w = self.wlat2d[valid].astype(np.float32)
        return float(_weighted_corr(o, f[None, :], w)[0])

    def score_rmse(self, field_anom, obs_grid, mask):
        """Latitude-weighted RMSE of a field anomaly vs a SWOT obs over `mask`.

        Both fields are de-meaned over the observed cells first, removing the
        offset difference between the SWOT (MSS) and GLORYS (library-mean)
        references so the RMSE reflects pattern + amplitude error, not the DC
        offset. Lower is better. Returns NaN if too few common cells.
        """
        valid = mask & np.isfinite(obs_grid) & self.ocean & np.isfinite(field_anom)
        if valid.sum() < 10:
            return np.nan
        o = obs_grid[valid].astype(np.float32)
        f = field_anom[valid].astype(np.float32)
        w = self.wlat2d[valid].astype(np.float32)
        W = w.sum()
        o = o - (w * o).sum() / W                       # de-mean over observed cells
        f = f - (w * f).sum() / W
        return float(np.sqrt((w * (f - o) ** 2).sum() / W))

    def distances(self, obs_grid, mask):
        """1 - weighted spatial correlation of `obs_grid` to every library state."""
        valid = mask & np.isfinite(obs_grid) & self.ocean
        o = obs_grid[valid].astype(np.float32)
        X = self.anom[:, valid]                             # (n, n_valid)
        w = self.wlat2d[valid].astype(np.float32)
        return 1.0 - _weighted_corr(o, X, w), int(valid.sum())

    def analogs(self, obs_grid, mask, k=10, lead=0):
        """Indices and distances of the K nearest library analogs (with a future)."""
        d, _ = self.distances(obs_grid, mask)
        valid_future = np.arange(self.n) + lead < self.n
        d = np.where(valid_future, d, np.inf)
        sel = np.argsort(d)[:k]
        return sel, d[sel]

    def forecast(self, obs_grid, mask, lead, k=10):
        """Gaussian-weighted mean of the K nearest analogs advanced `lead` steps."""
        sel, d_sel = self.analogs(obs_grid, mask, k=k, lead=lead)
        w = np.exp(-(d_sel / d_sel.mean()) ** 2)
        fc = np.tensordot(w, self.state[sel + lead], axes=1) / w.sum()
        return fc, sel, d_sel

    def clim_anomaly(self, date64):
        """Day-of-year seasonal climatology anomaly for a calendar date (baseline)."""
        surf = self.state if self.state.ndim == 3 else self.state[:, 0]
        d = _day_of_year(np.array([date64], dtype="datetime64[D]"))[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            c = np.nanmean(surf[_day_of_year(self.times) == d], axis=0)
        return c - self.mean_surf


def aggregate_skill(saf, obs_days, leads, swaths_for, k=15, min_cells=500):
    """Mean ACC and RMSE per lead over every available (obs-day, lead) pair.

    Parameters
    ----------
    saf : SwotAnalogForecaster
    obs_days : list of 'YYYY-MM-DD' strings that have SWOT swaths on disk.
    leads : iterable of lead times (days).
    swaths_for : callable, ``swaths_for(['YYYY-MM-DD', ...]) -> list of swaths``.
    k, min_cells : analogs kept; minimum observed cells to use an obs/verify day.

    Returns
    -------
    (leads, acc, rmse, counts) where `acc`/`rmse` are dicts keyed by
    {"analog","persistence","climatology"} of per-lead mean arrays, and `counts`
    is the number of contributing obs days per lead.
    """
    def _swaths(x):                      # swaths_for may return (swaths, labels)
        return x[0] if isinstance(x, tuple) else x

    names = ["analog", "persistence", "climatology"]
    leads = list(leads)
    acc = {n: {L: [] for L in leads} for n in names}
    rmse = {n: {L: [] for L in leads} for n in names}
    have = set(obs_days)
    for d0 in obs_days:
        t0 = np.datetime64(d0)
        obs_grid, mask = swath_to_grid(_swaths(swaths_for([d0])), saf.lon, saf.lat)
        if mask.sum() < min_cells:
            continue
        persist = saf.anom_of(saf.forecast(obs_grid, mask, 0, k=k)[0])   # analog nowcast
        for L in leads:
            vday = t0 + np.timedelta64(int(L), "D")
            if str(vday) not in have:
                continue
            obs2, mask2 = swath_to_grid(_swaths(swaths_for([str(vday)])), saf.lon, saf.lat)
            if mask2.sum() < min_cells:
                continue
            fc = saf.forecast(obs_grid, mask, int(L), k=k)[0]
            fields = {"analog": saf.anom_of(fc), "persistence": persist,
                      "climatology": saf.clim_anomaly(vday)}
            for n, fa in fields.items():
                acc[n][L].append(saf.score(fa, obs2, mask2))
                rmse[n][L].append(saf.score_rmse(fa, obs2, mask2))

    def means(dct):
        return {n: np.array([np.nanmean(dct[n][L]) if dct[n][L] else np.nan
                             for L in leads]) for n in names}

    counts = [sum(np.isfinite(acc["analog"][L])) for L in leads]
    return leads, means(acc), means(rmse), counts
