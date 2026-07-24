"""Unified model-analog forecasting engine.

One engine, three pluggable seams:

* a :class:`~sources.ObsSource` — where the observation comes from (the model
  itself, a SWOT swath, an OSTIA grid);
* an :class:`~distances.ObsDistance` — how the observation is compared to the
  library to rank analogs (classical spatial correlation now, a learned latent
  distance later);
* a list of error metrics — how a forecast is scored (ACC, RMSE, Loop Current
  front MHD), reported together.

The model itself is abstracted behind :class:`ModelLibrary` (GLORYS here, but only
variable/coordinate names and a state array are assumed). Analog *selection* stays
surface/observable; the forecast *target* (``ModelLibrary.state``) may carry depth
and is scored by the depth-aware metrics.
"""

import hashlib
import os
import warnings

import numpy as np
import xarray as xr

CACHE_DIR = "cache"


def climatology_of(state):
    """Time-mean of a (time, *field) state library, ignoring all-NaN land cells."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.asarray(state), axis=0)


def _day_of_year(times):
    """Day-of-year (1..366) for a datetime64 array."""
    days = times.astype("datetime64[D]")
    year_start = times.astype("datetime64[Y]").astype("datetime64[D]")
    return (days - year_start).astype("timedelta64[D]").astype(int) + 1


def seasonal_climatology(state, times):
    """Per-time day-of-year climatology: the multi-year mean for each calendar day.

    Returns an array shaped like `state` (time, *field) where entry t is the mean
    of every library state sharing t's day-of-year. Needs several years to be
    meaningful (with one year per day-of-year it just returns the state itself).
    """
    state = np.asarray(state)
    doy = _day_of_year(times)
    ref = np.empty_like(state, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for d in np.unique(doy):
            sel = np.where(doy == d)[0]
            ref[sel] = np.nanmean(state[sel], axis=0)
    return ref


def _cache_path(metric, source):
    """Cache file for `metric`'s matrix on `source`, invalidated by file mtime."""
    st = os.stat(source)
    tag = f"{os.path.abspath(source)}|{st.st_mtime_ns}|{metric.signature()}"
    h = hashlib.md5(tag.encode()).hexdigest()[:12]
    base = os.path.splitext(os.path.basename(source))[0]
    return os.path.join(CACHE_DIR, f"D_{base}_{metric.signature()}_{h}.npy")


def build_matrix(metric, obs, source=None, descriptors=None,
                 use_cache=True, verbose=True):
    """Return the analog dissimilarity matrix, loading from / saving to disk cache.

    The cache is keyed by the source file (path + modification time) and the
    metric signature, so it invalidates automatically when either changes.
    `descriptors` may be passed to reuse an already-computed description on a miss.
    `source` may be a path or an xr.Dataset carrying an `encoding["source"]` path.
    """
    src_path = _source_path(source)
    cpath = _cache_path(metric, src_path) if (use_cache and src_path) else None
    if cpath and os.path.exists(cpath):
        if verbose:
            print(f"  [cache] hit  {os.path.basename(cpath)}")
        return np.load(cpath)
    if descriptors is None:
        descriptors = metric.describe(obs)
    D = metric.distance_matrix(descriptors)
    if cpath:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(cpath, D)
        if verbose:
            print(f"  [cache] save {os.path.basename(cpath)}")
    return D


def _source_path(source):
    """Resolve a cache-key path from a str path or an xr.Dataset (its file source)."""
    if source is None:
        return None
    if isinstance(source, str):
        return source
    if isinstance(source, xr.Dataset):
        return source.encoding.get("source")
    return None


# --------------------------------------------------------------------------- #
# The model, abstracted.
# --------------------------------------------------------------------------- #
class ModelLibrary:
    """A library of model states — "the model" the pipeline forecasts and matches.

    Wraps the state array plus the surface anomaly, ocean mask, latitude weights and
    (seasonal) climatology every distance/metric needs. GLORYS-specific assumptions
    are confined to the default variable/coordinate names, which are parameters.

    Two variables, deliberately separable (surface-only-analog-selection): `obs_var`
    is the surface *observable* (SSH ``zos``) that analog selection sees, while `var`
    is the forecast *target* advanced to make forecasts — it may be a different,
    multi-depth field (e.g. ``thetao`` over depths [0, 320]). They coincide by
    default (both ``zos``).
    """

    def __init__(self, ds, var="zos", obs_var="zos", depths=None,
                 lon_name="longitude", lat_name="latitude", time_name="time",
                 seasonal=None, anomaly=None):
        self.ds = ds
        self.var = var
        self.obs_var = obs_var
        self.lon = ds[lon_name].values
        self.lat = ds[lat_name].values
        self.times = ds[time_name].values
        da = ds[var]
        if "depth" in da.dims:
            da = da.isel(depth=depths) if depths is not None else da.isel(depth=[0])
        self.state = da.values                          # (time, *field) forecast target
        self.n = self.state.shape[0]

        obs_da = ds[obs_var]                            # surface observable for selection
        if "depth" in obs_da.dims:
            obs_da = obs_da.isel(depth=0)
        surf = obs_da.values                            # (n, nlat, nlon) absolute surface
        self.surf = surf
        self.surf_da = xr.DataArray(
            surf, dims=(time_name, lat_name, lon_name),
            coords={time_name: self.times, lat_name: self.lat, lon_name: self.lon})
        self.mean_surf = climatology_of(surf)           # library mean surface
        # matching anomaly: default surface minus the library mean (offset removed,
        # seasonal cycle kept); pass `anomaly=` for a deseasonalized/mesoscale
        # selection field (as the oracle diagnostics do).
        if anomaly is None:
            anomaly = surf - self.mean_surf
        self.anom = np.asarray(anomaly, dtype=np.float32)
        self.ocean = np.isfinite(self.anom[0])          # constant land mask
        wlat = np.cos(np.deg2rad(self.lat))[:, None]
        self.wlat2d = np.broadcast_to(wlat, self.anom.shape[1:])

        n_years = len(np.unique(self.times.astype("datetime64[Y]")))
        self.seasonal = (n_years >= 3) if seasonal is None else seasonal
        self.clim_series = (seasonal_climatology(self.state, self.times)
                            if self.seasonal else None)
        self._static_clim = climatology_of(self.state)

    # -- anomaly / climatology helpers (surface) ----------------------------- #
    def anom_of(self, field):
        """Surface anomaly of a field, same reference as `self.anom`."""
        surf = field if field.ndim == 2 else field[0]
        return surf - self.mean_surf

    def clim_anomaly(self, date64):
        """Day-of-year seasonal climatology anomaly for a calendar date."""
        d = _day_of_year(np.array([np.datetime64(str(date64))], dtype="datetime64[D]"))[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            c = np.nanmean(self.surf[_day_of_year(self.times) == d], axis=0)
        return c - self.mean_surf

    def deseasonalize(self, field_anom, date64):
        """Strip the day-of-year seasonal cycle from a surface *anomaly* field."""
        return field_anom - self.clim_anomaly(date64)

    def clim_forecast(self, t):
        """Climatology baseline field valid at library index `t` (seasonal if available)."""
        if self.clim_series is not None:
            return self.clim_series[t]
        return self._static_clim

    def field_on(self, date64):
        """Nearest daily surface field to a calendar date, or None if out of range."""
        d = np.datetime64(str(date64))
        if d < self.times.min() or d > self.times.max():
            return None
        return self.surf[int(np.argmin(np.abs(self.times - d)))]


# --------------------------------------------------------------------------- #
# The engine.
# --------------------------------------------------------------------------- #
class AnalogForecaster:
    """Select library analogs of an observation and advance them into a forecast."""

    def __init__(self, library, distance, error_metrics=None, source=None,
                 use_cache=True):
        self.lib = library
        self.distance = distance.prepare(library, source=source, use_cache=use_cache)
        self.error_metrics = list(error_metrics) if error_metrics else []
        # expose library attributes for plotting/back-compat (viz reads these).
        for a in ("state", "n", "times", "lon", "lat", "anom", "ocean",
                  "mean_surf", "wlat2d", "surf"):
            setattr(self, a, getattr(library, a))

    # -- library delegates --------------------------------------------------- #
    def anom_of(self, field):
        return self.lib.anom_of(field)

    def clim_anomaly(self, date64):
        return self.lib.clim_anomaly(date64)

    def deseasonalize(self, field_anom, date64):
        return self.lib.deseasonalize(field_anom, date64)

    def clim_forecast(self, t):
        return self.lib.clim_forecast(t)

    def truth(self, t):
        return self.state[t]

    @property
    def primary_metric(self):
        return self.error_metrics[0] if self.error_metrics else None

    def error(self, forecast, truth, t=None, mask=None):
        return self.primary_metric(forecast, truth, t=t, mask=mask)

    # -- analog selection ---------------------------------------------------- #
    def _select(self, d, k, min_sep):
        """K nearest by `d`, optionally de-clustered to be ≥ min_sep indices apart."""
        order = np.argsort(d)
        if min_sep <= 0:
            sel = order[:k]
            return sel, d[sel]
        kept = []
        for idx in order:
            if not np.isfinite(d[idx]):
                break
            if all(abs(int(idx) - s) >= min_sep for s in kept):
                kept.append(int(idx))
                if len(kept) == k:
                    break
        sel = np.array(kept)
        return sel, d[sel]

    def n_candidates(self, t0, lead, exclude):
        a = np.arange(self.n)
        return int(np.sum((a + lead < self.n) & (np.abs(a - t0) > exclude)))

    # self-mode: the observation IS the library surface state at index t0
    def analogs(self, t0, lead, k, exclude=0, min_sep=0):
        d = self.distance.distance(self.surf[t0], self.ocean, self_index=int(t0))
        a = np.arange(self.n)
        invalid = (a + lead >= self.n) | (np.abs(a - t0) <= exclude)
        d = np.where(invalid, np.inf, d)
        return self._select(d, k, min_sep)

    def forecast(self, t0, lead, k=10, exclude=0, min_sep=0):
        """Gaussian-weighted mean of the K nearest analogs advanced `lead` steps."""
        sel, d_sel = self.analogs(t0, lead, k, exclude, min_sep)
        w = np.exp(-(d_sel / d_sel.mean()) ** 2)
        fc = np.tensordot(w, self.state[sel + lead], axes=1) / w.sum()
        return fc, sel, d_sel

    # obs-mode: an independent observation (a partial swath, an SST grid)
    def analogs_obs(self, obs_grid, mask, lead, k, min_sep=0, self_index=None):
        d = self.distance.distance(obs_grid, mask, self_index=self_index)
        valid_future = np.arange(self.n) + lead < self.n
        d = np.where(valid_future, d, np.inf)
        return self._select(d, k, min_sep)

    def forecast_obs(self, obs_grid, mask, lead, k=10, min_sep=0, self_index=None):
        sel, d_sel = self.analogs_obs(obs_grid, mask, lead, k, min_sep, self_index)
        w = np.exp(-(d_sel / d_sel.mean()) ** 2)
        fc = np.tensordot(w, self.state[sel + lead], axes=1) / w.sum()
        return fc, sel, d_sel

    # -- observation-space scores (masked spatial pattern; see swot_analog) --- #
    def score(self, field_anom, obs_grid, mask):
        """Weighted spatial correlation of a field anomaly to an obs over `mask`
        (verification twin of CorrelationDistance; higher = better, NaN if too few)."""
        from swot_analog import _weighted_corr

        valid = mask & np.isfinite(obs_grid) & self.ocean & np.isfinite(field_anom)
        if valid.sum() < 10:
            return np.nan
        o = obs_grid[valid].astype(np.float32)
        f = field_anom[valid].astype(np.float32)
        w = self.wlat2d[valid].astype(np.float32)
        return float(_weighted_corr(o, f[None, :], w)[0])

    def score_rmse(self, field_anom, obs_grid, mask):
        """Latitude-weighted RMSE vs an obs over `mask`, both de-meaned over the
        observed cells first (removes the reference-offset difference)."""
        valid = mask & np.isfinite(obs_grid) & self.ocean & np.isfinite(field_anom)
        if valid.sum() < 10:
            return np.nan
        o = obs_grid[valid].astype(np.float32)
        f = field_anom[valid].astype(np.float32)
        w = self.wlat2d[valid].astype(np.float32)
        W = w.sum()
        o = o - (w * o).sum() / W
        f = f - (w * f).sum() / W
        return float(np.sqrt((w * (f - o) ** 2).sum() / W))

    # -- multi-metric skill vs lead (self-source, dense truth) --------------- #
    def skill_self(self, leads, k=10, exclude=30, min_sep=0, stride=1, metrics=None):
        """Per-lead mean of every error metric for analog / persistence / climatology,
        scored against the dense future library state. Reports all metrics together."""
        metrics = metrics or self.error_metrics
        names = ["analog", "persistence", "climatology"]
        acc = {m.name: {nm: {L: [] for L in leads} for nm in names} for m in metrics}
        for lead in leads:
            for t0 in range(0, self.n, stride):
                if t0 + lead >= self.n or self.n_candidates(t0, lead, exclude) < k:
                    continue
                V = t0 + lead
                fc, _, _ = self.forecast(t0, lead, k=k, exclude=exclude, min_sep=min_sep)
                truth = self.truth(V)
                fields = {"analog": fc, "persistence": self.truth(t0),
                          "climatology": self.clim_forecast(V)}
                for m in metrics:
                    for nm, fld in fields.items():
                        acc[m.name][nm][lead].append(m(fld, truth, t=V, mask=self.ocean))
        out = {m.name: {nm: np.array([np.nanmean(acc[m.name][nm][L])
                                      if acc[m.name][nm][L] else np.nan
                                      for L in leads]) for nm in names}
               for m in metrics}
        return list(leads), out


def skill_curve(af, leads, k=10, exclude=5, stride=1):
    """Back-compatible single-metric skill curve (primary error metric).

    Mean forecast error vs lead for the analog, persistence and climatology
    forecasts, using the forecaster's primary error metric. Kept so existing
    drivers / ``viz.plot_skill`` continue to work; prefer ``af.skill_self`` for the
    full multi-metric report.
    """
    m = af.primary_metric.name
    _, out = af.skill_self(leads, k=k, exclude=exclude, stride=stride,
                           metrics=[af.primary_metric])
    return {nm: out[m][nm] for nm in ("analog", "persistence", "climatology")}
