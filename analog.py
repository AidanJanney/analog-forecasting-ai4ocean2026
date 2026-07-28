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
from abc import ABC, abstractmethod
from dataclasses import dataclass

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
        self._clim_series = None                         # built lazily on first use
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

    @property
    def clim_series(self):
        """Per-time seasonal climatology (time, *field), built on first use.

        Deferred because it is a full-size array only some workflows need — the
        SWOT→GLORYS scoring deseasonalizes via the on-the-fly ``clim_anomaly``
        instead, so a 30-year library never pays for it.
        """
        if self.seasonal and self._clim_series is None:
            self._clim_series = seasonal_climatology(self.state, self.times)
        return self._clim_series

    def clim_forecast(self, t):
        """Climatology baseline field valid at library index `t` (seasonal if available)."""
        cs = self.clim_series
        return cs[t] if cs is not None else self._static_clim

    def field_on(self, date64):
        """Nearest daily surface field to a calendar date, or None if out of range."""
        d = np.datetime64(str(date64))
        if d < self.times.min() or d > self.times.max():
            return None
        return self.surf[int(np.argmin(np.abs(self.times - d)))]


# --------------------------------------------------------------------------- #
# Selection result + forecast combiners — the seam between the two steps.
# --------------------------------------------------------------------------- #
@dataclass
class AnalogSet:
    """The outcome of analog *selection*, independent of any forecast.

    Holds the chosen library indices and their selection distances (smallest =
    most similar, in the order returned by the selector). Turn it into a forecast
    with a :class:`Combiner` (or ``Forecast.forecast``); the same
    ``AnalogSet`` can be advanced to many lead times or scored on its own.
    """

    indices: np.ndarray
    distances: np.ndarray

    def __len__(self):
        return len(self.indices)

    @property
    def weights(self):
        """Gaussian-on-distance ensemble weights, normalised to sum to 1."""
        d = np.asarray(self.distances, dtype=float)
        if d.size == 0:
            return d
        w = np.exp(-(d / d.mean()) ** 2)
        return w / w.sum()


class Combiner(ABC):
    """Turn a selection (:class:`AnalogSet`) into a forecast field at a given lead.

    This is the *forecasting* step, fully decoupled from selection: it sees only
    the chosen indices/weights and the state library, never the observation or the
    distance. Swap it to change how analogs are combined without touching selection.
    """

    @abstractmethod
    def __call__(self, analogs: AnalogSet, state, lead):
        ...


class EnsembleMean(Combiner):
    """Gaussian-distance-weighted mean of the analogs advanced `lead` steps (default)."""

    name = "ensemble-mean"

    def __call__(self, analogs, state, lead):
        return np.tensordot(analogs.weights, state[analogs.indices + lead], axes=1)


class BestAnalog(Combiner):
    """The single nearest analog advanced `lead` steps.

    Preferred for sharp frontal targets (e.g. the Loop Current), where averaging
    analogs displaces/blurs the front and the best single analog scores better."""

    name = "best-analog"

    def __call__(self, analogs, state, lead):
        return state[analogs.indices[0] + lead]


# --------------------------------------------------------------------------- #
# Selection and forecasting — two decoupled classes.
# --------------------------------------------------------------------------- #
class AnalogSelector:
    """Rank library analogs of an observation window. Selection only — no forecasting.

    Holds the model library and an ``ObsDistance``; :meth:`select` turns an
    :class:`obs_window.ObsWindow` into an :class:`AnalogSet` (indices +
    distances). One selector serves both workflows, because a one-day window *is*
    the single-observation case: the sequence distance over a single day reduces
    to that day's distance exactly.

    A self-referential window (its days carry ``self_index``, i.e.
    :class:`sources.SelfSource`) should be given an ``exclude`` window to block
    same-event leakage; a cross-source window (SWOT → GLORYS, disjoint eras)
    needs none.
    """

    def __init__(self, library, distance, source=None, use_cache=True):
        self.lib = library
        self.distance = distance.prepare(library, source=source, use_cache=use_cache)

    def select(self, window, lead=0, k=10, min_sep=0, exclude=0):
        """Return the K nearest analogs of `window` (with a valid `lead` future).

        `min_sep` de-clusters them to be ≥ that many days apart; `exclude`
        (self-referential windows only) drops candidates within that many days of
        the window's own library index.
        """
        n = self.lib.n
        d = window.sequence_distance(self.distance, n)
        j = np.arange(n)
        # Need a `lead`-day future, and enough history to align the whole window.
        invalid = (j + lead >= n) | (j < window.max_offset)
        self_index = window.self_index
        if self_index is not None and exclude > 0:           # self-source leakage guard
            invalid = invalid | (np.abs(j - self_index) <= exclude)
        return _knearest(np.where(invalid, np.inf, d), k, min_sep)


class Forecast:
    """Turn a selection into a forecast field. Forecasting only — no selection.

    Holds the model library and a default :class:`Combiner`; :meth:`forecast`
    advances the selected analogs `lead` steps and combines them (``EnsembleMean``
    by default, or pass ``combiner=BestAnalog()``). The same :class:`AnalogSet` can
    be forecast at many leads.
    """

    def __init__(self, library, combiner=None):
        self.lib = library
        self.combiner = combiner or EnsembleMean()

    def forecast(self, analogs, lead, combiner=None):
        return (combiner or self.combiner)(analogs, self.lib.state, lead)


def _knearest(d, k, min_sep):
    """The K smallest distances as an AnalogSet; de-clustered ≥ min_sep apart."""
    order = np.argsort(d)
    if min_sep <= 0:
        sel = order[:k]
    else:
        kept = []
        for idx in order:
            if not np.isfinite(d[idx]):
                break
            if all(abs(int(idx) - s) >= min_sep for s in kept):
                kept.append(int(idx))
                if len(kept) == k:
                    break
        sel = np.array(kept, dtype=int)
    return AnalogSet(sel, d[sel])


def skill_curve(selector, forecaster, leads, metrics, k=10, exclude=30, min_sep=0,
                stride=1, source=None, n_days=1):
    """Per-lead mean of each metric for analog / persistence / climatology,
    scored against the dense future library state (self-source).

    The observation is the library's own surface field at each target day (temporal
    exclusion on); the forecast is the combiner's output. Returns
    ``(leads, {metric_name: {analog|persistence|climatology: per-lead array}})``.
    """
    from obs_window import build_window            # local: obs_window is lower-level
    from sources import SelfSource

    src = source or SelfSource()
    lib = selector.lib
    n = lib.n
    a = np.arange(n)
    names = ["analog", "persistence", "climatology"]
    acc = {m.name: {nm: {L: [] for L in leads} for nm in names} for m in metrics}
    for lead in leads:
        for t0 in range(0, n, stride):
            if t0 + lead >= n or int(np.sum((a + lead < n) & (np.abs(a - t0) > exclude))) < k:
                continue
            window = build_window(src, lib.times[t0], lib, n_days=n_days)
            if window is None:
                continue
            sel = selector.select(window, lead, k, min_sep=min_sep, exclude=exclude)
            V = t0 + lead
            truth = lib.state[V]
            fields = {"analog": forecaster.forecast(sel, lead),
                      "persistence": lib.state[t0],
                      "climatology": lib.clim_forecast(V)}
            for m in metrics:
                for nm, fld in fields.items():
                    acc[m.name][nm][lead].append(m(fld, truth, t=V, mask=lib.ocean))
    out = {m.name: {nm: np.array([np.nanmean(acc[m.name][nm][L]) if acc[m.name][nm][L]
                                  else np.nan for L in leads]) for nm in names}
           for m in metrics}
    return list(leads), out
