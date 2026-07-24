"""Model-analog forecasting engine, agnostic to the choice of metrics.

Analogs are chosen with an ``AnalogMetric`` over surface *observations*; forecasts
are advanced in a (possibly multi-depth) *state* library and scored with an
``ErrorMetric``. The two metrics are independent — the observation space that
selects analogs (e.g. SWOT SSH) need not match the state space being forecast.
"""

import hashlib
import os
import warnings

import numpy as np

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
    """
    cpath = _cache_path(metric, source) if (use_cache and source) else None
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


class AnalogForecaster:
    def __init__(self, obs, analog_metric, state, error_metric,
                 source=None, use_cache=True, climatology=None):
        """
        Parameters
        ----------
        obs : xr.Dataset
            Surface observations that drive analog selection.
        analog_metric : AnalogMetric
            Dissimilarity used to rank analogs.
        state : np.ndarray, shape (time, *field)
            Library of states advanced to produce forecasts; `field` may include
            a depth axis, e.g. (time, depth, lat, lon).
        error_metric : ErrorMetric
            Metric used to verify a forecast against truth.
        source : str, optional
            Path of the dataset backing `obs`; enables the on-disk matrix cache.
        use_cache : bool
            Load/save the dissimilarity matrix from `cache/` when `source` is given.
        climatology : np.ndarray, optional
            Per-time reference field (time, *field), e.g. a seasonal climatology,
            used as the climatology baseline forecast. Falls back to the library
            time-mean when omitted.
        """
        self.analog_metric = analog_metric
        self.error_metric = error_metric
        self.state = np.asarray(state)
        self._clim_series = None if climatology is None else np.asarray(climatology)
        self.D = build_matrix(analog_metric, obs, source, use_cache=use_cache)
        self.n = self.D.shape[0]

    # -- analog selection ---------------------------------------------------- #
    def _candidates(self, t0, lead, exclude):
        """Library times with a known future and outside the exclusion window."""
        return [a for a in range(self.n)
                if a + lead < self.n and abs(a - t0) > exclude]

    def analogs(self, t0, lead, k, exclude):
        cand = self._candidates(t0, lead, exclude)
        d = self.D[t0, cand]
        sel = np.array(cand)[np.argsort(d)][:k]
        return sel, self.D[t0, sel]

    def n_candidates(self, t0, lead, exclude):
        return len(self._candidates(t0, lead, exclude))

    # -- forecasting --------------------------------------------------------- #
    def forecast(self, t0, lead, k=10, exclude=5):
        """Gaussian-weighted mean of the K nearest analogs advanced `lead` steps."""
        sel, d_sel = self.analogs(t0, lead, k, exclude)
        w = np.exp(-(d_sel / d_sel.mean()) ** 2)
        fc = np.tensordot(w, self.state[sel + lead], axes=1) / w.sum()
        return fc, sel, d_sel

    def truth(self, t):
        return self.state[t]

    def climatology(self):
        return climatology_of(self.state)

    def clim_forecast(self, t):
        """Climatology baseline forecast valid at time `t` (seasonal if provided)."""
        if self._clim_series is not None:
            return self._clim_series[t]
        return self.climatology()

    def error(self, forecast, truth, t=None):
        return self.error_metric(forecast, truth, t)


def skill_curve(af, leads, k=10, exclude=5, stride=1):
    """Mean forecast error vs lead time for the analog, persistence and
    climatology forecasts, averaged over valid targets in the library.

    Errors are scored at the verification time V = t0 + lead, so time-dependent
    references (e.g. a seasonal climatology) use the correct calendar day.
    `stride` subsamples targets (every `stride`-th) to speed up large libraries.
    """
    skill = {"analog": [], "persistence": [], "climatology": []}
    for lead in leads:
        ra, rp, rc = [], [], []
        for t0 in range(0, af.n, stride):
            if t0 + lead >= af.n or af.n_candidates(t0, lead, exclude) < k:
                continue
            V = t0 + lead
            fc, _, _ = af.forecast(t0, lead, k=k, exclude=exclude)
            truth = af.truth(V)
            ra.append(af.error(fc, truth, V))
            rp.append(af.error(af.truth(t0), truth, V))
            rc.append(af.error(af.clim_forecast(V), truth, V))
        skill["analog"].append(np.mean(ra))
        skill["persistence"].append(np.mean(rp))
        skill["climatology"].append(np.mean(rc))
    return skill
