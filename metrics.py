"""Pluggable metrics for analog forecasting.

* ``ErrorMetric`` — score a forecast against truth. The forecast state may span
  multiple depths, so error metrics operate on arbitrary (..., lat, lon) fields,
  and accept an optional coverage ``mask`` so partial-swath verification works.
  Reported together: ACC (:class:`AnomalyCorrelation` / :class:`SpatialCorrelation`),
  RMSE (:class:`WeightedRMSE` / :class:`DemeanedRMSE`) and the Loop Current front
  MHD in km (:class:`FrontMHDError`).

* ``AnalogMetric`` (below) is the OLD analog-*selection* interface. Selection now
  lives behind :class:`distances.ObsDistance` (``FrontMHDDistance`` /
  ``CorrelationDistance``); the classes here are kept only for backward
  compatibility and are no longer used by the drivers.
"""

from abc import ABC, abstractmethod

import numpy as np

from fronts import LEVEL, lon_scale, extract_fronts, front_distance


def _mask_2d(mask, shape):
    """Broadcast a (nlat, nlon) coverage mask over the leading axes of `shape`."""
    lead = (1,) * (len(shape) - 2)
    return np.broadcast_to(np.asarray(mask).reshape(lead + tuple(mask.shape)), shape)


# --------------------------------------------------------------------------- #
# DEPRECATED analog selection metrics — superseded by distances.ObsDistance.
# --------------------------------------------------------------------------- #
class AnalogMetric(ABC):
    """Dissimilarity between observed states, used to rank analogs."""

    name = "analog"

    @abstractmethod
    def describe(self, obs):
        """Return one descriptor per time step of the observation Dataset."""

    @abstractmethod
    def distance(self, a, b) -> float:
        """Dissimilarity between two descriptors."""

    def signature(self):
        """Short string identifying this metric + params (used for cache keys)."""
        return self.name

    def distance_matrix(self, descriptors):
        """Symmetric dissimilarity matrix over a list of descriptors."""
        n = len(descriptors)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = self.distance(descriptors[i], descriptors[j])
        return D


class FrontMHD(AnalogMetric):
    """Modified Hausdorff distance between Loop Current fronts (a fixed SSH contour).

    Longitudes are scaled to an ~isotropic grid before the distance is taken.
    """

    name = "front-MHD"

    def __init__(self, var="zos", level=LEVEL, max_points=200):
        self.var = var
        self.level = level
        self.max_points = max_points        # front vertices kept (None = all); caps MHD cost
        self._scale = 1.0

    def signature(self):
        return f"{self.name}_{self.var}_{self.level:g}_p{self.max_points}"

    def describe(self, obs):
        self._scale = lon_scale(obs["latitude"].values)
        return extract_fronts(obs[self.var], self.level, self.max_points)

    def distance(self, a, b):
        return front_distance(a, b, self._scale)


class SurfaceFieldRMSE(AnalogMetric):
    """Latitude-weighted RMSE between whole surface fields (e.g. gridded SSH).

    A pixel-space alternative to ``FrontMHD`` for selecting analogs.
    """

    name = "surface-RMSE"

    def __init__(self, var="zos"):
        self.var = var
        self._w = None

    def signature(self):
        return f"{self.name}_{self.var}"

    def describe(self, obs):
        self._w = np.cos(np.deg2rad(obs["latitude"].values))[:, None]
        da = obs[self.var]
        return [da.isel(time=t).values for t in range(da.sizes["time"])]

    def distance(self, a, b):
        m = np.isfinite(a) & np.isfinite(b)
        w = np.broadcast_to(self._w, a.shape)[m]
        return float(np.sqrt(np.sum(w * (a[m] - b[m]) ** 2) / np.sum(w)))


# --------------------------------------------------------------------------- #
# Forecast error metrics — operate on (possibly multi-depth) fields.
# --------------------------------------------------------------------------- #
class ErrorMetric(ABC):
    """Score a forecast field against truth.

    `t` is the (optional) valid-time index of the fields, used by metrics whose
    reference varies in time (e.g. a seasonal climatology); metrics that don't
    need it ignore it. `mask` is an optional (nlat, lon) coverage mask restricting
    the score to observed cells (e.g. a partial swath); None scores every finite
    cell. `dense_only` metrics (front geometry) skip partial-coverage truth.
    """

    name = "error"
    dense_only = False

    @abstractmethod
    def __call__(self, forecast, truth, t=None, mask=None) -> float:
        ...


class WeightedRMSE(ErrorMetric):
    """Latitude-weighted RMSE over ocean points.

    Latitude is assumed to be the second-to-last axis, so this handles both
    surface (lat, lon) and multi-depth (depth, lat, lon) fields unchanged.
    """

    name = "lat-weighted-RMSE"

    def __init__(self, latitude):
        self.wlat = np.cos(np.deg2rad(np.asarray(latitude, dtype=float)))

    def __call__(self, forecast, truth, t=None, mask=None):
        a = np.asarray(forecast)
        b = np.asarray(truth)
        shape = [1] * a.ndim
        shape[-2] = self.wlat.size          # broadcast weights along the lat axis
        w = np.broadcast_to(self.wlat.reshape(shape), a.shape)
        m = np.isfinite(a) & np.isfinite(b)
        if mask is not None:
            m = m & _mask_2d(mask, a.shape)
        return float(np.sqrt(np.sum(w[m] * (a[m] - b[m]) ** 2) / np.sum(w[m])))


class DemeanedRMSE(ErrorMetric):
    """Latitude-weighted RMSE of two *anomaly* fields, each de-meaned over the
    scored cells first — removing any residual offset so the RMSE reflects the
    pattern + amplitude error (the observation-space RMSE for cross-dataset SSH)."""

    name = "demeaned-RMSE"

    def __init__(self, latitude):
        self.wlat = np.cos(np.deg2rad(np.asarray(latitude, dtype=float)))[:, None]

    def __call__(self, forecast, truth, t=None, mask=None):
        a = np.asarray(forecast)
        b = np.asarray(truth)
        w2d = np.broadcast_to(self.wlat, a.shape)
        m = np.isfinite(a) & np.isfinite(b)
        if mask is not None:
            m = m & np.asarray(mask)
        if m.sum() < 10:
            return np.nan
        wa, fa, ba = w2d[m], a[m], b[m]
        W = wa.sum()
        fa = fa - (wa * fa).sum() / W
        ba = ba - (wa * ba).sum() / W
        return float(np.sqrt((wa * (fa - ba) ** 2).sum() / W))


class SpatialCorrelation(ErrorMetric):
    """Latitude-weighted spatial (pattern) correlation over the scored cells.

    The observation-space ACC used for cross-dataset verification: both fields are
    centered over the observed cells, so it is invariant to a constant offset.
    Operates on 2-D (lat, lon) anomaly fields. Perfect pattern match = 1."""

    name = "spatial-correlation"

    def __init__(self, latitude):
        self.wlat = np.cos(np.deg2rad(np.asarray(latitude, dtype=float)))[:, None]

    def __call__(self, forecast, truth, t=None, mask=None):
        from swot_analog import _weighted_corr

        a = np.asarray(forecast)
        b = np.asarray(truth)
        w2d = np.broadcast_to(self.wlat, a.shape)
        m = np.isfinite(a) & np.isfinite(b)
        if mask is not None:
            m = m & np.asarray(mask)
        if m.sum() < 10:
            return np.nan
        return float(_weighted_corr(b[m].astype(np.float32),
                                    a[m][None, :].astype(np.float32),
                                    w2d[m].astype(np.float32))[0])


class FrontMHDError(ErrorMetric):
    """Loop Current front position error: modified Hausdorff distance in km.

    Both fields are contoured at the fixed `level`-m ``zos`` isoline (offset-
    referenced to a common `ref_mean` datum so the contour tracks the front's
    position, not the basin-scale sea-level offset), and the MHD between the two
    fronts is returned in km. Lower is better. Requires dense absolute SSH fields —
    it is skipped on partial-swath / anomaly-only truth (``dense_only``)."""

    name = "front-MHD-km"
    dense_only = True

    def __init__(self, lon, lat, ocean, ref_mean, level=LEVEL, max_points=200,
                 deg_km=111.195):
        self.lon, self.lat, self.ocean = lon, lat, ocean
        self.ref_mean, self.level, self.max_points = ref_mean, level, max_points
        self.scale = lon_scale(lat)
        self.deg_km = deg_km

    def __call__(self, forecast, truth, t=None, mask=None):
        from fronts import loop_current_front

        f2 = forecast if forecast.ndim == 2 else forecast[0]
        t2 = truth if truth.ndim == 2 else truth[0]
        kw = dict(level=self.level, max_points=self.max_points)
        ff = loop_current_front(f2, self.lon, self.lat, self.ocean, self.ref_mean, **kw)
        tf = loop_current_front(t2, self.lon, self.lat, self.ocean, self.ref_mean, **kw)
        if len(ff) == 0 or len(tf) == 0:
            return np.nan
        return front_distance(ff, tf, self.scale) * self.deg_km


class AnomalyCorrelation(ErrorMetric):
    """Latitude-weighted anomaly correlation coefficient (ACC) vs a climatology.

    Anomalies are taken relative to a reference field (typically the library
    mean): ACC = <w f' t'> / sqrt(<w f'^2> <w t'^2>). Unlike RMSE, higher is
    better (perfect = 1); a climatology forecast has zero anomaly and scores 0.
    Latitude is assumed to be the second-to-last axis, so it handles both
    surface (lat, lon) and multi-depth (depth, lat, lon) fields.
    """

    name = "anomaly-correlation"

    def __init__(self, latitude, climatology):
        self.wlat = np.cos(np.deg2rad(np.asarray(latitude, dtype=float)))
        # climatology is either a static field (*field) or a per-time seasonal
        # series (time, *field); the latter is indexed by the valid-time `t`.
        self.clim = np.asarray(climatology, dtype=float)

    def _reference(self, field, t):
        if self.clim.ndim == field.ndim:             # static climatology
            return self.clim
        if t is None:                                # seasonal series, no time given
            raise ValueError("AnomalyCorrelation with a seasonal climatology "
                             "requires the valid-time index t")
        return self.clim[t]                          # seasonal reference at valid time

    def __call__(self, forecast, truth, t=None, mask=None):
        a = np.asarray(forecast)
        b = np.asarray(truth)
        c = self._reference(a, t)
        a = a - c                                    # forecast anomaly
        b = b - c                                    # truth anomaly
        shape = [1] * a.ndim
        shape[-2] = self.wlat.size
        w = np.broadcast_to(self.wlat.reshape(shape), a.shape)
        m = np.isfinite(a) & np.isfinite(b)
        if mask is not None:
            m = m & _mask_2d(mask, a.shape)
        wa = w[m]
        num = np.sum(wa * a[m] * b[m])
        den = np.sqrt(np.sum(wa * a[m] ** 2) * np.sum(wa * b[m] ** 2))
        return float(num / den) if den > 0 else 0.0  # 0 when a field has no anomaly
