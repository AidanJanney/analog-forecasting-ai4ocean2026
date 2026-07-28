"""Pluggable forecast error metrics — how a forecast is scored.

An :class:`ErrorMetric` scores a forecast field against truth. The forecast state
may span multiple depths, so metrics operate on arbitrary (..., lat, lon) fields,
and accept an optional coverage ``mask`` so partial-swath verification works.
Reported together: ACC (:class:`AnomalyCorrelation` / :class:`SpatialCorrelation`),
RMSE (:class:`WeightedRMSE` / :class:`DemeanedRMSE`) and the Loop Current front
MHD in km (:class:`FrontMHDError`).

This module is only about *scoring*. Analog **selection** is a separate seam and
lives behind :class:`distances.ObsDistance` (``CorrelationDistance`` /
``FrontMHDDistance``).
"""

from abc import ABC, abstractmethod

import numpy as np

from fronts import LEVEL, front_mask, front_mhd_km, grid_spacing_km


def _mask_2d(mask, shape):
    """Broadcast a (nlat, nlon) coverage mask over the leading axes of `shape`."""
    lead = (1,) * (len(shape) - 2)
    return np.broadcast_to(np.asarray(mask).reshape(lead + tuple(mask.shape)), shape)


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

    def __init__(self, lon, lat, ocean, ref_mean, level=LEVEL, main_only=False):
        self.ocean, self.ref_mean, self.level = ocean, ref_mean, level
        self.main_only = main_only
        self.sampling = grid_spacing_km(lon, lat)

    def _front(self, field):
        return front_mask(field, level=self.level, ocean=self.ocean,
                          ref_mean=self.ref_mean, main_only=self.main_only)

    def __call__(self, forecast, truth, t=None, mask=None):
        return front_mhd_km(self._front(forecast), self._front(truth), self.sampling)


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
