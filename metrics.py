"""Pluggable metrics for analog forecasting.

Two independent, user-selectable roles:

* ``AnalogMetric`` — dissimilarity between *observations* (surface fields such
  as gridded SSH, and eventually SWOT swaths). This is what selects analogs.
  Default: modified Hausdorff distance between Loop Current fronts (``FrontMHD``).

* ``ErrorMetric`` — score a forecast against truth. The forecast state may span
  multiple depths, so error metrics operate on arbitrary (..., lat, lon) fields.
  This need not be the same metric used to pick analogs.
"""

from abc import ABC, abstractmethod

import numpy as np

from fronts import LEVEL, lon_scale, extract_fronts, front_distance


# --------------------------------------------------------------------------- #
# Analog selection metrics — operate on surface observations.
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
    need it ignore it.
    """

    name = "error"

    @abstractmethod
    def __call__(self, forecast, truth, t=None) -> float:
        ...


class WeightedRMSE(ErrorMetric):
    """Latitude-weighted RMSE over ocean points.

    Latitude is assumed to be the second-to-last axis, so this handles both
    surface (lat, lon) and multi-depth (depth, lat, lon) fields unchanged.
    """

    name = "lat-weighted-RMSE"

    def __init__(self, latitude):
        self.wlat = np.cos(np.deg2rad(np.asarray(latitude, dtype=float)))

    def __call__(self, forecast, truth, t=None):
        a = np.asarray(forecast)
        b = np.asarray(truth)
        shape = [1] * a.ndim
        shape[-2] = self.wlat.size          # broadcast weights along the lat axis
        w = np.broadcast_to(self.wlat.reshape(shape), a.shape)
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.sqrt(np.sum(w[m] * (a[m] - b[m]) ** 2) / np.sum(w[m])))


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

    def __call__(self, forecast, truth, t=None):
        a = np.asarray(forecast)
        b = np.asarray(truth)
        c = self._reference(a, t)
        a = a - c                                    # forecast anomaly
        b = b - c                                    # truth anomaly
        shape = [1] * a.ndim
        shape[-2] = self.wlat.size
        w = np.broadcast_to(self.wlat.reshape(shape), a.shape)
        m = np.isfinite(a) & np.isfinite(b)
        wa = w[m]
        num = np.sum(wa * a[m] * b[m])
        den = np.sqrt(np.sum(wa * a[m] ** 2) * np.sum(wa * b[m] ** 2))
        return float(num / den) if den > 0 else 0.0  # 0 when a field has no anomaly
