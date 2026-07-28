"""Section 3 scoring: how a forecast is judged against what happened.

An :class:`ErrorMetric` takes ``{var: field}`` for the forecast and the truth
plus the valid time, and returns one number. Metrics declare their own units and
which direction is better, so the reporting code never carries a table of special
cases and a new metric needs no changes outside this file.

Every spatial mean is latitude-weighted. A grid cell at 31 N covers about 7% less
area than one at 18 N, so an unweighted domain mean over the Gulf box quietly
overweights the northern shelf; weighting by cos(latitude) makes the score an
area average of the field rather than an average over array elements.

Scoring is a different seam from selection. How library days are *ranked* lives
behind :mod:`..distance`; this module only judges the result, and a run normally
scores by several metrics whichever one did the ranking.
"""

from abc import ABC, abstractmethod

import numpy as np

from ..fronts import LEVEL, front_mask, front_mhd_km
from ..registry import Registry

METRICS = Registry("score metric")


class ErrorMetric(ABC):
    """Score one forecast against truth at a valid time."""

    name = "error"
    unit = ""
    higher_is_better = False

    @abstractmethod
    def __call__(self, forecast, truth, valid_time=None) -> float:
        """`forecast` and `truth` are ``{var: field}`` in physical units."""


def _values(field):
    return np.asarray(field.values if hasattr(field, "values") else field, dtype=float)


class WeightedRMSE(ErrorMetric):
    """Latitude-weighted root-mean-square error, in the field's own units."""

    def __init__(self, var, library):
        self.var = var
        self.name = f"{var}_rmse"
        self.unit = library.fields[var].unit
        self.w = _values(library.weights)

    def __call__(self, forecast, truth, valid_time=None):
        diff = _values(forecast[self.var]) - _values(truth[self.var])
        w = np.broadcast_to(self.w, diff.shape)
        ok = np.isfinite(diff)
        total = w[ok].sum()
        if total <= 0:
            return np.nan
        return float(np.sqrt((w[ok] * diff[ok] ** 2).sum() / total))


class AnomalyCorrelation(ErrorMetric):
    """Latitude-weighted anomaly correlation coefficient against the climatology.

    The spatial correlation between forecast and observed departures from the
    climatology of the day being forecast. 1 is a perfect anomaly pattern, 0 is no
    better than climatology, negative is worse — so unlike the error metrics,
    higher is better.

    Uncentred, following the usual verification convention: the fields are already
    departures from a mean, so no second mean is removed.
    """

    higher_is_better = True
    unit = "correlation"

    def __init__(self, var, library):
        self.var = var
        self.name = f"{var}_acc"
        self.field = library.fields[var]
        self.w = _values(library.weights)

    def __call__(self, forecast, truth, valid_time=None):
        clim = _values(self.field.climatology_at(valid_time))
        f = _values(forecast[self.var]) - clim
        o = _values(truth[self.var]) - clim
        w = np.broadcast_to(self.w, f.shape)
        ok = np.isfinite(f) & np.isfinite(o)
        f, o, w = f[ok], o[ok], w[ok]
        denominator = np.sqrt((w * f**2).sum() * (w * o**2).sum())
        return float((w * f * o).sum() / denominator) if denominator > 0 else np.nan


class FrontMHDError(ErrorMetric):
    """Loop Current front displacement between forecast and truth, in km.

    Both fields are contoured at the fixed `level` isoline and the modified
    Hausdorff distance between the two fronts is returned. `referenced` shifts
    both to a common ocean-mean datum first, which matters when the two sides come
    from different datasets or eras and not when they come from the same
    reanalysis.
    """

    name = "ssh_front_mhd"
    unit = "km"

    def __init__(self, var, library, level=LEVEL, referenced=False, main_only=False):
        self.var = var
        self.sampling = library.sampling
        self.level = level
        self.main_only = main_only
        self.kwargs = dict(level=level, main_only=main_only)
        if referenced:
            pool = library.pool(var).values
            ref_mean = float(np.nanmean(np.nanmean(pool, axis=0)[library.ocean]))
            self.kwargs.update(ocean=library.ocean, ref_mean=ref_mean)

    def __call__(self, forecast, truth, valid_time=None):
        return front_mhd_km(front_mask(_values(forecast[self.var]), **self.kwargs),
                            front_mask(_values(truth[self.var]), **self.kwargs),
                            self.sampling)


@METRICS.register("sst_rmse")
def _sst_rmse(library, **kw):
    return WeightedRMSE("sst", library)


@METRICS.register("ssh_rmse")
def _ssh_rmse(library, **kw):
    return WeightedRMSE("ssh", library)


@METRICS.register("sst_acc")
def _sst_acc(library, **kw):
    return AnomalyCorrelation("sst", library)


@METRICS.register("ssh_acc")
def _ssh_acc(library, **kw):
    return AnomalyCorrelation("ssh", library)


@METRICS.register("ssh_front_mhd")
def _ssh_front_mhd(library, level=LEVEL, referenced=False, main_only=False, **kw):
    return FrontMHDError("ssh", library, level=level, referenced=referenced,
                         main_only=main_only)


def build(names, library, **kwargs):
    """Construct the named metrics, failing early on an unknown one."""
    return [METRICS.create(name, library=library, **kwargs) for name in names]
