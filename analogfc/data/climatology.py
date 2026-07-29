"""Climatological normalization: the anomaly space analog selection works in.

Fields are standardized against a climatology before matching, so a day is
compared on its departure from what that calendar day normally looks like rather
than on the seasonal cycle every year shares. Two knobs, both config-exposed:

* ``group`` — what the statistics are grouped by. ``climday`` folds Feb 29 into
  Feb 28 (see :func:`noleap_dayofyear`); ``time.dayofyear`` and ``time.month``
  also work.
* ``reduce_dims`` — extra dims to average over. Empty keeps the statistics
  pointwise: one mean and standard deviation per group *per grid cell*.

:func:`to_physical` is the inverse, mapping a forecast anomaly back into physical
units against the climatology of the day being forecast. It is affine, which is
why an ensemble mean may be taken in either space and give the same answer.
"""

import warnings

import numpy as np
import xarray as xr


def noleap_dayofyear(time):
    """Day of year with Feb 29 folded into Feb 28.

    Feb 29 is day 60 of a leap year, so shifting day 60 onwards back by one both
    pools Feb 29 with Feb 28 (day 59) and keeps the rest of the year aligned with
    non-leap years (leap Mar 1 = 61 -> 60 = Mar 1). The result is 365 groups with
    equal sample counts; without it, day 366 is estimated from only the leap
    years in the record, which degenerates to a zero standard deviation (and a
    division by zero) on a shortened record.
    """
    doy = time.dt.dayofyear
    return xr.where(time.dt.is_leap_year & (doy >= 60), doy - 1, doy)


def with_climday(da):
    """Attach the `climday` coordinate `groupby` needs."""
    return da.assign_coords(climday=noleap_dayofyear(da.time))


def normalize(da, group="climday", reduce_dims=()):
    """Standardize a DataArray against its climatology.

    Returns ``(clim_mean, clim_std, normalized)`` — the statistics come back too,
    because the forecast rollout needs them to map anomalies back into physical
    units.
    """
    stat_dims = ["time", *reduce_dims]

    da = da.compute()
    grouped = da.groupby(group)
    # Land cells are NaN in every sample of every group, so nanmean/nanstd hit an
    # all-NaN slice by construction, not by error. numpy reports that through
    # warnings.warn rather than the floating-point trap np.errstate controls, so
    # the warning is silenced explicitly rather than left to print per group.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
        clim_mean = grouped.mean(dim=stat_dims)
        clim_std = grouped.std(dim=stat_dims)

    # Filling a preallocated output group by group keeps peak memory at two copies
    # of the field. Groupby arithmetic, `(da.groupby(g) - mean).groupby(g) / std`,
    # is tidier but builds a full-size temporary per operation, which overruns the
    # job's memory limit on the full record.
    group_dim = clim_mean.dims[0]
    labels = da[group].values
    values = da.values
    out = np.empty_like(values)
    for label in clim_mean[group_dim].values:
        in_group = labels == label
        mean = clim_mean.sel({group_dim: label}).values
        std = clim_std.sel({group_dim: label}).values
        out[in_group] = (values[in_group] - mean) / std

    return clim_mean, clim_std, da.copy(data=out)


def group_label(reference, valid_time, group="climday"):
    """The climatology group `valid_time` falls in, read off a `reference` array.

    Read from the data rather than derived from the timestamp so any `group`
    expression works — ``climday`` is a coordinate, not something recomputable
    from a bare date.
    """
    return int(reference.sel(time=valid_time)[group])


def to_physical(anomaly, label, clim_mean, clim_std):
    """Undo the standardization, turning an anomaly back into physical units.

    `label` is the climatology group of the day being forecast, from
    :func:`group_label`.
    """
    group_dim = clim_mean.dims[0]
    return clim_mean.sel({group_dim: label}) + anomaly * clim_std.sel({group_dim: label})
