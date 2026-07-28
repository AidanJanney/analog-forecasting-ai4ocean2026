"""Binning along-track swath observations onto the model grid.

The one geometric step between a satellite's native sampling and the pipeline:
everything downstream sees a (nlat, nlon) grid plus a coverage mask, so a
120 km-wide SWOT ribbon and a full model field are the same kind of object.
"""

import numpy as np


def _edges(centers):
    """Cell edges from monotonically-spaced cell centers."""
    c = np.asarray(centers, dtype=float)
    mid = (c[:-1] + c[1:]) / 2
    return np.r_[c[0] - (mid[0] - c[0]), mid, c[-1] + (c[-1] - mid[-1])]


def swath_to_grid(swaths, lon, lat):
    """Bin swath points onto the model grid.

    Parameters
    ----------
    swaths : list of (lon, lat, value) 1-D arrays (see :func:`..data.swot.load_swaths`)
    lon, lat : 1-D grid coordinate arrays (cell centers).

    Returns
    -------
    obs_grid : (nlat, nlon) float array, mean value per cell, NaN where unobserved.
    mask     : (nlat, nlon) bool array, True where at least one point fell.
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
