"""Shared front-extraction and modified-Hausdorff-distance helpers.

A "front" is the Loop Current edge, represented by a fixed sea-surface-height
(``zos``) contour. Each SSH field is reduced to the set of (lon, lat) vertices
tracing that isoline; MHD then measures how far apart two fronts are. Both the
dissimilarity analysis and the forecasting build on these helpers.

Fronts can be subsampled to ``max_points`` vertices: MHD is O(m^2) in the
points per front, so for large multi-year libraries a smooth contour is
well approximated by ~200 evenly-spaced points at a fraction of the cost.
"""

import matplotlib.pyplot as plt
import numpy as np

import mhd  # modified Hausdorff distance

# zos contour that marks the Loop Current edge for this Gulf of Mexico dataset.
LEVEL = 0.17  # m


def lon_scale(latitudes):
    """cos(reference latitude): scales longitudes to an ~isotropic grid.

    A degree of longitude is shorter than a degree of latitude away from the
    equator, so we scale lon by cos(mean lat) before computing distances.
    """
    return float(np.cos(np.deg2rad(np.mean(latitudes))))


def _subsample(points, max_points):
    """Evenly thin a point set to at most `max_points` vertices (None keeps all)."""
    if max_points is None or len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points).astype(int)
    return points[idx]


def _seg_length(seg):
    """Arc length of a contour segment (sum of vertex-to-vertex distances)."""
    if len(seg) < 2:
        return 0.0
    d = np.diff(seg, axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def zos_contour_points(field, level=LEVEL, max_points=None, main_only=False):
    """Return the (N, 2) array of raw (lon, lat) vertices of the `level` SSH contour.

    A single level can produce several disjoint segments (the main Loop Current
    filament plus detached eddies/rings). With `main_only=True`, only the longest
    segment (by arc length) is kept — for metrics that care about the dominant
    front rather than the whole frontal system. Default keeps all segments, the
    convention used by `FrontMHD`/`extract_fronts`.
    """
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(
        field["longitude"].values, field["latitude"].values,
        field.values, levels=[level],
    )
    segs = cs.allsegs[0]  # list of (M, 2) segments for this single level
    plt.close(fig_tmp)    # discard the throwaway figure used only for extraction
    if main_only and segs:
        segs = [max(segs, key=_seg_length)]
    pts = np.vstack(segs) if segs else np.empty((0, 2))
    return _subsample(pts, max_points)


def front_distance(a, b, scale):
    """MHD between two fronts, with longitudes scaled by `scale` (deg)."""
    s = np.array([scale, 1.0])
    return mhd.mhd(a * s, b * s)


def extract_fronts(zos, level=LEVEL, max_points=None):
    """Extract the front point set for every time step of a (time, lat, lon) DataArray."""
    return [zos_contour_points(zos.isel(time=t), level=level, max_points=max_points)
            for t in range(zos.sizes["time"])]


def loop_current_front(field, lon, lat, ocean, ref_mean, level=LEVEL,
                       max_points=200, main_only=True):
    """(N, 2) vertices of the Loop Current (`level`-m ``zos``) contour of a 2-D field.

    The field's ocean-domain mean is first shifted to `ref_mean` so the fixed
    contour is comparable across datasets/eras: the Gulf's mean SSH differs by a
    few cm between years/seasons, which would otherwise expand or shrink the
    `level` isoline for reasons unrelated to the Loop Current's position.
    Referencing to a common mean makes the contour track the front's
    *shape/position*, not the basin-scale sea-level offset.

    `field`, `ocean` are (nlat, nlon); `lon`, `lat` are the 1-D grid centers.
    With `main_only=True` (default) only the longest contour segment is kept — the
    Loop Current filament itself rather than detached eddies/rings that also cross
    `level`.
    """
    import xarray as xr  # local import keeps fronts.py light for non-grid callers

    adj = field - (float(np.nanmean(field[ocean])) - ref_mean)
    da = xr.DataArray(adj, dims=("latitude", "longitude"),
                      coords={"latitude": lat, "longitude": lon})
    return zos_contour_points(da, level=level, max_points=max_points,
                              main_only=main_only)
