"""Loop Current front extraction and front-to-front distance.

A "front" is the Loop Current edge, represented by a fixed sea-surface-height
(``zos``) contour. Here a front is a **boolean grid mask** — the cells the
contour passes through — and the distance between two fronts is the modified
Hausdorff distance (MHD) evaluated with a Euclidean distance transform.

Why the distance transform. MHD is

    MHD(A, B) = max( mean_a min_b d(a,b),  mean_b min_a d(a,b) )

so it needs, for every point of A, the distance to the nearest point of B. The
EDT of ``~B`` *is* that field, computed for every cell at once in O(N). Reading
it at A's cells gives the same nearest-neighbour distances a full pairwise
``cdist`` would, without the O(|A|·|B|) cost or the vertex subsampling it forces.
``mhd.py`` keeps the direct pairwise definition; the tests check the two agree.

Distances come out in **km**, not grid cells: :func:`grid_spacing_km` turns the
lon/lat spacing into physical cell sizes and hands them to the EDT as its
``sampling``, which also fixes the grid's anisotropy — a degree of longitude at
25 N is only ~0.9 of a degree of latitude, so an uncorrected EDT would overstate
zonal displacement by ~10%.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, label

# zos contour that marks the Loop Current edge for this Gulf of Mexico dataset.
LEVEL = 0.17  # m

# Degrees of latitude to km, for reporting front displacement in physical units.
DEG_KM = 111.195

# 8-connectivity: a front is a curve, so diagonal steps belong to the same segment.
_CONNECTIVITY = np.ones((3, 3), dtype=int)


def lon_scale(latitudes):
    """cos(reference latitude): scales longitudes to an ~isotropic grid.

    A degree of longitude is shorter than a degree of latitude away from the
    equator, so we scale lon by cos(mean lat) before computing distances.
    """
    return float(np.cos(np.deg2rad(np.mean(latitudes))))


def grid_spacing_km(lon, lat, deg_km=DEG_KM):
    """(dlat_km, dlon_km) physical cell size of a regular lon/lat grid.

    This is the ``sampling`` argument for the distance transform, in the array's
    axis order (latitude first). Longitude is scaled by cos(mean latitude), the
    same isotropy correction :func:`lon_scale` applies.
    """
    dlat = abs(float(np.mean(np.diff(np.asarray(lat, dtype=float))))) * deg_km
    dlon = abs(float(np.mean(np.diff(np.asarray(lon, dtype=float))))) * deg_km
    return (dlat, dlon * lon_scale(lat))


def _largest_segment(mask):
    """Keep only the largest connected component of a front mask."""
    lab, n = label(mask, structure=_CONNECTIVITY)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0                                  # background
    return lab == int(np.argmax(sizes))


def referenced(field, ocean=None, ref_mean=None):
    """Field shifted so its ocean-domain mean equals `ref_mean` (a no-op if None).

    The datum the fixed `level` contour assumes. Figures contour this so the line
    they draw is the same one :func:`front_mask` measures.
    """
    f = np.asarray(field, dtype=float)
    if f.ndim != 2:
        f = f[0]
    if ref_mean is None:
        return f
    where = np.isfinite(f) if ocean is None else ocean
    return f - (float(np.nanmean(f[where])) - ref_mean)


def front_mask(field, level=LEVEL, ocean=None, ref_mean=None, main_only=False):
    """Boolean mask of the Loop Current front: the `level` SSH contour cells.

    Returns the inside edge of the region at or above `level` — cells above it
    that are 4-adjacent to water below it. Requiring the neighbour to be *water*
    keeps coastlines out of the mask, which would otherwise trace every shore
    where high SSH meets land.

    `ref_mean` (with `ocean`) first shifts the field's ocean-domain mean to that
    reference, so the fixed contour is comparable across datasets and eras: the
    Gulf's mean SSH differs by a few cm between years, which would otherwise
    expand or shrink the `level` isoline for reasons unrelated to the Loop
    Current's position. Leave it None to contour the field as-is.

    `main_only` keeps just the largest connected segment — the Loop Current
    filament itself rather than detached eddies/rings that also cross `level`.
    """
    f = referenced(field, ocean=ocean, ref_mean=ref_mean)
    inside = np.nan_to_num(f, nan=-np.inf) >= level
    water_below = ~inside & np.isfinite(f)

    neighbour_below = np.zeros_like(inside)
    neighbour_below[1:, :] |= water_below[:-1, :]
    neighbour_below[:-1, :] |= water_below[1:, :]
    neighbour_below[:, 1:] |= water_below[:, :-1]
    neighbour_below[:, :-1] |= water_below[:, 1:]

    m = inside & neighbour_below
    return _largest_segment(m) if (main_only and m.any()) else m


def front_edt(mask, sampling):
    """Distance (km) from every grid cell to the nearest front cell of `mask`."""
    return distance_transform_edt(~mask, sampling=sampling)


def front_mhd_km(mask_a, mask_b, sampling, edt_a=None, edt_b=None, empty=np.nan):
    """Front displacement in km: the modified Hausdorff distance between two fronts.

    Pass `edt_a`/`edt_b` when a transform is already in hand (comparing many
    fronts against one truth computes the truth's EDT once). `empty` is returned
    when either front has no cells — NaN for scoring, so it drops out of a
    ``nanmean``; pass ``np.inf`` when ranking, so an empty front is never chosen.
    """
    if not mask_a.any() or not mask_b.any():
        return empty
    if edt_a is None:
        edt_a = front_edt(mask_a, sampling)
    if edt_b is None:
        edt_b = front_edt(mask_b, sampling)
    return float(max(edt_b[mask_a].mean(), edt_a[mask_b].mean()))


def front_masks(states, level=LEVEL, ocean=None, ref_mean=None, main_only=False):
    """Front mask for every state of a (time, nlat, nlon) array or DataArray."""
    arr = states.values if hasattr(states, "values") else np.asarray(states)
    return [front_mask(arr[t], level=level, ocean=ocean, ref_mean=ref_mean,
                       main_only=main_only) for t in range(arr.shape[0])]


def front_coverage(mask, observed):
    """Fraction of a front's cells that an observation actually saw.

    Answers "did this observation sample the Loop Current at all", as opposed to
    merely covering enough ocean somewhere in the box. 0 for an empty front.
    """
    n = int(mask.sum())
    return float((mask & observed).sum()) / n if n else 0.0


class FrontConvention:
    """Everything that defines *which* front is being measured, in one object.

    A front is not just a contour level: it is the level, the datum the level is
    taken against, whether detached segments count, the area being looked at, and
    the physical cell size that turns the result into km. Selection and scoring
    must agree on all five, or a run ranks library days by one front and measures
    its error against a different one — which produces no error, just numbers that
    quietly mean two things.

    So the bundle lives here, built once from the library and handed to both the
    :class:`~.distance.front_mhd.FrontMHD` distance and the
    :class:`~.forecast.score.FrontMHDError` metric.

    Parameters
    ----------
    sampling : (dlat_km, dlon_km)
        Physical cell size, so distances come out in km.
    level : float
        The SSH contour defining the front, in metres.
    ocean : (nlat, nlon) bool or None
        Land mask. Needed only when `ref_mean` is set.
    ref_mean : float or None
        Shift each field's mean over the scored cells to this datum before
        contouring, so the fixed level tracks the front's *position* rather than a
        basin-scale sea-level offset. None contours fields as they are, which is
        right when both sides come from the same reanalysis.
    main_only : bool
        Keep only the largest connected segment.
    region_mask : (nlat, nlon) bool or None
        Restrict the front to a sub-area. Cells outside it are excluded from the
        field before contouring, so they can neither carry front cells nor
        influence the datum.
    """

    def __init__(self, sampling, level=LEVEL, ocean=None, ref_mean=None,
                 main_only=False, region_mask=None):
        self.sampling = sampling
        self.level = level
        self.ocean = ocean
        self.ref_mean = ref_mean
        self.main_only = main_only
        self.region_mask = region_mask

    @classmethod
    def from_library(cls, library, level=LEVEL, referenced=False, main_only=False,
                     region_mask=None, var=None):
        """Build for a library, computing the datum from it when `referenced`.

        The datum is the mean over the cells actually being contoured — the ocean
        within the region — so a regional front is referenced to that region and
        not to a basin mean it never looks at.
        """
        ocean = library.ocean
        ref_mean = None
        if referenced:
            scored = ocean if region_mask is None else (ocean & region_mask)
            pool = library.pool(var or library.var).values
            # Indexed by `scored` before nanmean, so the permanently-NaN land
            # cells never enter the reduction — computing over the whole grid
            # first and discarding land afterwards would still hit an all-NaN
            # slice and warn.
            ref_mean = float(np.nanmean(pool[:, scored]))
        return cls(library.sampling, level=level, ocean=ocean, ref_mean=ref_mean,
                   main_only=main_only, region_mask=region_mask)

    def _restrict(self, field):
        """Blank out everything outside the region.

        NaN rather than a separate mask argument: `front_mask` already treats a
        non-finite cell as neither inside the contour nor water below it, which is
        exactly how a cell outside the region should behave — the same as cropping
        the array to the region would make its edge behave.
        """
        f = np.asarray(field, dtype=float)
        if f.ndim != 2:
            f = f[0]
        if self.region_mask is None:
            return f
        return np.where(self.region_mask, f, np.nan)

    def mask(self, field):
        """The front of one field, as a boolean grid mask."""
        return front_mask(self._restrict(field), level=self.level, ocean=self._ocean(),
                          ref_mean=self.ref_mean, main_only=self.main_only)

    def _ocean(self):
        if self.ocean is None or self.region_mask is None:
            return self.ocean
        return self.ocean & self.region_mask

    def masks(self, states):
        """Front masks for a (time, nlat, nlon) array or DataArray."""
        arr = states.values if hasattr(states, "values") else np.asarray(states)
        return [self.mask(arr[t]) for t in range(arr.shape[0])]

    def edt(self, mask):
        """Distance (km) from every cell to the nearest front cell of `mask`."""
        return front_edt(mask, self.sampling)

    def distance(self, mask_a, mask_b, edt_a=None, edt_b=None, empty=np.nan):
        """Front displacement in km between two front masks."""
        return front_mhd_km(mask_a, mask_b, self.sampling, edt_a=edt_a, edt_b=edt_b,
                            empty=empty)

    def describe(self):
        bits = [f"{self.level:g} m contour"]
        bits.append("referenced" if self.ref_mean is not None else "unreferenced")
        if self.main_only:
            bits.append("largest segment only")
        if self.region_mask is not None:
            bits.append("regional")
        return ", ".join(bits)
