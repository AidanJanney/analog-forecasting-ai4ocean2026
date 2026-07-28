"""Loop Current front displacement — selection on frontal geometry rather than field.

Ranks a library day by how far its Loop Current front sits from the target's,
as a modified Hausdorff distance in km. This ignores the field everywhere the
front is not, which is the point: two days can have similar basin-wide SSH and
put the Loop Current in quite different places, and for a Loop Current forecast
only the second difference matters.

Works on the *raw* field. The front is a property of physical SSH; the
standardized anomaly has exactly the mean structure that defines it removed.
"""

import numpy as np

from ..fronts import LEVEL, front_edt, front_mask, front_mhd_km
from .base import DISTANCES, ObsDistance


@DISTANCES.register("ssh_front_mhd")
class FrontMHD(ObsDistance):
    """Modified Hausdorff distance between Loop Current fronts, in km.

    `ref_mean` (with the ocean mask) shifts each field's ocean-domain mean to a
    common datum before contouring, so the fixed level tracks the front's
    *position* rather than a basin-scale sea-level offset. Leave it None to
    contour fields as they are, which is right when both sides come from the same
    reanalysis; set it when comparing across datasets or eras.

    `main_only` keeps just the largest connected segment — the Loop Current
    filament itself rather than detached rings that also cross the level.
    """

    name = "ssh_front_mhd"
    representation = "raw"
    _unit = "km"

    def __init__(self, var="ssh", level=LEVEL, referenced=False, main_only=False):
        self.var = var
        self.level = level
        self.referenced = referenced
        self.main_only = main_only

    def _mask_kwargs(self):
        if not self.referenced:
            return dict(level=self.level, main_only=self.main_only)
        return dict(level=self.level, ocean=self.library.ocean,
                    ref_mean=self.ref_mean, main_only=self.main_only)

    def prepare(self, library):
        self.library = library
        self.sampling = library.sampling
        self.ref_mean = None
        pool = library.pool(self.var, self.representation).compute()
        if self.referenced:
            mean_surf = np.nanmean(pool.values, axis=0)
            self.ref_mean = float(np.nanmean(mean_surf[library.ocean]))
        self.fronts = [front_mask(grid, **self._mask_kwargs()) for grid in pool.values]
        return self

    def distance(self, obs_grid, mask=None):
        target = front_mask(np.asarray(obs_grid), **self._mask_kwargs())
        # One transform serves every comparison: the EDT of the target front *is*
        # the "distance to the nearest target front cell" field the MHD needs.
        target_edt = front_edt(target, self.sampling) if target.any() else None
        # inf, not NaN: a day with no front must never be selected as an analog.
        return np.array([front_mhd_km(f, target, self.sampling, edt_b=target_edt,
                                      empty=np.inf) for f in self.fronts])
