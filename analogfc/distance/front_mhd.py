"""Loop Current front displacement — selection on frontal geometry rather than field.

Ranks a library day by how far its Loop Current front sits from the target's,
as a modified Hausdorff distance in km. This ignores the field everywhere the
front is not, which is the point: two days can have similar basin-wide SSH and
put the Loop Current in quite different places, and for a Loop Current forecast
only the second difference matters.

Works on the *raw* field. The front is a property of physical SSH; the
standardized anomaly has exactly the mean structure that defines it removed.

What counts as "the front" — level, datum, segments, area — lives in
:class:`~..fronts.FrontConvention`, shared with the scoring metric so the two
cannot drift apart.
"""

import numpy as np

from ..fronts import LEVEL, FrontConvention
from .base import DISTANCES, ObsDistance


@DISTANCES.register("ssh_front_mhd")
class FrontMHD(ObsDistance):
    """Modified Hausdorff distance between Loop Current fronts, in km.

    `referenced` shifts each field's mean over the scored cells to a common datum
    before contouring, so the fixed level tracks the front's position rather than
    a basin-scale sea-level offset. Leave it off when both sides come from the
    same reanalysis; turn it on across datasets or eras.

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

    def prepare(self, library, region_mask=None):
        self.library = library
        self.front = FrontConvention.from_library(
            library, level=self.level, referenced=self.referenced,
            main_only=self.main_only, region_mask=region_mask, var=self.var)
        pool = library.pool(self.var, self.representation).compute()
        self.fronts = self.front.masks(pool)
        return self

    def distance(self, obs_grid, mask=None):
        target = self.front.mask(obs_grid)
        # One transform serves every comparison: the EDT of the target front *is*
        # the "distance to the nearest target front cell" field the MHD needs.
        target_edt = self.front.edt(target) if target.any() else None
        # inf, not NaN: a day with no front must never be selected as an analog.
        return np.array([self.front.distance(f, target, edt_b=target_edt, empty=np.inf)
                         for f in self.fronts])
