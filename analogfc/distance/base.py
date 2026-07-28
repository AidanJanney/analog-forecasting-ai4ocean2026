"""The analog-selection seam: how similar an observation is to every library day.

An :class:`ObsDistance` answers one question — given a gridded observation on the
model grid, how dissimilar is it from each state in the library? Lower ranks
better. That single method is where the selection rule lives, so replacing the
classical distances with a learned one is a config string, not a code change.

A distance declares two things about the data it wants, and the driver wires the
observation source to match:

* ``var`` — which variable it compares (``ssh``, ``sst``, ...).
* ``representation`` — which anomaly space it works in. Front geometry needs the
  ``raw`` physical field; the RMSD selection needs the ``standardized`` anomaly;
  a cross-datum correlation needs the mean-removed but unscaled ``anomaly``.

Every spatial mean here is latitude-weighted, so a grid cell at 31 N does not
count the same as one covering ~7% more area at 18 N.

Ranking may be restricted to a sub-area with a region mask, independently of the
area a forecast is later *scored* over — see :mod:`..data.regions`.
"""

from abc import ABC, abstractmethod

import numpy as np

from ..registry import Registry

DISTANCES = Registry("distance")


class ObsDistance(ABC):
    """Dissimilarity of one gridded observation to every library state."""

    name = "distance"
    var = "ssh"
    representation = "raw"

    def prepare(self, library, region_mask=None):
        """Precompute the library-side representation. Returns self.

        `region_mask` restricts ranking to a sub-area of the domain (see
        :mod:`..data.regions`); None ranks over the whole thing.
        """
        self.library = library
        self.region_mask = region_mask
        return self

    @abstractmethod
    def distance(self, obs_grid, mask):
        """Distances from one observation to every library state -> ndarray (n,).

        `obs_grid` and `mask` are (nlat, nlon) on the model grid, NaN/False where
        unobserved. Return ``np.inf`` for a library day that cannot be scored, so
        it is never selected.
        """

    @property
    def unit(self):
        """Units the scores are reported in."""
        return getattr(self, "_unit", "")


def weighted_corr(o, X, w):
    """Weighted Pearson correlation between vector `o` (m,) and each row of `X` (n, m).

    Centering both sides over the compared cells is what makes this usable across
    datums: an observation referenced to a mean sea surface and a model field
    referenced to absolute topography differ by an offset this removes.
    """
    W = w.sum()
    mo = (w * o).sum() / W
    mx = (X * w).sum(axis=1) / W
    oc = o - mo
    Xc = X - mx[:, None]
    cov = (w * oc * Xc).sum(axis=1) / W
    vo = (w * oc * oc).sum() / W
    vx = (w * Xc * Xc).sum(axis=1) / W
    denom = np.sqrt(vo * vx)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, cov / denom, 0.0)


def weighted_rmsd(diff, w, axis):
    """Latitude-weighted RMS of `diff` over `axis`, ignoring non-finite cells."""
    valid = np.isfinite(diff)
    ww = np.where(valid, w, 0.0)
    sq = np.where(valid, diff, 0.0) ** 2
    total = ww.sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt((ww * sq).sum(axis=axis) / np.where(total > 0, total, np.nan))
