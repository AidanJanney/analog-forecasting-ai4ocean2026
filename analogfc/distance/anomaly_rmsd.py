"""Spatial RMSD between standardized anomalies — the default selection rule.

Ranks a library day by how far its standardized anomaly field sits from the
target's, in units of local standard deviations. Because the fields are
standardized per grid cell, a departure counts the same everywhere regardless of
how variable that cell is, and the score is comparable between variables.
"""

import numpy as np

from .base import DISTANCES, ObsDistance, weighted_rmsd


class AnomalyRMSD(ObsDistance):
    """Latitude-weighted spatial RMSD of standardized anomalies. Lower is better."""

    representation = "standardized"
    _unit = "sigma"

    def __init__(self, var="ssh"):
        self.var = var
        self.name = f"{var}_rmsd"

    def prepare(self, library):
        self.library = library
        self.pool = library.pool(self.var, self.representation).values   # (n, nlat, nlon)
        w = np.asarray(library.weights.values, dtype=float)
        self.w = np.broadcast_to(w, self.pool.shape[1:])
        return self

    def distance(self, obs_grid, mask):
        diff = self.pool - np.asarray(obs_grid, dtype=float)
        if mask is not None:
            diff = np.where(mask, diff, np.nan)
        w = np.broadcast_to(self.w, diff.shape)
        return weighted_rmsd(diff, w, axis=(-2, -1))


@DISTANCES.register("ssh_rmsd")
def _ssh_rmsd(**kw):
    return AnomalyRMSD(var="ssh", **kw)


@DISTANCES.register("sst_rmsd")
def _sst_rmsd(**kw):
    return AnomalyRMSD(var="sst", **kw)
