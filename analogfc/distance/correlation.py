"""Pattern correlation over the observed cells — the cross-datum selection rule.

The source-agnostic distance. Because it centres both the observation and each
library anomaly over the cells actually observed, it is invariant to the offset
and scale difference between an instrument's datum (SWOT ``ssha``, referenced to
a mean sea surface) and the model's (GLORYS ``zos``, absolute topography), so a
raw swath can be compared to the library without conversion. It is also the only
distance here that works on *partial* coverage.

Uses the ``anomaly`` representation — mean removed, but not rescaled per grid
cell. The standardized representation would divide each cell by its own standard
deviation, a spatially varying rescaling that an instrument anomaly has not had
applied to it, which would make the two sides incomparable.
"""

import numpy as np

from .base import DISTANCES, ObsDistance, weighted_corr


@DISTANCES.register("correlation")
class CorrelationDistance(ObsDistance):
    """``1 - latitude-weighted spatial correlation`` over the observed cells.

    Range 0..2; 0 is a perfect pattern match.
    """

    name = "correlation"
    representation = "anomaly"
    _unit = "1 - r"

    def __init__(self, var="ssh"):
        self.var = var

    def prepare(self, library):
        self.library = library
        self.anom = np.asarray(
            library.pool(self.var, self.representation).values, dtype=np.float32)
        self.ocean = library.ocean
        self.w = np.broadcast_to(np.asarray(library.weights.values, dtype=float),
                                 self.anom.shape[1:])
        return self

    def distance(self, obs_grid, mask=None):
        field = np.asarray(obs_grid)
        valid = np.isfinite(field) & self.ocean
        if mask is not None:
            valid = valid & mask
        o = field[valid].astype(np.float32)
        X = self.anom[:, valid]                       # (n, n_valid)
        w = self.w[valid].astype(np.float32)
        return 1.0 - weighted_corr(o, X, w)
