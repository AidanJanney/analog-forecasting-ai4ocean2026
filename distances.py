"""Pluggable observation-space distances that rank analogs.

An :class:`ObsDistance` answers one question: given a gridded observation
(`obs_grid`, `mask`) on the model grid, how dissimilar is it from *every* state
in the model library? It is the seam where the analog-selection rule lives, so
swapping the classical spatial-correlation distance for a learned latent-space
distance is a one-line change at the call site.

Two cost models share the interface:

* on-the-fly — :class:`CorrelationDistance` (and, later, :class:`LatentDistance`)
  compute obs-vs-library per query. They work for *any* source, including partial
  swaths, because they only ever compare the observed cells.
* precomputed matrix — :class:`FrontMHDDistance` reduces, for a self-source, to the
  cached n×n front dissimilarity matrix (``analog.build_matrix``), so the historic
  GLORYS-internal pipeline keeps its on-disk cache and O(n²/2) cost.

Selection stays *surface / observable*: distances see only the surface field a
sensor could measure. The multi-depth forecast target lives in the state library,
not here (see the surface-only-analog-selection design note).
"""

from abc import ABC, abstractmethod

import numpy as np

from fronts import (LEVEL, front_edt, front_mask, front_masks, front_mhd_km,
                    grid_spacing_km)
from swot_analog import _weighted_corr   # re-used correlation kernel (see swot_analog)


class ObsDistance(ABC):
    """Dissimilarity of a gridded observation to every library state."""

    name = "distance"

    def prepare(self, library, source=None, use_cache=True):
        """Precompute any library-side representation. Returns self."""
        self.library = library
        return self

    @abstractmethod
    def distance(self, obs_grid, mask, self_index=None):
        """Distances from one observation to every library state → ndarray (n,).

        `obs_grid`/`mask` are (nlat, nlon) on the model grid (NaN/False where
        unobserved). `self_index`, when given, is the library index the
        observation *is* (self-source) — matrix-backed distances use it as a fast
        path; on-the-fly distances ignore it.
        """

    def signature(self):
        """Short string identifying the distance + params (used for cache keys)."""
        return self.name


class CorrelationDistance(ObsDistance):
    """``1 − latitude-weighted spatial correlation`` of anomalies over observed cells.

    The default, source-agnostic distance. Because the correlation centers both
    the observation and each library anomaly over the observed cells, it is
    invariant to the offset/scale difference between an observation (e.g. SWOT
    ``ssha`` referenced to a mean sea surface) and the library (GLORYS ``zos``
    absolute topography) — so raw ``obs_grid`` may be passed as-is. Range 0..2.
    """

    name = "corr"

    def prepare(self, library, source=None, use_cache=True):
        self.library = library
        self.anom = library.anom                 # (n, nlat, nlon) library anomalies
        self.ocean = library.ocean
        self.wlat2d = library.wlat2d
        return self

    def distance(self, obs_grid, mask, self_index=None):
        # Self-source: the observation IS a library state, so compare in the same
        # anomaly space as the library (the raw surface still carries the spatial
        # mean-surface pattern, which would contaminate the correlation). A
        # cross-source observation (e.g. SWOT ssha) is already an anomaly-like
        # field and is used as-is — correlation centering absorbs its offset.
        field = self.anom[self_index] if self_index is not None else obs_grid
        valid = mask & np.isfinite(field) & self.ocean
        o = field[valid].astype(np.float32)
        X = self.anom[:, valid]                  # (n, n_valid)
        w = self.wlat2d[valid].astype(np.float32)
        return 1.0 - _weighted_corr(o, X, w)


class FrontMHDDistance(ObsDistance):
    """Modified Hausdorff distance between Loop Current fronts (a fixed SSH contour).

    Longitudes are scaled to an ~isotropic grid before the distance is taken. For a
    self-source the full symmetric matrix is built once and cached on disk (keyed by
    the source file + this distance's signature), reproducing the historic
    GLORYS-internal selector. For a cross-source (partial) observation the obs front
    is contoured from ``obs_grid`` and compared to each cached library front — note
    this needs an *absolute* SSH field (it will not work on an anomaly-only swath
    such as SWOT ``ssha``; pair those with :class:`CorrelationDistance`).
    """

    name = "front-MHD"

    def __init__(self, var="zos", level=LEVEL, main_only=False):
        self.var = var                           # kept in the signature for cache parity
        self.level = level
        self.main_only = main_only
        self.sampling = (1.0, 1.0)
        self.ref_mean = None
        self._fronts = None
        self._D = None

    @property
    def matrix(self):
        """The full n×n front dissimilarity matrix (available after prepare)."""
        return self._D

    @property
    def fronts(self):
        """The per-state library front masks (available after prepare)."""
        return self._fronts

    def signature(self):
        # "edt" distinguishes this from the older contour/cdist matrices, whose
        # cached .npy files are in different units and must not be reused.
        return f"{self.name}_{self.var}_{self.level:g}_edt"

    # -- reused by analog.build_matrix on a cache miss -----------------------  #
    def describe(self, obs):
        return self._fronts

    def distance_matrix(self, masks):
        """Symmetric n×n front MHD (km) over the library's front masks.

        ``M[j, i]`` is the mean distance from front *i*'s cells to front *j*, read
        off *j*'s distance transform, so the whole row costs one EDT plus a gather.
        MHD is the symmetric maximum, hence ``max(M, M.T)`` — n transforms in
        total rather than an O(n²) pairwise sweep.
        """
        n = len(masks)
        pts = [np.flatnonzero(m) for m in masks]
        counts = np.array([p.size for p in pts])
        ok = counts > 0
        M = np.full((n, n), np.inf)
        if not ok.any():
            return M
        allpts = np.concatenate(pts)
        offs = np.concatenate([[0], np.cumsum(counts)[:-1]])
        for j in range(n):
            if not ok[j]:
                continue
            edt = front_edt(masks[j], self.sampling).ravel()
            # reduceat returns junk for the zero-length segments; `ok` drops them.
            sums = np.add.reduceat(edt[allpts], offs)
            M[j, ok] = sums[ok] / counts[ok]
        D = np.maximum(M, M.T)
        np.fill_diagonal(D, 0.0)
        return D

    def prepare(self, library, source=None, use_cache=True):
        from analog import build_matrix     # local import avoids an import cycle

        self.library = library
        self.sampling = grid_spacing_km(library.lon, library.lat)
        self.ref_mean = float(np.nanmean(library.mean_surf[library.ocean]))
        self._fronts = front_masks(library.surf, level=self.level,
                                   ocean=library.ocean, ref_mean=self.ref_mean,
                                   main_only=self.main_only)
        self._D = build_matrix(self, library.ds, source=source,
                               descriptors=self._fronts, use_cache=use_cache)
        return self

    def distance(self, obs_grid, mask, self_index=None):
        if self._D is not None and self_index is not None:
            return self._D[self_index].copy()             # matrix fast path (self-source)
        obs_front = front_mask(obs_grid, level=self.level, ocean=mask,
                               ref_mean=self.ref_mean, main_only=self.main_only)
        obs_edt = front_edt(obs_front, self.sampling) if obs_front.any() else None
        # inf, not NaN: a state with no front must never be selected as an analog.
        return np.array([front_mhd_km(obs_front, f, self.sampling, edt_a=obs_edt,
                                      empty=np.inf) for f in self._fronts])


class LatentDistance(ObsDistance):
    """Distance in a learned latent-encoded space  ── STUB (not implemented).

    Intended design (gated by the oracle diagnostic — see oracle.py /
    metric_diagnostic.py, which measure whether a better *current-field* selection
    metric has headroom before this is worth building):

        z_i    = encoder(library.surf[i])          # per-state latent vector
        z_obs  = encoder(fill(obs_grid, mask))      # masked obs → latent
        distance = 1 − cos(z_obs, z_i)

    The encoder is trained so that the latent distance on the *current* field
    predicts *future* state similarity (the present-oracle ceiling): minimise
    ``| d_latent(t0, a) − (1 − S_future(t0, a)) |`` over library pairs. Only the
    surface/observable field is encoded — depth stays in the forecast target.
    """

    name = "latent"

    def __init__(self, encoder=None, **kwargs):
        self.encoder = encoder
        self._kwargs = kwargs

    def prepare(self, library, source=None, use_cache=True):
        raise NotImplementedError(
            "LatentDistance is a documented stub. Implement an encoder that maps "
            "the surface field to a latent vector and train it so current-field "
            "latent distance predicts future similarity (see oracle.py for the "
            "headroom gate that justifies building it). Until then, use "
            "CorrelationDistance.")

    def distance(self, obs_grid, mask, self_index=None):
        raise NotImplementedError("LatentDistance is a documented stub; see prepare().")


