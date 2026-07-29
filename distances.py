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

from fronts import LEVEL, lon_scale, zos_contour_points, extract_fronts, front_distance
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

    def __init__(self, var="zos", level=LEVEL, max_points=200, main_only=False):
        self.var = var                           # kept in the signature for cache parity
        self.level = level
        self.max_points = max_points
        self.main_only = main_only
        self._scale = 1.0
        self._fronts = None
        self._D = None

    @property
    def matrix(self):
        """The full n×n front dissimilarity matrix (available after prepare)."""
        return self._D

    @property
    def fronts(self):
        """The per-state library fronts / descriptors (available after prepare)."""
        return self._fronts

    def signature(self):
        # NOTE: must match metrics.FrontMHD's old signature so the on-disk cache
        # (D_..._front-MHD_zos_0.17_p200_*.npy) keeps hitting after the refactor.
        return f"{self.name}_{self.var}_{self.level:g}_p{self.max_points}"

    # -- reused by analog.build_matrix on a cache miss -----------------------  #
    def describe(self, obs):
        self._scale = lon_scale(obs["latitude"].values)
        return extract_fronts(obs[self.var], self.level, self.max_points)

    def distance_matrix(self, descriptors):
        n = len(descriptors)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = front_distance(descriptors[i], descriptors[j],
                                                   self._scale)
        return D

    def prepare(self, library, source=None, use_cache=True):
        from analog import build_matrix     # local import avoids an import cycle

        self.library = library
        self._scale = lon_scale(library.lat)
        self._fronts = extract_fronts(library.surf_da, self.level, self.max_points)
        self._D = build_matrix(self, library.ds, source=source,
                               descriptors=self._fronts, use_cache=use_cache)
        return self

    def distance(self, obs_grid, mask, self_index=None):
        if self._D is not None and self_index is not None:
            return self._D[self_index].copy()             # matrix fast path (self-source)
        obs_front = zos_contour_points(
            _as_da(obs_grid, self.library.lon, self.library.lat),
            level=self.level, max_points=self.max_points, main_only=self.main_only)
        return np.array([front_distance(obs_front, f, self._scale)
                         for f in self._fronts])


class LatentDistance(ObsDistance):
    """Distance in a learned, forecast-relevant latent space (see :mod:`latent`).

        z_i      = encoder(library.anom[i], full ocean)   # per-state latent vector
        z_obs    = encoder(obs_grid, mask)                # masked obs → latent
        distance = 1 − cos(z_obs, z_i)

    The encoder is trained so that latent distance on the *current* field predicts
    *future* state similarity — minimising ``|d_latent(t0, a) − (1 − S_future(t0, a))|``
    over library pairs — which is the one gap ``obs_gap.py`` finds both significant
    and reachable (D − C = +0.055 ACC). Because it is trained with random swath
    masks and takes the mask as an input channel, a partial SWOT view embeds near the
    full-field embedding of the same state; library states are encoded once at full
    coverage in :meth:`prepare`.

    Pass either a trained ``encoder`` (+ ``scale``) or ``path`` to weights written by
    ``latent.save`` (see ``train_latent.py``). Selection stays surface/observable:
    only ``library.anom`` is encoded.
    """

    name = "latent"

    def __init__(self, encoder=None, path=None, scale=None):
        if encoder is None and path is None:
            raise ValueError(
                "LatentDistance needs a trained encoder: pass encoder=/scale= or "
                "path= to weights from latent.save (train one with train_latent.py).")
        self.encoder = encoder
        self.path = path
        self.scale = scale
        self.meta = None
        self._Z = None

    @property
    def latents(self):
        """The (n, latent_dim) library embeddings (available after prepare)."""
        return self._Z

    def signature(self):
        lead = (self.meta or {}).get("lead", "?")
        return f"{self.name}_L{lead}"

    def prepare(self, library, source=None, use_cache=True):
        import latent as lat

        self.library = library
        if self.encoder is None:
            self.encoder, self.meta = lat.load(self.path)
            self.scale = self.meta["scale"]
        self._Z = lat.encode_all(self.encoder, library.anom, library.ocean, self.scale)
        self._Zn = self._Z / (np.linalg.norm(self._Z, axis=1, keepdims=True) + 1e-8)
        return self

    def distance(self, obs_grid, mask, self_index=None):
        import latent as lat

        # Self-source: the observation IS a library state, so encode the library's
        # own anomaly (matching the space the encoder was trained on). A cross-source
        # observation is already anomaly-like and is encoded as given.
        field = self.library.anom[self_index] if self_index is not None else obs_grid
        x = lat.encoder_inputs(field, mask & self.library.ocean, self.scale)
        z = np.asarray(self.encoder(x))
        z = z / (np.linalg.norm(z) + 1e-8)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return 1.0 - self._Zn @ z     # float32 BLAS may warn spuriously


class LearnedWeightDistance(ObsDistance):
    """``1 − corr_w`` under a *learned* spatial weight map (see :mod:`latent`).

    Identical to :class:`CorrelationDistance` except that the latitude weights are
    replaced by a map fitted so the current-field distance predicts *future*
    similarity (the D − C gap in ``obs_gap.py``). Because the map is initialised at
    ``cos(lat)``, this class starts exactly at the ``CorrelationDistance`` baseline
    and departs from it only where the training data pays for it — unlike a learned
    latent embedding, which must first re-learn the correlation it is replacing.

    Pass ``weights`` (an (nlat, nlon) array) or ``path`` to a ``.npz`` written by
    ``train_weights.py``. Offset/scale invariance is unchanged, so raw SWOT ``ssha``
    may be passed as-is.
    """

    name = "learned-w"

    def __init__(self, weights=None, path=None):
        if weights is None and path is None:
            raise ValueError("LearnedWeightDistance needs weights= or path= "
                             "(train one with train_weights.py).")
        self.path = path
        self.w2d = weights
        self.meta = None

    def signature(self):
        lead = (self.meta or {}).get("lead", "?")
        return f"{self.name}_L{lead}"

    def prepare(self, library, source=None, use_cache=True):
        if self.w2d is None:
            z = np.load(self.path)
            self.w2d = z["w2d"]
            self.meta = {k: z[k] for k in z.files if k != "w2d"}
        self.library = library
        self.anom = library.anom
        self.ocean = library.ocean
        assert self.w2d.shape == self.ocean.shape, "weight map != library grid"
        return self

    def distance(self, obs_grid, mask, self_index=None):
        field = self.anom[self_index] if self_index is not None else obs_grid
        valid = mask & np.isfinite(field) & self.ocean
        o = field[valid].astype(np.float32)
        X = self.anom[:, valid]
        w = self.w2d[valid].astype(np.float32)
        return 1.0 - _weighted_corr(o, X, w)


def _as_da(grid, lon, lat):
    """Wrap a (nlat, nlon) numpy field as a lon/lat DataArray for contouring."""
    import xarray as xr

    return xr.DataArray(grid, dims=("latitude", "longitude"),
                        coords={"latitude": lat, "longitude": lon})
