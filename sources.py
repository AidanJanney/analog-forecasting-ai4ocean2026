"""Pluggable observation sources that drive analog selection.

An :class:`ObsSource` answers "what is the observation on day/query X, gridded onto
the model grid?" and supplies the two verification hooks the skill loop needs:
``present_field`` (a dense field at the observation time, for persistence) and
``verify`` (the truth to score a forecast against). This is the seam that makes the
pipeline source-agnostic — the model comparing against itself (:class:`SelfSource`,
the historic GLORYS-internal case), a real satellite swath (:class:`SwotSource`),
or a gridded L4 product (:class:`OstiaSource`, a stub) all look identical to the
forecaster.

Every ``gridded`` returns ``(obs_grid, mask, self_index)`` where ``self_index`` is
the library index the observation *is* (self-source) or ``None`` (an independent
observation, so no temporal-exclusion leakage guard is needed).
"""

from abc import ABC, abstractmethod

import numpy as np

from swot_analog import swath_to_grid


class ObsSource(ABC):
    """A source of observations, gridded onto the model library's grid."""

    name = "source"
    self_referential = False   # True → obs is a library state; apply temporal exclusion

    @abstractmethod
    def queries(self, library):
        """Iterable of query keys (library indices, or 'YYYY-MM-DD' date strings)."""

    @abstractmethod
    def gridded(self, query, library):
        """Return (obs_grid, mask, self_index) on the library grid for `query`."""

    def present_field(self, query, library):
        """Dense surface field at the observation time (for persistence), or None."""
        return None

    def verify(self, query, lead, library):
        """Truth for a `lead`-day forecast: (field, mask, dense) or None if absent."""
        return None


class SelfSource(ObsSource):
    """Observation = the library's own surface state (the GLORYS-internal case).

    Reproduces the historic within-model analog pipeline: every library day is an
    observation of itself, verified against the dense future library state. Because
    obs and library are the same series, the forecaster applies a temporal
    exclusion window to prevent the target's own eddy event leaking into the pool.
    """

    name = "self"
    self_referential = True

    def __init__(self, indices=None):
        self._indices = indices                  # None → every library day

    def queries(self, library):
        return range(library.n) if self._indices is None else self._indices

    def gridded(self, t, library):
        return library.surf[t], library.ocean, int(t)

    def present_field(self, t, library):
        return library.surf[t]

    def verify(self, t, lead, library):
        v = t + lead
        if v >= library.n:
            return None
        return library.surf[v], library.ocean, True


class SwotSource(ObsSource):
    """Observation = a real SWOT KaRIn swath, binned onto the model grid.

    Selection uses the partial swath (an anomaly-only field → pair with
    ``CorrelationDistance``). Verification depends on ``truth_library``:

    * ``None`` → verify against the *future SWOT swath* (partial coverage;
      supports ACC/RMSE but not the dense Loop Current front MHD). Persistence
      falls back to the analog nowcast (no dense present field).
    * a dense ``ModelLibrary`` covering the SWOT era → verify against the dense
      field at obs-day + lead (enables the front MHD), and use the dense field at
      the obs day as the persistence baseline.
    """

    name = "swot"
    self_referential = False

    def __init__(self, swaths_for, truth_library=None, min_cells=500):
        self.swaths_for = swaths_for             # callable: [dates] -> swaths (or (swaths, labels))
        self.truth_library = truth_library
        self.min_cells = min_cells

    @staticmethod
    def _swaths(x):
        return x[0] if isinstance(x, tuple) else x

    def queries(self, library):
        raise NotImplementedError(
            "Pass the SWOT observation days explicitly (the dates with swaths on "
            "disk); SwotSource does not enumerate the archive itself.")

    def _grid_date(self, date, library):
        return swath_to_grid(self._swaths(self.swaths_for([str(date)])),
                             library.lon, library.lat)

    def gridded(self, date, library):
        obs_grid, mask = self._grid_date(date, library)
        return obs_grid, mask, None

    def present_field(self, date, library):
        if self.truth_library is None:
            return None
        return self.truth_library.field_on(date)

    def verify(self, date, lead, library):
        vday = np.datetime64(str(date)) + np.timedelta64(int(lead), "D")
        if self.truth_library is not None:
            field = self.truth_library.field_on(vday)
            if field is None:
                return None
            return field, library.ocean, True                 # dense truth
        obs2, mask2 = self._grid_date(vday, library)          # partial future swath
        if mask2.sum() < self.min_cells:
            return None
        return obs2, mask2, False


class OstiaSource(ObsSource):
    """Observation = OSTIA L4 SST anomaly, regridded onto the model grid ── STUB.

    Intended design: fetch the OSTIA daily L4 foundation SST (CMEMS
    ``SST_GLO_SST_L4_...`` or PO.DAAC), compute an SST *anomaly* relative to a
    climatology, and bilinearly regrid onto the model grid, returning
    ``(obs_grid, mask, None)`` exactly like :class:`SwotSource`. Pair it with
    :class:`~distances.CorrelationDistance` (SST anomaly is not an absolute SSH
    field, so the front-MHD selector does not apply). No OSTIA data is on disk yet.
    """

    name = "ostia"
    self_referential = False

    def __init__(self, directory="data/ostia"):
        self.directory = directory

    def queries(self, library):
        raise NotImplementedError("OstiaSource is a documented stub; pass dates explicitly.")

    def gridded(self, date, library):
        raise NotImplementedError(
            "OstiaSource is a documented stub. Implement: load the OSTIA L4 SST for "
            "`date`, subtract an SST climatology, and regrid onto "
            "(library.lat, library.lon) → return (obs_grid (nlat,nlon), mask, None).")
