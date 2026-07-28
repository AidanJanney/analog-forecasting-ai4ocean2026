"""Pluggable observation sources that drive analog selection.

An :class:`ObsSource` answers one question: *what did we observe on date X, on the
model grid?* — returning an :class:`obs_window.ObsDay` (a gridded field plus a
coverage mask) or None. That single method is the seam that makes the pipeline
source-agnostic: the model comparing against itself (:class:`SelfSource`, the
GLORYS-to-GLORYS case), a real satellite swath (:class:`SwotSource`), or a gridded
L4 product (:class:`OstiaSource`, a stub) all look identical to
:func:`obs_window.build_window` and everything downstream.

Instrument-specific policy lives here, not in the window layer — how many
co-located passes to keep on a day is a property of the observing system, so
``max_swaths_per_day`` belongs to :class:`SwotSource`.

Two optional verification hooks let a source drive a skill loop as well:
``present_field`` (a dense field at the observation time, for persistence) and
``verify`` (the truth a forecast is scored against).

Adding an instrument: implement :meth:`ObsSource.observe`, and it inherits
multi-day windowing, sequence matching, selection and the driver unchanged.
"""

from abc import ABC, abstractmethod

import numpy as np

from obs_window import ObsDay
from swot_analog import swath_to_grid


class ObsSource(ABC):
    """A source of observations, gridded onto the model library's grid."""

    name = "source"
    self_referential = False   # True → obs is a library state; apply temporal exclusion

    @abstractmethod
    def observe(self, date, library):
        """Return an :class:`obs_window.ObsDay` for `date`, or None if nothing usable.

        The returned day's ``offset`` is set by the window builder; implementations
        leave it at its default. Set ``self_index`` only for a self-referential
        source, where the observation *is* library state `self_index`.
        """

    def present_field(self, date, library):
        """Dense surface field at the observation time (for persistence), or None."""
        return None

    def verify(self, date, lead, library):
        """Truth for a `lead`-day forecast: (field, mask, dense) or None if absent."""
        return None


class SelfSource(ObsSource):
    """Observation = the library's own surface state (the GLORYS-internal case).

    Reproduces the within-model analog pipeline: every library day is an
    observation of itself, verified against the dense future library state.
    Because obs and library are the same series, the day carries ``self_index``,
    which makes :meth:`analog.AnalogSelector.select` apply a temporal exclusion
    window so the target's own eddy event cannot leak into its analog pool.
    """

    name = "self"
    self_referential = True

    def _index(self, date, library):
        """Library index for a calendar date, or None if outside the record."""
        d = np.datetime64(str(date)[:10], "D")
        days = library.times.astype("datetime64[D]")
        hit = np.flatnonzero(days == d)
        return int(hit[0]) if hit.size else None

    def observe(self, date, library):
        i = self._index(date, library)
        if i is None:
            return None
        return ObsDay(date=np.datetime64(str(date)[:10], "D"), grid=library.surf[i],
                      mask=library.ocean, self_index=i)

    def present_field(self, date, library):
        i = self._index(date, library)
        return None if i is None else library.surf[i]

    def verify(self, date, lead, library):
        i = self._index(date, library)
        if i is None or i + lead >= library.n:
            return None
        return library.surf[i + lead], library.ocean, True


class SwotSource(ObsSource):
    """Observation = real SWOT KaRIn swaths, binned onto the model grid.

    Selection uses the partial swath (an anomaly-only field → pair with
    ``CorrelationDistance``). ``max_swaths_per_day`` caps how many co-located
    passes a single day contributes, best-covered first; None keeps every pass.

    Verification depends on ``truth_library``:

    * ``None`` → verify against the *future SWOT swath* (partial coverage;
      supports ACC/RMSE but not the dense Loop Current front MHD). Persistence
      falls back to the analog nowcast (no dense present field).
    * a dense ``ModelLibrary`` covering the SWOT era → verify against the dense
      field at obs-day + lead (enables the front MHD), and use the dense field at
      the obs day as the persistence baseline.
    """

    name = "swot"
    self_referential = False

    def __init__(self, swaths_for, truth_library=None, max_swaths_per_day=None,
                 min_cells=500):
        self.swaths_for = swaths_for         # callable: [dates] -> swaths (or (swaths, labels))
        self.truth_library = truth_library
        self.max_swaths_per_day = max_swaths_per_day
        self.min_cells = min_cells

    @staticmethod
    def _pick(swaths, labels, max_swaths):
        """Keep at most `max_swaths` passes, the best-covered ones first."""
        if max_swaths is None or len(swaths) <= max_swaths:
            return swaths, labels
        order = sorted(np.argsort([-len(s[0]) for s in swaths])[:max_swaths].tolist())
        return [swaths[i] for i in order], [labels[i] for i in order]

    def _swaths_on(self, date):
        """(swaths, labels) for one date, after the per-day cap."""
        got = self.swaths_for([str(date)[:10]])
        swaths, labels = got if isinstance(got, tuple) else (got, [])
        swaths = list(swaths)
        labels = list(labels) if labels else [""] * len(swaths)
        return self._pick(swaths, labels, self.max_swaths_per_day)

    def _grid_date(self, date, library):
        swaths, _ = self._swaths_on(date)
        return swath_to_grid(swaths, library.lon, library.lat)

    def observe(self, date, library):
        swaths, labels = self._swaths_on(date)
        if not swaths:
            return None
        grid, mask = swath_to_grid(swaths, library.lon, library.lat)
        return ObsDay(date=np.datetime64(str(date)[:10], "D"), grid=grid, mask=mask,
                      n_obs=len(swaths), labels=labels)

    def present_field(self, date, library):
        if self.truth_library is None:
            return None
        return self.truth_library.field_on(date)

    def verify(self, date, lead, library):
        vday = np.datetime64(str(date)[:10], "D") + np.timedelta64(int(lead), "D")
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
    climatology, and bilinearly regrid onto the model grid, returning an
    :class:`obs_window.ObsDay` exactly like :class:`SwotSource`. Pair it with
    :class:`~distances.CorrelationDistance` (SST anomaly is not an absolute SSH
    field, so the front-MHD selector does not apply). No OSTIA data is on disk yet.
    """

    name = "ostia"
    self_referential = False

    def __init__(self, directory="data/ostia"):
        self.directory = directory

    def observe(self, date, library):
        raise NotImplementedError(
            "OstiaSource is a documented stub. Implement: load the OSTIA L4 SST for "
            "`date`, subtract an SST climatology, regrid onto (library.lat, "
            "library.lon), and return ObsDay(date=..., grid=..., mask=...).")
