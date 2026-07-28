"""Where an observation comes from — the section 1 seam.

An :class:`ObsSource` answers one question: *what did we observe on date X, on
the model grid?* It returns an :class:`~.windows.ObsDay` (a gridded field plus a
coverage mask) or None. That single method is what makes the pipeline
source-agnostic: the model observing itself, a real satellite swath, and a
gridded L4 product all look identical to :func:`~.windows.build_window` and to
everything downstream.

Instrument-specific policy lives here, not in the window layer — how many
co-located passes to keep on a day is a property of the observing system, so
``max_swaths_per_day`` belongs to :class:`SwotSource` and to nothing else.

A source is constructed knowing which variable and which anomaly representation
the chosen distance wants (``distance.var`` / ``distance.representation``), so
the driver never has to match them up by hand.

Adding an instrument: register an :class:`ObsSource` here and it inherits
multi-day windowing, sequence matching, selection, scoring and plotting unchanged.
"""

from abc import ABC, abstractmethod

import numpy as np

from ..registry import Registry
from .swath import swath_to_grid
from .windows import ObsDay

SOURCES = Registry("observation source")


class ObsSource(ABC):
    """A source of observations, gridded onto the model library's grid."""

    name = "source"

    @abstractmethod
    def observe(self, date, library):
        """Return an :class:`~.windows.ObsDay` for `date`, or None if nothing usable.

        The returned day's ``offset`` is set by the window builder; implementations
        leave it at its default.
        """

    def truth_field(self, date, library):
        """Dense field at `date` for verification/persistence, or None."""
        return None


@SOURCES.register("model")
class ModelSource(ObsSource):
    """Observation = the model's own state on that date.

    The GLORYS-to-GLORYS case. Serves whatever representation the distance asked
    for, which is why the same source works for a front-geometry selection (raw)
    and an RMSD selection (standardized).
    """

    name = "model"

    def __init__(self, var="ssh", representation="raw", library=None):
        self.var = var
        self.representation = representation
        self.library = library

    def observe(self, date, library):
        library = library or self.library
        when = np.datetime64(str(date)[:10], "D")
        if not library.covers(when):
            return None
        grid = library.at(self.var, str(when), self.representation)
        if "time" in grid.dims:                  # a date matches a whole day
            grid = grid.isel(time=0)
        return ObsDay(date=when, grid=grid.values, mask=library.ocean)

    def truth_field(self, date, library):
        library = library or self.library
        when = np.datetime64(str(date)[:10], "D")
        if not library.covers(when):
            return None
        field = library.at(self.var, str(when))
        return field.isel(time=0).values if "time" in field.dims else field.values


@SOURCES.register("swot")
class SwotSource(ObsSource):
    """Observation = real SWOT KaRIn swaths, binned onto the model grid.

    SWOT ``ssha`` is referenced to a mean sea surface and the model to absolute
    topography, so this must be paired with a distance that centres both sides
    over the observed cells (``correlation``). A front-geometry distance needs an
    absolute SSH field and will not work here.

    ``max_swaths_per_day`` caps how many co-located passes one day contributes,
    best-covered first; None keeps every pass.
    """

    name = "swot"

    def __init__(self, swaths_for, truth_library=None, max_swaths_per_day=None):
        self.swaths_for = swaths_for       # callable: [dates] -> swaths (or (swaths, labels))
        self.truth_library = truth_library
        self.max_swaths_per_day = max_swaths_per_day

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

    def observe(self, date, library):
        swaths, labels = self._swaths_on(date)
        if not swaths:
            return None
        grid, mask = swath_to_grid(swaths, library.lon, library.lat)
        return ObsDay(date=np.datetime64(str(date)[:10], "D"), grid=grid, mask=mask,
                      n_obs=len(swaths), labels=labels)

    def truth_field(self, date, library):
        """The dense field at `date`, from the verification library if there is one."""
        if self.truth_library is None:
            return None
        when = np.datetime64(str(date)[:10], "D")
        if not self.truth_library.covers(when):
            return None
        field = self.truth_library.at(self.truth_library.var, str(when))
        return field.isel(time=0).values if "time" in field.dims else field.values


@SOURCES.register("ostia")
class OstiaSource(ObsSource):
    """Observation = OSTIA L4 SST anomaly on the model grid — STUB.

    Kept as the worked example of what a new instrument has to provide. Intended
    design: fetch the OSTIA daily L4 foundation SST, subtract an SST climatology,
    bilinearly regrid onto the model grid, and return an ObsDay exactly as
    :class:`SwotSource` does. Pair it with the ``correlation`` distance — an SST
    anomaly is not an absolute SSH field, so the front distances do not apply.
    """

    name = "ostia"

    def __init__(self, directory="data/ostia"):
        self.directory = directory

    def observe(self, date, library):
        raise NotImplementedError(
            "OstiaSource is a documented stub. Implement: load the OSTIA L4 SST for "
            "`date`, subtract an SST climatology, regrid onto (library.lat, "
            "library.lon), and return ObsDay(date=..., grid=..., mask=...).")
