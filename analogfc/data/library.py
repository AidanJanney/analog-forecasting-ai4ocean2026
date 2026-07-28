"""The analog pool: which model states a forecast may be built from.

A :class:`ModelLibrary` is a time window of a :class:`~.glorys.FieldSet` — the
days analogs are drawn from — plus the grid quantities every section needs
(ocean mask, latitude weights, physical cell size). It is the single object
passed to a distance, a combiner and a metric, so none of them has to know where
the data came from.

The library window and the forecast target period are normally disjoint, which is
what makes the selection honest: an analog cannot be the target day itself, or a
neighbour of it, so no exclusion window is needed. The rollout still reads the
*full* record, because an analog selected in 1997 must be followed forward
through 1997 regardless of where the library window ends.
"""

import numpy as np

REPRESENTATIONS = ("raw", "standardized", "anomaly")


class ModelLibrary:
    """A time-sliced view of a FieldSet, with the grid metadata the pipeline needs.

    `fields` is the full record; `period` is the slice of it analogs may come
    from. :meth:`pool` serves the library-window array a distance ranks over,
    while :meth:`at` reads any date in the full record, which is how a selected
    analog is advanced past the end of the library window.
    """

    def __init__(self, fields, period=slice(None), var="ssh"):
        self.fields = fields
        self.period = period
        self.var = var
        self.times = fields.time.sel(time=period).values
        self.n = len(self.times)
        self.longitude = fields.longitude
        self.latitude = fields.latitude
        self.lon = fields.longitude.values
        self.lat = fields.latitude.values
        self.sampling = fields.sampling
        self.weights = fields.weights
        self._ocean = None

    @property
    def ocean(self):
        """Constant land mask: cells that are finite in the reference field."""
        if self._ocean is None:
            ref = self.fields[self.var].raw.isel(time=0).values
            self._ocean = np.isfinite(ref)
        return self._ocean

    # -- representations ----------------------------------------------------- #
    def series(self, var, representation="raw"):
        """The full-record DataArray of `var` in `representation`."""
        if representation not in REPRESENTATIONS:
            raise ValueError(f"unknown representation {representation!r}; "
                             f"choose from {list(REPRESENTATIONS)}")
        return getattr(self.fields[var], representation)

    def pool(self, var, representation="raw"):
        """The library window of `var` in `representation` — what a distance ranks."""
        return self.series(var, representation).sel(time=self.period)

    def at(self, var, when, representation="raw"):
        """`var` on one date, read from the full record (may be outside the pool)."""
        return self.series(var, representation).sel(time=when)

    def covers(self, when):
        """Whether the full record covers `when` (a rollout ran off the end if not)."""
        t = self.fields.time.values
        return bool(t.min() <= np.datetime64(when) <= t.max())

    def timestamp(self, date):
        """The record's exact stamp for a calendar date, or None if absent.

        GLORYS timestamps carry a 12:00 time of day, so a bare date parses to
        midnight and misses on an exact ``.sel``. Every entry point that takes a
        date from a config or a filename goes through here.
        """
        day = np.datetime64(str(date)[:10], "D")
        hit = np.flatnonzero(self.fields.time.values.astype("datetime64[D]") == day)
        return None if hit.size == 0 else self.fields.time.values[int(hit[0])]
