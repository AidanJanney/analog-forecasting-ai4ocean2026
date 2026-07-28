"""Multi-day, multi-instrument observation windows — the initialization for a forecast.

A single SWOT swath sees a ~120 km ribbon of the Gulf on one day. That is a thin
constraint on the circulation state, so this module lets an *observation window*
initialize a forecast instead: a run of consecutive days, each carrying however
many observations happened to fall in the box that day, all binned onto the model
grid.

Two knobs, both modular, matching how the observing system actually behaves:

* ``n_days`` — how many consecutive days the window spans. 1 reproduces the
  original single-day behaviour exactly.
* how many observations to keep per day — owned by the source, since it is
  instrument-specific (:class:`~.sources.SwotSource` takes ``max_swaths_per_day``).

The window is matched against the library as a *sequence*, not as one flattened
field (see :meth:`ObsWindow.sequence_scores`): day *o* of the window is compared
against the library state *o* days before the candidate. A candidate analog
therefore has to reproduce the whole observed evolution, not just the final
snapshot, and a front moving through the window is not smeared into a thick blur
the way flattening every day onto one grid would smear it.

This module is deliberately free of any instrument knowledge. A day is a gridded
field plus a coverage mask (:class:`ObsDay`), which is all any observation
reduces to, whatever its native geometry. Where those arrays come from is the
business of
:mod:`.sources`; how they are ranked against the library is the business of
:mod:`..distance`.

Two ways in, one representation out:

* :func:`build_window` — pull `n_days` back from a :class:`~.sources.ObsSource`.
* :func:`window_from_days` — hand over pre-gridded (grid, mask) pairs directly,
  for an observation type that has no ``ObsSource`` yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

DAY = np.timedelta64(1, "D")


@dataclass
class ObsDay:
    """One day of an observation window, gridded onto the model grid.

    `grid` is NaN where unobserved and `mask` is True where observed, so any
    observation type reduces to the same pair regardless of its native geometry.
    `offset` is how many days before the window's end this day sits (0 = the
    window's last/newest day), which is what aligns it to the library sequence.

    """

    date: np.datetime64
    grid: np.ndarray
    mask: np.ndarray
    offset: int = 0
    n_obs: int = 1
    labels: list = field(default_factory=list)

    @property
    def n_cells(self):
        return int(self.mask.sum())


@dataclass
class ObsWindow:
    """A run of consecutive observed days used to initialize one forecast.

    `days` is ordered newest-first (``offset`` 0, 1, 2, ...) and may be sparser
    than the nominal span: dates with no usable observation are simply absent, so
    a 5-day window over a 21-day-repeat orbit typically holds 1-3 days. `weights`
    is aligned to `days` and sums to 1.
    """

    days: list
    weights: np.ndarray
    end_date: np.datetime64
    n_days_requested: int = 1

    def __len__(self):
        return len(self.days)

    @property
    def dates(self):
        return [d.date for d in self.days]

    @property
    def max_offset(self):
        return max((d.offset for d in self.days), default=0)

    @property
    def n_cells(self):
        """Total observed cells across the window (days may overlap in space)."""
        return int(sum(d.n_cells for d in self.days))

    @property
    def n_obs(self):
        """Total observations (e.g. SWOT passes) across the window."""
        return int(sum(d.n_obs for d in self.days))

    @property
    def coverage_mask(self):
        """Union of the days' masks — everywhere the window saw *something*.

        This is what answers "did this window sample the Loop Current at all",
        and what the figures outline. It is deliberately not what the matching
        uses: :meth:`sequence_scores` keeps the days separate so each is
        compared against its own library day.
        """
        m = np.zeros_like(self.days[0].mask, dtype=bool)
        for d in self.days:
            m |= d.mask
        return m

    @property
    def composite(self):
        """Weighted mean of the observed days, for display only (NaN off-swath).

        Collapsing the window like this discards the time alignment, so it must
        never feed the matching — it exists so a multi-day window can be drawn as
        a single panel.
        """
        num = np.zeros_like(self.days[0].grid, dtype=float)
        den = np.zeros_like(num)
        for d, w in zip(self.days, self.weights):
            v = np.where(d.mask & np.isfinite(d.grid), d.grid, 0.0)
            num += w * v
            den += w * (d.mask & np.isfinite(d.grid))
        with np.errstate(invalid="ignore"):
            return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

    def sequence_scores(self, selector):
        """Scores against every library day, as a sequence match.

        Composes any :class:`~..distance.base.ObsDistance` over the window's days::

            D[j] = sum_o  w_o * d_o[j - o]

        where `j` is the library day aligned with the window's *end* and `o` is a
        day's offset back from that end. So a candidate has to reproduce the whole
        observed evolution, not just the final snapshot — and a front moving
        through the window is not smeared the way flattening every day onto one
        grid would smear it. Candidates too close to the start of the library to
        have a full history are marked infinite and never selected.

        Returns a DataArray indexed by time. With a one-day window this reduces to
        exactly that day's distance, so the windowed and single-day paths agree by
        construction.
        """
        n = selector.library.n
        total = np.zeros(n, dtype=float)
        for day, w in zip(self.days, self.weights):
            d = np.asarray(selector.distance.distance(day.grid, day.mask), dtype=float)
            o = int(day.offset)
            if o == 0:
                shifted = d
            else:
                shifted = np.full(n, np.inf)
                shifted[o:] = d[:-o]      # library day j aligns with the window's j - o
            total = total + w * shifted
        return xr.DataArray(total, coords={"time": selector.library.times}, dims="time")

    def describe(self):
        parts = [f"{str(d.date)[:10]}(-{d.offset}d, {d.n_obs}ob, {d.n_cells}c)"
                 for d in self.days]
        return f"window end {str(self.end_date)[:10]}: " + " ".join(parts)


# --------------------------------------------------------------------------- #
# Building windows.
# --------------------------------------------------------------------------- #
def _day_weights(offsets, scheme="uniform", halflife=3.0):
    """Per-day weights for the window, normalised to sum to 1.

    "uniform" treats every observed day alike. "recency" halves a day's influence
    every `halflife` days back, for when the newest observation should dominate
    the match (the forecast is launched from the window's end, so the state there
    matters most) while older days still constrain the evolution.
    """
    o = np.asarray(offsets, dtype=float)
    if scheme == "recency":
        w = 0.5 ** (o / float(halflife))
    elif scheme == "uniform":
        w = np.ones_like(o)
    else:
        raise ValueError(f"unknown weight scheme {scheme!r}; use 'uniform' or 'recency'")
    total = w.sum()
    return w / total if total > 0 else w


def _assemble(days, end, n_days, weights, halflife):
    """Order the days newest-first, weight them, and wrap them in an ObsWindow."""
    days = sorted(days, key=lambda d: d.offset)
    w = _day_weights([d.offset for d in days], scheme=weights, halflife=halflife)
    return ObsWindow(days=days, weights=w, end_date=end, n_days_requested=int(n_days))


def build_window(source, end_date, library, n_days=1, min_cells_per_day=1,
                 require_days=1, weights="uniform", halflife=3.0):
    """Build an :class:`ObsWindow` ending on `end_date` from an observation source.

    Walks back `n_days` from the window end, asking the source what it saw on each
    date. Because the source is the only thing that knows about instruments, this
    one function serves SWOT, a future L4 SST product, and the model observing
    itself — a new instrument implements :meth:`~.sources.ObsSource.observe` and
    inherits windowing, sequence matching and the whole driver.

    Parameters
    ----------
    source : ~.sources.ObsSource
        Supplies ``observe(date, library) -> ObsDay | None``.
    end_date : str or datetime64
        Last (newest) day of the window; the forecast is launched from here.
    library : ~.library.ModelLibrary
        The model library, which defines the grid the observation is binned onto.
    n_days : int
        Number of consecutive calendar days in the window (1 = a single day).
    min_cells_per_day : int
        A day contributing fewer observed grid cells than this is dropped.
    require_days : int
        Minimum usable days for the window to be returned at all; otherwise None.

    Returns
    -------
    ObsWindow or None
    """
    end = np.datetime64(str(end_date)[:10], "D")
    days = []
    for offset in range(int(n_days)):             # 0 = newest, counting back
        day = source.observe(end - offset * DAY, library)
        if day is None or day.n_cells < min_cells_per_day:
            continue
        day.offset = offset
        days.append(day)

    if len(days) < max(1, int(require_days)):
        return None
    return _assemble(days, end, n_days, weights, halflife)


def window_from_days(end_date, grids, masks, dates, weights="uniform", halflife=3.0):
    """Build a window from pre-gridded fields, bypassing :mod:`.sources`.

    The escape hatch for an observation type that has no ``ObsSource`` yet: produce
    (grid, mask) pairs on the model grid and their dates, and everything downstream
    — sequence matching, selection, forecasting, scoring — is unchanged. Prefer
    :func:`build_window` once the source exists, so the type is reusable.
    """
    end = np.datetime64(str(end_date)[:10], "D")
    days = []
    for g, m, d in zip(grids, masks, dates):
        date = np.datetime64(str(d)[:10], "D")
        days.append(ObsDay(date=date, grid=g, mask=np.asarray(m, dtype=bool),
                           offset=int((end - date) / DAY)))
    if not days:
        return None
    n_days = max(d.offset for d in days) + 1
    return _assemble(days, end, n_days, weights, halflife)


def available_windows(dates, stride=1):
    """Candidate window end-dates: every observed date, thinned by `stride`.

    `dates` is the list of dates with observations on disk (e.g.
    ``.swot.observed_dates()``). A window is only *attempted* here; whether it
    has enough usable days is decided by :func:`build_window`.
    """
    ends = sorted({np.datetime64(str(d)[:10], "D") for d in dates})
    return ends[::max(1, int(stride))]


def selection_dates(library, analogs):
    """Calendar dates of a selection's library indices (the window-end alignment)."""
    return [np.datetime_as_string(library.times[i], unit="D") for i in analogs.indices]
