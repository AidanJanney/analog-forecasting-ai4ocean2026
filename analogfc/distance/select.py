"""Turning distances into a chosen set of analogs.

Selection and forecasting are deliberately separate: an :class:`AnalogSet` is
just dates plus their distances, so one selection can be rolled out to many leads
or inspected on its own without a forecast ever being made.

The one policy here is the separation buffer. The nearest K library days are
usually the same eddy event on consecutive dates, which produces an "ensemble"
of one state repeated K times — confident and wrong. Requiring retained analogs
to sit at least `buffer_days` apart forces the ensemble to draw on genuinely
distinct occurrences.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr


@dataclass
class AnalogSet:
    """The outcome of selection: which days, how far, in rank order.

    Identified by calendar `times` rather than library indices, so advancing an
    analog is date arithmetic and stays correct regardless of where the library
    window ends or whether its time axis has gaps.
    """

    times: np.ndarray
    distances: np.ndarray
    metric: str = "distance"
    unit: str = ""

    def __len__(self):
        return len(self.times)

    @property
    def dates(self):
        """Analog dates as ``YYYY-MM-DD`` strings, in rank order."""
        return [pd.to_datetime(t).strftime("%Y-%m-%d") for t in self.times]

    def report(self, target_date):
        """Print the ranked analogs the way the run logs them."""
        print(f"\n--- Top {len(self)} analogs by {self.metric} for target {target_date} ---")
        for rank, (date, value) in enumerate(zip(self.dates, self.distances), 1):
            print(f"Rank {rank}: {date} | {self.metric} = {value:.4f} {self.unit}")


def select_top_k(scores, k, buffer_days=0, metric="distance", unit=""):
    """The K best-scoring days, kept at least `buffer_days` apart.

    `scores` is a DataArray indexed by `time`; lower is better. Non-finite scores
    are never selected, so a day a metric could not score (no front, say) drops
    out rather than ranking first.
    """
    order = scores.sortby(scores)
    buffer = np.timedelta64(int(buffer_days), "D")
    kept_times, kept_scores = [], []
    for time, value in zip(order.time.values, order.values):
        if not np.isfinite(value):
            break                       # sorted ascending, so the rest are worse
        if kept_times and buffer > np.timedelta64(0, "D"):
            gaps = np.abs((time - np.array(kept_times)).astype("timedelta64[D]"))
            if not np.all(gaps >= buffer):
                continue
        kept_times.append(time)
        kept_scores.append(float(value))
        if len(kept_times) >= k:
            break
    return AnalogSet(np.array(kept_times), np.array(kept_scores), metric, unit)


class AnalogSelector:
    """Rank the library against an observation and return the top analogs.

    Holds the library and a prepared :class:`~.base.ObsDistance`. One selector
    serves every workflow, because the observation is always the same thing by
    the time it gets here: a grid plus a coverage mask on the model grid.

    `region_mask` restricts ranking to a sub-area of the domain, independently of
    the area the resulting forecast is scored over (see :mod:`..data.regions`).
    """

    def __init__(self, library, distance, region_mask=None):
        self.library = library
        self.region_mask = region_mask
        self.distance = distance.prepare(library, region_mask=region_mask)

    def scores(self, obs_grid, mask=None):
        """One score per library day, as a DataArray indexed by time."""
        values = np.asarray(self.distance.distance(obs_grid, mask), dtype=float)
        return xr.DataArray(values, coords={"time": self.library.times}, dims="time")

    def select(self, obs_grid, mask=None, k=5, buffer_days=0):
        """Rank the library against one observation and take the top K."""
        return select_top_k(self.scores(obs_grid, mask), k=k, buffer_days=buffer_days,
                            metric=self.distance.name, unit=self.distance.unit)

    def select_window(self, window, k=5, buffer_days=0):
        """Rank against a multi-day observation window (see :mod:`..data.windows`)."""
        return select_top_k(window.sequence_scores(self), k=k, buffer_days=buffer_days,
                            metric=self.distance.name, unit=self.distance.unit)
