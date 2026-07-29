"""Spatial sub-areas of the loaded domain.

A lon/lat box within the domain, used to restrict *part* of
the pipeline to *part* of the field. Selection and scoring each take their own, so
a run can rank library days on one area and verify on another — rank on the
central Gulf where the Loop Current lives, then score over the whole basin to see
what that selection did everywhere, or rank on the whole field and score only on
the frontal zone you actually care about.

`domain` in a config still bounds what is *loaded*, and so what a region can
select from; regions are always subsets of it.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Region:
    """A lon/lat box. A bound left None is not applied on that side."""

    min_longitude: float = None
    max_longitude: float = None
    min_latitude: float = None
    max_latitude: float = None

    @classmethod
    def from_config(cls, value):
        """Build from a config entry: None, {}, or a mapping of bounds."""
        if value is None:
            return cls()
        if isinstance(value, Region):
            return value
        known = {"min_longitude", "max_longitude", "min_latitude", "max_latitude"}
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"unknown region bound(s) {sorted(unknown)}; "
                             f"choose from {sorted(known)}")
        return cls(**value)

    @property
    def is_whole(self):
        """True when no bound is applied, so the region is the whole domain."""
        return all(b is None for b in (self.min_longitude, self.max_longitude,
                                       self.min_latitude, self.max_latitude))

    @property
    def label(self):
        if self.is_whole:
            return "whole domain"
        def bound(lo, hi, unit):
            if lo is None and hi is None:
                return None
            return f"{'' if lo is None else f'{lo:g}'}..{'' if hi is None else f'{hi:g}'}{unit}"
        parts = [p for p in (bound(self.min_longitude, self.max_longitude, "E"),
                             bound(self.min_latitude, self.max_latitude, "N")) if p]
        return " ".join(parts)

    def mask(self, library):
        """(nlat, nlon) bool mask of the cells inside the box.

        Raises rather than returning an empty mask: a region that misses the
        loaded domain is a config error, and every score downstream would come
        back NaN with nothing to say why.
        """
        lon, lat = library.lon, library.lat
        in_lon = np.ones(lon.shape, dtype=bool)
        in_lat = np.ones(lat.shape, dtype=bool)
        if self.min_longitude is not None:
            in_lon &= lon >= self.min_longitude
        if self.max_longitude is not None:
            in_lon &= lon <= self.max_longitude
        if self.min_latitude is not None:
            in_lat &= lat >= self.min_latitude
        if self.max_latitude is not None:
            in_lat &= lat <= self.max_latitude
        m = in_lat[:, None] & in_lon[None, :]
        if not m.any():
            raise ValueError(
                f"region {self.label} selects no cells of the loaded domain "
                f"({lon.min():g}..{lon.max():g}E, {lat.min():g}..{lat.max():g}N). "
                f"Widen `domain`, or correct the region bounds.")
        return m

    def describe(self, library):
        """One line for the run log: the box and how much of the domain it keeps."""
        if self.is_whole:
            return f"whole domain ({library.latitude.size} x {library.longitude.size})"
        m = self.mask(library)
        rows, cols = np.any(m, axis=1).sum(), np.any(m, axis=0).sum()
        pct = 100.0 * m.sum() / m.size
        return f"{self.label} ({rows} x {cols}, {pct:.0f}% of the domain)"


WHOLE = Region()


def combine(mask, region_mask):
    """AND a region mask into an existing validity mask, tolerating None."""
    if region_mask is None:
        return mask
    return region_mask if mask is None else (mask & region_mask)
