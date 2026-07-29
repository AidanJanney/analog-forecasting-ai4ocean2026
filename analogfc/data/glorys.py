"""GLORYS reanalysis input: open the Zarr stores and build the field library.

This is where "getting data" ends and the rest of the pipeline begins. Everything
downstream sees a :class:`FieldSet` — named variables, each carrying its physical
field plus the anomaly representations selection works in — and never touches a
file path or a variable-name guess again.

Adding a variable is one entry in :data:`FIELD_SPECS` plus its name in the
config; nothing else changes.
"""

from dataclasses import dataclass, field as dataclass_field

import numpy as np
import xarray as xr

from ..fronts import grid_spacing_km
from . import climatology as clim

# Display and detection metadata per variable. `candidates` is tried in order
# against the dataset's variables, so a store naming SST `thetao` instead of
# `tos` needs no config change.
FIELD_SPECS = {
    "sst": dict(candidates=("tos", "thetao"), unit="°C", cmap="plasma",
                symmetric=False),
    "ssh": dict(candidates=("zos",), unit="m", cmap="RdBu_r", symmetric=True),
}


@dataclass
class Field:
    """One variable, in every representation the pipeline needs.

    `raw` is the physical field. The two anomaly representations are built on
    first use and cached, because they are full-size arrays and no single run
    needs both: front geometry reads `raw`, the RMSD selection and the rollout
    read `standardized`, and the cross-datum correlation reads `anomaly`.

    * `standardized` — ``(x - mu) / sigma`` against the per-group, per-gridpoint
      climatology. The space analog selection and the forecast rollout work in.
    * `anomaly` — ``x - time mean``. Mean removed but *not* rescaled per cell,
      which is what a correlation against an observation on a different datum
      needs: standardizing applies a spatially varying rescaling that a raw
      instrument anomaly has not had applied to it.
    """

    name: str
    raw: xr.DataArray
    unit: str
    cmap: str
    symmetric: bool
    group: str = "climday"
    reduce_dims: tuple = ()
    _cache: dict = dataclass_field(default_factory=dict, repr=False)

    def _standardize(self):
        if "standardized" not in self._cache:
            mean, std, norm = clim.normalize(self.raw, self.group, self.reduce_dims)
            self._cache.update(clim_mean=mean, clim_std=std, standardized=norm)
        return self._cache

    @property
    def standardized(self):
        return self._standardize()["standardized"]

    @property
    def clim_mean(self):
        return self._standardize()["clim_mean"]

    @property
    def clim_std(self):
        return self._standardize()["clim_std"]

    @property
    def anomaly(self):
        if "anomaly" not in self._cache:
            self._cache["anomaly"] = self.raw - self.raw.mean(dim="time")
        return self._cache["anomaly"]

    def to_physical(self, anomaly, valid_time):
        """Map a standardized anomaly back to physical units at `valid_time`."""
        label = clim.group_label(self.raw, valid_time, self.group)
        return clim.to_physical(anomaly, label, self.clim_mean, self.clim_std)

    def climatology_at(self, valid_time):
        """Climatological mean for the group `valid_time` falls in."""
        label = clim.group_label(self.raw, valid_time, self.group)
        group_dim = self.clim_mean.dims[0]
        return self.clim_mean.sel({group_dim: label})


class FieldSet:
    """The named variables of one run, on a shared grid.

    Behaves as a mapping (``fields["ssh"]``, ``for name in fields``) and carries
    the grid-derived quantities every section needs: the latitude weights that
    make spatial means area-correct, and the physical cell size that puts front
    distances in km.
    """

    def __init__(self, fields):
        self._fields = dict(fields)
        first = next(iter(self._fields.values())).raw
        self.longitude = first.longitude
        self.latitude = first.latitude
        self.time = first.time

    def __getitem__(self, name):
        return self._fields[name]

    def __iter__(self):
        return iter(self._fields)

    def __len__(self):
        return len(self._fields)

    def items(self):
        return self._fields.items()

    @property
    def weights(self):
        """cos(latitude) weights, broadcast over the grid.

        Every spatial mean in the pipeline is area-weighted with these: a grid
        cell at 31 N covers ~7% less area than one at 18 N, so an unweighted mean
        over the Gulf box systematically overcounts the northern shelf.
        """
        w = np.cos(np.deg2rad(self.latitude))
        return w.broadcast_like(self[next(iter(self))].raw.isel(time=0))

    @property
    def sampling(self):
        """(dlat_km, dlon_km) physical cell size, for front distances in km."""
        return grid_spacing_km(self.longitude.values, self.latitude.values)


def open_glorys(zarr_glob, lon_slice, lat_slice, variables=("sst", "ssh"),
                group="climday", reduce_dims=(), verbose=True):
    """Open the per-year Zarr stores and build the :class:`FieldSet`.

    The domain subset is applied here, before any statistic, ranking or plot, so
    every downstream number refers to the same box.
    """
    ds = xr.open_mfdataset(zarr_glob)
    ds = ds.sel(longitude=lon_slice, latitude=lat_slice)
    if verbose:
        print(f"Domain: {float(ds.longitude.min()):.2f} to {float(ds.longitude.max()):.2f} lon, "
              f"{float(ds.latitude.min()):.2f} to {float(ds.latitude.max()):.2f} lat "
              f"({ds.sizes['latitude']} x {ds.sizes['longitude']})")

    fields = {}
    for name in variables:
        spec = FIELD_SPECS[name]
        var = next((c for c in spec["candidates"] if c in ds), None)
        if var is None:
            raise KeyError(f"no variable for {name!r} in the store; tried "
                           f"{spec['candidates']}, found {list(ds.data_vars)}")
        da = ds[var]
        if "depth" in da.dims:          # SST is the surface level of a 3-D field
            da = da.isel(depth=0)
        fields[name] = Field(name=name, raw=clim.with_climday(da),
                             unit=spec["unit"], cmap=spec["cmap"],
                             symmetric=spec["symmetric"],
                             group=group, reduce_dims=tuple(reduce_dims))
    return FieldSet(fields)


def summarize(fields):
    """Min/max/mean/variance/std per field, printed and returned."""
    stats = {}
    for name, f in fields.items():
        da = f.raw
        s = dict(min=float(da.min().compute()), max=float(da.max().compute()),
                 mean=float(da.mean().compute()), variance=float(da.var().compute()),
                 std_dev=float(da.std().compute()))
        stats[name] = s
        print(f"\n[{name.upper()}] ({f.unit})")
        print(f"  Min:      {s['min']:.2f} {f.unit}")
        print(f"  Max:      {s['max']:.2f} {f.unit}")
        print(f"  Mean:     {s['mean']:.2f} {f.unit}")
        print(f"  Variance: {s['variance']:.4f}")
        print(f"  Std Dev:  {s['std_dev']:.2f} {f.unit}")
    return stats
