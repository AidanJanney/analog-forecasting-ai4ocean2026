# %% Model-analog forecasting — main workflow.
# Orchestrates the workflow modules end to end. Run cell-by-cell in an
# interactive window (cells are separated by "# %%") or top-to-bottom as a
# script. Modules:
#   fronts / metrics  — front extraction and the pluggable analog / error metrics
#   analog            — dissimilarity matrix (cached), forecaster, skill curve
#   viz               — figures (written to plots/)
#   download_glorys   — fetch the GLORYS library
#   swot_data         — retrieve SWOT surface observations (optional cell below)
import glob
import os

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import viz
import swot_data
from fronts import LEVEL
from metrics import (  # noqa: F401
    FrontMHD, SurfaceFieldRMSE, WeightedRMSE, AnomalyCorrelation,
)
from analog import (
    AnalogForecaster, build_matrix, skill_curve, climatology_of, seasonal_climatology,
)

# %% ---- CONFIG ----------------------------------------------------------- #
DATA_CANDIDATES = [
    "data/glorys_gom_zos_2004_2013.nc",    # 10-year zos library
    "data/glorys_gom_zos_2004.nc",         # single-year fallback
]
ANALOG_METRIC = FrontMHD()                 # analogs from surface SSH fronts (→ SWOT)
FORECAST_VAR = "zos"                       # variable to forecast (e.g. "thetao")
DEPTHS = None                              # None (surface) or slice(0, 10) for a 3-D var
PLOT_DEPTH = 0                             # depth index shown in the map panels
# Exclude analogs within ~1 month of the target: shorter windows leak the
# target's own eddy-shedding event into the analog pool and inflate skill.
K, EXCLUDE = 10, 30                        # analogs kept; temporal exclusion window (days)
MIN_SEP = 30                               # min days apart for the "most similar" pair
LEADS = list(range(1, 16))                 # lead times (days) for the skill curve
# -------------------------------------------------------------------------- #

# %% Load the GLORYS library.
# (Run download_glorys first if absent, e.g.:
#  python download_glorys.py download --variables zos --min-lon -98 --max-lon -78
#      --min-lat 18 --max-lat 32 --start 2004-01-01 --end 2004-12-31
#      --output-filename glorys_gom_zos_2004.nc --output-dir data)
path = next((p for p in DATA_CANDIDATES if os.path.exists(p)), DATA_CANDIDATES[-1])
ds = xr.open_dataset(path)
times = ds["time"].values
day = np.datetime_as_string(times, unit="D")
print(f"Loaded {path}: {ds.sizes['time']} states, {day[0]} .. {day[-1]}")

# %% Extract fronts and build the dissimilarity matrix (analog metric, cached).
contours = ANALOG_METRIC.describe(ds)      # descriptors (FrontMHD: (lon, lat) point sets)
D = build_matrix(ANALOG_METRIC, ds, source=path, descriptors=contours)

# %% Dissimilarity visualisations.
viz.plot_fields_and_fronts(ds["zos"], times, getattr(ANALOG_METRIC, "level", LEVEL))
viz.plot_dissimilarity_matrix(D, times, ANALOG_METRIC.name)
viz.plot_dissimilarity_vs_reference(D, times, ANALOG_METRIC.name, ref=0)
viz.plot_front_pairs(D, contours, times, min_sep=MIN_SEP)

# %% Build the forecaster (state library + error metric).
da = ds[FORECAST_VAR]
if "depth" in da.dims:
    da = da.isel(depth=DEPTHS) if DEPTHS is not None else da.isel(depth=[PLOT_DEPTH])
state = da.values                          # (time, lat, lon) or (time, depth, lat, lon)
# ACC anomaly reference: a day-of-year seasonal climatology when the library
# spans several years (removes the seasonal cycle so ACC reflects real skill),
# else the static library mean.
n_years = len(np.unique(times.astype("datetime64[Y]")))
if n_years >= 3:
    clim = seasonal_climatology(state, times)   # per-time (time, *field) reference
    print(f"ACC reference: day-of-year climatology over {n_years} years")
else:
    clim = climatology_of(state)                # static reference
    print("ACC reference: static library mean")
ERROR_METRIC = AnomalyCorrelation(ds["latitude"].values, clim)
af = AnalogForecaster(ds, ANALOG_METRIC, state, ERROR_METRIC, source=path,
                      climatology=clim if n_years >= 3 else None)
print(f"analog = {ANALOG_METRIC.name} | error = {ERROR_METRIC.name} | "
      f"forecast {FORECAST_VAR} {state.shape[1:]}")

# %% Forecast demo at one target / lead.
t0, lead = af.n // 2, 7
viz.plot_forecast_demo(af, t0, lead, day, forecast_var=FORECAST_VAR,
                       analog_name=ANALOG_METRIC.name,
                       units=da.attrs.get("units", ""), plot_depth=PLOT_DEPTH,
                       k=K, exclude=EXCLUDE)
# The analogs, their advanced futures, and the weighted average that combines them,
# shown at two lead times (+7 and +14 days).
viz.plot_analog_ensemble(af, t0, [lead, 14], day, k=K, exclude=EXCLUDE,
                         units=da.attrs.get("units", ""), plot_depth=PLOT_DEPTH)

# %% Forecast skill vs lead time (averaged over all valid targets).
skill = skill_curve(af, LEADS, k=K, exclude=EXCLUDE)
viz.plot_skill(LEADS, skill, FORECAST_VAR, ANALOG_METRIC.name,
               ERROR_METRIC.name, day[0][:4])

# %% SWOT surface observations over the same box.
# Downloads need a one-time Earthdata login (see swot_data docstring):
#   ! python -c "import earthaccess; earthaccess.login(strategy='interactive', persist=True)"
# swot_data.fetch_swot("2024-01-01", "2024-01-02")          # -> data/swot/*.nc
swot_paths = sorted(glob.glob("data/swot/*.nc"))
if swot_paths:
    swaths, labels = swot_data.load_swaths(swot_paths)      # grouped by pass, calibrated
    print(f"SWOT: {len(swaths)} passes over the box from {len(swot_paths)} granules")
    viz.plot_swot_swaths(swaths, bbox=swot_data.GOM_BBOX,
                         title="SWOT KaRIn SSHA (crossover-calibrated)")
    viz.plot_swot_panels(swaths, labels, bbox=swot_data.GOM_BBOX)
else:
    print("No SWOT granules yet — run swot_data.fetch_swot(start, end) first.")

plt.show()

# %%
