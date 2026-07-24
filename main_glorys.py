# %% GLORYS-internal model-analog forecasting.
# Analogs are selected *within* the GLORYS library (front-MHD on zos) and
# advanced to forecast the library state. Also tracks front dissimilarity.
# Companion script: main_swot.py (analogs selected from real SWOT observations).
# Modules: fronts / metrics (extraction + pluggable metrics), analog (cached
# dissimilarity matrix, forecaster, skill curve), viz (figures → plots/).
import os

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import viz
from distances import FrontMHDDistance
from metrics import WeightedRMSE, AnomalyCorrelation, FrontMHDError
from analog import (
    ModelLibrary, AnalogForecaster, skill_curve, climatology_of, seasonal_climatology,
)

# %% ---- CONFIG ----------------------------------------------------------- #
DATA_CANDIDATES = [
    "data/glorys_gom_zos_2004_2013.nc",    # 10-year zos library
    "data/glorys_gom_zos_2004.nc",         # single-year fallback
]
DISTANCE = FrontMHDDistance()              # analogs from surface SSH fronts (selection)
FORECAST_VAR = "zos"                       # variable to forecast (e.g. "thetao")
OBS_VAR = "zos"                            # surface observable that drives selection
DEPTHS = None                              # None (surface) or e.g. [0, 1, 2] for a 3-D var
PLOT_DEPTH = 0                             # depth index shown in the map panels
# Exclude analogs within ~1 month of the target: shorter windows leak the
# target's own eddy-shedding event into the analog pool and inflate skill.
K, EXCLUDE = 10, 30                        # analogs kept; temporal exclusion window (days)
MIN_SEP = 30                               # min days apart for the "most similar" pair
LEADS = list(range(1, 16))                 # lead times (days) for the skill curve
SKILL_STRIDE = 20                          # subsample targets for the (costly) 3-metric report
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

# %% Wrap the model, choose the ACC reference, and build the forecaster.
# ACC anomaly reference: a day-of-year seasonal climatology when the library spans
# several years (removes the seasonal cycle so ACC reflects real skill), else the
# static library mean.
lib = ModelLibrary(ds, var=FORECAST_VAR, obs_var=OBS_VAR, depths=DEPTHS)
n_years = len(np.unique(times.astype("datetime64[Y]")))
if n_years >= 3:
    clim = seasonal_climatology(lib.state, times)   # per-time (time, *field) reference
    print(f"ACC reference: day-of-year climatology over {n_years} years")
else:
    clim = climatology_of(lib.state)                # static reference
    print("ACC reference: static library mean")

# Report ACC + RMSE together, plus the Loop Current front error when forecasting SSH.
metrics_list = [AnomalyCorrelation(lib.lat, clim), WeightedRMSE(lib.lat)]
if FORECAST_VAR == OBS_VAR:                          # front MHD needs the SSH field
    ref_mean = float(np.nanmean(lib.mean_surf[lib.ocean]))
    metrics_list.append(FrontMHDError(lib.lon, lib.lat, lib.ocean, ref_mean))
# `source=path` keys the front-MHD matrix cache (built once inside prepare).
af = AnalogForecaster(lib, DISTANCE, error_metrics=metrics_list, source=path)
D, contours = DISTANCE.matrix, DISTANCE.fronts
print(f"analog = {DISTANCE.name} | error = {af.primary_metric.name} | "
      f"forecast {FORECAST_VAR} {lib.state.shape[1:]}")

# %% Dissimilarity visualisations.
viz.plot_fields_and_fronts(lib.surf_da, times, DISTANCE.level)
viz.plot_dissimilarity_matrix(D, times, DISTANCE.name)
viz.plot_dissimilarity_vs_reference(D, times, DISTANCE.name, ref=0)
viz.plot_front_pairs(D, contours, times, min_sep=MIN_SEP)

# %% Forecast demo at one target / lead.
da = ds[FORECAST_VAR]
t0, lead = af.n // 2, 7
viz.plot_forecast_demo(af, t0, lead, day, forecast_var=FORECAST_VAR,
                       analog_name=DISTANCE.name,
                       units=da.attrs.get("units", ""), plot_depth=PLOT_DEPTH,
                       k=K, exclude=EXCLUDE)
# The analogs, their advanced futures, and the weighted average that combines them,
# shown at two lead times (+7 and +14 days).
viz.plot_analog_ensemble(af, t0, [lead, 14], day, k=K, exclude=EXCLUDE,
                         units=da.attrs.get("units", ""), plot_depth=PLOT_DEPTH)

# %% Forecast skill vs lead time (averaged over all valid targets), all metrics.
_, skill_all = af.skill_self(LEADS, k=K, exclude=EXCLUDE, stride=SKILL_STRIDE)
skill = skill_curve(af, LEADS, k=K, exclude=EXCLUDE)   # primary metric (ACC), for plot_skill
viz.plot_skill(LEADS, skill, FORECAST_VAR, DISTANCE.name,
               af.primary_metric.name, day[0][:4])
for mname, base in skill_all.items():
    print(f"  {mname:22s} analog +{LEADS[-1]}d = {base['analog'][-1]:.3f} | "
          f"persistence = {base['persistence'][-1]:.3f}")

plt.show()

# %%
