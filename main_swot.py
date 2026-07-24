# %% SWOT-driven model-analog forecasting (real SWOT obs → GLORYS analogs).
# Analogs are selected from real SWOT KaRIn swaths against the GLORYS library:
# distance is measured in observation space (bin SWOT onto the GLORYS grid and
# correlate anomalies where observed, swot_analog). The analogs' futures give the
# forecast, verified against a *later* SWOT pass. Disjoint eras (SWOT 2023+ vs
# library 2004–2013) ⇒ every analog is independent, no exclusion window needed.
# Companion script: main_glorys.py (analogs selected within GLORYS).
import glob
import os
import re

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import viz
import swot_data
import swot_analog as sa
from analog import ModelLibrary, AnalogForecaster
from distances import CorrelationDistance

# %% ---- CONFIG ----------------------------------------------------------- #
DATA_CANDIDATES = [
    "data/glorys_gom_zos_2004_2013.nc",    # 10-year zos library (analog pool)
    "data/glorys_gom_zos_2004.nc",         # single-year fallback
]
SWOT_DIR = "data/swot"
K = 15                                     # analogs kept for SWOT-driven forecasts
K_SHOW = 8                                 # analogs shown in the illustration figure
SWOT_LEADS = range(1, 15)                  # lead times (days) for the skill aggregation
DEMO_LEADS = (7, 10)                       # leads shown in the forecast map
MIN_CELLS = 500                            # min observed grid cells to use a swath
# -------------------------------------------------------------------------- #

# %% Load the GLORYS library (the analog pool) and its surface state.
path = next((p for p in DATA_CANDIDATES if os.path.exists(p)), DATA_CANDIDATES[-1])
ds = xr.open_dataset(path)
times = ds["time"].values
day = np.datetime_as_string(times, unit="D")
lib = ModelLibrary(ds, var="zos")          # the model (GLORYS), abstracted
print(f"Loaded {path}: {ds.sizes['time']} states, {day[0]} .. {day[-1]}")

# %% SWOT observations over the same box.
# Downloads need a one-time Earthdata login (see swot_data docstring):
#   ! python -c "import earthaccess; earthaccess.login(strategy='interactive', persist=True)"
# swot_data.fetch_swot("2024-01-01", "2024-01-02")          # -> data/swot/*.nc
swot_paths = sorted(glob.glob(os.path.join(SWOT_DIR, "*.nc")))
if not swot_paths:
    print("No SWOT granules yet — run swot_data.fetch_swot(start, end) first.")
else:
    swaths, labels = swot_data.load_swaths(swot_paths)      # grouped by pass, calibrated
    print(f"SWOT: {len(swaths)} passes over the box from {len(swot_paths)} granules")
    viz.plot_swot_swaths(swaths, bbox=swot_data.GOM_BBOX,
                         title="SWOT KaRIn SSHA (crossover-calibrated)")
    viz.plot_swot_panels(swaths, labels, bbox=swot_data.GOM_BBOX)

# %% SWOT-driven analog selection & forecast.
# The observation is a real SWOT swath (an anomaly-like field), so analogs are
# selected in observation space with CorrelationDistance (no temporal exclusion —
# the SWOT era and the library are disjoint).
if swot_paths:
    af = AnalogForecaster(lib, CorrelationDistance())
    swot_days = sorted({re.search(r"_(\d{8})T", p).group(1) for p in swot_paths})
    obs_days = [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in swot_days]        # all days present

    # --- illustration: analogs of the earliest obs, and a forecast map ---
    illo = obs_days[:2]
    obs_grid, obs_mask = sa.swath_to_grid(swot_data.swaths_for_dates(illo)[0],
                                          af.lon, af.lat)
    sel, dist = af.analogs_obs(obs_grid, obs_mask, 0, K_SHOW)
    print(f"SWOT obs {illo[0]}: best GLORYS analog {day[sel[0]]} (r={1 - dist.min():.2f})")
    viz.plot_swot_analogs(obs_grid, obs_mask, af, sel, dist, day, k=K_SHOW)

    t0 = np.datetime64(illo[0])
    results = []
    for L in DEMO_LEADS:
        vday = t0 + np.timedelta64(L, "D")
        obs2, mask2 = sa.swath_to_grid(
            swot_data.swaths_for_dates([str(vday)])[0], af.lon, af.lat)
        if mask2.sum() < MIN_CELLS:
            continue
        # Deseasonalize forecast and truth so the maps/score show mesoscale skill.
        fc_anom = af.deseasonalize(af.anom_of(af.forecast_obs(obs_grid, obs_mask, L, k=K)[0]), vday)
        obs2 = af.deseasonalize(obs2, vday)
        results.append(dict(L=L, fc_anom=fc_anom, obs2=obs2, mask2=mask2,
                            r_analog=af.score(fc_anom, obs2, mask2)))
    if results:
        viz.plot_swot_forecast(af, illo[0], results)

    # --- aggregated skill (ACC + RMSE) over every obs day ---
    leads, acc, rmse, counts = sa.aggregate_skill(
        af, obs_days, SWOT_LEADS, swot_data.swaths_for_dates, k=K, min_cells=MIN_CELLS)
    viz.plot_swot_skill(leads, acc, rmse, counts, n_obs=len(obs_days))
    good = [i for i, c in enumerate(counts) if c]
    if good:
        i = good[0]
        print(f"aggregated skill @lead {leads[i]} (n={counts[i]}): "
              f"ACC analog/persist/clim = {acc['analog'][i]:.2f}/"
              f"{acc['persistence'][i]:.2f}/{acc['climatology'][i]:.2f}")

plt.show()

# %%
