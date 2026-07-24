# %% SWOT-selected analogs -> 14-day forecast, verified against dense GLORYS.
# Real 2024 SWOT swaths select analogs from the disjoint 2004-2013 GLORYS library
# (observation-space distance, swot_analog). The selected analogs are DE-CLUSTERED
# (>= MIN_SEP days apart, so they are distinct events), each advanced LEAD days, and
# EACH analog's forecast is scored separately against the real GLORYS zos field at
# the SWOT-obs date + LEAD (downloaded for 2024). Because the truth is a full field,
# verification is dense over the whole box and a real dense-persistence baseline
# (GLORYS@obs carried to obs+LEAD) is available. All scores are deseasonalized
# (mesoscale). Companion scripts: main_glorys.py, main_swot.py.
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
LIB_CANDIDATES = [
    # "data/glorys_gom_zos_2004_2013.nc",    # 10-year zos library (analog pool)
    # "data/glorys_gom_zos_2004.nc",         # single-year fallback
    "data/glorys_gom_zos_thetao_all.nc"
]
TRUTH_PATH = "data/glorys_gom_zos_2024jan.nc"   # dense GLORYS truth over the SWOT era
SWOT_DIR = "data/swot"
K = 10                                     # de-clustered analogs kept per obs day
MIN_SEP = 14                               # min days between selected analogs
LEAD = 14                                  # forecast lead time (days)
MIN_CELLS = 500                            # min observed SWOT cells to use an obs day
DEMO_DAY = "2024-06-23"                    # obs day shown in the per-analog map figure
# -------------------------------------------------------------------------- #

# %% Load the GLORYS library (analog pool) and the dense 2024 truth.
lib_path = next((p for p in LIB_CANDIDATES if os.path.exists(p)), LIB_CANDIDATES[-1])
ds = xr.open_dataset(lib_path).sel(time=slice("1993-01-01", "2023-12-31"))
times = ds["time"].values
day = np.datetime_as_string(times, unit="D")
lib = ModelLibrary(ds, var="zos")
af = AnalogForecaster(lib, CorrelationDistance())       # obs-space selection (SWOT swath)
print(f"Library {lib_path}: {ds.sizes['time']} states, {day[0]} .. {day[-1]}")

truth_ds = xr.open_dataset(TRUTH_PATH)
truth_times = truth_ds["time"].values
truth_state = truth_ds["zos"].values
assert np.allclose(truth_ds["longitude"].values, af.lon) and \
    np.allclose(truth_ds["latitude"].values, af.lat), "truth grid != library grid"
tmin, tmax = truth_times.min(), truth_times.max()
print(f"Truth   {TRUTH_PATH}: {truth_ds.sizes['time']} steps, "
      f"{str(tmin)[:10]} .. {str(tmax)[:10]}")


def field_on(date64):
    """Nearest daily GLORYS-2024 surface field to a calendar date."""
    i = int(np.argmin(np.abs(truth_times - np.datetime64(date64))))
    return truth_state[i]


# %% Obs days = SWOT days with coverage AND a truth field at obs+LEAD.
swot_paths = sorted(glob.glob(os.path.join(SWOT_DIR, "*.nc")))
swot_days = sorted({re.search(r"_(\d{8})T", p).group(1) for p in swot_paths})
obs_days_all = [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in swot_days]

obs_days, results, demo = [], [], None
for T in obs_days_all:
    # Skip obs days that don't have a truth field at obs+LEAD.
    vday = np.datetime64(T) + np.timedelta64(LEAD, "D")
    if not (tmin <= vday <= tmax):
        continue                                          # no dense truth at obs+LEAD

    # Load the SWOT swath for this obs day, grid it, and skip if too few cells.
    obs_grid, mask = sa.swath_to_grid(swot_data.swaths_for_dates([T])[0],
                                      af.lon, af.lat)
    if mask.sum() < MIN_CELLS:
        continue                                          # too little SWOT coverage

    # Run the analog forecast and verify against the dense truth.
    r = sa.analog_vs_glorys(af, obs_grid, mask, field_on(vday), vday,
                            persist_surf=field_on(T), lead=LEAD, k=K, min_sep=MIN_SEP)
    obs_days.append(T)
    results.append(r)
    if T == DEMO_DAY:
        demo = (T, obs_grid, mask, r)
    print(f"obs {T} -> truth {str(vday)[:10]}: Loop Current front MHD (km) "
          f"analog best={np.nanmin(r['lc_mhd']):.0f} med={np.nanmedian(r['lc_mhd']):.0f} "
          f"ens={r['lc_mhd_ens']:.0f}  persist={r['lc_mhd_persist']:.0f}   "
          f"[ACC best={np.nanmax(r['acc']):+.2f}]")

print(f"\nUsed {len(obs_days)} obs days (K={K}, min_sep={MIN_SEP}, lead={LEAD}).")

# %% Illustration: per-analog forecast vs GLORYS truth for one obs day.
if demo is None and results:                              # fall back to the first day
    T0, r0 = obs_days[0], results[0]
    og0, mk0 = sa.swath_to_grid(swot_data.swaths_for_dates([T0])[0], af.lon, af.lat)
    demo = (T0, og0, mk0, r0)
if demo is not None:
    T0, og0, mk0, r0 = demo
    viz.plot_analog_glorys_maps(af, T0, og0, mk0, r0["sel"], day, r0["fc_anoms"],
                                r0["truth"], r0["acc"], LEAD, dist=r0["dist"], k_show=6,
                                fc_fronts=r0["fc_fronts"], truth_front=r0["truth_front"],
                                lc_mhd=r0["lc_mhd"], init_anom=r0["init"],
                                init_front=r0["persist_front"])

# %% Summary: Loop Current front skill (primary) + full-field ACC/RMSE (context).
if results:
    viz.plot_analog_glorys_lc_skill(obs_days, results, LEAD)   # Loop Current metric
    viz.plot_analog_glorys_skill(obs_days, results, LEAD)      # full-field ACC/RMSE

    lc_best = np.array([np.nanmin(r["lc_mhd"]) for r in results])
    lc_med = np.array([np.nanmedian(r["lc_mhd"]) for r in results])
    lc_ens = np.array([r["lc_mhd_ens"] for r in results])
    lc_per = np.array([r["lc_mhd_persist"] for r in results])
    print(f"\nMean over obs days — Loop Current front MHD (km) @ lead {LEAD} "
          f"(lower = better):")
    print(f"  best analog   = {np.nanmean(lc_best):5.1f}")
    print(f"  median analog = {np.nanmean(lc_med):5.1f}")
    print(f"  ensemble mean = {np.nanmean(lc_ens):5.1f}")
    print(f"  persistence   = {np.nanmean(lc_per):5.1f}")
    verdict = ("beats" if np.nanmean(lc_best) < np.nanmean(lc_per) else "loses to")
    print(f"VERDICT: best SWOT-selected analog {verdict} dense persistence on the "
          f"Loop Current front at {LEAD} days.")

plt.show()

# %%
