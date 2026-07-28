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
from analog import ModelLibrary, AnalogSelector, Forecast
from distances import CorrelationDistance, LatentDistance
from fronts import LEVEL, loop_current_front

# %% ---- CONFIG ----------------------------------------------------------- #
LIB_CANDIDATES = [
    # "data/glorys_gom_zos_2004_2013.nc",    # 10-year zos library (analog pool)
    # "data/glorys_gom_zos_2004.nc",         # single-year fallback
    "data/glorys_gom_zos_thetao_all.nc"
]
TRUTH_PATH = "data/glorys_gom_zos_2024.nc"      # dense GLORYS truth over the SWOT era
                                                # (2024-01..07; falls back to 2024jan)
SWOT_DIR = "data/swot"
K = 10                                     # de-clustered analogs kept per obs day
K_SHOW = 6                                 # analogs shown in the illustration figures
MIN_SEP = 14                               # min days between selected analogs
LEAD = 14                                  # forecast lead time (days)
MIN_CELLS = 500                            # min observed SWOT cells to use an obs day
FRONT_COVER_MIN = 0.20                     # min fraction of the Loop Current front the
                                           # swath must sample (~one clean crossing; a
                                           # ~120 km swath spans ~15-20% of the front)
DEMO_DAY = None                            # obs day shown in the per-analog figures;
                                           # None = auto-pick the best LC-observed day
# -------------------------------------------------------------------------- #

# %% Load the GLORYS library (analog pool) and the dense 2024 truth.
lib_path = next((p for p in LIB_CANDIDATES if os.path.exists(p)), LIB_CANDIDATES[-1])
ds = xr.open_dataset(lib_path).sel(
    time=slice("1993-01-01", "2023-12-31") # Date range of the GLORYS library (analog pool)
)
times = ds["time"].values
day = np.datetime_as_string(times, unit="D")
lib = ModelLibrary(ds, var="zos")
selector = AnalogSelector(lib,
    CorrelationDistance()       # obs-space correlation over the SWOT swath
    # LatentDistance(),         # distance in a latent space (e.g., from an autoencoder)
)
forecaster = Forecast(lib)      # ensemble mean of the selected analogs
print(f"Library {lib_path}: {ds.sizes['time']} states, {day[0]} .. {day[-1]}")

truth_ds = xr.open_dataset(TRUTH_PATH)
truth_times = truth_ds["time"].values
truth_state = truth_ds["zos"].values
assert np.allclose(truth_ds["longitude"].values, lib.lon) and \
    np.allclose(truth_ds["latitude"].values, lib.lat), "truth grid != library grid"
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

# Loop Current fronts share a common datum (the library ocean mean) so the fixed
# 0.17-m contour tracks the front's position, not the basin-scale sea-level offset.
ref_mean = float(np.nanmean(lib.mean_surf[lib.ocean]))

obs_days, results, selections, covers, skipped = [], [], [], [], []
for T in obs_days_all:
    # Skip obs days that don't have a truth field at obs+LEAD.
    vday = np.datetime64(T) + np.timedelta64(LEAD, "D")
    if not (tmin <= vday <= tmax):
        continue                                          # no dense truth at obs+LEAD

    # Load the SWOT swath for this obs day, grid it, and skip if too few cells.
    obs_grid, mask = sa.swath_to_grid(swot_data.swaths_for_dates([T])[0],
                                      lib.lon, lib.lat)
    if mask.sum() < MIN_CELLS:
        continue                                          # too little SWOT coverage

    # Pick only days on which the swath actually samples the Loop Current: locate the
    # front in the dense GLORYS field at T and require the swath to observe enough of it.
    gT = field_on(T)
    lc_front = loop_current_front(gT, lib.lon, lib.lat, lib.ocean, ref_mean, level=LEVEL)
    cover = sa.front_coverage(lc_front, mask, lib.lon, lib.lat)
    if cover < FRONT_COVER_MIN:
        skipped.append((T, cover))
        continue                                          # Loop Current not observed

    # 1. IDENTIFY analogs (selection), 2. FORECAST them, 3. SCORE vs dense truth
    selection = selector.select(obs_grid, mask, LEAD, K, min_sep=MIN_SEP)
    ens = forecaster.forecast(selection, LEAD)
    r = sa.evaluate(lib, selection, ens, field_on(vday), vday,
                    persist_surf=gT, lead=LEAD)
    obs_days.append(T)
    results.append(r)
    selections.append(selection)
    covers.append(cover)
    print(f"obs {T} (LC {cover:.0%} observed) -> truth {str(vday)[:10]}: "
          f"Loop Current front MHD (km) "
          f"analog best={np.nanmin(r['lc_mhd']):.0f} med={np.nanmedian(r['lc_mhd']):.0f} "
          f"ens={r['lc_mhd_ens']:.0f}  persist={r['lc_mhd_persist']:.0f}   "
          f"[ACC best={np.nanmax(r['acc']):+.2f}]")

if skipped:
    print(f"\nSkipped (Loop Current under-observed, < {FRONT_COVER_MIN:.0%}): "
          + ", ".join(f"{t} {c:.0%}" for t, c in skipped))
print(f"\nUsed {len(obs_days)} obs days with the Loop Current observed "
      f"(K={K}, min_sep={MIN_SEP}, lead={LEAD}).")

# %% Illustration for one obs day: (a) which analogs were identified, then
#    (b) the forecast those analogs produce vs the dense GLORYS truth. Use DEMO_DAY
#    if it is among the LC-observed days, else the day that best observes the front.
if results:
    di = obs_days.index(DEMO_DAY) if DEMO_DAY in obs_days else int(np.argmax(covers))
    T0, r0, sel0 = obs_days[di], results[di], selections[di]
    og0, mk0 = sa.swath_to_grid(swot_data.swaths_for_dates([T0])[0], lib.lon, lib.lat)
    print(f"\nDemo day: {T0} (Loop Current {covers[di]:.0%} observed)")
    # (a) The identified analogs: the SWOT swath beside its top-K GLORYS matches.
    viz.plot_swot_analogs(og0, mk0, lib, sel0.indices, sel0.distances, day, k=K_SHOW)
    # (b) The forecast from those analogs (selection reused, not recomputed).
    viz.plot_analog_glorys_maps(lib, T0, og0, mk0, r0["sel"], day, r0["fc_anoms"],
                                r0["truth"], r0["acc"], LEAD, dist=r0["dist"], k_show=K_SHOW,
                                fc_fronts=r0["fc_fronts"], truth_front=r0["truth_front"],
                                lc_mhd=r0["lc_mhd"], init_anom=r0["init"],
                                init_front=r0["persist_front"], ens_anom=r0["ens_anom"],
                                ens_front=r0["ens_front"], acc_ens=r0["acc_ens"],
                                lc_mhd_ens=r0["lc_mhd_ens"], weights=r0["weights"])

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
