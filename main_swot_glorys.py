# %% SWOT-to-GLORYS analog forecasting.
#
# Real SWOT KaRIn swaths from the 2023-2025 target period initialize the forecast;
# analogs are drawn from the disjoint 1993-2022 GLORYS library and verified against
# the dense GLORYS field at obs + LEAD.
#
# This is the SWOT-to-GLORYS workflow and is deliberately distinct from the
# GLORYS-to-GLORYS one in analog_forecast.py:
#
#                     GLORYS -> GLORYS            SWOT -> GLORYS (here)
#   observation       a library state             a real swath, partial coverage
#   obs space         full field                  only the observed cells
#   datum             same as library             ssha vs zos -> correlation distance
#   leakage guard     temporal exclusion window   none needed: disjoint eras
#   initialization    one day                     a window of consecutive days
#
# The forecast is initialized from an *observation window* (obs_window): a run of
# consecutive days, each holding however many swaths fell in the box, matched
# against the library as a sequence rather than as one flattened field. Set
# `observation.n_days: 1` and `max_swaths_per_day: 1` for the single-swath case.
#
# Three seams, each swappable without touching the others:
#   sources.ObsSource    where the observation comes from  (SwotSource here)
#   distances.ObsDistance how it is ranked against the library (correlation here)
#   analog.Combiner      how the chosen analogs become a forecast (ensemble mean)
#
# Adding another observation type means implementing ObsSource.observe for it;
# windowing, sequence matching, selection and everything below are unchanged.
#
# Prerequisites:
#   python swot_data.py fetch-range --start 2023-01-01 --end 2025-12-31
#   (needs a one-time Earthdata login; see swot_data's module docstring)
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml

import obs_window as ow
import sources
import swot_data
import swot_analog as sa
import viz
from analog import ModelLibrary, Forecast, AnalogSelector
from distances import CorrelationDistance, FrontMHDDistance
from fronts import front_coverage, front_mask, referenced

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "config", "swot_glorys.yaml")

# parse_known_args so the script still runs cell by cell in Jupyter/VS Code,
# where sys.argv carries the kernel's own arguments.
parser = argparse.ArgumentParser(description="SWOT-to-GLORYS analog forecasting.")
parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to the YAML config.")
args, _ = parser.parse_known_args()

with open(args.config) as fh:
    config = yaml.safe_load(fh)
print(f"Config: {args.config}")


def config_path(value):
    """Resolve a config path against this script, not the config or the cwd."""
    return value if os.path.isabs(value) else os.path.normpath(
        os.path.join(SCRIPT_DIR, value))


ZARR_GLOB = config_path(config["data"]["zarr_glob"])
SWOT_DIR = config_path(config["data"]["swot_dir"])
POINTS_DIR = config_path(config["data"]["swot_points_dir"])
FIG_DIR = config_path(config["data"]["fig_dir"])

LIBRARY = config["periods"]["library"]
TARGET = config["periods"]["target"]
OBS = config["observation"]
K = config["analogs"]["k"]
MIN_SEP = config["analogs"]["min_sep"]
DISTANCE_NAME = config["analogs"]["distance"]
LEAD = config["forecast"]["lead_days"]
MAP_LEADS = config["forecast"]["map_leads"]      # leads shown as map columns
CURVE_DAYS = config["forecast"]["curve_days"]    # skill curves evaluated daily to here
FRONT_COVER_MIN = config["verification"]["front_cover_min"]
LEVEL = config["verification"]["ssh_contour_level"]
K_SHOW = config["figures"]["k_show"]
DEMO_DAY = config["figures"]["demo_day"]

DISTANCES = {"correlation": CorrelationDistance, "front_mhd": FrontMHDDistance}
if DISTANCE_NAME not in DISTANCES:
    raise ValueError(f"unknown distance {DISTANCE_NAME!r}; choose from {sorted(DISTANCES)}")

os.makedirs(FIG_DIR, exist_ok=True)

# %% Load GLORYS once, then split it into the analog pool and the dense truth.
# Both come from the same store, so they share a grid by construction and the
# forecast/truth comparison needs no regridding.
ds = xr.open_mfdataset(ZARR_GLOB, engine="zarr", combine="by_coords")
ds = ds[["zos"]].sel(
    longitude=slice(config["domain"]["min_longitude"], config["domain"]["max_longitude"]),
    latitude=slice(config["domain"]["min_latitude"], config["domain"]["max_latitude"]),
)
print(f"GLORYS {ZARR_GLOB}: {ds.sizes['time']} days, "
      f"{ds.sizes['latitude']} x {ds.sizes['longitude']} grid")

lib_ds = ds.sel(time=slice(LIBRARY["start"], LIBRARY["end"])).load()
truth_ds = ds.sel(time=slice(TARGET["start"], TARGET["end"])).load()

lib = ModelLibrary(lib_ds, var="zos")
truth = ModelLibrary(truth_ds, var="zos")
day = np.datetime_as_string(lib.times, unit="D")
print(f"  library : {lib.n} states, {day[0]} .. {day[-1]}  (analog pool)")
print(f"  truth   : {truth.n} states, "
      f"{np.datetime_as_string(truth.times[0], unit='D')} .. "
      f"{np.datetime_as_string(truth.times[-1], unit='D')}  (dense verification)")

# The library is matched as a *sequence*, so its days must be contiguous: index
# arithmetic (j - offset) is only day arithmetic if there are no gaps.
gaps = np.unique(np.diff(lib.times))
assert len(gaps) == 1, f"library time axis is not contiguous daily: {gaps}"

# %% Observation windows over the SWOT era.
swot_days = [d for d in swot_data.observed_dates(SWOT_DIR, POINTS_DIR)
             if TARGET["start"] <= d <= TARGET["end"]]
if not swot_days:
    raise SystemExit(
        f"No SWOT granules in {SWOT_DIR} for {TARGET['start']}..{TARGET['end']}.\n"
        f"Fetch them first:\n"
        f"  python swot_data.py fetch-range --start {TARGET['start']} "
        f"--end {TARGET['end']} --collection {config['swot']['collection']}")
print(f"SWOT: {len(swot_days)} observed days on disk, "
      f"{swot_days[0]} .. {swot_days[-1]}")


def swaths_for(dates):
    """This run's swath source: cached per-day points, falling back to granules."""
    return swot_data.swaths_for_dates(dates, directory=SWOT_DIR,
                                      bbox=tuple(config["swot"]["bbox"]),
                                      cache_dir=POINTS_DIR)


# The observation seam: SwotSource answers "what did SWOT see on date X, on the
# model grid". Swap it for another ObsSource to drive the same pipeline with a
# different instrument — nothing below this line is SWOT-specific.
source = sources.SwotSource(swaths_for, truth_library=truth,
                            max_swaths_per_day=OBS["max_swaths_per_day"])
selector = AnalogSelector(lib, DISTANCES[DISTANCE_NAME]())
forecaster = Forecast(lib)              # distance-weighted ensemble mean

# Loop Current fronts share a common datum (the library ocean mean) so the fixed
# 0.17 m contour tracks the front's position, not the basin-scale sea-level offset.
ref_mean = float(np.nanmean(lib.mean_surf[lib.ocean]))

# %% Roll out one forecast per usable window.
ends = ow.available_windows(swot_days, stride=OBS["stride"])
windows, obs_days, results, selections, covers = [], [], [], [], []
skipped = {"no_window": 0, "no_truth": 0, "front_unobserved": []}

for end in ends:
    vday = end + np.timedelta64(int(LEAD), "D")
    g_end, g_valid = truth.field_on(end), truth.field_on(vday)
    if g_end is None or g_valid is None:
        skipped["no_truth"] += 1               # no dense truth at the window end or +LEAD
        continue

    window = ow.build_window(
        source, end, lib,
        n_days=OBS["n_days"], min_cells_per_day=OBS["min_cells_per_day"],
        require_days=OBS["require_days"],
        weights=OBS["weights"], halflife=OBS["halflife"])
    if window is None:
        skipped["no_window"] += 1              # too little usable SWOT coverage
        continue

    # Only score days on which the window actually samples the Loop Current:
    # locate the front in the dense field at the window end and require the
    # window's union coverage to observe enough of it.
    front = front_mask(g_end, level=LEVEL, ocean=lib.ocean, ref_mean=ref_mean)
    cover = front_coverage(front, window.coverage_mask)
    if cover < FRONT_COVER_MIN:
        skipped["front_unobserved"].append((str(end), cover))
        continue

    # 1. IDENTIFY analogs from the window, 2. FORECAST them, 3. SCORE vs dense truth.
    selection = selector.select(window, lead=LEAD, k=K, min_sep=MIN_SEP)
    ens = forecaster.forecast(selection, LEAD)
    r = sa.evaluate(lib, selection, ens, g_valid, vday, persist_surf=g_end,
                    lead=LEAD, lc_level=LEVEL)

    windows.append(window)
    obs_days.append(str(end))
    results.append(r)
    selections.append(selection)
    covers.append(cover)
    print(f"window end {str(end)} ({len(window)}d, {window.n_obs}sw, "
          f"LC {cover:.0%} observed) -> truth {str(vday)[:10]}: "
          f"front MHD (km) best={np.nanmin(r['lc_mhd']):.0f} "
          f"med={np.nanmedian(r['lc_mhd']):.0f} ens={r['lc_mhd_ens']:.0f} "
          f"persist={r['lc_mhd_persist']:.0f}  [ACC best={np.nanmax(r['acc']):+.2f}]",
          flush=True)

print(f"\nUsed {len(obs_days)} windows "
      f"(n_days={OBS['n_days']}, max_swaths/day={OBS['max_swaths_per_day']}, "
      f"K={K}, min_sep={MIN_SEP}, lead={LEAD}).")
print(f"  skipped: {skipped['no_window']} without usable SWOT coverage, "
      f"{skipped['no_truth']} without dense truth at +{LEAD} d, "
      f"{len(skipped['front_unobserved'])} with the Loop Current under-observed "
      f"(< {FRONT_COVER_MIN:.0%}).")

if not results:
    raise SystemExit("No usable windows — loosen observation.min_cells_per_day or "
                     "verification.front_cover_min, or widen observation.n_days.")

# %% Illustration for one window: which analogs it identified, and what they forecast.
di = obs_days.index(DEMO_DAY) if DEMO_DAY in obs_days else int(np.argmax(covers))
w0, T0, r0, sel0 = windows[di], obs_days[di], results[di], selections[di]
og0, mk0 = w0.composite, w0.coverage_mask
print(f"\nDemo window: {w0.describe()}  (Loop Current {covers[di]:.0%} observed)")
print(f"  analogs: " + ", ".join(
    f"{d}({x:.2f})" for d, x in zip(ow.selection_dates(lib, sel0), sel0.distances)))

viz.plot_swot_analogs(og0, mk0, lib, sel0.indices, sel0.distances, day, k=K_SHOW,
                      outdir=FIG_DIR)

# %% The analog grid: one row per member (state, then misfit), leads as columns,
# ensemble and truth as the final rows, daily skill curves down the right.
end0 = np.datetime64(T0)
g_end0 = truth.field_on(end0)
k_grid = min(K_SHOW, len(sel0), len(viz.ANALOG_COLORS))
idx0 = sel0.indices[:k_grid]
sel_max = int(sel0.indices.max())            # ensemble needs every member's future


def ref(field):
    """Absolute SSH on the datum the fixed contour assumes (see fronts.referenced)."""
    return referenced(field, ocean=lib.ocean, ref_mean=ref_mean)


def usable(lead):
    """A lead is plottable only with dense truth and a future for every analog."""
    return (sel_max + int(lead) < lib.n
            and truth.field_on(end0 + np.timedelta64(int(lead), "D")) is not None)


map_leads = [int(L) for L in MAP_LEADS if usable(L)]
curve_leads = [L for L in range(0, CURVE_DAYS + 1) if usable(L)]
if not map_leads:
    raise SystemExit(f"None of forecast.map_leads={MAP_LEADS} has dense truth at {T0}.")

truth_states = [ref(truth.field_on(end0 + np.timedelta64(L, "D"))) for L in map_leads]
states = [[ref(lib.state[i + L]) for L in map_leads] for i in idx0]
states.append([ref(forecaster.forecast(sel0, L)) for L in map_leads])   # ensemble row
states.append(truth_states)                                            # truth row
init_states = ([ref(lib.state[i]) for i in idx0]
               + [ref(forecaster.forecast(sel0, 0)), ref(g_end0)])

sel_dates = ow.selection_dates(lib, sel0)
row_labels = ([f"analog {r + 1}\n{sel_dates[r]}" for r in range(k_grid)]
              + [f"ensemble\nK = {len(sel0)}", f"GLORYS truth\n{T0}"])

# Score every lead the same way the headline number is scored, so the curves and
# the printed summary cannot drift apart.
shape = (k_grid, len(curve_leads))
acc_m, rmse_m, mhd_m = (np.full(shape, np.nan) for _ in range(3))
acc_e, rmse_e, mhd_e = (np.full(len(curve_leads), np.nan) for _ in range(3))
acc_p, rmse_p, mhd_p = (np.full(len(curve_leads), np.nan) for _ in range(3))
for li, L in enumerate(curve_leads):
    vd = end0 + np.timedelta64(L, "D")
    rL = sa.evaluate(lib, sel0, forecaster.forecast(sel0, L), truth.field_on(vd), vd,
                     persist_surf=g_end0, lead=L, lc_level=LEVEL)
    acc_m[:, li], rmse_m[:, li], mhd_m[:, li] = (rL["acc"][:k_grid],
                                                 rL["rmse"][:k_grid],
                                                 rL["lc_mhd"][:k_grid])
    acc_e[li], rmse_e[li], mhd_e[li] = rL["acc_ens"], rL["rmse_ens"], rL["lc_mhd_ens"]
    acc_p[li], rmse_p[li], mhd_p[li] = (rL["acc_persist"], rL["rmse_persist"],
                                        rL["lc_mhd_persist"])

curves = {
    "ACC": dict(members=acc_m, ens=acc_e, persist=acc_p, unit="correlation"),
    "RMSE": dict(members=rmse_m, ens=rmse_e, persist=rmse_p, unit="m"),
    "front MHD": dict(members=mhd_m, ens=mhd_e, persist=mhd_p, unit="km"),
}
viz.plot_analog_grid(og0, mk0, init_states, states, truth_states, map_leads,
                     row_labels, curves, curve_leads, level=LEVEL, obs_day=T0,
                     n_analogs=k_grid, fname="analog_grid.png", outdir=FIG_DIR)

# %% Summary: Loop Current front skill (primary) + full-field ACC/RMSE (context).
viz.plot_analog_glorys_lc_skill(obs_days, results, LEAD, outdir=FIG_DIR)
viz.plot_analog_glorys_skill(obs_days, results, LEAD, outdir=FIG_DIR)

lc_best = np.array([np.nanmin(r["lc_mhd"]) for r in results])
lc_med = np.array([np.nanmedian(r["lc_mhd"]) for r in results])
lc_ens = np.array([r["lc_mhd_ens"] for r in results])
lc_per = np.array([r["lc_mhd_persist"] for r in results])
acc_ens = np.array([r["acc_ens"] for r in results])
acc_per = np.array([r["acc_persist"] for r in results])

print(f"\nMean over {len(obs_days)} windows — Loop Current front MHD (km) "
      f"@ lead {LEAD} (lower = better):")
print(f"  best analog   = {np.nanmean(lc_best):5.1f}")
print(f"  median analog = {np.nanmean(lc_med):5.1f}")
print(f"  ensemble mean = {np.nanmean(lc_ens):5.1f}")
print(f"  persistence   = {np.nanmean(lc_per):5.1f}")
print(f"Full-field ACC @ lead {LEAD} (higher = better): "
      f"ensemble {np.nanmean(acc_ens):+.2f}, persistence {np.nanmean(acc_per):+.2f}")
verdict = "beats" if np.nanmean(lc_best) < np.nanmean(lc_per) else "loses to"
print(f"VERDICT: the best SWOT-selected analog {verdict} dense persistence on the "
      f"Loop Current front at {LEAD} days.")
print(f"Figures written to {FIG_DIR}/")

plt.show()

# %%
