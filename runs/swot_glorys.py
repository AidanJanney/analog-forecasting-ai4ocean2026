# %%
"""SWOT-to-GLORYS analog forecasting.

Real SWOT KaRIn swaths from 2023-2025 initialize the forecast; analogs come from
the disjoint pre-2023 GLORYS library and are verified against the dense GLORYS
field at obs + lead.

The same three sections as runs/glorys_analog.py, with one option swapped in
each — which is the point of the structure:

    data      observation is a partial swath, not a model state   SwotSource
    distance  ranked on pattern correlation over observed cells   correlation
    forecast  identical: rollout, score, plot                     unchanged

Two things differ from the GLORYS-to-GLORYS run and both are handled by the
choice of options, not by separate code. Coverage is partial, so the distance
must work on a mask. The datums differ (``ssha`` against ``zos``), so the
distance must centre both sides — which the correlation distance does and the
front-geometry distance cannot, since it needs an absolute SSH field.

Prerequisites:
    python -m analogfc.data.swot fetch-range --start 2023-01-01 --end 2025-12-31
    (needs a one-time Earthdata login; see analogfc/data/swot.py)

    python runs/swot_glorys.py --config config/swot_glorys.yaml
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analogfc import config as cfg
from analogfc.data import glorys, sources, swot, windows as ow
from analogfc.data.library import ModelLibrary
from analogfc.data.regions import Region
from analogfc.distance import DISTANCES, AnalogSelector
from analogfc.forecast import plots, rollout as fc, score as scoring
from analogfc.fronts import front_coverage

config = cfg.load("config/swot_glorys.yaml", "SWOT-to-GLORYS analog forecasting.")

ZARR_GLOB = cfg.resolve(config["data"]["zarr_glob"])
SWOT_DIR = cfg.resolve(config["data"]["swot_dir"])
POINTS_DIR = cfg.resolve(config["data"]["swot_points_dir"])
FIG_DIR = cfg.resolve(config["data"]["fig_dir"])

LIBRARY = config["periods"]["library"]
TARGET = config["periods"]["target"]
OBS = config["observation"]
K = config["analogs"]["k"]
MIN_SEP = config["analogs"]["min_sep"]
DISTANCE_NAME = config["analogs"]["distance"]
LEAD = config["forecast"]["lead_days"]
MAP_LEADS = config["forecast"]["map_leads"]
CURVE_DAYS = config["forecast"]["curve_days"]
SCORE_METRIC_NAMES = config["forecast"].get(
    "score_metrics", ["ssh_rmse", "ssh_acc", "ssh_front_mhd"])
COMBINER = config["forecast"].get("combiner", "ensemble_mean")
SELECTION_REGION = Region.from_config(config["analogs"].get("region"))
SCORING_REGION = Region.from_config(config["forecast"].get("region"))
FRONT_COVER_MIN = config["verification"]["front_cover_min"]
LEVEL = config["verification"]["ssh_contour_level"]
K_SHOW = config["figures"]["k_show"]
DEMO_DAY = config["figures"]["demo_day"]

os.makedirs(FIG_DIR, exist_ok=True)

# %% -- Section 1: getting data ---------------------------------------------- #
# Library and truth come from the same store, so they share a grid by
# construction and the forecast/truth comparison needs no regridding.
lon_slice, lat_slice = cfg.domain_slices(config)
fields = glorys.open_glorys(ZARR_GLOB, lon_slice, lat_slice, variables=("ssh",),
                            group=config.get("climatology", {}).get("group", "climday"),
                            reduce_dims=tuple(
                                config.get("climatology", {}).get("reduce_dims", [])))

library = ModelLibrary(fields, period=slice(LIBRARY["start"], LIBRARY["end"]), var="ssh")
truth = ModelLibrary(fields, period=slice(TARGET["start"], TARGET["end"]), var="ssh")
print(f"  library : {library.n} states, "
      f"{np.datetime_as_string(library.times[0], unit='D')} .. "
      f"{np.datetime_as_string(library.times[-1], unit='D')}  (analog pool)")
print(f"  truth   : {truth.n} states, "
      f"{np.datetime_as_string(truth.times[0], unit='D')} .. "
      f"{np.datetime_as_string(truth.times[-1], unit='D')}  (dense verification)")

# The library is matched as a *sequence*, so its days must be contiguous: index
# arithmetic is only day arithmetic if there are no gaps.
gaps = np.unique(np.diff(library.times))
assert len(gaps) == 1, f"library time axis is not contiguous daily: {gaps}"

swot_days = [d for d in swot.observed_dates(SWOT_DIR, POINTS_DIR)
             if TARGET["start"] <= d <= TARGET["end"]]
if not swot_days:
    raise SystemExit(
        f"No SWOT granules in {SWOT_DIR} for {TARGET['start']}..{TARGET['end']}.\n"
        f"Fetch them first:\n"
        f"  python -m analogfc.data.swot fetch-range --start {TARGET['start']} "
        f"--end {TARGET['end']} --collection {config['swot']['collection']}")
print(f"SWOT: {len(swot_days)} observed days on disk, {swot_days[0]} .. {swot_days[-1]}")


def swaths_for(dates):
    """This run's swath source: cached per-day points, falling back to granules."""
    return swot.swaths_for_dates(dates, directory=SWOT_DIR,
                                 bbox=tuple(config["swot"]["bbox"]),
                                 cache_dir=POINTS_DIR)


source = sources.SOURCES.create("swot", swaths_for=swaths_for, truth_library=truth,
                                max_swaths_per_day=OBS["max_swaths_per_day"])

# %% -- Section 2: identifying analogs ---------------------------------------- #
selection_mask = None if SELECTION_REGION.is_whole else SELECTION_REGION.mask(library)
scoring_mask = None if SCORING_REGION.is_whole else SCORING_REGION.mask(library)
print(f"Select on: {SELECTION_REGION.describe(library)}")
print(f"Score on:  {SCORING_REGION.describe(library)}")

distance = DISTANCES.create(DISTANCE_NAME)
if DISTANCE_NAME == "ssh_front_mhd":
    distance.level = LEVEL
selector = AnalogSelector(library, distance, region_mask=selection_mask)

# %% Roll out one forecast per usable window.
metrics = scoring.build(SCORE_METRIC_NAMES, library, level=LEVEL, referenced=True,
                        region_mask=scoring_mask)

# Fronts are referenced to a common datum across the two eras, so the fixed
# contour tracks the front's position and not the basin-scale sea-level offset.
# The coverage check below must use the same convention the metric does.
front_convention = next(m for m in metrics if m.name == "ssh_front_mhd").front
primary = metrics[-1]

ends = ow.available_windows(swot_days, stride=OBS["stride"])
labels, per_window, selections, windows, covers = [], [], [], [], []
skipped = {"no_window": 0, "no_truth": 0, "front_unobserved": []}

for end in ends:
    vday = end + np.timedelta64(int(LEAD), "D")
    present = source.truth_field(end, library)
    if present is None or source.truth_field(vday, library) is None:
        skipped["no_truth"] += 1               # no dense truth at the window end or +LEAD
        continue

    window = ow.build_window(
        source, end, library,
        n_days=OBS["n_days"], min_cells_per_day=OBS["min_cells_per_day"],
        require_days=OBS["require_days"],
        weights=OBS["weights"], halflife=OBS["halflife"])
    if window is None:
        skipped["no_window"] += 1              # too little usable SWOT coverage
        continue

    # Only score days on which the window actually samples the Loop Current:
    # locate the front in the dense field at the window end and require the
    # window's union coverage to observe enough of it.
    front = front_convention.mask(present)
    cover = front_coverage(front, window.coverage_mask)
    if cover < FRONT_COVER_MIN:
        skipped["front_unobserved"].append((str(end), cover))
        continue

    # 1. IDENTIFY analogs from the window, 2. FORECAST them, 3. SCORE vs dense truth.
    selection = selector.select_window(window, k=K, buffer_days=MIN_SEP)
    result = fc.rollout(library, selection, library.timestamp(end), [LEAD],
                        variables=("ssh",), combiner=COMBINER)
    skill = fc.score(result, metrics)

    scores = {}
    for m in metrics:
        per_member, ens_curve = skill[m.name]
        persistence = m({"ssh": result.truth[LEAD]["ssh"].copy(data=present)},
                        result.truth[LEAD], result.valid_times[LEAD])
        scores[m.name] = dict(members=[c[0] for c in per_member],
                              ensemble=ens_curve[0], persistence=persistence)

    labels.append(str(end))
    per_window.append(scores)
    selections.append(selection)
    windows.append(window)
    covers.append(cover)
    p = scores[primary.name]
    print(f"window end {str(end)} ({len(window)}d, {window.n_obs}sw, "
          f"LC {cover:.0%} observed) -> truth {str(vday)[:10]}: "
          f"{primary.name} ({primary.unit}) best={np.nanmin(p['members']):.1f} "
          f"med={np.nanmedian(p['members']):.1f} ens={p['ensemble']:.1f} "
          f"persist={p['persistence']:.1f}", flush=True)

print(f"\nUsed {len(labels)} windows "
      f"(n_days={OBS['n_days']}, max_swaths/day={OBS['max_swaths_per_day']}, "
      f"K={K}, min_sep={MIN_SEP}, lead={LEAD}).")
print(f"  skipped: {skipped['no_window']} without usable SWOT coverage, "
      f"{skipped['no_truth']} without dense truth at +{LEAD} d, "
      f"{len(skipped['front_unobserved'])} with the Loop Current under-observed "
      f"(< {FRONT_COVER_MIN:.0%}).")

if not per_window:
    raise SystemExit("No usable windows — loosen observation.min_cells_per_day or "
                     "verification.front_cover_min, or widen observation.n_days.")

# %% -- Section 3: figures ---------------------------------------------------- #
di = labels.index(DEMO_DAY) if DEMO_DAY in labels else int(np.argmax(covers))
w0, T0, sel0 = windows[di], labels[di], selections[di]
print(f"\nDemo window: {w0.describe()}  (Loop Current {covers[di]:.0%} observed)")
print("  analogs: " + ", ".join(f"{d}({x:.2f})"
                                for d, x in zip(sel0.dates, sel0.distances)))

plots.plot_observation_analogs(w0.composite, w0.coverage_mask, library, sel0,
                               FIG_DIR, k=K_SHOW)

# The analog grid, evaluated daily so the curves are continuous even though the
# map columns stay at the coarser map_leads.
end0 = np.datetime64(T0)
curve_leads = [L for L in range(0, CURVE_DAYS + 1)
               if library.covers(end0 + np.timedelta64(L, "D"))]
map_leads = [int(L) for L in MAP_LEADS if L in curve_leads]
if not map_leads:
    raise SystemExit(f"None of forecast.map_leads={MAP_LEADS} has dense truth at {T0}.")

demo = fc.rollout(library, sel0, library.timestamp(end0), curve_leads,
                  variables=("ssh",), combiner=COMBINER)
demo_skill = fc.score(demo, metrics)
plots.plot_analog_grid("ssh", library, demo, demo_skill, metrics, map_leads, T0,
                       DISTANCE_NAME, FIG_DIR, "analog_grid.png",
                       contour_level=LEVEL)

plots.plot_window_skill(labels, per_window, metrics, LEAD, FIG_DIR,
                        "swot_window_skill.png")

# %% Summary.
print(f"\nMean over {len(labels)} windows @ lead {LEAD}:")
for m in metrics:
    best_fn = np.nanmax if m.higher_is_better else np.nanmin
    best = np.nanmean([best_fn(w[m.name]["members"]) for w in per_window])
    ens = np.nanmean([w[m.name]["ensemble"] for w in per_window])
    per = np.nanmean([w[m.name]["persistence"] for w in per_window])
    direction = "higher" if m.higher_is_better else "lower"
    print(f"  {m.name:>16s} ({m.unit}, {direction} = better): "
          f"best analog {best:7.3f} | ensemble {ens:7.3f} | persistence {per:7.3f}")
print(f"Figures written to {FIG_DIR}/")
