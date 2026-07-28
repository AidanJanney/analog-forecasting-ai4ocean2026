# %%
"""Analog forecasting of the Gulf of Mexico Loop Current — the one driver.

    python runs/forecast.py --config config/analog_forecast.yaml

Every run is the same loop: take a set of observation windows, rank the library
against each, roll the chosen analogs forward, score them. What used to be two
separate workflows is two settings of that loop:

    GLORYS -> GLORYS    1 window  x 31 leads   run.source: model
    SWOT   -> GLORYS   21 windows x  1 lead    run.source: swot

so there is one driver and one config schema. A single-day observation is a
one-day window, and a one-day window's sequence distance reduces exactly to that
day's distance — the collapse is real, not a special case bolted on.

Each stage is one call into a section of :mod:`analogfc`, and every choice below
is a config string:

    data      where the observation and the analog pool come from   [SOURCES]
    distance  how library days are ranked against the observation   [DISTANCES]
    forecast  how analogs are combined, scored and drawn            [COMBINERS, METRICS]

Runs as a script or cell by cell in Jupyter / VS Code.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analogfc import config as cfg
from analogfc.data import diagnostics, glorys, sources, swot, windows as ow
from analogfc.data.library import ModelLibrary
from analogfc.data.regions import Region
from analogfc.distance import DISTANCES, AnalogSelector
from analogfc.forecast import plots, rollout as fc, score as scoring
from analogfc.fronts import front_coverage

config = cfg.load("config/analog_forecast.yaml",
                  "Analog forecasting of the Gulf of Mexico Loop Current.")

SOURCE = config["run"]["source"]
ZARR_GLOB = cfg.resolve(config["data"]["zarr_glob"])
FIG_DIR = cfg.resolve(config["data"]["fig_dir"])

LIBRARY_DATES = slice(config["periods"]["library"]["start"],
                      config["periods"]["library"]["end"])
TARGET_DATES = slice(config["periods"]["target"]["start"],
                     config["periods"]["target"]["end"])

OBS = config["observation"]
K = config["analogs"]["k"]
BUFFER_DAYS = config["analogs"]["buffer_days"]
DISTANCE_NAME = config["analogs"]["distance"]
SSH_CONTOUR_LEVEL = config["analogs"]["ssh_contour_level"]
SELECTION_REGION = Region.from_config(config["analogs"].get("region"))

LEADS = config["forecast"]["leads"]
HEADLINE_LEAD = config["forecast"]["headline_lead"]
SCORE_TO = config["forecast"]["score_every_day_to"]
SCORE_METRIC_NAMES = config["forecast"]["score_metrics"]
COMBINER = config["forecast"].get("combiner", "ensemble_mean")
SCORING_REGION = Region.from_config(config["forecast"].get("region"))

FIGURES = config.get("figures", {})
K_SHOW = FIGURES.get("k_show", 6)
DEMO_DAY = FIGURES.get("demo_day")

os.makedirs(FIG_DIR, exist_ok=True)

# %% -- Section 1: getting data ---------------------------------------------- #
lon_slice, lat_slice = cfg.domain_slices(config)
VARIABLES = ("ssh",) if SOURCE == "swot" else ("sst", "ssh")
fields = glorys.open_glorys(
    ZARR_GLOB, lon_slice, lat_slice, variables=VARIABLES,
    group=config["climatology"]["group"],
    reduce_dims=tuple(config["climatology"]["reduce_dims"]))
stats = glorys.summarize(fields)

# %%
if FIGURES.get("diagnostics", False):
    diagnostics.run_all(fields["sst"], stats["sst"], FIG_DIR)

# %%
library = ModelLibrary(fields, period=LIBRARY_DATES, var="ssh")
truth = ModelLibrary(fields, period=TARGET_DATES, var="ssh")

distance = DISTANCES.create(DISTANCE_NAME)
if DISTANCE_NAME == "ssh_front_mhd":
    distance.level = SSH_CONTOUR_LEVEL

# The observation seam. `model` serves the library's own field in whatever
# representation the distance asked for; `swot` bins real swaths onto the grid.
if SOURCE == "swot":
    SWOT_DIR = cfg.resolve(config["data"]["swot_dir"])
    POINTS_DIR = cfg.resolve(config["data"]["swot_points_dir"])

    def swaths_for(dates):
        return swot.swaths_for_dates(dates, directory=SWOT_DIR,
                                     bbox=tuple(config["swot"]["bbox"]),
                                     cache_dir=POINTS_DIR)

    source = sources.SOURCES.create("swot", swaths_for=swaths_for,
                                    truth_library=truth,
                                    max_swaths_per_day=OBS.get("max_swaths_per_day"))
    observed = [d for d in swot.observed_dates(SWOT_DIR, POINTS_DIR)
                if TARGET_DATES.start <= d <= TARGET_DATES.stop]
    if not observed:
        raise SystemExit(
            f"No SWOT granules in {SWOT_DIR} for "
            f"{TARGET_DATES.start}..{TARGET_DATES.stop}.\nFetch them first:\n"
            f"  python -m analogfc.data.swot fetch-range "
            f"--start {TARGET_DATES.start} --end {TARGET_DATES.stop} "
            f"--collection {config['swot']['collection']}")
    print(f"SWOT: {len(observed)} observed days on disk, {observed[0]} .. {observed[-1]}")
else:
    source = sources.SOURCES.create("model", var=distance.var,
                                    representation=distance.representation,
                                    library=library)
    observed = [str(t)[:10] for t in truth.times]

# Which days to forecast. An explicit list is the single-target case; null takes
# every usable observed day in the target period.
if OBS["dates"]:
    ends = [np.datetime64(str(d)[:10], "D") for d in OBS["dates"]]
else:
    ends = ow.available_windows(observed, stride=OBS.get("stride", 1))

# %% -- Section 2: identifying analogs ---------------------------------------- #
selection_mask = None if SELECTION_REGION.is_whole else SELECTION_REGION.mask(library)
scoring_mask = None if SCORING_REGION.is_whole else SCORING_REGION.mask(library)
print(f"\nSelect on: {SELECTION_REGION.describe(library)}")
print(f"Score on:  {SCORING_REGION.describe(library)}")

selector = AnalogSelector(library, distance, region_mask=selection_mask)
metrics = scoring.build(SCORE_METRIC_NAMES, library, level=SSH_CONTOUR_LEVEL,
                        referenced=(SOURCE == "swot"), region_mask=scoring_mask)

# Coverage screening only means anything for a partial observation; a dense model
# field always sees the whole front.
COVER_MIN = OBS.get("front_cover_min") or 0.0
front_convention = next((m.front for m in metrics if m.name == "ssh_front_mhd"), None)

# %% -- Roll out one forecast per window -------------------------------------- #
# The demo window is scored every day out to the horizon, so its skill curves are
# continuous; the rest are scored at the headline lead only. With a single window
# the two coincide and the run reports the full by-lead table.
demo_index = ends.index(np.datetime64(DEMO_DAY, "D")) if DEMO_DAY in [str(e) for e in ends] else None
daily_leads = list(range(0, SCORE_TO + 1))

results, labels, selections, windows, covers = [], [], [], [], []
skipped = {"no_window": 0, "no_truth": 0, "front_unobserved": 0}

for end in ends:
    target_time = library.timestamp(end)
    if target_time is None or not library.covers(end + np.timedelta64(HEADLINE_LEAD, "D")):
        skipped["no_truth"] += 1
        continue

    window = ow.build_window(
        source, end, library, n_days=OBS.get("n_days", 1),
        min_cells_per_day=OBS.get("min_cells_per_day", 1),
        require_days=OBS.get("require_days", 1),
        weights=OBS.get("weights", "uniform"), halflife=OBS.get("halflife", 3.0))
    if window is None:
        skipped["no_window"] += 1
        continue

    cover = 1.0
    if COVER_MIN > 0 and front_convention is not None:
        present = source.truth_field(end, library)
        if present is not None:
            cover = front_coverage(front_convention.mask(present), window.coverage_mask)
        if cover < COVER_MIN:
            skipped["front_unobserved"] += 1
            continue

    analogs = selector.select_window(window, k=K, buffer_days=BUFFER_DAYS)
    if len(analogs) == 0:
        skipped["no_window"] += 1
        continue

    labels.append(str(end))
    selections.append(analogs)
    windows.append(window)
    covers.append(cover)
    results.append(None)                 # rolled out below, once the demo is known

if not results:
    raise SystemExit(
        f"No usable windows (skipped: {skipped['no_window']} without usable "
        f"observation coverage, {skipped['no_truth']} without truth at "
        f"+{HEADLINE_LEAD} d, {skipped['front_unobserved']} with the Loop Current "
        f"under-observed). Loosen observation.min_cells_per_day, "
        f"observation.front_cover_min, or widen observation.n_days.")

if demo_index is None or demo_index >= len(labels):
    demo_index = int(np.argmax(covers))          # best-observed window
single = len(labels) == 1

for i, (end, analogs) in enumerate(zip(labels, selections)):
    leads = daily_leads if i == demo_index else [HEADLINE_LEAD]
    target_time = library.timestamp(end)
    result = fc.rollout(library, analogs, target_time, leads,
                        variables=VARIABLES, combiner=COMBINER)
    # Persistence: the observed state at lead 0, carried forward unchanged. The
    # baseline every forecast has to beat, so it is scored on every window.
    persist = fc.score_baseline(
        result, metrics, fc.persistence_fields(library, target_time, VARIABLES))
    results[i] = (result, fc.score(result, metrics), persist)
    if not single:
        headline = metrics[-1]
        per_member, ens = results[i][1][headline.name]
        j = leads.index(HEADLINE_LEAD)
        members = [c[j] for c in per_member]
        print(f"window end {end} ({len(windows[i])}d, {windows[i].n_obs}ob"
              + (f", LC {covers[i]:.0%} observed" if COVER_MIN > 0 else "")
              + f") -> {headline.name} ({headline.unit}) "
              f"best={np.nanmin(members):.1f} med={np.nanmedian(members):.1f} "
              f"ens={ens[j]:.1f} persist={persist[headline.name][j]:.1f}", flush=True)

demo_result, demo_skill, demo_persist = results[demo_index]
demo_label, demo_analogs, demo_window = (labels[demo_index], selections[demo_index],
                                         windows[demo_index])

# %% -- Section 3: reporting and figures -------------------------------------- #
demo_analogs.report(demo_label)
fc.report_skill(demo_result, demo_skill, metrics, DISTANCE_NAME,
                len(demo_analogs), baseline=demo_persist)

if not single:
    print(f"\nUsed {len(labels)} windows of {len(ends)} candidates "
          f"(n_days={OBS.get('n_days', 1)}, K={K}, buffer_days={BUFFER_DAYS}, "
          f"headline lead={HEADLINE_LEAD}).")
    print(f"  skipped: {skipped['no_window']} without usable observation coverage, "
          f"{skipped['no_truth']} without truth at +{HEADLINE_LEAD} d, "
          f"{skipped['front_unobserved']} with the Loop Current under-observed.")
    print(f"\nMean over {len(labels)} windows @ lead {HEADLINE_LEAD}:")
    for m in metrics:
        best_fn = np.nanmax if m.higher_is_better else np.nanmin
        rows = []
        for result, skill, persist in results:
            j = result.leads.index(HEADLINE_LEAD)
            per_member, ens = skill[m.name]
            rows.append((best_fn([c[j] for c in per_member]), ens[j], persist[m.name][j]))
        best, ens, per = (np.nanmean([r[i] for r in rows]) for i in range(3))
        direction = "higher" if m.higher_is_better else "lower"
        print(f"  {m.name:>16s} ({m.unit}, {direction} = better): "
              f"best analog {best:8.3f} | ensemble {ens:8.3f} | persistence {per:8.3f}")

# %%
usable_leads = [L for L in LEADS if L in demo_result.leads]
for var in VARIABLES:
    plots.plot_analog_grid(
        var, library, demo_result, demo_skill, metrics, usable_leads, demo_label,
        DISTANCE_NAME, FIG_DIR, f"analog_grid_{var}_by_{DISTANCE_NAME}.png",
        contour_level=SSH_CONTOUR_LEVEL if var == "ssh" else None, max_rows=K_SHOW)

# %%
if SOURCE == "swot":
    plots.plot_observation_analogs(demo_window.composite, demo_window.coverage_mask,
                                   library, demo_analogs, FIG_DIR, k=K_SHOW)
else:
    plots.plot_target_vs_best(library, demo_analogs, library.timestamp(demo_label),
                              demo_label, FIG_DIR,
                              f"target_vs_best_analog_by_{DISTANCE_NAME}.png",
                              contour_level=SSH_CONTOUR_LEVEL)

# %%
if not single:
    per_window = []
    for result, skill, persist in results:
        j = result.leads.index(HEADLINE_LEAD)
        per_window.append({m.name: dict(members=[c[j] for c in skill[m.name][0]],
                                        ensemble=skill[m.name][1][j],
                                        persistence=persist[m.name][j])
                           for m in metrics})
    plots.plot_window_skill(labels, per_window, metrics, HEADLINE_LEAD, FIG_DIR)
    print(f"Figures written to {FIG_DIR}/")
