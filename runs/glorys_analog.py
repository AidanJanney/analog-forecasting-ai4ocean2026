# %%
"""Analog forecasting on the GLORYS Gulf of Mexico subset — GLORYS to GLORYS.

The observation is itself a GLORYS state, so selection sees the full field. The
library and target periods are disjoint, which is what keeps the selection
honest: the target day is not in the pool and neither are its neighbours.

Orchestration only. Each step is one call into a section of :mod:`analogfc`, and
every option below is a config string:

    data      open the record, build the fields          analogfc.data
    distance  rank library days against the target       analogfc.distance   [DISTANCES]
    forecast  roll out, score, plot                      analogfc.forecast   [METRICS]

Runs as a script or cell by cell in Jupyter / VS Code.

    python runs/glorys_analog.py --config config/analog_forecast.yaml
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analogfc import config as cfg
from analogfc.data import diagnostics, glorys
from analogfc.data.library import ModelLibrary
from analogfc.data.regions import Region
from analogfc.distance import DISTANCES, AnalogSelector
from analogfc.forecast import plots, rollout as fc, score as scoring

config = cfg.load("config/analog_forecast.yaml",
                  "Analog forecasting on the GLORYS Gulf of Mexico subset.")

ZARR_GLOB = cfg.resolve(config["data"]["zarr_glob"])
FIG_DIR = cfg.resolve(config["data"]["fig_dir"])

# Split the record into the analog library and the "observed" period we forecast for.
LIBRARY_DATES = slice(config["periods"]["library"]["start"],
                      config["periods"]["library"]["end"])
TARGET_DATE = config["analogs"]["target_date"]
SELECTION_METRIC = config["analogs"]["selection_metric"]
K = config["analogs"]["k"]
BUFFER_DAYS = config["analogs"]["buffer_days"]
SSH_CONTOUR_LEVEL = config["analogs"]["ssh_contour_level"]
FORECAST_LENGTH_DAYS = config["forecast"]["length_days"]
LEAD_DAYS = config["forecast"]["lead_days"]
SCORE_METRIC_NAMES = config["forecast"]["score_metrics"]
COMBINER = config["forecast"].get("combiner", "ensemble_mean")

# Selection and scoring each get their own area of the field. Omitted or null
# means the whole loaded domain, which is what every run did before these existed.
SELECTION_REGION = Region.from_config(config["analogs"].get("region"))
SCORING_REGION = Region.from_config(config["forecast"].get("region"))

if max(LEAD_DAYS) > FORECAST_LENGTH_DAYS:
    raise ValueError(
        f"lead_days reaches {max(LEAD_DAYS)} d but the forecast is only "
        f"{FORECAST_LENGTH_DAYS} d long")

os.makedirs(FIG_DIR, exist_ok=True)

# %% -- Section 1: getting data ---------------------------------------------- #
lon_slice, lat_slice = cfg.domain_slices(config)
fields = glorys.open_glorys(
    ZARR_GLOB, lon_slice, lat_slice,
    group=config["climatology"]["group"],
    reduce_dims=tuple(config["climatology"]["reduce_dims"]))
stats = glorys.summarize(fields)

# %%
diagnostics.run_all(fields["sst"], stats["sst"], FIG_DIR)

# %%
library = ModelLibrary(fields, period=LIBRARY_DATES, var="ssh")

# Timestamps carry a 12:00 time of day, so take the target stamp from the data
# rather than parsing TARGET_DATE, which would land on midnight and miss.
target_time = pd.Timestamp(fields["sst"].raw.sel(time=TARGET_DATE).time.values[0])

# %% -- Section 2: identifying analogs ---------------------------------------- #
selection_mask = None if SELECTION_REGION.is_whole else SELECTION_REGION.mask(library)
scoring_mask = None if SCORING_REGION.is_whole else SCORING_REGION.mask(library)
print(f"\nSelect on: {SELECTION_REGION.describe(library)}")
print(f"Score on:  {SCORING_REGION.describe(library)}")

distance = DISTANCES.create(SELECTION_METRIC)
if SELECTION_METRIC == "ssh_front_mhd":
    distance.level = SSH_CONTOUR_LEVEL

selector = AnalogSelector(library, distance, region_mask=selection_mask)
observation = library.at(distance.var, target_time, distance.representation).values
analogs = selector.select(observation, mask=None, k=K, buffer_days=BUFFER_DAYS)
analogs.report(TARGET_DATE)

# %% -- Section 3: rollout, scoring, plotting --------------------------------- #
# Errors are evaluated every day of the forecast, so the skill curves are
# continuous even though the map columns stay at the coarser LEAD_DAYS.
ERROR_LEADS = list(range(0, FORECAST_LENGTH_DAYS + 1))
metrics = scoring.build(SCORE_METRIC_NAMES, library, level=SSH_CONTOUR_LEVEL,
                        region_mask=scoring_mask)

result = fc.rollout(library, analogs, target_time, ERROR_LEADS, combiner=COMBINER)
skill = fc.score(result, metrics)
fc.report_skill(result, skill, metrics, SELECTION_METRIC, K)

# %%
for var in fields:
    plots.plot_analog_grid(
        var, library, result, skill, metrics, LEAD_DAYS, TARGET_DATE,
        SELECTION_METRIC, FIG_DIR, f"analog_grid_{var}_by_{SELECTION_METRIC}.png",
        contour_level=SSH_CONTOUR_LEVEL if var == "ssh" else None)

# %%
plots.plot_target_vs_best(
    library, analogs, target_time, TARGET_DATE, FIG_DIR,
    f"target_vs_best_analog_by_{SELECTION_METRIC}.png",
    contour_level=SSH_CONTOUR_LEVEL)
