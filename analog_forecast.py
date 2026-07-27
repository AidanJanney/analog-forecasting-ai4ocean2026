# %%
"""Analog forecasting on the GLORYS Gulf of Mexico subset.

Migrated out of the second half of download_glorys.ipynb, so that notebook now
only downloads/rechunks GLORYS and this script does the analog work.

Expects the per-year Zarr stores written by ../subset_glorys.py.
Run in the `data-access-ai4ocean2026` conda env.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy.ndimage import distance_transform_edt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "config", "analog_forecast.yaml")

# parse_known_args so the script still runs cell by cell in Jupyter/VS Code,
# where sys.argv carries the kernel's own arguments.
parser = argparse.ArgumentParser(description="Analog forecasting on the GLORYS Gulf of Mexico subset.")
parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to the YAML config file.")
args, _ = parser.parse_known_args()

with open(args.config) as fh:
    config = yaml.safe_load(fh)
print(f"Config: {args.config}")

# Paths in the config are relative to this script, not to the config file or the
# working directory, so a config can live anywhere and still resolve the same way.
def config_path(value):
    return value if os.path.isabs(value) else os.path.normpath(os.path.join(SCRIPT_DIR, value))


ZARR_GLOB = config_path(config["data"]["zarr_glob"])
FIG_DIR = config_path(config["data"]["fig_dir"])

# Split the record into the analog library and the "observed" period we forecast for.
LIBRARY_DATES = slice(config["periods"]["library"]["start"], config["periods"]["library"]["end"])
TARGET_DATES = slice(config["periods"]["target"]["start"], config["periods"]["target"]["end"])

TARGET_DATE = config["analogs"]["target_date"]
SELECTION_METRIC = config["analogs"]["selection_metric"]  # how library days are ranked
K = config["analogs"]["k"]  # number of analogs to keep
BUFFER_DAYS = config["analogs"]["buffer_days"]  # minimum separation between retained analogs
SSH_CONTOUR_LEVEL = config["analogs"]["ssh_contour_level"]  # metres, defines the Loop Current front
FORECAST_LENGTH_DAYS = config["forecast"]["length_days"]  # how far the rollout runs
LEAD_DAYS = config["forecast"]["lead_days"]  # leads shown as map columns
SCORE_METRIC_NAMES = config["forecast"]["score_metrics"]  # how forecasts are evaluated

# Bounding box applied before any statistics, ranking, or plotting. Any bound
# left null is not applied.
MIN_LONGITUDE = config["domain"]["min_longitude"]
MAX_LONGITUDE = config["domain"]["max_longitude"]
MIN_LATITUDE = config["domain"]["min_latitude"]
MAX_LATITUDE = config["domain"]["max_latitude"]

if max(LEAD_DAYS) > FORECAST_LENGTH_DAYS:
    raise ValueError(
        f"lead_days reaches {max(LEAD_DAYS)} d but the forecast is only {FORECAST_LENGTH_DAYS} d long"
    )

# Climatology the fields are standardized against before matching. See the config
# file for what the two knobs do.
CLIMATOLOGY_GROUP = config["climatology"]["group"]
CLIMATOLOGY_REDUCE_DIMS = tuple(config["climatology"]["reduce_dims"])

os.makedirs(FIG_DIR, exist_ok=True)

# %%

# Open your Zarr store if not already loaded
ds = xr.open_mfdataset(ZARR_GLOB)  # Adjust the filename as needed

# Coordinates are ascending, so a slice with either end None is a no-op.
ds = ds.sel(
    longitude=slice(MIN_LONGITUDE, MAX_LONGITUDE),
    latitude=slice(MIN_LATITUDE, MAX_LATITUDE),
)
print(
    f"Domain: {float(ds.longitude.min()):.2f} to {float(ds.longitude.max()):.2f} lon, "
    f"{float(ds.latitude.min()):.2f} to {float(ds.latitude.max()):.2f} lat "
    f"({ds.sizes['latitude']} x {ds.sizes['longitude']})"
)

# Identify SST and SSH variables in your dataset
# Adjust variable names if yours are named differently (e.g. 'thetao' instead of 'tos')
sst_var = "tos" if "tos" in ds else ("thetao" if "thetao" in ds else list(ds.data_vars)[0])
ssh_var = "zos" if "zos" in ds else list(ds.data_vars)[1]

# If SST has a depth dimension, slice the surface (depth=0)
sst_data = ds[sst_var].isel(depth=0) if "depth" in ds[sst_var].dims else ds[sst_var]
ssh_data = ds[ssh_var]


def noleap_dayofyear(time):
    """
    Day of year with Feb 29 folded into Feb 28, so no group is left with only the
    leap years to estimate from. Feb 29 is day 60 of a leap year, so shifting day
    60 onwards back by one both pools Feb 29 with Feb 28 (day 59) and keeps the
    rest of the year aligned with non-leap years (leap Mar 1 = 61 -> 60 = Mar 1).
    """
    doy = time.dt.dayofyear
    return xr.where(time.dt.is_leap_year & (doy >= 60), doy - 1, doy)


sst_data = sst_data.assign_coords(climday=noleap_dayofyear(sst_data.time))
ssh_data = ssh_data.assign_coords(climday=noleap_dayofyear(ssh_data.time))

stats = {}
for name, da in [("SST", sst_data), ("SSH", ssh_data)]:
    stats[name] = {
        "min": float(da.min().compute()),
        "max": float(da.max().compute()),
        "mean": float(da.mean().compute()),
        "variance": float(da.var().compute()),
        "std_dev": float(da.std().compute()),
    }

# Display results formatted
for var_name, s in stats.items():
    unit = "°C" if var_name == "SST" else "m"
    print(f"\n[{var_name}] ({unit})")
    print(f"  Min:      {s['min']:.2f} {unit}")
    print(f"  Max:      {s['max']:.2f} {unit}")
    print(f"  Mean:     {s['mean']:.2f} {unit}")
    print(f"  Variance: {s['variance']:.4f}")
    print(f"  Std Dev:  {s['std_dev']:.2f} {unit}")

# %%
# Spatial Variance
sst_data.var(dim="time").plot(cmap="plasma", robust=True)
plt.title("Spatial Variance of SST")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/sst_spatial_variance.png", dpi=150)
plt.show()

# %%
# Compute global stats
mean_val = stats["SST"]["mean"]
std_val = stats["SST"]["std_dev"]

# Define bounds
bounds_1std = (mean_val - std_val, mean_val + std_val)
bounds_2std = (mean_val - 2 * std_val, mean_val + 2 * std_val)

# Extract non-null grid points as a flat array
flat_data = sst_data.values.flatten()
flat_data = flat_data[~np.isnan(flat_data)]
total_points = len(flat_data)

# Count points within 1 and 2 std dev
count_1std = np.sum((flat_data >= bounds_1std[0]) & (flat_data <= bounds_1std[1]))
count_2std = np.sum((flat_data >= bounds_2std[0]) & (flat_data <= bounds_2std[1]))

# Calculate exact percentages
pct_1std = (count_1std / total_points) * 100
pct_2std = (count_2std / total_points) * 100

print(f"Mean: {mean_val:.2f} °C | Std Dev: {std_val:.2f} °C\n")
print(f"Exact Grid Points within Mean ± 1 Std Dev ({bounds_1std[0]:.2f}°C to {bounds_1std[1]:.2f}°C): {pct_1std:.2f}%")
print(f"Exact Grid Points within Mean ± 2 Std Dev ({bounds_2std[0]:.2f}°C to {bounds_2std[1]:.2f}°C): {pct_2std:.2f}%")

# %%
plt.figure(figsize=(10, 5))

# Plot the empirical distribution
plt.hist(flat_data, bins=60, density=True, color="teal", alpha=0.5)

# Add Mean and Standard Deviation boundary lines
plt.axvline(mean_val, color="red", linestyle="-", linewidth=2, label=f"Mean ({mean_val:.2f}°C)")
plt.axvline(bounds_1std[0], color="orange", linestyle="--", label=f"±1 Std Dev ({pct_1std:.1f}% of data)")
plt.axvline(bounds_1std[1], color="orange", linestyle="--")
plt.axvline(bounds_2std[0], color="purple", linestyle=":", label=f"±2 Std Dev ({pct_2std:.1f}% of data)")
plt.axvline(bounds_2std[1], color="purple", linestyle=":")

plt.title("SST Empirical Distribution", fontsize=13)
plt.xlabel("Sea Surface Temperature (°C)")
plt.ylabel("Density")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/sst_distribution.png", dpi=150)
plt.show()

# %%
# Compute key percentiles
quantiles = np.percentile(flat_data, [5, 25, 50, 75, 95])

print(f" 5th Percentile: {quantiles[0]:.2f} °C (Coldest 5% threshold)")
print(f"25th Percentile (Q1): {quantiles[1]:.2f} °C")
print(f"50th Percentile (Median): {quantiles[2]:.2f} °C")
print(f"75th Percentile (Q3): {quantiles[3]:.2f} °C")
print(f"95th Percentile: {quantiles[4]:.2f} °C (Warmest 5% threshold)")

del flat_data  # ~1.4 GB of the job's 10 GB, and nothing below needs it

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Spatial Variation (Time-Averaged SST Map)
sst_data.mean(dim="time").plot(ax=axes[0], cmap="plasma", cbar_kwargs={"label": "SST (°C)"})
axes[0].set_title("Spatial Variation: Mean SST")

# 2. Temporal Variation (Domain-Averaged Time Series)
sst_data.mean(dim=["latitude", "longitude"]).plot(ax=axes[1], color="navy")
axes[1].set_title("Temporal Variation: Domain-Averaged SST")
axes[1].set_ylabel("SST (°C)")
axes[1].grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/sst_spatial_temporal_variation.png", dpi=150)
plt.show()

# %%
def normalize_by_climatology(da, group=CLIMATOLOGY_GROUP, reduce_dims=CLIMATOLOGY_REDUCE_DIMS):
    """
    Normalize a DataArray by subtracting the climatological mean and dividing by
    the climatological std dev. Assumes 'time' is a dimension in da.

    group : str
        Grouping coordinate for the statistics, e.g. "time.dayofyear" for one
        mean/std per calendar day, "time.month" for one per month.
    reduce_dims : tuple of str
        Extra dims to reduce over when forming the statistics. Empty (the
        default) keeps them pointwise: one mean/std per group per grid cell.

    Returns (clim_mean, clim_std, normalized) — the statistics come back too so
    the forecast below can map anomalies back into physical units.
    """
    stat_dims = ["time", *reduce_dims]

    da = da.compute()
    grouped = da.groupby(group)
    clim_mean = grouped.mean(dim=stat_dims)
    clim_std = grouped.std(dim=stat_dims)

    # Filling a preallocated output group by group keeps peak memory at two copies
    # of the field. Groupby arithmetic, `(da.groupby(g) - mean).groupby(g) / std`,
    # is tidier but builds a full-size temporary per operation, which overruns the
    # job's memory limit on the full record.
    group_dim = clim_mean.dims[0]
    labels = da[group].values
    values = da.values
    out = np.empty_like(values)
    for label in clim_mean[group_dim].values:
        in_group = labels == label
        mean = clim_mean.sel({group_dim: label}).values
        std = clim_std.sel({group_dim: label}).values
        out[in_group] = (values[in_group] - mean) / std

    return clim_mean, clim_std, da.copy(data=out)

def select_top_k_buffer(data, rmsd, k=K, buffer = 14):
    """
    Select the top K analogs while ensuring they are at least 'buffer' days apart.
    """
    selected_dates = []
    sorted_data = data.sortby(rmsd)

    for date in sorted_data.time.values:
        if not selected_dates or all(abs((date - np.array(selected_dates)).astype('timedelta64[D]')) >= np.timedelta64(buffer, 'D')):
            selected_dates.append(date)
        if len(selected_dates) >= k:
            break

    return data.sel(time=selected_dates)

def front_mask(ssh_grid, level=SSH_CONTOUR_LEVEL):
    """
    Boolean mask of the Loop Current front: the SSH contour at `level`, following
    the standard 17 cm criterion (Leben 2005).

    Returns the inside edge of the region at or above `level` — cells above it
    that are 4-adjacent to water below it. Requiring the neighbour to be water
    keeps coastlines out of the mask, which would otherwise trace every shore
    where high SSH meets land.
    """
    inside = np.nan_to_num(ssh_grid, nan=-np.inf) >= level
    water_below = ~inside & ~np.isnan(ssh_grid)

    neighbour_below = np.zeros_like(inside)
    neighbour_below[1:, :] |= water_below[:-1, :]
    neighbour_below[:-1, :] |= water_below[1:, :]
    neighbour_below[:, 1:] |= water_below[:, :-1]
    neighbour_below[:, :-1] |= water_below[:, 1:]

    return inside & neighbour_below

def compute_mhd(mask_A, edt_A, mask_B, edt_B):
    """
    Modified Hausdorff Distance between two front masks, in grid units.

    edt_X is the Euclidean distance transform of ~mask_X, i.e. the distance from
    every grid cell to the nearest front point of X. Reading it at the other
    set's points gives the same nearest-neighbour distances as a full pairwise
    cdist, but in O(N) instead of O(|A|*|B|).
    """
    if not mask_A.any() or not mask_B.any():
        return np.inf  # Handle empty front edge cases

    return max(edt_B[mask_A].mean(), edt_A[mask_B].mean())

# %%
sst_clim_mean, sst_clim_std, sst_normalized = normalize_by_climatology(sst_data)
ssh_clim_mean, ssh_clim_std, ssh_normalized = normalize_by_climatology(ssh_data)

# Everything downstream reads variables out of this bundle, so adding one means
# adding an entry here rather than threading arguments through every function.
FIELDS = {
    "sst": {
        "raw": sst_data,
        "normalized": sst_normalized,
        "clim_mean": sst_clim_mean,
        "clim_std": sst_clim_std,
        "unit": "°C",
        "cmap": "plasma",
        "symmetric": False,
    },
    "ssh": {
        "raw": ssh_data,
        "normalized": ssh_normalized,
        "clim_mean": ssh_clim_mean,
        "clim_std": ssh_clim_std,
        "unit": "m",
        "cmap": "RdBu_r",
        "symmetric": True,
    },
}

# Timestamps carry a 12:00 time of day, so take the target stamp from the data
# rather than parsing TARGET_DATE, which would land on midnight and miss.
target_time = pd.Timestamp(sst_data.sel(time=TARGET_DATE).time.values[0])

# Map panels follow the domain's aspect, so a narrow longitude band does not come
# out stretched. Width is clamped so columns stay legible at either extreme.
PANEL_H = 2.4
PANEL_W = min(
    max(PANEL_H * float(sst_data.longitude.max() - sst_data.longitude.min())
        / float(sst_data.latitude.max() - sst_data.latitude.min()), 1.7),
    4.0,
)

# %%
# --- Selection metrics -------------------------------------------------------
# Each returns one score per library day; lower ranks better. To add a metric,
# write a function of no arguments and register it in SELECTION_METRICS.


def anomaly_rmsd(name):
    """Spatial RMSD of normalized anomalies against the target day."""
    library = FIELDS[name]["normalized"].sel(time=LIBRARY_DATES)
    target = FIELDS[name]["normalized"].sel(time=target_time)
    return np.sqrt(((library - target) ** 2).mean(dim=["latitude", "longitude"]))


def front_mhd(name="ssh"):
    """
    MHD between the target's Loop Current front and every library day's.

    Works on the raw field. The front is a property of physical SSH; the
    normalized anomaly has exactly the mean structure that defines it removed.
    """
    library_raw = FIELDS[name]["raw"].sel(time=LIBRARY_DATES).compute()
    target_mask = front_mask(FIELDS[name]["raw"].sel(time=target_time).values)
    target_edt = distance_transform_edt(~target_mask)

    scores = np.empty(library_raw.sizes["time"])
    for i, grid in enumerate(library_raw.values):
        mask = front_mask(grid)
        scores[i] = compute_mhd(target_mask, target_edt, mask, distance_transform_edt(~mask))
    return xr.DataArray(scores, coords={"time": library_raw.time}, dims="time")


SELECTION_METRICS = {
    "sst_rmsd": lambda: anomaly_rmsd("sst"),
    "ssh_rmsd": lambda: anomaly_rmsd("ssh"),
    "ssh_front_mhd": lambda: front_mhd("ssh"),
}
SELECTION_UNITS = {"sst_rmsd": "sigma", "ssh_rmsd": "sigma", "ssh_front_mhd": "grid units"}

if SELECTION_METRIC not in SELECTION_METRICS:
    raise ValueError(f"unknown selection_metric {SELECTION_METRIC!r}; choose from {sorted(SELECTION_METRICS)}")

selection_score = SELECTION_METRICS[SELECTION_METRIC]()
top_analogs = select_top_k_buffer(selection_score, selection_score, k=K, buffer=BUFFER_DAYS)

print(f"\n--- Top {K} analogs by {SELECTION_METRIC} for target {TARGET_DATE} ---")
for rank, (t, val) in enumerate(zip(top_analogs.time.values, top_analogs.values), 1):
    date_str = pd.to_datetime(t).strftime("%Y-%m-%d")
    print(f"Rank {rank}: {date_str} | {SELECTION_METRIC} = {val:.4f} {SELECTION_UNITS[SELECTION_METRIC]}")

# %%
# --- Forecast rollout --------------------------------------------------------


def to_physical(anomaly, valid_time, clim_mean, clim_std):
    """
    Undo the normalization at `valid_time`, turning a forecast anomaly back into
    physical units against the climatology of the day being forecast.
    """
    group_dim = clim_mean.dims[0]
    label = int(sst_data.sel(time=valid_time)[CLIMATOLOGY_GROUP])
    return clim_mean.sel({group_dim: label}) + anomaly * clim_std.sel({group_dim: label})


def member_forecast(name, analog_time, lead):
    """One analog's own forecast of `name` at `lead`, in physical units."""
    field = FIELDS[name]
    anomaly = field["normalized"].sel(time=pd.Timestamp(analog_time) + pd.Timedelta(days=lead))
    return to_physical(anomaly, target_time + pd.Timedelta(days=lead), field["clim_mean"], field["clim_std"])


def rollout(analog_times, leads):
    """
    Forecast every variable for every member at every lead, plus the observed
    truth. Members are mapped back through the valid day's climatology, which is
    affine, so the ensemble is exactly the mean of the members.

    Returns truth[lead][var], members[lead][member][var], ensemble[lead][var].
    """
    truth, members, ensemble = {}, {}, {}
    for lead in leads:
        valid_time = target_time + pd.Timedelta(days=lead)
        truth[lead] = {n: FIELDS[n]["raw"].sel(time=valid_time).compute() for n in FIELDS}
        members[lead] = [{n: member_forecast(n, t, lead) for n in FIELDS} for t in analog_times]
        ensemble[lead] = {n: sum(m[n] for m in members[lead]) / len(members[lead]) for n in FIELDS}
    return truth, members, ensemble


# --- Scoring metrics ---------------------------------------------------------
# Each takes {var: forecast}, {var: observed}, and the valid timestamp, and
# returns a scalar. Register a new one in SCORE_METRICS, along with its units and
# whether high or low scores are better.


def climatology_at(name, valid_time):
    """Climatological mean of `name` for the group `valid_time` falls in."""
    field = FIELDS[name]
    group_dim = field["clim_mean"].dims[0]
    label = int(sst_data.sel(time=valid_time)[CLIMATOLOGY_GROUP])
    return field["clim_mean"].sel({group_dim: label})


def field_rmse(name):
    def score(forecast, truth, valid_time):
        return float(np.sqrt(((forecast[name] - truth[name]) ** 2).mean()))

    return score


def field_acc(name):
    """
    Anomaly correlation coefficient: the spatial correlation between forecast and
    observed departures from climatology. 1 is a perfect anomaly pattern, 0 is no
    better than climatology, negative is worse.

    Uncentred, following the usual verification convention — the anomalies are
    already departures from a mean, so no second mean is removed.
    """

    def score(forecast, truth, valid_time):
        clim = climatology_at(name, valid_time)
        f = (forecast[name] - clim).values
        o = (truth[name] - clim).values
        usable = np.isfinite(f) & np.isfinite(o)
        f, o = f[usable], o[usable]
        denominator = np.sqrt((f**2).sum() * (o**2).sum())
        return float((f * o).sum() / denominator) if denominator > 0 else np.nan

    return score


def front_mhd_error(forecast, truth, valid_time):
    """Displacement between the forecast Loop Current front and the observed one."""
    forecast_mask = front_mask(forecast["ssh"].values)
    truth_mask = front_mask(truth["ssh"].values)
    return compute_mhd(
        forecast_mask, distance_transform_edt(~forecast_mask),
        truth_mask, distance_transform_edt(~truth_mask),
    )


SCORE_METRICS = {
    "sst_rmse": field_rmse("sst"),
    "ssh_rmse": field_rmse("ssh"),
    "sst_acc": field_acc("sst"),
    "ssh_acc": field_acc("ssh"),
    "ssh_front_mhd": front_mhd_error,
}
SCORE_UNITS = {
    "sst_rmse": "°C", "ssh_rmse": "m",
    "sst_acc": "correlation", "ssh_acc": "correlation",
    "ssh_front_mhd": "grid units",
}
# ACC is a skill score, so it ranks the opposite way to the error metrics.
SCORE_HIGHER_IS_BETTER = {"sst_acc", "ssh_acc"}

for metric in SCORE_METRIC_NAMES:
    if metric not in SCORE_METRICS:
        raise ValueError(f"unknown score metric {metric!r}; choose from {sorted(SCORE_METRICS)}")

# Errors are evaluated every day of the forecast, so the skill curves are
# continuous even though the map columns stay at the coarser LEAD_DAYS.
ERROR_LEADS = list(range(0, FORECAST_LENGTH_DAYS + 1))
truth_by_lead, members_by_lead, ensemble_by_lead = rollout(top_analogs.time.values, ERROR_LEADS)

valid_times = {lead: target_time + pd.Timedelta(days=lead) for lead in ERROR_LEADS}

skill = {}
for metric in SCORE_METRIC_NAMES:
    fn = SCORE_METRICS[metric]
    per_member = [
        [fn(members_by_lead[lead][i], truth_by_lead[lead], valid_times[lead]) for lead in ERROR_LEADS]
        for i in range(K)
    ]
    ensemble_curve = [fn(ensemble_by_lead[lead], truth_by_lead[lead], valid_times[lead]) for lead in ERROR_LEADS]
    skill[metric] = (per_member, ensemble_curve)

print(f"\n--- Ensemble skill by lead (selection: {SELECTION_METRIC}, K = {K}) ---")
print("lead  " + "  ".join(f"{m:>16s}" for m in SCORE_METRIC_NAMES))
print("      " + "  ".join(f"{'(' + SCORE_UNITS[m] + ')':>16s}" for m in SCORE_METRIC_NAMES))
for j, lead in enumerate(ERROR_LEADS):
    row = "  ".join(f"{skill[m][1][j]:>16.4f}" for m in SCORE_METRIC_NAMES)
    print(f"{lead:>4d}  {row}")

print(f"\n--- Best single member vs ensemble, by metric ---")
for metric in SCORE_METRIC_NAMES:
    per_member, ensemble_curve = skill[metric]
    member_means = [np.mean(curve) for curve in per_member]
    # ACC ranks the other way round from the error metrics.
    best = int(np.argmax(member_means) if metric in SCORE_HIGHER_IS_BETTER else np.argmin(member_means))
    best_date = pd.to_datetime(top_analogs.time.values[best]).strftime("%Y-%m-%d")
    print(
        f"{metric:>16s}: ensemble {np.mean(ensemble_curve):.4f}  |  "
        f"best member {best_date} (rank {best + 1}) {member_means[best]:.4f} {SCORE_UNITS[metric]}"
    )

# %%
# Categorical hues for the individual analog members, assigned in fixed order.
# Validated for colourblind separation against a light surface.
ANALOG_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
ENSEMBLE_COLOR = "#0b0b0b"  # the ensemble is an aggregate, not a peer series


def _strip_ticks(ax, keep_x, keep_y):
    """Interior panels of a dense grid carry no tick labels; the edges do."""
    ax.set_xlabel("")
    if not keep_x:
        ax.set_xticklabels([])
    if not keep_y:
        ax.set_yticklabels([])


def plot_analog_grid(var, analog_times, filename):
    """
    One row per analog (state, then error against truth), then the ensemble mean,
    then the observed truth. Columns are LEAD_DAYS. The final column carries one
    panel per scoring metric, each showing every member and the ensemble against
    daily lead.

    For SSH the observed front is drawn on every panel and each member's own
    front dashed over it, so displacement can be read directly.
    """
    field = FIELDS[var]
    n = len(analog_times)
    if n > len(ANALOG_COLORS):
        raise ValueError(f"{n} analogs but only {len(ANALOG_COLORS)} categorical colours defined")

    contour_level = SSH_CONTOUR_LEVEL if var == "ssh" else None
    truths = [truth_by_lead[lead][var] for lead in LEAD_DAYS]
    states = [[members_by_lead[lead][i][var] for lead in LEAD_DAYS] for i in range(n)]
    states.append([ensemble_by_lead[lead][var] for lead in LEAD_DAYS])
    states.append(truths)
    errors = [[states[r][j] - truths[j] for j in range(len(LEAD_DAYS))] for r in range(n + 1)]

    row_labels = [f"Analog {i + 1}\n{pd.to_datetime(t).strftime('%Y-%m-%d')}" for i, t in enumerate(analog_times)]
    row_labels += [f"Ensemble mean\nK = {n}", f"Observed\n{TARGET_DATE}"]

    state_min = min(float(f.min()) for row in states for f in row)
    state_max = max(float(f.max()) for row in states for f in row)
    if field["symmetric"]:
        state_max = max(abs(state_min), abs(state_max))
        state_min = -state_max
    err_max = max(float(np.nanpercentile(np.abs(f.values), 99)) for row in errors for f in row)

    n_rows = 2 * n + 3  # state+error per analog, state+error for the ensemble, truth state
    n_cols = len(LEAD_DAYS) + 2  # lead columns, a spacer, then the skill curves
    # A trailing short row holds the two shared colorbars, and an empty spacer
    # column keeps the curve tick labels off the last map column. The curve column
    # gets a floor in absolute inches so it stays readable when panels are narrow.
    curve_ratio = max(1.5, 3.2 / PANEL_W)
    fig = plt.figure(
        figsize=(PANEL_W * (len(LEAD_DAYS) + 0.3 + curve_ratio) + 1.0, PANEL_H * n_rows + 1.0)
    )
    gs = fig.add_gridspec(
        n_rows + 1, n_cols,
        height_ratios=[1] * n_rows + [0.18],
        width_ratios=[1] * len(LEAD_DAYS) + [0.3, curve_ratio],
        hspace=0.14, wspace=0.06,
    )

    state_axes, error_axes = [], []
    state_art = err_art = None
    last_row = 2 * (n + 1)  # the truth row, which keeps its x tick labels
    for r in range(n + 2):  # analogs, ensemble, truth
        state_row = 2 * r
        for j, lead in enumerate(LEAD_DAYS):
            ax = fig.add_subplot(gs[state_row, j])
            state_axes.append(ax)
            state_art = states[r][j].plot(ax=ax, cmap=field["cmap"], vmin=state_min, vmax=state_max,
                                          add_colorbar=False)
            if contour_level is not None:
                ax.contour(truths[j].longitude, truths[j].latitude, truths[j].values,
                           levels=[contour_level], colors="k", linewidths=1.2)
                if r < n + 1:  # the member's own front, to read displacement against truth
                    ax.contour(states[r][j].longitude, states[r][j].latitude, states[r][j].values,
                               levels=[contour_level],
                               colors=[ANALOG_COLORS[r] if r < n else ENSEMBLE_COLOR],
                               linewidths=1.2, linestyles="--")
            ax.set_title(f"lead +{lead} d" if state_row == 0 else "")
            ax.set_ylabel(row_labels[r] if j == 0 else "")
            _strip_ticks(ax, keep_x=state_row == last_row, keep_y=j == 0)

            if r < n + 1:  # truth has no error row
                ax_e = fig.add_subplot(gs[state_row + 1, j])
                error_axes.append(ax_e)
                err_art = errors[r][j].plot(ax=ax_e, cmap="PuOr_r", vmin=-err_max, vmax=err_max,
                                            add_colorbar=False)
                ax_e.set_title("")
                ax_e.set_ylabel(f"error ({field['unit']})" if j == 0 else "")
                _strip_ticks(ax_e, keep_x=False, keep_y=j == 0)

    # Two shared colorbars in the trailing row. Distinct diverging pairs, so a state
    # panel is never mistaken for an error panel at a glance.
    half = max(1, len(LEAD_DAYS) // 2)
    fig.colorbar(state_art, cax=fig.add_subplot(gs[n_rows, :half]), orientation="horizontal",
                 label=f"State ({field['unit']})")
    fig.colorbar(err_art, cax=fig.add_subplot(gs[n_rows, half:len(LEAD_DAYS)]),
                 orientation="horizontal", label=f"Forecast − observed ({field['unit']})")

    # Final column: one panel per scoring metric, evaluated daily.
    n_metrics = len(SCORE_METRIC_NAMES)
    block = n_rows // n_metrics
    for mi, metric in enumerate(SCORE_METRIC_NAMES):
        r0 = mi * block
        r1 = n_rows if mi == n_metrics - 1 else (mi + 1) * block
        ax = fig.add_subplot(gs[r0:r1, -1])
        per_member, ensemble_curve = skill[metric]
        for i in range(n):
            ax.plot(ERROR_LEADS, per_member[i], color=ANALOG_COLORS[i], linewidth=2,
                    label=row_labels[i].replace("\n", " "))
        ax.plot(ERROR_LEADS, ensemble_curve, color=ENSEMBLE_COLOR, linewidth=3, label="Ensemble mean")
        ax.set_ylabel(f"{metric} ({SCORE_UNITS[metric]})")
        ax.grid(True, linestyle=":", alpha=0.5)
        if mi == n_metrics - 1:
            ax.set_xlabel("Lead (days)")
        if mi == 0:
            ax.set_title(f"Skill by lead\nanalogs selected by {SELECTION_METRIC}")
            ax.legend(loc="upper left", fontsize="small")

    fig.savefig(f"{FIG_DIR}/{filename}", dpi=130, bbox_inches="tight")
    plt.show()


for var in FIELDS:
    plot_analog_grid(var, top_analogs.time.values, f"analog_grid_{var}_by_{SELECTION_METRIC}.png")

# %%
# Target against the best analog, both variables, at lead 0.
best_date = pd.to_datetime(top_analogs.time.values[0]).strftime("%Y-%m-%d")
fig, axes = plt.subplots(
    len(FIELDS), 2, figsize=(2 * PANEL_W * 1.9 + 1.0, PANEL_H * 1.9 * len(FIELDS)), squeeze=False
)

for row, (var, field) in enumerate(FIELDS.items()):
    target_field = field["raw"].sel(time=target_time).compute()
    analog_field = field["raw"].sel(time=top_analogs.time.values[0]).compute()
    vmin = min(float(target_field.min()), float(analog_field.min()))
    vmax = max(float(target_field.max()), float(analog_field.max()))
    if field["symmetric"]:
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax
    kwargs = dict(cmap=field["cmap"], vmin=vmin, vmax=vmax, cbar_kwargs={"label": f"{var.upper()} ({field['unit']})"})

    for col, (data, label) in enumerate([(target_field, f"Target {TARGET_DATE}"), (analog_field, f"Best analog {best_date}")]):
        data.plot(ax=axes[row][col], **kwargs)
        if var == "ssh":
            axes[row][col].contour(data.longitude, data.latitude, data.values,
                                   levels=[SSH_CONTOUR_LEVEL], colors="k", linewidths=1.5)
        axes[row][col].set_title(f"{var.upper()} — {label}")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/target_vs_best_analog_by_{SELECTION_METRIC}.png", dpi=150)
plt.show()

# %%
