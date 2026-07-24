# %% Is the SWOT analog-selection metric hindering the forecast? (oracle ladder)
# The SWOT-driven forecast loses to dense persistence. To find out whether the
# *selection metric* is the bottleneck (vs. a predictability ceiling), we re-select
# analogs using targets a real metric can't have and score the forecasts identically.
# Because `analog_vs_glorys` chooses analogs ONLY from (obs_grid, mask), each rung is
# one call with a different selection target:
#
#   metric-analog        SWOT swath  + swath mask      the real metric (current)
#   obs-oracle (mask)    GLORYS@T    + swath mask      perfect obs, same coverage
#   present-oracle       GLORYS@T    + full ocean      perfect obs, full coverage
#   present-oracle (ds)  deseas@T    + full ocean      mesoscale-only selection
#   future-oracle        GLORYS@T+14 + full ocean      best library match to truth (lead 0)
#
# All are scored against the dense GLORYS truth at T+14 with full-field ACC AND the
# Loop Current front MHD. Reading: present-oracle >> metric-analog => the metric is
# hindering; future-oracle ~ persistence => the library/predictability is the ceiling.
# Companion: main_swot_glorys.py (the production SWOT->GLORYS forecast).
import glob
import os
import re

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import viz
import swot_data
import swot_analog as sa
from analog import ModelLibrary, AnalogForecaster, seasonal_climatology
from distances import CorrelationDistance

# %% ---- CONFIG ----------------------------------------------------------- #
LIB_CANDIDATES = ["data/glorys_gom_zos_2004_2013.nc", "data/glorys_gom_zos_2004.nc"]
TRUTH_PATH = "data/glorys_gom_zos_2024jan.nc"
SWOT_DIR = "data/swot"
K, MIN_SEP, LEAD = 10, 14, 14
MIN_CELLS = 500
# -------------------------------------------------------------------------- #

# %% Library + dense 2024 truth; a raw and a deseasonalized-selection forecaster.
lib_path = next((p for p in LIB_CANDIDATES if os.path.exists(p)), LIB_CANDIDATES[-1])
ds = xr.open_dataset(lib_path)
times = ds["time"].values
state = ds["zos"].values
saf = AnalogForecaster(ModelLibrary(ds, var="zos"), CorrelationDistance())    # seasonal-kept
deseas_anom = state - seasonal_climatology(state, times)              # mesoscale residual
saf_ds = AnalogForecaster(ModelLibrary(ds, var="zos", anomaly=deseas_anom),
                          CorrelationDistance())                      # mesoscale selection
print(f"Library {lib_path}: {ds.sizes['time']} states")

truth_ds = xr.open_dataset(TRUTH_PATH)
truth_times = truth_ds["time"].values
truth_state = truth_ds["zos"].values
assert np.allclose(truth_ds["longitude"].values, saf.lon) and \
    np.allclose(truth_ds["latitude"].values, saf.lat), "truth grid != library grid"
tmin, tmax = truth_times.min(), truth_times.max()


def field_on(date64):
    """Nearest daily GLORYS-2024 surface field to a calendar date."""
    i = int(np.argmin(np.abs(truth_times - np.datetime64(date64))))
    return truth_state[i]


# %% Per obs day: run the selection ladder; every rung is one analog_vs_glorys call.
swot_paths = sorted(glob.glob(os.path.join(SWOT_DIR, "*.nc")))
swot_days = sorted({re.search(r"_(\d{8})T", p).group(1) for p in swot_paths})
obs_days_all = [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in swot_days]

VARIANTS = ["metric-analog", "obs-oracle (mask)", "present-oracle",
            "present-oracle (ds)", "future-oracle"]
runs = {v: [] for v in VARIANTS}          # per-variant list of analog_vs_glorys results
obs_days = []

for T in obs_days_all:
    vday = np.datetime64(T) + np.timedelta64(LEAD, "D")
    if not (tmin <= vday <= tmax):
        continue
    obs_grid, mask = sa.swath_to_grid(swot_data.swaths_for_dates([T])[0], saf.lon, saf.lat)
    if mask.sum() < MIN_CELLS:
        continue
    gT, gV = field_on(T), field_on(vday)              # dense GLORYS @T and @T+14
    gT_ds = saf.deseasonalize(saf.anom_of(gT), T)     # deseasonalized dense @T
    gV_ds = saf.deseasonalize(saf.anom_of(gV), vday)  # deseasonalized dense truth @T+14
    kw = dict(persist_surf=gT, k=K, min_sep=MIN_SEP)

    runs["metric-analog"].append(
        sa.analog_vs_glorys(saf, obs_grid, mask, gV, vday, lead=LEAD, **kw))
    runs["obs-oracle (mask)"].append(
        sa.analog_vs_glorys(saf, gT, mask, gV, vday, lead=LEAD, **kw))
    runs["present-oracle"].append(
        sa.analog_vs_glorys(saf, gT, saf.ocean, gV, vday, lead=LEAD, **kw))
    runs["present-oracle (ds)"].append(
        sa.analog_vs_glorys(saf_ds, gT_ds, saf_ds.ocean, gV, vday, lead=LEAD, **kw))
    runs["future-oracle"].append(          # best library mesoscale match to the truth, lead 0
        sa.analog_vs_glorys(saf_ds, gV_ds, saf_ds.ocean, gV, vday, lead=0, **kw))
    obs_days.append(T)

print(f"Used {len(obs_days)} obs days (K={K}, min_sep={MIN_SEP}, lead={LEAD}).\n")

# %% Aggregate: mean best-analog and ensemble skill per variant, both metrics.
def agg(results, key, reduce):
    return float(np.nanmean([reduce(r[key]) for r in results]))

persist_acc = agg(runs["metric-analog"], "acc_persist", lambda x: x)      # same for all
persist_lc = agg(runs["metric-analog"], "lc_mhd_persist", lambda x: x)

rows = {}
for v in VARIANTS:
    R = runs[v]
    rows[v] = dict(
        acc_best=agg(R, "acc", np.nanmax), acc_ens=agg(R, "acc_ens", lambda x: x),
        lc_best=agg(R, "lc_mhd", np.nanmin), lc_ens=agg(R, "lc_mhd_ens", lambda x: x))

hdr = f"{'variant':22s} {'ACC best':>9s} {'ACC ens':>8s} | {'LC best':>8s} {'LC ens':>7s}"
print(hdr)
print("-" * len(hdr))
for v in VARIANTS:
    r = rows[v]
    print(f"{v:22s} {r['acc_best']:+9.3f} {r['acc_ens']:+8.3f} | "
          f"{r['lc_best']:8.1f} {r['lc_ens']:7.1f}")
print(f"{'persistence':22s} {persist_acc:+9.3f} {'—':>8s} | {persist_lc:8.1f} {'—':>7s}")
print("   (ACC higher = better; Loop Current MHD km lower = better)")

# %% Verdict — the headline gaps.
# metric headroom: how much a perfect *current* dense obs improves over the real metric
d_acc = rows["present-oracle"]["acc_best"] - rows["metric-analog"]["acc_best"]
d_lc = rows["metric-analog"]["lc_best"] - rows["present-oracle"]["lc_best"]   # +ve = oracle better
# coverage vs obs-error split (obs-oracle uses perfect values over the SAME swath mask)
cov_acc = rows["present-oracle"]["acc_best"] - rows["obs-oracle (mask)"]["acc_best"]
err_acc = rows["obs-oracle (mask)"]["acc_best"] - rows["metric-analog"]["acc_best"]
# library ceiling = best achievable using the truth (ensemble for ACC, best-analog for the front)
ceil_acc = max(rows["future-oracle"]["acc_best"], rows["future-oracle"]["acc_ens"])
ceil_lc = min(rows["future-oracle"]["lc_best"], rows["future-oracle"]["lc_ens"])
print("\nVERDICT:")
print(f"  a better CURRENT metric (present-oracle - metric-analog) = {d_acc:+.3f} ACC, "
      f"{d_lc:+.1f} km  ({'some skill left on the table' if d_acc > 0.05 or d_lc > 3 else 'metric ~ at ceiling'})")
print(f"    of which coverage (mask->full)={cov_acc:+.3f} ACC, obs-error (SWOT->perfect)={err_acc:+.3f} ACC")
print(f"  even the perfect current metric still loses to persistence: "
      f"present-oracle {rows['present-oracle']['acc_best']:+.3f} vs persistence {persist_acc:+.3f} ACC, "
      f"{rows['present-oracle']['lc_best']:.1f} vs {persist_lc:.1f} km")
print(f"  library ceiling (uses the truth): {ceil_acc:+.3f} ACC / {ceil_lc:.1f} km  vs persistence "
      f"{persist_acc:+.3f} / {persist_lc:.1f}  -> headroom exists but is unreachable from the current field")
print(f"  deseasonalizing selection: ACC {rows['present-oracle (ds)']['acc_best'] - rows['present-oracle']['acc_best']:+.3f}, "
      f"LC {rows['present-oracle']['lc_best'] - rows['present-oracle (ds)']['lc_best']:+.1f} km (≈0 -> seasonal cycle is not the problem)")

# %% Summary figure.
viz.plot_metric_ladder(VARIANTS, rows, persist_acc, persist_lc, lead=LEAD)

plt.show()

# %%
