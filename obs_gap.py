# %% Where is the SWOT->GLORYS analog skill actually lost? (gap decomposition)
# The metric ladder (metric_diagnostic.py) asks "is the selection metric the
# bottleneck?". This script asks the sharper question that gates building a LEARNED
# distance: of the skill a perfect selector would gain, how much is recoverable by a
# better DISTANCE and how much is simply missing data?
#
# Only two things vary across the rungs -- WHAT is observed and WHERE. The obs-side
# representation is held fixed: every rung enters the distance as a library-referenced
# anomaly, so no rung is handicapped by an absolute-vs-anomaly mismatch.
#
#   A swot     real SWOT ssha        + swath mask    the deployed cross-source distance
#   B perfect  anom(GLORYS@T)        + swath mask    perfect values, SAME footprint
#   C dense    anom(GLORYS@T)        + full ocean    perfect values, full coverage
#   D future   deseas(GLORYS@T+LEAD) + full ocean    library ceiling (uses the truth)
#
#   B - A  representation gap : the CEILING on any learned SWOT->GLORYS distance --
#                               it cannot beat having the model's own true values.
#   C - B  coverage gap       : unreachable by ANY distance (missing data).
#   D - C  forecast-relevance : headroom for a distance that sees only the CURRENT
#                               field but ranks by FUTURE similarity -> the ML target.
#
# Gaps are reported as paired per-day differences with a bootstrap CI, because the
# usable sample is small (a couple of dozen obs days clear the front-coverage filter).
# Companions: metric_diagnostic.py (the 5-rung ladder), main_swot_glorys.py.
import glob
import os
import re

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import swot_data
import swot_analog as sa
from analog import ModelLibrary, AnalogSelector, Forecast, seasonal_climatology
from distances import CorrelationDistance
from fronts import LEVEL, loop_current_front

# %% ---- CONFIG ----------------------------------------------------------- #
LIB_CANDIDATES = ["data/glorys_gom_zos_2004_2013.nc", "data/glorys_gom_zos_2004.nc"]
TRUTH_PATH = "data/glorys_gom_zos_2024.nc"
SWOT_DIR = "data/swot"
K, MIN_SEP, LEAD = 10, 14, 14
MIN_CELLS = 500                  # min observed SWOT cells to use an obs day
FRONT_COVER_MIN = 0.20           # min fraction of the Loop Current front observed
N_BOOT = 20000
# -------------------------------------------------------------------------- #

lib_path = next((p for p in LIB_CANDIDATES if os.path.exists(p)), LIB_CANDIDATES[-1])
ds = xr.open_dataset(lib_path)
times, state = ds["time"].values, ds["zos"].values
lib = ModelLibrary(ds, var="zos")
lib_ds = ModelLibrary(ds, var="zos",                       # mesoscale selection space
                      anomaly=state - seasonal_climatology(state, times))
selector = AnalogSelector(lib, CorrelationDistance())
selector_ds = AnalogSelector(lib_ds, CorrelationDistance())
forecaster = Forecast(lib)
print(f"Library {lib_path}: {ds.sizes['time']} states")

truth_ds = xr.open_dataset(TRUTH_PATH)
truth_times, truth_state = truth_ds["time"].values, truth_ds["zos"].values
assert np.allclose(truth_ds["longitude"].values, lib.lon) and \
    np.allclose(truth_ds["latitude"].values, lib.lat), "truth grid != library grid"
tmin, tmax = truth_times.min(), truth_times.max()
print(f"Truth   {TRUTH_PATH}: {truth_ds.sizes['time']} steps, "
      f"{str(tmin)[:10]} .. {str(tmax)[:10]}")


def field_on(date64):
    """Nearest daily GLORYS-2024 surface field to a calendar date."""
    return truth_state[int(np.argmin(np.abs(truth_times - np.datetime64(date64))))]


# %% Per obs day: run the four rungs, keeping PER-DAY scores for paired statistics.
ref_mean = float(np.nanmean(lib.mean_surf[lib.ocean]))
swot_days = sorted({re.search(r"_(\d{8})T", p).group(1)
                    for p in glob.glob(os.path.join(SWOT_DIR, "*.nc"))})
obs_days_all = [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in swot_days]

RUNGS = ["A swot", "B perfect", "C dense", "D future"]
acc = {v: [] for v in RUNGS}          # best-analog full-field mesoscale ACC
lcm = {v: [] for v in RUNGS}          # best-analog Loop Current front MHD (km)
pacc, plc = [], []                    # dense persistence baselines
overlap, tied = [], []                # selection degeneracy diagnostics
obs_days = []

for T in obs_days_all:
    vday = np.datetime64(T) + np.timedelta64(LEAD, "D")
    if not (tmin <= vday <= tmax):
        continue                                       # no dense truth at obs+LEAD
    obs_grid, mask = sa.swath_to_grid(swot_data.swaths_for_dates([T])[0],
                                      lib.lon, lib.lat)
    if mask.sum() < MIN_CELLS:
        continue                                       # too little SWOT coverage
    gT, gV = field_on(T), field_on(vday)
    lc_front = loop_current_front(gT, lib.lon, lib.lat, lib.ocean, ref_mean, level=LEVEL)
    if sa.front_coverage(lc_front, mask, lib.lon, lib.lat) < FRONT_COVER_MIN:
        continue                                       # Loop Current not observed

    def rung(sel_, obs_, mask_, lead=LEAD):
        """select -> forecast -> score against the dense truth. Returns (AnalogSet, dict)."""
        s = sel_.select(obs_, mask_, lead, K, min_sep=MIN_SEP)
        e = forecaster.forecast(s, lead)
        return s, sa.evaluate(lib, s, e, gV, vday, persist_surf=gT, lead=lead)

    gT_anom = lib.anom_of(gT)                          # same space as lib.anom
    gV_ds = lib.deseasonalize(lib.anom_of(gV), vday)   # mesoscale truth @T+LEAD

    picks = {}
    for name, (sel_, obs_, mask_, lead) in {
            "A swot":    (selector,    obs_grid, mask,        LEAD),
            "B perfect": (selector,    gT_anom,  mask,        LEAD),
            "C dense":   (selector,    gT_anom,  lib.ocean,   LEAD),
            "D future":  (selector_ds, gV_ds,    lib_ds.ocean, 0)}.items():
        s, r = rung(sel_, obs_, mask_, lead)
        picks[name] = set(s.indices.tolist())
        acc[name].append(np.nanmax(r["acc"]))
        lcm[name].append(np.nanmin(r["lc_mhd"]))
    pacc.append(r["acc_persist"])
    plc.append(r["lc_mhd_persist"])

    # Degeneracy: does swapping SWOT for perfect obs even change WHICH analogs win,
    # and how tied is the top of the ranking?
    overlap.append(len(picks["A swot"] & picks["B perfect"]) / K)
    d = selector.distance.distance(obs_grid, mask)
    d = np.sort(d[np.isfinite(d)])
    tied.append((d[K - 1] - d[0]) / (np.median(d) - d[0]))
    obs_days.append(T)

n = len(obs_days)
print(f"\nUsed {n} obs days with the Loop Current observed "
      f"(K={K}, min_sep={MIN_SEP}, lead={LEAD}).\n")

# %% Rung table.
hdr = f"{'rung':24s} {'ACC best':>9s}  {'LC best km':>10s}"
print(hdr); print("-" * len(hdr))
for v in RUNGS:
    print(f"{v:24s} {np.nanmean(acc[v]):+9.3f}  {np.nanmean(lcm[v]):10.1f}")
print(f"{'persistence (dense)':24s} {np.nanmean(pacc):+9.3f}  {np.nanmean(plc):10.1f}")
print("  (ACC higher = better; Loop Current MHD km lower = better)")


# %% Paired gaps with bootstrap CIs.
def paired(a, b, label, higher_better, unit="", seed=0):
    """Bootstrap CI + win count on the paired difference b - a."""
    d = np.asarray(b, float) - np.asarray(a, float)
    d = d[np.isfinite(d)]
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, d.size, (N_BOOT, d.size))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    wins = int((d > 0).sum()) if higher_better else int((d < 0).sum())
    sig = "" if lo <= 0 <= hi else "  *"
    print(f"  {label:26s} {d.mean():+7.3f}{unit}  [{lo:+.3f}, {hi:+.3f}]  "
          f"better on {wins:2d}/{d.size}{sig}")
    return d.mean(), lo, hi


print("\nPAIRED GAPS (best analog; * = 95% CI excludes 0)")
print(" full-field mesoscale ACC (higher = better)")
paired(acc["A swot"], acc["B perfect"], "representation  B - A", True)
paired(acc["B perfect"], acc["C dense"], "coverage        C - B", True)
paired(acc["C dense"], acc["D future"], "forecast-relev. D - C", True)
paired(pacc, acc["A swot"], "A - persistence", True)
print(" Loop Current front MHD, km (lower = better)")
paired(lcm["A swot"], lcm["B perfect"], "representation  B - A", False)
paired(lcm["B perfect"], lcm["C dense"], "coverage        C - B", False)
paired(lcm["C dense"], lcm["D future"], "forecast-relev. D - C", False)
paired(plc, lcm["A swot"], "A - persistence", False)

print(f"\nSELECTION DEGENERACY")
print(f"  top-{K} shared between real SWOT and perfect obs: {np.mean(overlap):.0%}")
print(f"  top-{K} distance span / span to library median : {np.mean(tied):.1%}")
print("  (few shared picks + a nearly tied top => many library states are\n"
      "   interchangeably good, so fidelity of the comparison buys little;\n"
      "   breaking the tie in a FORECAST-RELEVANT way is the D - C target.)")

# %% Figure.
viz_rows = {v: dict(acc_best=float(np.nanmean(acc[v])), lc_best=float(np.nanmean(lcm[v])))
            for v in RUNGS}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, key, lab, base in [
        (axes[0], "acc_best", "+%dd full-field mesoscale ACC  (higher = better)" % LEAD,
         float(np.nanmean(pacc))),
        (axes[1], "lc_best", "+%dd Loop Current front MHD (km)  (lower = better)" % LEAD,
         float(np.nanmean(plc)))]:
    ax.plot(RUNGS, [viz_rows[v][key] for v in RUNGS], "o-", color="tab:blue",
            label="best analog")
    ax.axhline(base, ls=":", color="k", label="persistence")
    ax.set_ylabel(lab)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3)
    ax.legend()
fig.suptitle("Where SWOT-analog skill is lost: representation (B-A) vs coverage (C-B) "
             "vs forecast-relevance (D-C)")
fig.tight_layout()
os.makedirs("plots", exist_ok=True)
fig.savefig("plots/obs_gap.png", dpi=140)
print("\nSaved plots/obs_gap.png")

plt.show()

# %%
