# %% Does the learned distance actually recover the D - C gap?
# Two independent questions, in order of directness:
#
#   1. INTRINSIC -- on library states held out in time, does latent distance rank
#      candidates by future similarity better than correlation does? This is what the
#      encoder was trained for and needs no forecast pipeline.
#   2. EXTRINSIC -- swap CorrelationDistance for LatentDistance in the real
#      SWOT->GLORYS pipeline and re-run the obs_gap.py rungs. Paired CIs, same days.
#
# A learned metric can win (1) and still not move (2): obs_gap.py shows the top of
# the ranking is nearly tied, so a better ordering only pays off if it reorders the
# analogs whose FUTURES differ. Reporting both keeps that distinction visible.
import glob
import os
import re

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import swot_data
import swot_analog as sa
import latent as lat
from analog import ModelLibrary, AnalogSelector, Forecast, seasonal_climatology
from distances import CorrelationDistance, LatentDistance, LearnedWeightDistance
from fronts import LEVEL, loop_current_front

# %% ---- CONFIG ----------------------------------------------------------- #
LIB_CANDIDATES = ["data/glorys_gom_zos_2004_2013.nc", "data/glorys_gom_zos_2004.nc"]
TRUTH_PATH = "data/glorys_gom_zos_2024.nc"
ENC_PATH = "cache/latent_encoder_L14.eqx"
W_PATH = "cache/learned_weights_L14.npz"
SWOT_DIR = "data/swot"
K, MIN_SEP, LEAD = 10, 14, 14
MIN_CELLS, FRONT_COVER_MIN = 500, 0.20
VAL_FRAC = 0.15               # must match train_latent.py
N_BOOT = 20000
# -------------------------------------------------------------------------- #

lib_path = next((p for p in LIB_CANDIDATES if os.path.exists(p)), LIB_CANDIDATES[-1])
ds = xr.open_dataset(lib_path)
times, state = ds["time"].values, ds["zos"].values
lib = ModelLibrary(ds, var="zos")
lib_ds = ModelLibrary(ds, var="zos", anomaly=state - seasonal_climatology(state, times))
forecaster = Forecast(lib)

corr_sel = AnalogSelector(lib, CorrelationDistance())
lat_dist = LatentDistance(path=ENC_PATH)
lat_sel = AnalogSelector(lib, lat_dist)
w_dist = LearnedWeightDistance(path=W_PATH)
w_sel = AnalogSelector(lib, w_dist)
oracle_sel = AnalogSelector(lib_ds, CorrelationDistance())
print(f"Library {lib_path}: {ds.sizes['time']} states")
print(f"  latent encoder {ENC_PATH}\n  learned weights {W_PATH}")


# %% ---------------- 1. INTRINSIC: ranking quality, held out in time --------- #
def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


deseas = state - seasonal_climatology(state, times)
S = lat.future_similarity(deseas, lib.lat, lib.ocean)
n = lib.n
usable = n - LEAD
val_lo = int(usable * (1 - VAL_FRAC))          # states the encoder never trained on
Zn = lat_dist.latents / (np.linalg.norm(lat_dist.latents, axis=1, keepdims=True) + 1e-8)

METHODS = ["correlation", "latent", "learned-w"]
rng = np.random.default_rng(0)
probe = rng.choice(np.arange(val_lo, usable), size=120, replace=False)
rows = {m: [] for m in METHODS}
top_rows = {m: [] for m in METHODS}
l1 = {m: [] for m in METHODS}
TAU = 0.35                                   # must match the training emphasis
for t0 in probe:
    cand = np.arange(usable)
    cand = cand[np.abs(cand - t0) > 30]
    target = 1.0 - S[t0 + LEAD, cand + LEAD]                    # future dissimilarity
    d = {"correlation": corr_sel.distance.distance(lib.anom[t0], lib.ocean)[cand],
         "latent": 1.0 - Zn[cand] @ Zn[t0],
         "learned-w": w_sel.distance.distance(lib.anom[t0], lib.ocean)[cand]}
    wt = np.exp(-target / TAU)
    for m in METHODS:
        rows[m].append(spearman(d[m], target))
        top_rows[m].append(float(np.mean(target[np.argsort(d[m])[:K]])))
        l1[m].append(float((wt * np.abs(d[m] - target)).sum() / wt.sum()))

print(f"\n1. INTRINSIC ranking quality (120 states held out in time, lead {LEAD}):")
print(f"  {'':26s}" + "".join(f"{m:>14s}" for m in METHODS))
print(f"  {'weighted L1 to target':26s}" + "".join(f"{np.mean(l1[m]):14.3f}" for m in METHODS)
      + "   (lower = better; this is the training objective)")
print(f"  {'Spearman(d, future diss.)':26s}" + "".join(f"{np.mean(rows[m]):+14.3f}" for m in METHODS))
print(f"  {f'future diss. of top-{K}':26s}" + "".join(f"{np.mean(top_rows[m]):14.3f}" for m in METHODS)
      + "   (lower = better)")


# %% ---------------- 2. EXTRINSIC: the real SWOT->GLORYS pipeline ------------ #
truth_ds = xr.open_dataset(TRUTH_PATH)
truth_times, truth_state = truth_ds["time"].values, truth_ds["zos"].values
tmin, tmax = truth_times.min(), truth_times.max()
field_on = lambda d: truth_state[int(np.argmin(np.abs(truth_times - np.datetime64(d))))]
ref_mean = float(np.nanmean(lib.mean_surf[lib.ocean]))

swot_days = sorted({re.search(r"_(\d{8})T", p).group(1)
                    for p in glob.glob(os.path.join(SWOT_DIR, "*.nc"))})
obs_days_all = [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in swot_days]

RUNGS = ["A swot (corr)", "A swot (latent)", "A swot (learned-w)",
         "C dense (corr)", "C dense (latent)", "C dense (learned-w)",
         "D future-oracle"]
acc = {v: [] for v in RUNGS}
lcm = {v: [] for v in RUNGS}
pacc, plc, obs_days = [], [], []

for T in obs_days_all:
    vday = np.datetime64(T) + np.timedelta64(LEAD, "D")
    if not (tmin <= vday <= tmax):
        continue
    obs_grid, mask = sa.swath_to_grid(swot_data.swaths_for_dates([T])[0],
                                      lib.lon, lib.lat)
    if mask.sum() < MIN_CELLS:
        continue
    gT, gV = field_on(T), field_on(vday)
    lcf = loop_current_front(gT, lib.lon, lib.lat, lib.ocean, ref_mean, level=LEVEL)
    if sa.front_coverage(lcf, mask, lib.lon, lib.lat) < FRONT_COVER_MIN:
        continue

    gT_anom = lib.anom_of(gT)
    gV_ds = lib.deseasonalize(lib.anom_of(gV), vday)

    for name, (sel_, obs_, mask_, lead) in {
            "A swot (corr)":       (corr_sel,   obs_grid, mask,         LEAD),
            "A swot (latent)":     (lat_sel,    obs_grid, mask,         LEAD),
            "A swot (learned-w)":  (w_sel,      obs_grid, mask,         LEAD),
            "C dense (corr)":      (corr_sel,   gT_anom,  lib.ocean,    LEAD),
            "C dense (latent)":    (lat_sel,    gT_anom,  lib.ocean,    LEAD),
            "C dense (learned-w)": (w_sel,      gT_anom,  lib.ocean,    LEAD),
            "D future-oracle":     (oracle_sel, gV_ds,    lib_ds.ocean, 0)}.items():
        s = sel_.select(obs_, mask_, lead, K, min_sep=MIN_SEP)
        e = forecaster.forecast(s, lead)
        r = sa.evaluate(lib, s, e, gV, vday, persist_surf=gT, lead=lead)
        acc[name].append(np.nanmax(r["acc"]))
        lcm[name].append(np.nanmin(r["lc_mhd"]))
    pacc.append(r["acc_persist"])
    plc.append(r["lc_mhd_persist"])
    obs_days.append(T)

print(f"\n2. EXTRINSIC skill on {len(obs_days)} SWOT obs days "
      f"(K={K}, min_sep={MIN_SEP}, lead={LEAD}):\n")
hdr = f"{'rung':22s} {'ACC best':>9s}  {'LC best km':>10s}"
print(hdr); print("-" * len(hdr))
for v in RUNGS:
    print(f"{v:22s} {np.nanmean(acc[v]):+9.3f}  {np.nanmean(lcm[v]):10.1f}")
print(f"{'persistence (dense)':22s} {np.nanmean(pacc):+9.3f}  {np.nanmean(plc):10.1f}")
print("  (ACC higher = better; Loop Current MHD km lower = better)")


def paired(a, b, label, higher_better, seed=0):
    d = np.asarray(b, float) - np.asarray(a, float)
    d = d[np.isfinite(d)]
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, d.size, (N_BOOT, d.size))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    wins = int((d > 0).sum()) if higher_better else int((d < 0).sum())
    sig = "" if lo <= 0 <= hi else "  *"
    print(f"  {label:30s} {d.mean():+7.3f}  [{lo:+.3f}, {hi:+.3f}]  "
          f"better on {wins:2d}/{d.size}{sig}")


print("\nPAIRED  learned - correlation, same days (* = 95% CI excludes 0)")
for learner in ["latent", "learned-w"]:
    print(f" [{learner}]  full-field mesoscale ACC (higher = better)")
    paired(acc["A swot (corr)"], acc[f"A swot ({learner})"], "SWOT swath  (rung A)", True)
    paired(acc["C dense (corr)"], acc[f"C dense ({learner})"], "dense field (rung C)", True)
    print(f" [{learner}]  Loop Current front MHD, km (lower = better)")
    paired(lcm["A swot (corr)"], lcm[f"A swot ({learner})"], "SWOT swath  (rung A)", False)
    paired(lcm["C dense (corr)"], lcm[f"C dense ({learner})"], "dense field (rung C)", False)

gapC = np.nanmean(acc["D future-oracle"]) - np.nanmean(acc["C dense (corr)"])
print(f"\nFraction of the D - C headroom ({gapC:+.3f} ACC) recovered on dense fields:")
for learner in ["latent", "learned-w"]:
    got = np.nanmean(acc[f"C dense ({learner})"]) - np.nanmean(acc["C dense (corr)"])
    print(f"  {learner:12s} {got / gapC:+6.0%}   ({got:+.3f} ACC)")

# %% Figure.
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, tbl, lab, base in [
        (axes[0], acc, f"+{LEAD}d full-field mesoscale ACC (higher = better)",
         float(np.nanmean(pacc))),
        (axes[1], lcm, f"+{LEAD}d Loop Current front MHD km (lower = better)",
         float(np.nanmean(plc)))]:
    vals = [np.nanmean(tbl[v]) for v in RUNGS]
    cols = ["tab:blue", "tab:orange", "tab:red"] * 2 + ["tab:green"]
    ax.bar(range(len(RUNGS)), vals, color=cols)
    ax.set_xticks(range(len(RUNGS)))
    ax.set_xticklabels(RUNGS, rotation=20, ha="right")
    ax.axhline(base, ls=":", color="k", label="persistence")
    ax.set_ylabel(lab); ax.grid(alpha=0.3, axis="y"); ax.legend()
fig.suptitle("Learned latent distance vs correlation, on the D - C (forecast-relevance) gap")
fig.tight_layout()
os.makedirs("plots", exist_ok=True)
fig.savefig("plots/latent_eval.png", dpi=140)
print("\nSaved plots/latent_eval.png")

plt.show()

# %%
