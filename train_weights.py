# %% Train the learned spatial weight map for analog selection (see latent.py).
# distance = 1 - corr_w(current field), with w fitted so the CURRENT-field distance
# predicts FUTURE dissimilarity 1 - corr(X[t0+LEAD], X[a+LEAD]) on deseasonalized
# fields -- the D - C gap in obs_gap.py.
#
# w starts at cos(lat), i.e. exactly the deployed CorrelationDistance, so training
# can only move away from the baseline where the data pays for it. The fitted map is
# the interpretable output: where the current surface field determines the 14-day future.
import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import latent as lat
from analog import ModelLibrary, seasonal_climatology

# %% ---- CONFIG ----------------------------------------------------------- #
LIB_CANDIDATES = ["data/glorys_gom_zos_2004_2013.nc", "data/glorys_gom_zos_2004.nc"]
OUT_PATH = "cache/learned_weights_L14.npz"
LEAD = 14
STEPS, BATCH, LR = 1500, 64, 3e-2
EXCLUDE = 30                  # drop pairs within this many days (same-event leakage)
TAU = 0.35                    # emphasis on similar pairs (np.inf = unweighted)
SMOOTH = 1e-3                 # total-variation penalty on log-weights
VAL_FRAC = 0.15
SEED = 0
# -------------------------------------------------------------------------- #

lib_path = next((p for p in LIB_CANDIDATES if os.path.exists(p)), LIB_CANDIDATES[-1])
ds = xr.open_dataset(lib_path)
times, state = ds["time"].values, ds["zos"].values
lib = ModelLibrary(ds, var="zos")
print(f"Library {lib_path}: {ds.sizes['time']} states, grid {lib.anom.shape[1:]}")

print("Building the future-similarity matrix (deseasonalized) ...")
S = lat.future_similarity(state - seasonal_climatology(state, times), lib.lat, lib.ocean)

print(f"Fitting weights (lead={LEAD}, steps={STEPS}, batch={BATCH}, tau={TAU}, "
      f"smooth={SMOOTH}) ...")
w2d, meta = lat.train_weights(lib.anom, S, lib.ocean, lib.lat, LEAD, steps=STEPS,
                              batch=BATCH, lr=LR, exclude=EXCLUDE, tau=TAU,
                              smooth=SMOOTH, val_frac=VAL_FRAC, seed=SEED)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
np.savez(OUT_PATH, w2d=w2d, lead=LEAD, smooth=SMOOTH, tau=TAU,
         baseline=meta["baseline"])
print(f"Saved {OUT_PATH}")

# %% The fitted map, as a ratio to the cos(lat) baseline it started from.
base = meta["baseline"]
ratio = np.where(lib.ocean, w2d / np.where(base > 0, base, np.nan), np.nan)
h = np.array(meta["history"])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
im = axes[0].pcolormesh(lib.lon, lib.lat, ratio, cmap="RdBu_r",
                        vmin=np.nanpercentile(ratio, 2), vmax=np.nanpercentile(ratio, 98))
fig.colorbar(im, ax=axes[0], label="learned weight / cos(lat) baseline")
axes[0].set_title(f"Where the current field decides the +{LEAD}d future")
axes[0].set_xlabel("longitude"); axes[0].set_ylabel("latitude")

axes[1].plot(h[:, 0], h[:, 1], label="train")
axes[1].plot(h[:, 0], h[:, 2], label="val (held out in time)")
axes[1].set_xlabel("step"); axes[1].set_ylabel("weighted L1 + TV")
axes[1].set_title("Fit"); axes[1].grid(alpha=0.3); axes[1].legend()
fig.tight_layout()
os.makedirs("plots", exist_ok=True)
fig.savefig("plots/learned_weights.png", dpi=140)
print("Saved plots/learned_weights.png")
print(f"weight ratio: min {np.nanmin(ratio):.2f}  median {np.nanmedian(ratio):.2f}  "
      f"max {np.nanmax(ratio):.2f}")

plt.show()

# %%
