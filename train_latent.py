# %% Train the forecast-relevant latent analog distance (see latent.py).
# The encoder sees the CURRENT surface anomaly and is trained so that latent
# distance reproduces FUTURE dissimilarity, 1 - corr(X[t0+LEAD], X[a+LEAD]), on
# deseasonalized (mesoscale) fields. That is the D - C gap in obs_gap.py -- the only
# gap there that is both significant and reachable from the current field.
#
# The encoder sees `lib.anom` (the same selection space the deployed
# CorrelationDistance uses, so LatentDistance is a drop-in at the call site), while
# the TARGET is built from deseasonalized futures (the space the oracle is defined
# in). Input space and target space are deliberately decoupled.
#
# Evaluate with eval_latent.py.
import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import latent as lat
from analog import ModelLibrary, seasonal_climatology

# %% ---- CONFIG ----------------------------------------------------------- #
LIB_CANDIDATES = ["data/glorys_gom_zos_2004_2013.nc", "data/glorys_gom_zos_2004.nc"]
OUT_PATH = "cache/latent_encoder_L14.eqx"
LEAD = 14                     # the lead the distance is trained for
LATENT_DIM = 64
STEPS = 4000
BATCH = 48                    # M states per step -> M^2 supervised pairs
LR = 3e-4
EXCLUDE = 30                  # drop pairs within this many days (same-event leakage)
TAU = 0.35                    # emphasis on similar pairs (np.inf = unweighted L1)
P_FULL = 0.35                 # fraction of samples at full ocean coverage
VAL_FRAC = 0.15               # held out in time (the last 15% of the library)
SEED = 0
# -------------------------------------------------------------------------- #

lib_path = next((p for p in LIB_CANDIDATES if os.path.exists(p)), LIB_CANDIDATES[-1])
ds = xr.open_dataset(lib_path)
times, state = ds["time"].values, ds["zos"].values
lib = ModelLibrary(ds, var="zos")
print(f"Library {lib_path}: {ds.sizes['time']} states, grid {lib.anom.shape[1:]}")

# %% Supervision: future similarity on deseasonalized (mesoscale) anomalies.
print("Building the future-similarity matrix (deseasonalized) ...")
deseas = state - seasonal_climatology(state, times)
S = lat.future_similarity(deseas, lib.lat, lib.ocean)
print(f"  S {S.shape}  mean off-diagonal similarity {np.mean(S[~np.eye(len(S), dtype=bool)]):+.3f}")

# %% Train.
print(f"Training (lead={LEAD}, dim={LATENT_DIM}, steps={STEPS}, batch={BATCH}, "
      f"tau={TAU}, p_full={P_FULL}) ...")
model, meta = lat.train_encoder(
    lib.anom, S, lib.ocean, LEAD, latent_dim=LATENT_DIM, steps=STEPS, batch=BATCH,
    lr=LR, exclude=EXCLUDE, tau=TAU, p_full=P_FULL, val_frac=VAL_FRAC, seed=SEED)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
lat.save(OUT_PATH, model, meta)
print(f"Saved {OUT_PATH}  (scale={meta['scale']:.4f})")

# %% Learning curve.
h = np.array(meta["history"])
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(h[:, 0], h[:, 1], label="train")
ax.plot(h[:, 0], h[:, 2], label="val (held out in time)")
ax.set_xlabel("step"); ax.set_ylabel("weighted L1  |d_latent - (1 - S_future)|")
ax.set_title(f"Latent analog distance, lead {LEAD}")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()
os.makedirs("plots", exist_ok=True)
fig.savefig("plots/latent_training.png", dpi=140)
print("Saved plots/latent_training.png")

plt.show()

# %%
