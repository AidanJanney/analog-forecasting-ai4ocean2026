# %% Analog skill-ceiling (oracle) diagnostic.
# Gate before building a learned ("ML") distance metric: does the analog library
# even contain futures that beat persistence, and does the current front-MHD
# metric leave skill on the table? See oracle.py for the four strategies.
import os

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import viz
from metrics import FrontMHD
from analog import build_matrix
from oracle import ceiling_curves

# %% ---- CONFIG ----------------------------------------------------------- #
DATA_CANDIDATES = [
    # "data/glorys_gom_zos_2004_2013.nc",
    # "data/glorys_gom_zos_2004.nc",
    "data/glorys_gom_zos_thetao_all.nc"
]
K, EXCLUDE = 10, 30                         # analogs kept; temporal exclusion (days)
LEADS = list(range(1, 16))
STRIDE = 20                                 # subsample targets for speed
# -------------------------------------------------------------------------- #

path = next((p for p in DATA_CANDIDATES if os.path.exists(p)), DATA_CANDIDATES[-1])
ds = xr.open_dataset(path)
times = ds["time"].values
state = ds["zos"].values
print(f"Loaded {path}: {ds.sizes['time']} states")

# %% Front-MHD distance matrix (from the existing cache; no re-extraction).
D = build_matrix(FrontMHD(), ds, source=path)

# %% Four skill-vs-lead curves and the verdict.
leads, curves = ceiling_curves(state, times, ds["latitude"].values, D,
                               LEADS, k=K, exclude=EXCLUDE, stride=STRIDE)
viz.plot_ceiling(leads, curves)

means = {nm: float(np.nanmean(v)) for nm, v in curves.items()}
for nm, m in means.items():
    print(f"  {nm:15s} mean ACC = {m:+.3f}")
print("\nVERDICT:")
gap_fo = means["future-oracle"] - means["persistence"]
gap_me = means["present-oracle"] - means["metric-analog"]
v_fo = "headroom" if gap_fo > 0.03 else "NO headroom (predictability ceiling)"
v_me = ("a better current metric could help → ML worth trying" if gap_me > 0.03
        else "front-MHD already near the current-similarity ceiling")
print(f"  future-oracle − persistence   = {gap_fo:+.3f}  ({v_fo})")
print(f"  present-oracle − metric-analog = {gap_me:+.3f}  ({v_me})")

plt.show()

# %%
