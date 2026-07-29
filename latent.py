"""A learned, forecast-relevant analog distance (Equinox/JAX).

This targets the one gap the diagnostics say is both real and reachable (see
``obs_gap.py``): a distance that sees only the **current** surface field but ranks
library states by their **future** similarity.

    representation  B - A  = +0.019 ACC  [-0.014, +0.052]   not significant
    coverage        C - B  = +0.058 ACC  [+0.025, +0.094]   no distance can fix
    forecast-relev. D - C  = +0.055 ACC  [+0.033, +0.078]   <- what this targets

Correlation of the current field is near-degenerate at the top of the ranking (the
top-10 distances span ~7% of the way to the library median), so many library states
look interchangeably good. The encoder's job is to break that tie in the direction
that matters: states whose *futures* agree should land close together *now*.

Design
------
``d_latent(x, y) = 1 - cos(f(x), f(y))``  with ``f`` a small CNN, trained so that

    d_latent(t0, a)  ~=  1 - S_future(t0, a) = 1 - corr(X[t0+lead], X[a+lead])

over library pairs (the design already written into ``distances.LatentDistance``).
Both sides live in [0, 2], so the regression is well posed. Pairs are supervised
*within a batch*: encoding M states yields M^2 supervised pairs, so the M^2 target
block is free once the similarity matrix is precomputed.

Two properties make it usable from a partial SWOT swath:

* the encoder takes **two channels** — the masked field and the mask itself — so it
  always knows what it was allowed to see;
* training applies random **swath-shaped masks**, so a partial view of a state embeds
  near the full-field embedding of that same state. Library states are encoded at full
  ocean coverage once, in ``prepare``; only the query carries a swath mask.

Selection stays surface/observable: only ``library.anom`` is encoded. The multi-depth
forecast target is untouched (see the surface-only-analog-selection note).
"""

import pickle

import numpy as np

import equinox as eqx
import jax
import jax.numpy as jnp
import optax


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
class Encoder(eqx.Module):
    """(2, nlat, nlon) -> (latent_dim,).  Channels are [masked field, mask]."""

    convs: list
    head: eqx.nn.Linear
    latent_dim: int = eqx.field(static=True)

    def __init__(self, key, latent_dim=64, widths=(16, 32, 64, 64)):
        keys = jax.random.split(key, len(widths) + 1)
        chans = (2,) + tuple(widths)
        self.convs = [
            eqx.nn.Conv2d(chans[i], chans[i + 1], 3, stride=2, padding=1, key=keys[i])
            for i in range(len(widths))
        ]
        self.head = eqx.nn.Linear(widths[-1], latent_dim, key=keys[-1])
        self.latent_dim = latent_dim

    def __call__(self, x):
        for conv in self.convs:
            x = jax.nn.gelu(conv(x))
        return self.head(jnp.mean(x, axis=(1, 2)))       # global average pool


def _unit(z, eps=1e-8):
    return z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + eps)


def latent_distance(za, zb):
    """1 - cosine similarity, in [0, 2] — the same range as 1 - correlation."""
    return 1.0 - _unit(za) @ _unit(zb).T


# --------------------------------------------------------------------------- #
# Inputs and supervision
# --------------------------------------------------------------------------- #
def encoder_inputs(field, mask, scale):
    """Stack a field and its coverage mask into the encoder's 2 channels.

    NaNs (land, unobserved) are zeroed in the value channel; the mask channel keeps
    the distinction between "zero anomaly" and "not seen".
    """
    m = np.asarray(mask, dtype=bool) & np.isfinite(field)
    v = np.where(m, np.nan_to_num(np.asarray(field, dtype=np.float32)), 0.0) / scale
    return np.stack([v.astype(np.float32), m.astype(np.float32)])


def swath_masks(ocean, rng, n, width_frac=(0.06, 0.22)):
    """Random diagonal-band masks approximating SWOT swath coverage over the box.

    A band of random angle, offset and width, intersected with the ocean mask. This
    is deliberately *synthetic* — SWOT's repeating orbit makes the real footprint
    distribution known a priori, but training on bands rather than the 2024 passes
    keeps the encoder from being tuned to the exact evaluation geometry.
    """
    nlat, nlon = ocean.shape
    yy, xx = np.mgrid[0:nlat, 0:nlon]
    out = np.empty((n, nlat, nlon), dtype=bool)
    diag = np.hypot(nlat, nlon)
    for i in range(n):
        th = rng.uniform(0, np.pi)
        proj = xx * np.cos(th) + yy * np.sin(th)
        half = rng.uniform(*width_frac) * diag / 2
        centre = rng.uniform(proj.min(), proj.max())
        out[i] = (np.abs(proj - centre) < half) & ocean
    return out


def future_similarity(anom_future, lat, ocean):
    """Weighted spatial correlation between every pair of (deseasonalized) states.

    ``S[i, j]`` is the similarity of state i to state j; the training target for a
    pair ``(t0, a)`` at lead L is ``1 - S[t0 + L, a + L]``.
    """
    A = np.asarray(anom_future, dtype=np.float32)
    X = A[:, ocean]
    w = np.broadcast_to(np.cos(np.deg2rad(lat)).astype(np.float32)[:, None],
                        A.shape[1:])[ocean]
    W = w.sum()
    mu = (X * w).sum(1, keepdims=True) / W
    Xc = (X - mu) * np.sqrt(w)
    norm = np.sqrt((Xc ** 2).sum(1, keepdims=True))
    Xhat = np.divide(Xc, norm, out=np.zeros_like(Xc), where=norm > 0)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return (Xhat @ Xhat.T).astype(np.float32)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_encoder(anom, S, ocean, lead, *, latent_dim=64, steps=3000, batch=48,
                  lr=3e-4, exclude=30, tau=0.35, p_full=0.35, seed=0, scale=None,
                  val_frac=0.15, log_every=250, verbose=True):
    """Fit an :class:`Encoder` so latent distance now predicts future dissimilarity.

    Parameters
    ----------
    anom : (n, nlat, nlon) selection-space anomalies (what the encoder sees).
    S    : (n, n) future-similarity matrix from :func:`future_similarity`.
    lead : forecast lead the distance is trained for (the target uses S[i+lead, j+lead]).
    exclude : pairs closer than this many days are dropped (same-event leakage).
    tau  : emphasis on *similar* pairs, ``w = exp(-target / tau)``. Selection only
           reads the top of the ranking, but a plain L1 fit is dominated by the many
           uninformative near-orthogonal pairs; this reweights toward the pairs that
           decide the top-K. Set ``tau=np.inf`` for the unweighted fit.
    p_full : fraction of samples encoded at full ocean coverage (the rest get a
           random swath band), so one encoder serves both dense and swath queries.
    val_frac : final fraction of the library held out in time for validation.

    Returns ``(encoder, history)``.
    """
    n = anom.shape[0]
    scale = float(np.nanstd(anom[np.isfinite(anom)])) if scale is None else scale
    usable = n - lead                                  # need i+lead to exist
    n_train = int(usable * (1 - val_frac))
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    key, ksub = jax.random.split(key)
    model = Encoder(ksub, latent_dim=latent_dim)
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    # Pre-stack the value channel once; masks are applied per sample.
    V = np.where(np.isfinite(anom), np.nan_to_num(anom), 0.0).astype(np.float32) / scale

    def sample(lo, hi, rg):
        """A batch of states from [lo, hi) plus its M x M target block and weights."""
        idx = rg.choice(np.arange(lo, hi), size=batch, replace=False)
        masks = swath_masks(ocean, rg, batch)
        full = rg.random(batch) < p_full
        masks[full] = ocean
        x = np.stack([V[idx] * masks, masks.astype(np.float32)], axis=1)
        tgt = 1.0 - S[np.ix_(idx + lead, idx + lead)]
        keep = np.abs(idx[:, None] - idx[None, :]) > exclude   # drop same-event pairs
        w = np.exp(-tgt / tau) if np.isfinite(tau) else np.ones_like(tgt)
        return (jnp.asarray(x), jnp.asarray(tgt), jnp.asarray(w * keep))

    def loss_fn(m, x, tgt, w):
        z = eqx.filter_vmap(m)(x)
        d = latent_distance(z, z)
        return jnp.sum(w * jnp.abs(d - tgt)) / (jnp.sum(w) + 1e-8)

    @eqx.filter_jit
    def step(m, st, x, tgt, w):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(m, x, tgt, w)
        updates, st = opt.update(grads, st, eqx.filter(m, eqx.is_inexact_array))
        return eqx.apply_updates(m, updates), st, loss

    @eqx.filter_jit
    def evaluate(m, x, tgt, w):
        return loss_fn(m, x, tgt, w)

    history = []
    vrng = np.random.default_rng(seed + 1)
    for i in range(1, steps + 1):
        model, opt_state, loss = step(model, opt_state, *sample(0, n_train, rng))
        if i % log_every == 0 or i == steps:
            vl = float(np.mean([float(evaluate(model, *sample(n_train, usable, vrng)))
                                for _ in range(8)]))
            history.append((i, float(loss), vl))
            if verbose:
                print(f"  step {i:5d}  train {float(loss):.4f}  val {vl:.4f}")
    return model, dict(history=history, scale=scale, lead=lead, latent_dim=latent_dim)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save(path, model, meta):
    with open(path, "wb") as fh:
        pickle.dump(meta, fh)
        eqx.tree_serialise_leaves(fh, model)


def load(path):
    with open(path, "rb") as fh:
        meta = pickle.load(fh)
        skeleton = Encoder(jax.random.PRNGKey(0), latent_dim=meta["latent_dim"])
        model = eqx.tree_deserialise_leaves(fh, skeleton)
    return model, meta


# --------------------------------------------------------------------------- #
# Learned spatial weights — the encoder's lesson, applied
# --------------------------------------------------------------------------- #
# Training the CNN above revealed the flaw in learning the distance from scratch:
# plain correlation reaches weighted L1 = 0.160 on the very target the encoder is
# fit to, while the trained encoder reaches only 0.228. Comparing ~25k ocean cells
# directly is a high-capacity operation, and squeezing it through a 64-dim pooled
# bottleneck loses more than the learned forecast-relevance recovers.
#
# So learn the INCREMENT instead. Keep the correlation kernel at full spatial
# resolution and learn only the weight map it is taken under:
#
#     d(o, x) = 1 - corr_w(o, x),      w = softplus(theta) on the model grid
#
# initialised at theta = softplus^-1(cos(lat)), i.e. *exactly* the current
# CorrelationDistance, so the fit starts from the baseline and can only depart from
# it if the data pays for it. The learned object is directly interpretable: which
# regions of the current surface field actually determine the 14-day future.

def _softplus_inv(y):
    return np.log(np.expm1(np.clip(y, 1e-6, None)))


def _corr_under_weights(X, w):
    """Pairwise weighted correlation of rows of X (M, m) under weights w (m,)."""
    W = w.sum()
    mu = (X * w).sum(1, keepdims=True) / W
    Xc = (X - mu) * jnp.sqrt(w)
    nrm = jnp.sqrt((Xc ** 2).sum(1, keepdims=True))
    Xhat = Xc / (nrm + 1e-8)
    return Xhat @ Xhat.T


def train_weights(anom, S, ocean, lat_deg, lead, *, steps=1500, batch=64, lr=3e-2,
                  exclude=30, tau=0.35, smooth=1e-3, val_frac=0.15, seed=0,
                  log_every=150, verbose=True):
    """Fit a spatial weight map so ``1 - corr_w`` predicts future dissimilarity.

    Starts exactly at the latitude-weighted correlation baseline. ``smooth`` is the
    weight of a total-variation penalty on the log-weights, which keeps the map
    physically readable rather than a per-cell fit to noise.

    Returns ``(w2d, meta)`` with ``w2d`` the (nlat, nlon) weight map.
    """
    n = anom.shape[0]
    usable = n - lead
    n_train = int(usable * (1 - val_frac))
    nlat, nlon = ocean.shape

    w0 = np.broadcast_to(np.cos(np.deg2rad(lat_deg))[:, None], (nlat, nlon))
    theta = jnp.asarray(_softplus_inv(np.asarray(w0, dtype=np.float32)))
    # Ocean columns are selected once, outside the traced function: boolean masks
    # are not concrete under jit, and this keeps the per-step array (M, m) small.
    oc_idx = np.flatnonzero(ocean.reshape(-1))
    A = jnp.asarray(np.where(np.isfinite(anom), np.nan_to_num(anom), 0.0)
                    .astype(np.float32).reshape(anom.shape[0], -1)[:, oc_idx])
    oc_idx = jnp.asarray(oc_idx)

    def loss_fn(theta, idx, tgt, keep):
        w = jax.nn.softplus(theta).reshape(-1)[oc_idx]
        d = 1.0 - _corr_under_weights(A[idx], w)
        wt = jnp.exp(-tgt / tau) * keep if np.isfinite(tau) else keep
        fit = jnp.sum(wt * jnp.abs(d - tgt)) / (jnp.sum(wt) + 1e-8)
        lw = jnp.log(jax.nn.softplus(theta) + 1e-6)
        tv = (jnp.abs(jnp.diff(lw, axis=0)).mean() + jnp.abs(jnp.diff(lw, axis=1)).mean())
        return fit + smooth * tv

    opt = optax.adam(lr)
    opt_state = opt.init(theta)

    @eqx.filter_jit
    def step(theta, st, idx, tgt, keep):
        loss, g = jax.value_and_grad(loss_fn)(theta, idx, tgt, keep)
        upd, st = opt.update(g, st)
        return optax.apply_updates(theta, upd), st, loss

    def sample(lo, hi, rg):
        idx = rg.choice(np.arange(lo, hi), size=batch, replace=False)
        tgt = 1.0 - S[np.ix_(idx + lead, idx + lead)]
        keep = (np.abs(idx[:, None] - idx[None, :]) > exclude).astype(np.float32)
        return jnp.asarray(idx), jnp.asarray(tgt), jnp.asarray(keep)

    rng = np.random.default_rng(seed)
    vrng = np.random.default_rng(seed + 1)
    history = []
    for i in range(1, steps + 1):
        theta, opt_state, loss = step(theta, opt_state, *sample(0, n_train, rng))
        if i % log_every == 0 or i == steps:
            vl = float(np.mean([float(loss_fn(theta, *sample(n_train, usable, vrng)))
                                for _ in range(6)]))
            history.append((i, float(loss), vl))
            if verbose:
                print(f"  step {i:5d}  train {float(loss):.4f}  val {vl:.4f}")

    w2d = np.asarray(jax.nn.softplus(theta)) * ocean
    return w2d, dict(history=history, lead=lead, smooth=smooth, tau=tau,
                     baseline=np.asarray(w0) * ocean)


def encode_all(model, anom, ocean, scale, batch=64):
    """Encode every library state at full ocean coverage -> (n, latent_dim)."""
    V = np.where(np.isfinite(anom), np.nan_to_num(anom), 0.0).astype(np.float32) / scale
    m = np.broadcast_to(ocean.astype(np.float32), V.shape)
    f = eqx.filter_jit(eqx.filter_vmap(model))
    out = []
    for i in range(0, V.shape[0], batch):
        x = jnp.asarray(np.stack([V[i:i + batch] * m[i:i + batch],
                                  m[i:i + batch]], axis=1))
        out.append(np.asarray(f(x)))
    return np.concatenate(out)
