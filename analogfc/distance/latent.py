"""Distance in a learned latent space — STUB (not implemented).

Kept as a worked example of what a new selection rule has to provide: a `var`, a
`representation`, a `prepare` that builds the library side once, and a `distance`
returning one score per library day. Nothing else in the pipeline changes.

Intended design, gated by the oracle diagnostic (does a better *current-field*
selection metric have headroom before this is worth building?)::

    z_i   = encoder(library state i)          # per-state latent vector
    z_obs = encoder(fill(obs_grid, mask))     # masked observation -> latent
    distance = 1 - cos(z_obs, z_i)

The encoder is trained so that latent distance on the *current* field predicts
*future* state similarity: minimise ``| d_latent(t0, a) - (1 - S_future(t0, a)) |``
over library pairs.
"""

from .base import DISTANCES, ObsDistance


@DISTANCES.register("latent")
class LatentDistance(ObsDistance):
    """Learned latent-space distance — not implemented."""

    name = "latent"
    representation = "anomaly"

    def __init__(self, var="ssh", encoder=None):
        self.var = var
        self.encoder = encoder

    def prepare(self, library):
        raise NotImplementedError(
            "LatentDistance is a documented stub. Implement an encoder mapping the "
            "field to a latent vector, trained so current-field latent distance "
            "predicts future similarity. Until then use 'correlation' or "
            "'ssh_front_mhd'.")

    def distance(self, obs_grid, mask=None):
        raise NotImplementedError("LatentDistance is a documented stub; see prepare().")
