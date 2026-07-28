"""How the selected analogs become one forecast.

The combiner sees only the chosen analogs and their fields — never the
observation or the distance — so changing how an ensemble is formed never
touches how it was selected.
"""

import numpy as np

from ..registry import Registry

COMBINERS = Registry("combiner")


@COMBINERS.register("ensemble_mean")
def ensemble_mean(members, distances=None):
    """Plain mean of the members. The default.

    Every retained analog is already required to clear the selection threshold
    and the separation buffer, so weighting them further mostly re-expresses the
    ranking rather than adding information.
    """
    return sum(members) / len(members)


@COMBINERS.register("best_analog")
def best_analog(members, distances=None):
    """The single nearest analog.

    Preferred for sharp frontal targets: averaging analogs whose fronts sit in
    different places blurs the front into a broad gradient that is nowhere, which
    scores well on RMSE and badly on front displacement.
    """
    return members[0]


@COMBINERS.register("weighted_mean")
def weighted_mean(members, distances=None):
    """Gaussian-on-distance weighted mean, so a close analog counts for more."""
    if distances is None:
        return ensemble_mean(members)
    d = np.asarray(distances, dtype=float)
    if d.size == 0 or not np.isfinite(d).any() or d.mean() == 0:
        return ensemble_mean(members)
    w = np.exp(-(d / d.mean()) ** 2)
    w = w / w.sum()
    return sum(wi * m for wi, m in zip(w, members))
