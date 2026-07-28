"""Section 3 — rolling out predictions, scoring them, and plotting them.

Selection is already done by the time anything here runs: these take an
:class:`~..distance.select.AnalogSet` and turn it into fields, numbers and
figures. Two registries are open for extension — :data:`~.combine.COMBINERS`
(how analogs become one forecast) and :data:`~.score.METRICS` (how it is judged).

The submodules ``rollout`` and ``score`` each define a function of the same name,
so those functions are not re-exported here; import the modules and call
``rollout.rollout(...)`` / ``score.build(...)``.
"""

from . import combine, plots, rollout, score
from .combine import COMBINERS
from .rollout import Rollout
from .score import METRICS, ErrorMetric

__all__ = ["COMBINERS", "METRICS", "ErrorMetric", "Rollout",
           "combine", "plots", "rollout", "score"]
