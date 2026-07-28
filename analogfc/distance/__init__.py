"""Section 2 — identifying analogs.

Importing this package registers every built-in distance, so
``DISTANCES.names()`` is the list a config may choose from:

===================  ==========================================================
``ssh_rmsd``         RMSD of standardized SSH anomalies
``sst_rmsd``         RMSD of standardized SST anomalies
``ssh_front_mhd``    Loop Current front displacement, km
``correlation``      1 - pattern correlation over observed cells (cross-datum)
``latent``           1 - cosine in a learned embedding space (needs a checkpoint)
===================  ==========================================================

To add one, drop a module here with an ``@DISTANCES.register("name")`` class and
import it below.
"""

from .base import DISTANCES, ObsDistance, weighted_corr, weighted_rmsd
from .select import AnalogSelector, AnalogSet, select_top_k

from . import anomaly_rmsd, correlation, front_mhd, latent    # noqa: F401  (registration)

__all__ = ["DISTANCES", "ObsDistance", "AnalogSelector", "AnalogSet",
           "select_top_k", "weighted_corr", "weighted_rmsd"]
