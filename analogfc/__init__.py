"""Analog forecasting of the Gulf of Mexico Loop Current.

Three sections, each a pluggable seam:

    data      -> getting data          library, fields, observation sources
    distance  -> identifying analogs   how library days are ranked      [DISTANCES]
    forecast  -> rollout / scoring     combining, scoring, plotting     [COMBINERS, METRICS]

A run wires one option from each together; adding an option means registering a
class in the relevant section and naming it in a config, never editing a driver.
"""

from . import config, fronts, registry

__all__ = ["config", "fronts", "registry"]
