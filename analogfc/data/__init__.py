"""Section 1 — getting data.

Turns files on disk into the two things the rest of the pipeline consumes: a
:class:`~.library.ModelLibrary` of candidate analog states, and observations on
the model grid (:class:`~.windows.ObsDay` / :class:`~.windows.ObsWindow`) served
by a :class:`~.sources.ObsSource`.

To add an observation type, register an ``ObsSource`` in :mod:`.sources`; it
inherits windowing, sequence matching, selection and scoring unchanged.
"""

from .glorys import Field, FieldSet, open_glorys, summarize
from .library import ModelLibrary

__all__ = ["Field", "FieldSet", "ModelLibrary", "open_glorys", "summarize"]
