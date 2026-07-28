"""Name -> factory lookup, one per pluggable seam.

Each of the three sections owns a registry, so adding an option is a decorated
class plus a config string and never an edit to a driver:

    @DISTANCES.register("my_metric")
    class MyDistance(ObsDistance):
        ...

`create` raises with the full list of valid names, which is what a config typo
should produce instead of a KeyError three frames down.
"""


class Registry:
    """The registered options for one seam (a distance, a source, a metric...)."""

    def __init__(self, kind):
        self.kind = kind
        self._items = {}

    def register(self, name):
        """Decorator: bind `name` to the decorated class/factory."""
        def add(obj):
            if name in self._items:
                raise ValueError(f"{self.kind} {name!r} is already registered")
            self._items[name] = obj
            return obj
        return add

    def get(self, name):
        try:
            return self._items[name]
        except KeyError:
            raise ValueError(
                f"unknown {self.kind} {name!r}; choose from {self.names()}") from None

    def create(self, name, **kwargs):
        """Instantiate the option registered under `name`."""
        return self.get(name)(**kwargs)

    def names(self):
        return sorted(self._items)

    def __contains__(self, name):
        return name in self._items
