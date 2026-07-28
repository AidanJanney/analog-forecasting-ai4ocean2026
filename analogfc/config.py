"""Run configuration: YAML in, resolved paths and constructed options out.

Paths inside a config are resolved against the repository root, not the config
file or the working directory, so a config can live anywhere and still point at
the same data. This preserves the behaviour the drivers had when they resolved
against their own directory: the drivers moved into ``runs/`` but the configs
still say ``../glorys_gom*_subset.zarr``, meaning "next to the repo".
"""

import argparse
import os

import yaml

# analogfc/config.py -> analogfc -> repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve(value):
    """Resolve a config path against the repo root; absolute paths pass through."""
    if value is None or os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(ROOT, value))


def load(default_config, description):
    """Parse ``--config`` and load it.

    ``parse_known_args`` so the drivers still run cell by cell in Jupyter and
    VS Code, where ``sys.argv`` carries the kernel's own arguments.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=resolve(default_config),
                        help="Path to the YAML config file.")
    args, _ = parser.parse_known_args()
    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    print(f"Config: {args.config}")
    return config


def domain_slices(config):
    """(longitude, latitude) slices for the configured domain.

    Coordinates are ascending, so a slice with either end None is a no-op — that
    is how a null bound in the config means "not applied".
    """
    d = config["domain"]
    return (slice(d["min_longitude"], d["max_longitude"]),
            slice(d["min_latitude"], d["max_latitude"]))
