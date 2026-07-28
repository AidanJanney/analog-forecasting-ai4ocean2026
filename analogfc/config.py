"""Run configuration: YAML in, resolved paths and constructed options out.

Paths inside a config are resolved against the repository root, not the config
file or the working directory, so a config can live anywhere and still point at
the same data.
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
    return validate(config, args.config)


def domain_slices(config):
    """(longitude, latitude) slices for the configured domain.

    Coordinates are ascending, so a slice with either end None is a no-op — that
    is how a null bound in the config means "not applied".
    """
    d = config["domain"]
    return (slice(d["min_longitude"], d["max_longitude"]),
            slice(d["min_latitude"], d["max_latitude"]))


# --------------------------------------------------------------------------- #
# Schema.
#
# There is one schema for every run. The GLORYS-to-GLORYS and SWOT-to-GLORYS
# workflows used to have their own, which shared 15 keys and diverged on 26 —
# including three concepts with two names each, and a `lead_days` that was a list
# of map columns in one and a single headline lead in the other. Anything written
# against the old vocabulary is rejected by name rather than ignored, because the
# failure mode that matters is a key silently not being read.
# --------------------------------------------------------------------------- #
RENAMED = {
    ("analogs", "selection_metric"): "analogs.distance",
    ("analogs", "min_sep"): "analogs.buffer_days",
    ("analogs", "target_date"): "observation.dates (a list of dates, or null for all)",
    ("verification", "ssh_contour_level"): "analogs.ssh_contour_level",
    ("verification", "front_cover_min"): "observation.front_cover_min",
    ("forecast", "length_days"): "forecast.score_every_day_to",
    ("forecast", "curve_days"): "forecast.score_every_day_to",
    ("forecast", "map_leads"): "forecast.leads",
    ("forecast", "lead_days"):
        "forecast.leads (the list of map columns) and/or forecast.headline_lead "
        "(the single summary lead) — the old key meant one in the GLORYS schema "
        "and the other in the SWOT schema",
}

SECTIONS = {"run", "data", "domain", "periods", "climatology", "swot",
            "observation", "analogs", "forecast", "figures"}

REQUIRED = {
    "run": ["source"],
    "data": ["zarr_glob", "fig_dir"],
    "periods": ["library", "target"],
    "observation": ["dates"],
    "analogs": ["distance", "k", "buffer_days", "ssh_contour_level"],
    "forecast": ["leads", "headline_lead", "score_every_day_to", "score_metrics"],
}


def validate(config, path=""):
    """Reject anything written against the old two-schema vocabulary.

    Raises with the replacement key named, so a stale config is a one-line fix
    rather than a hunt through the diff.
    """
    where = f" in {path}" if path else ""
    problems = []

    for (section, key), replacement in RENAMED.items():
        if isinstance(config.get(section), dict) and key in config[section]:
            problems.append(f"  {section}.{key}  ->  {replacement}")
    if "verification" in config:
        problems.append("  the whole [verification] section was split between "
                        "[analogs] and [observation]")
    if problems:
        raise ValueError(
            f"config{where} uses the old pre-unification key names:\n"
            + "\n".join(problems)
            + "\n\nThere is now one schema for every run; see config/analog_forecast.yaml.")

    unknown = set(config) - SECTIONS
    if unknown:
        raise ValueError(f"config{where} has unknown section(s) {sorted(unknown)}; "
                         f"choose from {sorted(SECTIONS)}")

    for section, keys in REQUIRED.items():
        if section not in config:
            raise ValueError(f"config{where} is missing the [{section}] section")
        missing = [k for k in keys if k not in config[section]]
        if missing:
            raise ValueError(f"config{where}: [{section}] is missing {missing}")

    forecast = config["forecast"]
    horizon = forecast["score_every_day_to"]
    if max(forecast["leads"]) > horizon:
        raise ValueError(
            f"config{where}: forecast.leads reaches {max(forecast['leads'])} d but "
            f"forecast.score_every_day_to is only {horizon} d")
    if forecast["headline_lead"] > horizon:
        raise ValueError(
            f"config{where}: forecast.headline_lead is {forecast['headline_lead']} d "
            f"but forecast.score_every_day_to is only {horizon} d")

    source = config["run"]["source"]
    if source == "swot":
        for key in ("swot_dir", "swot_points_dir"):
            if key not in config["data"]:
                raise ValueError(f"config{where}: run.source is 'swot' but "
                                 f"data.{key} is not set")
    return config
