"""Retrieve SWOT Low-Rate SSH data from NASA PO.DAAC via ``earthaccess``.

SWOT/KaRIn measures sea surface height along two wide swaths. For the Loop
Current we use the Level-2 Low-Rate SSH **Expert** product (``ssha_karin``),
subset to a bounding box and time range. These swaths are the surface
observations that will eventually drive analog selection (see
``metrics.AnalogMetric``) — the SWOT counterpart of the GLORYS ``zos`` fronts.

Search (CMR) is public; **downloads require an Earthdata login**. Set one up
once with either of:

    earthaccess.login(strategy="interactive", persist=True)   # writes ~/.netrc
    # or export EARTHDATA_USERNAME / EARTHDATA_PASSWORD

Note: SWOT science data begins ~2023-07 (21-day science orbit; a 1-day cal/val
phase runs ~2023-04..2023-07), so temporal queries must fall in the SWOT era.
"""

from __future__ import annotations

import argparse
import os

import earthaccess

# PO.DAAC SWOT Level-2 Low-Rate SSH science collections (v2.0). "Expert" carries
# the full set of geophysical fields and corrections; "Basic" is a lighter subset.
DEFAULT_SHORT_NAME = "SWOT_L2_LR_SSH_EXPERT_2.0"
SSHA_VAR = "ssha_karin"                 # sea-surface-height anomaly on the swath

# Gulf of Mexico Loop Current box — matches the GLORYS subset in download_glorys.
GOM_BBOX = (-98.0, 18.0, -78.0, 32.0)   # (min_lon, min_lat, max_lon, max_lat)


def login(strategy="netrc", persist=True):
    """Authenticate with Earthdata (needed for downloads, not for search)."""
    return earthaccess.login(strategy=strategy, persist=persist)


def search_swot(start, end, bbox=GOM_BBOX, short_name=DEFAULT_SHORT_NAME, count=-1):
    """Search PO.DAAC for SWOT LR SSH granules over a bbox and time range.

    Parameters
    ----------
    start, end : str
        ISO datetimes, e.g. "2024-01-01" / "2024-01-08".
    bbox : (min_lon, min_lat, max_lon, max_lat)
        Bounding box in degrees (-180..180).
    short_name : str
        PO.DAAC collection short name.
    count : int
        Max granules (-1 for all matches).

    Returns
    -------
    list
        earthaccess granule results (pass to :func:`download`).
    """
    return earthaccess.search_data(
        short_name=short_name,
        temporal=(start, end),
        bounding_box=bbox,
        count=count,
    )


def download(granules, out_dir="data/swot"):
    """Download granules to `out_dir` (requires an Earthdata login). Returns paths."""
    os.makedirs(out_dir, exist_ok=True)
    return earthaccess.download(granules, local_path=out_dir)


def fetch_swot(start, end, bbox=GOM_BBOX, short_name=DEFAULT_SHORT_NAME,
               out_dir="data/swot", count=-1):
    """Search + login + download in one call. Returns local file paths."""
    granules = search_swot(start, end, bbox=bbox, short_name=short_name, count=count)
    print(f"Found {len(granules)} SWOT granules for {short_name} "
          f"{start}..{end} over {bbox}")
    if not granules:
        return []
    login()
    return download(granules, out_dir=out_dir)


def load_swath(path, bbox=GOM_BBOX, var=SSHA_VAR, qual_max=0, xover=True):
    """Open one LR SSH granule and return in-box (lon, lat, ssha) as 1-D arrays.

    SWOT longitudes are 0..360 and are wrapped to -180..180 here. Pixels are kept
    only where the quality flag ``<var>_qual <= qual_max`` (0 = good); raw KaRIn
    SSHA contains large edge/land artefacts that must be masked out.

    When ``xover`` is set, the crossover calibration ``height_cor_xover`` is added
    to the SSHA. This is essential: it removes the large cross-track-correlated
    (roll/phase) error that otherwise dominates the ~0.3 m mesoscale signal
    (per-swath std drops from ~1.2 m to ~0.08 m). The returned points feed a
    surface analog metric.
    """
    import numpy as np
    import xarray as xr

    ds = xr.open_dataset(path)
    lon = ds["longitude"].values
    lat = ds["latitude"].values
    ssha = ds[var].values.copy()
    lon = np.where(lon > 180.0, lon - 360.0, lon)

    min_lon, min_lat, max_lon, max_lat = bbox
    m = (
        (lon >= min_lon) & (lon <= max_lon)
        & (lat >= min_lat) & (lat <= max_lat)
        & np.isfinite(ssha)
    )
    qual = f"{var}_qual"
    if qual in ds:                       # keep good-quality SSHA pixels only
        m &= np.nan_to_num(ds[qual].values, nan=1e9) <= qual_max
    if xover and "height_cor_xover" in ds:
        ssha = ssha + ds["height_cor_xover"].values   # crossover calibration
        # drop pixels where the crossover calibration itself is flagged/missing
        m &= np.isfinite(ds["height_cor_xover"].values)
        if "height_cor_xover_qual" in ds:
            m &= np.nan_to_num(ds["height_cor_xover_qual"].values, nan=1e9) <= qual_max
    return lon[m], lat[m], ssha[m]


def load_swaths(paths, bbox=GOM_BBOX, var=SSHA_VAR, prefer="_PGC0_"):
    """Load many granules, grouped by satellite pass. Returns (swaths, labels).

    `swaths` is a list of (lon, lat, ssha) arrays (one per pass, tiles merged);
    `labels` are "pass <n>  <date> <time>Z" strings.

    A granule exists in several processings (validated ``_PGC0_``, interim
    ``_PIC0_``) and PO.DAAC subsetting can return byte-identical duplicates
    (trailing ``_01`` / ``_02``). For each pass (keyed by ``cycle`` + ``pass``
    number) we keep the ``prefer`` processing when it has in-box pixels, else the
    alternate — per-pass rather than global, so a preferred file that is empty
    in-box (its crossover calibration missing/flagged) no longer shadows a good
    alternate — then collapse exact-duplicate points so a swath is never
    double-counted.
    """
    import re
    import numpy as np

    pat = re.compile(
        r"_(\d{3})_(\d{3})_(\d{8}T\d{6})_\d{8}T\d{6}_[A-Za-z0-9]+_\d{2}\.nc$")

    # Bucket every granule under its (cycle, pass), tagging preferred processings.
    groups = {}
    for f in sorted(paths):
        mm = pat.search(os.path.basename(f))
        lon, lat, ssha = load_swath(f, bbox=bbox, var=var)
        if mm:
            cycle, pas, start = mm.groups()
            key = (cycle, pas)
        else:                                    # unrecognised name: keep as-is
            key, pas, start = (f, None), f, ""
        g = groups.setdefault(key, {"pas": pas, "start": start, "files": []})
        g["files"].append((prefer in f, lon, lat, ssha))

    swaths, labels = [], []
    for _key, g in sorted(groups.items()):
        pref = [t for t in g["files"] if t[0] and t[3].size]   # non-empty preferred
        use = pref or [t for t in g["files"] if t[3].size]     # else any non-empty
        if not use:
            continue
        lon = np.concatenate([t[1] for t in use])
        lat = np.concatenate([t[2] for t in use])
        ssha = np.concatenate([t[3] for t in use])
        _, uniq = np.unique(np.column_stack([lon, lat]), axis=0, return_index=True)
        lon, lat, ssha = lon[uniq], lat[uniq], ssha[uniq]      # drop duplicate points
        swaths.append((lon, lat, ssha))
        labels.append(f"pass {g['pas']}  {g['start'][:8]} {g['start'][9:13]}Z")
    return swaths, labels


def swaths_for_dates(dates, directory="data/swot", bbox=GOM_BBOX, var=SSHA_VAR,
                     prefer="_PGC0_"):
    """Load swaths whose acquisition start date is in `dates` (list of 'YYYY-MM-DD').

    Filters the granules in `directory` by the date in their filename, then groups
    them by pass via :func:`load_swaths`. Returns (swaths, labels).
    """
    import glob
    import re

    want = {d.replace("-", "") for d in dates}          # 'YYYYMMDD'
    paths = []
    for p in glob.glob(os.path.join(directory, "*.nc")):
        m = re.search(r"_(\d{8})T\d{6}_", p)
        if m and m.group(1) in want:
            paths.append(p)
    return load_swaths(paths, bbox=bbox, var=var, prefer=prefer)


def _parse_args():
    p = argparse.ArgumentParser(description="Retrieve SWOT LR SSH from PO.DAAC.")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--start", required=True, help="ISO datetime, e.g. 2024-01-01")
    common.add_argument("--end", required=True, help="ISO datetime, e.g. 2024-01-08")
    common.add_argument("--short-name", default=DEFAULT_SHORT_NAME)
    common.add_argument("--bbox", nargs=4, type=float, default=list(GOM_BBOX),
                        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    common.add_argument("--count", type=int, default=-1)

    sub.add_parser("search", parents=[common], help="List matching granules (no login).")
    dl = sub.add_parser("download", parents=[common], help="Search + download granules.")
    dl.add_argument("--out-dir", default="data/swot")
    return p.parse_args()


def main():
    args = _parse_args()
    bbox = tuple(args.bbox)
    if args.command == "search":
        granules = search_swot(args.start, args.end, bbox=bbox,
                               short_name=args.short_name, count=args.count)
        print(f"Found {len(granules)} granules for {args.short_name} "
              f"{args.start}..{args.end} over {bbox}")
        for g in granules[:20]:
            print(" ", g)
    elif args.command == "download":
        paths = fetch_swot(args.start, args.end, bbox=bbox,
                           short_name=args.short_name, out_dir=args.out_dir,
                           count=args.count)
        print(f"Downloaded {len(paths)} files to {args.out_dir}")


if __name__ == "__main__":
    main()
