"""Retrieve SWOT Low-Rate SSH data from NASA PO.DAAC via ``earthaccess``.

SWOT/KaRIn measures sea surface height along two wide swaths. For the Loop
Current we use the Level-2 Low-Rate SSH **Expert** product (``ssha_karin``),
subset to a bounding box and time range. These swaths are the surface
observations that drive analog selection, reaching the pipeline through
``sources.SwotSource`` — the SWOT counterpart of the GLORYS ``zos`` fronts.

Search (CMR) is public; **downloads require an Earthdata login**. Set one up
once with either of:

    earthaccess.login(strategy="interactive", persist=True)   # writes ~/.netrc
    # or export EARTHDATA_USERNAME / EARTHDATA_PASSWORD

Note: SWOT KaRIn data does not exist before 2023-03-28. A 1-day-repeat cal/val
phase runs 2023-03-28..~2023-07-10 (a few fixed swaths revisited *daily* — dense
in time, sparse in space), then the 21-day science orbit starts 2023-07-26. A
request for a target period starting 2023-01-01 therefore returns nothing until
late March 2023; :func:`fetch_range` reports the gap rather than failing.
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np

# `earthaccess` is imported inside the three functions that need it, not here.
# Only the *download* path uses it; once a period has been reduced to the per-day
# point cache, every read function below works from disk alone (and the granules
# themselves can be deleted). Keeping the import lazy means a machine that only
# ever reads the cache — a training or forecast node — does not need Earthdata
# tooling installed at all.

# PO.DAAC SWOT Level-2 Low-Rate SSH collections. "Expert" carries the full set of
# geophysical fields and corrections; "Basic" is a lighter subset. Which one you
# want depends on the period being covered (verified against CMR, 2026-07):
#
#   expert_d    2023-03-28 .. 2025-12-31  version D, includes the cal/val phase
#               and the whole 2023-2025 target period. The default.
#   expert_2.0  2023-07-26 .. 2025-05-03  version C, validated science orbit only.
#
# Both are the same measurement processed differently, so a run can use either;
# `collection=` is a config knob throughout this module.
COLLECTIONS = {
    "expert_d": "SWOT_L2_LR_SSH_EXPERT_D",
    "expert_2.0": "SWOT_L2_LR_SSH_EXPERT_2.0",
}
DEFAULT_COLLECTION = "expert_d"
DEFAULT_SHORT_NAME = COLLECTIONS[DEFAULT_COLLECTION]
SSHA_VAR = "ssha_karin"                 # sea-surface-height anomaly on the swath

# First day any SWOT KaRIn data exists — requests earlier than this return nothing.
SWOT_EPOCH = "2023-03-28"

# Filename regex marking the *validated* processing stream, preferred over the
# interim one when both exist for a pass. The tag letter tracks the collection
# version (C -> _PGC0_/_PIC0_, D -> _PGD0_/_PID0_), so match the family, not a
# literal: "PG" = validated ground processing, "PI" = interim.
PREFER_RE = r"_PG[A-Z]\d_"

# Gulf of Mexico Loop Current box — matches the GLORYS subset in download_glorys.
GOM_BBOX = (-98.0, 18.0, -78.0, 32.0)   # (min_lon, min_lat, max_lon, max_lat)


def resolve_collection(name):
    """Map a collection alias ('expert_d') or a raw short name to a CMR short name."""
    if name is None:
        return DEFAULT_SHORT_NAME
    return COLLECTIONS.get(name, name)


def login(strategy="netrc", persist=True):
    """Authenticate with Earthdata (needed for downloads, not for search)."""
    import earthaccess

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
    import earthaccess

    return earthaccess.search_data(
        short_name=short_name,
        temporal=(start, end),
        bounding_box=bbox,
        count=count,
    )


def download(granules, out_dir="data/swot"):
    """Download granules to `out_dir` (requires an Earthdata login). Returns paths."""
    import earthaccess

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


def _granule_name(granule):
    """Best-effort local filename for an earthaccess granule result."""
    for link in granule.data_links():
        base = os.path.basename(link)
        if base.endswith(".nc"):
            return base
    return None


def fetch_range(start, end, bbox=GOM_BBOX, collection=DEFAULT_COLLECTION,
                out_dir="data/swot", chunk_days=30, skip_existing=True):
    """Download a long period in chunks, skipping granules already on disk.

    Fetching the full 2023-2025 target period is ~1800 granules / tens of GB over
    the Gulf of Mexico, which is far too much for one `earthaccess.download` call
    to survive: any network fault loses the whole batch. This walks the period in
    `chunk_days` windows and, with `skip_existing`, re-checks the output directory
    each time, so an interrupted fetch is resumed simply by re-running the same
    command.

    `collection` is an alias from :data:`COLLECTIONS` or a raw CMR short name.
    Returns the list of paths present on disk for the requested period.
    """
    short_name = resolve_collection(collection)
    t0, t1 = np.datetime64(start, "D"), np.datetime64(end, "D")
    if t1 < np.datetime64(SWOT_EPOCH, "D"):
        print(f"No SWOT data before {SWOT_EPOCH}; requested {start}..{end}.")
        return []
    if t0 < np.datetime64(SWOT_EPOCH, "D"):
        print(f"NOTE: SWOT starts {SWOT_EPOCH}; clipping request {start} -> {SWOT_EPOCH}.")
        t0 = np.datetime64(SWOT_EPOCH, "D")

    os.makedirs(out_dir, exist_ok=True)
    logged_in = False
    got = []
    chunk = np.timedelta64(int(chunk_days), "D")
    while t0 <= t1:
        c1 = min(t0 + chunk - np.timedelta64(1, "D"), t1)
        granules = search_swot(str(t0), str(c1), bbox=bbox, short_name=short_name)
        if skip_existing:
            have = set(os.listdir(out_dir))
            todo = [g for g in granules if (_granule_name(g) or "") not in have]
        else:
            todo = list(granules)
        print(f"{str(t0)}..{str(c1)}: {len(granules)} granules, {len(todo)} to download",
              flush=True)
        if todo:
            if not logged_in:
                login()
                logged_in = True
            got += download(todo, out_dir=out_dir)
        t0 = c1 + np.timedelta64(1, "D")

    paths = sorted(glob.glob(os.path.join(out_dir, "*.nc")))
    print(f"{len(got)} newly downloaded; {len(paths)} granules now in {out_dir}")
    return paths


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


def load_swaths(paths, bbox=GOM_BBOX, var=SSHA_VAR, prefer=PREFER_RE):
    """Load many granules, grouped by satellite pass. Returns (swaths, labels).

    `swaths` is a list of (lon, lat, ssha) arrays (one per pass, tiles merged);
    `labels` are "pass <n>  <date> <time>Z" strings.

    A granule exists in several processings and PO.DAAC subsetting can return
    byte-identical duplicates (trailing ``_01`` / ``_02``). For each pass (keyed by
    ``cycle`` + ``pass`` number) we keep a `prefer`-matching processing when it has
    in-box pixels, else the alternate — per-pass rather than global, so a preferred
    file that is empty in-box (its crossover calibration missing/flagged) no longer
    shadows a good alternate — then collapse exact-duplicate points so a swath is
    never double-counted.

    `prefer` is a regex matched against the filename, not a literal, because the
    processing tag is collection-specific: version C granules are ``_PGC0_`` /
    ``_PIC0_`` while version D granules are ``_PGD0_`` / ``_PID0_``. The default
    :data:`PREFER_RE` selects the validated ``PG*`` stream over the interim
    ``PI*`` one in either collection.
    """
    import numpy as np

    pat = re.compile(
        r"_(\d{3})_(\d{3})_(\d{8}T\d{6})_\d{8}T\d{6}_[A-Za-z0-9]+_\d{2}\.nc$")
    pref_re = re.compile(prefer)

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
        g["files"].append((bool(pref_re.search(f)), lon, lat, ssha))

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


def granule_dates(directory="data/swot"):
    """Sorted 'YYYY-MM-DD' acquisition dates of the granules in `directory`."""
    days = set()
    for p in glob.glob(os.path.join(directory, "*.nc")):
        m = re.search(r"_(\d{8})T\d{6}_", os.path.basename(p))
        if m:
            d = m.group(1)
            days.add(f"{d[:4]}-{d[4:6]}-{d[6:]}")
    return sorted(days)


# --------------------------------------------------------------------------- #
# Compact per-day point cache.
#
# A raw Expert granule is ~30 MB of full-swath geophysics, of which the Gulf box
# keeps a few tens of thousands of points. The analog driver asks for the same
# dates repeatedly (once per day of every observation window), so re-opening the
# granules each time dominates the runtime of a multi-year run. Reducing each day
# once to its in-box (lon, lat, ssha) points turns that into a ~1 MB npz read.
# --------------------------------------------------------------------------- #
POINTS_DIR = "data/swot_points"


def _points_path(date, cache_dir=POINTS_DIR):
    return os.path.join(cache_dir, f"{str(date).replace('-', '')}.npz")


def build_points_cache(directory="data/swot", cache_dir=POINTS_DIR, bbox=GOM_BBOX,
                       var=SSHA_VAR, prefer=PREFER_RE, dates=None, overwrite=False):
    """Reduce every granule day to a compact npz of in-box swath points.

    Writes one file per date holding the concatenated points of that day's passes
    plus the offsets that split them back into individual swaths, so
    :func:`swaths_for_dates` returns exactly what it would have from the granules.
    Returns the list of dates cached.
    """
    os.makedirs(cache_dir, exist_ok=True)
    dates = list(dates) if dates is not None else granule_dates(directory)
    done = []
    for d in dates:
        out = _points_path(d, cache_dir)
        if os.path.exists(out) and not overwrite:
            done.append(d)
            continue
        swaths, labels = _swaths_from_granules([d], directory, bbox, var, prefer)
        # Store as one flat point array + split offsets; npz has no ragged support.
        lon = np.concatenate([s[0] for s in swaths]) if swaths else np.empty(0)
        lat = np.concatenate([s[1] for s in swaths]) if swaths else np.empty(0)
        ssha = np.concatenate([s[2] for s in swaths]) if swaths else np.empty(0)
        splits = np.cumsum([len(s[0]) for s in swaths])[:-1] if swaths else np.empty(0, int)
        np.savez_compressed(out, lon=lon.astype(np.float32), lat=lat.astype(np.float32),
                            ssha=ssha.astype(np.float32),
                            splits=np.asarray(splits, dtype=np.int64),
                            labels=np.array(labels, dtype=object), allow_pickle=True)
        done.append(d)
    return done


def cached_dates(cache_dir=POINTS_DIR):
    """Sorted 'YYYY-MM-DD' dates present in the reduced point cache.

    The cache is self-sufficient: once a period is reduced, the bulky granules can
    be deleted and every workflow below still runs, so date discovery must not
    depend on the granules still being on disk.
    """
    days = set()
    for p in glob.glob(os.path.join(cache_dir, "*.npz")):
        d = os.path.splitext(os.path.basename(p))[0]
        if len(d) == 8 and d.isdigit():
            days.add(f"{d[:4]}-{d[4:6]}-{d[6:]}")
    return sorted(days)


def observed_dates(directory="data/swot", cache_dir=POINTS_DIR):
    """All dates with SWOT observations available, from the cache and/or granules."""
    return sorted(set(granule_dates(directory)) | set(cached_dates(cache_dir)))


def load_points_day(date, cache_dir=POINTS_DIR):
    """Load one cached day as (swaths, labels), or None if it is not cached."""
    path = _points_path(date, cache_dir)
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=True)
    lon, lat, ssha = z["lon"], z["lat"], z["ssha"]
    splits = z["splits"]
    swaths = [(a, b, c) for a, b, c in zip(np.split(lon, splits), np.split(lat, splits),
                                           np.split(ssha, splits)) if a.size]
    return swaths, list(z["labels"])


def _swaths_from_granules(dates, directory, bbox, var, prefer):
    """Load swaths for `dates` straight from the raw granules in `directory`."""
    want = {str(d).replace("-", "") for d in dates}          # 'YYYYMMDD'
    paths = []
    for p in glob.glob(os.path.join(directory, "*.nc")):
        m = re.search(r"_(\d{8})T\d{6}_", p)
        if m and m.group(1) in want:
            paths.append(p)
    return load_swaths(paths, bbox=bbox, var=var, prefer=prefer)


def swaths_for_dates(dates, directory="data/swot", bbox=GOM_BBOX, var=SSHA_VAR,
                     prefer=PREFER_RE, cache_dir=POINTS_DIR):
    """Load swaths whose acquisition start date is in `dates` (list of 'YYYY-MM-DD').

    Reads the compact per-day cache written by :func:`build_points_cache` when it
    is present (and falls back to the raw granules otherwise), then returns
    (swaths, labels) with one entry per satellite pass. Pass ``cache_dir=None`` to
    force reading the granules.
    """
    dates = [str(d) for d in dates]
    if cache_dir is None:
        return _swaths_from_granules(dates, directory, bbox, var, prefer)

    swaths, labels, uncached = [], [], []
    for d in dates:
        hit = load_points_day(d, cache_dir)
        if hit is None:
            uncached.append(d)
        else:
            swaths += hit[0]
            labels += hit[1]
    if uncached:                       # days not reduced yet: read their granules
        s, l = _swaths_from_granules(uncached, directory, bbox, var, prefer)
        swaths += s
        labels += l
    return swaths, labels


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

    # The target-period fetch: chunked and resumable, by collection alias.
    fr = sub.add_parser("fetch-range",
                        help="Download a long period in resumable chunks.")
    fr.add_argument("--start", default="2023-01-01")
    fr.add_argument("--end", default="2025-12-31")
    fr.add_argument("--collection", default=DEFAULT_COLLECTION,
                    help=f"alias {sorted(COLLECTIONS)} or a raw CMR short name")
    fr.add_argument("--bbox", nargs=4, type=float, default=list(GOM_BBOX),
                    metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    fr.add_argument("--out-dir", default="data/swot")
    fr.add_argument("--chunk-days", type=int, default=30)
    fr.add_argument("--no-cache", action="store_true",
                    help="skip building the compact per-day point cache afterwards")

    ca = sub.add_parser("cache", help="Reduce downloaded granules to per-day points.")
    ca.add_argument("--directory", default="data/swot")
    ca.add_argument("--cache-dir", default=POINTS_DIR)
    ca.add_argument("--bbox", nargs=4, type=float, default=list(GOM_BBOX),
                    metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    ca.add_argument("--overwrite", action="store_true")
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
    elif args.command == "fetch-range":
        fetch_range(args.start, args.end, bbox=bbox, collection=args.collection,
                    out_dir=args.out_dir, chunk_days=args.chunk_days)
        if not args.no_cache:
            days = build_points_cache(directory=args.out_dir, bbox=bbox)
            print(f"Cached {len(days)} days of swath points.")
    elif args.command == "cache":
        days = build_points_cache(directory=args.directory, cache_dir=args.cache_dir,
                                  bbox=bbox, overwrite=args.overwrite)
        print(f"Cached {len(days)} days into {args.cache_dir}")


if __name__ == "__main__":
    main()
