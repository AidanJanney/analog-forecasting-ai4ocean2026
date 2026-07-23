"""Download GLORYS12 (Copernicus Marine) data for a given region, time, and variables.

Requires the `data-access-ai4ocean2026` conda environment (has `copernicusmarine`).
REQUIRED once - log in once with your Copernicus Marine account:

    copernicusmarine login

Two steps:
  1. download_glorys  — subset and write to disk (NetCDF).
  2. rechunk_to_zarr  — open with xr.open_mfdataset, rechunk, write Zarr store.
"""

from __future__ import annotations

import argparse
import glob

import copernicusmarine
import xarray as xr

# GLORYS12 global ocean physics reanalysis, daily means, 1/12 deg.
DEFAULT_DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
DEFAULT_VARIABLES = ["thetao", "so", "uo", "vo", "zos"]


def download_glorys(
    variables,
    min_lon,
    max_lon,
    min_lat,
    max_lat,
    start_datetime,
    end_datetime,
    dataset_id=DEFAULT_DATASET_ID,
    min_depth=None,
    max_depth=None,
    output_filename=None,
    output_dir=".",
):
    """Download a GLORYS subset to NetCDF via copernicusmarine.subset.

    Parameters
    ----------
    variables : list[str]
        Variable names, e.g. ["thetao", "so", "uo", "vo", "zos"].
    min_lon, max_lon, min_lat, max_lat : float
        Bounding box in degrees.
    start_datetime, end_datetime : str
        ISO datetimes, e.g. "2020-01-01" / "2020-01-31".
    dataset_id : str
        Copernicus Marine dataset id (default: GLORYS12 daily reanalysis).
    min_depth, max_depth : float, optional
        Depth range in metres.
    output_filename : str, optional
        Output file name (default: derived from dataset id).
    output_dir : str
        Directory to write into.

    Returns
    -------
    str
        Path to the written NetCDF file.
    """
    response = copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=list(variables),
        minimum_longitude=min_lon,
        maximum_longitude=max_lon,
        minimum_latitude=min_lat,
        maximum_latitude=max_lat,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        minimum_depth=min_depth,
        maximum_depth=max_depth,
        output_filename=output_filename,
        output_directory=output_dir,
        file_format="netcdf",
        overwrite=True,
    )
    path = str(response.file_path)
    print(f"Downloaded {path}")
    return path


def rechunk_to_zarr(input_path, output_path, chunks, use_dask=False):
    """Open NetCDF file(s) with xr.open_mfdataset, rechunk, and write Zarr.

    Parameters
    ----------
    input_path : str
        Path or glob pattern to input NetCDF file(s).
    output_path : str
        Path for the output Zarr store.
    chunks : dict
        Chunk sizes, e.g. {"time": 30, "latitude": 200, "longitude": 200}.
    use_dask : bool
        Pass parallel=True to open_mfdataset (uses dask for parallel I/O).
    """
    files = sorted(glob.glob(input_path)) or input_path
    ds = xr.open_mfdataset(files, combine="by_coords", parallel=use_dask)
    ds = ds.chunk(chunks)
    ds.to_zarr(output_path, mode="w")
    print(f"Wrote {output_path}")


def _parse_args():
    p = argparse.ArgumentParser(description="Download GLORYS data from Copernicus Marine.")
    sub = p.add_subparsers(dest="command", required=True)

    # --- download subcommand ---
    dl = sub.add_parser("download", help="Subset and download to NetCDF.")
    dl.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    dl.add_argument("--variables", nargs="+", default=DEFAULT_VARIABLES)
    dl.add_argument("--min-lon", type=float, required=True)
    dl.add_argument("--max-lon", type=float, required=True)
    dl.add_argument("--min-lat", type=float, required=True)
    dl.add_argument("--max-lat", type=float, required=True)
    dl.add_argument("--start", dest="start_datetime", required=True)
    dl.add_argument("--end", dest="end_datetime", required=True)
    dl.add_argument("--min-depth", type=float, default=None)
    dl.add_argument("--max-depth", type=float, default=None)
    dl.add_argument("--output-filename", default=None)
    dl.add_argument("--output-dir", default=".")

    # --- rechunk subcommand ---
    rc = sub.add_parser("rechunk", help="Rechunk NetCDF(s) to a Zarr store.")
    rc.add_argument("input_path", help="Path or glob to input NetCDF file(s).")
    rc.add_argument("output_path", help="Path for the output Zarr store.")
    rc.add_argument(
        "--chunks",
        required=True,
        help='JSON dict, e.g. \'{"time": 30, "latitude": 200, "longitude": 200}\'',
    )
    rc.add_argument("--use-dask", action="store_true", help="Use dask parallel I/O.")

    return p.parse_args()


def main():
    import json

    args = _parse_args()
    if args.command == "download":
        download_glorys(
            variables=args.variables,
            min_lon=args.min_lon,
            max_lon=args.max_lon,
            min_lat=args.min_lat,
            max_lat=args.max_lat,
            start_datetime=args.start_datetime,
            end_datetime=args.end_datetime,
            dataset_id=args.dataset_id,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            output_filename=args.output_filename,
            output_dir=args.output_dir,
        )
    elif args.command == "rechunk":
        rechunk_to_zarr(
            input_path=args.input_path,
            output_path=args.output_path,
            chunks=json.loads(args.chunks),
            use_dask=args.use_dask,
        )

if __name__ == "__main__":
    main()