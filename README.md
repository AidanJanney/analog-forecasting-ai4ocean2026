# Analog forecasting — Gulf of Mexico

Analog forecasts of Gulf of Mexico SST and SSH from the GLORYS12 reanalysis.
Library days are ranked against a target day by spatial RMSD of normalized
anomalies, and by Modified Hausdorff Distance between Loop Current fronts taken
as the 17 cm SSH contour. The top-K analogs are rolled forward and verified.

## Layout

| Path | Purpose |
|---|---|
| `analog_forecast.py` | Analog selection and forecast rollout. |
| `config/analog_forecast.yaml` | Default run configuration. |
| `config/select_*.yaml` | One per selection metric, for comparing them. |
| `config/smoke_test.yaml` | Six-year configuration for testing. |
| `submit_analog_forecast.pbs` | Casper PBS batch submission. |
| `../subset_glorys.py` | Subset the GDEX GLORYS mirror to per-year Zarr stores. |
| `../submit_subset.pbs` | Casper job array over years for the above. |
| `download_glorys.py`, `download_glorys.ipynb` | Download GLORYS from Copernicus Marine. |
| `analog_forecast_notes.md` | Design decisions and results. |

## 1. Environment

```bash
conda env create -f env_data_access.yml
conda activate data-access-ai4ocean2026
```

To use the environment as a Jupyter kernel:

```bash
conda run -n data-access-ai4ocean2026 python -m ipykernel install --user --name data-access-ai4ocean2026
```

## 2. Obtain the data

Both routes produce one Zarr store per year, named `glorys_gom_<year>_subset.zarr`.

### On Casper: subset the local mirror

`subset_glorys.py` reads the GDEX GLORYS mirror at `/gdex/data/d010049`. No
credentials required. Edit the region, depth range, and variables at the top of
the file.

Single year:

```bash
python subset_glorys.py 2004
```

All years, one job per year:

```bash
qsub submit_subset.pbs
```

### Off Casper: download from Copernicus Marine

Requires a free Copernicus Marine account. Authenticate once:

```bash
copernicusmarine login
```

Download, then rechunk to Zarr:

```bash
python download_glorys.py download \
    --min-lon -98 --max-lon -78 --min-lat 18 --max-lat 32 \
    --min-depth 0 --max-depth 300 \
    --start 1993-01-01 --end 1993-12-31 \
    --output-filename glorys_gom_1993.nc --output-dir data

python download_glorys.py rechunk data/glorys_gom_1993.nc glorys_gom_1993_subset.zarr \
    --chunks '{"time": -1, "depth": 1, "latitude": -1, "longitude": -1}'
```

`download_glorys.ipynb` walks through the same two steps interactively.

## 3. Run a forecast

```bash
python analog_forecast.py                                    # config/analog_forecast.yaml
python analog_forecast.py --config config/smoke_test.yaml
```

On Casper:

```bash
qsub submit_analog_forecast.pbs
qsub -v CONFIG=config/smoke_test.yaml submit_analog_forecast.pbs
```

The full 33-year record peaks near 18 GB. Interactive sessions default to 10 GB;
request more before running it by hand:

```bash
qsub -I -A P93300012 -q casper -l select=1:ncpus=8:mem=128GB -l walltime=06:00:00
```

`analog_forecast.py` is a `# %%` cell script and also runs cell by cell in VS Code
or Jupyter.

### Output

Written to `data.fig_dir`:

| File | Contents |
|---|---|
| `sst_spatial_variance.png` | SST variance over time. |
| `sst_distribution.png` | SST histogram with mean and standard deviation bounds. |
| `sst_spatial_temporal_variation.png` | Time-mean map and domain-mean series. |
| `target_vs_best_analogs.png` | Target and best analog by each ranking, SST and SSH. |
| `analog_forecast_leads.png` | Observed, ensemble forecast, and error per lead. |
| `analog_grid_ssh_mhd.png` | Per-analog state and error by lead, SSH with front overlay. |
| `analog_grid_sst_rmsd.png` | Per-analog state and error by lead, SST. |

Analog dates, scores, and domain RMSE by lead are printed to stdout.

## 4. Configuration

Paths are resolved relative to `analog_forecast.py`, not the config file.

| Key | Meaning |
|---|---|
| `data.zarr_glob` | Glob matching the per-year Zarr stores. |
| `data.fig_dir` | Output directory for figures. |
| `domain.min_longitude` | Western bound. Applied before any statistics, ranking, or plotting. `null` for no bound. |
| `domain.max_longitude` | Eastern bound. `null` for no bound. |
| `domain.min_latitude` | Southern bound. `null` for no bound. |
| `domain.max_latitude` | Northern bound. `null` for no bound. |
| `periods.library` | Date range analogs are drawn from. |
| `periods.target` | Date range containing the day being forecast. |
| `climatology.group` | Grouping for the normalization statistics. `climday` folds Feb 29 into Feb 28; `time.dayofyear` and `time.month` also accepted. |
| `climatology.reduce_dims` | Extra dims to average over. Empty gives one mean and standard deviation per group per grid cell. |
| `analogs.target_date` | Day being forecast. |
| `analogs.selection_metric` | How library days are ranked: `sst_rmsd`, `ssh_rmsd`, or `ssh_front_mhd`. |
| `analogs.k` | Number of analogs retained. |
| `analogs.buffer_days` | Minimum separation between retained analogs. Nothing to do with forecast length. |
| `analogs.ssh_contour_level` | SSH contour defining the Loop Current front, in metres. |
| `forecast.length_days` | How far the rollout runs. Errors are evaluated every day out to here. |
| `forecast.lead_days` | Leads shown as map columns in the analog grids. Must fall within `length_days`. |
| `forecast.score_metrics` | Metrics every forecast is scored by: `sst_rmse`, `ssh_rmse`, `sst_acc`, `ssh_acc`, `ssh_front_mhd`. ACC is a skill score, so higher is better; the rest are errors. |

## 5. Testing a new metric on a small subset

Ranking the full library takes several minutes per run. Work against a six-year
subset first.

1. Subset the years needed. `config/smoke_test.yaml` expects 1993–1998:

   ```bash
   for year in 1993 1994 1995 1996 1997 1998; do
       python ../subset_glorys.py $year
   done
   ```

   Or submit them as an array by narrowing `#PBS -J` in `submit_subset.pbs`.

2. Copy the test config and point it at those years:

   ```bash
   cp config/smoke_test.yaml config/my_metric.yaml
   ```

   Set `data.zarr_glob` to match the subset, `data.fig_dir` to a separate
   directory, and keep `periods.library` and `periods.target` disjoint.

3. Add the metric to `analog_forecast.py`. A ranking needs two pieces:

   - a score per library day, as a `DataArray` indexed by `time`
   - `select_top_k_buffer(score, score, k=K, buffer=BUFFER_DAYS)` to take the
     top-K while enforcing the separation

   Follow the MHD block, which builds `mhd_scores` over `ssh_library_raw` and
   wraps it in a `DataArray`. Lower scores rank better.

4. Run and inspect:

   ```bash
   python analog_forecast.py --config config/my_metric.yaml
   ```

   `plot_analog_grid` accepts any set of analog times, so a new ranking can be
   plotted by passing its `top_*.time.values`.

5. Once the metric is settled, run the full record with
   `config/analog_forecast.yaml`.

A six-year library is too short for meaningful analogs. Use it to check that code
runs and figures render, not to judge skill.
