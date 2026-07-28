# Analog forecasting — Gulf of Mexico

Analog forecasts of the Gulf of Mexico Loop Current from the GLORYS12 reanalysis.
There are two workflows, sharing the same library and front metrics:

| Workflow | Observation | Library | Driver |
|---|---|---|---|
| **GLORYS → GLORYS** | a GLORYS state (full field) | GLORYS | `analog_forecast.py` |
| **SWOT → GLORYS** | real SWOT KaRIn swaths (partial) | GLORYS | `main_swot_glorys.py` |

They are deliberately separate. In the GLORYS-to-GLORYS case the observation is
itself a library state, so the ranking sees the full field and needs a temporal
exclusion window to stop the target's own event leaking into its analog pool. In
the SWOT-to-GLORYS case the observation is a 2023–2025 satellite swath and the
pool is pre-2023 GLORYS: coverage is partial, the datums differ (`ssha` against
`zos`), and the two eras are disjoint so no exclusion is needed.

## Layout

| Path | Purpose |
|---|---|
| `analog_forecast.py` | GLORYS → GLORYS selection and forecast rollout. Standalone: it predates the module layer below and does not import it. |
| `main_swot_glorys.py` | SWOT → GLORYS driver (see §6). |
| `obs_window.py` | Multi-day / multi-swath observation windows and sequence matching. |
| `swot_data.py` | SWOT retrieval from PO.DAAC, and the reduced per-day point cache. |
| `analog.py` | `ModelLibrary`, analog selection, forecast combiners. |
| `distances.py` | Pluggable observation-space distances (correlation, front MHD). |
| `sources.py` | Pluggable observation sources (SWOT, self, OSTIA stub). |
| `fronts.py` | Loop Current front masks and front-to-front MHD in km. Shared by both workflows. |
| `mhd.py` | Direct pairwise MHD definition. Reference implementation the tests check `fronts.py` against. |
| `metrics.py` | Forecast scoring metrics (ACC, RMSE, front MHD). |
| `viz.py` | Figures. |
| `config/analog_forecast.yaml` | GLORYS → GLORYS run configuration. |
| `config/swot_glorys.yaml` | SWOT → GLORYS run configuration. |
| `config/select_*.yaml` | One per selection metric, for comparing them. |
| `config/smoke_test.yaml` | Six-year configuration for testing. |
| `submit_analog_forecast.pbs` | Casper PBS batch submission. |
| `../subset_glorys.py` | Subset the GDEX GLORYS mirror to per-year Zarr stores. |
| `../submit_subset.pbs` | Casper job array over years for the above. |
| `download_glorys.py`, `download_glorys.ipynb` | Download GLORYS from Copernicus Marine. |
| `analog_forecast_notes.md` | Design decisions and results. |

### Module architecture

`main_swot_glorys.py` is one pipeline built from three swappable seams. Each has a
single abstract method, and changing one never touches the others:

```
sources.ObsSource     .observe(date, library) -> ObsDay      where the observation comes from
        |                                                    (SwotSource, SelfSource, OstiaSource stub)
        v
obs_window.build_window(source, end, library, n_days=...)     n consecutive days -> one ObsWindow
        |
        v
distances.ObsDistance .distance(grid, mask) -> (n,)          how it is ranked against the library
        |                                                    (CorrelationDistance, FrontMHDDistance)
        v
analog.AnalogSelector .select(window, lead, k) -> AnalogSet   which library states are the analogs
        |
        v
analog.Combiner       (analogs, state, lead) -> field        how they become a forecast
                                                             (EnsembleMean, BestAnalog)
```

Selection and forecasting are decoupled: an `AnalogSet` is just indices plus
distances, so the same selection can be rolled out to many leads or scored on its
own. `metrics.ErrorMetric` then scores the result (ACC, RMSE, front MHD in km).

### The Loop Current front

Both workflows use one definition, in `fronts.py`. A front is a **boolean grid
mask**: the cells the 0.17 m `zos` contour passes through (Leben 2005), taken as
the inside edge of the region at or above the level so coastlines are excluded.

Distance between two fronts is the modified Hausdorff distance, evaluated with a
Euclidean distance transform — the EDT of one front *is* the "distance to the
nearest front cell" field the MHD needs, so a comparison costs one transform
instead of an O(|A|·|B|) pairwise sweep, with no vertex subsampling. `mhd.py`
holds the direct pairwise definition and the tests confirm the two agree exactly.

Results are in **km**: `fronts.grid_spacing_km` gives the transform the physical
cell size, which also corrects the grid's anisotropy (a degree of longitude at
25 N is ~0.9 of a degree of latitude). Two options matter:

| Option | Effect |
|---|---|
| `ref_mean` | Shift the field's ocean mean to a common datum before contouring, so the fixed level tracks the front's *position* rather than a basin-scale sea-level offset. Essential when comparing across eras. |
| `main_only` | Keep only the largest connected segment — the Loop Current filament itself, not detached rings that also cross the level. |

There is one selector and one window builder for both workflows — a one-day window
is the single-observation case, and the sequence distance over a single day
reduces to that day's distance exactly.

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

   Follow `front_mhd()`, which scores every library day against the target and
   wraps the result in a `DataArray`. Lower scores rank better; return `np.inf`
   for a day that cannot be scored, so it is never selected.

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

## 6. SWOT → GLORYS forecasts

Real SWOT KaRIn swaths from 2023–2025 initialize a forecast; analogs come from the
disjoint pre-2023 GLORYS library and are verified against the dense GLORYS field at
obs + lead.

### 6.1 Fetch the SWOT data

Downloads need a one-time NASA Earthdata login (searching does not):

```bash
python -c "import earthaccess; earthaccess.login(strategy='interactive', persist=True)"
```

Then fetch the target period. The fetch walks the period in chunks and skips
granules already on disk, so re-running the same command resumes an interrupted
download:

```bash
python swot_data.py fetch-range --start 2023-01-01 --end 2025-12-31
```

SWOT KaRIn data does not exist before **2023-03-28**, so a request starting
2023-01-01 is clipped to that date and reports the gap. Two collections are
available; `--collection` takes either alias or a raw CMR short name:

| Alias | Coverage | Notes |
|---|---|---|
| `expert_d` (default) | 2023-03-28 – 2025-12-31 | Covers the whole target period, including the 1-day-repeat cal/val phase. |
| `expert_2.0` | 2023-07-26 – 2025-05-03 | Validated science orbit only. |

Over the Gulf box that is ~1800 granules / ~55 GB. The fetch then reduces each day
to a compact `.npz` of in-box points (~1 MB/day) under `data/swot_points/`. That
cache is what the driver reads, so the bulky granules can be deleted afterwards:

```bash
python swot_data.py cache          # reduce granules already downloaded
```

### 6.2 Run

```bash
python main_swot_glorys.py                                 # config/swot_glorys.yaml
python main_swot_glorys.py --config config/my_swot.yaml
```

### 6.3 The observation window

Each forecast is initialized from an **observation window**: a run of consecutive
days, each carrying however many swaths fell in the box that day. Both knobs are
modular, so one swath on one day and several co-located swaths over several days
are the same code path:

```yaml
observation:
  n_days: 3                 # consecutive days in the window (1 = a single day)
  max_swaths_per_day: null  # null keeps every co-located pass; 1 uses the best
  min_cells_per_day: 200    # drop a day that barely clips the box
  require_days: 1           # minimum usable days for the window to be forecast
  weights: uniform          # or "recency": halve a day's weight every `halflife` days
```

The window is matched against the library as a **sequence**, not as one flattened
field. Day *o* of the window is compared against the library state *o* days before
each candidate, and the distances are combined:

```
D[j] = Σ_o  w_o · d( obs[end − o], GLORYS[j − o] )
```

so an analog has to reproduce the observed *evolution*, not just the final
snapshot. With `n_days: 1` this reduces exactly to the single-day distance.

### 6.4 Output

Written to `data.fig_dir` (default `figures_swot_glorys/`):

| File | Contents |
|---|---|
| `swot_analogs.png` | The window's binned SSHA beside its top-K GLORYS analogs. |
| `analog_grid.png` | One row per analog (state, then misfit), then the ensemble and the truth. Columns are the SWOT observation and each lead in `forecast.map_leads`, with the 0.17 m contour drawn on every panel — black for the truth, dashed in the row's colour for its own. The right column carries daily ACC, RMSE and front-MHD curves for every member against the ensemble and persistence. |
| `analog_glorys_lc_skill.png` | Loop Current front MHD per window vs persistence. |
| `analog_glorys_skill.png` | Full-field ACC/RMSE per window vs persistence. |

### 6.5 Adding another observation type

`sources.ObsSource` is the seam, and it has exactly one required method:

```python
class MySource(ObsSource):
    def observe(self, date, library):
        """Return an obs_window.ObsDay for `date`, or None if nothing usable."""
        grid, mask = ...            # (nlat, nlon) on library.lon / library.lat
        return ObsDay(date=date, grid=grid, mask=mask)
```

Implement that — see the `OstiaSource` stub — and `obs_window.build_window` gives
it multi-day windowing, sequence matching, selection and the whole driver
unchanged. Instrument-specific policy (how many observations to keep per day, how
to read the files) belongs in the source; the window layer imports nothing but
numpy and never learns what an instrument is.

For a one-off set of already-gridded fields you can skip `sources` entirely and
build a window with `obs_window.window_from_days(end_date, grids, masks, dates)`.

Pair anomaly-like observations with `distance: correlation`; it centres both
fields over the observed cells, so a sensor's datum need not match GLORYS `zos`.
The `front_mhd` distance needs an *absolute* SSH field and will not work on an
anomaly-only swath.
