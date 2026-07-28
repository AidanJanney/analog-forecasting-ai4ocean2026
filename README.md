# Analog forecasting — Gulf of Mexico

Analog forecasts of the Gulf of Mexico Loop Current from the GLORYS12 reanalysis.

A run is three choices, one per section, wired together by a config file:

```
data      →  where the observation and the analog pool come from
distance  →  how library days are ranked against the observation
forecast  →  how the analogs are combined, scored and drawn
```

Two workflows differ only in which options they pick:

| Workflow | Observation | Distance | Driver |
|---|---|---|---|
| **GLORYS → GLORYS** | a GLORYS state (full field) | `ssh_front_mhd`, `ssh_rmsd`, `sst_rmsd` | `runs/glorys_analog.py` |
| **SWOT → GLORYS** | real SWOT KaRIn swaths (partial) | `correlation` | `runs/swot_glorys.py` |

In both, the library and target periods are disjoint, so the target day cannot
leak into its own analog pool and no exclusion window is needed. What genuinely
differs is the SWOT case's partial coverage and its datum: `ssha` is referenced to
a mean sea surface and GLORYS `zos` to absolute topography, so it needs a distance
that centres both sides over the observed cells. That is a choice of option, not a
separate code path.

## Layout

| Path | Purpose |
|---|---|
| `analogfc/registry.py` | Name → factory lookup. One registry per seam. |
| `analogfc/config.py` | YAML loading and path resolution. |
| `analogfc/fronts.py` | Loop Current front masks and front-to-front MHD in km. |
| `analogfc/mhd.py` | Direct pairwise MHD. The reference the tests check `fronts.py` against. |
| **`analogfc/data/`** | **Section 1 — getting data.** |
| `data/glorys.py` | Open the Zarr stores; `Field` / `FieldSet` and their representations. |
| `data/climatology.py` | `climday` grouping, standardization, and the map back to physical units. |
| `data/library.py` | `ModelLibrary` — the analog pool plus grid metadata. |
| `data/sources.py` | `ObsSource` registry: `model`, `swot`, `ostia` (stub). |
| `data/windows.py` | Multi-day observation windows and sequence matching. |
| `data/swot.py` | SWOT retrieval from PO.DAAC and the per-day point cache. |
| `data/swath.py` | Bin along-track points onto the model grid. |
| `data/diagnostics.py` | Input-data figures (variance, distribution, variation). |
| **`analogfc/distance/`** | **Section 2 — identifying analogs.** |
| `distance/base.py` | `ObsDistance`, the `DISTANCES` registry, weighted kernels. |
| `distance/anomaly_rmsd.py` | `ssh_rmsd`, `sst_rmsd` — RMSD of standardized anomalies. |
| `distance/front_mhd.py` | `ssh_front_mhd` — Loop Current front displacement in km. |
| `distance/correlation.py` | `correlation` — pattern correlation over observed cells. |
| `distance/latent.py` | `latent` — learned latent-space distance (stub). |
| `distance/select.py` | `AnalogSet`, `AnalogSelector`, top-K with a separation buffer. |
| **`analogfc/forecast/`** | **Section 3 — rollout, scoring, plotting.** |
| `forecast/combine.py` | `COMBINERS`: `ensemble_mean`, `best_analog`, `weighted_mean`. |
| `forecast/rollout.py` | Advance a selection to every lead, in physical units. |
| `forecast/score.py` | `METRICS`: `sst_rmse`, `ssh_rmse`, `sst_acc`, `ssh_acc`, `ssh_front_mhd`. |
| `forecast/plots.py` | Analog grids, target-vs-best, per-window skill. |
| `runs/glorys_analog.py` | GLORYS → GLORYS driver. Orchestration only. |
| `runs/swot_glorys.py` | SWOT → GLORYS driver. |
| `tests/test_analogfc.py` | The suite. No data files, no pytest: `python tests/test_analogfc.py`. |
| `config/*.yaml` | Run configurations. |
| `submit_analog_forecast.pbs` | Casper PBS batch submission. |
| `../subset_glorys.py` | Subset the GDEX GLORYS mirror to per-year Zarr stores. |
| `download_glorys.py`, `download_glorys.ipynb` | Download GLORYS from Copernicus Marine. |
| `analog_forecast_notes.md` | Design decisions and results. |

### Adding an option

Each section is a registry. An option is a decorated class and a config string —
no driver changes, and every other section is untouched.

**A distance** (how analogs are ranked). Declare which variable and which anomaly
representation it consumes, and the observation source is wired to match:

```python
# analogfc/distance/my_metric.py
@DISTANCES.register("my_metric")
class MyDistance(ObsDistance):
    var, representation = "ssh", "standardized"   # or "raw" / "anomaly"

    def prepare(self, library):
        self.pool = library.pool(self.var, self.representation).values
        return self

    def distance(self, obs_grid, mask=None):
        """One score per library day; lower is better, np.inf if unscorable."""
        return ...
```

Import it in `analogfc/distance/__init__.py`, then set `selection_metric: my_metric`.

**A score metric** (how a forecast is judged). Register a factory taking the
library; declare `unit` and `higher_is_better` and the reporting follows:

```python
@METRICS.register("ssh_bias")
def _ssh_bias(library, **kw):
    return MyBias("ssh", library)
```

Then add `ssh_bias` to `forecast.score_metrics`.

**An observation type.** Implement one method and it inherits multi-day
windowing, sequence matching, selection, scoring and plotting unchanged:

```python
@SOURCES.register("my_instrument")
class MySource(ObsSource):
    def observe(self, date, library):
        """Return an ObsDay for `date`, or None if nothing usable."""
        grid, mask = ...            # (nlat, nlon) on library.lon / library.lat
        return ObsDay(date=date, grid=grid, mask=mask)
```

See `data/sources.py:OstiaSource` for a worked stub. For a one-off set of
already-gridded fields you can skip sources entirely and build a window with
`windows.window_from_days(end_date, grids, masks, dates)`.

**A combiner** (how analogs become one forecast): register a function of
`(members, distances)` in `forecast/combine.py` and set `forecast.combiner`.

### Anomaly representations

Distances declare which one they need, because using the wrong one is a silent
error rather than a crash:

| Representation | Definition | Used by |
|---|---|---|
| `raw` | the physical field | front geometry — the front is a property of physical SSH, and the standardized anomaly has exactly the mean structure that defines it removed |
| `standardized` | `(x − μ)/σ` per climatology group per grid cell | the RMSD selections and the forecast rollout |
| `anomaly` | `x −` time mean | cross-datum correlation — mean removed but *not* rescaled per cell, since an instrument anomaly has had no such rescaling applied |

They are built on first use, so a run only pays for the ones it asks for.

### The Loop Current front

Both workflows use one definition, in `analogfc/fronts.py`. A front is a **boolean
grid mask**: the cells the 0.17 m `zos` contour passes through (Leben 2005), taken
as the inside edge of the region at or above the level so coastlines are excluded.

Distance between two fronts is the modified Hausdorff distance, evaluated with a
Euclidean distance transform — the EDT of one front *is* the "distance to the
nearest front cell" field the MHD needs, so a comparison costs one transform
instead of an O(|A|·|B|) pairwise sweep, with no vertex subsampling. `mhd.py`
holds the direct pairwise definition and `tests/test_analogfc.py` confirms the
two agree exactly.

Results are in **km**: `fronts.grid_spacing_km` gives the transform the physical
cell size, which also corrects the grid's anisotropy (a degree of longitude at
25 N is ~0.9 of a degree of latitude). Two options matter:

| Option | Effect |
|---|---|
| `referenced` | Shift each field's ocean mean to a common datum before contouring, so the fixed level tracks the front's *position* rather than a basin-scale sea-level offset. Off within one reanalysis; on when comparing across datasets or eras. |
| `main_only` | Keep only the largest connected segment — the Loop Current filament itself, not detached rings that also cross the level. Unstable where the domain's eastern cut clips the contour, so it is off by default. |

### Latitude weighting

Every spatial mean — the RMSD selections, RMSE, ACC — is weighted by
cos(latitude). A grid cell at 31 N covers about 7% less area than one at 18 N, so
an unweighted domain mean over the Gulf box overweights the northern shelf. Front
MHD is geometric and unaffected.

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
python runs/glorys_analog.py                                 # config/analog_forecast.yaml
python runs/glorys_analog.py --config config/smoke_test.yaml
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

Both drivers are `# %%` cell scripts and also run cell by cell in VS Code or
Jupyter.

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

Paths are resolved relative to the repository root, not the config file.

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
| `analogs.selection_metric` | How library days are ranked. Any name in `DISTANCES`: `sst_rmsd`, `ssh_rmsd`, `ssh_front_mhd`. |
| `analogs.k` | Number of analogs retained. |
| `analogs.buffer_days` | Minimum separation between retained analogs. Nothing to do with forecast length. |
| `analogs.ssh_contour_level` | SSH contour defining the Loop Current front, in metres. |
| `forecast.length_days` | How far the rollout runs. Errors are evaluated every day out to here. |
| `forecast.lead_days` | Leads shown as map columns in the analog grids. Must fall within `length_days`. |
| `forecast.score_metrics` | Metrics every forecast is scored by. Any names in `METRICS`: `sst_rmse`, `ssh_rmse`, `sst_acc`, `ssh_acc`, `ssh_front_mhd`. ACC is a skill score, so higher is better; the rest are errors. |
| `forecast.combiner` | How the analogs become one forecast. Any name in `COMBINERS`: `ensemble_mean` (plain mean), `best_analog`, `weighted_mean`. |

## 5. Testing a new option on a small subset

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

3. Add the option — see "Adding an option" above — and name it in the config.

4. Check nothing else moved:

   ```bash
   python tests/test_analogfc.py
   python runs/glorys_analog.py --config config/my_metric.yaml
   ```

   The suite runs on synthetic data in about a second and needs no GLORYS files,
   so it is the fast check; the smoke config is the slow one.

5. Once the option is settled, run the full record with
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
python -m analogfc.data.swot fetch-range --start 2023-01-01 --end 2025-12-31
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
python -m analogfc.data.swot cache          # reduce granules already downloaded
```

### 6.2 Run

```bash
python runs/swot_glorys.py                                 # config/swot_glorys.yaml
python runs/swot_glorys.py --config config/my_swot.yaml
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
| `observation_analogs.png` | The window's binned SSHA beside its top-K GLORYS analogs. |
| `analog_grid.png` | One row per analog (state, then error), then the ensemble and the truth. Columns are each lead in `forecast.map_leads`, with the 0.17 m contour drawn on every panel — black for the truth, dashed in the row's colour for its own. The right column carries one daily skill curve per metric in `forecast.score_metrics`. |
| `swot_window_skill.png` | One panel per metric: each analog, the ensemble and persistence, per observation window. |

### 6.5 Adding another observation type

See "Adding an option" above: register an `ObsSource` and the whole driver is
unchanged. Instrument-specific policy (how many observations to keep per day, how
to read the files) belongs in the source; the window layer never learns what an
instrument is.

Pair anomaly-like observations with `distance: correlation`, which centres both
fields over the observed cells so a sensor's datum need not match GLORYS `zos`.
The `ssh_front_mhd` distance needs an *absolute* SSH field and will not work on an
anomaly-only swath.
