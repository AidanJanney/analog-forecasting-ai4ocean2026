# Analog forecasting — design notes and results

Design decisions and results. See `README.md` for setup and usage.
Last updated 2026-07-28.

## What changed

### 2026-07-28 — three sections, one implementation each

The work was split across two standalone drivers that had grown parallel
implementations of the same ideas: two climatologies, two RMSE/ACC pairs, two
analog-grid renderers, two ways of spacing the top-K apart. It is now a package
with one section per stage — `analogfc/data`, `analogfc/distance`,
`analogfc/forecast` — each a registry, so an option is a decorated class plus a
config string and the drivers hold orchestration only.

Where the two stacks disagreed numerically, `analog_forecast.py`'s behaviour is
the one kept: standardized `climday` anomalies, unreferenced front contours, a
plain-mean ensemble, a calendar-day separation buffer, and front distances in km.
The SWOT path's variants were dropped rather than carried alongside. Two of its
pieces survive as *options*, because they answer a real question the GLORYS path
never faces — `correlation` (partial coverage, mismatched datum) and
`referenced` front contouring (comparing across eras).

Also removed: the n×n front dissimilarity matrix and its on-disk cache. It only
paid off for a self-referential library, and with the library and target periods
disjoint the target-EDT approach is O(n) rather than O(n²) — one transform of the
target front, then read every library front off it.

Verified by running the restructured driver with latitude weights forced to 1:
stdout identical to the original on all three selection metrics across 31 leads
and 5 score metrics, and all 18 figures byte-identical.

### 2026-07-28 — latitude weighting

Every spatial mean is now weighted by cos(latitude): the RMSD selections, RMSE
and ACC. A cell at 31 N covers ~7% less area than one at 18 N, so the unweighted
domain mean was overweighting the northern shelf. Front MHD is geometric and
unchanged.

The effect on the six-year smoke test is small and in the expected direction:

| | change |
|---|---|
| `ssh_front_mhd` selection and scores | none — the metric never took a spatial mean |
| `ssh_rmsd` / `sst_rmsd` selection | same analogs, scores shifted in the 4th decimal; one `sst_rmsd` analog moved by a day (1995-09-23 → 1995-09-22) |
| RMSE / ACC scores | ~0.5% |

### 2026-07-28 — one driver, one config schema

The two drivers were a symptom; the two config *schemas* were the disease. They
shared 15 keys and diverged on 26, gave three concepts two names each
(`selection_metric`/`distance`, `buffer_days`/`min_sep`, and `ssh_contour_level`
in two different sections), and — worst — `forecast.lead_days` was a list of map
columns in one schema and a single headline lead in the other. Same key, different
type, no way to catch a mistake.

That divergence had already caused a bug: a bulk edit adding `region:` to every
config put it under `[climatology]` in the SWOT file, because that section sits
between `[analogs]` and `[forecast]` there but not elsewhere. Nothing read it.

Now one `runs/forecast.py` and one schema, with `run.source` choosing where
observations come from. The unification is real rather than a dispatch: every run
is windows x leads, and the old GLORYS workflow is `n_windows == 1` while the SWOT
one is `n_leads == 1`. Old key names are rejected by name rather than ignored,
since a silently-unread key is the failure that matters.

Also restored in the merge: **persistence** is now reported for every run, not
just the SWOT one. On the six-year smoke test the analog ensemble loses to
persistence on 4 of 5 metrics — the same pattern the real SWOT run showed, and a
useful reminder that the baseline is strong over short leads in quiet periods.

Verified: the GLORYS run is bit-identical through the new driver on all three
selection metrics, all 18 figures included, and the SWOT run reproduces its
numbers exactly (0.145 / 0.522 / 29.517).

### 2026-07-28 — first real SWOT -> GLORYS run

87 Expert-D granules for 2023-11-01..2023-12-15 (2.8 GB, reduced to 5.6 MB of
in-box points). 21 usable windows at lead 14, from 45 observed days: 8 dropped for
too little SWOT coverage, 16 because the window saw under 20% of the Loop Current
front.

| metric | best analog | ensemble | persistence |
|---|---|---|---|
| front MHD (km) | 29.5 | 43.0 | **22.1** |
| ACC | 0.522 | 0.377 | **0.887** |
| RMSE (m) | 0.145 | 0.150 | **0.081** |

**Persistence wins on all three.** What this does not establish: the sample is 21
windows inside a single 45-day stretch of one season, and 14-day persistence in a
quiescent Loop Current period is a strong baseline — the front barely moved. It is
a check that the pipeline runs and produces defensible numbers, not a verdict on
the method. A fair test needs the full 2023-2025 record (~1800 granules, ~55 GB)
so the sample spans eddy-shedding events, where persistence should degrade sharply
and an analog has something to beat it with.

Two things in the numbers are worth acting on:

* **The ensemble is worse than the best analog on front MHD** (43.0 vs 29.5) while
  being about equal on RMSE. This is the blurring already predicted in the
  `best_analog` combiner's docstring: averaging analogs whose fronts sit in
  different places produces a broad gradient that is nowhere, which reads fine to
  RMSE and badly to a front metric. Worth running `combiner: best_analog` for
  frontal targets.
* **One-day windows produce ensemble blow-ups** — 163.7 km and 155.4 km, both from
  the only two windows with a single observed day. With one swath the selection is
  weakly constrained and the K analogs disagree wildly. Raising
  `observation.require_days` to 2 would drop them.

### Earlier

The work used to live in one notebook, `download_glorys.ipynb`, whose second half
had drifted into analysis. It is now two pieces:

| File | Job |
|---|---|
| `download_glorys.ipynb` | Download GLORYS from Copernicus Marine and rechunk to Zarr. 14 cells, ends at "Read the Zarr store back". |
| `runs/glorys_analog.py` | Everything downstream: statistics, analog selection, forecast rollout. |

`../subset_glorys.py` is the path actually used for the data on disk — it subsets
the local GDEX mirror (`/gdex/data/d010049`) rather than downloading, and writes
one Zarr store per year.

## Data

33 per-year Zarr stores, `glorys_gom_{1993..2025}_subset.zarr`:

- 12,053 daily timesteps, 1993-01-01 to 2025-12-31
- Gulf of Mexico, 18–32 N, 98–78 W → 169 × 241 grid at 1/12°
- `thetao` (28 depth levels, 0–266 m) and `zos`
- `int16` on disk with scale/offset; xarray decodes to `float64`
- Timestamps carry a 12:00 time of day — this matters when selecting by date

SST is `thetao` at `depth=0`; SSH is `zos`.

## Pipeline

1. Open all years with `open_mfdataset`, restrict to `domain`, attach a `climday` coord
2. Summary statistics for SST and SSH
3. Diagnostics: spatial variance map, ±1/±2σ coverage, distribution, percentiles,
   spatial + temporal variation panels
4. Normalize both fields against a per-day, per-gridpoint climatology
5. Split into library (1993-01-01 – 2022-06-30) and target (2023-01-01 – 2025-12-31)
6. Rank library days against the target by `analogs.selection_metric`
7. Roll the top-K analogs forward daily to `forecast.length_days`
8. Score every forecast by all of `forecast.score_metrics`

## Design choices

### Configuration lives in YAML

All run parameters are in `config/*.yaml`, selected with `--config`. Paths inside
a config resolve against the repository root, not the config file or the working
directory, so a config can live anywhere and still find the same data.

`--config` is parsed with `parse_known_args` so the file still runs cell by cell
in Jupyter and VS Code, where `sys.argv` carries the kernel's own arguments.

`config/smoke_test.yaml` runs six years in about a minute, and
`tests/test_analogfc.py` runs on synthetic data in about a second. It replaces the
edit-the-script-and-hope loop that earlier changes were validated with.

### Climatology day: Feb 29 folded into Feb 28

`noleap_dayofyear` shifts day-of-year back by one for day 60 onward in leap years.
That does two things at once: Feb 29 (day 60) joins Feb 28 (day 59), and leap-year
Mar 1 (61 → 60) lands on non-leap Mar 1 (60), so the rest of the year stays
aligned. Result is 365 groups with equal sample counts.

Without this, day 366 is estimated from only the 8 leap years in the record. On a
shortened record it degenerates completely — a 5-year test window leaves day 366
with a single sample, zero standard deviation, and division by zero.

Exposed as a `climday` coordinate so `groupby` can use it directly.

### Normalization statistics are pointwise and per-day, and adjustable

```yaml
climatology:
  group: climday      # or "time.dayofyear", "time.month"
  reduce_dims: []     # [] = pointwise; ["latitude","longitude"] = domain-wide
```

`stat_dims = ["time", *reduce_dims]`, so an empty list gives one mean and one
standard deviation per group *per grid cell*. Both knobs were verified to work.

### Normalization fills a preallocated array group by group

The idiomatic form is `(da.groupby(g) - clim_mean).groupby(g) / clim_std`. It
builds a full-size temporary per operation — measured at 7.77 GB for a single
variable in float32, so roughly 15 GB in float64.

The version in the script computes the climatology, allocates one output array,
and fills it group by group. Each group's slice is a few MB, so peak stays at two
copies of the field. Verified bit-identical to the original implementation:
`max |old − new| = 0.0` over four years, with identical NaN masks.

An earlier attempt kept the array lazy and let dask handle the groupby. That is
worse, not better — `flox` is not installed, so xarray falls back to a path that
materializes the whole array during the shuffle.

### The Loop Current front is the 17 cm SSH contour

`front_mask` returns the inside edge of the region at or above
`analogs.ssh_contour_level` — cells above the level that are 4-adjacent to *water*
below it. Requiring the neighbour to be water keeps coastlines out of the mask,
which would otherwise trace every shore where high SSH meets land.

The 17 cm criterion (Leben 2005) is defined on altimetric SSH with its own mean
dynamic topography reference, and GLORYS `zos` is height above geoid, so the level
was checked before adoption rather than assumed. It transfers directly: the
0.17 m contour traces the extended Loop in 2010, 2015, and 2023, the retracted
state in 2020, and shed warm-core eddies in the western Gulf. No offset needed.

This replaced an SST gradient-magnitude threshold, and the change is not cosmetic.
The SST version put ~14,500 front points across the domain every day, so every day
looked alike: its full-record top five spanned 0.5523–0.5679 MHD, a 2.8% spread.
The SSH contour spans 3.72–4.44 across its top five, a 19% spread, and it selects
paired dates from single years rather than scattering across the record.

### MHD computed by distance transform, not pairwise distances

`scipy.spatial.distance.cdist` on two front point sets cost 1.27 s per pair at the
old SST threshold — about 4 hours over a 30-year library.

`scipy.ndimage.distance_transform_edt` on the complement of a front mask gives, at
every grid cell, the distance to the nearest front point. Reading it at the other
set's points yields the same nearest-neighbour distances in O(N). Cost per pair:
4 ms.

This is exact, not an approximation — both methods return 0.7336892639 on the same
test pair, to all printed digits. It is what makes scoring the entire library
viable rather than a ±30-day seasonal window.

### MHD runs on raw SSH, RMSD on standardized anomalies

The Loop Current front is a property of the physical SSH field. Normalizing
pointwise by a daily climatology removes the mean structure that defines it, so
the MHD branch reads raw SSH split by the same dates. The RMSD branch stays in
normalized space.

Both rankings go through the same `select_top_k_buffer` with a 14-day separation,
so the two lists are comparable.

### Forecast rollout stays in anomaly space

The first version averaged **raw** SST at `analog_date + lead`. That is wrong here.
Analogs are ranked in normalized-anomaly space, which is season-blind — a mid-June
target legitimately matches August and September days. Averaging their raw SST
imports each analog's own season, producing a bias that grows with lead.

The rollout averages the normalized anomaly at `analog_date + lead` and maps back
through the climatology of the **valid** day via `to_physical`. On a six-year test
this roughly halved the error at every lead:

| Lead | raw rollout | anomaly rollout |
|---|---|---|
| +0 d | 0.645 °C | 0.400 °C |
| +5 d | 1.315 °C | 0.608 °C |
| +10 d | 1.213 °C | 0.558 °C |
| +15 d | 1.291 °C | 0.577 °C |

This is why `normalize_by_climatology` returns `(clim_mean, clim_std, normalized)`.
Individual members are mapped back the same way, so the ensemble row is exactly
the mean of the member rows.

### Plot conventions

- SST fields: `plasma`, one shared scale so panels compare directly
- SSH fields: `RdBu_r`, symmetric about zero
- Error fields: `PuOr_r`, symmetric about zero — a different diverging pair from
  the SSH state panels, so state and error are never confused in one figure
- Error limits at the 99th percentile of |error|, not the max — a few coastal
  points run several degrees off and would otherwise flatten the whole field
- Analog members use a fixed categorical order, validated for colourblind
  separation (worst adjacent ΔE 9.1 protan, normal-vision floor 19.6). The
  ensemble is near-black and heavier, since it is an aggregate rather than a peer
  series. `K > 7` raises rather than cycling hues.

## Bugs found in the migrated code

1. **`select_top_k_buffer` raised on its first call.** `date - np.array([])`
   produces a float64 array, and numpy refuses to subtract it from a `datetime64`.
   Guarded with `if not selected_dates or all(...)`.
2. **`K` was read as a global from above its own definition.** Now a `k=` parameter.
3. **The top-K print loops formatted a 3-D field as a scalar.** `sst_library` was
   passed as `data`, so `top_sst.values` was a lat/lon field, not a score.
4. **Date selection missed by 12 hours.** Timestamps sit at 12:00, so
   `pd.to_datetime("2023-06-15")` lands on midnight and raises `KeyError`. The
   target stamp is now taken from the data.

## Environment

Run through `submit_analog_forecast.pbs` on Casper: 8 cpus, 128 GB, ~1m30s for
the full record. Interactive sessions default to 10 GB, which is not enough — see
`README.md` for the `qsub -I` line.

A float32 working cast was used while the workflow was confined to a 10 GB
interactive session. It has been dropped; there is no precision argument for it,
only a memory one, and memory is no longer the constraint. Six years in float64
peaks at 3.6 GB, which extrapolates to roughly 20 GB for the full record. That
figure is extrapolated, not measured — PBS reported zero for a job this short.
`/usr/bin/time` is now wired into the submission script so the next run records it.

`flox` is not installed, which is why dask groupby reductions behave badly.
`seaborn` is not installed; the distribution plot uses `plt.hist` rather than
`sns.histplot(kde=True)`, so the KDE curve is gone.

## Results — full record, target 2023-06-15

Job 5364629, 1m37s wall.

**Top-5 SST analogs (spatial RMSD on normalized anomalies)**

| Rank | Date | RMSD |
|---|---|---|
| 1 | 2020-08-20 | 0.7710 |
| 2 | 2020-07-12 | 0.8191 |
| 3 | 1998-08-26 | 0.8446 |
| 4 | 2009-09-16 | 0.8471 |
| 5 | 1997-04-01 | 0.8675 |

**Top-5 SSH analogs (spatial RMSD on normalized anomalies)**

| Rank | Date | RMSD |
|---|---|---|
| 1 | 2019-05-31 | 0.7395 |
| 2 | 2021-10-22 | 0.8450 |
| 3 | 2021-11-10 | 0.8740 |
| 4 | 2019-05-17 | 0.8900 |
| 5 | 2019-06-15 | 0.9165 |

**Top-5 Loop Current front analogs (MHD, grid units)**

| Rank | Date | MHD |
|---|---|---|
| 1 | 2001-09-02 | 3.7236 |
| 2 | 2021-10-16 | 3.8764 |
| 3 | 2021-09-28 | 4.2601 |
| 4 | 2001-10-03 | 4.2869 |
| 5 | 2017-08-19 | 4.4401 |

**SST forecast skill, K = 5** — 0.437, 0.759, 0.877, 1.087 °C at leads 0, 5, 10,
15. Monotonic with lead. Identical to the float32 run to three decimals.

**Record-wide statistics**

| | SST | SSH |
|---|---|---|
| Min | 6.99 °C | −1.25 m |
| Max | 35.86 °C | 1.51 m |
| Mean | 26.70 °C | 0.03 m |
| Std dev | 3.02 °C | 0.23 m |

SST percentiles: 5th 21.11, Q1 24.92, median 27.25, Q3 29.09, 95th 30.29 °C.
71.21% of grid points within ±1σ, 95.79% within ±2σ.

### Reading the results

**The rankings disagree, informatively.** SST RMSD selects July–September days,
matching the anomaly field. Loop Current MHD selects August–October days paired
within single years (two from 2001, two from 2021), matching front geometry. They
measure different things.

**The ensemble mean is not the best forecast.** In `analog_grid_ssh_mhd.png`,
analog 2 (2021-10-16) beats the five-member mean at every lead — 0.092 m against
0.113 m at lead 0, 0.124 against 0.135 at lead 15. Analog 3 (2021-09-28) also
beats it at short lead. Averaging displaced sharp fronts produces a blurred front
that matches neither, so equal-weight averaging is a poor estimator for a feature
this sharp. Worth testing rank-weighted or single-best-analog forecasts.

**MHD rank does not predict SSH forecast skill.** Rank 1 (2001-09-02) has among
the worst RMSE of the five; rank 2 has the best. MHD scores front geometry at
lead 0, RMSE scores the whole field over the forecast. A good initial front match
is not the same as a good forecast.

**The target case is a Loop Current eddy shedding event.** The observed row shows
the extended Loop pinching into a detached eddy by lead +10 to +15. The two 2021
analogs capture it; the 2001 and 2017 analogs do not.

**SST forecasts under-warm with lead**, in a coherent basin-wide pattern. Two
plausible contributors, not yet separated: 2023 was an exceptionally warm year in
the Gulf and the 1993–2022 library may not contain analogs warm enough; and the
ensemble mean regresses toward climatology by construction.

## Selection metric comparison — full record, target 2023-06-15

Three runs, identical except for `analogs.selection_metric`. Every run scores its
forecast by all five metrics. Ensemble scores averaged over daily leads 0–30.
Current domain 92.5–85 W, 18–32 N. Jobs 5364896/8/9.

| Selected by ↓ | sst_rmse ↓ | ssh_rmse ↓ | sst_acc ↑ | ssh_acc ↑ | ssh_front_mhd ↓ |
|---|---|---|---|---|---|
| `sst_rmsd` | 1.0128 | 0.2079 | 0.1425 | −0.2406 | 24.3192 |
| `ssh_rmsd` | **0.8083** | **0.1074** | **0.7638** | **0.7832** | 6.9608 |
| `ssh_front_mhd` | 0.8658 | 0.1085 | 0.6838 | 0.7534 | **6.3620** |

`ssh_rmsd` wins four of five. `ssh_front_mhd` takes front displacement, narrowly,
and is close behind on everything else. The two SSH-based selections are nearly
interchangeable; SST-based selection is not competitive.

### ACC separates the metrics where RMSE did not

On RMSE alone `ssh_rmsd` and `ssh_front_mhd` looked equivalent (0.1074 vs 0.1085
on SSH). ACC pulls them apart and, more importantly, exposes how badly `sst_rmsd`
fails:

- `sst_rmsd` gives **ssh_acc = −0.24**: the SSH anomaly pattern is *anti*-correlated
  with observations, worse than forecasting climatology.
- `sst_rmsd` gives **sst_acc = 0.14** — almost no anomaly skill even for the field
  it selected on.
- `ssh_rmsd` gives sst_acc = 0.76, better SST anomaly skill than selecting on SST
  by a factor of five.

Matching SST anomalies selects days with similar surface temperature patterns and
no constraint on circulation. Matching SSH anomalies selects days with a similar
circulation state, which then governs both fields over the following month.

### Results move with the domain and forecast length

This is the fourth configuration tested, and the ranking has changed each time:

| Configuration | Winner |
|---|---|
| Full domain 98–78 W, 15 d | `ssh_rmsd` on all three |
| 90 W cut, 15 d | `ssh_front_mhd` on all three |
| 92.5–85 W, 30 d | `ssh_rmsd` and `ssh_front_mhd` near-tied |
| 92.5–85 W, 30 d, with ACC | `ssh_rmsd` on four of five |

The domain and lead range move the answer about as much as the metric choice
does. From one target date, the safe conclusion is the qualitative one — SSH-based
selection beats SST-based selection, decisively and in every configuration — not
the ordering between the two SSH metrics.

### Ensemble averaging

An earlier full-domain run had the best single member beating the ensemble on
`ssh_front_mhd` by 3–4× in every case, which looked like clean evidence that
averaging smears a sharp feature. It does not generalise. Under the current
domain, with ACC:

| Selected by | metric | ensemble | best member |
|---|---|---|---|
| `ssh_rmsd` | sst_acc | **0.7638** | 0.6649 |
| `ssh_rmsd` | ssh_acc | **0.7832** | 0.7517 |
| `ssh_front_mhd` | sst_acc | 0.6838 | **0.7062** |
| `ssh_front_mhd` | ssh_front_mhd | **6.3620** | 6.4751 |
| `sst_rmsd` | ssh_acc | −0.2406 | **0.0585** |

Averaging helps when members agree on the circulation state and hurts when they
do not. Under `sst_rmsd` the members disagree badly enough that the ensemble is
worse than its worst-case member on ACC.

### Front displacement jumps mid-forecast

In the `ssh_front_mhd` run the ensemble front error sits near 2.6–2.8 grid cells
through lead 10, then jumps to 9.1 at lead 15 and stays near 8.5–9.4 to lead 30.
A step that sharp is a topology change — the Loop Current pinching off an eddy —
not gradual drift. MHD compares point sets, so gaining or losing a closed contour
moves it discontinuously. Worth knowing when reading the curve.

`ssh_front_mhd` as a *selection* score and as an *error* score are not directly
comparable: selection compares raw library fields against the raw target, while
scoring compares climatology-remapped forecasts against observations.

## Open items

- **Climatology leakage.** The climatology is computed over the whole record,
  including the 2023–2025 target period, so the target leaks into the baseline it
  is measured against. This matters more now that the forecast is mapped back
  through that same climatology. Restricting it to `periods.library` is a one-line
  change and would likely make the cold bias larger and more honest.
- **No baseline.** Persistence and climatology forecasts would say whether any of
  the skill above is worth anything.
- **One target date only.** Skill from a single 2023-06-15 case is anecdotal.
- **Equal-weight ensemble.** Helps on the eastern Gulf where members agree on
  front position, hurts badly under `sst_rmsd` where they do not. Rank weighting
  or combining per-member contours rather than fields may be more robust than
  either.
- **The 90 W cut was chosen, not tuned.** It changes which selection metric wins,
  so the result is sensitive to it. Worth checking a couple of other bounds
  before treating the ranking as settled.
- **A coastal outlier near 29.5 N, 89 W** runs several degrees off at every lead —
  probably a river-mouth cell worth masking.
- **The summary-statistics block makes ten full passes** over the data (five
  separate `.compute()` calls per variable). One pass would do.
- **The Caribbean segment of the 17 cm contour** is nearly always present and
  largely static, so it contributes little discrimination to the MHD. Restricting
  the front to a Gulf sub-box may sharpen the ranking.
