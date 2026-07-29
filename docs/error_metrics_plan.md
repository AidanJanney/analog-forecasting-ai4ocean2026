# Diagnostic error metrics for the analog forecasting system

**Branch:** `with_errors_RKF`
**Status:** proposed, implementation in `analogfc/diagnostics/`
**Scope:** read-only diagnostics consuming existing outputs (`Rollout`, `AnalogSet`,
`ObsWindow`, `ModelLibrary`). No changes to `rollout.py`, `score.py`, `select.py`,
`windows.py`, or any other file under active development by collaborators.

## 1. Motivation

The repository already reports total forecast error at each lead
(`forecast.score.score()` → `sst_rmse`, `ssh_rmse`, `sst_acc`, `ssh_acc`,
`ssh_front_mhd`), and compares against a persistence baseline
(`rollout.score_baseline`). That answers *"how big is the error"* but not *why*:
whether error comes from a bad initial match, a bad predicted evolution, or a
combination that happens to cancel and look better than it is.

This document defines three additional diagnostics that decompose total error
into interpretable pieces, and specifies exactly what data each one consumes
from the existing pipeline. They are implemented in
`analogfc/diagnostics/error_metrics.py` as pure functions operating on a
`Rollout` (plus, for one function, an `ObsWindow`/`ModelLibrary` pair) — nothing
they do can affect analog selection, rollout, or scoring.

## 2. Definitions

All three metrics reuse the **latitude-weighted squared-error accounting**
already implemented in `WeightedRMSE` (`analogfc/forecast/score.py`): the same
`cos(latitude)` weight array (`library.weights`), the same finite/region mask
combination, and the same `{var: field}` calling convention. No weighting logic
is reimplemented; existing `ErrorMetric` instances are called directly wherever
possible.

Let `X_a(L)` be an analog member's (or the ensemble's) forecast of a variable at
lead `L`, and `X_t(L)` the corresponding truth (`rollout.truth[L]`), both from
`rollout.py`'s `rollout()` output. `||·||_W` denotes the weighted RMS defined by
`WeightedRMSE`: `sqrt( Σ w_i (·)_i^2 / Σ w_i )` over finite, in-region cells.

### 2.1 Lead-zero reconstruction error — `E_recon`

```
E_recon = || X_a(0) − X_t(0) ||_W
```

How well the analog's own initial state matches truth at lead 0, before any
forecast dynamics are applied. This is the "sanity/bug check" metric from the
original experiment list: for GLORYS→GLORYS with the target day *in* the
library, `E_recon` should be ~0 by construction (the analog's own history is the
truth); for held-out GLORYS or SWOT/GOES-initialized runs it need not be, and a
large value here means the *selection* step failed, independent of anything the
forecast dynamics do afterward.

**Obs-vs-reanalysis variant** (`reconstruction_vs_obs`): for the SWOT/GOES
workflows, this also directly compares the *observation itself*
(`ObsWindow.composite`, masked to `ObsWindow.coverage_mask`) against the GLORYS
truth field at the window's end date (`library.at(var, valid_time)`) —
independent of any analog. This isolates instrument/sampling error from analog
selection error: if the raw obs already disagrees with reanalysis before any
analog is chosen, no selection or dynamics change can fix it.

**Implementation:** `reconstruction_error(rollout, metrics)` and
`reconstruction_vs_obs(window, library, metrics, valid_time)` in
`error_metrics.py`. Reuses `metric.__call__` unmodified — the only new logic is
selecting lead 0 and constructing the obs-composite input.

### 2.2 Analog tendency (increment) error — `E_tend`

```
ΔX_a(L) = X_a(L) − X_a(0)          ΔX_t(L) = X_t(L) − X_t(0)
E_tend(L) = || ΔX_a(L) − ΔX_t(L) ||_W
```

How far off the analog's *predicted evolution* is from the true evolution, with
the lead-0 mismatch subtracted out first. This isolates the dynamical/analog
skill from the initial-condition skill: an analog can start from the wrong state
(`E_recon` large) but still correctly predict the *shape* of the change
(`E_tend` small), or vice versa. Reported as a curve over `leads != 0`.

**Implementation:** `tendency_error(rollout, metrics)`. Builds the increment
dicts explicitly (`_diff_dict`) and calls the existing `metric.__call__` on
them — `WeightedRMSE` doesn't need to know it's being handed an increment
rather than a raw field, since it only computes a weighted squared difference
of whatever two `{var: array}` dicts it receives.

### 2.3 Cross term (cancellation diagnostic)

The total forecast error at lead `L`,

```
E_total(L) = || X_a(L) − X_t(L) ||_W
```

(already computed by `score.score()`) decomposes, in the weighted-squared-error
sense, as

```
E_total(L)^2  ≈  E_recon^2 + E_tend(L)^2 + cross(L)

cross(L) = 2 · < X_a(0) − X_t(0),  ΔX_a(L) − ΔX_t(L) >_W
```

where `<·,·>_W` is the weighted inner product with the same weights and mask as
`E_recon`/`E_tend`. `cross(L)` is the diagnostic of interest: a large-magnitude
**negative** cross term means the initial-condition error and the tendency
error point in *opposite* directions and partially cancel — the forecast looks
better (`E_total` smaller) than either error component alone would suggest,
which is exactly the "errors cancel and prop up a false high RMSE" case from the
original notes. A large **positive** cross term means the two errors compound.

This decomposition is only meaningful for the squared-error family
(`WeightedRMSE`); it does not apply to `AnomalyCorrelation` (not a squared-error
quantity) or `FrontMHDError` (geometric, non-additive). `cross_term()` therefore
silently restricts itself to metrics that are `isinstance(m, WeightedRMSE)` and
skips the rest — this is a mathematical constraint, not an implementation
shortcut.

**Implementation:** `cross_term(rollout, metrics)`. Reuses `metric.w` and
`metric.region_mask` (already computed once at metric-construction time) rather
than recomputing weights.

### 2.4 Verification, not assumption

Because the decomposition above is an approximation only in that it assumes
`E_total`, `E_recon`, `E_tend` are RMS quantities being combined as if they were
plain sums of squares — i.e., the identity
`E_total^2 = E_recon^2 + E_tend^2 + cross` holds **exactly** for a single grid
cell's squared error but only approximately once different cells' weighted
means are combined this way, because RMS does not distribute over sums the way
plain sums of squares do. **This will be numerically checked**, not assumed:
`tests/test_error_metrics.py` (to be written next) constructs a synthetic
`Rollout` and checks the identity numerically to confirm how close the
approximation is in practice, and whether a per-cell (unaggregated) version of
the decomposition is needed instead for the cross-term map described below.

## 3. What each metric needs as input (mapped to existing repo objects)

| Metric | Needs from `Rollout` | Needs from elsewhere |
|---|---|---|
| `E_recon` | `truth[0]`, `members[0]`, `ensemble[0]`, `valid_times[0]` | existing `metrics` list (`score.build(...)`) |
| `E_recon` (obs variant) | — | `ObsWindow.composite`, `.coverage_mask`; `ModelLibrary.at(var, valid_time)` |
| `E_tend` | `truth[0]`, `truth[L]`, `members[0]`, `members[L]`, `ensemble[0]`, `ensemble[L]` for each `L` in `leads` | same `metrics` list |
| cross term | same as `E_tend`, plus `metric.w`, `metric.region_mask` | none beyond `WeightedRMSE` internals |

All are read directly off objects `rollout.py`, `select.py`, and `windows.py`
already construct; no new fields are added to any collaborator dataclass.

## 4. Experiment coverage

Per the original four experiments, here is what these three metrics answer in
each case:

1. **GLORYS→GLORYS, trained-in library (sanity check).** `E_recon` should be
   ≈0 (target day *is* in its own history) — if not, something in selection or
   rollout is broken. `E_tend`/cross term meaningful only insofar as `k>1`
   analogs are combined; with the exact self-match at rank 1, ensemble
   dilution from other members is itself a diagnostic.

2. **Held-out GLORYS from GLORYS.** `E_recon` measures how well analog
   selection alone can match an unseen day using only `analogs.distance` (no
   dynamics involved yet). `E_tend` measures whether the chosen analogs, even
   if imperfectly matched at t0, still predict the right *change*. Cross term
   flags whether apparent skill is genuine or cancellation.

3. **SWOT→GLORYS (in/out of library).** `E_recon` (obs variant) isolates
   sampling/instrument disagreement with GLORYS from selection error;
   `E_recon` (analog variant) isolates selection error given the sparse
   observation; `E_tend`/cross term as above. This is the most informative
   split, since sparse coverage is the dominant new source of error.

4. **SWOT + GOES → GLORYS.** Same as (3), rerun with an added variable —
   compare `E_recon`/`E_tend` curves against (3) to quantify GOES SST's
   marginal contribution to reconstruction vs. tendency skill separately
   (rather than only the combined `E_total`, which cannot distinguish which
   piece GOES actually helped).

## 5. Known gaps / blockers

- `analogfc/data/regions.py` (providing `regions.combine`) is imported by
  `score.py`, `anomaly_rmsd.py`, and `correlation.py` but does not exist
  anywhere in the branch's git history (`git log --all -- "*regions.py"` is
  empty). This blocks running `score.py` at all currently, not just these
  diagnostics. `error_metrics.py` includes a local `_combine_mask` fallback
  for standalone testing only — **not** a substitute for the real module,
  which the team needs to add or restore.

## 6. Deliverables checklist

- [x] `analogfc/diagnostics/error_metrics.py` — `reconstruction_error`,
      `reconstruction_vs_obs`, `tendency_error`, `cross_term`
- [x] `docs/error_metrics_plan.md` — this document
- [ ] `tests/test_error_metrics.py` — synthetic-data numerical check of the
      decomposition identity (§2.4), following the style of
      `tests/test_analogfc.py` (no data files, plain `python` entry point)
- [ ] `analogfc/diagnostics/plotting.py` — time series of `E_recon`/`E_tend`/
      cross-term across leads; scatter of `E_recon` vs `E_tend` colored by
      cross-term sign; spatial map of the per-cell cross-term product
      (`w_i(X_a(0)-X_t(0))_i(ΔX_a(L)-ΔX_t(L))_i`, before summation) to localize
      *where* cancellation occurs rather than only its domain-mean value
- [ ] Resolution of the `regions.py` gap (owned by the team, not this
      diagnostics work)