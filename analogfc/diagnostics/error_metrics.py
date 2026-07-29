"""Diagnostic error metrics built on top of `forecast.rollout.Rollout`.

Three metrics, all read-only consumers of existing outputs — no changes to
`rollout.py`, `score.py`, `select.py`, or `windows.py`.

    E_recon(0)  : lead-zero reconstruction error (analog vs truth at t0)
    E_tend(L)   : analog tendency error (increment mismatch, t0 -> lead L)
    cross(L)    : 2 * <initial mismatch, tendency mismatch>_W

so that, for each metric's own weighted-squared-error accounting,

    E_total(L)^2  ~=  E_recon(0)^2 + E_tend(L)^2 + cross(L)

A large-magnitude negative `cross(L)` flags a forecast whose low RMSE is a
cancellation of an initial-condition error against a compensating tendency
error, rather than genuine skill.

Every function requires `0` to be in `rollout.leads` — lead zero is the
reference state and the increment baseline. This is asserted, not handled
silently: a rollout that never carries lead 0 cannot support these metrics.
"""

from __future__ import annotations

import numpy as np

from ..forecast.score import ErrorMetric, WeightedRMSE


def _require_lead_zero(rollout):
    if 0 not in rollout.leads:
        raise ValueError(
            "reconstruction/tendency/cross-term diagnostics need lead 0 in "
            f"rollout.leads; got {rollout.leads}. Re-run rollout() with 0 "
            "included among `leads`."
        )


def _field_dict(fields):
    """Pass through a `{var: array-like}` mapping unchanged (already the shape
    every ErrorMetric expects)."""
    return fields


def _diff_dict(a, b, var):
    """{var: a[var] - b[var]} as plain ndarrays, matching what ErrorMetric.__call__
    expects from `forecast`/`truth`."""
    av = np.asarray(a[var].values if hasattr(a[var], "values") else a[var], dtype=float)
    bv = np.asarray(b[var].values if hasattr(b[var], "values") else b[var], dtype=float)
    return {var: av - bv}


# --------------------------------------------------------------------------- #
# 1. Lead-zero reconstruction error.
# --------------------------------------------------------------------------- #
def reconstruction_error(rollout, metrics):
    """Score every member and the ensemble against truth, at lead 0 only.

    Returns ``{metric_name: (per_member_value, ensemble_value)}`` — a single
    number per member/ensemble rather than a curve, since there is only one
    lead involved.
    """
    _require_lead_zero(rollout)
    valid0 = rollout.valid_times[0]
    truth0 = rollout.truth[0]
    k = len(rollout.analogs)
    out = {}
    for metric in metrics:
        per_member = [metric(rollout.members[0][i], truth0, valid0) for i in range(k)]
        ens = metric(rollout.ensemble[0], truth0, valid0)
        out[metric.name] = (per_member, ens)
    return out


def reconstruction_vs_obs(window, library, metrics, valid_time):
    """Compare the observation itself (not any analog) against reanalysis at t0.

    For the SWOT/GOES workflows: is the *observation* consistent with GLORYS at
    the window's end date, restricted to the cells actually observed? Uses
    `window.composite` masked to `window.coverage_mask`, so unobserved cells
    never enter the weighted mean (`ErrorMetric` already drops non-finite
    values via its own `ok` mask).

    `library.at(var, valid_time)` supplies the truth field the same way
    `rollout.py` does.
    """
    obs = window.composite
    obs_masked = np.where(window.coverage_mask, obs, np.nan)
    out = {}
    for metric in metrics:
        forecast = {metric.var: obs_masked}
        truth = {metric.var: library.at(metric.var, valid_time).compute()}
        out[metric.name] = metric(forecast, truth, valid_time)
    return out


# --------------------------------------------------------------------------- #
# 2. Analog tendency (increment) error.
# --------------------------------------------------------------------------- #
def tendency_error(rollout, metrics):
    """Score the increment (t0 -> lead) mismatch, per member and ensemble.

    Returns ``{metric_name: (per_member, ensemble_curve)}`` over
    ``rollout.leads`` excluding lead 0 (the increment from lead 0 to itself is
    identically zero and not informative).
    """
    _require_lead_zero(rollout)
    k = len(rollout.analogs)
    leads = [L for L in rollout.leads if L != 0]
    out = {}
    for metric in metrics:
        var = metric.var
        per_member = []
        for i in range(k):
            curve = []
            for L in leads:
                analog_incr = _diff_dict(rollout.members[L][i], rollout.members[0][i], var)
                true_incr = _diff_dict(rollout.truth[L], rollout.truth[0], var)
                curve.append(metric(analog_incr, true_incr, rollout.valid_times[L]))
            per_member.append(curve)
        ensemble_curve = []
        for L in leads:
            analog_incr = _diff_dict(rollout.ensemble[L], rollout.ensemble[0], var)
            true_incr = _diff_dict(rollout.truth[L], rollout.truth[0], var)
            ensemble_curve.append(metric(analog_incr, true_incr, rollout.valid_times[L]))
        out[metric.name] = (per_member, ensemble_curve)
    return out, leads


# --------------------------------------------------------------------------- #
# 3. Initial-mismatch x tendency-mismatch cross term.
# --------------------------------------------------------------------------- #
def _combine_mask(finite_mask, region_mask):
    """Local stand-in for `data.regions.combine`, which does not exist in this
    branch yet (`score.py` imports it but the module is absent/uncommitted).

    Matches the two-argument contract `score.py`'s metrics rely on: finite
    cells, further restricted to `region_mask` if one is given.
    """
    if region_mask is None:
        return finite_mask
    return finite_mask & np.asarray(region_mask, dtype=bool)


def _weighted_inner(a, b, w, region_mask):
    """Sum(w * a * b) / Sum(w) over finite, in-region cells."""
    ok = _combine_mask(np.isfinite(a) & np.isfinite(b), region_mask)
    w = np.broadcast_to(w, a.shape)
    total = w[ok].sum()
    if total <= 0:
        return np.nan
    return float((w[ok] * a[ok] * b[ok]).sum() / total)


def cross_term(rollout, metrics):
    """2 * <initial mismatch, tendency mismatch>_W, per member and ensemble.

    Only defined for `WeightedRMSE`-family metrics, since it needs a weight
    array (`metric.w`) and squared-error semantics; skips any metric that
    lacks a `.w` attribute (e.g. `FrontMHDError`, which is geometric, not a
    weighted-squared-error quantity that decomposes this way).

    Returns ``{metric_name: (per_member, ensemble_curve)}`` over leads != 0,
    plus a `check` entry per metric giving `E_recon^2 + E_tend^2 + cross`
    alongside the actual `E_total^2` from `score.score()`, so the
    decomposition can be verified rather than trusted.
    """
    _require_lead_zero(rollout)
    k = len(rollout.analogs)
    leads = [L for L in rollout.leads if L != 0]
    out = {}
    for metric in metrics:
        if not isinstance(metric, WeightedRMSE):
            continue
        var = metric.var
        w, region_mask = metric.w, metric.region_mask

        def _initial_mismatch(analog0, truth0):
            a = np.asarray(analog0[var].values if hasattr(analog0[var], "values")
                           else analog0[var], dtype=float)
            t = np.asarray(truth0[var].values if hasattr(truth0[var], "values")
                           else truth0[var], dtype=float)
            return a - t

        per_member = []
        for i in range(k):
            init_mismatch = _initial_mismatch(rollout.members[0][i], rollout.truth[0])
            curve = []
            for L in leads:
                analog_incr = np.asarray(
                    rollout.members[L][i][var].values
                    if hasattr(rollout.members[L][i][var], "values")
                    else rollout.members[L][i][var], dtype=float
                ) - np.asarray(
                    rollout.members[0][i][var].values
                    if hasattr(rollout.members[0][i][var], "values")
                    else rollout.members[0][i][var], dtype=float
                )
                true_incr = np.asarray(
                    rollout.truth[L][var].values
                    if hasattr(rollout.truth[L][var], "values")
                    else rollout.truth[L][var], dtype=float
                ) - np.asarray(
                    rollout.truth[0][var].values
                    if hasattr(rollout.truth[0][var], "values")
                    else rollout.truth[0][var], dtype=float
                )
                tend_mismatch = analog_incr - true_incr
                curve.append(2.0 * _weighted_inner(init_mismatch, tend_mismatch, w, region_mask))
            per_member.append(curve)

        init_mismatch_ens = _initial_mismatch(rollout.ensemble[0], rollout.truth[0])
        ensemble_curve = []
        for L in leads:
            analog_incr = np.asarray(
                rollout.ensemble[L][var].values if hasattr(rollout.ensemble[L][var], "values")
                else rollout.ensemble[L][var], dtype=float
            ) - np.asarray(
                rollout.ensemble[0][var].values if hasattr(rollout.ensemble[0][var], "values")
                else rollout.ensemble[0][var], dtype=float
            )
            true_incr = np.asarray(
                rollout.truth[L][var].values if hasattr(rollout.truth[L][var], "values")
                else rollout.truth[L][var], dtype=float
            ) - np.asarray(
                rollout.truth[0][var].values if hasattr(rollout.truth[0][var], "values")
                else rollout.truth[0][var], dtype=float
            )
            tend_mismatch = analog_incr - true_incr
            ensemble_curve.append(2.0 * _weighted_inner(init_mismatch_ens, tend_mismatch, w, region_mask))

        out[metric.name] = (per_member, ensemble_curve)
    return out, leads