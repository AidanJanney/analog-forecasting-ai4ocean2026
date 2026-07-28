"""Advancing a selection into a forecast, lead by lead.

Each analog is followed forward through its own history: the forecast at lead L
is what actually happened L days after that analog day. Members are carried in
standardized-anomaly space and mapped back through the climatology of the day
being *forecast*, not the analog's own day — an October analog forecasting a June
target must be re-expressed against June's climatology or it carries October's
seasonal cycle with it.

That mapping is affine, which is why the ensemble may be combined in either
space and give the same field.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .combine import COMBINERS


@dataclass
class Rollout:
    """A forecast at every lead: the truth, each member, and the combination.

    ``truth[lead][var]``, ``members[lead][member][var]``, ``ensemble[lead][var]``,
    all in physical units.
    """

    leads: list
    truth: dict
    members: dict
    ensemble: dict
    valid_times: dict
    analogs: object

    def __len__(self):
        return len(self.leads)


def member_forecast(library, var, analog_time, lead, valid_time):
    """One analog's own forecast of `var` at `lead`, in physical units."""
    field = library.fields[var]
    anomaly = field.standardized.sel(
        time=pd.Timestamp(analog_time) + pd.Timedelta(days=lead))
    return field.to_physical(anomaly, valid_time)


def rollout(library, analogs, target_time, leads, variables=None,
            combiner="ensemble_mean"):
    """Forecast every variable, for every member, at every lead.

    `combiner` names an entry in :data:`~.combine.COMBINERS`.
    """
    variables = list(library.fields) if variables is None else list(variables)
    combine = COMBINERS.get(combiner)
    target_time = pd.Timestamp(target_time)

    truth, members, ensemble, valid_times = {}, {}, {}, {}
    for lead in leads:
        valid = target_time + pd.Timedelta(days=lead)
        valid_times[lead] = valid
        truth[lead] = {v: library.at(v, valid).compute() for v in variables}
        members[lead] = [
            {v: member_forecast(library, v, t, lead, valid) for v in variables}
            for t in analogs.times]
        ensemble[lead] = {
            v: combine([m[v] for m in members[lead]], analogs.distances)
            for v in variables}
    return Rollout(list(leads), truth, members, ensemble, valid_times, analogs)


def report_skill(rollout, skill, metrics, selection_metric, k, baseline=None):
    """Print the per-lead ensemble table and the best-member comparison.

    `baseline` is an optional ``{metric: per-lead curve}`` from
    :func:`score_baseline` — the persistence comparison, reported last.
    """
    print(f"\n--- Ensemble skill by lead (selection: {selection_metric}, K = {k}) ---")
    print("lead  " + "  ".join(f"{m.name:>16s}" for m in metrics))
    print("      " + "  ".join(f"{'(' + m.unit + ')':>16s}" for m in metrics))
    for j, lead in enumerate(rollout.leads):
        row = "  ".join(f"{skill[m.name][1][j]:>16.4f}" for m in metrics)
        print(f"{lead:>4d}  {row}")

    print("\n--- Best single member vs ensemble, by metric ---")
    for metric in metrics:
        per_member, ensemble_curve = skill[metric.name]
        member_means = [np.mean(curve) for curve in per_member]
        # A skill score ranks the opposite way round from an error metric.
        best = int(np.argmax(member_means) if metric.higher_is_better
                   else np.argmin(member_means))
        print(f"{metric.name:>16s}: ensemble {np.mean(ensemble_curve):.4f}  |  "
              f"best member {rollout.analogs.dates[best]} (rank {best + 1}) "
              f"{member_means[best]:.4f} {metric.unit}")

    if baseline is None:
        return
    print("\n--- vs persistence (mean over the scored leads) ---")
    for metric in metrics:
        _, ensemble_curve = skill[metric.name]
        ens, per = np.mean(ensemble_curve), np.nanmean(baseline[metric.name])
        beats = (ens > per) if metric.higher_is_better else (ens < per)
        print(f"{metric.name:>16s}: ensemble {ens:.4f}  |  persistence {per:.4f}  "
              f"{metric.unit}  -> ensemble {'beats' if beats else 'loses to'} persistence")


def persistence_fields(library, target_time, variables):
    """The observed state at lead 0 — the persistence forecast for every lead.

    The baseline any forecast has to beat: assume nothing changes. Over a
    quiescent Loop Current period it is a strong one, which is exactly why it
    belongs in every report rather than only in the runs where it loses.
    """
    return {v: library.at(v, target_time).compute() for v in variables}


def score_baseline(rollout, metrics, fields):
    """Score one fixed field set against the truth at every lead."""
    return {m.name: [m(fields, rollout.truth[lead], rollout.valid_times[lead])
                     for lead in rollout.leads] for m in metrics}


def score(rollout, metrics):
    """Score every member and the ensemble at every lead.

    Returns ``{metric_name: (per_member, ensemble_curve)}`` where `per_member` is
    one curve per analog over the rollout's leads.
    """
    k = len(rollout.analogs)
    out = {}
    for metric in metrics:
        per_member = [
            [metric(rollout.members[lead][i], rollout.truth[lead],
                    rollout.valid_times[lead]) for lead in rollout.leads]
            for i in range(k)]
        ensemble_curve = [metric(rollout.ensemble[lead], rollout.truth[lead],
                                 rollout.valid_times[lead]) for lead in rollout.leads]
        out[metric.name] = (per_member, ensemble_curve)
    return out
