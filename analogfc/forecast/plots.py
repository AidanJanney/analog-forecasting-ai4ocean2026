"""Section 3 figures: what the forecast looked like and how well it did.

Each function takes already-computed inputs — a :class:`~.rollout.Rollout`, the
scored skill curves, the metric objects — so the orchestration stays in the
driver and a figure can be reproduced from a saved result without re-running the
pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt

# Categorical hues for the individual analog members, assigned in fixed order.
# Validated for colourblind separation against a light surface.
ANALOG_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
ENSEMBLE_COLOR = "#0b0b0b"      # the ensemble is an aggregate, not a peer series


def panel_size(fields, height=2.4):
    """(width, height) for one map panel, following the domain's aspect ratio.

    Width is clamped so a narrow longitude band does not come out as an
    unreadable sliver, and a wide one does not crowd out the skill curves.
    """
    lon, lat = fields.longitude, fields.latitude
    aspect = float(lon.max() - lon.min()) / float(lat.max() - lat.min())
    return min(max(height * aspect, 1.7), 4.0), height


def _strip_ticks(ax, keep_x, keep_y):
    """Interior panels of a dense grid carry no tick labels; the edges do."""
    ax.set_xlabel("")
    if not keep_x:
        ax.set_xticklabels([])
    if not keep_y:
        ax.set_yticklabels([])


def plot_analog_grid(var, library, rollout, skill, metrics, map_leads,
                     target_date, selection_metric, fig_dir, filename,
                     contour_level=None, obs=None):
    """One row per analog (state, then error), then the ensemble, then the truth.

    Columns are `map_leads`, optionally preceded by the observation that drove the
    selection. The final column carries one panel per scoring metric, each showing
    every member and the ensemble against daily lead — so a member that looks good
    on the maps at one lead can be checked against its whole trajectory.

    For a field with a `contour_level` the observed front is drawn on every panel
    and each member's own front dashed over it, so displacement reads directly off
    the panel instead of having to be inferred from the colour difference.
    """
    field = library.fields[var]
    analogs = rollout.analogs
    n = len(analogs)
    if n > len(ANALOG_COLORS):
        raise ValueError(f"{n} analogs but only {len(ANALOG_COLORS)} categorical colours defined")

    truths = [rollout.truth[lead][var] for lead in map_leads]
    states = [[rollout.members[lead][i][var] for lead in map_leads] for i in range(n)]
    states.append([rollout.ensemble[lead][var] for lead in map_leads])
    states.append(truths)
    errors = [[states[r][j] - truths[j] for j in range(len(map_leads))]
              for r in range(n + 1)]

    row_labels = [f"Analog {i + 1}\n{d}" for i, d in enumerate(analogs.dates)]
    row_labels += [f"Ensemble mean\nK = {n}", f"Observed\n{target_date}"]

    state_min = min(float(f.min()) for row in states for f in row)
    state_max = max(float(f.max()) for row in states for f in row)
    if field.symmetric:
        state_max = max(abs(state_min), abs(state_max))
        state_min = -state_max
    err_max = max(float(np.nanpercentile(np.abs(f.values), 99)) for row in errors for f in row)

    panel_w, panel_h = panel_size(library.fields)
    has_obs = obs is not None
    n_map_cols = len(map_leads) + (1 if has_obs else 0)
    n_rows = 2 * n + 3          # state+error per analog, state+error for the ensemble, truth
    n_cols = n_map_cols + 2     # map columns, a spacer, then the skill curves
    # A trailing short row holds the two shared colorbars, and an empty spacer
    # column keeps the curve tick labels off the last map column. The curve column
    # gets a floor in absolute inches so it stays readable when panels are narrow.
    curve_ratio = max(1.5, 3.2 / panel_w)
    fig = plt.figure(
        figsize=(panel_w * (n_map_cols + 0.3 + curve_ratio) + 1.0, panel_h * n_rows + 1.0))
    gs = fig.add_gridspec(
        n_rows + 1, n_cols,
        height_ratios=[1] * n_rows + [0.18],
        width_ratios=[1] * n_map_cols + [0.3, curve_ratio],
        hspace=0.14, wspace=0.06,
    )

    state_art = err_art = None
    last_row = 2 * (n + 1)      # the truth row, which keeps its x tick labels
    for r in range(n + 2):      # analogs, ensemble, truth
        state_row = 2 * r
        colour = ANALOG_COLORS[r] if r < n else ENSEMBLE_COLOR

        if has_obs:             # the observation that drove the selection
            ax = fig.add_subplot(gs[state_row, 0])
            obs.plot(ax=ax, cmap=field.cmap, vmin=state_min, vmax=state_max,
                     add_colorbar=False)
            if contour_level is not None and r < n + 1:
                ax.contour(states[r][0].longitude, states[r][0].latitude,
                           states[r][0].values, levels=[contour_level],
                           colors=[colour], linewidths=1.2, linestyles="--")
            ax.set_title(f"observation\n{target_date}" if state_row == 0 else "")
            ax.set_ylabel(row_labels[r])
            _strip_ticks(ax, keep_x=state_row == last_row, keep_y=True)

        for j, lead in enumerate(map_leads):
            col = j + (1 if has_obs else 0)
            ax = fig.add_subplot(gs[state_row, col])
            state_art = states[r][j].plot(ax=ax, cmap=field.cmap, vmin=state_min,
                                          vmax=state_max, add_colorbar=False)
            if contour_level is not None:
                ax.contour(truths[j].longitude, truths[j].latitude, truths[j].values,
                           levels=[contour_level], colors="k", linewidths=1.2)
                if r < n + 1:   # the member's own front, to read displacement against truth
                    ax.contour(states[r][j].longitude, states[r][j].latitude,
                               states[r][j].values, levels=[contour_level],
                               colors=[colour], linewidths=1.2, linestyles="--")
            ax.set_title(f"lead +{lead} d" if state_row == 0 else "")
            ax.set_ylabel(row_labels[r] if col == 0 else "")
            _strip_ticks(ax, keep_x=state_row == last_row, keep_y=col == 0)

            if r < n + 1:       # truth has no error row
                ax_e = fig.add_subplot(gs[state_row + 1, col])
                err_art = errors[r][j].plot(ax=ax_e, cmap="PuOr_r", vmin=-err_max,
                                            vmax=err_max, add_colorbar=False)
                ax_e.set_title("")
                ax_e.set_ylabel(f"error ({field.unit})" if col == 0 else "")
                _strip_ticks(ax_e, keep_x=False, keep_y=col == 0)

    # Two shared colorbars in the trailing row. Distinct diverging pairs, so a state
    # panel is never mistaken for an error panel at a glance.
    half = max(1, n_map_cols // 2)
    fig.colorbar(state_art, cax=fig.add_subplot(gs[n_rows, :half]), orientation="horizontal",
                 label=f"State ({field.unit})")
    fig.colorbar(err_art, cax=fig.add_subplot(gs[n_rows, half:n_map_cols]),
                 orientation="horizontal", label=f"Forecast − observed ({field.unit})")

    # Final column: one panel per scoring metric, evaluated daily.
    n_metrics = len(metrics)
    block = n_rows // n_metrics
    for mi, metric in enumerate(metrics):
        r0 = mi * block
        r1 = n_rows if mi == n_metrics - 1 else (mi + 1) * block
        ax = fig.add_subplot(gs[r0:r1, -1])
        per_member, ensemble_curve = skill[metric.name]
        for i in range(n):
            ax.plot(rollout.leads, per_member[i], color=ANALOG_COLORS[i], linewidth=2,
                    label=row_labels[i].replace("\n", " "))
        ax.plot(rollout.leads, ensemble_curve, color=ENSEMBLE_COLOR, linewidth=3,
                label="Ensemble mean")
        ax.set_ylabel(f"{metric.name} ({metric.unit})")
        ax.grid(True, linestyle=":", alpha=0.5)
        if mi == n_metrics - 1:
            ax.set_xlabel("Lead (days)")
        if mi == 0:
            ax.set_title(f"Skill by lead\nanalogs selected by {selection_metric}")
            ax.legend(loc="upper left", fontsize="small")

    fig.savefig(f"{fig_dir}/{filename}", dpi=130, bbox_inches="tight")
    plt.show()
    return fig


def plot_target_vs_best(library, analogs, target_time, target_date, fig_dir,
                        filename, contour_var="ssh", contour_level=None):
    """The target day beside the best analog, every variable, at lead 0.

    The sanity check on selection: whatever the metric optimized, these two maps
    should look alike to the eye.
    """
    fields = library.fields
    panel_w, panel_h = panel_size(fields)
    best_date = analogs.dates[0]
    fig, axes = plt.subplots(len(fields), 2,
                             figsize=(2 * panel_w * 1.9 + 1.0, panel_h * 1.9 * len(fields)),
                             squeeze=False)

    for row, (var, field) in enumerate(fields.items()):
        target_field = field.raw.sel(time=target_time).compute()
        analog_field = field.raw.sel(time=analogs.times[0]).compute()
        vmin = min(float(target_field.min()), float(analog_field.min()))
        vmax = max(float(target_field.max()), float(analog_field.max()))
        if field.symmetric:
            vmax = max(abs(vmin), abs(vmax))
            vmin = -vmax
        kwargs = dict(cmap=field.cmap, vmin=vmin, vmax=vmax,
                      cbar_kwargs={"label": f"{var.upper()} ({field.unit})"})

        for col, (data, label) in enumerate([(target_field, f"Target {target_date}"),
                                             (analog_field, f"Best analog {best_date}")]):
            data.plot(ax=axes[row][col], **kwargs)
            if var == contour_var and contour_level is not None:
                axes[row][col].contour(data.longitude, data.latitude, data.values,
                                       levels=[contour_level], colors="k", linewidths=1.5)
            axes[row][col].set_title(f"{var.upper()} — {label}")

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/{filename}", dpi=150)
    plt.show()
    return fig


def plot_window_skill(labels, per_window, metrics, lead, fig_dir,
                      filename="window_skill.png"):
    """Per-window forecast skill: each analog, the ensemble, and persistence.

    One panel per metric. Used by the multi-window drivers, where the question is
    not "how did this one forecast do" but "does the method beat persistence
    across the whole observed period".
    """
    x = np.arange(len(labels))
    fig, axes = plt.subplots(len(metrics), 1,
                             figsize=(max(7, 1.1 * len(x) + 2), 3.2 * len(metrics)),
                             sharex=True, squeeze=False)
    for ai, metric in enumerate(metrics):
        ax = axes[ai][0]
        for xi, w in zip(x, per_window):
            first = xi == 0
            members = w[metric.name]["members"]
            ax.scatter([xi] * len(members), members, s=18, color="#4a90d9", alpha=0.6,
                       zorder=2, label="analogs" if first else None)
            ax.scatter([xi], [w[metric.name]["ensemble"]], marker="D", s=45,
                       color="#1a7d3c", zorder=3, label="ensemble" if first else None)
            if w[metric.name].get("persistence") is not None:
                ax.scatter([xi], [w[metric.name]["persistence"]], marker="x", s=55,
                           color="k", zorder=3, label="persistence" if first else None)
        ax.set_ylabel(f"+{lead}d {metric.name} ({metric.unit})")
        ax.grid(True, alpha=0.3)
        if ai == 0:
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title(f"{lead}-day forecast skill per observation window")
    axes[-1][0].set_xlabel("observation window")
    axes[-1][0].set_xticks(x)
    axes[-1][0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/{filename}", dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_observation_analogs(obs_grid, mask, library, analogs, fig_dir,
                             filename="observation_analogs.png", k=None,
                             var="ssh"):
    """The observation beside the analog fields it selected.

    The footprint is outlined on each analog so the match can be checked where the
    observation actually saw something, which for a partial swath is a small part
    of the box.
    """
    k = len(analogs) if k is None else min(k, len(analogs))
    obs = np.where(mask, obs_grid, np.nan)
    series = library.series(var, "anomaly")
    fields = [series.sel(time=t).values for t in analogs.times[:k]]
    pool = np.concatenate([obs[np.isfinite(obs)]] + [f[np.isfinite(f)] for f in fields])
    vmax = float(np.nanpercentile(np.abs(pool), 98))
    kw = dict(origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ncol = k + 1
    aspect_wh = obs.shape[1] / obs.shape[0]
    ph = 2.0
    fig, ax = plt.subplots(1, ncol, figsize=(ncol * ph * aspect_wh + 1.0, ph + 1.2),
                           constrained_layout=True, squeeze=False)
    ax = ax[0]
    ax[0].imshow(obs, **kw)
    ax[0].set_title("observation", fontsize=8, weight="bold")
    for s in ax[0].spines.values():
        s.set(color="#7a3b9e", linewidth=2)
    im = None
    for c, (date, d) in enumerate(zip(analogs.dates[:k], analogs.distances[:k]), start=1):
        im = ax[c].imshow(fields[c - 1], **kw)
        ax[c].contour(mask, levels=[0.5], colors="k", linewidths=0.5)
        ax[c].set_title(f"analog {c}\n{date}  d={d:.2f}", fontsize=8)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    if im is not None:
        fig.colorbar(im, ax=ax, location="bottom", shrink=0.5, aspect=50, pad=0.02,
                     label="anomaly (m)")
    fig.suptitle(f"observation-selected analogs (top-{k} by {analogs.metric})",
                 fontsize=12, weight="bold")
    fig.savefig(f"{fig_dir}/{filename}", dpi=180, bbox_inches="tight")
    plt.show()
    return fig
