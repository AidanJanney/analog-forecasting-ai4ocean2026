"""Figures for the SWOT -> GLORYS analog-forecasting workflow.

Each function builds one figure, saves it under `outdir` (default ``plots/``),
and returns the figure. They take already-computed inputs (a ``ModelLibrary``,
front masks, the per-window result dicts from ``swot_analog.evaluate``) so the
orchestration stays in the driver.

Fronts arrive as boolean grid masks (see :mod:`fronts`), which are drawn directly
as their own cells — no contour extraction is needed for display.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

PLOTS = "plots"


def _save(fig, name, outdir, dpi=150):
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, name), dpi=dpi, bbox_inches="tight")
    return fig


def plot_swot_analogs(obs_grid, mask, lib, sel, dist, day, k=None,
                      fname="swot_analogs.png", outdir=PLOTS):
    """Binned SWOT observation beside its top GLORYS analog anomaly fields.

    `lib` is a ModelLibrary; `sel`/`dist` are analog indices and
    distances (1 - correlation). The swath footprint is outlined on each analog so
    the eye can check the match where SWOT observed.
    """
    k = len(sel) if k is None else min(k, len(sel))
    obs = np.where(mask, obs_grid, np.nan)
    fields = [lib.anom[i] for i in sel[:k]]
    pool = np.concatenate([obs[np.isfinite(obs)]]
                          + [f[np.isfinite(f)] for f in fields])
    vmax = float(np.nanpercentile(np.abs(pool), 98))
    kw = dict(origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ncol = k + 1
    aspect_wh = obs.shape[1] / obs.shape[0]
    ph = 2.0
    fig, ax = plt.subplots(1, ncol, figsize=(ncol * ph * aspect_wh + 1.0, ph + 1.2),
                           constrained_layout=True, squeeze=False)
    ax = ax[0]
    ax[0].imshow(obs, **kw)
    ax[0].set_title("SWOT obs\n(binned SSHA)", fontsize=8, weight="bold")
    for s in ax[0].spines.values():
        s.set(color="#7a3b9e", linewidth=2)
    im = None
    for c, (i, d) in enumerate(zip(sel[:k], dist[:k]), start=1):
        im = ax[c].imshow(fields[c - 1], **kw)
        ax[c].contour(mask, levels=[0.5], colors="k", linewidths=0.5)
        ax[c].set_title(f"analog {c}\n{day[i]}  r={1 - d:.2f}", fontsize=8)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    if im is not None:
        fig.colorbar(im, ax=ax, location="bottom", shrink=0.5, aspect=50,
                     pad=0.02, label="SSHA anomaly (m)")
    fig.suptitle("SWOT-selected GLORYS analogs   (top-K by masked SSHA correlation)",
                 fontsize=12, weight="bold")
    return _save(fig, fname, outdir, dpi=180)






def plot_analog_glorys_lc_skill(obs_days, results, lead,
                                fname="analog_glorys_lc_skill.png", outdir=PLOTS):
    """Per-obs-day Loop Current forecast skill: front MHD (km), lower = better.

    For each obs day: the K de-clustered analogs' Loop Current front error at `lead`
    days (dots), the ensemble mean (diamond) and dense persistence (x). The Loop
    Current front is the offset-referenced 0.17-m ``zos`` contour; MHD is the modified
    Hausdorff distance to the truth front.
    """
    x = np.arange(len(obs_days))
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(x) + 2), 5))
    for xi, r in zip(x, results):
        first = xi == 0
        ax.scatter([xi] * len(r["lc_mhd"]), r["lc_mhd"], s=18, color="#4a90d9",
                   alpha=0.6, zorder=2, label="analogs" if first else None)
        ax.scatter([xi], [r["lc_mhd_ens"]], marker="D", s=45, color="#1a7d3c",
                   zorder=3, label="ensemble" if first else None)
        ax.scatter([xi], [r["lc_mhd_persist"]], marker="x", s=55, color="k",
                   zorder=3, label="persistence" if first else None)
    ax.set_ylabel(f"+{lead}d Loop Current front MHD (km)")
    ax.set_xlabel("SWOT observation day")
    ax.set_xticks(x)
    ax.set_xticklabels(obs_days, rotation=45, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"SWOT-selected analog {lead}-day Loop Current forecast skill "
                 f"(front MHD; lower = better)")
    return _save(fig, fname, outdir)


def plot_analog_glorys_skill(obs_days, results, lead,
                             fname="analog_glorys_skill.png", outdir=PLOTS):
    """Per-obs-day distribution of per-analog skill vs persistence / climatology.

    For each obs day: the K de-clustered analogs' `lead`-day forecast ACC/RMSE
    (dots), the Gaussian-weighted ensemble mean (diamond) and the dense persistence
    baseline (x). The deseasonalized climatology sits at ACC = 0 by construction.
    """
    x = np.arange(len(obs_days))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(max(7, 1.1 * len(x) + 2), 7),
                                 sharex=True)
    for xi, r in zip(x, results):
        first = xi == 0
        a1.scatter([xi] * len(r["acc"]), r["acc"], s=18, color="#4a90d9",
                   alpha=0.6, zorder=2, label="analogs" if first else None)
        a1.scatter([xi], [r["acc_ens"]], marker="D", s=45, color="#1a7d3c",
                   zorder=3, label="ensemble" if first else None)
        a1.scatter([xi], [r["acc_persist"]], marker="x", s=55, color="k",
                   zorder=3, label="persistence" if first else None)
        a2.scatter([xi] * len(r["rmse"]), r["rmse"], s=18, color="#4a90d9", alpha=0.6)
        a2.scatter([xi], [r["rmse_ens"]], marker="D", s=45, color="#1a7d3c")
        a2.scatter([xi], [r["rmse_persist"]], marker="x", s=55, color="k")
    a1.axhline(0, color="gray", lw=0.8)          # deseasonalized climatology floor
    a1.set_ylabel(f"+{lead}d mesoscale ACC")
    a2.set_ylabel(f"+{lead}d mesoscale RMSE (m)")
    a2.set_xlabel("SWOT observation day")
    a1.set_xticks(x)
    a1.set_xticklabels(obs_days, rotation=45, ha="right", fontsize=8)
    for ax in (a1, a2):
        ax.grid(True, alpha=0.3)
    a1.legend(loc="upper right", fontsize=8)
    a1.set_title(f"SWOT-selected analog {lead}-day forecast skill vs dense GLORYS truth "
                 f"(each analog scored separately)")
    return _save(fig, fname, outdir)


# --------------------------------------------------------------------------- #
# The analog grid: one row per member (state, then misfit), leads as columns.
# Ported from analog_forecast.py's plot_analog_grid for the SWOT -> GLORYS run.
# --------------------------------------------------------------------------- #
# Categorical hues for the individual analog members, assigned in fixed order.
# Validated for colourblind separation against a light surface.
ANALOG_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
ENSEMBLE_COLOR = "#0b0b0b"      # the ensemble is an aggregate, not a peer series


def _strip_ticks(ax, keep_x=False, keep_y=False):
    """Interior panels of a dense grid carry no tick labels; the edges do."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")


def plot_analog_grid(obs_grid, obs_mask, init_states, states, truth_states,
                     map_leads, row_labels, curves, curve_leads,
                     level=0.17, unit="m", obs_day="", n_analogs=None,
                     fname="analog_grid.png", outdir=PLOTS):
    """One row per analog (state, then misfit), then the ensemble, then the truth.

    Columns are the SWOT observation that initialized the forecast, then one map
    per lead in `map_leads`, then a column of daily skill curves — one panel per
    metric in `curves`, each showing every member against the ensemble and
    persistence.

    The Loop Current contour at `level` is drawn on every map: the truth's in
    black, and each row's own front dashed in that row's colour, so the position
    error can be read straight off the panel. Fields must already be referenced to
    the common datum (``fronts.referenced``) so the contour drawn here is the one
    ``fronts.front_mask`` measures.

    Parameters
    ----------
    obs_grid, obs_mask : (nlat, nlon)
        The observation window's composite SSHA and its coverage mask.
    init_states : list of (nlat, nlon), length n_analogs + 2
        Each row's field at lead 0, for the observation column's contour.
    states : list of lists, [row][lead]
        Absolute fields per row per map lead. Rows are the analogs, then the
        ensemble, then the truth — so ``len(states) == n_analogs + 2``.
    truth_states : list of (nlat, nlon), one per map lead
        The verification field, subtracted to form each misfit row.
    curves : dict
        ``{metric_name: {"members": (k, nL), "ens": (nL,), "persist": (nL,),
        "unit": str, "higher_is_better": bool}}``.
    """
    n = len(states) - 2 if n_analogs is None else n_analogs
    if n > len(ANALOG_COLORS):
        raise ValueError(f"{n} analogs but only {len(ANALOG_COLORS)} categorical colours")
    nL = len(map_leads)
    row_colors = [ANALOG_COLORS[r] for r in range(n)] + [ENSEMBLE_COLOR, "k"]

    # Misfit rows exist for every row that is a forecast (not the truth itself).
    errors = [[states[r][j] - truth_states[j] for j in range(nL)] for r in range(n + 1)]

    finite = [f[np.isfinite(f)] for row in states for f in row]
    pool = np.concatenate(finite)
    smin, smax = float(np.nanpercentile(pool, 1)), float(np.nanpercentile(pool, 99))
    emax = max(float(np.nanpercentile(np.abs(f[np.isfinite(f)]), 99))
               for row in errors for f in row)
    skw = dict(origin="lower", cmap="RdBu_r", vmin=smin, vmax=smax)
    ekw = dict(origin="lower", cmap="PuOr_r", vmin=-emax, vmax=emax)

    aspect_wh = states[0][0].shape[1] / states[0][0].shape[0]
    ph = 2.2
    pw = min(max(ph * aspect_wh, 1.5), 4.0)
    n_rows = 2 * (n + 1) + 1               # state+misfit per forecast row, plus truth
    n_metrics = max(1, len(curves))
    curve_ratio = max(2.0, 4.0 / pw)
    fig = plt.figure(figsize=(pw * (nL + 1 + 0.35 + curve_ratio) + 1.4,
                              ph * n_rows + 1.4))
    # Explicit margins: the default 0.1 top/bottom would leave inches of dead
    # space on a figure this tall.
    gs = fig.add_gridspec(
        n_rows + 1, nL + 3,
        height_ratios=[1] * n_rows + [0.16],
        width_ratios=[1] * (nL + 1) + [0.35, curve_ratio],
        left=0.05, right=0.94, top=1 - 0.5 / (ph * n_rows), bottom=0.02,
        hspace=0.12, wspace=0.05)

    # SWOT ssha is referenced to a mean sea surface and zos to the library mean, so
    # show the swath with its offset removed. One scale for every row, or the same
    # observation would appear differently down the column.
    obs_show = np.where(obs_mask, obs_grid - np.nanmean(obs_grid[obs_mask]), np.nan)
    ovmax = float(np.nanpercentile(np.abs(obs_show[np.isfinite(obs_show)]), 98))
    okw = dict(origin="lower", cmap="RdBu_r", vmin=-ovmax, vmax=ovmax)

    state_art = err_art = None
    for r in range(n + 2):                 # analogs, ensemble, truth
        sr = 2 * r
        colour = row_colors[r]

        # -- the observation that drove the selection, with this row's own front --
        a = fig.add_subplot(gs[sr, 0])
        a.set_facecolor("#e8e8e8")         # unobserved (off-swath) cells
        a.imshow(obs_show, **okw)
        a.contour(obs_mask.astype(float), levels=[0.5], colors="k", linewidths=0.5)
        a.contour(init_states[r], levels=[level], colors=[colour],
                  linewidths=1.3, linestyles="--")
        if sr == 0:
            a.set_title(f"SWOT obs {obs_day}", fontsize=9)
        a.set_ylabel(row_labels[r], fontsize=8)
        _strip_ticks(a)
        for s in a.spines.values():
            s.set(color="#7a3b9e", linewidth=1.6)

        for j, lead in enumerate(map_leads):
            ax = fig.add_subplot(gs[sr, j + 1])
            state_art = ax.imshow(states[r][j], **skw)
            ax.contour(truth_states[j], levels=[level], colors="k", linewidths=1.1)
            if r < n + 1:                  # the truth row's own front is the black one
                ax.contour(states[r][j], levels=[level], colors=[colour],
                           linewidths=1.2, linestyles="--")
            if sr == 0:
                ax.set_title(f"lead +{lead} d", fontsize=9)
            _strip_ticks(ax)

            if r < n + 1:                  # truth has no misfit row
                ae = fig.add_subplot(gs[sr + 1, j + 1])
                err_art = ae.imshow(errors[r][j], **ekw)
                if j == 0:
                    ae.set_ylabel(f"misfit ({unit})", fontsize=8)
                _strip_ticks(ae)

    # Two shared colorbars, distinct diverging pairs so a state panel is never
    # mistaken for a misfit panel at a glance.
    half = max(1, (nL + 1) // 2)
    fig.colorbar(state_art, cax=fig.add_subplot(gs[n_rows, :half]),
                 orientation="horizontal", label=f"SSH ({unit})")
    fig.colorbar(err_art, cax=fig.add_subplot(gs[n_rows, half:nL + 1]),
                 orientation="horizontal", label=f"forecast − truth ({unit})")

    # -- daily skill curves, one panel per metric ---------------------------- #
    block = n_rows // n_metrics
    for mi, (name, c) in enumerate(curves.items()):
        r0 = mi * block
        r1 = n_rows if mi == n_metrics - 1 else (mi + 1) * block
        ax = fig.add_subplot(gs[r0:r1, -1])
        for i in range(min(n, c["members"].shape[0])):
            ax.plot(curve_leads, c["members"][i], color=ANALOG_COLORS[i],
                    lw=1.8, label=row_labels[i].replace("\n", " ") if mi == 0 else None)
        ax.plot(curve_leads, c["ens"], color=ENSEMBLE_COLOR, lw=3,
                label="ensemble mean" if mi == 0 else None)
        if c.get("persist") is not None:
            ax.plot(curve_leads, c["persist"], color="#888888", lw=2, ls=":",
                    label="persistence" if mi == 0 else None)
        ax.set_ylabel(f"{name} ({c['unit']})", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        if mi == n_metrics - 1:
            ax.set_xlabel("lead (days)")
        if mi == 0:
            ax.set_title("skill by lead", fontsize=10)
            ax.legend(loc="best", fontsize=7)

    fig.suptitle(f"SWOT-selected analogs → forecast vs GLORYS   obs {obs_day}   "
                 f"(dashed = row's own {level:g} m contour, black = truth)",
                 fontsize=13, weight="bold", y=1 - 0.12 / (ph * n_rows))
    return _save(fig, fname, outdir, dpi=130)
