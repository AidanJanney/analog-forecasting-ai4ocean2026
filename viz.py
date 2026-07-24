"""Plotting for the analog-forecasting workflow.

Each function builds one figure, saves it under `outdir` (default ``plots/``),
and returns the figure. They take already-computed inputs (fronts, dissimilarity
matrix, an ``AnalogForecaster``) so the orchestration lives in ``main.py``.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from fronts import zos_contour_points

PLOTS = "plots"


def _save(fig, name, outdir, dpi=150):
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, name), dpi=dpi, bbox_inches="tight")
    return fig


def _time_ticks(times):
    """Tick indices/labels for a time axis: months for short spans, years for long."""
    days = times.astype("datetime64[D]")
    dom = (days - days.astype("datetime64[M]")).astype("timedelta64[D]").astype(int) + 1
    month_idx = np.where(dom == 1)[0]
    if len(month_idx) > 24:                     # multi-year: label year starts only
        month = (days.astype("datetime64[M]")
                 - days.astype("datetime64[Y]")).astype("timedelta64[M]").astype(int)
        yr_idx = np.where((dom == 1) & (month == 0))[0]
        return yr_idx, np.datetime_as_string(times[yr_idx], unit="Y")
    return month_idx, np.datetime_as_string(times[month_idx], unit="M")


# --------------------------------------------------------------------------- #
# Dissimilarity of fronts
# --------------------------------------------------------------------------- #
def plot_fields_and_fronts(zos, times, level, a=0, b=None, outdir=PLOTS):
    """Two example SSH fields with their extracted fronts overlaid."""
    day = np.datetime_as_string(times, unit="D")
    if b is None:
        b = zos.sizes["time"] // 2
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for k, t in zip((0, 1), (a, b)):
        zos.isel(time=t).plot(x="longitude", y="latitude", ax=ax[k],
                              vmin=float(zos.min()), vmax=float(zos.max()), cmap="RdBu_r")
        pts = zos_contour_points(zos.isel(time=t), level)
        ax[k].plot(pts[:, 0], pts[:, 1], "k.", ms=2)
        ax[k].set_title(f"{day[t]}   (front = zos {level} m)")
    return _save(fig, "fields_and_fronts.png", outdir)


def plot_dissimilarity_matrix(D, times, metric_name, outdir=PLOTS):
    """Full pairwise dissimilarity matrix with month ticks."""
    day = np.datetime_as_string(times, unit="D")
    idx, lbl = _time_ticks(times)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(D, origin="lower", cmap="viridis")
    ax.set_xticks(idx)
    ax.set_xticklabels(lbl, rotation=90)
    ax.set_yticks(idx)
    ax.set_yticklabels(lbl)
    ax.set_title(f"Pairwise {metric_name} dissimilarity ({day[0][:4]})")
    fig.colorbar(im, ax=ax, label=f"{metric_name} (deg)")
    return _save(fig, "dissimilarity_matrix.png", outdir)


def plot_dissimilarity_vs_reference(D, times, metric_name, ref=0, outdir=PLOTS):
    """Dissimilarity of every state to a reference day (seasonal drift)."""
    day = np.datetime_as_string(times, unit="D")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, D[ref], lw=1)
    ax.set_xlabel("date")
    ax.set_ylabel(f"{metric_name} to {day[ref]} (deg)")
    ax.set_title("Front dissimilarity relative to the reference day")
    ax.grid(True, alpha=0.3)
    return _save(fig, "dissimilarity_vs_reference.png", outdir)


def plot_front_pairs(D, contours, times, min_sep=30, outdir=PLOTS):
    """The most similar and most dissimilar pairs of fronts.

    `min_sep` (in time steps) excludes pairs closer together in time than this
    when picking the most-similar pair. Adjacent days are trivially similar
    because the Loop Current is autocorrelated day-to-day; a minimum separation
    surfaces genuinely *recurring* states — the point of analog forecasting.
    """
    day = np.datetime_as_string(times, unit="D")
    n = D.shape[0]
    idx = np.arange(n)
    near = np.abs(idx[:, None] - idx[None, :]) <= min_sep   # trivial near-in-time pairs
    Dsim = np.where(near, np.inf, D)                         # mask them for the min
    i_sim, j_sim = np.unravel_index(np.argmin(Dsim), D.shape)
    i_dis, j_dis = np.unravel_index(np.argmax(D), D.shape)
    fig, axp = plt.subplots(1, 2, figsize=(13, 6))
    for k, (i, j, kind) in enumerate([
        (i_sim, j_sim, f"Most similar (≥{min_sep}d apart)"),
        (i_dis, j_dis, "Most dissimilar"),
    ]):
        axp[k].plot(contours[i][:, 0], contours[i][:, 1], ".", ms=3, label=day[i])
        axp[k].plot(contours[j][:, 0], contours[j][:, 1], ".", ms=3, label=day[j])
        axp[k].set_aspect("equal")
        axp[k].set_xlabel("longitude")
        axp[k].set_ylabel("latitude")
        axp[k].set_title(f"{kind} fronts  (MHD = {D[i, j]:.4f} deg)")
        axp[k].legend()
    gap = abs(i_sim - j_sim)
    print(f"Most similar (≥{min_sep}d):  {day[i_sim]} / {day[j_sim]}  "
          f"({gap}d apart)  MHD = {D[i_sim, j_sim]:.4f} deg")
    print(f"Most dissimilar:      {day[i_dis]} / {day[j_dis]}   MHD = {D[i_dis, j_dis]:.4f} deg")
    return _save(fig, "front_pairs.png", outdir)


# --------------------------------------------------------------------------- #
# Forecasting
# --------------------------------------------------------------------------- #
def plot_forecast_demo(af, t0, lead, day, forecast_var="", analog_name="",
                       units="", plot_depth=0, k=10, exclude=5, outdir=PLOTS):
    """Truth / analog forecast / persistence / error, verified against baselines."""
    V = t0 + lead
    fc, sel, d_sel = af.forecast(t0, lead, k=k, exclude=exclude)
    truth = af.truth(V)
    print(f"Forecast {day[t0]} + {lead}d  ->  {day[V]}")
    print(f"  analogs: {', '.join(day[sel])}")
    print(f"  analog      = {af.error(fc, truth, V):.4f}")
    print(f"  persistence = {af.error(af.truth(t0), truth, V):.4f}")
    print(f"  climatology = {af.error(af.clim_forecast(V), truth, V):.4f}")

    def to2d(field):
        return field if field.ndim == 2 else field[plot_depth]

    t2d, f2d = to2d(truth), to2d(fc)
    fig, axf = plt.subplots(2, 2, figsize=(12, 8))
    kw = dict(origin="lower", cmap="RdBu_r", vmin=np.nanmin(t2d), vmax=np.nanmax(t2d))
    cbar_kw = dict(fraction=0.046, pad=0.04)   # match colorbar height to the map
    for a, field, title in [
        (axf[0, 0], t2d, f"truth  {day[t0 + lead]}"),
        (axf[0, 1], f2d, "analog forecast"),
        (axf[1, 0], to2d(af.truth(t0)), f"persistence  {day[t0]}"),
    ]:
        imf = a.imshow(field, **kw)
        a.set_title(title)
        fig.colorbar(imf, ax=a, label=units or None, **cbar_kw)
    im_e = axf[1, 1].imshow(f2d - t2d, origin="lower", cmap="coolwarm")
    axf[1, 1].set_title("forecast - truth")
    fig.colorbar(im_e, ax=axf[1, 1],
                 label=f"error ({units})" if units else "error", **cbar_kw)
    fig.suptitle(f"{forecast_var} forecast (analogs by {analog_name})")
    return _save(fig, f"{forecast_var}_forecast_demo.png", outdir)


# --------------------------------------------------------------------------- #
# SWOT observations
# --------------------------------------------------------------------------- #
def _swot_clim(swaths):
    """Robust symmetric SSHA colour limit (98th percentile) pooled over swaths."""
    ssha = np.concatenate([s[2] for s in swaths]) if swaths else np.array([0.0])
    return float(np.nanpercentile(np.abs(ssha), 98))


def plot_swot_swaths(swaths, bbox=None, title="SWOT KaRIn SSHA",
                     fname="swot_swaths.png", outdir=PLOTS):
    """Composite scatter of SSHA from several SWOT swaths over the region.

    `swaths` is a list of (lon, lat, ssha) 1-D arrays (see swot_data.load_swath).
    """
    vmax = _swot_clim(swaths)
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = None
    for lon, lat, ssha in swaths:
        sc = ax.scatter(lon, lat, c=ssha, s=2, cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, linewidths=0)
    if bbox:
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect("equal")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title)
    if sc is not None:
        fig.colorbar(sc, ax=ax, label="SSHA (m)", fraction=0.046, pad=0.04)
    return _save(fig, fname, outdir)


def plot_swot_panels(swaths, labels, bbox=None, vmax=None,
                     fname="swot_passes.png", outdir=PLOTS):
    """Small multiples: one SSHA map per SWOT pass, on a shared colour scale."""
    if vmax is None:
        vmax = _swot_clim(swaths)
    m = len(swaths)
    ncol = min(3, m)
    nrow = int(np.ceil(m / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow),
                             squeeze=False)
    sc = None
    for ax, (lon, lat, ssha), lab in zip(axes.ravel(), swaths, labels):
        sc = ax.scatter(lon, lat, c=ssha, s=3, cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, linewidths=0)
        if bbox:
            ax.set_xlim(bbox[0], bbox[2])
            ax.set_ylim(bbox[1], bbox[3])
        ax.set_aspect("equal")
        ax.set_title(lab, fontsize=9)
    for ax in axes.ravel()[m:]:
        ax.axis("off")
    if sc is not None:
        fig.colorbar(sc, ax=axes, label="SSHA (m)", fraction=0.03, pad=0.02)
    return _save(fig, fname, outdir)


def plot_swot_analogs(obs_grid, mask, af, sel, dist, day, k=None,
                      fname="swot_analogs.png", outdir=PLOTS):
    """Binned SWOT observation beside its top GLORYS analog anomaly fields.

    `af` is an analog.AnalogForecaster; `sel`/`dist` are analog indices and
    distances (1 - correlation). The swath footprint is outlined on each analog so
    the eye can check the match where SWOT observed.
    """
    k = len(sel) if k is None else min(k, len(sel))
    obs = np.where(mask, obs_grid, np.nan)
    fields = [af.anom[i] for i in sel[:k]]
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


def plot_swot_forecast(af, obs_date, results, fname="swot_forecast.png", outdir=PLOTS):
    """Per-lead: SWOT-driven dense forecast (anomaly) beside the future SWOT truth.

    `results` is a list of dicts with keys L, fc_anom, obs2, mask2, r_analog.
    """
    leads = [r["L"] for r in results]
    fields = [r["fc_anom"] for r in results] + \
             [np.where(r["mask2"], r["obs2"], np.nan) for r in results]
    pool = np.concatenate([f[np.isfinite(f)] for f in fields])
    vmax = float(np.nanpercentile(np.abs(pool), 98))
    kw = dict(origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    nrow = len(results)
    aspect_wh = af.anom.shape[2] / af.anom.shape[1]
    ph = 2.0
    fig, ax = plt.subplots(nrow, 2, figsize=(2 * ph * aspect_wh + 1.2, nrow * ph + 1.4),
                           constrained_layout=True, squeeze=False)
    im = None
    for r, res in enumerate(results):
        im = ax[r, 0].imshow(res["fc_anom"], **kw)
        ax[r, 0].contour(res["mask2"], levels=[0.5], colors="k", linewidths=0.5)
        ax[r, 0].set_title(f"FORECAST +{res['L']}d   r={res['r_analog']:.2f}",
                           fontsize=9, weight="bold")
        for s in ax[r, 0].spines.values():
            s.set(color="#1a7d3c", linewidth=2)
        ax[r, 1].imshow(np.where(res["mask2"], res["obs2"], np.nan), **kw)
        ax[r, 1].set_title(f"SWOT truth +{res['L']}d", fontsize=9, weight="bold")
        for s in ax[r, 1].spines.values():
            s.set(color="#7a3b9e", linewidth=2)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    if im is not None:
        fig.colorbar(im, ax=ax, location="bottom", shrink=0.5, aspect=50,
                     pad=0.02, label="mesoscale SSHA anomaly (m)")
    fig.suptitle(f"SWOT-driven model-analog forecast (deseasonalized)   obs {obs_date}",
                 fontsize=12, weight="bold")
    return _save(fig, fname, outdir, dpi=180)


def plot_swot_skill(leads, acc, rmse, counts, n_obs, fname="swot_skill.png",
                    outdir=PLOTS):
    """Observation-space skill vs lead: anomaly correlation (top) and RMSE (bottom).

    `acc`/`rmse` are dicts {baseline: per-lead array}; `counts` is the number of
    obs days contributing per lead; `n_obs` the total obs days aggregated over.
    """
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for name in acc:
        a1.plot(leads, acc[name], marker="o", label=name)
        a2.plot(leads, rmse[name], marker="o", label=name)
    a1.axhline(0, color="gray", lw=0.8)
    a1.set_ylabel("mesoscale anomaly correlation")
    a2.set_ylabel("mesoscale RMSE (m)")
    a2.set_xlabel("lead time (days)")
    for ax in (a1, a2):
        ax.grid(True, alpha=0.3)
    a1.legend()
    # sample count per lead along the top
    top = np.nanmax([np.nanmax(v) for v in acc.values()])
    for L, c in zip(leads, counts):
        a1.annotate(str(c), (L, top), fontsize=7, ha="center", va="bottom",
                    color="gray")
    a1.set_title(f"SWOT-driven forecast skill (deseasonalized / mesoscale)   "
                 f"(aggregated over {n_obs} obs days; n per lead shown above)")
    return _save(fig, fname, outdir)


def _front_px(front, lon, lat):
    """Map (lon,lat) front vertices to imshow pixel coords for a regular grid."""
    if front is None or len(front) == 0:
        return np.empty(0), np.empty(0)
    px = (front[:, 0] - lon[0]) / (lon[1] - lon[0])
    py = (front[:, 1] - lat[0]) / (lat[1] - lat[0])
    return px, py


def plot_analog_glorys_maps(saf, obs_day, obs_grid, mask, sel, day, fc_anoms,
                            truth_anom, accs, lead, dist=None, k_show=6,
                            fc_fronts=None, truth_front=None, lc_mhd=None,
                            init_anom=None, init_front=None,
                            fname="analog_glorys_maps.png", outdir=PLOTS):
    """SWOT obs | analog | GLORYS init | +lead forecast | dense GLORYS truth.

    Each row is one SWOT-selected (de-clustered) analog. The first column is the SWOT
    swath that drove the selection (gridded, NaN off-swath, offset removed since ssha
    and zos have different references) and the third is the dense GLORYS truth at the
    *obs* day — the initial condition the forecast has to move away from — so each row
    reads left-to-right as what was observed, what matched it, where the ocean actually
    started, where the analog says it goes, and where it actually went. Titles carry the
    analog date, the forecast's dense ACC and its Loop Current front error (MHD, km).
    The Loop Current front is overlaid on the initial condition (orange), the forecast
    (green) and the truth (black), with the truth front also drawn on each forecast
    panel so the position error is visible. All model maps are deseasonalized (the
    initial condition at the obs day, the rest at the verification day) and share a
    diverging scale.
    """
    k = min(k_show, len(sel))
    sel_fields = [saf.deseasonalize(saf.anom[i], saf.times[i]) for i in sel[:k]]
    fcs = fc_anoms[:k]
    pool = np.concatenate([truth_anom[np.isfinite(truth_anom)]]
                          + ([] if init_anom is None else [init_anom[np.isfinite(init_anom)]])
                          + [f[np.isfinite(f)] for f in sel_fields + list(fcs)])
    vmax = float(np.nanpercentile(np.abs(pool), 98))
    kw = dict(origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    tf_px = _front_px(truth_front, saf.lon, saf.lat)
    if_px = _front_px(init_front, saf.lon, saf.lat)
    # SWOT ssha is referenced to a mean sea surface, zos to the library mean: the
    # selection kernel is offset-invariant, so show the swath with its offset removed.
    obs_show = np.where(mask, obs_grid - np.nanmean(obs_grid[mask]), np.nan)

    cols = ["swot", "analog", "init", "forecast", "truth"]
    if init_anom is None:
        cols.remove("init")
    c = {name: j for j, name in enumerate(cols)}
    nc = len(cols)

    aspect_wh = truth_anom.shape[1] / truth_anom.shape[0]
    ph = 1.9
    fig, ax = plt.subplots(k, nc, figsize=(nc * ph * aspect_wh + 1.2, k * ph + 1.2),
                           constrained_layout=True, squeeze=False)
    im = None
    for r in range(k):
        i = sel[r]
        a_obs = ax[r, c["swot"]]
        a_obs.set_facecolor("#e8e8e8")                   # unobserved (off-swath) cells
        a_obs.imshow(obs_show, **kw)
        a_obs.set_title(f"SWOT obs {obs_day}", fontsize=8)
        for s in a_obs.spines.values():
            s.set(color="#2a6ebb", linewidth=2)
        ax[r, c["analog"]].imshow(sel_fields[r], **kw)
        ax[r, c["analog"]].contour(mask, levels=[0.5], colors="k", linewidths=0.4)
        rr = "" if dist is None else f"  r={1 - dist[r]:.2f}"
        ax[r, c["analog"]].set_title(f"analog {r + 1}: {day[i]}{rr}", fontsize=8)
        if init_anom is not None:
            a_ic = ax[r, c["init"]]
            a_ic.imshow(init_anom, **kw)
            a_ic.plot(*if_px, ".", ms=1.1, color="#b5651d")       # initial LC front
            a_ic.set_title(f"GLORYS init {obs_day}", fontsize=8)
            for s in a_ic.spines.values():
                s.set(color="#d9a05b", linewidth=2)
        a_fc = ax[r, c["forecast"]]
        im = a_fc.imshow(fcs[r], **kw)
        if fc_fronts is not None:                       # forecast LC front (green)
            fx, fy = _front_px(fc_fronts[r], saf.lon, saf.lat)
            a_fc.plot(fx, fy, ".", ms=1.1, color="#127a2e")
        a_fc.plot(*tf_px, ".", ms=0.8, color="k", alpha=0.7)      # truth front
        lc = "" if lc_mhd is None else f"   LC={lc_mhd[r]:.0f}km"
        a_fc.set_title(f"forecast +{lead}d   ACC={accs[r]:+.2f}{lc}",
                       fontsize=8, weight="bold")
        for s in a_fc.spines.values():
            s.set(color="#1a7d3c", linewidth=2)
        ax[r, c["truth"]].imshow(truth_anom, **kw)
        ax[r, c["truth"]].plot(*tf_px, ".", ms=1.1, color="k")    # Loop Current
        ax[r, c["truth"]].set_title(f"GLORYS truth +{lead}d", fontsize=8)
        for s in ax[r, c["truth"]].spines.values():
            s.set(color="#b5651d", linewidth=2)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
        a.set_xlim(0, truth_anom.shape[1] - 1)
        a.set_ylim(0, truth_anom.shape[0] - 1)
    if im is not None:
        fig.colorbar(im, ax=ax, location="bottom", shrink=0.4, aspect=60,
                     pad=0.02, label="deseasonalized SSHA anomaly (m)")
    fig.suptitle(f"SWOT-selected analogs → {lead}-day forecast vs GLORYS   obs {obs_day}"
                 f"   (Loop Current: orange = initial, green = forecast, black = truth)",
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


def plot_metric_ladder(variants, rows, persist_acc, persist_lc, lead,
                       fname="metric_ladder.png", outdir=PLOTS):
    """Oracle selection ladder: forecast skill as the selection target improves.

    `rows[v]` holds `acc_best/acc_ens/lc_best/lc_ens` for each variant `v` (from the
    real SWOT metric up to future-oracle). Left panel: full-field ACC (higher=better);
    right: Loop Current front MHD in km (lower=better). Best-analog and ensemble are
    shown per variant, with the dense-persistence baseline drawn as a reference line —
    the gap from `metric-analog` to `present-oracle` says whether a better selection
    metric could help.
    """
    x = np.arange(len(variants))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))

    a1.plot(x, [rows[v]["acc_best"] for v in variants], "o-", color="#4a90d9",
            label="best analog", zorder=3)
    a1.plot(x, [rows[v]["acc_ens"] for v in variants], "D--", color="#1a7d3c",
            label="ensemble", zorder=3)
    a1.axhline(persist_acc, color="k", ls=":", lw=1.5, label="persistence")
    a1.axhline(0, color="gray", lw=0.8)
    a1.set_ylabel(f"+{lead}d full-field mesoscale ACC  (higher = better)")

    a2.plot(x, [rows[v]["lc_best"] for v in variants], "o-", color="#4a90d9",
            label="best analog", zorder=3)
    a2.plot(x, [rows[v]["lc_ens"] for v in variants], "D--", color="#1a7d3c",
            label="ensemble", zorder=3)
    a2.axhline(persist_lc, color="k", ls=":", lw=1.5, label="persistence")
    a2.set_ylabel(f"+{lead}d Loop Current front MHD (km)  (lower = better)")

    for ax in (a1, a2):
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Does a better analog-selection metric help? "
                 f"(selection target improves left → right, {lead}-day forecast)",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    return _save(fig, fname, outdir)


def plot_ceiling(leads, curves, fname="ceiling.png", outdir=PLOTS):
    """Analog skill-ceiling (oracle) diagnostic: deseasonalized ACC vs lead.

    `curves` maps each strategy name (persistence, metric-analog, present-oracle,
    future-oracle) to its per-lead mean ACC. The gap from `metric-analog` up to
    `present-oracle` is the skill a better *current* selection metric could recover
    (ML headroom); the gap from `persistence` up to `future-oracle` is the library's
    intrinsic predictability ceiling. Higher is better.
    """
    style = {                                        # (color, linestyle, marker)
        "persistence": ("k", ":", "x"),
        "metric-analog": ("#4a90d9", "-", "o"),
        "present-oracle": ("#1a7d3c", "--", "D"),
        "future-oracle": ("#b5651d", "-.", "s"),
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, vals in curves.items():
        c, ls, mk = style.get(name, ("gray", "-", "."))
        ax.plot(leads, vals, color=c, linestyle=ls, marker=mk, label=name, zorder=3)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("lead time (days)")
    ax.set_ylabel("deseasonalized ACC  (higher = better)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Analog skill ceiling: is the selection metric or the library the "
                 "bottleneck?\n(metric-analog → present-oracle = ML headroom; "
                 "persistence → future-oracle = predictability ceiling)")
    return _save(fig, fname, outdir)


def plot_analog_ensemble(af, t0, leads, day, k=8, exclude=30, units="",
                         plot_depth=0, fname="analog_ensemble.png", outdir=PLOTS):
    """Visualise one analog forecast at one or more lead times.

    The K analogs are selected once (from the target's front); each is then
    advanced to every requested lead. Layout — columns are
    ``analog_1 .. analog_K | forecast | target/truth | error``:
      row 0        — analogs at their *selection* time, plus the target
      one row/lead — analog futures, the weighted-average forecast, truth, error
    The error uses a diverging scale shared across leads, so its growth shows.
    """
    leads = [leads] if np.isscalar(leads) else list(leads)
    Lmax = max(leads)
    sel, d_sel = af.analogs(t0, Lmax, k, exclude)    # one analog set for all leads
    w = np.exp(-(d_sel / d_sel.mean()) ** 2)
    w = w / w.sum()                                  # normalised weights (sum to 1)

    def to2d(f):
        return f if f.ndim == 2 else f[plot_depth]

    tar = to2d(af.truth(t0))
    truths, fcasts, errs, accs = {}, {}, {}, {}
    for L in leads:
        fc = np.tensordot(w, af.state[sel + L], axes=1)   # weighted mean of futures
        tr = af.truth(t0 + L)
        truths[L], fcasts[L] = to2d(tr), to2d(fc)
        errs[L] = fcasts[L] - truths[L]
        accs[L] = af.error(fc, tr, t0 + L)

    # shared colour scales: one for SSH fields, one for errors (across leads)
    ssh = np.concatenate([tar[np.isfinite(tar)]]
                         + [truths[L][np.isfinite(truths[L])] for L in leads])
    vmax = float(np.nanpercentile(np.abs(ssh), 98))
    ep = np.concatenate([errs[L][np.isfinite(errs[L])] for L in leads])
    evmax = float(np.nanpercentile(np.abs(ep), 98)) or vmax
    kw = dict(origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ekw = dict(origin="lower", cmap="coolwarm", vmin=-evmax, vmax=evmax)

    ncol = k + 3                             # analogs | forecast | target/truth | error
    nrow = 1 + len(leads)
    aspect_wh = tar.shape[1] / tar.shape[0]
    panel_h = 1.7
    fig, ax = plt.subplots(
        nrow, ncol,
        figsize=(ncol * panel_h * aspect_wh + 1.2, nrow * panel_h + 1.8),
        constrained_layout=True, squeeze=False,
    )
    cF, cT, cE = k, k + 1, k + 2             # forecast / target-truth / error columns

    # -- selection row: the K analogs, then the target --
    im_ssh = None
    for c, (a, wi) in enumerate(zip(sel, w)):
        im_ssh = ax[0, c].imshow(to2d(af.truth(a)), **kw)
        ax[0, c].set_title(f"analog {c + 1}\n{day[a]}  w={wi * 100:.0f}%", fontsize=8)
    ax[0, cF].axis("off")
    ax[0, cE].axis("off")
    ax[0, cT].imshow(tar, **kw)
    ax[0, cT].set_title(f"TARGET\n{day[t0]}", fontsize=8, weight="bold")

    # -- one row per lead: analog futures, forecast, truth, error --
    im_err = None
    for r, L in enumerate(leads, start=1):
        for c, a in enumerate(sel):
            ax[r, c].imshow(to2d(af.truth(a + L)), **kw)
            ax[r, c].set_title(f"+{L}d", fontsize=7)
        ax[r, cF].imshow(fcasts[L], **kw)
        ax[r, cF].set_title(f"FORECAST +{L}d\nACC={accs[L]:.3f}", fontsize=8, weight="bold")
        ax[r, cT].imshow(truths[L], **kw)
        ax[r, cT].set_title(f"TRUTH +{L}d\n{day[t0 + L]}", fontsize=8, weight="bold")
        im_err = ax[r, cE].imshow(errs[L], **ekw)
        ax[r, cE].set_title(f"ERROR +{L}d", fontsize=8, weight="bold")

    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    ax[0, 0].set_ylabel("selection", fontsize=9)
    for r, L in enumerate(leads, start=1):
        ax[r, 0].set_ylabel(f"+{L} days", fontsize=9)

    def frame(a, color):
        for s in a.spines.values():
            s.set(color=color, linewidth=2)
    frame(ax[0, cT], "#333333")                     # target
    for r in range(1, nrow):
        frame(ax[r, cF], "#1a7d3c")                 # forecast
        frame(ax[r, cT], "#333333")                 # truth
        frame(ax[r, cE], "#7a3b9e")                 # error

    # horizontal colorbars along the bottom: SSH under the field block, error under its column
    if im_ssh is not None:
        fig.colorbar(im_ssh, ax=ax[:, :cE], location="bottom", shrink=0.6,
                     aspect=55, pad=0.02, label=units or "field")
    if im_err is not None:
        fig.colorbar(im_err, ax=ax[:, cE], location="bottom", shrink=0.9,
                     aspect=8, pad=0.02, label=f"error ({units})" if units else "error")
    fig.suptitle(f"Analog ensemble forecast   {day[t0]}   "
                 f"leads {', '.join(f'+{L}d' for L in leads)}   "
                 f"(K={k}, exclude={exclude}d)", fontsize=13, weight="bold")
    return _save(fig, fname, outdir, dpi=200)


def plot_skill(leads, skill, forecast_var, analog_name, error_name, year, outdir=PLOTS):
    """Forecast error vs lead time for analog, persistence and climatology."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, vals in skill.items():
        ax.plot(leads, vals, marker="o", label=name)
    ax.set_xlabel("lead time (days)")
    ax.set_ylabel(error_name)
    ax.set_title(f"Model-analog skill: {forecast_var} forecast, "
                 f"analogs by {analog_name} ({year})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, f"{forecast_var}_skill.png", outdir)
