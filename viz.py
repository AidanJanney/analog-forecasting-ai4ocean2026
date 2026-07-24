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
