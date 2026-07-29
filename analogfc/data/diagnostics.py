"""Input-data diagnostics: what the record looks like before any forecasting.

These describe the *data*, not a forecast, which is why they live in the data
section. Three figures and the distribution statistics that go with them.
"""

import numpy as np
import matplotlib.pyplot as plt


def _save(fig_dir, name, dpi=150):
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/{name}", dpi=dpi)
    plt.show()


def spatial_variance(field, fig_dir):
    """Map of a field's variance over time."""
    field.raw.var(dim="time").plot(cmap=field.cmap, robust=True)
    plt.title(f"Spatial Variance of {field.name.upper()}")
    _save(fig_dir, f"{field.name}_spatial_variance.png")


def distribution(field, stats, fig_dir):
    """Histogram with mean and +/-1, +/-2 standard-deviation bounds.

    Also reports what fraction of grid points fall inside each band, which is the
    check that the field is close enough to normal for a standardized anomaly to
    mean what it usually means.
    """
    mean_val = stats["mean"]
    std_val = stats["std_dev"]
    bounds_1std = (mean_val - std_val, mean_val + std_val)
    bounds_2std = (mean_val - 2 * std_val, mean_val + 2 * std_val)

    flat_data = field.raw.values.flatten()
    flat_data = flat_data[~np.isnan(flat_data)]
    total_points = len(flat_data)

    count_1std = np.sum((flat_data >= bounds_1std[0]) & (flat_data <= bounds_1std[1]))
    count_2std = np.sum((flat_data >= bounds_2std[0]) & (flat_data <= bounds_2std[1]))
    pct_1std = (count_1std / total_points) * 100
    pct_2std = (count_2std / total_points) * 100

    unit = field.unit
    print(f"Mean: {mean_val:.2f} {unit} | Std Dev: {std_val:.2f} {unit}\n")
    print(f"Exact Grid Points within Mean ± 1 Std Dev ({bounds_1std[0]:.2f}{unit} to {bounds_1std[1]:.2f}{unit}): {pct_1std:.2f}%")
    print(f"Exact Grid Points within Mean ± 2 Std Dev ({bounds_2std[0]:.2f}{unit} to {bounds_2std[1]:.2f}{unit}): {pct_2std:.2f}%")

    plt.figure(figsize=(10, 5))
    plt.hist(flat_data, bins=60, density=True, color="teal", alpha=0.5)
    plt.axvline(mean_val, color="red", linestyle="-", linewidth=2, label=f"Mean ({mean_val:.2f}{unit})")
    plt.axvline(bounds_1std[0], color="orange", linestyle="--", label=f"±1 Std Dev ({pct_1std:.1f}% of data)")
    plt.axvline(bounds_1std[1], color="orange", linestyle="--")
    plt.axvline(bounds_2std[0], color="purple", linestyle=":", label=f"±2 Std Dev ({pct_2std:.1f}% of data)")
    plt.axvline(bounds_2std[1], color="purple", linestyle=":")
    plt.title(f"{field.name.upper()} Empirical Distribution", fontsize=13)
    plt.xlabel(f"{field.name.upper()} ({unit})")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    _save(fig_dir, f"{field.name}_distribution.png")

    quantiles = np.percentile(flat_data, [5, 25, 50, 75, 95])
    print(f" 5th Percentile: {quantiles[0]:.2f} {unit} (Coldest 5% threshold)")
    print(f"25th Percentile (Q1): {quantiles[1]:.2f} {unit}")
    print(f"50th Percentile (Median): {quantiles[2]:.2f} {unit}")
    print(f"75th Percentile (Q3): {quantiles[3]:.2f} {unit}")
    print(f"95th Percentile: {quantiles[4]:.2f} {unit} (Warmest 5% threshold)")
    del flat_data      # ~1.4 GB of the job's 10 GB, and nothing below needs it


def spatial_temporal_variation(field, fig_dir):
    """Time-mean map beside the domain-mean series."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    field.raw.mean(dim="time").plot(ax=axes[0], cmap=field.cmap,
                                    cbar_kwargs={"label": f"{field.name.upper()} ({field.unit})"})
    axes[0].set_title(f"Spatial Variation: Mean {field.name.upper()}")
    field.raw.mean(dim=["latitude", "longitude"]).plot(ax=axes[1], color="navy")
    axes[1].set_title(f"Temporal Variation: Domain-Averaged {field.name.upper()}")
    axes[1].set_ylabel(f"{field.name.upper()} ({field.unit})")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    _save(fig_dir, f"{field.name}_spatial_temporal_variation.png")


def run_all(field, stats, fig_dir):
    """Every input diagnostic for one field, in the order the run reports them."""
    spatial_variance(field, fig_dir)
    distribution(field, stats, fig_dir)
    spatial_temporal_variation(field, fig_dir)
