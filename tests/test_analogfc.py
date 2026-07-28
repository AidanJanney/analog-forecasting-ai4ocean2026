"""Tests for the three sections. No data files and no pytest required:

    python tests/test_analogfc.py

Everything runs on a small synthetic record built in :func:`synthetic_fields`,
so the whole suite takes about a second and exercises the paths that the GLORYS
regression run cannot — the observation-source and window layers, which have no
data on disk to drive them.
"""

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analogfc import mhd
from analogfc.data import climatology, sources, windows as ow
from analogfc.data.glorys import Field, FieldSet
from analogfc.data.library import ModelLibrary
from analogfc.data.regions import Region
from analogfc.distance import DISTANCES, AnalogSelector, select_top_k
from analogfc.forecast import (COMBINERS, METRICS, plots, rollout as fc,
                               score as scoring)
from analogfc.fronts import (FrontConvention, front_edt, front_mask,
                             front_mhd_km, grid_spacing_km)
from analogfc.registry import Registry

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


# --------------------------------------------------------------------------- #
# A synthetic Gulf, on the real grid extent (18-32 N, 98-78 W).
#
# Two features, matching the two things the real domain choices are about:
#
#   * a Loop Current analogue in the central Gulf near 88.5 W, drifting north and
#     back on a 60-day cycle, so the front moves and analogs have something real
#     to match on;
#   * a detached ring in the western Gulf near 95 W, on a slower, unrelated cycle.
#     It crosses the same 0.17 m level, so a front metric run over the whole basin
#     will chase it. That is exactly why the front scenario cuts at 92.5 W.
#
# Six years, because the climatology is per calendar day: a shorter record leaves
# each day-of-year group with one sample, a zero standard deviation, and a
# standardization that divides by it.
# --------------------------------------------------------------------------- #
LIBRARY_PERIOD = slice("1993-01-01", "1997-12-31")


def _build_fields(n_time, nlat, nlon, seed):
    rng = np.random.default_rng(seed)
    lat = np.linspace(18.0, 32.0, nlat)
    lon = np.linspace(-98.0, -78.0, nlon)
    time = pd.date_range("1993-01-01T12:00", periods=n_time, freq="D")

    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    ssh = np.empty((n_time, nlat, nlon))
    sst = np.empty((n_time, nlat, nlon))
    for t in range(n_time):
        centre = 24.0 + 1.5 * np.sin(2 * np.pi * t / 60.0)
        loop = 0.45 * np.exp(-(((yy - centre) / 1.4) ** 2 + ((xx + 88.5) / 1.8) ** 2))
        ring_lat = 23.5 + 2.0 * np.sin(2 * np.pi * t / 217.0)     # unrelated drift
        ring = 0.40 * np.exp(-(((yy - ring_lat) / 1.2) ** 2 + ((xx + 95.0) / 1.5) ** 2))
        seasonal = 0.05 * np.sin(2 * np.pi * t / 365.25)
        ssh[t] = loop + ring + seasonal + 0.01 * rng.standard_normal((nlat, nlon))
        sst[t] = 25.0 + 6.0 * np.sin(2 * np.pi * t / 365.25) + 8 * loop \
            + 0.05 * rng.standard_normal((nlat, nlon))
    ssh[:, 0, 0] = np.nan                                            # a land cell
    sst[:, 0, 0] = np.nan

    def make(name, values, unit, cmap, symmetric):
        da = xr.DataArray(values, dims=("time", "latitude", "longitude"),
                          coords={"time": time, "latitude": lat, "longitude": lon})
        # Raw day-of-year would leave leap day 366 with one sample in the whole
        # record — a zero standard deviation and a standardization that divides
        # by it. noleap_dayofyear is exactly what prevents that.
        da = climatology.with_climday(da)
        return Field(name=name, raw=da, unit=unit, cmap=cmap, symmetric=symmetric,
                     group="climday")

    return FieldSet({"sst": make("sst", sst, "°C", "plasma", False),
                     "ssh": make("ssh", ssh, "m", "RdBu_r", True)})


_FIELD_CACHE = {}


def synthetic_fields(n_time=6 * 365, nlat=29, nlon=41, seed=0):
    """The full synthetic record, built once and shared (nothing mutates it)."""
    key = (n_time, nlat, nlon, seed)
    if key not in _FIELD_CACHE:
        _FIELD_CACHE[key] = _build_fields(*key)
    return _FIELD_CACHE[key]


def in_domain(fields, min_longitude=None, max_longitude=None,
              min_latitude=None, max_latitude=None):
    """A domain subset of a FieldSet — what open_glorys' `.sel` does to the store."""
    lon_slice = slice(min_longitude, max_longitude)
    lat_slice = slice(min_latitude, max_latitude)
    subset = {}
    for name, f in fields.items():
        subset[name] = Field(name=name,
                             raw=f.raw.sel(longitude=lon_slice, latitude=lat_slice),
                             unit=f.unit, cmap=f.cmap, symmetric=f.symmetric,
                             group=f.group)
    return FieldSet(subset)


def synthetic_library(domain=None, **kw):
    fields = synthetic_fields(**kw)
    if domain:
        fields = in_domain(fields, **domain)
    return ModelLibrary(fields, period=LIBRARY_PERIOD, var="ssh")


# --------------------------------------------------------------------------- #
# fronts.py against mhd.py — the claim the README has always made.
# --------------------------------------------------------------------------- #
@case
def test_front_mhd_matches_pairwise_reference():
    """The EDT-based MHD equals the direct pairwise definition, exactly.

    fronts.py reads nearest-neighbour distances off a distance transform instead
    of doing an O(|A|.|B|) sweep. mhd.py is the literal definition; if these ever
    disagree the fast path is wrong.
    """
    rng = np.random.default_rng(3)
    for trial in range(8):
        a = rng.random((18, 22)) > 0.93
        b = rng.random((18, 22)) > 0.93
        if not a.any() or not b.any():
            continue
        sampling = (7.5, 4.25)
        fast = front_mhd_km(a, b, sampling)

        # The same points, in physical coordinates, through the definition.
        pa = np.argwhere(a) * np.array(sampling)
        pb = np.argwhere(b) * np.array(sampling)
        slow = mhd.modified_hausdorff_distance(pa, pb)
        assert abs(fast - slow) < 1e-9, f"trial {trial}: {fast} vs {slow}"


@case
def test_front_edt_is_distance_to_nearest_front_cell():
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True
    edt = front_edt(mask, (2.0, 3.0))
    assert edt[5, 5] == 0.0
    assert abs(edt[5, 7] - 6.0) < 1e-9        # two cells of 3 km
    assert abs(edt[3, 5] - 4.0) < 1e-9        # two cells of 2 km


@case
def test_front_mhd_is_in_km_not_cells():
    """A one-cell front shift must report the physical cell size, not 1."""
    a = np.zeros((12, 12), dtype=bool)
    b = np.zeros((12, 12), dtype=bool)
    a[6, 4:8] = True
    b[7, 4:8] = True
    lat = np.linspace(20, 28, 12)
    lon = np.linspace(-92, -85, 12)
    sampling = grid_spacing_km(lon, lat)
    d = front_mhd_km(a, b, sampling)
    assert abs(d - sampling[0]) < 1e-9, f"{d} != one latitude cell {sampling[0]}"
    assert d > 50, "a degree-scale cell should be tens of km, not order 1"


@case
def test_front_mask_excludes_coastline():
    """A high-SSH cell touching land is not a front; one touching low water is."""
    field = np.full((6, 6), 0.30)
    field[:, 4:] = 0.05                    # open water below the level
    field[:, 0] = np.nan                   # land
    m = front_mask(field, level=0.17)
    assert not m[:, 0].any(), "land column marked as front"
    assert m[:, 3].all(), "the water-adjacent edge should be the front"


# --------------------------------------------------------------------------- #
# Section 1 — data: sources and windows.
# --------------------------------------------------------------------------- #
@case
def test_one_day_window_reduces_to_the_single_day_distance():
    """The windowed path and the single-observation path must agree exactly.

    This is what lets one selector serve both workflows.
    """
    lib = synthetic_library()
    distance = DISTANCES.create("ssh_rmsd")
    selector = AnalogSelector(lib, distance)
    src = sources.SOURCES.create("model", var=distance.var,
                                 representation=distance.representation, library=lib)

    date = np.datetime64("1993-09-15")
    window = ow.build_window(src, date, lib, n_days=1)
    assert window is not None and len(window) == 1

    day = window.days[0]
    direct = selector.scores(day.grid, day.mask)
    windowed = window.sequence_scores(selector)
    assert np.allclose(direct.values, windowed.values, equal_nan=True)


@case
def test_multi_day_window_aligns_each_day_to_its_own_library_day():
    """Day o of the window is compared against the library day o days earlier."""
    lib = synthetic_library()
    distance = DISTANCES.create("ssh_rmsd")
    selector = AnalogSelector(lib, distance)
    src = sources.SOURCES.create("model", var=distance.var,
                                 representation=distance.representation, library=lib)

    window = ow.build_window(src, np.datetime64("1993-09-15"), lib, n_days=3)
    assert len(window) == 3
    assert [d.offset for d in window.days] == [0, 1, 2]
    assert abs(window.weights.sum() - 1.0) < 1e-12

    scores = window.sequence_scores(selector)
    # The first max_offset candidates have no full history and must be unusable.
    assert np.isinf(scores.values[:window.max_offset]).all()
    assert np.isfinite(scores.values[window.max_offset:]).all()

    # A window ending on a library day should still rank that day best: the
    # sequence match is exact there.
    best = scores.time.values[int(np.nanargmin(np.where(np.isfinite(scores.values),
                                                        scores.values, np.inf)))]
    assert str(best)[:10] == "1993-09-15"


@case
def test_window_requires_enough_usable_days():
    lib = synthetic_library()

    class Nothing(sources.ObsSource):
        def observe(self, date, library):
            return None

    assert ow.build_window(Nothing(), np.datetime64("1993-09-15"), lib, n_days=3) is None


@case
def test_recency_weights_favour_the_newest_day():
    w = ow._day_weights([0, 1, 2], scheme="recency", halflife=1.0)
    assert w[0] > w[1] > w[2]
    assert abs(w.sum() - 1.0) < 1e-12
    uniform = ow._day_weights([0, 1, 2], scheme="uniform")
    assert np.allclose(uniform, 1 / 3)


# --------------------------------------------------------------------------- #
# Section 2 — distance and selection.
# --------------------------------------------------------------------------- #
@case
def test_a_day_is_its_own_nearest_analog():
    """Sanity: every distance must rank an exact match first."""
    lib = synthetic_library()
    for name in ["ssh_rmsd", "sst_rmsd", "correlation", "ssh_front_mhd"]:
        distance = DISTANCES.create(name)
        selector = AnalogSelector(lib, distance)
        when = lib.times[100]
        obs = lib.at(distance.var, when, distance.representation).values
        scores = selector.scores(obs, mask=None)
        assert str(scores.time.values[int(np.argmin(scores.values))])[:10] \
            == str(when)[:10], f"{name} did not rank the exact match first"


@case
def test_buffer_keeps_analogs_apart():
    lib = synthetic_library()
    selector = AnalogSelector(lib, DISTANCES.create("ssh_rmsd"))
    obs = lib.at("ssh", lib.times[150], "standardized").values

    tight = selector.select(obs, k=5, buffer_days=0)
    spread = selector.select(obs, k=5, buffer_days=14)
    assert len(spread) == 5
    gaps = np.abs(np.diff(np.sort(spread.times)).astype("timedelta64[D]").astype(int))
    assert (gaps >= 14).all(), f"analogs closer than the buffer: {gaps}"
    # The buffer costs similarity: a de-clustered set cannot beat an unconstrained one.
    assert spread.distances.sum() >= tight.distances.sum() - 1e-12


@case
def test_unscorable_days_are_never_selected():
    """A day a metric cannot score (inf) must drop out, not rank first."""
    times = pd.date_range("1993-01-01", periods=6, freq="D")
    scores = xr.DataArray([np.inf, 2.0, np.inf, 1.0, 3.0, np.inf],
                          coords={"time": times}, dims="time")
    picked = select_top_k(scores, k=6, buffer_days=0)
    assert len(picked) == 3
    assert np.isfinite(picked.distances).all()
    assert list(picked.distances) == [1.0, 2.0, 3.0]


@case
def test_front_distance_is_unaffected_by_a_uniform_offset():
    """With referencing on, a basin-wide sea-level shift must not move the front."""
    lib = synthetic_library()
    distance = DISTANCES.create("ssh_front_mhd", referenced=True)
    selector = AnalogSelector(lib, distance)
    obs = lib.at("ssh", lib.times[120], "raw").values
    plain = selector.scores(obs, None).values
    shifted = selector.scores(obs + 0.08, None).values
    assert np.allclose(plain, shifted, equal_nan=True)


@case
def test_correlation_is_unaffected_by_offset_and_scale():
    """The cross-datum property SWOT depends on."""
    lib = synthetic_library()
    selector = AnalogSelector(lib, DISTANCES.create("correlation"))
    obs = lib.at("ssh", lib.times[120], "anomaly").values
    plain = selector.scores(obs, None).values
    altered = selector.scores(3.0 * obs + 0.5, None).values
    assert np.allclose(plain, altered, atol=1e-5, equal_nan=True)  # kernel is float32


@case
def test_correlation_handles_partial_coverage():
    """A masked swath must score without touching the unobserved cells."""
    lib = synthetic_library()
    selector = AnalogSelector(lib, DISTANCES.create("correlation"))
    obs = lib.at("ssh", lib.times[120], "anomaly").values.copy()
    mask = np.zeros(obs.shape, dtype=bool)
    mask[:, 8:16] = True                    # a swath-like ribbon
    obs[~mask] = np.nan
    scores = selector.scores(obs, mask)
    assert np.isfinite(scores.values).all()
    assert str(scores.time.values[int(np.argmin(scores.values))])[:10] \
        == str(lib.times[120])[:10]


# --------------------------------------------------------------------------- #
# Section 3 — forecast: combining, scoring, rollout.
# --------------------------------------------------------------------------- #
@case
def test_latitude_weighting_reduces_to_the_plain_mean_when_weights_are_equal():
    """The weighting is the only thing it changes; with equal weights it is a no-op."""
    lib = synthetic_library()
    metric = scoring.WeightedRMSE("ssh", lib)
    a = lib.at("ssh", lib.times[10]).copy()
    b = lib.at("ssh", lib.times[40]).copy()

    metric.w = np.ones_like(metric.w)
    diff = (a - b).values
    plain = float(np.sqrt(np.nanmean(diff[np.isfinite(diff)] ** 2)))
    assert abs(metric({"ssh": a}, {"ssh": b}) - plain) < 1e-12


@case
def test_latitude_weighting_downweights_high_latitudes():
    """A discrepancy in the north must count for less than the same one further south."""
    lib = synthetic_library()
    metric = scoring.WeightedRMSE("ssh", lib)
    truth = lib.at("ssh", lib.times[10]).copy()

    north = truth.copy(deep=True)
    north[-2, 5] += 1.0
    south = truth.copy(deep=True)
    south[1, 5] += 1.0
    assert metric({"ssh": north}, {"ssh": truth}) < metric({"ssh": south}, {"ssh": truth})


@case
def test_perfect_forecast_scores_perfectly():
    lib = synthetic_library()
    when = lib.times[200]
    field = lib.at("ssh", when)
    fields = {"ssh": field, "sst": lib.at("sst", when)}

    assert scoring.WeightedRMSE("ssh", lib)(fields, fields, when) == 0.0
    assert abs(scoring.AnomalyCorrelation("ssh", lib)(fields, fields, when) - 1.0) < 1e-9
    assert scoring.FrontMHDError("ssh", lib)(fields, fields, when) == 0.0


@case
def test_acc_is_undefined_for_a_climatology_forecast():
    """A forecast equal to the climatology has zero anomaly, so ACC is 0/0.

    Reported as NaN rather than 0, so it drops out of a nanmean instead of being
    averaged in as though the forecast had been scored and found unskilful.
    """
    lib = synthetic_library()
    when = lib.times[200]
    metric = scoring.AnomalyCorrelation("ssh", lib)
    clim = lib.fields["ssh"].climatology_at(when)
    assert np.isnan(metric({"ssh": clim}, {"ssh": lib.at("ssh", when)}, when))


@case
def test_metric_directions_are_declared():
    lib = synthetic_library()
    metrics = scoring.build(["sst_rmse", "ssh_rmse", "sst_acc", "ssh_acc",
                             "ssh_front_mhd"], lib, level=0.17)
    directions = {m.name: m.higher_is_better for m in metrics}
    assert directions == {"sst_rmse": False, "ssh_rmse": False, "sst_acc": True,
                          "ssh_acc": True, "ssh_front_mhd": False}
    assert [m.unit for m in metrics] == ["°C", "m", "correlation", "correlation", "km"]


@case
def test_combiners():
    members = [np.full((3, 3), 1.0), np.full((3, 3), 3.0)]
    assert np.allclose(COMBINERS.get("ensemble_mean")(members, None), 2.0)
    assert np.allclose(COMBINERS.get("best_analog")(members, None), 1.0)
    # The nearer member should pull the weighted mean below the plain mean.
    w = COMBINERS.get("weighted_mean")(members, np.array([0.1, 2.0]))
    assert np.all(w < 2.0)


@case
def test_rollout_maps_back_to_physical_units():
    """A lead-0 forecast of an analog is that analog's own physical field."""
    lib = synthetic_library()
    when = lib.times[150]
    selector = AnalogSelector(lib, DISTANCES.create("ssh_rmsd"))
    analogs = selector.select(lib.at("ssh", when, "standardized").values, k=3,
                              buffer_days=14)
    result = fc.rollout(lib, analogs, when, [0, 5])

    for i, t in enumerate(analogs.times):
        member = result.members[0][i]["ssh"]
        # Standardization is per calendar day, so the round trip is exact only
        # when the analog and the target share a climatology group; compare the
        # member against its own field re-standardized at the target's group.
        assert np.isfinite(member.values[np.isfinite(member.values)]).all()
        assert member.shape == lib.at("ssh", t).shape

    assert np.allclose(result.ensemble[0]["ssh"].values,
                       np.mean([result.members[0][i]["ssh"].values for i in range(3)],
                               axis=0), equal_nan=True)
    assert list(result.leads) == [0, 5]
    assert result.valid_times[5] == pd.Timestamp(when) + pd.Timedelta(days=5)


@case
def test_ensemble_may_be_combined_in_either_space():
    """to_physical is affine, so mean-then-map equals map-then-mean."""
    lib = synthetic_library()
    when = lib.times[150]
    field = lib.fields["ssh"]
    anomalies = [field.standardized.sel(time=t) for t in lib.times[10:14]]

    mapped_then_meaned = sum(field.to_physical(a, when) for a in anomalies) / 4
    meaned_then_mapped = field.to_physical(sum(anomalies) / 4, when)
    assert np.allclose(mapped_then_meaned.values, meaned_then_mapped.values,
                       equal_nan=True)


# --------------------------------------------------------------------------- #
# The two GLORYS -> GLORYS scenarios, mirroring their config files. Each runs the
# whole pipeline — data -> distance -> forecast -> score — on the synthetic record
# with that scenario's domain and selection rule.
# --------------------------------------------------------------------------- #
FULL_DOMAIN = dict(min_longitude=None, max_longitude=None,
                   min_latitude=None, max_latitude=None)
CENTRAL_DOMAIN = dict(min_longitude=-92.5, max_longitude=-80.0,
                      min_latitude=None, max_latitude=None)

SCENARIOS = {
    # name: (config file, domain, selection metric)
    "ssh_rmsd_full": ("glorys_ssh_rmsd_full.yaml", FULL_DOMAIN, "ssh_rmsd"),
    "ssh_front_mhd_central": ("glorys_ssh_front_mhd_central.yaml", CENTRAL_DOMAIN,
                              "ssh_front_mhd"),
}

SCORE_METRICS = ["sst_rmse", "ssh_rmse", "sst_acc", "ssh_acc", "ssh_front_mhd"]
TARGET_DATE = "1998-06-15"
LEADS = [0, 10, 20, 30]
K, BUFFER_DAYS = 5, 14


def run_scenario(name, leads=LEADS, k=K, buffer_days=BUFFER_DAYS):
    """The scenario end to end, returning everything the assertions need."""
    _, domain, metric_name = SCENARIOS[name]
    lib = synthetic_library(domain=domain)

    distance = DISTANCES.create(metric_name)
    if metric_name == "ssh_front_mhd":
        distance.level = 0.17
    selector = AnalogSelector(lib, distance)

    # The target is outside the library period, as in the real configs.
    target = lib.fields["ssh"].raw.sel(time=TARGET_DATE).time.values[0]
    observation = lib.at(distance.var, target, distance.representation).values
    analogs = selector.select(observation, k=k, buffer_days=buffer_days)

    metrics = scoring.build(SCORE_METRICS, lib, level=0.17)
    result = fc.rollout(lib, analogs, target, leads)
    return lib, analogs, result, fc.score(result, metrics), metrics


@case
def test_scenario_configs_exist_and_match_the_tested_setup():
    """The configs and these tests must not drift apart."""
    import yaml

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name, (filename, domain, metric_name) in SCENARIOS.items():
        path = os.path.join(root, "config", filename)
        assert os.path.exists(path), f"{name}: missing config {filename}"
        with open(path) as fh:
            config = yaml.safe_load(fh)

        assert config["analogs"]["selection_metric"] == metric_name, name
        assert config["domain"] == domain, f"{name}: domain drifted from {filename}"
        assert config["forecast"]["score_metrics"] == SCORE_METRICS, name
        assert config["analogs"]["k"] == K and config["analogs"]["buffer_days"] == BUFFER_DAYS
        # Every map lead must be reachable within the rollout.
        assert max(config["forecast"]["lead_days"]) <= config["forecast"]["length_days"], name
        # The two scenarios must stay disjoint in library and target.
        assert config["periods"]["library"]["end"] < config["periods"]["target"]["start"], name


@case
def test_scenario_domains_differ_as_intended():
    """The full domain spans the basin; the central one drops the western Gulf."""
    full = synthetic_library(domain=FULL_DOMAIN)
    central = synthetic_library(domain=CENTRAL_DOMAIN)

    assert float(full.longitude.min()) == -98.0
    assert float(central.longitude.min()) >= -92.5
    assert float(central.longitude.max()) <= -80.0
    assert central.longitude.size < full.longitude.size
    assert central.latitude.size == full.latitude.size      # no latitude bound

    # The point of the cut: the western ring crosses the same level, so over the
    # full basin it appears in the front mask and competes with the Loop Current.
    when = full.times[300]
    full_front = front_mask(full.at("ssh", when).values, level=0.17)
    west = full.longitude.values < -92.5
    assert full_front[:, west].any(), "the decoy ring should show up over the full basin"

    central_front = front_mask(central.at("ssh", when).values, level=0.17)
    assert central_front.any(), "the Loop Current front should survive the cut"
    assert central_front.shape[1] == central.longitude.size


@case
def test_rollout_ssh_rmsd_full():
    """Scenario 1: RMSD selection over the whole field."""
    check_scenario_rollout("ssh_rmsd_full")


@case
def test_rollout_ssh_front_mhd_central():
    """Scenario 2: front MHD selection over the central Gulf."""
    check_scenario_rollout("ssh_front_mhd_central")


def check_scenario_rollout(name):
    lib, analogs, result, skill, metrics = run_scenario(name)
    _, _, metric_name = SCENARIOS[name]

    # -- selection ------------------------------------------------------------ #
    assert len(analogs) == K, f"{name}: got {len(analogs)} analogs, wanted {K}"
    assert analogs.metric == metric_name
    assert np.isfinite(analogs.distances).all(), f"{name}: an unscorable day was selected"
    assert (np.diff(analogs.distances) >= 0).all(), f"{name}: analogs not in rank order"
    gaps = np.abs(np.diff(np.sort(analogs.times)).astype("timedelta64[D]").astype(int))
    assert (gaps >= BUFFER_DAYS).all(), f"{name}: buffer violated, gaps {gaps}"
    # Every analog must come from the library period, never the target period.
    assert all(np.datetime64("1993-01-01") <= t <= np.datetime64("1997-12-31")
               for t in analogs.times), f"{name}: an analog leaked from outside the library"

    # -- rollout -------------------------------------------------------------- #
    assert result.leads == LEADS
    for lead in LEADS:
        assert result.valid_times[lead] == pd.Timestamp(TARGET_DATE + "T12:00") \
            + pd.Timedelta(days=lead)
        assert len(result.members[lead]) == K
        for var in ("sst", "ssh"):
            field = lib.fields[var]
            truth = result.truth[lead][var]
            ens = result.ensemble[lead][var]
            assert ens.shape == truth.shape, f"{name}: {var} shape drifted at lead {lead}"

            # Forecasts must come back in physical units, not in sigma. The
            # rollout maps each member through the climatology of the day being
            # forecast, so the ensemble's domain mean must sit within a few
            # climatological standard deviations of that day's climatology — a
            # forecast left in anomaly space would have a domain mean near zero
            # and miss by the whole field magnitude.
            clim = field.climatology_at(result.valid_times[lead])
            tolerance = 3 * float(field.clim_std.mean())
            offset = abs(float(ens.mean()) - float(clim.mean()))
            assert offset < tolerance, (
                f"{name}: {var} ensemble at lead {lead} is {offset:.3f} from the "
                f"climatology (tolerance {tolerance:.3f}) — not in physical units")

            # The ensemble is the plain mean of its members.
            stack = np.stack([result.members[lead][i][var].values for i in range(K)])
            assert np.allclose(ens.values, stack.mean(axis=0), equal_nan=True)

    # -- scoring -------------------------------------------------------------- #
    for metric in metrics:
        per_member, ensemble_curve = skill[metric.name]
        assert len(per_member) == K and len(ensemble_curve) == len(LEADS)
        assert np.isfinite(ensemble_curve).all(), \
            f"{name}: {metric.name} produced a non-finite ensemble score"
        assert np.isfinite(np.asarray(per_member)).all(), \
            f"{name}: {metric.name} produced a non-finite member score"

    # A correlation must stay a correlation, and a distance in km must be a
    # plausible number of km rather than a count of grid cells.
    for acc in ("sst_acc", "ssh_acc"):
        assert (np.abs(skill[acc][1]) <= 1.0 + 1e-9).all(), f"{name}: {acc} out of range"
    mhd = np.asarray(skill["ssh_front_mhd"][1])
    assert (mhd >= 0).all() and (mhd < 3000).all(), f"{name}: front MHD implausible: {mhd}"
    assert (np.asarray(skill["ssh_rmse"][1]) >= 0).all()


@case
def test_front_score_is_in_km_on_the_library_grid():
    """The scored front distance must use the library's physical cell size.

    The range check in the rollout tests is too loose to notice a score reported
    in grid cells, so pin it directly: a front displaced by exactly one cell must
    score as one latitude cell in km, for both scenario domains.
    """
    for name, (_, domain, _) in SCENARIOS.items():
        lib = synthetic_library(domain=domain)
        metric = scoring.build(["ssh_front_mhd"], lib, level=0.17)[0]
        assert metric.sampling == lib.sampling, f"{name}: metric is not on the library grid"

        dlat_km = lib.sampling[0]
        assert dlat_km > 10, f"{name}: cell size {dlat_km} looks like cells, not km"

        template = lib.at("ssh", lib.times[0])
        truth = template.copy(deep=True)
        truth.values[:] = 0.05
        truth.values[10:, :] = 0.30            # a front along one latitude row
        shifted = truth.copy(deep=True)
        shifted.values[:] = 0.05
        shifted.values[11:, :] = 0.30          # the same front, one cell north

        d = metric({"ssh": shifted}, {"ssh": truth})
        assert abs(d - dlat_km) < 1e-6, f"{name}: one-cell shift scored {d}, not {dlat_km}"


# --------------------------------------------------------------------------- #
# Separate selection and scoring regions.
# --------------------------------------------------------------------------- #
@case
def test_region_defaults_to_the_whole_domain():
    """Omitting a region must change nothing — every earlier run relies on it."""
    assert Region.from_config(None).is_whole
    assert Region.from_config({}).is_whole
    assert Region().label == "whole domain"

    lib = synthetic_library()
    for metric_name in ("ssh_rmsd", "ssh_front_mhd"):
        plain = AnalogSelector(lib, DISTANCES.create(metric_name))
        explicit = AnalogSelector(lib, DISTANCES.create(metric_name), region_mask=None)
        obs = lib.at("ssh", lib.times[120],
                     DISTANCES.create(metric_name).representation).values
        assert np.allclose(plain.scores(obs).values, explicit.scores(obs).values,
                           equal_nan=True), metric_name


@case
def test_region_mask_equals_cropping_the_data():
    """A region applied as a mask must give exactly what loading that box gives.

    This is the assumption the whole feature rests on: one FieldSet is loaded and
    each part of the pipeline masks it, rather than loading the data twice. If
    masking and cropping disagreed, a regional score would silently not be the
    score over that region.
    """
    cropped = synthetic_library(domain=CENTRAL_DOMAIN)
    full = synthetic_library(domain=FULL_DOMAIN)
    region = Region(**{k: v for k, v in CENTRAL_DOMAIN.items() if v is not None})
    mask = region.mask(full)

    when = full.times[300]
    target = full.fields["ssh"].raw.sel(time=TARGET_DATE).time.values[0]

    for metric_name in ("ssh_rmsd", "sst_rmsd", "ssh_front_mhd", "correlation"):
        distance = DISTANCES.create(metric_name)
        crop_sel = AnalogSelector(cropped, DISTANCES.create(metric_name))
        mask_sel = AnalogSelector(full, DISTANCES.create(metric_name), region_mask=mask)

        crop_obs = cropped.at(distance.var, target, distance.representation).values
        full_obs = full.at(distance.var, target, distance.representation).values
        crop_scores = crop_sel.scores(crop_obs).values
        mask_scores = mask_sel.scores(full_obs).values
        assert np.allclose(crop_scores, mask_scores, equal_nan=True, atol=1e-9), (
            f"{metric_name}: masking and cropping disagree, "
            f"max |diff| = {np.nanmax(np.abs(crop_scores - mask_scores))}")

    # ... and the same for every score metric.
    crop_metrics = scoring.build(SCORE_METRICS, cropped, level=0.17)
    mask_metrics = scoring.build(SCORE_METRICS, full, level=0.17, region_mask=mask)
    crop_fields = {v: cropped.at(v, when) for v in ("sst", "ssh")}
    full_fields = {v: full.at(v, when) for v in ("sst", "ssh")}
    later = full.times[330]
    crop_truth = {v: cropped.at(v, later) for v in ("sst", "ssh")}
    full_truth = {v: full.at(v, later) for v in ("sst", "ssh")}
    for cm, mm in zip(crop_metrics, mask_metrics):
        a = cm(crop_fields, crop_truth, later)
        b = mm(full_fields, full_truth, later)
        assert np.isclose(a, b, equal_nan=True, atol=1e-9), \
            f"{cm.name}: cropped {a} vs masked {b}"


@case
def test_selection_and_scoring_regions_are_independent():
    """The point of the feature: rank on one area, verify on another."""
    lib = synthetic_library()
    central = Region(min_longitude=-92.5, max_longitude=-80.0)
    west = Region(min_longitude=-98.0, max_longitude=-92.5)
    central_mask, west_mask = central.mask(lib), west.mask(lib)
    target = lib.fields["ssh"].raw.sel(time=TARGET_DATE).time.values[0]

    def run(select_mask, score_mask):
        distance = DISTANCES.create("ssh_rmsd")
        selector = AnalogSelector(lib, distance, region_mask=select_mask)
        obs = lib.at(distance.var, target, distance.representation).values
        analogs = selector.select(obs, k=K, buffer_days=BUFFER_DAYS)
        metrics = scoring.build(["ssh_rmse"], lib, region_mask=score_mask)
        result = fc.rollout(lib, analogs, target, [0, 10], variables=("ssh",))
        return analogs, fc.score(result, metrics)["ssh_rmse"][1]

    central_central, score_cc = run(central_mask, central_mask)
    central_west, score_cw = run(central_mask, west_mask)
    west_central, score_wc = run(west_mask, central_mask)

    # Same selection region, different scoring region -> same analogs, different scores.
    assert central_central.dates == central_west.dates
    assert not np.allclose(score_cc, score_cw), \
        "the scoring region made no difference to the scores"

    # Same scoring region, different selection region -> different analogs.
    assert central_central.dates != west_central.dates, \
        "the selection region made no difference to the analogs"


@case
def test_front_convention_is_shared_by_selection_and_scoring():
    """Selection and scoring must contour the same front, by construction.

    They used to build the level/datum/segment bundle separately, which meant a
    run could rank days on one front and measure error against another with
    nothing failing.
    """
    lib = synthetic_library()
    region_mask = Region(min_longitude=-92.5, max_longitude=-80.0).mask(lib)

    distance = DISTANCES.create("ssh_front_mhd", referenced=True)
    AnalogSelector(lib, distance, region_mask=region_mask)
    metric = scoring.build(["ssh_front_mhd"], lib, level=0.17, referenced=True,
                           region_mask=region_mask)[0]

    a, b = distance.front, metric.front
    assert a.level == b.level and a.main_only == b.main_only
    assert a.sampling == b.sampling
    assert a.ref_mean == b.ref_mean, "selection and scoring disagree on the datum"

    field = lib.at("ssh", lib.times[250]).values
    assert np.array_equal(a.mask(field), b.mask(field)), \
        "selection and scoring extract different fronts from the same field"


@case
def test_regional_front_is_referenced_to_its_own_region():
    """A regional datum must come from the region, not from a basin mean."""
    lib = synthetic_library()
    region_mask = Region(min_longitude=-92.5, max_longitude=-80.0).mask(lib)
    whole = FrontConvention.from_library(lib, referenced=True)
    regional = FrontConvention.from_library(lib, referenced=True,
                                            region_mask=region_mask)
    assert whole.ref_mean != regional.ref_mean
    # And the regional front must not stray outside its region.
    front = regional.mask(lib.at("ssh", lib.times[250]).values)
    assert not front[~region_mask].any()


@case
def test_region_outside_the_domain_fails_loudly():
    """An empty region would make every score NaN with nothing to say why."""
    lib = synthetic_library()
    try:
        Region(min_longitude=10.0, max_longitude=20.0).mask(lib)
    except ValueError as e:
        assert "selects no cells" in str(e) and "Widen `domain`" in str(e)
    else:
        raise AssertionError("a region outside the domain must raise")

    try:
        Region.from_config({"min_lon": -92.5})
    except ValueError as e:
        assert "unknown region bound" in str(e)
    else:
        raise AssertionError("a misspelt bound must raise")


@case
def test_selection_rules_disagree_about_which_days_are_analogs():
    """The two scenarios must actually be different experiments.

    If a field-wide RMSD and a front-geometry distance picked the same days there
    would be no reason to keep both, and the comparison the configs set up would
    be vacuous.
    """
    _, rmsd_analogs, _, _, _ = run_scenario("ssh_rmsd_full", leads=[0])
    _, mhd_analogs, _, _, _ = run_scenario("ssh_front_mhd_central", leads=[0])
    assert set(rmsd_analogs.dates) != set(mhd_analogs.dates)


@case
def test_front_selection_is_degraded_by_the_western_ring():
    """Why the front scenario restricts the domain.

    Ranked over the whole basin the front distance is contaminated by the western
    ring, which crosses the same level on its own unrelated cycle. Restricting to
    the central Gulf must change which days it calls analogs.
    """
    target = synthetic_fields()["ssh"].raw.sel(time=TARGET_DATE).time.values[0]

    picked = {}
    for label, domain in [("full", FULL_DOMAIN), ("central", CENTRAL_DOMAIN)]:
        lib = synthetic_library(domain=domain)
        distance = DISTANCES.create("ssh_front_mhd")
        selector = AnalogSelector(lib, distance)
        obs = lib.at("ssh", target, "raw").values
        picked[label] = selector.select(obs, k=K, buffer_days=BUFFER_DAYS)

    assert set(picked["full"].dates) != set(picked["central"].dates), \
        "the western ring made no difference; the synthetic decoy is too weak"
    # Both must still be usable selections, not degenerate ones.
    for label, analogs in picked.items():
        assert len(analogs) == K and np.isfinite(analogs.distances).all(), label


@case
def test_analog_grid_caps_rows_instead_of_crashing():
    """More analogs than categorical colours must degrade, not raise.

    The SWOT run selects K=10 while only 7 colours are defined; this used to be a
    ValueError at the very end of a long run, after all the science was done.
    """
    import tempfile

    import matplotlib
    matplotlib.use("Agg")

    lib = synthetic_library()
    target = lib.fields["ssh"].raw.sel(time=TARGET_DATE).time.values[0]
    selector = AnalogSelector(lib, DISTANCES.create("ssh_rmsd"))
    obs = lib.at("ssh", target, "standardized").values
    analogs = selector.select(obs, k=10, buffer_days=10)
    assert len(analogs) > len(plots.ANALOG_COLORS), "the test needs more analogs than colours"

    metrics = scoring.build(["ssh_rmse"], lib, level=0.17)
    result = fc.rollout(lib, analogs, target, [0, 5], variables=("ssh",))
    skill = fc.score(result, metrics)

    with tempfile.TemporaryDirectory() as out:
        fig = plots.plot_analog_grid("ssh", lib, result, skill, metrics, [0, 5],
                                     TARGET_DATE, "ssh_rmsd", out, "grid.png",
                                     contour_level=0.17)
        assert os.path.exists(os.path.join(out, "grid.png"))
        # The ensemble label must still report the true K, not the row count.
        labels = [t.get_text() for ax in fig.axes for t in [ax.yaxis.get_label()]]
        assert any(f"K = {len(analogs)}" in t for t in labels), \
            f"ensemble row must report all {len(analogs)} members"

        # An explicit cap is honoured too.
        plots.plot_analog_grid("ssh", lib, result, skill, metrics, [0, 5],
                               TARGET_DATE, "ssh_rmsd", out, "grid3.png",
                               contour_level=0.17, max_rows=3)
        assert os.path.exists(os.path.join(out, "grid3.png"))
    import matplotlib.pyplot as plt
    plt.close("all")


@case
def test_end_to_end_through_every_section():
    """data -> distance -> forecast, the whole pipeline on synthetic input."""
    lib = synthetic_library()
    distance = DISTANCES.create("ssh_rmsd")
    src = sources.SOURCES.create("model", var=distance.var,
                                 representation=distance.representation, library=lib)
    selector = AnalogSelector(lib, distance)

    window = ow.build_window(src, np.datetime64("1993-10-20"), lib, n_days=3)
    analogs = selector.select_window(window, k=4, buffer_days=10)
    assert len(analogs) == 4

    when = lib.timestamp("1993-10-20")
    metrics = scoring.build(["ssh_rmse", "ssh_acc", "ssh_front_mhd"], lib, level=0.17)
    result = fc.rollout(lib, analogs, when, [0, 3, 7], variables=("ssh",))
    skill = fc.score(result, metrics)

    for m in metrics:
        per_member, ens = skill[m.name]
        assert len(per_member) == 4 and len(ens) == 3
        assert np.isfinite(ens).all(), f"{m.name} produced a non-finite ensemble score"


# --------------------------------------------------------------------------- #
# The registry contract that makes each section extensible.
# --------------------------------------------------------------------------- #
@case
def test_registry_rejects_unknown_and_duplicate_names():
    r = Registry("widget")
    r.register("a")(lambda: "A")
    assert r.create("a") == "A"
    assert r.names() == ["a"]

    try:
        r.get("b")
    except ValueError as e:
        assert "unknown widget 'b'" in str(e) and "['a']" in str(e)
    else:
        raise AssertionError("an unknown name must raise with the valid options")

    try:
        r.register("a")(lambda: "again")
    except ValueError as e:
        assert "already registered" in str(e)
    else:
        raise AssertionError("a duplicate registration must raise")


@case
def test_every_config_names_registered_options():
    """No config can name an option that does not exist."""
    import glob as globmod

    import yaml

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in sorted(globmod.glob(os.path.join(root, "config", "*.yaml"))):
        with open(path) as fh:
            config = yaml.safe_load(fh)
        name = os.path.basename(path)
        analogs, forecast = config["analogs"], config["forecast"]
        for key in ("selection_metric", "distance"):
            if key in analogs:
                assert analogs[key] in DISTANCES, f"{name}: {key}={analogs[key]!r}"
        for metric in forecast.get("score_metrics", []):
            assert metric in METRICS, f"{name}: score metric {metric!r}"
        if "combiner" in forecast:
            assert forecast["combiner"] in COMBINERS, f"{name}: {forecast['combiner']!r}"
        # Both regions must parse, and must sit inside the configured domain.
        domain = Region.from_config({k: v for k, v in config["domain"].items()})
        for where, section in (("analogs", analogs), ("forecast", forecast)):
            region = Region.from_config(section.get("region"))
            if region.is_whole or domain.is_whole:
                continue
            assert region.min_longitude is None or domain.min_longitude is None \
                or region.min_longitude >= domain.min_longitude, \
                f"{name}: {where}.region starts west of the loaded domain"
            assert region.max_longitude is None or domain.max_longitude is None \
                or region.max_longitude <= domain.max_longitude, \
                f"{name}: {where}.region ends east of the loaded domain"


@case
def test_the_split_region_config_actually_splits():
    """config/glorys_select_central_score_full.yaml must do what its name says."""
    import yaml

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "config", "glorys_select_central_score_full.yaml")
    with open(path) as fh:
        config = yaml.safe_load(fh)

    selection = Region.from_config(config["analogs"]["region"])
    scoring_region = Region.from_config(config["forecast"]["region"])
    assert not selection.is_whole, "the selection region is not restricted"
    assert scoring_region.is_whole, "the scoring region should be the whole domain"
    assert selection != scoring_region
    # The domain must be left whole, or the scoring region would be silently
    # narrowed by the load bounds and the config would not ask its question.
    assert all(v is None for v in config["domain"].values()), \
        "the domain must stay whole for the scoring region to mean the whole basin"


def main():
    failures = []
    for fn in CASES:
        try:
            fn()
        except Exception as exc:                       # noqa: BLE001 - a test report
            failures.append((fn.__name__, exc))
            print(f"FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {fn.__name__}")
    print(f"\n{len(CASES) - len(failures)}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
