"""Optional cleaning diagnostics, the RANSAC seed, and input-mutation behavior."""

import copy
import inspect

import numpy as np
import pytest

from eegprep import eeg_checkset, eeg_emptyset, pop_eegfiltnew, pop_select
from eegprep.plugins.clean_rawdata.clean_channels import clean_channels
from eegprep.plugins.clean_rawdata.clean_channels_nolocs import clean_channels_nolocs
from eegprep.plugins.clean_rawdata.clean_drifts import clean_drifts
from eegprep.plugins.clean_rawdata.clean_flatlines import clean_flatlines
from eegprep.plugins.clean_rawdata.clean_windows import clean_windows
from eegprep.plugins.clean_rawdata.private.ransac import calc_projector
from eegprep.plugins.clean_rawdata.private.windows import matlab_window_offsets


def _recording(
    n_channels: int = 4,
    n_samples: int = 2000,
    srate: float = 128.0,
    dtype: np.dtype = np.float64,
    *,
    noisy: bool = False,
    flat_channel: int | None = None,
    constant: bool = False,
) -> dict:
    rng = np.random.default_rng(0)
    if constant:
        data = np.full((n_channels, n_samples), 5.0)
    else:
        data = rng.normal(size=(1, n_samples)) + 0.05 * rng.normal(size=(n_channels, n_samples))
        if noisy:
            data[-1] += rng.normal(scale=30.0, size=n_samples)
        if flat_channel is not None:
            data[flat_channel] = 5.0
    data = np.asarray(data, dtype=dtype)
    eeg = eeg_emptyset()
    eeg.update(
        data=data,
        nbchan=n_channels,
        pnts=n_samples,
        trials=1,
        srate=srate,
        xmin=0.0,
        xmax=(n_samples - 1) / srate,
        event=[],
        chanlocs=[
            {
                "labels": f"E{index + 1}",
                "X": float(np.cos(2.0 * np.pi * index / n_channels)),
                "Y": float(np.sin(2.0 * np.pi * index / n_channels)),
                "Z": 0.3 + 0.1 * index,
            }
            for index in range(n_channels)
        ],
    )
    eeg = eeg_checkset(eeg)
    eeg["data"] = data
    eeg["etc"] = {}
    return eeg


def _channel_kwargs(**overrides):
    values = {
        "corr_threshold": -1.0,
        "noise_threshold": 1.0e9,
        "window_len": 5,
        "max_broken_time": 0.4,
        "num_samples": 3,
        "subset_size": 0.25,
    }
    values.update(overrides)
    return values


def test_default_cleaning_returns_keep_their_shape_and_values():
    """Requesting diagnostics is opt-in. A default call keeps today's result."""
    source = _recording()
    channels = clean_channels(copy.deepcopy(source), **_channel_kwargs())
    channels_with_report = clean_channels(
        copy.deepcopy(source),
        **_channel_kwargs(),
        return_diagnostics=True,
    )
    assert isinstance(channels, dict)
    assert isinstance(channels_with_report, tuple) and len(channels_with_report) == 2
    np.testing.assert_array_equal(channels["data"], channels_with_report[0]["data"])
    assert "clean_channels" not in channels.get("etc", {})

    nolocs = clean_channels_nolocs(copy.deepcopy(source), min_corr=0.0, linenoise_aware=False)
    nolocs_with_report = clean_channels_nolocs(
        copy.deepcopy(source),
        min_corr=0.0,
        linenoise_aware=False,
        return_diagnostics=True,
    )
    assert len(nolocs) == 2
    assert len(nolocs_with_report) == 3
    np.testing.assert_array_equal(nolocs[0]["data"], nolocs_with_report[0]["data"])
    np.testing.assert_array_equal(nolocs[1], nolocs_with_report[1])

    windows = clean_windows(copy.deepcopy(source), window_len=1.0, window_overlap=0.0)
    windows_with_report = clean_windows(
        copy.deepcopy(source),
        window_len=1.0,
        window_overlap=0.0,
        return_diagnostics=True,
    )
    assert len(windows) == 2
    assert len(windows_with_report) == 3
    np.testing.assert_array_equal(windows[0]["data"], windows_with_report[0]["data"])
    np.testing.assert_array_equal(windows[1], windows_with_report[1])

    flat = clean_flatlines(copy.deepcopy(source), max_flatline_duration=5.0)
    flat_with_report = clean_flatlines(copy.deepcopy(source), max_flatline_duration=5.0, return_diagnostics=True)
    assert isinstance(flat, dict)
    assert isinstance(flat_with_report, tuple) and len(flat_with_report) == 2
    np.testing.assert_array_equal(flat["data"], flat_with_report[0]["data"])


def test_default_seed_is_5489_and_a_chosen_seed_reaches_calc_projector():
    """The default seed reproduces RandomState(5489). Another seed is the projector's stream."""
    assert inspect.signature(clean_channels).parameters["seed"].default == 5489
    source = _recording()
    kwargs = _channel_kwargs()
    default = clean_channels(copy.deepcopy(source), **kwargs)
    explicit, explicit_report = clean_channels(
        copy.deepcopy(source),
        **kwargs,
        seed=5489,
        return_diagnostics=True,
    )
    other, other_report = clean_channels(
        copy.deepcopy(source),
        **kwargs,
        seed=7,
        return_diagnostics=True,
    )
    np.testing.assert_array_equal(default["data"], explicit["data"])
    assert explicit_report["seed"] == 5489
    assert other_report["seed"] == 7
    expected_default = calc_projector(
        explicit_report["usable_locations"],
        explicit_report["criteria"]["num_samples"],
        explicit_report["criteria"]["subset_count"],
        stream=np.random.RandomState(5489),
    )
    expected_other = calc_projector(
        other_report["usable_locations"],
        other_report["criteria"]["num_samples"],
        other_report["criteria"]["subset_count"],
        stream=np.random.RandomState(7),
    )
    np.testing.assert_array_equal(explicit_report["projector"], expected_default)
    np.testing.assert_array_equal(other_report["projector"], expected_other)
    assert not np.array_equal(explicit_report["projector"], other_report["projector"])


def test_clean_channels_diagnostics_are_the_arrays_that_decide_removal():
    """Noise z-scores and correlation flags are the values compared with the criteria."""
    source = _recording(noisy=True)
    _dataset, quiet = clean_channels(
        copy.deepcopy(source),
        **_channel_kwargs(noise_threshold=1.0e9),
        return_diagnostics=True,
    )
    scores = quiet["noise_zscores"]
    assert scores is not None and scores.shape == (source["nbchan"],)
    ranked = np.argsort(scores)
    assert scores[ranked[-1]] > scores[ranked[-2]]
    threshold = float((scores[ranked[-1]] + scores[ranked[-2]]) / 2.0)
    noisy = int(ranked[-1])
    prior = np.ones(source["nbchan"], dtype=bool)
    prepared = copy.deepcopy(source)
    prepared["etc"] = {"clean_channel_mask": prior.copy()}
    cleaned, report = clean_channels(
        prepared,
        **_channel_kwargs(noise_threshold=threshold),
        return_diagnostics=True,
    )

    assert report["index_base"] == 0
    assert report["criteria"]["noise_criterion_applied"] is True
    np.testing.assert_array_equal(report["noise_flags"], report["noise_zscores"] > threshold)
    np.testing.assert_array_equal(
        report["correlation_flags"], report["correlations"] < report["criteria"]["corr_threshold"]
    )
    np.testing.assert_array_equal(report["flagged_window_counts"], np.sum(report["correlation_flags"], axis=1))
    expected = np.zeros(source["nbchan"], dtype=bool)
    expected[report["usable_channels"]] = (
        report["flagged_window_counts"] * report["window_length"] > report["criteria"]["max_broken_time_samples"]
    )
    expected |= report["noise_flags"]
    np.testing.assert_array_equal(report["detected_channels"], expected)
    np.testing.assert_array_equal(report["removed_channels"], expected)
    assert report["removal_suppressed"] is None
    assert report["correlations"].shape == (len(report["usable_channels"]), len(report["window_offsets"]))
    assert cleaned["nbchan"] == source["nbchan"] - 1
    assert prepared["nbchan"] == source["nbchan"]
    np.testing.assert_array_equal(prepared["etc"]["clean_channel_mask"], prior)
    np.testing.assert_array_equal(report["prior_clean_channel_mask"], prior)
    assert report["clean_channel_mask"].shape == (source["nbchan"],)
    assert bool(report["clean_channel_mask"][noisy]) is False
    assert int(np.sum(report["clean_channel_mask"])) == source["nbchan"] - 1
    np.testing.assert_array_equal(cleaned["etc"]["clean_channel_mask"], report["clean_channel_mask"])


def test_clean_channels_below_100_hz_has_no_noise_scores():
    """The noise detector does not run at 100 Hz, so the report does not invent scores."""
    source = _recording(n_samples=500, srate=100.0)
    _cleaned, report = clean_channels(
        source,
        **_channel_kwargs(window_len=2),
        return_diagnostics=True,
    )
    assert report["criteria"]["noise_criterion_applied"] is False
    assert report["noise_scores"] is None
    assert report["noise_zscores"] is None
    assert not report["noise_flags"].any()
    assert report["correlations"].shape[0] == len(report["usable_channels"])
    assert report["window_offsets"].ndim == 1
    assert int(report["window_offsets"][0]) == 0


def test_nolocs_diagnostics_keep_sorted_correlations_and_the_retained_rows():
    """The report keeps the sorted absolute correlations and the rows the threshold uses."""
    source = _recording(n_samples=400, srate=100.0)
    cleaned, removed, report = clean_channels_nolocs(
        copy.deepcopy(source),
        min_corr=0.45,
        ignored_quantile=0.1,
        window_len=1.0,
        max_broken_time=0.5,
        linenoise_aware=False,
        return_diagnostics=True,
    )
    retained = report["criteria"]["retained_correlation_rows"]
    correlations = report["sorted_absolute_correlations"]
    expected_flags = np.all(correlations[retained] < 0.45, axis=0)
    np.testing.assert_array_equal(report["flagged"], expected_flags)
    np.testing.assert_array_equal(removed, report["detected_channels"])
    assert correlations.shape[0] == source["nbchan"]
    assert correlations.shape[1] == source["nbchan"]
    assert report["window_offsets"].shape[0] == correlations.shape[2]
    assert "best_correlation" not in report
    assert cleaned["nbchan"] <= source["nbchan"]


def test_all_flagged_nolocs_channels_are_kept_and_the_mask_stays_unset():
    """Every flagged channel is reported, and a retained-channel mask of all false is not stored."""
    source = _recording(n_samples=400, srate=100.0)
    cleaned, removed, report = clean_channels_nolocs(
        source,
        min_corr=1.0,
        ignored_quantile=0.1,
        window_len=1.0,
        max_broken_time=0.5,
        linenoise_aware=False,
        return_diagnostics=True,
    )
    assert removed.all()
    assert report["detected_channels"].all()
    assert not report["removed_channels"].any()
    assert report["removal_suppressed"] == (
        "All channels are flagged bad according to the used criterion: not removing anything."
    )
    assert cleaned is source
    assert cleaned["nbchan"] == source["nbchan"]
    assert "clean_channel_mask" not in cleaned["etc"]
    assert report["clean_channel_mask"] is None


def test_all_flat_channels_are_reported_without_writing_a_retention_mask():
    """A recording that is entirely flat keeps its mask and reports why nothing was removed."""
    source = _recording(constant=True)
    prior = np.ones(source["nbchan"], dtype=bool)
    with_prior = copy.deepcopy(source)
    with_prior["etc"] = {"clean_channel_mask": prior.copy()}

    cleaned, report = clean_flatlines(source, max_flatline_duration=5.0, return_diagnostics=True)
    kept, kept_report = clean_flatlines(with_prior, max_flatline_duration=5.0, return_diagnostics=True)

    assert cleaned is source
    assert report["detected_channels"].all()
    assert not report["removed_channels"].any()
    assert report["removal_suppressed"] == "All channels have a flat-line portion; not removing anything."
    assert "clean_channel_mask" not in cleaned["etc"]
    assert report["clean_channel_mask"] is None
    assert kept is with_prior
    np.testing.assert_array_equal(kept["etc"]["clean_channel_mask"], prior)
    np.testing.assert_array_equal(kept_report["prior_clean_channel_mask"], prior)
    np.testing.assert_array_equal(kept_report["clean_channel_mask"], prior)
    assert kept_report["clean_channel_mask"].all()


def test_one_flat_channel_composes_the_prior_mask_and_leaves_the_caller():
    """Removing one channel returns a new dataset. The caller's mask stays the prior one."""
    source = _recording(flat_channel=0)
    prior = np.ones(source["nbchan"], dtype=bool)
    source["etc"] = {"clean_channel_mask": prior.copy()}
    caller_data = source["data"]
    cleaned, report = clean_flatlines(source, max_flatline_duration=1.0, return_diagnostics=True)

    assert cleaned is not source
    assert source["nbchan"] == 4
    assert source["data"] is caller_data
    assert source["etc"]["clean_channel_mask"] is not cleaned["etc"]["clean_channel_mask"]
    np.testing.assert_array_equal(source["etc"]["clean_channel_mask"], prior)
    assert cleaned["nbchan"] == 3
    expected = np.array([False, True, True, True])
    np.testing.assert_array_equal(cleaned["etc"]["clean_channel_mask"], expected)
    np.testing.assert_array_equal(report["prior_clean_channel_mask"], prior)
    np.testing.assert_array_equal(report["clean_channel_mask"], expected)
    np.testing.assert_array_equal(report["detected_channels"], np.array([True, False, False, False]))
    np.testing.assert_array_equal(report["removed_channels"], report["detected_channels"])
    assert report["removal_suppressed"] is None


def test_window_diagnostics_expose_offsets_zscores_and_prior_mask_composition():
    """Window offsets and z-scores are the ones that build the sample mask."""
    source = _recording(n_samples=2000, srate=100.0)
    n_samples = source["pnts"]
    window_len = 1.0
    overlap = 0.0
    cleaned, sample_mask, report = clean_windows(
        copy.deepcopy(source),
        max_bad_channels=0.25,
        zthresholds=(-3.5, 5.0),
        window_len=window_len,
        window_overlap=overlap,
        return_diagnostics=True,
    )
    n_window = float(window_len) * float(source["srate"])
    step = n_window * (1.0 - overlap)
    np.testing.assert_array_equal(report["window_offsets"], matlab_window_offsets(n_samples, n_window, step))
    assert report["zscores"].shape == (source["nbchan"], len(report["window_offsets"]))
    assert report["window_length"] == int(np.floor(n_window))
    assert not np.allclose(report["zscores"], 0.0)
    zmin, zmax = report["criteria"]["zthresholds"]
    ordered = np.sort(report["zscores"], axis=0)
    max_bad = report["criteria"]["max_bad_channels"]
    expected = np.zeros(ordered.shape[1], dtype=bool)
    if zmax > 0:
        high = max(min(source["nbchan"] - max_bad - 1, source["nbchan"] - 1), 0)
        expected |= ordered[high] > zmax
    if zmin < 0:
        low = max(min(max_bad, source["nbchan"] - 1), 0)
        expected |= ordered[low] < zmin
    np.testing.assert_array_equal(report["window_flags"], expected)
    np.testing.assert_array_equal(sample_mask, report["sample_mask"])
    np.testing.assert_array_equal(cleaned["etc"]["clean_sample_mask"], sample_mask)
    assert sample_mask.shape == (n_samples,)

    prior = np.ones(n_samples + 4, dtype=bool)
    prior[:4] = False
    prepared = copy.deepcopy(source)
    prepared["etc"] = {"clean_sample_mask": prior.copy()}
    caller_data = prepared["data"]
    revised, revised_mask, revised_report = clean_windows(
        prepared,
        max_bad_channels=0.25,
        zthresholds=(-3.5, 5.0),
        window_len=window_len,
        window_overlap=overlap,
        return_diagnostics=True,
    )
    assert prepared["data"] is caller_data
    np.testing.assert_array_equal(prepared["etc"]["clean_sample_mask"], prior)
    np.testing.assert_array_equal(revised_report["prior_clean_sample_mask"], prior)
    np.testing.assert_array_equal(revised_report["clean_sample_mask"][prior], revised_mask)
    np.testing.assert_array_equal(revised["etc"]["clean_sample_mask"], revised_report["clean_sample_mask"])
    assert revised is not prepared


def test_cleaning_functions_follow_the_documented_mutation_contract():
    """Float32 inputs follow the copy and dtype behavior documented for each function."""
    source = _recording(dtype=np.float32)
    original = source["data"].copy()

    drifts = copy.deepcopy(source)
    drifts_data = drifts["data"]
    drifted = clean_drifts(drifts)
    assert drifted is drifts
    assert drifts["data"] is not drifts_data
    assert drifts["data"].dtype == np.float64
    assert drifts["etc"]["clean_drifts_kernel"].ndim == 1

    flat = copy.deepcopy(source)
    flat_data = flat["data"]
    flat_result = clean_flatlines(flat, max_flatline_duration=5.0)
    assert flat_result is flat
    assert flat["data"] is flat_data
    assert flat["data"].dtype == np.float32
    assert flat["etc"] == {}

    channels = copy.deepcopy(source)
    channels_data = channels["data"]
    cleaned = clean_channels(channels, **_channel_kwargs())
    assert cleaned is channels
    assert channels["data"] is not channels_data
    assert channels["data"].dtype == np.float64
    np.testing.assert_array_equal(channels["data"], original.astype(np.float64))
    assert "clean_channel_mask" not in channels["etc"]

    nolocs = copy.deepcopy(source)
    nolocs_data = nolocs["data"]
    nolocs_result, _removed = clean_channels_nolocs(nolocs, min_corr=0.0, linenoise_aware=False)
    assert nolocs_result is nolocs
    assert nolocs["data"] is not nolocs_data
    assert nolocs["data"].dtype == np.float64
    np.testing.assert_array_equal(nolocs["data"], original.astype(np.float64))
    assert "clean_channel_mask" not in nolocs["etc"]

    windows = copy.deepcopy(source)
    windows_data = windows["data"]
    windowed, _mask = clean_windows(windows, window_len=1.0, window_overlap=0.0)
    assert windowed is not windows
    assert windows["data"] is windows_data
    assert windows["data"].dtype == np.float32
    assert windows["etc"] == {}
    assert windowed["data"].dtype == np.float32
    assert "clean_sample_mask" in windowed["etc"]

    filtered_input = copy.deepcopy(source)
    filtered_data = filtered_input["data"]
    filtered = pop_eegfiltnew(filtered_input, locutoff=1.0, plotfreqz=False)
    assert filtered is not filtered_input
    assert filtered_input["data"] is filtered_data
    assert filtered_input["data"].dtype == np.float32
    assert filtered_input["etc"] == {}

    selected_input = copy.deepcopy(source)
    selected_data = selected_input["data"]
    selected = pop_select(selected_input, channel=list(range(source["nbchan"])))
    assert selected is not selected_input
    assert selected_input["data"] is selected_data
    assert selected_input["data"].dtype == np.float32
    assert selected_input["etc"] == {}


def test_clean_channels_positional_criteria_still_select_the_correlation_threshold():
    """Existing positional calls keep their meaning once seed is a keyword."""
    kept = clean_channels(
        _recording(),
        -1.0,
        1.0e9,
        5,
        0.4,
        3,
        0.25,
    )
    assert kept["nbchan"] == 4
    with pytest.raises(TypeError):
        clean_channels(_recording(), -1.0, 1.0e9, 5, 0.4, 3, 0.25, 7)
