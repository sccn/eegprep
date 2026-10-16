"""MATLAB comparisons for the ds003061 run-1 stages used by the manuscript harness.

Each stage test loads one saved dataset and runs that stage in Python and in
EEGLAB. The chained pipeline is opt-in (``--run-heavy-reference``) because it
is the only check that lets earlier differences accumulate.

MATLAB ``option_single`` is turned off for these calls and restored afterward.
The snapshot reads the ``option_file`` left in the base workspace by
``eeglab_options`` (``~/eeg_options.m`` when that file is the one EEGLAB
loads) and writes those bytes back. EEGLAB otherwise stores the shared input
as single, so Picard and interpolation see different numbers from the Python
float64 arrays. Picard must be mind-inria/picard 617ce6a (identity
initialization). The EEGLAB PICARD 2.0 extension draws a seeded random start.

The numeric ceilings are unchanged. MATLAB comparison data is the double array
from the function result; ``pop_saveset`` still writes the dataset file as single.
The chain matches rejected components by absolute correlation of mixing
columns, so a permutation or sign flip still has to reject the same components.
"""

from __future__ import annotations

import copy
import json
import platform
import re
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.io
from scipy.optimize import linear_sum_assignment

from eegprep.functions.adminfunc.eeglabcompat import MatlabWrapper, _restore_double_eeg_data
from eegprep.functions.miscfunc.event_utils import is_boundary_event
from eegprep.functions.popfunc.eeg_interp import eeg_interp
from eegprep.functions.popfunc.eeg_picard import eeg_picard
from eegprep.functions.popfunc.pop_epoch import pop_epoch
from eegprep.functions.popfunc.pop_load_frombids import pop_load_frombids
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_reref import pop_reref
from eegprep.functions.popfunc.pop_rmbase import pop_rmbase
from eegprep.functions.popfunc.pop_saveset import pop_saveset
from eegprep.functions.popfunc.pop_select import pop_select
from eegprep.functions.popfunc.pop_subcomp import pop_subcomp
from eegprep.plugins.clean_rawdata.clean_artifacts import clean_artifacts
from eegprep.plugins.ICLabel.iclabel import iclabel
from eegprep.plugins.ICLabel.pop_icflag import pop_icflag
from tests.eeg_options_snapshot import eeg_options_snapshot, restore_eeg_options
from tests.eeglab_tests.backend import skip_without_matlab_plugin

SUBJECT = "sub-001"
EPOCH_TYPES = ["standard", "oddball", "oddball_with_reponse"]
EPOCH_LIMITS = [-0.5, 1.0]
BASELINE_MS = [-500.0, 0.0]
ICFLAG_THRESHOLDS = np.array(
    [
        [np.nan, np.nan],
        [0.9, 1.0],
        [0.9, 1.0],
        [np.nan, np.nan],
        [np.nan, np.nan],
        [np.nan, np.nan],
        [np.nan, np.nan],
    ]
)
CLEAN_PARAMS = {
    "FlatlineCriterion": 5.0,
    "ChannelCriterion": 0.8,
    "LineNoiseCriterion": 4.0,
    "Highpass": (0.25, 0.75),
    "BurstCriterion": 20.0,
    "WindowCriterion": 0.25,
    "BurstRejection": "on",
    "Distance": "Euclidian",
    "WindowCriterionTolerances": (-np.inf, 7.0),
}
REREF_ATOL = 1e-5
CLEAN_ATOL = 1e-5
EPOCH_ATOL = 1e-5
RMBASE_ATOL = 1e-6
INTERP_KEPT_ATOL = 1e-5
INTERP_NEW_ATOL = 1e-4
SUBCOMP_ATOL = 1e-6
PICARD_AMARI_MAX = 1e-5
PICARD_CORR_MIN = 0.9999
EVENT_ATOL = 1e-6
EXPECTED_SUMMARY = Path(__file__).resolve().parent / "parity_expected" / "ds003061_pipeline.json"
EXACT_SUMMARY_KEYS = {
    "channels_removed",
    "boundaries",
    "pnts",
    "trials",
    "rejected",
    "epoch_failed",
}
MAX_SUMMARY_KEYS = {"max_abs", "rms", "amari"}
MIN_SUMMARY_KEYS = {"corr_min", "corr_mean"}


def amari_distance(unmixing, mixing):
    """Return the Amari index of ``unmixing @ mixing`` (0 for a signed permutation)."""
    product = np.asarray(unmixing, dtype=float) @ np.asarray(mixing, dtype=float)
    magnitude = np.abs(product)
    width = magnitude.shape[0]
    if magnitude.shape != (width, width):
        raise AssertionError(f"AMARI product shape {magnitude.shape}")
    row_scale = magnitude.max(axis=1)
    column_scale = magnitude.max(axis=0)
    if np.any(row_scale == 0) or np.any(column_scale == 0):
        raise AssertionError("AMARI input has a zero row or column")
    row_term = np.sum(magnitude / row_scale[:, None]) - width
    column_term = np.sum(magnitude / column_scale[None, :]) - width
    return float((row_term + column_term) / (2 * width))


def _mixing_correlation(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 2 or left.shape != right.shape:
        raise AssertionError(f"mixing shape {left.shape} vs {right.shape}")
    components = left.shape[1]
    correlation = np.empty((components, components))
    for row in range(components):
        for column in range(components):
            correlation[row, column] = np.corrcoef(left[:, row], right[:, column])[0, 1]
    if not np.isfinite(correlation).all():
        raise AssertionError("mixing-matrix correlation was not finite")
    return correlation


def matched_mixing_correlation(left, right):
    """Return mean and minimum absolute correlation of uniquely matched mixing columns."""
    correlation = _mixing_correlation(left, right)
    assignment = linear_sum_assignment(-np.abs(correlation))
    matched = np.abs(correlation[assignment])
    return float(matched.mean()), float(matched.min())


def matched_components(left, right):
    """Return the right-hand column matched to each left-hand mixing column.

    Matching uses absolute correlation and is a bijection. Each pair must be
    the unique best match in its row and column, at or above
    ``PICARD_CORR_MIN``, with every runner-up below that bar.
    """
    correlation = np.abs(_mixing_correlation(left, right))
    components = correlation.shape[0]
    if components == 0:
        return np.zeros(0, dtype=int)
    rows, columns = linear_sum_assignment(-correlation)
    partner = np.empty(components, dtype=int)
    partner[rows] = columns
    if len(set(partner.tolist())) != components:
        raise AssertionError("component match is not one-to-one")
    for python_index, matlab_index in enumerate(partner.tolist()):
        score = float(correlation[python_index, matlab_index])
        row = correlation[python_index]
        column = correlation[:, matlab_index]
        if int(np.argmax(row)) != matlab_index or np.count_nonzero(row == score) != 1:
            raise AssertionError(f"component {python_index} has no unique row match")
        if int(np.argmax(column)) != python_index or np.count_nonzero(column == score) != 1:
            raise AssertionError(f"component {python_index} -> {matlab_index} has no unique column match")
        runner_up = max(
            float(np.max(np.delete(row, matlab_index))) if components > 1 else -np.inf,
            float(np.max(np.delete(column, python_index))) if components > 1 else -np.inf,
        )
        if score < PICARD_CORR_MIN or runner_up >= PICARD_CORR_MIN:
            raise AssertionError(
                f"component {python_index} -> {matlab_index} correlation {score:.6f} "
                f"is not separated from the runner-up {runner_up:.6f}"
            )
    return partner


def test_amari_distance_is_zero_for_a_signed_permutation():
    generator = np.random.default_rng(0)
    mixing = generator.standard_normal((8, 8))
    signs = np.diag(generator.choice([-1.0, 1.0], size=8))
    permutation = np.eye(8)[generator.permutation(8)]
    unmixing = np.linalg.pinv(mixing)
    other_unmixing = signs @ permutation @ unmixing
    other_mixing = np.linalg.pinv(other_unmixing)

    assert amari_distance(other_unmixing, mixing) < 1e-8
    mean_correlation, min_correlation = matched_mixing_correlation(other_mixing, mixing)
    assert mean_correlation > 1 - 1e-8
    assert min_correlation > 1 - 1e-8


def test_pipeline_expected_summary_file_is_a_platform_map():
    payload = json.loads(EXPECTED_SUMMARY.read_text(encoding="utf-8"))
    recorded = payload["platforms"]["linux_x86_64_R2025b"]
    assert set(recorded) == {f"sub-{index:03d}" for index in range(1, 14)}


def test_rejected_components_follow_a_signed_permutation():
    generator = np.random.default_rng(1)
    mixing = generator.standard_normal((64, 5))
    order = np.array([3, 0, 4, 1, 2])
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    other = mixing[:, order] * signs
    partner = matched_components(other, mixing)

    np.testing.assert_array_equal(partner, order)
    python_eeg = {"icawinv": other, "reject": {"gcompreject": [1, 0, 1, 0, 0]}}
    matlab_flags = np.zeros(5, dtype=int)
    matlab_flags[order[[0, 2]]] = 1
    matlab_eeg = {"icawinv": mixing, "reject": {"gcompreject": matlab_flags}}
    _assert_same_rejected_components("sub-signed", python_eeg, matlab_eeg)

    matlab_eeg["reject"]["gcompreject"] = np.array([1, 0, 0, 0, 0])
    with pytest.raises(AssertionError, match="rejected"):
        _assert_same_rejected_components("sub-signed", python_eeg, matlab_eeg)


def test_component_match_rejects_an_ambiguous_runner_up():
    mixing = np.eye(4)
    tied = mixing.copy()
    tied[:, 1] = mixing[:, 0]
    with pytest.raises(AssertionError, match="runner-up|unique"):
        matched_components(mixing, tied)


def _run1_eeg_set(suite_root, subject):
    directory = Path(suite_root) / "ds003061" / subject / "eeg"
    matches = sorted(directory.glob("*_run-1_eeg.set"))
    if len(matches) != 1:
        pytest.fail(f"expected one run-1 .set under {directory}, found {[str(path) for path in matches]}")
    return matches[0]


def _import_subject(suite_root, subject):
    imported = pop_load_frombids(
        str(_run1_eeg_set(suite_root, subject)),
        eventtype="value",
        bidsevent=True,
        bidschanloc=False,
        infer_locations="standard-10-5-342ch.locs",
        verbose=False,
    )
    selected = pop_select(imported, chantype="EEG")
    if subject != "sub-012" and int(selected["nbchan"]) != 64:
        raise AssertionError(f"{subject} EEG selection has {selected['nbchan']} channels, expected 64")
    return selected


def _reload(eeg, path):
    pop_saveset(eeg, str(path))
    return pop_loadset(str(path))


def _chanlocs(eeg):
    locations = eeg.get("chanlocs")
    if locations is None:
        return []
    if isinstance(locations, dict):
        return [locations]
    if isinstance(locations, np.ndarray):
        if locations.size == 0:
            return []
        locations = locations.tolist()
    return list(locations)


def _labels(eeg):
    return [str(location.get("labels", "")) for location in _chanlocs(eeg)]


def _events(eeg):
    events = eeg.get("event")
    if events is None:
        return []
    if isinstance(events, dict):
        events = [events]
    elif isinstance(events, np.ndarray):
        events = [] if events.size == 0 else events.tolist()
    return [event for event in events if isinstance(event, dict)]


def _event_rows(eeg):
    rows = []
    for event in _events(eeg):
        event_type = event.get("type", "")
        if isinstance(event_type, bytes):
            event_type = event_type.decode("utf-8")
        duration = event.get("duration", 0) or 0
        rows.append((str(event_type), float(event.get("latency")), float(duration)))
    return rows


def _assert_events_match(python_eeg, matlab_eeg):
    python_rows = _event_rows(python_eeg)
    matlab_rows = _event_rows(matlab_eeg)
    assert len(python_rows) == len(matlab_rows), f"{len(python_rows)} Python events vs {len(matlab_rows)} MATLAB events"
    for index, (python_row, matlab_row) in enumerate(zip(python_rows, matlab_rows)):
        assert python_row[0] == matlab_row[0], f"event {index} type {python_row[0]} vs {matlab_row[0]}"
        assert abs(python_row[1] - matlab_row[1]) <= EVENT_ATOL, (
            f"event {index} latency {python_row[1]} vs {matlab_row[1]}"
        )
        assert np.isclose(python_row[2], matlab_row[2], rtol=0.0, atol=EVENT_ATOL, equal_nan=True), (
            f"event {index} duration {python_row[2]} vs {matlab_row[2]}"
        )


def _boundary_count(eeg):
    return sum(1 for event in _events(eeg) if is_boundary_event(event))


def _removed_labels(before, after):
    kept = set(_labels(after))
    return [label for label in _labels(before) if label not in kept]


def _flags(eeg):
    reject = eeg.get("reject")
    if not isinstance(reject, dict):
        return np.zeros(0, dtype=int)
    values = np.asarray(reject.get("gcompreject", []), dtype=float).ravel()
    return (values != 0).astype(int)


def _assert_same_rejected_components(subject, python_eeg, matlab_eeg):
    partner = matched_components(python_eeg["icawinv"], matlab_eeg["icawinv"])
    python_flags = _flags(python_eeg)
    matlab_flags = _flags(matlab_eeg)
    if python_flags.shape != (partner.size,) or matlab_flags.shape != (partner.size,):
        raise AssertionError(
            f"{subject} rejection flags {python_flags.shape} and {matlab_flags.shape} "
            f"do not match {partner.size} components"
        )
    python_rejected = {int(index) for index in np.flatnonzero(python_flags)}
    matlab_rejected = {int(index) for index in np.flatnonzero(matlab_flags)}
    mapped = {int(partner[index]) for index in python_rejected}
    assert mapped == matlab_rejected, (
        f"{subject} rejected Python components {sorted(python_rejected)} "
        f"map to {sorted(mapped)}, MATLAB rejected {sorted(matlab_rejected)}; "
        f"python {_threshold_report(python_eeg)}; matlab {_threshold_report(matlab_eeg)}"
    )


def _numeric_diff(python_eeg, matlab_eeg, name):
    python_data = np.asarray(python_eeg["data"], dtype=np.float64)
    matlab_data = np.asarray(matlab_eeg["data"], dtype=np.float64)
    assert python_data.shape == matlab_data.shape, f"{name} shape {python_data.shape} vs {matlab_data.shape}"
    difference = np.abs(python_data - matlab_data)
    max_abs = float(difference.max()) if difference.size else 0.0
    rms = float(np.sqrt(np.mean(difference * difference))) if difference.size else 0.0
    warnings.warn(f"{name}: max_abs={max_abs:.6e} uV rms={rms:.6e} uV", UserWarning, stacklevel=2)
    return max_abs, rms


def _assert_data_close(python_eeg, matlab_eeg, atol, name):
    max_abs, rms = _numeric_diff(python_eeg, matlab_eeg, name)
    assert max_abs <= atol, f"{name} max_abs={max_abs:.6e} uV exceeds {atol:.6e}"
    return max_abs, rms


def _matlab_string(path):
    return str(path).replace("\\", "/").replace("'", "''")


def _matlab_restore_montage(engine, reduced, full, directory):
    reduced_path = Path(directory) / "reduced.set"
    full_path = Path(directory) / "full.set"
    output_path = Path(directory) / "interp.set"
    double_path = Path(directory) / "interp.double.mat"
    pop_saveset(reduced, str(reduced_path))
    pop_saveset(full, str(full_path))
    engine.eval(
        "reduced = pop_loadset('" + _matlab_string(reduced_path) + "'); "
        "reference = pop_loadset('" + _matlab_string(full_path) + "'); "
        "out = eeg_interp(reduced, reference.chanlocs, 'spherical'); "
        "eegprep_roundtrip_data = double(out.data); "
        "save('" + _matlab_string(double_path) + "', 'eegprep_roundtrip_data', '-v7'); "
        "pop_saveset(out, '" + _matlab_string(output_path) + "'); "
        "clear reduced reference out eegprep_roundtrip_data;",
        nargout=0,
    )
    restored = pop_loadset(str(output_path))
    _restore_double_eeg_data(restored, str(double_path))
    return restored


class _Side:
    """One preprocessing side. ``engine`` None selects the Python functions."""

    def __init__(self, engine=None):
        self.wrapper = None if engine is None else MatlabWrapper(engine)
        self.engine = engine

    def reref(self, eeg):
        if self.wrapper is None:
            return pop_reref(eeg, [])
        return self.wrapper.pop_reref(eeg, np.empty((0, 0)))

    def clean(self, eeg):
        if self.wrapper is None:
            cleaned, _highpass, _burst, _removed = clean_artifacts(eeg, **CLEAN_PARAMS)
            return cleaned
        return self.wrapper.clean_artifacts(eeg, **CLEAN_PARAMS)

    def picard(self, eeg):
        if self.wrapper is None:
            return eeg_picard(eeg)
        return self.wrapper.eeg_picard(eeg)

    def label(self, eeg):
        if self.wrapper is None:
            return iclabel(eeg)
        return self.wrapper.iclabel(eeg)

    def icflag(self, eeg):
        if self.wrapper is None:
            return pop_icflag(eeg, ICFLAG_THRESHOLDS, gui=False)
        return self.wrapper.pop_icflag(eeg, ICFLAG_THRESHOLDS)

    def subcomp(self, eeg):
        if self.wrapper is None:
            return pop_subcomp(eeg, [], 0, gui=False)
        return self.wrapper.pop_subcomp(eeg, np.array([]), np.float64(0.0))

    def epoch(self, eeg):
        if self.wrapper is None:
            epoched, _indices = pop_epoch(eeg, EPOCH_TYPES, EPOCH_LIMITS)
            return epoched
        return self.wrapper.pop_epoch(eeg, EPOCH_TYPES, np.asarray(EPOCH_LIMITS, dtype=float))

    def rmbase(self, eeg):
        if self.wrapper is None:
            return pop_rmbase(eeg, BASELINE_MS, gui=False)
        return self.wrapper.pop_rmbase(eeg, np.asarray(BASELINE_MS, dtype=float))


def _unmixing(eeg):
    return np.asarray(eeg["icaweights"], dtype=float) @ np.asarray(eeg["icasphere"], dtype=float)


def _as_float64(eeg):
    copied = copy.deepcopy(eeg)
    copied["data"] = np.asarray(copied["data"], dtype=np.float64)
    return copied


def _require_clean(engine):
    skip_without_matlab_plugin(engine, "clean_artifacts", "clean_rawdata")


def _require_iclabel(engine):
    skip_without_matlab_plugin(engine, "iclabel", "ICLabel")
    skip_without_matlab_plugin(engine, "pop_icflag", "ICLabel")


def _require_picard(engine):
    skip_without_matlab_plugin(engine, "eeg_picard", "PICARD")
    skip_without_matlab_plugin(engine, "picard_python_port", "PICARD")
    source = str(engine.eval("fileread(which('eeg_picard'))", nargout=1))
    if "picard_python_port" not in source:
        pytest.fail(
            "MATLAB eeg_picard does not call picard_python_port. Install mind-inria/picard "
            "617ce6ae669cfbd1dfedba6a5fce39a602b99ac3. The EEGLAB PICARD 2.0 extension uses a seeded random start."
        )


@pytest.fixture
def matlab(eeglab_matlab_engine):
    options_path, original = eeg_options_snapshot(eeglab_matlab_engine)
    eeglab_matlab_engine.eval("pop_editoptions('option_single', 0);", nargout=0)
    try:
        yield MatlabWrapper(eeglab_matlab_engine)
    finally:
        restore_eeg_options(options_path, original)
        eeglab_matlab_engine.eval("clear functions", nargout=0)
        eeglab_matlab_engine.eval("clear EEG OUT args reduced reference out", nargout=0)


@pytest.fixture(scope="session")
def p300_imported(eeglab_suite_root, tmp_path_factory):
    directory = tmp_path_factory.mktemp("p300-imported")
    return _reload(_import_subject(eeglab_suite_root, SUBJECT), directory / "imported.set")


@pytest.fixture(scope="session")
def p300_reref(p300_imported, tmp_path_factory):
    directory = tmp_path_factory.mktemp("p300-reref")
    return _reload(pop_reref(p300_imported, []), directory / "reref.set")


@pytest.fixture(scope="session")
def p300_clean(p300_reref, tmp_path_factory):
    directory = tmp_path_factory.mktemp("p300-clean")
    cleaned, _highpass, _burst, _removed = clean_artifacts(p300_reref, **CLEAN_PARAMS)
    return _reload(cleaned, directory / "clean.set")


@pytest.fixture(scope="session")
def p300_picard(p300_clean):
    return eeg_picard(p300_clean)


@pytest.fixture
def pipeline_parity(request):
    if not request.config.getoption("--run-heavy-reference"):
        pytest.skip("heavy EEGLAB reference test; pass --run-heavy-reference to run")


def test_average_reference_matches_matlab_on_64_channel_float32(p300_imported, matlab):
    """Guard the column-major float32 accumulation against EEGLAB on a real 64-channel recording."""
    source = p300_imported
    assert int(source["nbchan"]) == 64
    assert np.asarray(source["data"]).dtype == np.float32

    python_eeg = _Side().reref(source)
    matlab_eeg = _Side(matlab.engine).reref(source)

    assert _labels(python_eeg) == _labels(matlab_eeg)
    _assert_data_close(python_eeg, matlab_eeg, REREF_ATOL, f"{SUBJECT} pop_reref")


def test_clean_artifacts_matches_matlab_with_burst_rejection(p300_reref, matlab):
    _require_clean(matlab.engine)
    source = p300_reref
    python_eeg = _Side().clean(source)
    matlab_eeg = _Side(matlab.engine).clean(source)

    assert _removed_labels(source, python_eeg) == _removed_labels(source, matlab_eeg)
    assert _labels(python_eeg) == _labels(matlab_eeg)
    _assert_events_match(python_eeg, matlab_eeg)
    assert _boundary_count(python_eeg) == _boundary_count(matlab_eeg)
    assert int(python_eeg["pnts"]) == int(matlab_eeg["pnts"])
    warnings.warn(
        f"{SUBJECT} clean_artifacts: removed={len(_removed_labels(source, python_eeg))} "
        f"boundaries={_boundary_count(python_eeg)} pnts={int(python_eeg['pnts'])}",
        UserWarning,
        stacklevel=1,
    )
    _assert_data_close(python_eeg, matlab_eeg, CLEAN_ATOL, f"{SUBJECT} clean_artifacts")


def test_picard_amari_and_matched_correlation_on_cleaned_data(p300_clean, p300_picard, matlab):
    _require_picard(matlab.engine)
    assert np.asarray(p300_clean["data"]).dtype == np.float64
    matlab_eeg = _Side(matlab.engine).picard(p300_clean)
    python_unmixing = _unmixing(p300_picard)
    matlab_unmixing = _unmixing(matlab_eeg)
    assert python_unmixing.shape == matlab_unmixing.shape, (
        f"unmixing shape {python_unmixing.shape} vs {matlab_unmixing.shape}"
    )

    distance = amari_distance(python_unmixing, np.asarray(matlab_eeg["icawinv"], dtype=float))
    mean_correlation, min_correlation = matched_mixing_correlation(p300_picard["icawinv"], matlab_eeg["icawinv"])
    warnings.warn(
        f"{SUBJECT} picard: amari={distance:.6e} corr_mean={mean_correlation:.6f} corr_min={min_correlation:.6f}",
        UserWarning,
        stacklevel=1,
    )
    assert distance <= PICARD_AMARI_MAX, f"AMARI {distance:.6e} exceeds {PICARD_AMARI_MAX:.6e}"
    assert min_correlation >= PICARD_CORR_MIN, (
        f"min matched correlation {min_correlation:.6f} is below {PICARD_CORR_MIN}"
    )


def test_epoch_and_time_range_baseline_match_matlab_after_sample_rejection(p300_clean, matlab, tmp_path):
    source = p300_clean
    assert _boundary_count(source) > 0, "sample rejection did not insert a boundary event to epoch across"

    python_epoch = _Side().epoch(source)
    matlab_epoch = _Side(matlab.engine).epoch(source)
    assert int(python_epoch["trials"]) == int(matlab_epoch["trials"])
    assert int(python_epoch["trials"]) > 1
    assert int(python_epoch["pnts"]) == int(matlab_epoch["pnts"])
    _assert_events_match(python_epoch, matlab_epoch)
    _assert_data_close(python_epoch, matlab_epoch, EPOCH_ATOL, f"{SUBJECT} pop_epoch")

    shared_epoch = _reload(python_epoch, tmp_path / "epoch.set")
    times = np.asarray(shared_epoch["times"], dtype=float).ravel()
    assert times[0] <= BASELINE_MS[0] and times[-1] >= BASELINE_MS[1]
    python_baseline = _Side().rmbase(shared_epoch)
    matlab_baseline = _Side(matlab.engine).rmbase(shared_epoch)
    assert int(python_baseline["trials"]) == int(matlab_baseline["trials"])
    _assert_data_close(python_baseline, matlab_baseline, RMBASE_ATOL, f"{SUBJECT} pop_rmbase")


def test_interp_restores_the_full_montage_like_matlab(p300_reref, matlab, tmp_path):
    full = _as_float64(p300_reref)
    count = int(full["nbchan"])
    drop = set(range(0, count, 8))
    keep = [index for index in range(count) if index not in drop]
    reduced = copy.deepcopy(full)
    reduced["data"] = np.asarray(full["data"])[keep]
    reduced["chanlocs"] = [_chanlocs(full)[index] for index in keep]
    reduced["nbchan"] = len(keep)
    for field in ("icaweights", "icasphere", "icawinv", "icaact"):
        reduced[field] = np.array([])
    reduced["icachansind"] = np.array([], dtype=int)

    full_loaded = _reload(full, tmp_path / "full.set")
    reduced_loaded = _reload(reduced, tmp_path / "reduced.set")
    python_eeg = eeg_interp(reduced_loaded, _chanlocs(full_loaded), method="spherical")
    matlab_eeg = _matlab_restore_montage(matlab.engine, reduced_loaded, full_loaded, tmp_path)

    assert _labels(python_eeg) == _labels(matlab_eeg) == _labels(full_loaded)
    kept = np.array([label in set(_labels(reduced_loaded)) for label in _labels(python_eeg)])
    python_data = np.asarray(python_eeg["data"], dtype=np.float64)
    matlab_data = np.asarray(matlab_eeg["data"], dtype=np.float64)
    assert python_data.shape == matlab_data.shape
    kept_diff = float(np.abs(python_data[kept] - matlab_data[kept]).max())
    new_diff = float(np.abs(python_data[~kept] - matlab_data[~kept]).max())
    warnings.warn(
        f"{SUBJECT} eeg_interp: kept_max_abs={kept_diff:.6e} uV new_max_abs={new_diff:.6e} uV dropped={len(drop)}",
        UserWarning,
        stacklevel=1,
    )
    assert kept_diff <= INTERP_KEPT_ATOL, f"preserved channels differ by {kept_diff:.6e} uV"
    assert new_diff <= INTERP_NEW_ATOL, f"interpolated channels differ by {new_diff:.6e} uV"


def test_icflag_and_subcomp_match_matlab_on_the_same_classification(p300_picard, matlab):
    _require_iclabel(matlab.engine)
    classified = iclabel(p300_picard)
    python_flagged = _Side().icflag(classified)
    matlab_flagged = _Side(matlab.engine).icflag(classified)
    python_flags = _flags(python_flagged)
    matlab_flags = _flags(matlab_flagged)
    warnings.warn(
        f"{SUBJECT} pop_icflag: rejected python={int(python_flags.sum())} matlab={int(matlab_flags.sum())}",
        UserWarning,
        stacklevel=1,
    )
    assert python_flags.shape == matlab_flags.shape
    assert np.array_equal(python_flags, matlab_flags)

    python_eeg = _Side().subcomp(python_flagged)
    matlab_eeg = _Side(matlab.engine).subcomp(python_flagged)
    assert _labels(python_eeg) == _labels(matlab_eeg)
    _assert_data_close(python_eeg, matlab_eeg, SUBCOMP_ATOL, f"{SUBJECT} pop_subcomp")


def _error_text(exc):
    return f"{type(exc).__name__}: {exc}"[:500]


def _call(function, eeg):
    try:
        return function(copy.deepcopy(eeg)), None
    except Exception as exc:
        return None, _error_text(exc)


def _step_pair(name, python_side, matlab_side, python_eeg, matlab_eeg):
    python_out, python_error = _call(python_side, python_eeg)
    matlab_out, matlab_error = _call(matlab_side, matlab_eeg)
    if python_error or matlab_error:
        raise AssertionError(f"{name} failed: python={python_error} matlab={matlab_error}")
    assert _labels(python_out) == _labels(matlab_out), name
    max_abs, rms = _numeric_diff(python_out, matlab_out, name)
    return python_out, matlab_out, max_abs, rms


def _chain_subject(subject, imported, engine):
    """Run the harness order on one subject. Each side consumes its own previous output."""
    python = _Side()
    matlab = _Side(engine)
    summary = {}
    python_ref, matlab_ref, max_abs, rms = _step_pair("reref", python.reref, matlab.reref, imported, imported)
    summary["reref"] = {"max_abs": max_abs, "rms": rms}
    python_ref = _as_float64(python_ref)
    matlab_ref = _as_float64(matlab_ref)

    python_eeg, matlab_eeg, max_abs, rms = _step_pair(
        "clean_artifacts", python.clean, matlab.clean, python_ref, matlab_ref
    )
    summary["clean_artifacts"] = {
        "max_abs": max_abs,
        "rms": rms,
        "channels_removed": len(_removed_labels(python_ref, python_eeg)),
        "boundaries": _boundary_count(python_eeg),
        "pnts": int(python_eeg["pnts"]),
    }
    assert _removed_labels(python_ref, python_eeg) == _removed_labels(matlab_ref, matlab_eeg)
    _assert_events_match(python_eeg, matlab_eeg)

    python_eeg, matlab_eeg, _max_abs, _rms = _step_pair("picard", python.picard, matlab.picard, python_eeg, matlab_eeg)
    distance = amari_distance(_unmixing(python_eeg), np.asarray(matlab_eeg["icawinv"], dtype=float))
    mean_correlation, min_correlation = matched_mixing_correlation(python_eeg["icawinv"], matlab_eeg["icawinv"])
    summary["picard"] = {"amari": distance, "corr_min": min_correlation, "corr_mean": mean_correlation}
    warnings.warn(
        f"{subject} chain picard: amari={distance:.6e} corr_min={min_correlation:.6f}",
        UserWarning,
        stacklevel=1,
    )

    python_eeg, matlab_eeg, _max_abs, _rms = _step_pair("iclabel", python.label, matlab.label, python_eeg, matlab_eeg)
    python_eeg, matlab_eeg, _max_abs, _rms = _step_pair("icflag", python.icflag, matlab.icflag, python_eeg, matlab_eeg)
    python_flags = _flags(python_eeg)
    warnings.warn(
        f"{subject} pop_icflag: python={_threshold_report(python_eeg)} matlab={_threshold_report(matlab_eeg)}",
        UserWarning,
        stacklevel=1,
    )
    _assert_same_rejected_components(subject, python_eeg, matlab_eeg)
    rejected = int(python_flags.sum())
    python_eeg, matlab_eeg, max_abs, rms = _step_pair("subcomp", python.subcomp, matlab.subcomp, python_eeg, matlab_eeg)
    summary["subcomp"] = {"max_abs": max_abs, "rms": rms, "rejected": rejected}

    python_restored = eeg_interp(copy.deepcopy(python_eeg), _chanlocs(python_ref), method="spherical")
    with tempfile.TemporaryDirectory(prefix="eegprep-chain-interp-") as directory:
        matlab_restored = _matlab_restore_montage(engine, matlab_eeg, matlab_ref, Path(directory))
    assert _labels(python_restored) == _labels(matlab_restored) == _labels(python_ref)
    max_abs, rms = _numeric_diff(python_restored, matlab_restored, f"{subject} interp")
    summary["interp"] = {"max_abs": max_abs, "rms": rms}

    python_epoch, python_error = _call(python.epoch, python_restored)
    matlab_epoch, matlab_error = _call(matlab.epoch, matlab_restored)
    if python_error or matlab_error:
        assert python_error and matlab_error, f"{subject} epoch disagreed: python={python_error} matlab={matlab_error}"
        summary["epoch_failed"] = True
        warnings.warn(f"{subject} epoch failed on both sides", UserWarning, stacklevel=1)
        return summary

    assert int(python_epoch["trials"]) == int(matlab_epoch["trials"])
    _assert_events_match(python_epoch, matlab_epoch)
    python_base, _matlab_base, max_abs, rms = _step_pair(
        "pop_rmbase", python.rmbase, matlab.rmbase, python_epoch, matlab_epoch
    )
    summary["epoch"] = {"max_abs": max_abs, "rms": rms, "trials": int(python_base["trials"])}
    summary["epoch_failed"] = False
    return summary


def _platform_key(engine):
    version = engine.eval("version", nargout=1)
    match = re.search(r"R20\d{2}[ab]", str(version))
    release = match.group(0) if match else "unknown"
    return f"{platform.system().lower()}_{platform.machine().lower()}_{release}"


def _assert_stored_summary(measured, ceiling):
    assert set(measured) == set(ceiling), f"subjects {sorted(measured)} vs stored {sorted(ceiling)}"
    for subject, stats in measured.items():
        expected = ceiling[subject]
        assert set(stats) == set(expected), f"{subject} keys {sorted(stats)} vs {sorted(expected)}"
        for key, value in stats.items():
            target = expected[key]
            if isinstance(value, dict):
                _assert_summary_block(f"{subject} {key}", value, target)
            else:
                _assert_summary_value(f"{subject} {key}", key, value, target)


def _assert_summary_block(label, measured, ceiling):
    assert set(measured) == set(ceiling), f"{label} keys {sorted(measured)} vs {sorted(ceiling)}"
    for key, value in measured.items():
        _assert_summary_value(f"{label} {key}", key, value, ceiling[key])


def _assert_summary_value(label, key, value, target):
    if key in EXACT_SUMMARY_KEYS or isinstance(target, (bool, str)):
        assert value == target, f"{label}: {value} != {target}"
        return
    if key in MAX_SUMMARY_KEYS:
        limit = max(float(target) * 1.05, float(target) + 1e-8)
        assert float(value) <= limit, f"{label}: {value} exceeds stored ceiling {target}"
        return
    if key in MIN_SUMMARY_KEYS:
        assert float(value) >= float(target) - 1e-6, f"{label}: {value} is below stored {target}"
        return
    raise AssertionError(f"{label}: no comparison rule for {key}")


def _subjects(suite_root):
    root = Path(suite_root) / "ds003061"
    if not root.is_dir():
        pytest.fail(f"ds003061 is not in the eeglab_tests checkout at {root}")
    names = sorted(path.name for path in root.glob("sub-*") if path.is_dir())
    return [name for name in names if (_run1_set_count(root, name))]


def _run1_set_count(root, subject):
    return len(list((root / subject / "eeg").glob("*_run-1_eeg.set"))) == 1


def _classifications(eeg):
    etc = eeg.get("etc")
    if not isinstance(etc, dict):
        return None
    record = etc.get("ic_classification")
    if not isinstance(record, dict):
        return None
    iclabel_record = record.get("ICLabel")
    if not isinstance(iclabel_record, dict):
        return None
    values = iclabel_record.get("classifications")
    if values is None:
        return None
    return np.asarray(values, dtype=float)


def _threshold_report(eeg):
    values = _classifications(eeg)
    if values is None or values.ndim != 2 or values.shape[1] < 3:
        return "no ICLabel classifications"
    parts = []
    for name, column in (("Muscle", 1), ("Eye", 2)):
        scores = values[:, column]
        index = int(np.argmin(np.abs(scores - 0.9)))
        parts.append(f"{name}[{index + 1}]={scores[index]:.8f}")
    return " ".join(parts)


def _saved_strings(value):
    array = np.asarray(value)
    if array.dtype.kind in "SU":
        return [str(item) for item in array.ravel()]
    strings = []
    for item in np.asarray(value, dtype=object).ravel():
        if isinstance(item, np.ndarray):
            if item.dtype.kind in "SU":
                strings.append("".join(item.astype(str).ravel()))
            elif item.size == 0:
                strings.append("")
            else:
                strings.append(str(item.reshape(-1)[0]))
        elif isinstance(item, bytes):
            strings.append(item.decode())
        else:
            strings.append(str(item))
    return strings


def _matlab_import_run1(engine, bids_root, subject, output_dir, mat_path):
    """Import one subject with MATLAB pop_importbids and save run-1 arrays."""
    bids = _matlab_string(bids_root)
    out = _matlab_string(output_dir)
    mat = _matlab_string(mat_path)
    subject_id = subject.replace("'", "''")
    engine.eval(
        "clear ALLEEG STUDY EEG; "
        f"[~, ALLEEG] = pop_importbids('{bids}', "
        f"'outputdir', '{out}', "
        "'bidsevent', 'on', "
        "'bidschanloc', 'off', "
        "'bidscoord', 'off', "
        "'eventtype', 'value', "
        f"'subjects', {{'{subject_id}'}}); "
        "if isempty(ALLEEG), error('pop_importbids returned no datasets'); end; "
        "keep = []; "
        "for k = 1:numel(ALLEEG), "
        "name = ALLEEG(k).filename; "
        "if contains(name, '_run-1_'), keep(end+1) = k; end; "
        "end; "
        "if numel(keep) ~= 1, error('expected one run-1 dataset, found %d', numel(keep)); end; "
        "EEG = ALLEEG(keep); "
        "n = numel(EEG.event); "
        "event_type = cell(1, n); "
        "event_latency = nan(1, n); "
        "event_duration = nan(1, n); "
        "for iEvent = 1:n, "
        "event_type{iEvent} = char(string(EEG.event(iEvent).type)); "
        "event_latency(iEvent) = double(EEG.event(iEvent).latency); "
        "event_duration(iEvent) = double(EEG.event(iEvent).duration); "
        "end; "
        "nbchan = double(EEG.nbchan); "
        "pnts = double(EEG.pnts); "
        "srate = double(EEG.srate); "
        "trials = double(EEG.trials); "
        "data = double(EEG.data); "
        f"save('{mat}', 'event_type', 'event_latency', 'event_duration', "
        "'nbchan', 'pnts', 'srate', 'trials', 'data', '-v7'); "
        "clear ALLEEG STUDY EEG data event_type event_latency event_duration;",
        nargout=0,
    )


def test_load_frombids_matches_matlab_pop_importbids(pipeline_parity, matlab, eeglab_suite_root, tmp_path):
    """Compare chain step 1 with MATLAB pop_importbids, not a second Python import."""
    skip_without_matlab_plugin(matlab.engine, "pop_importbids", "EEG-BIDS")
    bids_root = Path(eeglab_suite_root) / "ds003061"
    subjects = _subjects(eeglab_suite_root)
    if "sub-001" not in subjects:
        pytest.fail("ds003061 sub-001 run-1 is not in the eeglab_tests checkout")

    for subject in subjects:
        python_eeg = pop_load_frombids(
            str(_run1_eeg_set(eeglab_suite_root, subject)),
            eventtype="value",
            bidsevent=True,
            bidschanloc=False,
            infer_locations="standard-10-5-342ch.locs",
            verbose=False,
        )
        mat_path = tmp_path / f"{subject}-run1.mat"
        output_dir = tmp_path / subject
        output_dir.mkdir(parents=True, exist_ok=True)
        _matlab_import_run1(matlab.engine, bids_root, subject, output_dir, mat_path)
        saved = scipy.io.loadmat(mat_path, squeeze_me=True, chars_as_strings=True)

        assert int(python_eeg["nbchan"]) == int(saved["nbchan"]), subject
        assert int(python_eeg["pnts"]) == int(saved["pnts"]), subject
        assert int(python_eeg["trials"]) == int(saved["trials"]), subject
        assert float(python_eeg["srate"]) == float(saved["srate"]), subject
        python_data = np.asarray(python_eeg["data"], dtype=np.float64)
        matlab_data = np.asarray(saved["data"], dtype=np.float64)
        assert python_data.shape == matlab_data.shape, f"{subject} data {python_data.shape} vs {matlab_data.shape}"
        data_diff = float(np.max(np.abs(python_data - matlab_data))) if python_data.size else 0.0
        assert data_diff == 0.0, f"{subject} import data max_abs={data_diff:.6e}"

        python_rows = _event_rows(python_eeg)
        matlab_types = _saved_strings(saved["event_type"])
        matlab_latency = np.asarray(saved["event_latency"], dtype=float).ravel()
        matlab_duration = np.asarray(saved["event_duration"], dtype=float).ravel()
        assert len(python_rows) == len(matlab_types) == matlab_latency.size == matlab_duration.size, (
            f"{subject} events {len(python_rows)} vs {len(matlab_types)}"
        )
        for index, python_row in enumerate(python_rows):
            assert python_row[0] == matlab_types[index], (
                f"{subject} event {index} type {python_row[0]} vs {matlab_types[index]}"
            )
            assert abs(python_row[1] - float(matlab_latency[index])) <= EVENT_ATOL, (
                f"{subject} event {index} latency {python_row[1]} vs {matlab_latency[index]}"
            )
            assert np.isclose(python_row[2], float(matlab_duration[index]), atol=EVENT_ATOL, equal_nan=True), (
                f"{subject} event {index} duration {python_row[2]} vs {matlab_duration[index]}"
            )


def test_chained_pipeline_matches_stored_platform_summary(pipeline_parity, matlab, eeglab_suite_root, request):
    """Compare the full chain, including sub-012, with the stored summary for this platform."""
    _require_clean(matlab.engine)
    _require_picard(matlab.engine)
    _require_iclabel(matlab.engine)
    subjects = _subjects(eeglab_suite_root)
    if "sub-012" not in subjects:
        pytest.fail("ds003061 sub-012 run-1 is not in the eeglab_tests checkout")
    measured = {}
    for subject in subjects:
        warnings.warn(f"chain {subject}", UserWarning, stacklevel=1)
        measured[subject] = _chain_subject(subject, _import_subject(eeglab_suite_root, subject), matlab.engine)

    key = _platform_key(matlab.engine)
    payload = json.loads(EXPECTED_SUMMARY.read_text(encoding="utf-8"))
    sub012 = measured["sub-012"]
    assert sub012.get("epoch_failed") is False, sub012
    assert int(sub012["epoch"]["trials"]) == 71

    if request.config.getoption("--record-parity-summary"):
        payload.setdefault("platforms", {})[key] = measured
        EXPECTED_SUMMARY.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        warnings.warn(f"recorded pipeline summary for {key}", UserWarning, stacklevel=1)
        return

    stored = payload.get("platforms", {}).get(key)
    rendered = json.dumps(measured, indent=2)
    if stored is None:
        raise AssertionError(
            f"no stored pipeline summary for {key}. Re-run with --record-parity-summary and commit the "
            f"updated {EXPECTED_SUMMARY.name}. Measured:\n{rendered}"
        )
    _assert_stored_summary(measured, stored)
