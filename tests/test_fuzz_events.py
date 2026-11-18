"""Event-aware fuzz targets. MATLAB is not required."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from eegprep.functions.popfunc.pop_saveset import pop_saveset
from tests.fixtures import create_test_eeg
from tests.fuzz.harness import (
    EEGSpec,
    FakeOracle,
    FuzzCase,
    Outcome,
    PythonBackend,
    SideResult,
    case_from_payload,
    data_sha256,
    dispatch_matlab,
    dispatch_python,
    draw_cases,
    evaluate_case,
    freeze_params,
    judge,
    load_tolerances,
    materialize_eeg,
    side_from_call,
    thaw_params,
    write_repro,
)
from tests.fuzz.targets import extra_sanity, rmbase_boundary_offset

_U = 2.0**-53
_GOLDEN = Path(__file__).resolve().parent / "fuzz" / "golden_draws_20261012.json"


def _spec() -> EEGSpec:
    return EEGSpec((0, 1), 0, 8, 1, -1, 0, 0)


def _case(function: str) -> FuzzCase:
    return FuzzCase(function, _spec(), freeze_params({}))


def _side(**changes) -> SideResult:
    side = SideResult(True, np.ones((2, 4)), 128.0, None, None, xmax=1.0, xmin=0.0)
    return replace(side, **changes)


def _event(latency: float, kind: str = "square") -> tuple:
    return (kind, latency, None, None, None)


@pytest.mark.parametrize(
    "option, selection",
    [
        ("channel", [1, 1, 4, 2]),
        ("channel", ["Ch1", "Ch2", "Ch4"]),
        ("nochannel", [1, 1, 4, 2]),
    ],
)
def test_select_adapter_converts_channel_numbers_only(option, selection):
    eeg = create_test_eeg(n_channels=4, n_samples=8, n_trials=2)
    params = {option: selection, "trial": [1]}
    case = FuzzCase("pop_select", _spec(), freeze_params(params))
    expected = [0, 1, 3] if option == "channel" else [2]

    out = dispatch_python(case, eeg)

    np.testing.assert_array_equal(out["data"], eeg["data"][expected, :, 0])
    assert [chan["labels"] for chan in out["chanlocs"]] == [eeg["chanlocs"][index]["labels"] for index in expected]
    assert thaw_params(case.params) == params


def test_select_duplicate_draw_uses_cropped_channel_count():
    duplicate = None
    for seed_value in range(20261105, 20261125):
        for case in draw_cases("pop_select", seed_value=seed_value, examples=8):
            params = thaw_params(case.params)
            if params.get("channel") == [1, 1, len(case.spec.channels), 2]:
                duplicate = case
                break
        if duplicate is not None:
            break
    assert duplicate is not None
    nbchan = len(duplicate.spec.channels)
    eeg = materialize_eeg(duplicate)
    out = dispatch_python(duplicate, eeg)
    np.testing.assert_array_equal(out["data"], eeg["data"][[0, 1, nbchan - 1]])


class _ShiftedOracle:
    def run(self, case, eeg):
        side = PythonBackend().run(case, eeg)
        if side.data is None:
            return side
        return replace(side, data=side.data + 1.0)


_DRAW_SCRIPT = """
import hashlib
import json
from pathlib import Path
from tests.fuzz.catalog import FUNCTION_NAMES
from tests.fuzz.harness import draw_cases, spec_to_json, thaw_params
golden = json.loads(Path("tests/fuzz/golden_draws_20261012.json").read_text())
original = ("pop_eegfiltnew", "pop_reref", "pop_resample", "clean_asr", "pop_clean_rawdata")
blob = []
for name in FUNCTION_NAMES:
    drawn = []
    for case in draw_cases(name, seed_value=20261012, examples=4):
        drawn.append({
            "function": case.function,
            "params": thaw_params(case.params),
            "sanity": case.sanity,
            "spec": spec_to_json(case.spec),
        })
    if name in original:
        assert drawn == golden[name], name
    blob.append({"function": name, "cases": drawn})
print("DRAWHASH " + hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest())
"""


def _draw_in_subprocess(prelude: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", prelude + "\n" + _DRAW_SCRIPT],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )


def test_original_draws_match_the_golden_snapshot():
    completed = _draw_in_subprocess("")
    assert completed.returncode == 0, completed.stdout + completed.stderr


# Hypothesis 6.169 copies integer literals from every already-imported local
# module into its integer draws. These imports are the two that moved the
# golden file: the sigprocfunc package (its eager init loads the submodule
# literals a lazy init would leave out) and gauss (fdlibm exp constants).
# The probe is a new local module full of literals outside the -100..100 band
# Hypothesis already upweights, so a later module is covered too.
_INTEGER_LITERAL_PROBE = """
import importlib.util
import sys
import tempfile
from pathlib import Path
probe = Path(tempfile.gettempdir()) / f"eegprep_fuzz_integer_literals_{__import__('os').getpid()}.py"
values = [128, 256, 512, 1024, 2048, 4096, 0x3FF00000, 0x40862E42, 0x3FE62E43]
values.extend(range(1000, 1600))
probe.write_text("\\n".join(f"LITERAL_{index} = {value}" for index, value in enumerate(values)) + "\\n")
spec = importlib.util.spec_from_file_location("eegprep_fuzz_integer_literals", probe)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
import eegprep.functions.sigprocfunc
import eegprep.functions.miscfunc.gauss
"""


@pytest.mark.parametrize(
    "prelude",
    [
        "",
        "import eegprep.functions.sigprocfunc.eegfilt",
        "import eegprep.plugins.firfilt.pop_eegfiltnew",
        "import eegprep.functions.sigprocfunc.runica",
        "import tests.fuzz.targets",
        "import eegprep.functions.sigprocfunc",
        "import eegprep.functions.miscfunc.gauss",
        _INTEGER_LITERAL_PROBE,
    ],
)
def test_draws_do_not_depend_on_import_order(prelude):
    fresh = _draw_in_subprocess("")
    other = _draw_in_subprocess(prelude)
    assert fresh.returncode == 0, fresh.stdout + fresh.stderr
    assert other.returncode == 0, other.stdout + other.stderr
    fresh_hash = [line for line in fresh.stdout.splitlines() if line.startswith("DRAWHASH ")]
    other_hash = [line for line in other.stdout.splitlines() if line.startswith("DRAWHASH ")]
    assert fresh_hash == other_hash


def test_old_repro_spec_materializes_like_a_legacy_crop():
    payload = {
        "function": "pop_reref",
        "params": {"exclude": [], "keepref": "off", "ref": []},
        "sanity": False,
        "spec": {
            "channels": [0, 1],
            "decimate": 1,
            "length": 32,
            "nan_channel": -1,
            "nan_length": 0,
            "nan_start": 0,
            "start": 4,
        },
    }
    loaded = case_from_payload(payload)
    legacy = FuzzCase(
        "pop_reref",
        EEGSpec((0, 1), 4, 32, 1, -1, 0, 0),
        freeze_params(payload["params"]),
    )
    assert loaded.spec.events == "none"
    assert loaded.pipeline == ()
    assert data_sha256(materialize_eeg(loaded)["data"]) == data_sha256(materialize_eeg(legacy)["data"])


def test_new_sanity_cases_match_under_the_fake_oracle():
    oracle = FakeOracle("match", PythonBackend())
    tolerances = load_tolerances()
    for case in extra_sanity(EEGSpec, FuzzCase):
        outcome = evaluate_case(case, materialize_eeg, PythonBackend(), oracle, tolerances)
        assert outcome.both_ok, (case.function, outcome.error_class, outcome.detail)
        assert not outcome.diverged, (case.function, outcome.error_class, outcome.detail)


def test_event_mask_and_meta_mismatches_use_their_error_class():
    tolerances = load_tolerances()
    epoch = _case("pop_epoch")
    latency = tolerances["pop_epoch"]
    matched = _side(events=(_event(10.0),))
    assert judge(matched, matched, latency, epoch, "digest").error_class == "match"
    shifted = _side(events=(_event(10.0 + 1e-12),))
    assert judge(shifted, matched, latency, epoch, "digest").error_class == "match"
    far = _side(events=(_event(10.0 + 1e-10),))
    assert judge(far, matched, latency, epoch, "digest").error_class == "event-latency"
    dropped = _side(events=())
    assert judge(dropped, matched, latency, epoch, "digest").error_class == "event-count"
    renamed = _side(events=(_event(10.0, "rt"),))
    assert judge(renamed, matched, latency, epoch, "digest").error_class == "event-field"
    present = _side(events=(("square", 10.0, 0.0, None, None),))
    assert judge(present, matched, latency, epoch, "digest").error_class == "event-field"
    missing = float("nan")
    nan_duration = _side(events=(("square", 10.0, missing, None, None),))
    assert judge(nan_duration, nan_duration, latency, epoch, "digest").error_class == "match"
    assert judge(nan_duration, matched, latency, epoch, "digest").error_class == "event-field"

    windows = _case("clean_windows")
    mask = np.array([True, False, True, True])
    flipped = np.array([True, True, True, True])
    left = _side(clean_sample_mask=flipped)
    right = _side(clean_sample_mask=mask)
    assert judge(left, right, tolerances["clean_windows"], windows, "digest").error_class == "mask"

    labels = _case("eeg_eegrej")
    meta = tolerances["eeg_eegrej"]
    oracle = _side(labels=("Fz", "Cz"))
    swapped = _side(labels=("Cz", "Fz"))
    assert judge(swapped, oracle, meta, labels, "digest").error_class == "meta"
    # 0.75 has a unit in the last place of u, so these offsets are exact.
    baseline = _side(xmax=0.75)
    beyond = _side(xmax=0.75 + 5.0 * _U)
    assert judge(beyond, baseline, meta, labels, "digest").error_class == "meta"
    within = _side(xmax=0.75 + 3.0 * _U)
    assert judge(within, baseline, meta, labels, "digest").error_class == "match"


def test_new_target_draws_are_deterministic_per_seed():
    first = draw_cases("pop_epoch", seed_value=20261017, examples=3)
    second = draw_cases("pop_epoch", seed_value=20261017, examples=3)
    other = draw_cases("pop_epoch", seed_value=20261018, examples=3)
    assert first == second
    assert first != other


def test_event_repro_round_trip_keeps_events(tmp_path):
    case = next(item for item in extra_sanity(EEGSpec, FuzzCase) if item.function == "pop_epoch")
    eeg = materialize_eeg(case)
    outcome = Outcome(case, True, True, "event-latency", 0.0, 0.0, "detail")
    path = write_repro(tmp_path, outcome, 20261017, eeg)
    payload = json.loads(path.read_text())
    assert payload["spec"]["events"] == "keep"
    loaded = case_from_payload(payload)
    again = materialize_eeg(loaded)
    assert data_sha256(again["data"]) == data_sha256(np.asarray(eeg["data"]))
    assert len(list(np.asarray(again["event"]).ravel())) > 0
    assert int(again["trials"]) >= 1


def test_rmbase_boundary_offset_is_not_filed():
    spec = EEGSpec(
        tuple(range(32)),
        0,
        256,
        1,
        -1,
        0,
        0,
        events="keep+boundary",
        boundaries=((40.0, 4.0),),
    )
    offset = FuzzCase(
        "pop_rmbase",
        spec,
        freeze_params({"timerange": [], "pointrange": [5, 6, 7, 8, 40], "chanlist": []}),
    )
    prefix = FuzzCase(
        "pop_rmbase",
        spec,
        freeze_params({"timerange": [], "pointrange": [1, 2, 3, 4], "chanlist": []}),
    )
    eeg = materialize_eeg(offset)
    assert rmbase_boundary_offset(offset, eeg)
    assert not rmbase_boundary_offset(prefix, eeg)
    tolerances = load_tolerances()
    skipped = evaluate_case(offset, materialize_eeg, PythonBackend(), _ShiftedOracle(), tolerances)
    assert skipped.diverged is False
    assert "known divergence" in skipped.detail
    matched = evaluate_case(offset, materialize_eeg, PythonBackend(), PythonBackend(), tolerances)
    assert matched.diverged is False
    assert matched.error_class == "match"
    assert "known divergence" not in matched.detail
    filed = evaluate_case(prefix, materialize_eeg, PythonBackend(), _ShiftedOracle(), tolerances)
    assert filed.diverged is True
    assert filed.error_class == "numeric"


def test_openneuro_override_without_the_dataset_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("EEGPREP_OPENNEURO_ROOT", str(tmp_path))
    from tests.fuzz.targets import _openneuro_root

    with pytest.raises(FileNotFoundError):
        _openneuro_root()


def test_continuous_draws_include_every_loaded_recording():
    from tests.fuzz.targets import _continuous_recordings, _corpus

    loaded = {rec.name for rec in _corpus().values() if rec.trials == 1}
    assert {rec.name for rec in _continuous_recordings()} == loaded
    assert "eeglab_data" in loaded
    assert "eeglab_data_epochs_ica" not in loaded


def test_pipeline_crop_covers_thirty_seconds():
    from tests.fuzz.targets import recording

    case = next(iter(draw_cases("pipeline", seed_value=20261017, examples=1)))
    rec = recording(case.spec.source)
    assert case.spec.length == min(rec.pnts, int(round(30.0 * rec.srate)))


def test_pipeline_explicit_reref_nan_spread_is_a_match():
    spec = EEGSpec((0, 1, 2, 3), 0, 8, 1, 1, 2, 2)
    explicit = freeze_params({"ref": (0,), "keepref": "on", "exclude": []})
    case = FuzzCase(
        "pipeline",
        spec,
        freeze_params({"profile": "reref-nan"}),
        pipeline=(("pop_reref", explicit),),
    )
    chain = FuzzCase(
        "pipeline",
        spec,
        freeze_params({"profile": "reref-nan"}),
        pipeline=(
            ("pop_reref", explicit),
            ("pop_reref", freeze_params({"ref": [], "keepref": "off", "exclude": []})),
        ),
    )
    later_nans = []

    class _Matrix:
        def run_raw(self, step, eeg):
            if not dict(step.params).get("ref"):
                later_nans.append(int(np.isnan(np.asarray(eeg["data"])).sum()))
                return dispatch_python(step, eeg)
            data = np.asarray(eeg["data"], dtype=np.float64)
            refmatrix = np.eye(data.shape[0])
            refmatrix[:, 0] -= 1.0
            with np.errstate(invalid="ignore"):
                spread = refmatrix @ data
            out = eeg.copy()
            out["data"] = spread
            return out

    outcome = evaluate_case(chain, materialize_eeg, PythonBackend(), _Matrix(), load_tolerances())
    assert outcome.diverged is False
    assert outcome.error_class == "match"
    assert later_nans == [2]

    class _Shifted:
        def run_raw(self, step, eeg):
            out = dispatch_python(step, eeg)
            out = out.copy()
            out["data"] = np.array(np.asarray(out["data"]), dtype=np.float64, copy=True)
            out["data"][2, 0] += 1.0
            return out

    shifted = evaluate_case(case, materialize_eeg, PythonBackend(), _Shifted(), load_tolerances())
    assert shifted.diverged is True
    assert shifted.error_class == "pop_reref:numeric"


def test_zero_trials_event_drop_is_not_filed():
    eeg = {
        "data": np.arange(20, dtype=float).reshape(1, 20),
        "nbchan": 1,
        "pnts": 20,
        "trials": 0,
        "srate": 100.0,
        "xmin": 0.0,
        "xmax": 0.19,
        "event": [{"type": "stim", "latency": 5.0}],
        "chanlocs": [{"labels": "Cz"}],
    }
    case = FuzzCase(
        "eeg_eegrej",
        EEGSpec((0,), 0, 20, 1, -1, 0, 0),
        freeze_params({"regions": [[1, 2]], "eegplot": False}),
    )

    class _Dropped:
        def run(self, step, dataset):
            out = dispatch_python(step, dataset)
            out["event"] = [
                event
                for event in out["event"]
                if str(event.get("type")).lower() == "boundary" or float(event.get("latency") or 0) <= 1
            ]
            return side_from_call(out)

    outcome = evaluate_case(case, lambda _case: eeg, PythonBackend(), _Dropped(), load_tolerances())
    assert outcome.diverged is False
    assert "trials was 0" in outcome.detail

    class _Shifted:
        def run(self, step, dataset):
            out = dispatch_python(step, dataset)
            out["data"] = np.array(np.asarray(out["data"]), copy=True)
            out["data"][0, 0] += 1.0
            return side_from_call(out)

    shifted = evaluate_case(case, lambda _case: eeg, PythonBackend(), _Shifted(), load_tolerances())
    assert shifted.diverged is True
    assert shifted.error_class == "numeric"

    class _Retyped(_Dropped):
        def run(self, step, dataset):
            out = dispatch_python(step, dataset)
            out["event"] = [
                {**event, "type": "other"} if str(event.get("type")).lower() == "boundary" else event
                for event in out["event"]
                if str(event.get("type")).lower() == "boundary" or float(event.get("latency") or 0) <= 1
            ]
            return side_from_call(out)

    retyped = evaluate_case(case, lambda _case: eeg, PythonBackend(), _Retyped(), load_tolerances())
    assert retyped.diverged is True

    missing = dict(eeg)
    del missing["trials"]
    omitted = evaluate_case(case, lambda _case: missing, PythonBackend(), _Dropped(), load_tolerances())
    assert omitted.diverged is True


def test_pipeline_stale_trials_continues_from_python():
    eeg = {
        "data": np.arange(20, dtype=float).reshape(1, 20),
        "nbchan": 1,
        "pnts": 20,
        "trials": 0,
        "srate": 100.0,
        "xmin": 0.0,
        "xmax": 0.19,
        "event": [{"type": "stim", "latency": 5.0}],
        "chanlocs": [{"labels": "Cz"}],
    }
    seen = []
    case = FuzzCase(
        "pipeline",
        EEGSpec((0,), 0, 20, 1, -1, 0, 0),
        freeze_params({"profile": "stale-trials"}),
        pipeline=(
            ("eeg_eegrej", freeze_params({"regions": [[1, 2]], "eegplot": False})),
            ("pop_reref", freeze_params({"ref": [], "keepref": "off", "exclude": []})),
        ),
    )

    class _Oracle:
        def run_raw(self, step, dataset):
            if step.function != "eeg_eegrej":
                seen.append([float(event["latency"]) for event in dataset["event"]])
                return dispatch_python(step, dataset)
            out = dispatch_python(step, dataset)
            out["event"] = [event for event in out["event"] if float(event.get("latency") or 0) <= 1]
            return out

    outcome = evaluate_case(case, lambda _case: eeg, PythonBackend(), _Oracle(), load_tolerances())
    assert outcome.diverged is False
    assert seen == [[0.5, 3.0]]


def test_eegrej_edge_draw_can_store_trials_zero():
    found = False
    for case in draw_cases("eeg_eegrej", seed_value=20261023, examples=40):
        if case.params and dict(case.params).get("stale_trials") is not True:
            continue
        eeg = materialize_eeg(case)
        assert eeg["trials"] == 0
        found = True
        break
    assert found


def test_pipeline_continues_after_the_known_rmbase_divergence():
    spec = EEGSpec(
        tuple(range(32)),
        0,
        256,
        1,
        -1,
        0,
        0,
        events="keep+boundary",
        boundaries=((40.0, 4.0),),
    )
    case = FuzzCase(
        "pipeline",
        spec,
        freeze_params({"profile": "edge"}),
        pipeline=(
            ("pop_rmbase", freeze_params({"timerange": [], "pointrange": [5, 6, 7, 8, 40], "chanlist": []})),
            ("pop_reref", freeze_params({"ref": [], "keepref": "off", "exclude": []})),
        ),
    )

    class _LaterDrift:
        def run_raw(self, step, eeg):
            out = dispatch_python(step, eeg)
            data = np.array(np.asarray(out["data"]), dtype=np.float64, copy=True)
            data += 10.0 if step.function == "pop_reref" else 1.0
            copied = out.copy()
            copied["data"] = data
            return copied

    outcome = evaluate_case(case, materialize_eeg, PythonBackend(), _LaterDrift(), load_tolerances())
    assert outcome.diverged is True
    assert outcome.error_class == "pop_reref:numeric"


def test_pipeline_resyncs_to_the_oracle_output():
    case = next(item for item in extra_sanity(EEGSpec, FuzzCase) if item.function == "pipeline")

    class _Drift:
        def run_raw(self, step, eeg):
            raw = dispatch_python(step, eeg)
            if step.function != "pop_eegfiltnew":
                return raw
            out = raw.copy()
            data = np.array(np.asarray(out["data"]), dtype=np.float64, copy=True)
            data[0] += 1e-8
            out["data"] = data
            return out

    outcome = evaluate_case(case, materialize_eeg, PythonBackend(), _Drift(), load_tolerances())
    assert outcome.error_class == "match", outcome.detail


def test_span_artifact_hits_the_flatline_boundary():
    from tests.fuzz.targets import _jitter_threshold, _span_sample_count, _write_artifact

    srate = 128.0
    duration = 0.25
    limit = duration * srate
    jitter = 20.0
    threshold = _jitter_threshold(jitter)
    assert _span_sample_count(limit, "below") == 32
    assert _span_sample_count(limit, "equal") == 33
    assert _span_sample_count(limit, "above") == 34
    for place, removed in (("below", False), ("equal", False), ("above", True)):
        count = _span_sample_count(limit, place)
        samples = np.linspace(-30.0, 30.0, count + 5)
        _write_artifact(samples, f"span-{place}-jitter-below-near0-{int(jitter)}", 0, count, srate)
        diffs = np.abs(np.diff(samples[:count]))
        assert np.all(diffs < threshold)
        assert np.any(diffs > 0)
        assert abs(samples[count] - samples[count - 1]) > threshold
        flat = np.pad(np.abs(np.diff(samples)) < threshold, 1)
        edges = np.where(np.diff(flat) > 0)[0]
        intervals = np.reshape(edges, (-1, 2))
        longest = int(np.max(intervals[:, 1] - intervals[:, 0]))
        assert (longest > limit) is removed
    for relation, check in (
        ("below", lambda diffs: np.all(diffs < threshold) and np.any(diffs > 0)),
        ("equal", lambda diffs: np.all(diffs == threshold)),
        ("above", lambda diffs: np.all(diffs > threshold)),
    ):
        samples = np.zeros(16, dtype=np.float64)
        _write_artifact(samples, f"span-equal-jitter-{relation}-near0-{int(jitter)}", 0, 8, srate)
        assert check(np.abs(np.diff(samples[:8])))

    recorded = np.full(40, np.float64(-26.488021850585973))
    _write_artifact(recorded, "span-equal-jitter-equal-recorded-20", 0, 16, srate)
    assert np.all(np.abs(np.diff(recorded[:16])) == threshold)


def test_jitter_is_not_a_constant_and_channel_drops_clear_ica():
    from tests.fuzz.targets import _write_artifact

    samples = np.linspace(-200.0, 200.0, 16)
    _write_artifact(samples, "jitter20", 0, 16, 256.0)
    assert len(set(np.round(samples, 12))) > 1
    kept = materialize_eeg(
        FuzzCase(
            "clean_channels",
            EEGSpec(tuple(range(32)), 0, 256, 1, -1, 0, 0, source="eeglab_data_with_ica_tmp", events="keep"),
            freeze_params({}),
        )
    )
    dropped = materialize_eeg(
        FuzzCase(
            "clean_channels",
            EEGSpec(tuple(range(16)), 0, 256, 1, -1, 0, 0, source="eeglab_data_with_ica_tmp", events="keep"),
            freeze_params({}),
        )
    )
    assert np.asarray(kept["icaweights"]).size > 0
    assert np.asarray(dropped["icaweights"]).size == 0
    assert np.asarray(dropped["icachansind"]).size == 0


def _windows_case(length: int) -> FuzzCase:
    return FuzzCase(
        "clean_windows",
        EEGSpec((0,), 0, length, 1, -1, 0, 0, srate_hz=128.0),
        freeze_params(
            {
                "max_bad_channels": 0.2,
                "zthresholds": [-3.5, 5.0],
                "window_len": 1.0,
                "window_overlap": 0.66,
            }
        ),
    )


def _recording(length: int) -> dict:
    return {
        "data": np.zeros((1, length)),
        "nbchan": 1,
        "pnts": length,
        "trials": 1,
        "srate": 128.0,
        "xmin": 0.0,
        "xmax": (length - 1) / 128.0,
        "event": [],
        "chanlocs": [{"labels": "Cz"}],
    }


class _RaisingBackend:
    def __init__(self, exc: Exception):
        self.exc = exc

    def run(self, case, eeg):
        return SideResult(False, None, None, type(self.exc).__name__, str(self.exc))


def test_short_clean_windows_errors_are_a_match():
    tolerances = load_tolerances()
    short = evaluate_case(
        _windows_case(160),
        lambda case: _recording(case.spec.length),
        _RaisingBackend(ValueError("Too few windows")),
        _RaisingBackend(RuntimeError("Index in position 1 is invalid")),
        tolerances,
    )
    assert short.error_class == "match"
    assert short.both_ok
    assert not short.diverged

    long = evaluate_case(
        _windows_case(2048),
        lambda case: _recording(case.spec.length),
        _RaisingBackend(ValueError("Too few windows")),
        _RaisingBackend(RuntimeError("Index in position 1 is invalid")),
        tolerances,
    )
    assert long.diverged
    assert long.error_class.startswith("exception-mismatch")


def test_pipeline_short_clean_windows_errors_are_a_match(monkeypatch):
    def boom(step, eeg):
        raise ValueError("Too few windows")

    monkeypatch.setattr("tests.fuzz.harness.dispatch_python", boom)
    windows = _windows_case(160).params
    case = FuzzCase(
        "pipeline",
        EEGSpec((0,), 0, 160, 1, -1, 0, 0, srate_hz=128.0),
        freeze_params({"profile": "short-windows"}),
        pipeline=(("clean_windows", windows),),
    )
    outcome = evaluate_case(
        case,
        lambda _case: _recording(160),
        PythonBackend(),
        FakeOracle("match", PythonBackend()),
        load_tolerances(),
    )
    assert outcome.error_class == "match", outcome.detail
    assert not outcome.diverged


def test_matlab_dispatch_replaces_none_chaninfo_before_save():
    captured = {}

    class Engine:
        def pop_resample(self, eeg, freq):
            captured["eeg"] = eeg
            return eeg

    eeg = _recording(8)
    eeg["times"] = np.arange(8) / 128.0
    eeg["icachansind"] = np.array([])
    eeg["chaninfo"] = {"plotrad": None, "shrink": None, "nosedir": "+X"}
    case = FuzzCase("pop_resample", EEGSpec((0,), 0, 8, 1, -1, 0, 0), freeze_params({"freq": 64.0}))
    dispatch_matlab(Engine(), case, eeg)

    assert eeg["chaninfo"]["plotrad"] is None
    saved = captured["eeg"]["chaninfo"]
    assert saved["plotrad"].size == 0
    assert saved["shrink"].size == 0
    assert saved["nosedir"] == "+X"
    with tempfile.TemporaryDirectory() as tmp:
        pop_saveset(captured["eeg"], os.path.join(tmp, "chaninfo.set"))
