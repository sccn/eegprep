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


class _ShiftedOracle:
    def run(self, case, eeg):
        side = PythonBackend().run(case, eeg)
        if side.data is None:
            return side
        return replace(side, data=side.data + 1.0)


_DRAW_SCRIPT = """
import json
from pathlib import Path
from tests.fuzz.harness import draw_cases, spec_to_json, thaw_params
golden = json.loads(Path("tests/fuzz/golden_draws_20261012.json").read_text())
names = ("pop_eegfiltnew", "pop_reref", "pop_resample", "clean_asr", "pop_clean_rawdata")
for name in names:
    drawn = []
    for case in draw_cases(name, seed_value=20261012, examples=4):
        drawn.append({
            "function": case.function,
            "params": thaw_params(case.params),
            "sanity": case.sanity,
            "spec": spec_to_json(case.spec),
        })
    assert drawn == golden[name], name
"""


def test_original_draws_match_the_golden_snapshot():
    completed = subprocess.run(
        [sys.executable, "-c", _DRAW_SCRIPT],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_original_draws_do_not_depend_on_import_order():
    completed = subprocess.run(
        [sys.executable, "-c", "import tests.fuzz.targets\n" + _DRAW_SCRIPT],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


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
    case = FuzzCase(
        "pipeline",
        spec,
        freeze_params({"profile": "reref-nan"}),
        pipeline=(("pop_reref", freeze_params({"ref": (0,), "keepref": "on", "exclude": []})),),
    )

    class _Matrix:
        def run_raw(self, step, eeg):
            data = np.asarray(eeg["data"], dtype=np.float64)
            refmatrix = np.eye(data.shape[0])
            refmatrix[:, 0] -= 1.0
            with np.errstate(invalid="ignore"):
                spread = refmatrix @ data
            out = eeg.copy()
            out["data"] = spread
            return out

    outcome = evaluate_case(case, materialize_eeg, PythonBackend(), _Matrix(), load_tolerances())
    assert outcome.diverged is False
    assert outcome.error_class == "match"

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
