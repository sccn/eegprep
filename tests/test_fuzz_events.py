"""Event-aware fuzz targets. MATLAB is not required."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from tests.fuzz.harness import (
    EEGSpec,
    FakeOracle,
    FuzzCase,
    Outcome,
    PythonBackend,
    SideResult,
    case_from_payload,
    data_sha256,
    draw_cases,
    evaluate_case,
    freeze_params,
    judge,
    load_tolerances,
    materialize_eeg,
    spec_to_json,
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


def _dump(case: FuzzCase) -> dict:
    return {
        "function": case.function,
        "params": thaw_params(case.params),
        "sanity": case.sanity,
        "spec": spec_to_json(case.spec),
    }


class _ShiftedOracle:
    def run(self, case, eeg):
        side = PythonBackend().run(case, eeg)
        if side.data is None:
            return side
        return replace(side, data=side.data + 1.0)


def test_original_draws_match_the_golden_snapshot():
    golden = json.loads(_GOLDEN.read_text())
    for name in ("pop_eegfiltnew", "pop_reref", "pop_resample", "clean_asr", "pop_clean_rawdata"):
        drawn = [_dump(case) for case in draw_cases(name, seed_value=20261012, examples=4)]
        assert drawn == golden[name]


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
    filed = evaluate_case(prefix, materialize_eeg, PythonBackend(), _ShiftedOracle(), tolerances)
    assert filed.diverged is True
    assert filed.error_class == "numeric"
