"""Nightly strategy coverage that does not live in the loadset replay file."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import strategies as st

from tests.fuzz.harness import (
    EEGSpec,
    FuzzCase,
    SideResult,
    draw_cases,
    freeze_params,
    run_session,
    side_from_error,
    thaw_params,
)


class _Backend:
    def __init__(self, fn):
        self.fn = fn

    def run(self, case, eeg):
        try:
            data = np.asarray(self.fn(case, eeg), dtype=float)
            return SideResult(True, data, float(eeg["srate"]), None, None)
        except Exception as exc:
            return side_from_error(exc)


def _ones(case: FuzzCase) -> dict:
    return {"data": np.ones((len(case.spec.channels), 8), dtype=float), "srate": 128.0}


def _asr_case(n_channels: int, entry: str) -> FuzzCase:
    return FuzzCase(
        "clean_asr",
        EEGSpec(tuple(range(n_channels)), 0, 8, 1, -1, 0, 0),
        freeze_params({"entry": entry, "cutoff": 5.0, "ref_maxbadchannels": "off", "window_len": None}),
    )


@pytest.mark.parametrize("seed_value", list(range(20261014, 20261021)))
def test_each_night_draws_pop_clean_rawdata(seed_value):
    raw = draw_cases("pop_clean_rawdata", seed_value=seed_value, examples=4)
    asr = draw_cases("clean_asr", seed_value=seed_value, examples=4)
    assert len(raw) == 4
    assert len(asr) == 4
    assert {thaw_params(case.params)["entry"] for case in raw} == {"clean_rawdata"}
    assert {thaw_params(case.params)["entry"] for case in asr} == {"clean_asr"}


def test_rawdata_divergence_shrinks_on_its_own_strategy(tmp_path):
    def python(case, eeg):
        del eeg
        value = 1.0 if thaw_params(case.params)["entry"] == "clean_rawdata" and len(case.spec.channels) >= 3 else 0.0
        return np.full((len(case.spec.channels), 8), value)

    def oracle(case, eeg):
        del eeg
        return np.zeros((len(case.spec.channels), 8))

    result = run_session(
        seed_value=1,
        output_dir=tmp_path,
        examples=8,
        shrink_budget=40,
        max_shrinks=2,
        python_backend=_Backend(python),
        oracle_backend=_Backend(oracle),
        strategies={
            "clean_asr": st.integers(1, 8).map(lambda n: _asr_case(n, "clean_asr")),
            "pop_clean_rawdata": st.integers(1, 8).map(lambda n: _asr_case(n, "clean_rawdata")),
        },
        sanity=[_asr_case(1, "clean_asr")],
        materialize=_ones,
    )
    assert result.harness_ok
    assert result.functions["pop_clean_rawdata"]["examples"] == 8
    assert result.functions["clean_asr"]["divergences"] == 0
    assert result.divergences
    confirmed = result.divergences[0]
    assert confirmed.report == "pop_clean_rawdata"
    assert thaw_params(confirmed.case.params)["entry"] == "clean_rawdata"
    assert len(confirmed.case.spec.channels) == 3
