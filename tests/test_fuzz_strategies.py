"""Nightly strategy coverage that does not live in the loadset replay file."""

from __future__ import annotations

import numpy as np
import pytest

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
            "clean_asr": lambda draw: _asr_case(draw.integers(1, 8), "clean_asr"),
            "pop_clean_rawdata": lambda draw: _asr_case(draw.integers(1, 8), "clean_rawdata"),
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


def test_draw_integers_are_fixed_for_a_seed():
    from tests.fuzz.draw import Draw

    draw = Draw(20261012)
    assert [draw.integers(0, 1000) for _ in range(8)] == [82, 868, 434, 528, 35, 317, 33, 559]


def test_flat_channel_crop_is_constant():
    from tests.fuzz.harness import materialize_eeg

    spec = EEGSpec((0, 1, 2, 3), 0, 32, 1, -1, 0, 0, flat_channel=1)
    case = FuzzCase("pop_reref", spec, freeze_params({"ref": (), "keepref": "off", "exclude": ()}))
    data = np.asarray(materialize_eeg(case)["data"])
    assert np.all(data[1] == 0.0)
    assert not np.all(data[0] == 0.0)


def test_original_strategies_cover_real_calls_and_edges():
    filters = draw_cases("pop_eegfiltnew", seed_value=20261012, examples=40)
    filter_params = [thaw_params(case.params) for case in filters]
    assert sum(params["filtorder"] is None for params in filter_params) >= 20
    assert any(params["locutoff"] in {0.5, 1.0, 2.0} and params["hicutoff"] is None for params in filter_params)
    assert any(params["hicutoff"] in {30.0, 40.0, 45.0} and params["revfilt"] is False for params in filter_params)
    assert any(len(case.spec.channels) >= 32 for case in filters)
    assert any(case.spec.flat_channel >= 0 or case.spec.length < 256 for case in filters)
    assert any(params.get("filtorder") == 101 or params.get("hicutoff") == 64.0 for params in filter_params)

    rerefs = [thaw_params(case.params) for case in draw_cases("pop_reref", seed_value=20261012, examples=40)]
    assert any(not params["ref"] and not params["exclude"] and "refloc" not in params for params in rerefs)
    assert any(params["ref"] or params.get("refloc") or params["keepref"] == "on" for params in rerefs)
    assert any(params["exclude"] for params in rerefs)

    resamples = draw_cases("pop_resample", seed_value=20261012, examples=40)
    frequencies = {thaw_params(case.params)["freq"] for case in resamples}
    assert frequencies & {64.0, 100.0, 128.0, 250.0, 256.0}
    assert any(case.spec.source == "eeglab_data_epochs_ica" for case in resamples)
    assert any("fc" in thaw_params(case.params) for case in resamples)

    raw_params = [
        thaw_params(case.params) for case in draw_cases("pop_clean_rawdata", seed_value=20261012, examples=40)
    ]
    assert all(params["entry"] == "clean_rawdata" for params in raw_params)
    assert any(params.get("FlatlineCriterion") == 5.0 and params.get("BurstCriterion") == 20.0 for params in raw_params)
    assert any(params.get("Highpass") == [0.25, 0.75] and params.get("BurstCriterion") == 5.0 for params in raw_params)
    assert any("cutoff" in params for params in raw_params)
