"""Tests for the portable ASR state: ``clean_asr(return_state=True)`` and ``clean_asr(state=...)``."""

import json
from pathlib import Path

import numpy as np
import pytest

from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.plugins.clean_rawdata.asr_calibrate import asr_calibrate
from eegprep.plugins.clean_rawdata.asr_state import STATE_FORMAT, state_from_json, state_to_json
from eegprep.plugins.clean_rawdata.clean_asr import clean_asr

SAMPLE = Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data.set"
MODES = [None, "calib", "all"]
MODE_IDS = ["euclidean", "calib", "riemannian"]


def reload(state):
    """Write the state as JSON text and read it back, as a stored record would be."""
    return state_from_json(json.loads(json.dumps(state_to_json(state))))


@pytest.fixture(scope="module")
def eeg():
    source = pop_loadset(SAMPLE)
    data = np.asarray(source["data"][:, : 40 * 128], dtype=np.float64)
    return {
        "data": data,
        "srate": float(source["srate"]),
        "nbchan": data.shape[0],
        "pnts": data.shape[1],
        "trials": 1,
        "xmin": 0.0,
        "xmax": (data.shape[1] - 1) / float(source["srate"]),
        "event": [],
        "chanlocs": source["chanlocs"],
    }


@pytest.fixture(scope="module", params=MODES, ids=MODE_IDS)
def run(request, eeg):
    cleaned, state = clean_asr(eeg, return_state=True, useriemannian=request.param)
    return request.param, cleaned, state


@pytest.fixture(scope="module")
def state(eeg):
    return clean_asr(eeg, return_state=True)[1]


def test_reapplied_state_reproduces_the_cleaned_data_exactly(eeg, run):
    _, cleaned, saved = run
    assert not np.array_equal(cleaned["data"], eeg["data"])
    again = clean_asr(eeg, state=reload(saved))
    assert np.array_equal(again["data"], cleaned["data"])


def test_a_state_can_be_reused_and_is_not_changed_by_use(eeg, run):
    _, cleaned, saved = run
    before = json.dumps(state_to_json(saved))
    assert np.array_equal(clean_asr(eeg, state=saved)["data"], cleaned["data"])
    assert np.array_equal(clean_asr(eeg, state=saved)["data"], cleaned["data"])
    assert json.dumps(state_to_json(saved)) == before


def test_asking_for_the_state_does_not_change_the_data(eeg, run):
    mode, cleaned, _ = run
    assert np.array_equal(clean_asr(eeg, useriemannian=mode)["data"], cleaned["data"])


def test_state_survives_json_exactly(run):
    _, _, saved = run
    back = reload(saved)
    assert back.keys() == saved.keys()
    for key, value in saved.items():
        if isinstance(value, np.ndarray):
            assert back[key].dtype == np.float64
            assert np.array_equal(back[key], value), key
        else:
            assert back[key] == value, key


def test_returned_state_is_the_calibration_before_processing(eeg):
    _, saved = clean_asr(eeg, ref_maxbadchannels="off", return_state=True)
    calibration = asr_calibrate(eeg["data"], eeg["srate"], cutoff=5.0, maxmem=64)
    for key, value in calibration.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(saved[key], value), key
        else:
            assert saved[key] == value, key
    assert saved["format"] == STATE_FORMAT
    assert saved["srate"] == eeg["srate"]
    assert saved["channels"] == [chan["labels"] for chan in eeg["chanlocs"]]


def test_processing_arguments_travel_with_the_state(eeg):
    options = dict(cutoff=3.0, window_len=1.0, step_size=40, max_dims=0.5, maxmem=32, ref_maxbadchannels="off")
    cleaned, saved = clean_asr(eeg, return_state=True, **options)
    recorded = {key: saved[key] for key in ("cutoff", "window_len", "step_size", "max_dims", "maxmem")}
    assert recorded == {key: options[key] for key in recorded}
    assert np.array_equal(clean_asr(eeg, state=reload(saved))["data"], cleaned["data"])
    assert not np.array_equal(clean_asr(eeg, ref_maxbadchannels="off")["data"], cleaned["data"])


@pytest.mark.parametrize(("key", "value"), [("max_dims", 0.0), ("step_size", 8), ("window_len", 1.0)])
def test_reapplying_uses_the_processing_arguments_recorded_in_the_state(eeg, state, key, value):
    changed = clean_asr(eeg, state=dict(state, **{key: value}))
    assert not np.array_equal(changed["data"], clean_asr(eeg, state=state)["data"])


def test_reapplying_riemannian_uses_the_recorded_memory_budget(eeg):
    _, saved = clean_asr(eeg, return_state=True, useriemannian="all")
    split = clean_asr(eeg, state=dict(saved, maxmem=8))
    assert not np.array_equal(split["data"], clean_asr(eeg, state=saved)["data"])


def test_default_call_returns_a_single_eeg(eeg):
    out = clean_asr(eeg, ref_maxbadchannels="off")
    assert isinstance(out, dict)
    assert out["data"].shape == eeg["data"].shape
    pair = clean_asr(eeg, ref_maxbadchannels="off", return_state=True)
    assert isinstance(pair, tuple)
    assert len(pair) == 2


def test_state_for_another_channel_order_is_refused(eeg, state):
    order = np.arange(eeg["nbchan"])[::-1]
    reordered = dict(eeg, data=eeg["data"][order], chanlocs=eeg["chanlocs"][order])
    with pytest.raises(ValueError, match="channels"):
        clean_asr(reordered, state=state)


def test_state_for_another_sampling_rate_is_refused(eeg, state):
    with pytest.raises(ValueError, match="srate"):
        clean_asr(dict(eeg, srate=256.0), state=state)


def test_state_without_labels_still_checks_the_channel_count():
    rng = np.random.default_rng(7)

    def unlabeled(count):
        return {"data": rng.standard_normal((count, 1000)), "srate": 250.0, "nbchan": count}

    _, saved = clean_asr(unlabeled(4), ref_maxbadchannels="off", return_state=True)
    assert saved["channels"] is None
    with pytest.raises(ValueError, match="channels"):
        clean_asr(unlabeled(5), state=saved)


def test_a_calibration_dictionary_is_not_a_state(eeg):
    calibration = asr_calibrate(eeg["data"], eeg["srate"])
    with pytest.raises(ValueError, match="format"):
        clean_asr(eeg, state=calibration)


@pytest.mark.parametrize(
    "option",
    [
        {"max_dims": 0.5},
        {"cutoff": 3.0},
        {"window_len": 1.0},
        {"useriemannian": "calib"},
        {"ref_maxbadchannels": "off"},
    ],
    ids=lambda option: next(iter(option)),
)
def test_arguments_the_state_owns_are_refused(eeg, state, option):
    with pytest.raises(ValueError, match=next(iter(option))):
        clean_asr(eeg, state=state, **option)


def test_changed_state_element_changes_the_data(eeg, run):
    _, cleaned, saved = run
    document = json.loads(json.dumps(state_to_json(saved)))
    document["M"]["data"][0][0] *= 1.5
    changed = clean_asr(eeg, state=state_from_json(document))
    assert not np.array_equal(changed["data"], cleaned["data"])


def _format(document):
    document["format"] = "eegprep.asr-state.v2"


def _missing(document):
    del document["T"]


def _shape(document):
    document["M"]["shape"] = [1, 1]


def _ragged(document):
    document["M"]["data"][0] = [1.0]


def _not_finite(document):
    document["T"]["data"][0][0] = float("nan")


def _dtype(document):
    document["B"]["dtype"] = "float32"


def _unknown(document):
    document["extra"] = 1


def _carried_buffer(document):
    document["carry"] = document["M"]


@pytest.mark.parametrize(
    ("edit", "key"),
    [
        (_format, "format"),
        (_missing, "T"),
        (_shape, "M"),
        (_ragged, "M"),
        (_not_finite, "T"),
        (_dtype, "B"),
        (_unknown, "extra"),
        (_carried_buffer, "carry"),
    ],
    ids=lambda value: value.__name__.strip("_") if callable(value) else value,
)
def test_malformed_documents_are_refused_naming_the_key(state, edit, key):
    document = json.loads(json.dumps(state_to_json(state)))
    edit(document)
    with pytest.raises(ValueError, match=rf"^{key}: "):
        state_from_json(json.loads(json.dumps(document)))


def test_state_to_json_refuses_non_finite_numbers(state):
    broken = dict(state, T=np.full_like(state["T"], np.nan))
    with pytest.raises(ValueError, match="T"):
        state_to_json(broken)
