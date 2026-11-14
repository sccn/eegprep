"""Tests for the portable ASR state: ``clean_asr(return_state=True)`` and ``clean_asr(state=...)``."""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.plugins.clean_rawdata.asr_calibrate import asr_calibrate
from eegprep.plugins.clean_rawdata.asr_state import STATE_FORMAT, state_from_json, state_to_json
from eegprep.plugins.clean_rawdata.asr_state import check_state
from eegprep.plugins.clean_rawdata.clean_asr import _warn_if_maxmem_cannot_reproduce, clean_asr

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


@pytest.mark.parametrize(("key", "value"), [("max_dims", 0.0), ("step_size", 8), ("window_len", 1.0), ("maxmem", 8)])
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
        {"step_size": 40},
        {"maxmem": 32},
        {"ref_tolerances": (-3.0, 5.0)},
        {"ref_wndlen": 2.0},
    ],
    ids=lambda option: next(iter(option)),
)
def test_arguments_the_state_owns_are_refused(eeg, state, option):
    with pytest.raises(ValueError, match=next(iter(option))):
        clean_asr(eeg, state=state, **option)


def test_arguments_equal_to_the_state_or_to_the_default_are_accepted(eeg):
    options = dict(cutoff=3.0, step_size=40, ref_maxbadchannels="off")
    cleaned, saved = clean_asr(eeg, return_state=True, **options)
    same = dict(cutoff=3.0, step_size=40, window_len=0.5, max_dims=0.66, maxmem=64, useriemannian=None)
    assert np.array_equal(clean_asr(eeg, state=saved, **same)["data"], cleaned["data"])
    assert np.array_equal(clean_asr(eeg, state=saved, cutoff=5.0, step_size=None)["data"], cleaned["data"])
    with pytest.raises(ValueError, match="cutoff"):
        clean_asr(eeg, state=saved, cutoff=4.0)


@pytest.mark.parametrize("spelling", [False, "off", "none"])
def test_every_spelling_of_euclidean_matches_a_euclidean_state(eeg, state, spelling):
    assert np.array_equal(
        clean_asr(eeg, state=state, useriemannian=spelling)["data"], clean_asr(eeg, state=state)["data"]
    )


@pytest.mark.parametrize("spelling", [True, "on", "all"])
def test_every_spelling_of_riemannian_matches_a_riemannian_state(eeg, spelling):
    cleaned, saved = clean_asr(eeg, return_state=True, useriemannian="all")
    assert np.array_equal(clean_asr(eeg, state=saved, useriemannian=spelling)["data"], cleaned["data"])
    with pytest.raises(ValueError, match="useriemannian"):
        clean_asr(eeg, state=saved, useriemannian="calib")


def test_a_list_of_tolerances_matches_the_tuple_default(eeg, state):
    assert np.array_equal(
        clean_asr(eeg, state=state, ref_tolerances=[-3.5, 5.5])["data"], clean_asr(eeg, state=state)["data"]
    )
    with pytest.raises(ValueError, match="ref_tolerances"):
        clean_asr(eeg, state=state, ref_tolerances=[-3.0, 5.0])


def test_a_budget_below_the_processing_overhead_warns_when_the_state_is_asked_for_or_applied(eeg, state):
    # 32 channels and 32 lookahead samples reserve 32 * 32 * 32 * 24 bytes, 0.75 MB: 1 MB is the smallest budget.
    with pytest.warns(UserWarning, match=r"maxmem=None .*Use maxmem=1 or more"):
        clean_asr(eeg, maxmem=None, return_state=True)
    with pytest.warns(UserWarning, match=r"maxmem=0 .*Use maxmem=1 or more"):
        clean_asr(eeg, state=dict(state, maxmem=0))
    with pytest.warns(UserWarning, match=r"maxmem=None .*Use maxmem=1 or more"):
        clean_asr(eeg, state=dict(state, maxmem=None))


def test_a_sufficient_budget_and_a_plain_call_do_not_warn(eeg, state):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        clean_asr(eeg, return_state=True)
        clean_asr(eeg, state=state)
        clean_asr(eeg, maxmem=None)


def test_the_overhead_threshold_matches_what_asr_process_reserves():
    # 16 channels and 512 lookahead samples reserve exactly 3 MiB, the budget asr_process still accepts.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _warn_if_maxmem_cannot_reproduce(3, 16, 512)
    with pytest.warns(UserWarning, match="Use maxmem=3 or more"):
        _warn_if_maxmem_cannot_reproduce(2, 16, 512)


def test_a_whole_number_step_size_is_recorded_as_an_integer(eeg):
    cleaned, saved = clean_asr(eeg, ref_maxbadchannels="off", step_size=40.0, return_state=True)
    assert type(saved["step_size"]) is int
    assert np.array_equal(clean_asr(eeg, state=reload(saved))["data"], cleaned["data"])
    _, saved = clean_asr(eeg, ref_maxbadchannels="off", step_size=np.int64(40), return_state=True)
    assert type(saved["step_size"]) is int


def test_a_fractional_step_size_cannot_be_recorded_but_still_runs_unrecorded(eeg):
    with pytest.raises(ValueError, match="step_size"):
        clean_asr(eeg, ref_maxbadchannels="off", step_size=32.5, return_state=True)
    assert clean_asr(eeg, ref_maxbadchannels="off", step_size=32.5)["data"].shape == eeg["data"].shape


@pytest.mark.parametrize(
    "make",
    [
        lambda state: [1],
        lambda state: dict(state, useriemannian="all"),
        lambda state: {"format": STATE_FORMAT},
        lambda state: dict(state, T=state["T"].astype(np.float32)),
    ],
    ids=["not_a_dict", "euclidean_keys_with_all", "only_a_format_key", "float32_array"],
)
def test_a_state_in_memory_is_checked_like_a_document(eeg, state, make):
    with pytest.raises(ValueError):
        clean_asr(eeg, state=make(state))
    with pytest.raises(ValueError):
        check_state(make(state))


def test_labeled_and_unlabeled_states_never_mix(eeg, state):
    unlabeled = {key: value for key, value in eeg.items() if key != "chanlocs"}
    with pytest.raises(ValueError, match="channels"):
        clean_asr(unlabeled, state=state)
    _, bare = clean_asr(unlabeled, ref_maxbadchannels="off", return_state=True)
    with pytest.raises(ValueError, match="channels"):
        clean_asr(eeg, state=bare)


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


def _document(state):
    return json.loads(json.dumps(state_to_json(state)))


def _set(key, value):
    def edit(document):
        document[key] = value

    return edit


def _drop(key):
    def edit(document):
        del document[key]

    return edit


def _set_shape(document):
    document["M"]["shape"] = "32x32"


def _set_shape_number(document):
    document["M"]["shape"] = 2


def _negative_shape(document):
    document["M"]["shape"] = [-1]


def _boolean_shape(document):
    document["M"]["shape"] = [True, True]


def _array_without_data(document):
    document["M"] = {"dtype": "float64", "shape": [1, 1]}


DECODE_REFUSALS = {
    "not_an_object": (lambda document: [], "document"),
    "no_useriemannian": (_drop("useriemannian"), "useriemannian"),
    "unknown_useriemannian": (_set("useriemannian", "bogus"), "useriemannian"),
    "array_is_a_list": (_set("M", [1.0]), "M"),
    "array_without_data": (_array_without_data, "M"),
    "shape_is_text": (_set_shape, "M"),
    "shape_is_a_number": (_set_shape_number, "M"),
    "shape_is_negative": (_negative_shape, "M"),
    "shape_holds_booleans": (_boolean_shape, "M"),
    "last_trivial_is_text": (_set("last_trivial", "yes"), "last_trivial"),
    "unknown_compatibility": (_set("compatibility", "loose"), "compatibility"),
    "channels_is_text": (_set("channels", "Cz"), "channels"),
    "channels_hold_numbers": (_set("channels", [1, 2]), "channels"),
    "zero_step_size": (_set("step_size", 0), "step_size"),
    "fractional_step_size": (_set("step_size", 32.5), "step_size"),
    "boolean_step_size": (_set("step_size", True), "step_size"),
    "text_cutoff": (_set("cutoff", "5"), "cutoff"),
    "boolean_cutoff": (_set("cutoff", True), "cutoff"),
    "nan_cutoff": (_set("cutoff", float("nan")), "cutoff"),
    "infinite_max_dims": (_set("max_dims", float("inf")), "max_dims"),
    "zero_srate": (_set("srate", 0), "srate"),
    "negative_window_len": (_set("window_len", -1.0), "window_len"),
    "text_maxmem": (_set("maxmem", "64"), "maxmem"),
}


@pytest.mark.parametrize(("edit", "key"), list(DECODE_REFUSALS.values()), ids=list(DECODE_REFUSALS))
def test_state_from_json_refuses_each_invalid_field_naming_it(state, edit, key):
    document = _document(state)
    result = edit(document)
    if result is not None:
        document = result
    with pytest.raises(ValueError, match=rf"^{key}: "):
        state_from_json(json.loads(json.dumps(document)))


def test_state_from_json_accepts_null_where_a_state_may_hold_null(state):
    document = _document(state)
    document["sos"] = None
    document["maxmem"] = None
    back = state_from_json(document)
    assert back["sos"] is None
    assert back["maxmem"] is None


ENCODE_REFUSALS = {
    "float32_array": (lambda state: dict(state, T=state["T"].astype(np.float32)), "T"),
    "integer_array": (lambda state: dict(state, T=state["T"].astype(np.int64)), "T"),
    "nan_cutoff": (lambda state: dict(state, cutoff=float("nan")), "cutoff"),
    "infinite_window_len": (lambda state: dict(state, window_len=float("inf")), "window_len"),
    "unsupported_type": (lambda state: dict(state, cutoff=1 + 2j), "cutoff"),
    "not_a_dict": (lambda state: [1], "state"),
}


@pytest.mark.parametrize(("make", "key"), list(ENCODE_REFUSALS.values()), ids=list(ENCODE_REFUSALS))
def test_state_to_json_refuses_each_invalid_value_naming_it(state, make, key):
    with pytest.raises(ValueError, match=rf"^{key}: "):
        state_to_json(make(state))


def test_numpy_scalars_and_tuples_are_written_as_plain_json(state):
    odd = dict(
        state,
        cutoff=np.float64(5.0),
        step_size=np.int64(32),
        maxmem=np.int64(64),
        last_trivial=np.True_,
        channels=tuple(state["channels"]),
    )
    back = state_from_json(json.loads(json.dumps(state_to_json(odd))))
    assert type(back["step_size"]) is int
    assert back["last_trivial"] is True
    assert back["channels"] == list(state["channels"])
