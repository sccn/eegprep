"""Release-readiness checks for rejection counts (issue 551, stream 6)."""

from pathlib import Path

import numpy as np
import pytest

from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher
from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.popfunc.eeg_rejsuperpose import eeg_rejsuperpose
from eegprep.functions.popfunc.pop_autorej import pop_autorej
from eegprep.functions.popfunc.pop_eegthresh import pop_eegthresh_dialog_spec
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_rejchan import pop_rejchan, pop_rejchan_dialog_spec
from eegprep.functions.popfunc.pop_rejcont import pop_rejcont, pop_rejcont_dialog_spec
from eegprep.functions.popfunc.pop_rejepoch import pop_rejepoch
from eegprep.functions.popfunc.pop_rejkurt import pop_rejkurt
from eegprep.functions.popfunc.pop_rejspec import pop_rejspec
from eegprep.functions.popfunc.pop_rejtrend import pop_rejtrend
from eegprep.plugins.firfilt.pop_eegfiltnew import pop_eegfiltnew
from tests.eeglab_tests import eeglab_test
from tests.fixtures import SAMPLE_DATASET_PATH, create_test_eeg


EPOCH_DATASET_PATH = Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data_epochs_ica.set"
REJCONT_REGIONS = [
    [321, 480],
    [3489, 3936],
    [4289, 4608],
    [7713, 7904],
    [13505, 13664],
    [26561, 26816],
    [27201, 27360],
    [27553, 27840],
    [28033, 28544],
]
FFT_DATA_EPOCHS = [
    1,
    3,
    6,
    7,
    10,
    12,
    14,
    16,
    17,
    20,
    21,
    23,
    25,
    27,
    30,
    31,
    32,
    35,
    36,
    37,
    39,
    41,
    42,
    44,
    45,
    46,
    47,
    48,
    49,
    51,
    52,
    59,
    61,
    63,
    64,
    66,
    67,
    70,
    71,
    73,
    74,
    75,
    77,
    78,
]
FFT_ICA_EPOCHS = [
    2,
    3,
    4,
    5,
    7,
    8,
    10,
    13,
    14,
    15,
    16,
    17,
    19,
    20,
    21,
    22,
    23,
    24,
    26,
    27,
    28,
    29,
    31,
    33,
    34,
    36,
    39,
    40,
    43,
    45,
    47,
    49,
    50,
    51,
    52,
    54,
    56,
    57,
    59,
    60,
    68,
    71,
    73,
    77,
    78,
    79,
]


def _continuous():
    return pop_loadset(str(SAMPLE_DATASET_PATH))


def _epochs():
    return pop_loadset(str(EPOCH_DATASET_PATH))


def _indices(value) -> list[int]:
    array = np.asarray(value).reshape(-1)
    return [int(item) for item in array]


def _rejcont(eeglab_backend, EEG):
    return eeglab_backend(
        "pop_rejcont",
        EEG,
        "elecrange",
        np.arange(1, 33, dtype=float)[None, :],
        "freqlimit",
        np.array([[20.0, 40.0]]),
        "threshold",
        10.0,
        "epochlength",
        0.5,
        "contiguous",
        4.0,
        "addlength",
        0.25,
        "taper",
        "hamming",
        "overlap",
        0.25,
        "onlyreturnselection",
        "on",
        "verbose",
        "off",
        "eegplot",
        "off",
        nargout=2,
    )


def _rejspec(eeglab_backend, EEG, icacomp, method, threshold):
    rows = int(EEG["nbchan"]) if icacomp else int(np.asarray(EEG["icaweights"]).shape[0])
    return eeglab_backend(
        "pop_rejspec",
        EEG,
        float(icacomp),
        "elecrange",
        np.arange(1, rows + 1, dtype=float)[None, :],
        "method",
        method,
        "threshold",
        np.asarray(threshold, dtype=float).reshape(1, 2),
        "freqlimits",
        np.array([[15.0, 30.0]]),
        "eegplotreject",
        0.0,
        nargout=2,
    )


def test_pop_rejcont_keeps_the_sample_recording_regions():
    out, selected = pop_rejcont(
        _continuous(),
        "elecrange",
        list(range(1, 33)),
        "freqlimit",
        [20, 40],
        "threshold",
        10,
        "epochlength",
        0.5,
        "contiguous",
        4,
        "addlength",
        0.25,
        "taper",
        "hamming",
        "overlap",
        0.25,
    )

    assert out["pnts"] == 28008
    assert np.asarray(selected, dtype=int).tolist() == REJCONT_REGIONS


def test_pop_rejcont_keeps_the_same_regions_after_1hz_highpass():
    filtered = pop_eegfiltnew(_continuous(), 1)
    out, selected = pop_rejcont(
        filtered,
        "elecrange",
        list(range(1, 33)),
        "freqlimit",
        [20, 40],
        "threshold",
        10,
        "epochlength",
        0.5,
        "contiguous",
        4,
        "addlength",
        0.25,
        "taper",
        "hamming",
        "overlap",
        0.25,
    )

    assert out["pnts"] == 28008
    assert np.asarray(selected, dtype=int).tolist() == REJCONT_REGIONS


def test_pop_rejcont_scroll_spacing_and_dialog_have_no_help(monkeypatch):
    captured = {}

    def fake_eegplot(data, **kwargs):
        captured["spacing"] = kwargs["spacing"]
        return lambda winrej: None

    monkeypatch.setattr("eegprep.functions.popfunc.pop_rejcont.eegplot", fake_eegplot)
    out, selected = pop_rejcont(
        _continuous(),
        "elecrange",
        list(range(1, 33)),
        "threshold",
        1e9,
        "eegplot",
        "on",
    )

    assert captured["spacing"] == 80
    assert selected.size == 0
    assert out["pnts"] == 30504
    spec = pop_rejcont_dialog_spec(_continuous())
    assert spec.show_help_button is False
    assert spec.help_text is None


def test_pop_rejchan_normalized_channels_match_the_sample_sets():
    continuous = _continuous()
    epoched = _epochs()
    _, kurt, _measure = pop_rejchan(
        continuous, "elec", list(range(1, 33)), "measure", "kurt", "threshold", 5, "norm", "on"
    )
    _, probability, _measure = pop_rejchan(
        continuous, "elec", list(range(1, 33)), "measure", "prob", "threshold", 5, "norm", "on"
    )
    _, epoch_kurt, _measure = pop_rejchan(
        epoched, "elec", list(range(1, 33)), "measure", "kurt", "threshold", 5, "norm", "on"
    )

    assert kurt == [1, 2, 6]
    assert probability == [1, 2]
    assert epoch_kurt == [1, 2]


def test_pop_rejchan_dialog_and_history_follow_vararg2str():
    spec = pop_rejchan_dialog_spec(_epochs())
    tags = controls_by_tag(spec)

    callback = tags["measure"].callback
    assert tags["measure"].string == "Probability|Kurtosis|Spectrum"
    assert tags["freqrange"].enabled is False
    assert tags["freq_label"].enabled is False
    assert callback is not None
    assert callback.params["enabled_index"] == 3
    _, command = pop_rejchan(
        _continuous(),
        "elec",
        list(range(1, 33)),
        "threshold",
        5,
        "norm",
        "on",
        "measure",
        "kurt",
        return_com=True,
    )
    assert command == "EEG = pop_rejchan(EEG, 'elec',[1:32] ,'threshold',5,'norm','on','measure','kurt');"


def test_pop_rejspec_fft_epoch_lists_stay_unchanged():
    epoched = _epochs()
    _, data_epochs = pop_rejspec(
        epoched,
        1,
        "elecrange",
        list(range(1, 33)),
        "method",
        "fft",
        "threshold",
        [-30, 30],
        "freqlimits",
        [15, 30],
        "eegplotreject",
        0,
    )
    _, ica_epochs = pop_rejspec(
        epoched,
        0,
        "elecrange",
        list(range(1, 33)),
        "method",
        "fft",
        "threshold",
        [-30, 30],
        "freqlimits",
        [15, 30],
        "eegplotreject",
        0,
    )

    assert data_epochs == FFT_DATA_EPOCHS
    assert ica_epochs == FFT_ICA_EPOCHS


def test_pop_rejspec_multitaper_marks_epoch_37_and_the_ica_list():
    epoched = _epochs()
    _, data_35 = pop_rejspec(
        epoched,
        1,
        "elecrange",
        list(range(1, 33)),
        "method",
        "multitaper",
        "threshold",
        [-35, 35],
        "freqlimits",
        [15, 30],
        "eegplotreject",
        0,
    )
    _, data_30 = pop_rejspec(
        epoched,
        1,
        "elecrange",
        list(range(1, 33)),
        "method",
        "multitaper",
        "threshold",
        [-30, 30],
        "freqlimits",
        [15, 30],
        "eegplotreject",
        0,
    )
    _, ica = pop_rejspec(
        epoched,
        0,
        "elecrange",
        list(range(1, 33)),
        "method",
        "multitaper",
        "threshold",
        [-30, 30],
        "freqlimits",
        [15, 30],
        "eegplotreject",
        0,
    )

    assert data_35 == [37]
    assert data_30 == [37]
    assert ica == [13, 14, 15, 26, 79]


def test_pop_autorej_channel_mode_keeps_the_kurtosis_epoch_eeglab_drops():
    # pop_autorej.m:224 reads icarejkurt in channel mode, so EEGLAB omits epoch 16.
    _out, rejected = pop_autorej(_epochs(), "eegplot", "off")

    assert 16 in rejected
    assert rejected == [1, 2, 16, 28, 46, 70]


def test_ica_trend_and_kurtosis_defaults_mark_the_audited_epochs():
    epoched = _epochs()
    trend = pop_rejtrend(epoched, 0, list(range(1, 33)), int(epoched["pnts"]), 0.5, 0.3, 0, 0)
    kurtosis, _local, _global, _count = pop_rejkurt(epoched, 0, list(range(1, 33)), 5, 5, 0, 0)

    assert int(np.asarray(trend["reject"]["icarejconst"]).sum()) == 80
    assert (np.flatnonzero(np.asarray(kurtosis["reject"]["icarejkurt"])) + 1).tolist() == [9, 10, 16, 53, 69]


def test_component_threshold_dialog_defaults_to_the_first_five_components():
    spec = pop_eegthresh_dialog_spec(_epochs(), 0)

    assert controls_by_tag(spec)["elecrange"].value == "1:5"


def test_export_marks_copies_rejglobal_onto_the_manual_field():
    session = EEGPrepSession()
    eeg = create_test_eeg(n_channels=3, n_samples=8, n_trials=3, srate=100)
    eeg["reject"]["rejthresh"] = np.array([True, False, True])
    session.store_current(eeg, new=True)

    MenuActionDispatcher(session).dispatch("eeg_rejsuperpose:data_to_ica")

    reject = session.EEG["reject"]
    assert reject["icarejmanual"].tolist() == [True, False, True]
    assert reject["rejglobal"].tolist() == [True, False, True]
    assert session.ALLCOM[-1] == "EEG.reject.icarejmanual = EEG.reject.rejglobal;"
    assert "eeg_rejsuperpose(EEG, 1, 1, 1, 1, 1, 1, 1, 1)" in session.ALLCOM[-2]


def test_export_ica_marks_copies_the_audited_kurtosis_epochs():
    session = EEGPrepSession()
    marked, *_rest = pop_rejkurt(_epochs(), 0, list(range(1, 33)), 5, 5, 0, 0)
    session.store_current(marked, new=True)

    MenuActionDispatcher(session).dispatch("eeg_rejsuperpose:ica_to_data")

    reject = session.EEG["reject"]
    assert (np.flatnonzero(np.asarray(reject["rejmanual"])) + 1).tolist() == [9, 10, 16, 53, 69]
    np.testing.assert_array_equal(reject["rejmanual"], reject["rejglobal"])
    assert session.ALLCOM[-1] == "EEG.reject.rejmanual = EEG.reject.rejglobal;"


def test_reject_marked_epochs_superposes_before_removing_trials():
    session = EEGPrepSession()
    eeg = create_test_eeg(n_channels=3, n_samples=8, n_trials=3, srate=100)
    eeg["reject"]["icarejmanual"] = np.array([False, True, False])
    session.store_current(eeg, new=True)

    MenuActionDispatcher(session).dispatch("pop_rejepoch:ica")

    assert session.EEG["trials"] == 2
    assert "eeg_rejsuperpose(EEG, 0, 1, 1, 1, 1, 1, 1, 1)" in session.ALLCOM[-2]
    assert session.ALLCOM[-1].startswith("EEG = pop_rejepoch(")


def test_pop_rejepoch_names_the_empty_dataset_error():
    eeg = create_test_eeg(n_channels=2, n_samples=8, n_trials=3, srate=100)
    eeg["reject"]["rejglobal"] = np.ones(3, dtype=bool)

    with pytest.raises(ValueError, match="pop_rejepoch: rejecting every epoch"):
        pop_rejepoch(eeg)


def test_eeg_rejsuperpose_public_typerej_still_means_data():
    eeg = create_test_eeg(n_channels=2, n_samples=8, n_trials=2, srate=100)
    eeg["reject"]["rejmanual"] = np.array([True, False])
    out, command = eeg_rejsuperpose(eeg, 1, 1, 0, 0, 0, 0, 0, 0, return_com=True)

    assert command.startswith("EEG = eeg_rejsuperpose(EEG, 1,")
    assert out["reject"]["rejglobal"].tolist() == [True, False]


@eeglab_test("functions/popfunc/pop_rejcont.m", "test_reference_pop_rejcont_sample_regions")
def test_reference_pop_rejcont_sample_regions(eeglab_backend):
    _matlab_eeg, matlab_regions = _rejcont(eeglab_backend, eeglab_backend("pop_loadset", str(SAMPLE_DATASET_PATH)))
    python_eeg, python_regions = pop_rejcont(
        _continuous(),
        "elecrange",
        list(range(1, 33)),
        "freqlimit",
        [20, 40],
        "threshold",
        10,
        "epochlength",
        0.5,
        "contiguous",
        4,
        "addlength",
        0.25,
        "taper",
        "hamming",
        "overlap",
        0.25,
        "onlyreturnselection",
        "on",
        "verbose",
        "off",
    )

    assert python_regions.astype(int).tolist() == REJCONT_REGIONS
    assert np.rint(np.asarray(matlab_regions, dtype=float)).astype(int).tolist() == python_regions.astype(int).tolist()


@eeglab_test("functions/popfunc/pop_rejchan.m", "test_reference_pop_rejchan_sample_channels")
def test_reference_pop_rejchan_sample_channels(eeglab_backend):
    electrodes = np.arange(1, 33, dtype=float)[None, :]
    matlab_kurt = eeglab_backend(
        "pop_rejchan",
        eeglab_backend("pop_loadset", str(SAMPLE_DATASET_PATH)),
        "elec",
        electrodes,
        "measure",
        "kurt",
        "threshold",
        5.0,
        "norm",
        "on",
        nargout=3,
    )
    matlab_probability = eeglab_backend(
        "pop_rejchan",
        eeglab_backend("pop_loadset", str(SAMPLE_DATASET_PATH)),
        "elec",
        electrodes,
        "measure",
        "prob",
        "threshold",
        5.0,
        "norm",
        "on",
        nargout=3,
    )
    _, kurt, _measure = pop_rejchan(
        _continuous(), "elec", list(range(1, 33)), "measure", "kurt", "threshold", 5, "norm", "on"
    )
    _, probability, _measure = pop_rejchan(
        _continuous(), "elec", list(range(1, 33)), "measure", "prob", "threshold", 5, "norm", "on"
    )

    assert kurt == [1, 2, 6]
    assert probability == [1, 2]
    assert _indices(matlab_kurt[1]) == kurt
    assert _indices(matlab_probability[1]) == probability


@eeglab_test("functions/popfunc/pop_rejspec.m", "test_reference_pop_rejspec_multitaper_epochs")
def test_reference_pop_rejspec_multitaper_epochs(eeglab_backend):
    loaded = eeglab_backend("pop_loadset", str(EPOCH_DATASET_PATH))
    _matlab_data, matlab_data_epochs = _rejspec(eeglab_backend, loaded, 1, "multitaper", [-35.0, 35.0])
    _matlab_ica, matlab_ica_epochs = _rejspec(eeglab_backend, loaded, 0, "multitaper", [-30.0, 30.0])
    _fft_data, matlab_fft = _rejspec(eeglab_backend, loaded, 1, "fft", [-30.0, 30.0])
    python = _epochs()
    _, data_epochs = pop_rejspec(
        python,
        1,
        "elecrange",
        list(range(1, 33)),
        "method",
        "multitaper",
        "threshold",
        [-35, 35],
        "freqlimits",
        [15, 30],
        "eegplotreject",
        0,
    )
    _, ica_epochs = pop_rejspec(
        python,
        0,
        "elecrange",
        list(range(1, 33)),
        "method",
        "multitaper",
        "threshold",
        [-30, 30],
        "freqlimits",
        [15, 30],
        "eegplotreject",
        0,
    )

    assert data_epochs == [37]
    assert ica_epochs == [13, 14, 15, 26, 79]
    assert _indices(matlab_data_epochs) == data_epochs
    assert _indices(matlab_ica_epochs) == ica_epochs
    assert _indices(matlab_fft) == FFT_DATA_EPOCHS
