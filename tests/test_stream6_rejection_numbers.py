"""Release-readiness checks for rejection counts (issue 551, stream 6)."""

import logging
import os
from pathlib import Path

import numpy as np
import pytest

from eegprep.functions.adminfunc.eeg_checkset import _matlab_num2str
from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.popfunc.eeg_rejsuperpose import eeg_rejsuperpose
from eegprep.functions.popfunc.pop_autorej import pop_autorej
from eegprep.functions.popfunc._eegplot_rejection import (
    MARKED_EPOCHS_NOTICE,
    open_epoched_rejection_browser,
)
from eegprep.functions.popfunc.pop_eegthresh import pop_eegthresh, pop_eegthresh_dialog_spec
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_rejchan import pop_rejchan, pop_rejchan_dialog_spec
from eegprep.functions.popfunc.pop_rejcont import pop_rejcont, pop_rejcont_dialog_spec
from eegprep.functions.popfunc.pop_rejepoch import pop_rejepoch
from eegprep.functions.popfunc.pop_rejkurt import pop_rejkurt
from eegprep.functions.popfunc.pop_rejspec import pop_rejspec, pop_rejspec_dialog_spec
from eegprep.functions.popfunc.pop_rejtrend import pop_rejtrend
from eegprep.plugins.firfilt.pop_eegfiltnew import pop_eegfiltnew
from tests.eeglab_tests import eeglab_test
from tests.eeglab_tests.recordings import _records
from tests.fixtures import SAMPLE_DATASET_PATH, create_test_eeg


EPOCH_DATASET_PATH = Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data_epochs_ica.set"
REJCONT_LOW_REGIONS = [
    [1, 4832],
    [5057, 6272],
    [6401, 7008],
    [7361, 11488],
    [12513, 12832],
    [12929, 13760],
    [13857, 14848],
    [15009, 17344],
    [17441, 18688],
    [18817, 19136],
    [19233, 22208],
    [22593, 23456],
    [23649, 24288],
    [24385, 26816],
    [27105, 29632],
    [29761, 30496],
]
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


def _rejcont(eeglab_backend, EEG, threshold=10.0):
    return eeglab_backend(
        "pop_rejcont",
        EEG,
        "elecrange",
        np.arange(1, 33, dtype=float)[None, :],
        "freqlimit",
        np.array([[20.0, 40.0]]),
        "threshold",
        np.asarray(threshold, dtype=float).reshape(1, -1),
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


def _sample_rejcont(eeg, threshold, freqlimit=None):
    return pop_rejcont(
        eeg,
        "elecrange",
        list(range(1, 33)),
        "freqlimit",
        [20, 40] if freqlimit is None else freqlimit,
        "threshold",
        threshold,
        "epochlength",
        0.5,
        "contiguous",
        4,
        "addlength",
        0.25,
        "taper",
        "hamming",
    )


def test_pop_rejcont_keeps_the_sample_recording_regions(caplog):
    with caplog.at_level(logging.WARNING, logger="eegprep"):
        out, selected = _sample_rejcont(_continuous(), 10)

    assert "Field 'urevent'" not in caplog.text
    assert out["pnts"] == 28008
    assert np.asarray(selected, dtype=int).tolist() == REJCONT_REGIONS


def test_pop_rejcont_two_sided_and_cell_thresholds():
    continuous = _continuous()
    wide, wide_regions = _sample_rejcont(continuous, [-50, 10])
    narrow, narrow_regions = _sample_rejcont(continuous, [-5, 10])
    cell, cell_regions = _sample_rejcont(continuous, [[-50, 10]], freqlimit=[[20, 40]])
    dialog_text, dialog_regions = _sample_rejcont(continuous, "-50 10")
    _out, command = pop_rejcont(
        continuous,
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
        return_com=True,
    )

    assert wide["pnts"] == 28008
    assert np.asarray(wide_regions, dtype=int).tolist() == REJCONT_REGIONS
    assert narrow["pnts"] == 3496
    assert np.asarray(narrow_regions, dtype=int).tolist() == REJCONT_LOW_REGIONS
    assert cell["pnts"] == wide["pnts"]
    assert np.asarray(cell_regions, dtype=int).tolist() == REJCONT_REGIONS
    assert dialog_text["pnts"] == 28008
    assert np.asarray(dialog_regions, dtype=int).tolist() == REJCONT_REGIONS
    assert "eegplot" not in command
    assert command.splitlines()[1].startswith("% EEG = pop_rejcont(EEG, ")
    assert command.splitlines()[2] == (
        "EEG = pop_select( EEG, 'nopoint',[321 480;3489 3936;4289 4608;7713 7904;"
        "13505 13664;26561 26816;27201 27360;27553 27840;28033 28544]);"
    )


def test_pop_rejcont_blank_zeroes_float32_sample_columns():
    eeg = _continuous()
    eeg["data"] = np.asarray(eeg["data"], dtype=np.float32)
    original = np.array(eeg["data"], copy=True)

    out, _regions = pop_rejcont(
        eeg,
        "elecrange",
        list(range(1, 33)),
        "freqlimit",
        [20, 40],
        "threshold",
        10,
        "taper",
        "none",
        "correct",
        "blank",
        "eegplot",
        "off",
    )
    data = np.asarray(out["data"])
    zeroed = np.flatnonzero(np.all(data == 0, axis=0) & np.any(original != 0, axis=0))

    assert data.shape == original.shape
    assert zeroed.size == 1056
    assert np.all(data[:, zeroed] == 0)
    assert np.array_equal(np.asarray(eeg["data"]), original)


def test_pop_rejcont_names_a_two_number_add_trails_value():
    eeg = create_test_eeg(n_channels=2, n_samples=40, n_trials=1, srate=100)

    with pytest.raises(ValueError, match="pop_rejcont: Add trails before and after regions") as error:
        pop_rejcont(eeg, "addlength", "1 2", "eegplot", "off")

    assert "0.25" in str(error.value)
    assert "'1 2'" in str(error.value)


def test_pop_eegthresh_history_and_dialog_follow_num2str():
    eeg = create_test_eeg(n_channels=32, n_samples=4, n_trials=2, srate=128)
    eeg["xmin"] = -1.0
    eeg["xmax"] = 1.9921875
    spec = pop_eegthresh_dialog_spec(_epochs(), 1)

    _out, command = pop_eegthresh(eeg, 1, list(range(1, 33)), -75, 75, -1, 1.9921875, 0, 0, return_com=True)

    assert controls_by_tag(spec)["starttime"].value == "-1"
    assert controls_by_tag(spec)["endtime"].value == "1.9922"
    assert command == "EEG = pop_eegthresh(EEG,1,[1:32] ,-75,75,-1,1.9922,0,0);"
    assert _matlab_num2str(1.9921875) == "1.9922"
    assert _matlab_num2str(12.34567) == "12.3457"
    assert _matlab_num2str(123.456789) == "123.4568"


def test_marked_epochs_notice_closes_instead_of_staying_in_the_session_list(monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6 import QtWidgets
    except ImportError as exc:
        pytest.skip(f"Qt is not loadable: {exc}")

    from eegprep.functions.popfunc import _eegplot_rejection as rejection

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = QtWidgets.QMainWindow()
    window.show()
    app.processEvents()
    before = list(rejection._notice_boxes)

    rejection.show_marked_epochs_notice()
    app.processEvents()
    boxes = window.findChildren(QtWidgets.QMessageBox, "marked_epochs_notice")
    assert boxes
    assert boxes[0].parent() is window
    assert rejection._notice_boxes == before
    boxes[0].close()
    app.processEvents()
    assert window.findChildren(QtWidgets.QMessageBox, "marked_epochs_notice") == []
    window.close()

    monkeypatch.setattr(rejection, "_notice_parent", lambda: None)
    rejection.show_marked_epochs_notice()
    app.processEvents()
    held = [box for box in rejection._notice_boxes if box not in before]
    assert len(held) == 1
    held[0].close()
    app.processEvents()
    assert held[0] not in rejection._notice_boxes


def test_update_marks_shows_the_eeglab_notice(monkeypatch):
    notices = []
    eeg = create_test_eeg(n_channels=2, n_samples=8, n_trials=2, srate=100)

    def fake_eegplot(_data, *args, **kwargs):
        del args
        kwargs["command_callback"](np.zeros((0, 5)))
        return "window"

    monkeypatch.setattr(
        "eegprep.functions.popfunc._eegplot_rejection.eegplot",
        fake_eegplot,
    )
    monkeypatch.setattr(
        "eegprep.functions.popfunc._eegplot_rejection.show_marked_epochs_notice",
        lambda parent=None: notices.append(MARKED_EPOCHS_NOTICE),
    )

    open_epoched_rejection_browser(
        eeg,
        data=np.asarray(eeg["data"]),
        icacomp=1,
        elecrange=[1],
        kind="thresh",
        superpose=0,
        reject=0,
        command="",
        show=True,
    )
    open_epoched_rejection_browser(
        eeg,
        data=np.asarray(eeg["data"]),
        icacomp=1,
        elecrange=[1],
        kind="thresh",
        superpose=0,
        reject=1,
        command="",
        show=True,
    )

    assert notices == [MARKED_EPOCHS_NOTICE]
    assert notices[0].startswith("Epochs (=trials) marked for rejection have been noted.")


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
    norm = tags["norm"].callback
    assert norm is not None
    assert norm.name == "toggle_checked_text"
    assert norm.params["target"] == "normlab"
    assert tags["normlab"].string == "Z-score threshold [max] or [min max]"
    assert norm.params["unchecked"] == "Absolute threshold [max] or [min max]"
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
    assert "eeg_rejsuperpose(EEG, 0, 1, 1, 1, 1, 1, 1, 1)" in session.ALLCOM[-3]
    assert "EEG = pop_rejepoch( EEG, 2,0);" in session.ALLCOM
    assert any("pop_newset" in line for line in session.ALLCOM)


def test_pop_rejspec_dialog_defaults_to_fft():
    spec = pop_rejspec_dialog_spec(_epochs(), 1)
    method = controls_by_tag(spec)["method"]

    assert method.string == "FFT|Multitaper"
    assert method.value == 1


def test_threshold_75_marks_71_epochs_and_export_copies_them():
    epochs = _epochs()
    xmin = float(np.asarray(epochs["xmin"]).reshape(-1)[0])
    xmax = float(np.asarray(epochs["xmax"]).reshape(-1)[0])
    marked, rejected = pop_eegthresh(epochs, 1, list(range(1, 33)), -75, 75, xmin, xmax, 0, 0)
    session = EEGPrepSession()
    session.store_current(marked, new=True)

    MenuActionDispatcher(session).dispatch("eeg_rejsuperpose:data_to_ica")

    assert len(rejected) == 71
    assert int(np.asarray(session.EEG["reject"]["icarejmanual"]).sum()) == 71


def test_pop_rejepoch_history_compresses_ranges_and_records_confirm_zero():
    eeg = create_test_eeg(n_channels=2, n_samples=8, n_trials=10, srate=100)
    marks = np.zeros(10, dtype=bool)
    marks[:6] = True
    marks[8] = True

    _out, command = pop_rejepoch(eeg, marks, 0, return_com=True)

    assert command == "EEG = pop_rejepoch( EEG, [1:6 9] ,0);"


def test_pop_rejepoch_confirm_minus_one_keeps_an_empty_dataset():
    eeg = create_test_eeg(n_channels=2, n_samples=8, n_trials=3, srate=100)
    eeg["reject"]["rejglobal"] = np.ones(3, dtype=bool)

    empty = pop_rejepoch(eeg, confirm=-1)

    assert empty["trials"] == 0
    assert np.asarray(empty["data"]).shape[-1] == 0


def test_pop_rejepoch_names_the_empty_dataset_error():
    eeg = create_test_eeg(n_channels=2, n_samples=8, n_trials=3, srate=100)
    eeg["reject"]["rejglobal"] = np.ones(3, dtype=bool)

    with pytest.raises(ValueError, match="pop_rejepoch: rejecting every epoch"):
        pop_rejepoch(eeg)


def test_pop_autorej_names_a_non_numeric_dialog_field():
    with pytest.raises(ValueError, match="pop_autorej: Threshold limit \\(microV\\)") as error:
        pop_autorej(_epochs(), "threshold", "abc", "eegplot", "off")

    assert "1000" in str(error.value)


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


@eeglab_test("functions/popfunc/pop_rejcont.m", "test_reference_pop_rejcont_two_sided_thresholds")
def test_reference_pop_rejcont_two_sided_thresholds(eeglab_backend):
    matlab = eeglab_backend("pop_loadset", str(SAMPLE_DATASET_PATH))
    for threshold, expected in (([-50.0, 10.0], REJCONT_REGIONS), ([-5.0, 10.0], REJCONT_LOW_REGIONS)):
        _eeg, regions = _rejcont(eeglab_backend, matlab, threshold)
        assert np.rint(np.asarray(regions, dtype=float)).astype(int).tolist() == expected


@eeglab_test("functions/popfunc/pop_eegthresh.m", "test_reference_threshold_75_marks_71_epochs")
def test_reference_threshold_75_marks_71_epochs(eeglab_backend):
    epochs = _epochs()
    xmin = float(np.asarray(epochs["xmin"]).reshape(-1)[0])
    xmax = float(np.asarray(epochs["xmax"]).reshape(-1)[0])
    _marked, rejected = eeglab_backend(
        "pop_eegthresh",
        eeglab_backend("pop_loadset", str(EPOCH_DATASET_PATH)),
        1.0,
        np.arange(1, 33, dtype=float)[None, :],
        -75.0,
        75.0,
        xmin,
        xmax,
        0.0,
        0.0,
        nargout=2,
    )

    assert _indices(rejected) and len(_indices(rejected)) == 71


@eeglab_test("functions/popfunc/pop_autorej.m", "test_reference_pop_autorej_eeglab_omits_epoch_16")
def test_reference_pop_autorej_eeglab_omits_epoch_16(eeglab_backend, request):
    if request.config.getoption("--eeglab-backend") != "matlab":
        pytest.skip("pop_autorej.m:224 reads icarejkurt; EEGPrep's rejkurt result is asserted separately")
    _eeg, rejected = eeglab_backend(
        "pop_autorej",
        eeglab_backend("pop_loadset", str(EPOCH_DATASET_PATH)),
        "eegplot",
        "off",
        "nogui",
        "on",
        nargout=2,
    )

    assert sorted(_indices(rejected)) == [1, 2, 28, 46, 70]


@eeglab_test(
    "functions/miscfunc/eeg_regepochs.m",
    "test_reference_eeg_regepochs_keeps_events_inside_overlapping_epochs",
)
def test_reference_eeg_regepochs_keeps_events_inside_overlapping_epochs(eeglab_backend):
    expected = (
        (0.25, [0.0, 0.5], 952, 2212, 1904, 1107),
        (0.5, [0.0, 2.0], 473, 2506, 1892, 630),
    )
    for recurrence, limits, trials, events, x_events, urevents in expected:
        result = eeglab_backend(
            "eeg_regepochs",
            eeglab_backend("pop_loadset", str(SAMPLE_DATASET_PATH)),
            float(recurrence),
            np.asarray(limits, dtype=float).reshape(1, 2),
            np.nan,
        )
        assert int(np.asarray(result["trials"]).reshape(-1)[0]) == trials
        assert _record_count(result.get("event")) == events
        assert _type_count(result.get("event"), "X") == x_events
        assert _record_count(result.get("urevent")) == urevents


def test_type_count_reads_matlab_struct_event_arrays():
    events = np.array(
        [("X     ", 1.0), ("square", 2.0), ("X", 3.0)],
        dtype=[("type", "U6"), ("latency", float)],
    )
    chars = np.zeros(2, dtype=[("type", "U1", (1, 6)), ("latency", float)])
    chars["type"][0, 0] = list("X     ")
    chars["latency"][0] = 1.0
    chars["type"][1, 0] = list("square")
    chars["latency"][1] = 2.0

    assert _record_count(events) == 3
    assert _type_count(events, "X") == 2
    assert _type_count(chars, "X") == 1
    assert (
        _type_count(np.array([{"type": "X", "latency": 1.0}, {"type": "rt", "latency": 2.0}], dtype=object), "X") == 1
    )


def _record_count(value) -> int:
    if value is None:
        return 0
    if isinstance(value, np.ndarray):
        return int(value.size)
    if isinstance(value, dict):
        return 1
    return len(value)


def _type_count(value, event_type: str) -> int:
    return sum(
        _matlab_event_type(record.get("type")) == event_type for record in _records(value) if isinstance(record, dict)
    )


def _matlab_event_type(value) -> str:
    if isinstance(value, bytes):
        return value.decode("latin-1").strip()
    if isinstance(value, str):
        return value.strip()
    array = np.asarray(value)
    if array.dtype == object and array.size == 1:
        return _matlab_event_type(array.reshape(-1)[0])
    if array.dtype.kind in "SU":
        return "".join(str(item) for item in array.ravel(order="F")).strip()
    return str(value).strip()


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
