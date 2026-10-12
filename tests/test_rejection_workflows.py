import copy
from pathlib import Path

import numpy as np
import pytest

import eegprep.functions.popfunc._eegplot_rejection as eegplot_rejection_module
import eegprep.functions.popfunc.pop_rejcont as pop_rejcont_module
from eegprep.functions.adminfunc.console import _console_python_command
from eegprep.functions.popfunc.eeg_rejsuperpose import eeg_rejsuperpose
from eegprep.functions.popfunc._rejection import (
    jointprob,
    jointprob_marks,
    kurtosis_marks,
    rejkurt,
)
from eegprep.functions.popfunc.pop_eegthresh import pop_eegthresh
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_rejchan import pop_rejchan
from eegprep.functions.popfunc.pop_rejcont import pop_rejcont
from eegprep.functions.popfunc.pop_rejepoch import pop_rejepoch
from eegprep.functions.popfunc.pop_rejmenu import pop_rejmenu
from eegprep.functions.popfunc.pop_rejspec import pop_rejspec
from eegprep.functions.popfunc.pop_selectcomps import pop_selectcomps
from eegprep.functions.sigprocfunc.eegplot import eegplot
from eegprep.plugins.ICLabel.pop_viewprops import pop_viewprops
from tests.eeglab_tests import assert_matlab_near, eeglab_test, load_matlab_test_fixture
from tests.fixtures import SAMPLE_DATASET_PATH, create_test_eeg


def _source_epoch_sample(backend, suite_root):
    return backend("pop_loadset", str(suite_root / "eeglab/sample_data/eeglab_data_epochs_ica.set"))


def _source_empty_rejection_eeg(backend, trials):
    eeg = backend("eeg_emptyset")
    eeg["nbchan"], eeg["trials"] = 2.0, float(trials)
    return eeg


def _source_allow_empty_dataset(backend, function, *args):
    # The three upstream mypop_* helpers suppress only this specific error.
    # Completing a smoke workflow therefore does not prove every call succeeded.
    try:
        return backend(function, *args, nargout=2)
    except Exception as error:
        if "Error: dataset is empty" not in str(error):
            raise
        return np.empty((0, 0)), np.empty((0, 0))


@eeglab_test("unittesting_popfunc/pop_eegthresh/popfunc_pop_eegthresh_wrapperTest.m", "test_test_pop_eegthresh")
def test_reference_pop_eegthresh(eeglab_backend, eeglab_suite_root):
    # Exact rng('default'); rng(1) inputs, reproducible by the adjacent .m script.
    inputs = load_matlab_test_fixture(Path(__file__).parent / "matlab/pop_eegthresh_source_inputs.mat")
    _source_epoch_sample(eeglab_backend, eeglab_suite_root)
    component_selection = 0
    for mode in (1.0, 0.0):
        for superpose, reject in ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)):
            for shifted in (False, True):
                for selection in range(4):
                    # All 64 source cases reload, including those following a rejection.
                    eeg = _source_epoch_sample(eeglab_backend, eeglab_suite_root)
                    if selection == 0:
                        count = np.asarray(eeg["nbchan"]).item() if mode else eeg["icaweights"].shape[0]
                        if not mode and superpose == reject == 0 and not shifted:
                            count = np.size(eeg["icachansind"])
                        channels = np.arange(1.0, count + 1)[None, :]
                    elif selection == 1:
                        channels = np.empty((0, 0)) if mode else np.array([[1.0]])
                    elif mode:
                        channels = inputs["Elements"]
                    else:
                        channels = inputs["ComponentElements"][component_selection : component_selection + 1]
                        component_selection += 1
                    low, high = -100.0, 100.0
                    if selection == 2:
                        low, high = -inputs["Thresh"], inputs["Thresh"]
                    elif selection == 3:
                        low, high = inputs["LThresh"], inputs["HThresh"]
                    start, end = eeg["xmin"], eeg["xmax"]
                    if shifted:
                        shift = (end - start) / 10
                        start, end = start + shift, end - shift
                    _source_allow_empty_dataset(
                        eeglab_backend,
                        "pop_eegthresh",
                        eeg,
                        mode,
                        channels,
                        low,
                        high,
                        start,
                        end,
                        superpose,
                        reject,
                    )


@eeglab_test("unittesting_popfunc/eeg_rejsuperpose/popfunc_eeg_rejsuperpose_wrapperTest.m", "test_pass_empty")
def test_reference_rejsuperpose_empty(eeglab_backend):
    eeg = _source_empty_rejection_eeg(eeglab_backend, 5)
    expected = eeglab_backend("eeg_emptyset")
    expected["reject"] = {"rejglobal": np.zeros((1, 5)), "rejglobalE": np.zeros((2, 5))}
    result = eeglab_backend("eeg_rejsuperpose", eeg, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for field in ("rejglobal", "rejglobalE"):
        assert_matlab_near(result["reject"][field], expected["reject"][field])


@eeglab_test("unittesting_popfunc/eeg_rejsuperpose/popfunc_eeg_rejsuperpose_wrapperTest.m", "test_pass_zero")
def test_reference_rejsuperpose_zero(eeglab_backend):
    eeg = _source_empty_rejection_eeg(eeglab_backend, 5)
    expected = eeglab_backend("eeg_emptyset")
    # MATLAB's first nested assignment turns eeg_emptyset's [] into a struct.
    eeg["reject"], expected["reject"] = {}, {}
    for field in ("rejmanual", "rejfreq"):
        eeg["reject"][field] = np.zeros((1, 5))
        eeg["reject"][field + "E"] = np.zeros((2, 5))
        expected["reject"][field] = np.zeros((1, 5))
        expected["reject"][field + "E"] = np.zeros((2, 5))
    expected["reject"]["rejglobal"] = np.zeros((1, 5))
    expected["reject"]["rejglobalE"] = np.zeros((2, 5))
    result = eeglab_backend("eeg_rejsuperpose", eeg, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    for field in ("rejmanual", "rejmanualE", "rejfreq", "rejfreqE", "rejglobal", "rejglobalE"):
        assert_matlab_near(result["reject"][field], expected["reject"][field])


@eeglab_test("unittesting_popfunc/eeg_rejsuperpose/popfunc_eeg_rejsuperpose_wrapperTest.m", "test_pass_general")
def test_reference_rejsuperpose_general(eeglab_backend):
    eeg = _source_empty_rejection_eeg(eeglab_backend, 5)
    expected = eeglab_backend("eeg_emptyset")
    marks = {
        "rejmanual": np.array([[0, 0, 0, 1, 0]], dtype=float),
        "rejfreq": np.array([[1, 0, 0, 1, 0]], dtype=float),
        "rejmanualE": np.array([[0, 0, 0, 1, 0], [0, 1, 0, 0, 0]], dtype=float),
        "rejfreqE": np.array([[0, 0, 0, 0, 1], [0, 0, 0, 0, 1]], dtype=float),
    }
    eeg["reject"] = copy.deepcopy(marks)
    expected["reject"] = marks
    expected["reject"]["rejglobal"] = np.array([[1, 0, 0, 1, 0]], dtype=float)
    expected["reject"]["rejglobalE"] = np.array([[0, 0, 0, 1, 1], [0, 1, 0, 0, 1]], dtype=float)
    result = eeglab_backend("eeg_rejsuperpose", eeg, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    for field in ("rejmanual", "rejmanualE", "rejfreq", "rejfreqE", "rejglobal", "rejglobalE"):
        assert_matlab_near(result["reject"][field], expected["reject"][field])


def _source_all_rejection_marks(eeg, prefix):
    eeg["reject"] = {}
    for index, field in enumerate(("rejmanual", "rejthresh", "rejconst", "rejjp", "rejkurt", "rejfreq")):
        eeg["reject"][prefix + field] = np.eye(6)[index : index + 1]
        eeg["reject"][prefix + field + "E"] = np.eye(6)[[index, 5 - index]]


@eeglab_test("unittesting_popfunc/eeg_rejsuperpose/popfunc_eeg_rejsuperpose_wrapperTest.m", "test_pass_all")
def test_reference_rejsuperpose_all(eeglab_backend):
    eeg = _source_empty_rejection_eeg(eeglab_backend, 6)
    _source_all_rejection_marks(eeg, "")
    expected = eeglab_backend("eeg_emptyset")
    expected["reject"] = {"rejglobal": np.ones((1, 6)), "rejglobalE": np.ones((2, 6))}
    result = eeglab_backend("eeg_rejsuperpose", eeg, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
    for field in ("rejglobal", "rejglobalE"):
        assert_matlab_near(result["reject"][field], expected["reject"][field])


@eeglab_test("unittesting_popfunc/eeg_rejsuperpose/popfunc_eeg_rejsuperpose_wrapperTest.m", "test_pass_all_ica")
def test_reference_rejsuperpose_all_ica(eeglab_backend):
    eeg = _source_empty_rejection_eeg(eeglab_backend, 6)
    eeg["data"] = np.zeros((2, 100, 6))
    eeg["icachansind"] = np.array([[1.0, 2.0]])
    _source_all_rejection_marks(eeg, "ica")
    expected = eeglab_backend("eeg_emptyset")
    expected["reject"] = {"rejglobal": np.ones((1, 6)), "rejglobalE": np.ones((2, 6))}
    result = eeglab_backend("eeg_rejsuperpose", eeg, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
    for field in ("rejglobal", "rejglobalE"):
        assert_matlab_near(result["reject"][field], expected["reject"][field])


@eeglab_test("unittesting_popfunc/pop_rejepoch/popfunc_pop_rejepoch_wrapperTest.m", "test_test_pop_rejepoch")
def test_reference_pop_rejepoch(eeglab_backend, eeglab_suite_root):
    eeg = _source_epoch_sample(eeglab_backend, eeglab_suite_root)
    rejection = np.ones((80, 1))
    indices = np.array([1, 3, 6, 25, 46, 67, 78]) - 1
    rejection[indices] -= 1
    eeglab_backend("pop_rejepoch", eeg, rejection, 0.0)
    eeglab_backend("pop_rejepoch", eeg, rejection.astype(bool), 0.0)
    eeglab_backend("pop_rejepoch", eeg, np.zeros((80, 1), dtype=bool), 0.0)


@eeglab_test("unittesting_popfunc/pop_rejchan/popfunc_pop_rejchan_wrapperTest.m", "test_test_pop_rejchan")
def test_reference_pop_rejchan(eeglab_backend, eeglab_suite_root):
    eeg = _source_epoch_sample(eeglab_backend, eeglab_suite_root)
    all_channels = np.arange(1.0, 33.0)[None, :]
    selected = np.array([[2.0, 4.0, 5.0]])
    for channels, threshold, measure, norm in (
        (selected, [[5.0]], "kurt", "off"),
        (all_channels, [[5.0]], "kurt", "off"),
        (selected, [[5.0, 5.0, 5.0]], "kurt", "off"),
        (selected, [[5.0]], "prob", "off"),
        (all_channels, [[5.0]], "kurt", "on"),
        (all_channels, [[5.0]], "prob", "off"),
        (all_channels, [[5.0, 1.0]], "kurt", "off"),
        (np.arange(1.0, 5.0)[None, :], [[5.0]], "kurt", "on"),
    ):
        eeglab_backend(
            "pop_rejchan",
            eeg,
            "elec",
            channels,
            "threshold",
            np.array(threshold),
            "measure",
            measure,
            "norm",
            norm,
            nargout=4,
        )


@eeglab_test("unittesting_popfunc/pop_jointprob/popfunc_pop_jointprob_wrapperTest.m", "test_test_pop_jointprob")
@pytest.mark.gui  # Original vistype=1 calls open EEGPrep's browser.
def test_reference_pop_jointprob(eeglab_backend, eeglab_suite_root):
    eeg = _source_epoch_sample(eeglab_backend, eeglab_suite_root)
    all_channels = np.arange(1.0, np.asarray(eeg["nbchan"]).item() + 1)[None, :]
    for flags in ((0, 1, 0), (0, 1, 1), (0, 0, 1), (0, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1), (1, 0, 0)):
        eeglab_backend("pop_jointprob", eeg, 1.0, all_channels, 3.0, 3.0, *map(float, flags), nargout=5)
    for mode, channels in (
        (0.0, np.arange(1.0, np.size(eeg["icachansind"]) + 1)[None, :]),
        (1.0, np.array([[2.0, 4.0, 5.0, 16.0, 32.0]])),
        (0.0, np.array([[2.0, 4.0, 5.0, 16.0, 30.0]])),
    ):
        eeglab_backend("pop_jointprob", eeg, mode, channels, 3.0, 3.0, 0.0, 1.0, 0.0, nargout=5)


@eeglab_test("unittesting_popfunc/pop_rejkurt/popfunc_pop_rejkurt_wrapperTest.m", "test_test_pop_rejkurt")
@pytest.mark.gui  # Original vistype=1 calls open EEGPrep's browser.
def test_reference_pop_rejkurt(eeglab_backend, eeglab_suite_root):
    eeg = _source_epoch_sample(eeglab_backend, eeglab_suite_root)
    for mode, count in ((1.0, 32), (0.0, 30)):
        channels = np.arange(1.0, count + 1)[None, :]
        for flags in ((0, 1, 0), (0, 1, 1), (0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1), (1, 0, 0), (1, 0, 1)):
            eeglab_backend("pop_rejkurt", eeg, mode, channels, 3.0, 3.0, *map(float, flags), nargout=4)
        eeglab_backend("pop_rejkurt", eeg, mode, np.array([[2.0, 4.0, 5.0]]), 3.0, 3.0, 0.0, 1.0, 0.0, nargout=4)


@eeglab_test("unittesting_popfunc/pop_rejtrend/popfunc_pop_rejtrend_wrapperTest.m", "test_test_pop_rejtrend")
def test_reference_pop_rejtrend(eeglab_backend, eeglab_suite_root):
    eeg = _source_epoch_sample(eeglab_backend, eeglab_suite_root)
    for mode, count, min_r in ((1.0, 32, 0.3), (0.0, 30, 0.5)):
        channels = np.arange(1.0, count + 1)[None, :]
        for superpose, reject in ((0.0, 1.0), (0.0, 0.0), (1.0, 1.0), (1.0, 0.0)):
            _source_allow_empty_dataset(
                eeglab_backend,
                "pop_rejtrend",
                eeg,
                mode,
                channels,
                384.0,
                0.5,
                min_r,
                superpose,
                reject,
                0.0,
            )
        for selection, window, slope, minimum in (
            (channels, 384.0, 0.5, 1.0),
            (channels, 384.0, 0.5, 0.5),
            (channels, 384.0, 10.0, min_r),
            (channels, 1000.0, 0.5, min_r),
            (np.array([[2.0, 4.0, 5.0]]), 384.0, 0.5, min_r),
        ):
            _source_allow_empty_dataset(
                eeglab_backend,
                "pop_rejtrend",
                eeg,
                mode,
                selection,
                window,
                slope,
                minimum,
                0.0,
                1.0,
                0.0,
            )


@eeglab_test("unittesting_popfunc/pop_rejspec/popfunc_pop_rejspec_wrapperTest.m", "test_test_pop_rejspec")
def test_reference_pop_rejspec(eeglab_backend, eeglab_suite_root, request):
    eeg = _source_epoch_sample(eeglab_backend, eeglab_suite_root)
    # The native source gates FFT by EEGLAB version and legacy PMTM by toolbox
    # licensing. Python has no MATLAB license API; exercise both workflows there.
    run_fft, run_legacy = True, True
    if request.config.getoption("--eeglab-backend") == "matlab":
        version = eeglab_backend("eeg_getversion")
        engine = request.getfixturevalue("eeglab_matlab_engine")
        run_fft = bool(np.any(np.asarray(engine.str2num(version[:2])) > 9))
        run_legacy = bool(
            engine.eval("license('test', 'signal_toolbox') && license('checkout', 'signal_toolbox') && exist('psd')")
        )
    if run_fft:
        eeglab_backend(
            "pop_rejspec",
            eeg,
            1.0,
            "elecrange",
            np.arange(1.0, 33.0)[None, :],
            "threshold",
            np.array([[-30.0, 30.0]]),
            "freqlimits",
            np.array([[15.0, 30.0]]),
            "eegplotplotallrej",
            0.0,
            "eegplotreject",
            1.0,
            "method",
            "fft",
            nargout=2,
        )
    if run_legacy:
        _source_allow_empty_dataset(
            eeglab_backend,
            "pop_rejspec",
            eeg,
            1.0,
            np.array([[2.0, 4.0, 5.0]]),
            -10.0,
            100.0,
            2.0,
            15.0,
            0.0,
            1.0,
        )
        for mode, count, flags in (
            (1.0, 32, ((0.0, 0.0), (1.0, 1.0), (1.0, 0.0))),
            (0.0, np.size(eeg["icachansind"]), ((0.0, 1.0), (0.0, 0.0), (1.0, 1.0), (1.0, 0.0))),
        ):
            for superpose, reject in flags:
                _source_allow_empty_dataset(
                    eeglab_backend,
                    "pop_rejspec",
                    eeg,
                    mode,
                    np.arange(1.0, count + 1)[None, :],
                    -30.0,
                    30.0,
                    15.0,
                    30.0,
                    superpose,
                    reject,
                )


def _epoched_eeg() -> dict:
    rng = np.random.default_rng(4)
    eeg = create_test_eeg(n_channels=4, n_samples=80, n_trials=5, srate=100)
    data = rng.normal(0, 0.05, (4, 80, 5))
    data[0, 10:20, 1] = 25
    data[1, :, 2] += np.linspace(0, 8, 80)
    data[2, :, 3] += 4 * np.sin(2 * np.pi * 25 * np.arange(80) / 100)
    eeg["data"] = data
    eeg["icaweights"] = np.eye(4)
    eeg["icasphere"] = np.eye(4)
    eeg["icawinv"] = np.eye(4)
    eeg["icachansind"] = np.arange(4)
    eeg["icaact"] = None
    eeg["reject"] = {}
    return eeg


def test_pop_eegthresh_marks_epochs_and_emits_replayable_python():
    eeg = _epoched_eeg()

    out, com = pop_eegthresh(eeg, 1, [1], -10, 10, 0, 0.79, 0, 0, return_com=True)
    component_out, component_rejected = pop_eegthresh(eeg, 0, [1], -10, 10, 0, 0.79, 1, 0)

    assert out["reject"]["rejthresh"].tolist() == [False, True, False, False, False]
    assert out["reject"]["rejthreshE"][0].tolist() == [False, True, False, False, False]
    assert component_rejected == [2]
    assert component_out["reject"]["icarejthresh"].tolist() == [False, True, False, False, False]
    assert _console_python_command(com) == (
        "EEG = pop_eegthresh(EEG, icacomp=1, elecrange=[1], negthresh=[-10], "
        "posthresh=[10], starttime=[0], endtime=[0.79], superpose=0, reject=0)"
    )


def test_eegplot_accepts_epoched_data_after_spectral_rejection_marks():
    # The upstream TODO describes eegplot failing after abnormal-spectrum
    # rejection, but its payload is fully commented and calls unrelated
    # pop_biosig. Preserve the reported workflow as an executable regression.
    eeg = _epoched_eeg()
    marked, rejected = pop_rejspec(
        eeg,
        1,
        "method",
        "fft",
        "elecrange",
        [3],
        "threshold",
        [-10, 10],
        "freqlimits",
        [20, 30],
        "eegplotreject",
        0,
    )

    model = eegplot(marked, show=False)

    assert rejected
    assert marked["reject"]["rejfreq"].any()
    assert model.data.mode == "epoched"
    assert model.data.data.shape == marked["data"].shape
    assert model.data.total_samples == marked["pnts"] * marked["trials"]


def test_reject_on_browser_accept_removes_epochs_without_immediate_rejection(monkeypatch):
    accepted = []

    def fake_eegplot(_data, *args, **kwargs):
        del args
        kwargs["command_callback"](kwargs["winrej"])
        return "window"

    monkeypatch.setattr(eegplot_rejection_module, "eegplot", fake_eegplot)
    eeg = _epoched_eeg()

    out, _command = pop_eegthresh(
        eeg,
        1,
        [1],
        -10,
        10,
        0,
        0.79,
        0,
        1,
        topcommand="reject",
        command_callback=lambda eeg_out, command: accepted.append(eeg_out),
        return_com=True,
    )

    assert out["trials"] == eeg["trials"]
    assert accepted[0]["trials"] == eeg["trials"] - 1


def test_pop_rejcont_display_defers_history_command_until_browser_accept(monkeypatch):
    accepted = []
    eeg = create_test_eeg(n_channels=2, n_samples=120, n_trials=1, srate=100)
    time = np.arange(120) / 100
    eeg["data"][0] = 100 * np.sin(2 * np.pi * 30 * time)

    def fake_eegplot(_data, *args, **kwargs):
        del args
        kwargs["command_callback"](kwargs["winrej"])
        return "window"

    monkeypatch.setattr(pop_rejcont_module, "eegplot", fake_eegplot)

    out, command = pop_rejcont(
        eeg,
        "elecrange",
        [1],
        "freqlimit",
        [20, 40],
        "threshold",
        0,
        "epochlength",
        0.2,
        "contiguous",
        1,
        "eegplot",
        "on",
        command_callback=lambda eeg_out, accept_command: accepted.append((eeg_out, accept_command)),
        return_com=True,
    )

    assert out is eeg
    assert command == ""
    assert accepted[0][1].startswith("EEG = pop_rejcont(EEG, ")


def test_jointprob_global_marks_match_eeglab_trial_rows_for_duplicate_channels():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(3, 12, 4))
    elecrange = [3, 1, 3]

    reject, row_marks, local_scores, global_scores = jointprob_marks(data, elecrange, 0.8, 0.8)

    selected = [2, 0, 2]
    expected_local_scores, expected_local = jointprob(data[selected], 0.8, normalize=1)
    global_data = data[selected].transpose(2, 0, 1).reshape(data.shape[2], -1)
    expected_global_scores, expected_global = jointprob(global_data, 0.8, normalize=1)
    expected_reject = expected_local.any(axis=0) | expected_global.ravel()

    np.testing.assert_allclose(local_scores, expected_local_scores)
    np.testing.assert_allclose(global_scores, expected_global_scores.ravel())
    np.testing.assert_array_equal(reject, expected_reject)
    np.testing.assert_array_equal(row_marks[0], expected_local[1])
    np.testing.assert_array_equal(row_marks[2], expected_local[2])


def test_jointprob_three_dimensional_scores_match_upstream():
    signal = np.empty((3, 4, 2), dtype=float)
    signal[:, :, 0] = [[1, 1, 3, 4], [1, 2, 1, 4], [1, 2, 3, 4]]
    signal[:, :, 1] = [[1, 2, 3, 4], [1, 2, 1, 4], [2, 2, 3, 4]]
    expected = np.asarray(
        [
            [-np.sum(np.log([3 / 8, 3 / 8, 2 / 8, 2 / 8])), -np.sum(np.log([3 / 8, 1 / 8, 2 / 8, 2 / 8]))],
            [-np.sum(np.log([4 / 8, 2 / 8, 4 / 8, 2 / 8])), -np.sum(np.log([4 / 8, 2 / 8, 4 / 8, 2 / 8]))],
            [-np.sum(np.log([1 / 8, 3 / 8, 2 / 8, 2 / 8])), -np.sum(np.log([3 / 8, 3 / 8, 2 / 8, 2 / 8]))],
        ]
    )

    scores, rejected = jointprob(signal)

    np.testing.assert_allclose(scores, expected)
    np.testing.assert_array_equal(rejected, np.zeros((3, 2), dtype=bool))


def test_jointprob_computed_and_precomputed_thresholds_match_upstream():
    signal = np.asarray([[1, 1, 5], [1, 2, 1], [1, 2, 5]])
    expected = np.asarray(
        [
            -np.sum(np.log([2 / 3, 2 / 3, 1 / 3])),
            -np.sum(np.log([2 / 3, 1 / 3, 2 / 3])),
            -np.sum(np.log([1 / 3, 1 / 3, 1 / 3])),
        ]
    )[:, np.newaxis]

    scores, rejected = jointprob(signal, 2)
    reused, reused_rejected = jointprob(signal, 2, expected)

    np.testing.assert_allclose(scores, expected)
    np.testing.assert_allclose(reused, expected)
    np.testing.assert_array_equal(rejected, [[False], [False], [True]])
    np.testing.assert_array_equal(reused_rejected, rejected)


def test_jointprob_uses_matlab_sample_standard_deviation_for_normalization():
    signal = np.asarray([[1, 1, 3], [1, 2, 1], [1, 2, 3]])
    raw, _ = jointprob(signal)

    normalized, rejected = jointprob(signal, 0, normalize=1)
    expected = (raw - raw.mean()) / raw.std(ddof=1)

    np.testing.assert_allclose(normalized, expected)
    np.testing.assert_array_equal(rejected, np.zeros((3, 1), dtype=bool))

    trials = np.stack([signal, signal[:, ::-1]], axis=2)
    raw_3d, _ = jointprob(trials)
    normalized_3d, _ = jointprob(trials, 0, normalize=1)
    expected_3d = raw_3d - raw_3d.mean(axis=1, keepdims=True)
    std_3d = raw_3d.std(axis=1, ddof=1, keepdims=True)
    std_3d[std_3d == 0] = 1
    expected_3d /= std_3d
    assert normalized_3d.shape == (3, 2)
    np.testing.assert_allclose(normalized_3d, expected_3d)


def test_jointprob_global_threshold_can_reject_when_local_threshold_does_not():
    data = np.array(
        [
            [
                [4.0, 3.0],
                [2.0, 1.0],
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 4.0],
                [3.0, 4.0],
                [2.0, 3.0],
                [4.0, 3.0],
                [3.0, 2.0],
                [2.0, 4.0],
                [1.0, 4.0],
                [3.0, 0.0],
            ],
            [
                [1.0, 4.0],
                [2.0, 0.0],
                [3.0, 3.0],
                [4.0, 0.0],
                [0.0, 4.0],
                [0.0, 2.0],
                [0.0, 1.0],
                [2.0, 2.0],
                [2.0, 0.0],
                [0.0, 0.0],
                [0.0, 3.0],
                [2.0, 3.0],
            ],
        ]
    )

    reject, row_marks, _local_scores, global_scores = jointprob_marks(data, [1, 2], 10, 0.5)

    np.testing.assert_array_equal(row_marks.any(axis=0), [False, False])
    np.testing.assert_allclose(global_scores, [1 / np.sqrt(2), -1 / np.sqrt(2)])
    np.testing.assert_array_equal(reject, [True, True])


def test_kurtosis_global_marks_match_eeglab_trial_rows_for_duplicate_channels():
    rng = np.random.default_rng(7)
    data = rng.normal(size=(3, 16, 4))
    elecrange = [2, 1, 2]

    reject, row_marks, local_scores, global_scores = kurtosis_marks(data, elecrange, 0.6, 0.6)

    selected = [1, 0, 1]
    expected_local_scores, expected_local = rejkurt(data[selected], 0.6, normalize=1)
    global_data = data[selected].transpose(2, 0, 1).reshape(data.shape[2], -1)
    expected_global_scores, expected_global = rejkurt(global_data, 0.6, normalize=1)
    expected_reject = expected_local.any(axis=0) | expected_global.ravel()

    np.testing.assert_allclose(local_scores, expected_local_scores)
    np.testing.assert_allclose(global_scores, expected_global_scores.ravel())
    np.testing.assert_array_equal(reject, expected_reject)
    np.testing.assert_array_equal(row_marks[0], expected_local[1])
    np.testing.assert_array_equal(row_marks[1], expected_local[2])


def test_kurtosis_global_threshold_can_reject_when_local_threshold_does_not():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(2, 12, 2))
    data[:, :, 1] *= 0.1

    reject, row_marks, _local_scores, global_scores = kurtosis_marks(data, [1, 2], 10, 0.5)

    np.testing.assert_array_equal(row_marks.any(axis=0), [False, False])
    np.testing.assert_allclose(global_scores, [1 / np.sqrt(2), -1 / np.sqrt(2)])
    np.testing.assert_array_equal(reject, [True, True])


def test_eeg_rejsuperpose_and_pop_rejepoch_remove_marked_epochs():
    eeg = _epoched_eeg()
    eeg["reject"]["rejmanual"] = np.array([False, True, False, False, True])
    eeg["reject"]["rejmanualE"] = np.zeros((4, 5), dtype=bool)
    eeg["reject"]["rejmanualE"][0, 1] = True
    eeg["reject"]["rejmanualE"][1, 4] = True

    marked, com = eeg_rejsuperpose(eeg, 1, 1, 0, 0, 0, 0, 0, 0, return_com=True)

    assert marked["reject"]["rejglobal"].tolist() == [False, True, False, False, True]
    assert marked["reject"]["rejglobalE"].shape == (4, 5)
    removed, reject_com = pop_rejepoch(copy.deepcopy(marked), marked["reject"]["rejglobal"], 0, return_com=True)
    assert removed["trials"] == 3
    assert _console_python_command(com) == "EEG = eeg_rejsuperpose(EEG, 1, 1, 0, 0, 0, 0, 0, 0)"
    assert _console_python_command(reject_com) == "EEG = pop_rejepoch(EEG, tmprej=[2, 5], confirm=0)"


def test_eeg_rejsuperpose_only_crosses_trial_marks_between_data_and_ica_families():
    eeg = _epoched_eeg()
    eeg["icaweights"] = np.ones((2, 4))
    eeg["icawinv"] = np.ones((4, 2))
    eeg["reject"] = {
        "rejmanual": np.array([False, True, False, False, False]),
        "rejmanualE": np.zeros((4, 5), dtype=bool),
        "icarejmanual": np.array([False, False, True, False, False]),
        "icarejmanualE": np.ones((2, 5), dtype=bool),
    }

    marked = eeg_rejsuperpose(eeg, 1, 1, 0, 0, 0, 0, 0, 1)

    assert marked["reject"]["rejglobal"].tolist() == [False, True, True, False, False]
    assert marked["reject"]["rejglobalE"].shape == (4, 5)
    assert not marked["reject"]["rejglobalE"].any()


def test_pop_rejmenu_can_combine_marks_without_browser():
    eeg = _epoched_eeg()
    eeg["reject"]["rejthresh"] = np.array([False, True, False, False, False])
    eeg["reject"]["rejthreshE"] = np.zeros((4, 5), dtype=bool)

    out, com = pop_rejmenu(eeg, 1, gui=False, return_com=True)

    assert out["reject"]["rejglobal"].tolist() == [False, True, False, False, False]
    assert _console_python_command(com) == "EEG = eeg_rejsuperpose(EEG, 1, 1, 1, 1, 1, 1, 1, 1)"


def test_pop_rejchan_current_suite_probability_and_kurtosis_options():
    rng = np.random.default_rng(91)
    eeg = create_test_eeg(n_channels=5, n_samples=40, n_trials=3, srate=100)
    eeg["data"] = rng.normal(size=(5, 40, 3))
    options = (
        ([2, 4, 5], [5], "kurt", "off"),
        ([1, 2, 3, 4, 5], [5], "kurt", "off"),
        ([2, 4, 5], [5, 5, 5], "kurt", "off"),
        ([2, 4, 5], [5], "prob", "off"),
        ([1, 2, 3, 4, 5], [5], "kurt", "on"),
        ([1, 2, 3, 4, 5], [5], "prob", "off"),
        ([1, 2, 3, 4, 5], [5, 1], "kurt", "off"),
        ([1, 2, 3, 4], [5], "kurt", "on"),
    )

    for channels, threshold, measure_name, norm in options:
        out, rejected, measure = pop_rejchan(
            eeg,
            "elec",
            channels,
            "threshold",
            threshold,
            "measure",
            measure_name,
            "norm",
            norm,
            "indexonly",
            "on",
        )
        assert measure.shape == (len(channels),)
        assert np.isfinite(measure).all()
        assert set(rejected).issubset(channels)
        assert out["nbchan"] == eeg["nbchan"]

    removal_eeg = create_test_eeg(n_channels=2, n_samples=20, n_trials=1, srate=100)
    removal_eeg["data"] = np.zeros((2, 20))
    removal_eeg["data"][0, 10] = 100
    removed, rejected, _measure = pop_rejchan(removal_eeg, "measure", "std", "threshold", 5)
    assert rejected == [1]
    assert removed["nbchan"] == 1


def test_rejection_component_threshold_recomputes_stale_stored_icaact():
    eeg = _epoched_eeg()
    eeg["icaweights"] = 2.0 * np.eye(4)
    eeg["icasphere"] = np.eye(4)
    eeg["icaact"] = np.zeros((4, eeg["pnts"], eeg["trials"]))

    out, rejected = pop_eegthresh(eeg, 0, [1], -40, 40, 0, 0.79, 0, 0)

    assert rejected == [2]
    assert out["reject"]["icarejthresh"].tolist() == [False, True, False, False, False]


def test_pop_rejchan_scripted_default_threshold_matches_eeglab():
    eeg = create_test_eeg(n_channels=2, n_samples=20, n_trials=1, srate=100)
    eeg["data"] = np.zeros((2, 20))
    eeg["data"][0, 10] = 100

    _out, rejected_channels, _measure = pop_rejchan(eeg, "measure", "std", "indexonly", "on")

    assert rejected_channels == []


def test_pop_rejcont_history_replays_effectful_mode_and_overlap_options():
    sample = pop_loadset(SAMPLE_DATASET_PATH)

    _out, command = pop_rejcont(
        sample,
        "elecrange",
        [1],
        "freqlimit",
        [20, 40],
        "threshold",
        1e9,
        "epochlength",
        0.5,
        "overlap",
        0.1,
        "mode",
        "mean",
        "onlyreturnselection",
        "on",
        return_com=True,
    )

    assert _console_python_command(command) == (
        "EEG = pop_rejcont(EEG, elecrange=[1], freqlimit=[20, 40], threshold=1000000000, "
        "epochlength=0.5, overlap=0.1, mode='mean', onlyreturnselection='on')"
    )


def test_component_selection_and_viewprops_are_replayable_without_scrolling_browser():
    eeg = _epoched_eeg()

    selected, select_com = pop_selectcomps(eeg, [1, 3], reject=[2], plot=False, return_com=True)
    figures, props_com = pop_viewprops(eeg, 0, [1, 2], plot=False, return_com=True)

    assert selected["reject"]["gcompreject"].tolist() == [False, True, False, False]
    assert figures == []
    assert _console_python_command(select_com) == "EEG = pop_selectcomps(EEG, compnum=[1, 3], reject=[2])"
    assert _console_python_command(props_com) == (
        "pop_viewprops(EEG, typecomp=0, chanorcomp=[1, 2], spec_opt=[], erp_opt=[], scroll_event=1, classifier_name='')"
    )
