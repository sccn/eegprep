from __future__ import annotations

import copy

import numpy as np
import pytest
from scipy.stats import pearsonr

from eegprep.functions.popfunc.pop_averef import pop_averef
from eegprep.functions.popfunc.pop_findmatchingcomps import pop_findmatchingcomps
from eegprep.functions.popfunc.pop_fusechanrej import pop_fusechanrej
from eegprep.functions.popfunc.pop_icathresh import pop_icathresh
from eegprep.functions.popfunc.pop_rejchanspec import pop_rejchanspec
from eegprep.functions.popfunc.pop_chansel import pop_chansel_resolve
from eegprep.functions.popfunc.pop_topochansel import pop_topochansel
from eegprep.functions.sigprocfunc.eegthresh import eegthresh
from eegprep.functions.sigprocfunc.entropy_rej import entropy_rej
from eegprep.functions.sigprocfunc.ica_helpers import compvar, eeg_getica, eeg_pvaf, icaact, icaproj, icavar
from eegprep.functions.sigprocfunc.kurt import kurt
from eegprep.functions.sigprocfunc.realproba import realproba
from eegprep.functions.sigprocfunc.rejtrend import rejtrend
from tests.eeglab_tests import assert_matlab_near, eeglab_test


SIGPROC_ROOT = "unittesting_sigprocfunc"


@eeglab_test(f"{SIGPROC_ROOT}/fdr/sigprocfunc_fdr_wrapperTest.m", "test_test_fdr")
def test_reference_fdr(eeglab_backend):
    data = np.random.default_rng(1).standard_normal((30, 4))
    data[:, 3] = data.sum(axis=1)
    # This is the source's corrcoef p-value input, not the fdr under test.
    probabilities = pearsonr(data.T[:, None, :], data.T[None, :, :], axis=-1).pvalue
    for arguments in ((), (0.8,), (0.05,), (0.5,), (0.5, "nonParametric")):
        eeglab_backend("fdr", probabilities, *arguments, nargout=2)


@eeglab_test(f"{SIGPROC_ROOT}/kurt/sigprocfunc_kurt_wrapperTest.m", "test_pass_general")
@eeglab_test(f"{SIGPROC_ROOT}/kurt/sigprocfunc_kurt_wrapperTest.m", "test_pass_column")
@eeglab_test(f"{SIGPROC_ROOT}/kurt/sigprocfunc_kurt_wrapperTest.m", "test_pass_bernoulli")
@eeglab_test(f"{SIGPROC_ROOT}/kurt/sigprocfunc_kurt_wrapperTest.m", "test_pass_positive")
def test_reference_kurt(eeglab_backend):
    data = np.arange(1.0, 10.0)[None, :]
    expected = (708 / (225 / 4)) / 9 - 3
    assert_matlab_near(eeglab_backend("kurt", data), [[expected]])
    assert_matlab_near(eeglab_backend("kurt", data.T), [[expected]])
    balanced = np.concatenate((np.zeros(1_000_000), np.ones(1_000_000)))[None, :]
    assert_matlab_near(eeglab_backend("kurt", balanced), [[-2.0]])
    assert_matlab_near(
        eeglab_backend("kurt", np.array([[1, 1, 1, 1, 1, 1, 0, 1, 1, 1]], dtype=float)), [[(0.657 / 0.01) / 10 - 3]]
    )


@eeglab_test(f"{SIGPROC_ROOT}/realproba/sigprocfunc_realproba_wrapperTest.m", "test_pass_general")
@eeglab_test(f"{SIGPROC_ROOT}/realproba/sigprocfunc_realproba_wrapperTest.m", "test_pass_discrete")
def test_reference_realproba(eeglab_backend):
    data = np.array([[1, 2, 3]], dtype=float)
    probabilities, distribution = eeglab_backend("realproba", data, nargout=2)
    assert_matlab_near(probabilities, [[1, 1, 1]])
    assert_matlab_near(distribution, [[1]])
    assert_matlab_near(np.sum(distribution, axis=0, keepdims=True), [[1]])
    probabilities, distribution = eeglab_backend("realproba", data, 10.0, nargout=2)
    assert_matlab_near(probabilities, [[1 / 3, 1 / 3, 1 / 3]])
    assert_matlab_near(distribution, [[1 / 3, 0, 0, 0, 1 / 3, 0, 0, 0, 0, 1 / 3]])
    assert_matlab_near([[np.sum(distribution)]], [[1]])


@eeglab_test(f"{SIGPROC_ROOT}/entropy_rej/sigprocfunc_entropy_rej_wrapperTest.m", "test_pass_1d")
@eeglab_test(f"{SIGPROC_ROOT}/entropy_rej/sigprocfunc_entropy_rej_wrapperTest.m", "test_pass_transposed")
@eeglab_test(f"{SIGPROC_ROOT}/entropy_rej/sigprocfunc_entropy_rej_wrapperTest.m", "test_pass_one_arg")
@eeglab_test(f"{SIGPROC_ROOT}/entropy_rej/sigprocfunc_entropy_rej_wrapperTest.m", "test_pass_reject")
def test_reference_entropy_rej_vectors(eeglab_backend):
    data = np.array([[1, 1, 2]], dtype=float)
    probability = np.array([2 / 3, 2 / 3, 1 / 3])
    expected = -np.sum(probability * np.log(probability))
    for arguments in (
        (data, 3.0, np.empty((0, 0)), 0.0, 1000.0),
        (data.T, 3.0, np.empty((0, 0)), 0.0, 1000.0),
        (data,),
        (data, 3.0, expected, 0.0, 1000.0),
    ):
        entropy, rejected = eeglab_backend("entropy_rej", *arguments, nargout=2)
        assert_matlab_near(entropy, [[expected]])
        assert_matlab_near(rejected, [[0]])


@eeglab_test(f"{SIGPROC_ROOT}/entropy_rej/sigprocfunc_entropy_rej_wrapperTest.m", "test_pass_2d")
@eeglab_test(f"{SIGPROC_ROOT}/entropy_rej/sigprocfunc_entropy_rej_wrapperTest.m", "test_pass_2d_norm")
@eeglab_test(f"{SIGPROC_ROOT}/entropy_rej/sigprocfunc_entropy_rej_wrapperTest.m", "test_pass_3d")
def test_reference_entropy_rej_matrices(eeglab_backend):
    data = np.array([[1, 1, 2], [1, 2, 3]], dtype=float)
    probabilities = np.array([[2 / 3, 2 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]])
    expected = -np.sum(probabilities * np.log(probabilities), axis=1, keepdims=True)
    entropy, rejected = eeglab_backend("entropy_rej", data, 3.0, np.empty((0, 0)), 0.0, 1000.0, nargout=2)
    assert_matlab_near(entropy, expected)
    assert_matlab_near(rejected, np.zeros((2, 1)))
    entropy, rejected = eeglab_backend("entropy_rej", data, 3.0, np.empty((0, 0)), 1.0, 1000.0, nargout=2)
    assert_matlab_near(entropy, [[-np.sqrt(2) / 2], [np.sqrt(2) / 2]])
    assert_matlab_near(rejected, np.zeros((2, 1)))
    assert_matlab_near(np.sum(entropy, axis=0, keepdims=True), [[0]])
    data = np.stack((data, [[2, 1, 1], [2, 1, 3]]), axis=2)
    entropy, rejected = eeglab_backend("entropy_rej", data, 3.0, np.empty((0, 0)), 0.0, 1000.0, nargout=2)
    assert_matlab_near(entropy, np.repeat(expected, 2, axis=1))
    assert_matlab_near(rejected, np.zeros((2, 2)))


@eeglab_test(f"{SIGPROC_ROOT}/entropy_rej/sigprocfunc_entropy_rej_wrapperTest.m", "test_pass_3d_norm")
def test_reference_entropy_rej_normalized_epochs(eeglab_backend):
    data = np.stack(([[1, 1, 2], [2, 1, 1]], [[1, 2, 3], [2, 1, 3]]), axis=2).astype(float)
    entropy, rejected = eeglab_backend("entropy_rej", data, 3.0, np.empty((0, 0)), 1.0, 1000.0, nargout=2)
    assert_matlab_near(entropy, [[np.sqrt(2) / 2, -np.sqrt(2) / 2], [np.sqrt(2) / 2, -np.sqrt(2) / 2]])
    assert_matlab_near(rejected, np.zeros((2, 2)))


def _reference_threshold_data():
    return np.array(
        [
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
            [[1, 4, 7, 10, 13], [2, 5, 8, 11, 14], [3, 6, 9, 12, 15]],
        ],
        dtype=float,
    )


@eeglab_test(f"{SIGPROC_ROOT}/eegthresh/sigprocfunc_eegthresh_wrapperTest.m", "test_pass_general")
@eeglab_test(f"{SIGPROC_ROOT}/eegthresh/sigprocfunc_eegthresh_wrapperTest.m", "test_pass_two_epochs")
def test_reference_eegthresh(eeglab_backend):
    data = _reference_threshold_data()
    accepted, rejected, selected, electrodes = eeglab_backend(
        "eegthresh",
        data,
        3.0,
        np.array([[1, 2]], dtype=float),
        2.0,
        13.0,
        np.array([[1, 15]], dtype=float),
        1.0,
        15.0,
        nargout=4,
    )
    assert_matlab_near(accepted, [[2, 3]])
    assert_matlab_near(rejected, [[1, 4, 5]])
    assert_matlab_near(selected, data[:, :, [1, 2]])
    assert_matlab_near(electrodes, [[1, 1, 1], [1, 0, 1]])
    accepted, rejected, selected, electrodes = eeglab_backend(
        "eegthresh",
        data[:, :, :2],
        3.0,
        np.array([[1, 2]], dtype=float),
        2.0,
        13.0,
        np.array([[1, 6]], dtype=float),
        1.0,
        6.0,
        nargout=4,
    )
    assert_matlab_near(accepted, [[2]])
    assert_matlab_near(rejected, [[1]])
    assert_matlab_near(selected, data[:, :, 1])
    assert_matlab_near(electrodes, [[1], [1]])


@eeglab_test(f"{SIGPROC_ROOT}/eegthresh/sigprocfunc_eegthresh_wrapperTest.m", "test_pass_one_elec")
def test_reference_eegthresh_one_electrode(eeglab_backend):
    data = _reference_threshold_data()
    data[0, 2, 4] = 13
    accepted, rejected, selected, electrodes = eeglab_backend(
        "eegthresh", data, 3.0, 2.0, 2.0, 13.0, np.array([[1, 15]], dtype=float), 1.0, 15.0, nargout=4
    )
    assert_matlab_near(accepted, [[2, 3, 4]])
    assert_matlab_near(rejected, [[1, 5]])
    assert_matlab_near(selected, np.array([[[4, 7, 10], [5, 8, 11], [6, 9, 12]]]))
    assert_matlab_near(electrodes, [[1, 1]])


@eeglab_test(f"{SIGPROC_ROOT}/eegthresh/sigprocfunc_eegthresh_wrapperTest.m", "test_pass_rej_all")
def test_reference_eegthresh_all_rejected(eeglab_backend):
    accepted, rejected, selected, electrodes = eeglab_backend(
        "eegthresh",
        _reference_threshold_data(),
        3.0,
        np.array([[1, 2]], dtype=float),
        2.0,
        11.0,
        np.array([[1, 15]], dtype=float),
        1.0,
        15.0,
        nargout=4,
    )
    assert_matlab_near(accepted, np.empty((1, 0)))
    assert_matlab_near(rejected, [[1, 2, 3, 4, 5]])
    assert_matlab_near(selected, np.empty((0, 0)))
    assert_matlab_near(electrodes, [[1, 1, 1, 1, 1], [1, 0, 0, 1, 1]])


@eeglab_test(f"{SIGPROC_ROOT}/eegthresh/sigprocfunc_eegthresh_wrapperTest.m", "test_pass_rej_nothing")
def test_reference_eegthresh_none_rejected(eeglab_backend):
    data = _reference_threshold_data()
    accepted, rejected, selected, electrodes = eeglab_backend(
        "eegthresh",
        data,
        3.0,
        np.array([[1, 2]], dtype=float),
        1.0,
        15.0,
        np.array([[1, 15]], dtype=float),
        1.0,
        15.0,
        nargout=4,
    )
    assert_matlab_near(accepted, [[1, 2, 3, 4, 5]])
    assert_matlab_near(rejected, np.empty((1, 0)))
    assert_matlab_near(selected, data)
    assert_matlab_near(electrodes, np.empty((2, 0)))


@eeglab_test(f"{SIGPROC_ROOT}/jointprob/sigprocfunc_jointprob_wrapperTest.m", "test_pass_1d_col")
@eeglab_test(f"{SIGPROC_ROOT}/jointprob/sigprocfunc_jointprob_wrapperTest.m", "test_pass_1d_row")
@eeglab_test(f"{SIGPROC_ROOT}/jointprob/sigprocfunc_jointprob_wrapperTest.m", "test_pass_3d")
@eeglab_test(f"{SIGPROC_ROOT}/jointprob/sigprocfunc_jointprob_wrapperTest.m", "test_pass_jp_threshold")
@eeglab_test(f"{SIGPROC_ROOT}/jointprob/sigprocfunc_jointprob_wrapperTest.m", "test_pass_threshold")
@eeglab_test(f"{SIGPROC_ROOT}/jointprob/sigprocfunc_jointprob_wrapperTest.m", "test_pass_normalize_2d")
def test_reference_jointprob(eeglab_backend):
    vector = np.array([[1, 1, 3]], dtype=float)
    for data in (vector, vector.T):
        probability, rejected = eeglab_backend("jointprob", data, nargout=2)
        assert_matlab_near(probability, [[-np.sum(np.log([2 / 3, 2 / 3, 1 / 3]))]])
        assert_matlab_near(rejected, [[0]])
    data = np.stack(
        ([[1, 1, 3, 4], [1, 2, 1, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 1, 4], [2, 2, 3, 4]]), axis=2
    ).astype(float)
    expected = -np.log(
        np.array([[[3, 3, 2, 2], [3, 1, 2, 2]], [[4, 2, 4, 2], [4, 2, 4, 2]], [[1, 3, 2, 2], [3, 3, 2, 2]]]) / 8
    ).sum(axis=2)
    probability, rejected = eeglab_backend("jointprob", data, nargout=2)
    assert_matlab_near(probability, expected)
    assert_matlab_near(rejected, np.zeros((3, 2)))
    data = np.array([[1, 1, 5], [1, 2, 1], [1, 2, 5]], dtype=float)
    expected = -np.log(np.array([[2, 2, 1], [2, 1, 2], [1, 1, 1]]) / 3).sum(axis=1, keepdims=True)
    probability, rejected = eeglab_backend("jointprob", data, 2.0, nargout=2)
    assert_matlab_near(probability, expected)
    assert_matlab_near(rejected, [[0], [0], [1]])
    _probability, rejected = eeglab_backend("jointprob", data, 2.0, expected, nargout=2)
    assert_matlab_near(rejected, [[0], [0], [1]])
    data = np.array([[1, 1, 3], [1, 2, 1], [1, 2, 3]], dtype=float)
    probability, rejected = eeglab_backend("jointprob", data, 0.0, np.empty((0, 0)), 1.0, nargout=2)
    assert_matlab_near(probability, (expected - expected.mean()) / expected.std(ddof=1))
    assert_matlab_near(rejected, np.zeros((3, 1)))


@eeglab_test(f"{SIGPROC_ROOT}/rejkurt/sigprocfunc_rejkurt_wrapperTest.m", "test_test_rejkurt")
def test_reference_rejkurt(eeglab_backend, eeglab_suite_root):
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data_epochs_ica.set"))
    for arguments in ((), (2.0,), (1.5, np.ones((32, 80))), (0.5, np.empty((0, 0)), 1.0), (0.5, np.empty((0, 0)), 2.0)):
        eeglab_backend("rejkurt", eeg["data"], *arguments, nargout=2)
    eeglab_backend("rejkurt", np.random.default_rng(1).random((32, 80, 384)), 0.5, np.empty((0, 0)), 2.0, nargout=2)


@eeglab_test(f"{SIGPROC_ROOT}/rejtrend/sigprocfunc_rejtrend_wrapperTest.m", "test_test_rejtrend")
def test_reference_rejtrend(eeglab_backend, eeglab_suite_root):
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data_epochs_ica.set"))
    for arguments in (
        (384.0, 0.5, 0.3),
        (384.0, 0.5, 1.0),
        (384.0, 0.5, 0.0),
        (384.0, 10.0, 0.3),
        (1000.0, 0.5, 0.3),
        (384.0, 0.5, 0.3, 2.0),
        (384.0, 0.5, 1.0, 3.0),
        (384.0, 0.5, 0.0, 10.0),
        (384.0, 10.0, 0.3, 100.0),
        (1000.0, 0.5, 0.3, 1.0),
    ):
        eeglab_backend("rejtrend", eeg["data"], *arguments)


def _eeg(data: np.ndarray) -> dict:
    return {
        "setname": "phase7",
        "data": data,
        "nbchan": int(data.shape[0]),
        "pnts": int(data.shape[1]),
        "trials": 1 if data.ndim == 2 else int(data.shape[2]),
        "srate": 100.0,
        "xmin": 0.0,
        "xmax": (data.shape[1] - 1) / 100.0,
        "chanlocs": [{"labels": f"Ch{index + 1}"} for index in range(data.shape[0])],
        "event": [],
        "urevent": [],
        "epoch": [],
        "icaweights": np.eye(data.shape[0]),
        "icasphere": np.eye(data.shape[0]),
        "icawinv": np.eye(data.shape[0]),
        "icaact": np.array([]),
        "icachansind": np.arange(data.shape[0]),
        "reject": {},
        "stats": {},
    }


def test_pop_averef_delegates_to_reref_and_keeps_legacy_history():
    eeg = _eeg(np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]))

    out, command = pop_averef(eeg, return_com=True)

    np.testing.assert_allclose(out["data"].mean(axis=0), np.zeros(3), atol=1e-12)
    assert command == "EEG = pop_averef( EEG, 0);"


def test_pop_findmatchingcomps_marks_highly_correlated_component():
    eeg = _eeg(np.zeros((3, 10)))
    eeg["icawinv"] = np.array([[1.0, 0.0, 0.2], [0.0, 1.0, 0.1], [0.0, 0.0, 1.0]])
    match = eeg["icawinv"][:, [1]]

    out, matchic, matchinput = pop_findmatchingcomps(eeg, "matchcomps", match, "corrthresh", 0.99, "rejflag", 1)

    assert matchic == [2]
    assert matchinput == [1]
    np.testing.assert_array_equal(out["reject"]["gcompreject"], [0, 1, 0])


def test_pop_findmatchingcomps_uses_rejected_maps_from_dataset():
    eeg = _eeg(np.zeros((3, 10)))
    eeg["icawinv"] = np.eye(3)
    other = copy.deepcopy(eeg)
    other["reject"] = {"gcompreject": np.array([0, 0, 1])}

    out, matchic, matchinput = pop_findmatchingcomps(eeg, "dataset", other, "corrthresh", 0.99)

    assert out["setname"] == eeg["setname"]
    assert matchic == [3]
    assert matchinput == []


def test_pop_fusechanrej_keeps_common_channels_within_subject_session():
    first = _eeg(np.arange(12, dtype=float).reshape(3, 4))
    second = _eeg(np.arange(12, 24, dtype=float).reshape(3, 4))
    first["subject"] = second["subject"] = "S01"
    first["session"] = second["session"] = 1
    first["chanlocs"] = [{"labels": "A"}, {"labels": "B"}, {"labels": "C"}]
    second["chanlocs"] = [{"labels": "B"}, {"labels": "C"}, {"labels": "D"}]

    out, command = pop_fusechanrej([first, second], return_com=True)

    assert command == "ALLEEG = pop_fusechanrej(ALLEEG);"
    assert [[chan["labels"] for chan in eeg["chanlocs"]] for eeg in out] == [["B", "C"], ["B", "C"]]
    assert [eeg["data"].shape[0] for eeg in out] == [2, 2]


def test_pop_fusechanrej_matches_common_channels_case_insensitively():
    first = _eeg(np.arange(8, dtype=float).reshape(2, 4))
    second = _eeg(np.arange(8, 16, dtype=float).reshape(2, 4))
    first["subject"] = second["subject"] = "S01"
    first["session"] = second["session"] = 1
    first["chanlocs"] = [{"labels": "Fz"}, {"labels": "Cz"}]
    second["chanlocs"] = [{"labels": "fz"}, {"labels": "cz"}]

    out = pop_fusechanrej([first, second])

    assert [[chan["labels"] for chan in eeg["chanlocs"]] for eeg in out] == [["Fz", "Cz"], ["fz", "cz"]]


def test_pop_icathresh_sets_component_rejection_flags():
    eeg = _eeg(np.zeros((3, 10)))
    eeg["stats"] = {
        "compenta": np.array([1.0, 5.0, 6.0]),
        "compkurta": np.array([1.0, 5.0, 2.0]),
        "compkurtdist": np.array([1.0, 2.0, 12.0]),
    }

    out, command = pop_icathresh(eeg, [4, 4, 10], "current", 25, 0, return_com=True)

    np.testing.assert_array_equal(out["reject"]["gcompreject"], [0, 1, 1])
    assert command == "EEG = pop_icathresh(EEG, [4 4 10], 'current', 25, 0);"


@eeglab_test(
    "unittesting_popfunc/pop_rejchanspec/popfunc_pop_rejchanspec_wrapperTest.m",
    "test_test_pop_rejchanspec",
)
def test_pop_rejchanspec_rejects_spectral_outlier_and_returns_history():
    eeg = _eeg(np.zeros((3, 8)))
    specdata = np.array([[1.0, 2.0, 1.0], [1.0, 50.0, 1.0], [1.0, 2.0, 1.0]])
    specfreqs = np.array([10.0, 40.0, 60.0])

    out, command = pop_rejchanspec(
        eeg,
        "elec",
        [1, 2, 3],
        "freqlims",
        [35, 45],
        "absthresh",
        [0, 20],
        "specdata",
        specdata,
        "specfreqs",
        specfreqs,
        return_com=True,
    )

    assert out["nbchan"] == 2
    assert [chan["labels"] for chan in out["chanlocs"]] == ["Ch1", "Ch3"]
    assert command.startswith("EEG = pop_rejchanspec(EEG, ")


def test_pop_topochansel_resolves_indices_and_labels_without_gui():
    chanlocs = [{"labels": "Fz"}, {"labels": "Cz"}, {"labels": "Pz"}]

    chanlist, names, name_text = pop_topochansel(chanlocs, "Cz Pz", gui=False)
    cell_output, selected_names, selected_text, command = pop_topochansel(
        chanlocs, [1, 3], cellstrout="on", gui=False, return_com=True
    )

    assert chanlist == [2, 3]
    assert names == ["Cz", "Pz"]
    assert name_text == "Cz Pz"
    assert cell_output == ["Fz", "Pz"]
    assert selected_names == ["Fz", "Pz"]
    assert selected_text == "Fz Pz"
    assert command.startswith("pop_topochansel(")


def test_pop_topochansel_uses_canonical_chansel_resolver():
    # The non-GUI selection resolution must match pop_chansel's resolver so the
    # two former parsers cannot drift (e.g. on comma-separated labels).
    chanlocs = [{"labels": "Fz"}, {"labels": "Cz"}, {"labels": "Pz"}]
    chanlist, _names, _text = pop_topochansel(chanlocs, "Pz, Fz", gui=False)
    _values, expected = pop_chansel_resolve(chanlocs, "Pz, Fz")
    assert chanlist == expected


def test_ica_helpers_match_simple_projection_identities():
    data = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    weights = np.eye(2)
    eeg = _eeg(data)

    np.testing.assert_allclose(icaact(data, weights), data)
    np.testing.assert_allclose(icaproj(data, weights, [1]), np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(icavar(data, weights, np.eye(2), [1]), [[1.0, 4.0, 9.0]])
    np.testing.assert_allclose(eeg_getica(eeg, [2]), data[[1], :, np.newaxis])
    projected, pvaf = compvar(data, data, np.eye(2), [1])
    np.testing.assert_allclose(projected, data)
    assert pvaf == 100.0
    total_pvaf, channel_pvaf, variances = eeg_pvaf(eeg, [1])
    assert total_pvaf == 100.0
    np.testing.assert_allclose(channel_pvaf, [100.0, 100.0])
    assert variances.shape == (2,)


@eeglab_test("unittesting_popfunc/eeg_getica/popfunc_eeg_getica_wrapperTest.m", "test_test_eeg_getica")
def test_eeg_getica_current_suite_all_single_and_multiple_components():
    data = np.arange(48, dtype=float).reshape(6, 4, 2)
    eeg = _eeg(data)

    all_components = eeg_getica(eeg)
    first_component = eeg_getica(eeg, 1)
    selected_components = eeg_getica(eeg, [5, 6])

    np.testing.assert_array_equal(all_components, data)
    np.testing.assert_array_equal(first_component, data[[0]])
    np.testing.assert_array_equal(selected_components, data[[4, 5]])


def test_eeg_pvaf_maps_full_channel_selection_to_icachansind_subset():
    data = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [100.0, 200.0, 300.0, 400.0],
            [2.0, 4.0, 6.0, 8.0],
        ]
    )
    eeg = _eeg(data)
    eeg["icachansind"] = np.array([0, 2])
    eeg["icaweights"] = np.eye(2)
    eeg["icasphere"] = np.eye(2)
    eeg["icawinv"] = np.eye(2)

    total_pvaf, channel_pvaf, variances = eeg_pvaf(eeg, [2], chans=[3])

    assert total_pvaf == 100.0
    np.testing.assert_allclose(channel_pvaf, [100.0])
    np.testing.assert_allclose(variances, [np.var(data[2])])


def test_kurt_uses_eeglab_sample_standard_deviation_formula():
    values = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    centered = values - values.mean(axis=0, keepdims=True)
    expected = np.sum(centered**4, axis=0) / np.std(values, axis=0, ddof=1) ** 4 / values.shape[0] - 3.0

    np.testing.assert_allclose(kurt(values), expected)
    np.testing.assert_allclose(expected, [-7 / 3, -7 / 3])


def test_python_regression_kurt_row_and_column_vectors_match_upstream_moment_formula():
    values = np.arange(1.0, 10.0)
    expected = (708.0 / (225.0 / 4.0)) / 9.0 - 3.0

    assert kurt(values) == pytest.approx(expected)
    assert kurt(values[:, np.newaxis]) == pytest.approx(expected)


def test_python_regression_kurt_matches_upstream_bernoulli_cases():
    balanced = np.concatenate([np.zeros(10_000), np.ones(10_000)])
    sparse_zero = np.asarray([1, 1, 1, 1, 1, 1, 0, 1, 1, 1], dtype=float)

    for values in (balanced, sparse_zero):
        centered = values - values.mean()
        expected = np.sum(centered**4) / np.std(values, ddof=1) ** 4 / values.size - 3
        assert kurt(values) == pytest.approx(expected)


def test_python_regression_realproba_default_bin_count_matches_eeglab():
    probabilities, distribution = realproba(np.array([1.0, 2.0, 3.0]))

    np.testing.assert_allclose(probabilities, np.ones(3))
    np.testing.assert_allclose(distribution, np.ones(1))


def test_python_regression_realproba_explicit_discretization_matches_upstream_bins():
    probabilities, distribution = realproba(np.array([1.0, 2.0, 3.0]), 10)

    expected = np.zeros(10)
    expected[[0, 4, 9]] = 1 / 3
    np.testing.assert_allclose(probabilities, np.full(3, 1 / 3))
    np.testing.assert_allclose(distribution, expected)


def test_python_regression_realproba_equal_values_have_well_defined_probabilities():
    probabilities, distribution = realproba(np.ones(3), 3)

    np.testing.assert_allclose(probabilities, np.ones(3))
    np.testing.assert_allclose(distribution, np.full(3, 1 / 3))


def test_python_regression_entropy_rej_vector_orientation_and_defaults_match_upstream():
    expected = -np.sum(np.asarray([2 / 3, 2 / 3, 1 / 3]) * np.log([2 / 3, 2 / 3, 1 / 3]))

    for data in (np.asarray([1, 1, 2]), np.asarray([[1], [1], [2]])):
        entropy, rejected = entropy_rej(data)
        np.testing.assert_allclose(entropy, [[expected]])
        np.testing.assert_array_equal(rejected, [[False]])


def test_python_regression_entropy_rej_two_dimensional_scores_and_sample_normalization_match_upstream():
    data = np.asarray([[1, 1, 2], [1, 2, 3]])
    expected = np.asarray(
        [
            -np.sum(np.asarray([2 / 3, 2 / 3, 1 / 3]) * np.log([2 / 3, 2 / 3, 1 / 3])),
            -np.sum(np.full(3, 1 / 3) * np.log(np.full(3, 1 / 3))),
        ]
    )[:, np.newaxis]

    entropy, rejected = entropy_rej(data, 3, None, 0, 1000)
    normalized, normalized_rejected = entropy_rej(data, 3, None, 1, 1000)

    np.testing.assert_allclose(entropy, expected)
    np.testing.assert_allclose(normalized, [[-np.sqrt(2) / 2], [np.sqrt(2) / 2]])
    np.testing.assert_array_equal(rejected, np.zeros((2, 1), dtype=bool))
    np.testing.assert_array_equal(normalized_rejected, np.zeros((2, 1), dtype=bool))


def test_python_regression_entropy_rej_three_dimensional_trials_match_upstream():
    data = np.empty((2, 3, 2), dtype=float)
    data[:, :, 0] = [[1, 1, 2], [1, 2, 3]]
    data[:, :, 1] = [[2, 1, 1], [2, 1, 3]]
    expected_raw = np.asarray(
        [
            -np.sum(np.asarray([2 / 3, 2 / 3, 1 / 3]) * np.log([2 / 3, 2 / 3, 1 / 3])),
            -np.sum(np.full(3, 1 / 3) * np.log(np.full(3, 1 / 3))),
        ]
    )

    raw, _ = entropy_rej(data, 3, None, 0, 1000)
    normalized, rejected = entropy_rej(data, 3, None, 1, 1000)

    np.testing.assert_allclose(raw, [[expected_raw[0], expected_raw[0]], [expected_raw[1], expected_raw[1]]])
    # These synthetic trials have identical entropy per channel.
    np.testing.assert_allclose(normalized, np.zeros((2, 2)))
    np.testing.assert_array_equal(rejected, np.zeros((2, 2), dtype=bool))


def test_python_regression_entropy_rej_precomputed_scores_only_apply_threshold():
    expected = -np.sum(np.asarray([2 / 3, 2 / 3, 1 / 3]) * np.log([2 / 3, 2 / 3, 1 / 3]))

    entropy, rejected = entropy_rej([1, 1, 2], 3, [expected], 0, 1000)

    np.testing.assert_allclose(entropy, [expected])
    np.testing.assert_array_equal(rejected, [False])


def test_python_regression_eegthresh_matches_upstream_selected_and_rejected_trials():
    data = np.empty((2, 3, 5), dtype=float)
    data[0] = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
    data[1] = [[1, 4, 7, 10, 13], [2, 5, 8, 11, 14], [3, 6, 9, 12, 15]]

    accepted, rejected, selected, electrodes = eegthresh(data, 3, [1, 2], 2, 13, [1, 15], 1, 15)

    np.testing.assert_array_equal(accepted, [2, 3])
    np.testing.assert_array_equal(rejected, [1, 4, 5])
    np.testing.assert_array_equal(selected, data[:, :, [1, 2]])
    np.testing.assert_array_equal(electrodes, [[True, True, True], [True, False, True]])

    accepted_two, rejected_two, selected_two, electrodes_two = eegthresh(data[:, :, :2], 3, [1, 2], 2, 13, [1, 6], 1, 6)
    np.testing.assert_array_equal(accepted_two, [2])
    np.testing.assert_array_equal(rejected_two, [1])
    np.testing.assert_array_equal(selected_two, data[:, :, [1]])
    np.testing.assert_array_equal(electrodes_two, [[True], [True]])


def test_python_regression_eegthresh_preserves_all_channels_when_testing_one_electrode():
    data = np.empty((2, 3, 5), dtype=float)
    data[0] = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 13]]
    data[1] = [[1, 4, 7, 10, 13], [2, 5, 8, 11, 14], [3, 6, 9, 12, 15]]

    accepted, rejected, selected, electrodes = eegthresh(data, 3, [2], 2, 13, [1, 15], 1, 15)

    np.testing.assert_array_equal(accepted, [2, 3, 4])
    np.testing.assert_array_equal(rejected, [1, 5])
    np.testing.assert_array_equal(selected, data[:, :, [1, 2, 3]])
    np.testing.assert_array_equal(electrodes, [[True, True]])


def test_python_regression_eegthresh_handles_all_and_no_rejections():
    data = np.empty((2, 3, 5), dtype=float)
    data[0] = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
    data[1] = [[1, 4, 7, 10, 13], [2, 5, 8, 11, 14], [3, 6, 9, 12, 15]]

    accepted, rejected, selected, _ = eegthresh(data, 3, [1, 2], 2, 11, [1, 15], 1, 15)
    assert accepted.size == 0
    np.testing.assert_array_equal(rejected, [1, 2, 3, 4, 5])
    assert selected.shape == (2, 3, 0)

    accepted, rejected, selected, electrodes = eegthresh(data, 3, [1, 2], 1, 15, [1, 15], 1, 15)
    np.testing.assert_array_equal(accepted, [1, 2, 3, 4, 5])
    assert rejected.size == 0
    np.testing.assert_array_equal(selected, data)
    assert electrodes.shape == (2, 0)


def test_python_regression_eegthresh_accepts_continuous_and_single_epoch_shapes():
    data = np.arange(12, dtype=float).reshape(2, 6)

    continuous = eegthresh(data, 3, [1], -1, 20, [1, 6], 1, 6)
    one_epoch = eegthresh(data[:, :, np.newaxis], 6, [1], -1, 20, [1, 6], 1, 6)

    assert continuous[2].shape == (2, 6)
    assert one_epoch[2].shape == (2, 6, 1)


def test_rejection_helper_compatibility_outputs_are_eeglab_facing():
    signal = np.array([[[0.0, 0.0], [0.5, 2.0], [0.0, 0.0]]])

    accepted, rejected, newsignal, elec = eegthresh(signal, 3, [1], [-1], [1], [0, 1], [0], [1])
    trend_reject, trend_rows = rejtrend(np.repeat(signal, 2, axis=1), 3, 0.1, 0.1)
    probabilities, distribution = realproba(np.array([0.0, 0.0, 1.0, 1.0]), 2)

    np.testing.assert_array_equal(accepted, [1])
    np.testing.assert_array_equal(rejected, [2])
    assert newsignal.shape == (1, 3, 1)
    np.testing.assert_array_equal(elec, [[True]])
    assert trend_reject.shape == (2,)
    assert trend_rows.shape == (1, 2)
    np.testing.assert_allclose(probabilities, [0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(distribution, [0.5, 0.5])


def test_python_regression_rejtrend_upstream_parameter_combinations_preserve_trial_contract():
    rng = np.random.default_rng(14)
    signal = rng.normal(size=(6, 1000, 9))
    signal[0, :, 2] += np.linspace(0, 20, 1000)

    for pointrange, maxslope, min_r, step in (
        (384, 0.5, 0.3, None),
        (384, 0.5, 1, None),
        (384, 0.5, 0, None),
        (384, 10, 0.3, None),
        (1000, 0.5, 0.3, None),
        (384, 0.5, 0.3, 2),
        (384, 0.5, 1, 3),
        (384, 0.5, 0, 10),
        (384, 10, 0.3, 100),
        (1000, 0.5, 0.3, 1),
    ):
        rejected, row_marks = rejtrend(signal, pointrange, maxslope, min_r, step)
        assert rejected.shape == (9,)
        assert row_marks.shape == (6, 9)
        np.testing.assert_array_equal(rejected, row_marks.any(axis=0))
