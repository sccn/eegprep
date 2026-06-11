from __future__ import annotations

import copy

import numpy as np

from eegprep.functions.popfunc.pop_averef import pop_averef
from eegprep.functions.popfunc.pop_findmatchingcomps import pop_findmatchingcomps
from eegprep.functions.popfunc.pop_fusechanrej import pop_fusechanrej
from eegprep.functions.popfunc.pop_icathresh import pop_icathresh
from eegprep.functions.popfunc.pop_rejchanspec import pop_rejchanspec
from eegprep.functions.popfunc.pop_topochansel import pop_topochansel
from eegprep.functions.sigprocfunc.eegthresh import eegthresh
from eegprep.functions.sigprocfunc.ica_helpers import compvar, eeg_getica, eeg_pvaf, icaact, icaproj, icavar
from eegprep.functions.sigprocfunc.kurt import kurt
from eegprep.functions.sigprocfunc.realproba import realproba
from eegprep.functions.sigprocfunc.rejtrend import rejtrend


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


def test_kurt_uses_eeglab_population_moment_formula():
    values = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    centered = values - values.mean(axis=0, keepdims=True)
    expected = np.mean(centered**4, axis=0) / np.mean(centered**2, axis=0) ** 2 - 3.0

    np.testing.assert_allclose(kurt(values), expected)
    np.testing.assert_allclose(expected, [-1.5, -1.5])


def test_realproba_default_bin_count_matches_eeglab():
    probabilities, distribution = realproba(np.array([0.0, 1.0]))

    np.testing.assert_allclose(probabilities, [0.5, 0.5])
    assert distribution.shape == (1000,)


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
