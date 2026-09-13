"""Behavioral ports of current EEGLAB administrative data-access tests."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from eegprep import eeg_checkchanlocs, eeg_getdatact
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_saveset import pop_saveset
from tests.eeglab_tests import eeglab_test


def _eeg(*, epoched: bool = False) -> dict:
    data = np.arange(3 * 20, dtype=float).reshape(3, 20)
    pnts = 5 if epoched else 20
    trials = 4 if epoched else 1
    data = data.reshape(3, pnts, trials, order="F") if epoched else data
    weights = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, -0.25], [0.25, 0.0, 1.0]])
    inverse = np.linalg.inv(weights)
    flattened = data.reshape(3, -1, order="F")
    activity = (weights @ flattened).reshape(3, pnts, trials, order="F")
    if not epoched:
        activity = activity[:, :, 0]
    return {
        "data": data,
        "nbchan": 3,
        "pnts": pnts,
        "trials": trials,
        "srate": 100.0,
        "xmin": 0.0,
        "xmax": (pnts - 1) / 100.0,
        "chanlocs": np.asarray(
            [
                {"labels": "EEG Fz", "X": 1.0, "Y": 0.0, "Z": 0.0},
                {"labels": "RDA_Cz", "theta": 0.0, "radius": 0.0},
                {"labels": "'Pz'", "theta": 180.0, "radius": 0.5},
            ],
            dtype=object,
        ),
        "urchanlocs": np.array([], dtype=object),
        "chaninfo": {},
        "event": [],
        "icaweights": weights,
        "icasphere": np.eye(3),
        "icawinv": inverse,
        "icachansind": np.arange(3),
        "icaact": activity,
    }


@eeglab_test(
    "unittesting_adminfunc/eeg_checkchanlocs/adminfunc_eeg_checkchanlocs_wrapperTest.m",
    "test_test_eeg_checkchanlocs",
)
def test_current_eeg_checkchanlocs_normalizes_continuous_epoched_and_extended_locations(caplog):
    continuous = eeg_checkchanlocs(_eeg())
    assert [location["labels"] for location in continuous["chanlocs"]] == ["Fz", "Cz", "Pz"]
    assert all(all(location[field] is not None for field in ("X", "Y", "Z")) for location in continuous["chanlocs"])
    np.testing.assert_allclose(
        [[location[axis] for axis in ("X", "Y", "Z")] for location in continuous["chanlocs"]],
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]],
        atol=1e-12,
    )
    assert continuous["chaninfo"] == {"plotrad": None, "shrink": None, "nosedir": "+X", "nodatchans": []}
    assert "urchan" not in continuous["chanlocs"][0]

    extended = _eeg()
    extended["chanlocs"] = list(extended["chanlocs"]) + [{"labels": "test"}, {"labels": "test"}]
    with caplog.at_level("WARNING"):
        checked = eeg_checkchanlocs(extended)
    assert len(checked["chanlocs"]) == 5
    assert "same label" in caplog.text

    epoched = eeg_checkchanlocs(_eeg(epoched=True))
    assert epoched["data"].shape == (3, 5, 4)
    assert [location["labels"] for location in epoched["chanlocs"]] == ["Fz", "Cz", "Pz"]


@eeglab_test(
    "unittesting_adminfunc/eeg_getdatact/adminfunc_eeg_getdatact_wrapperTest.m",
    "test_test_eeg_getdatact",
)
def test_current_eeg_getdatact_channel_component_trial_and_removal_cases():
    continuous = _eeg()
    np.testing.assert_array_equal(eeg_getdatact(continuous), continuous["data"][:, :, None])
    np.testing.assert_array_equal(
        eeg_getdatact(continuous, "channel", [1, 3], "trialindices", 1, "verbose", "on"),
        continuous["data"][[0, 2], :, None],
    )
    np.testing.assert_array_equal(
        eeg_getdatact(continuous, "channel", [1, 3], "trialindices", 1, "verbose", "off"),
        continuous["data"][[0, 2], :, None],
    )

    epoched = _eeg(epoched=True)
    np.testing.assert_array_equal(eeg_getdatact(epoched), epoched["data"])
    np.testing.assert_array_equal(
        eeg_getdatact(epoched, channel=[1, 3], trialindices=1),
        epoched["data"][[0, 2], :, :1],
    )
    np.testing.assert_array_equal(
        eeg_getdatact(epoched, channel=[1, 3], trialindices="1:4"),
        epoched["data"][[0, 2]],
    )
    np.testing.assert_array_equal(
        eeg_getdatact(epoched, component=[1, 3], trialindices=1), epoched["icaact"][[0, 2], :, :1]
    )

    expected = epoched["data"] - np.einsum(
        "c,st->cst",
        epoched["icawinv"][:, 0],
        epoched["icaact"][0],
    )
    np.testing.assert_allclose(eeg_getdatact(epoched, rmcomps=[1]), expected, atol=1e-12)


def test_eeg_getdatact_computes_uncached_activity_backprojection_and_two_dimensional_shape():
    eeg = _eeg(epoched=True)
    eeg["icaact"] = np.array([])
    components = eeg_getdatact(eeg, component=[1, 2], trialindices=[2, 4])
    expected = (eeg["icaweights"] @ eeg["data"].reshape(3, -1, order="F")).reshape(3, 5, 4, order="F")
    np.testing.assert_allclose(components, expected[[0, 1]][:, :, [1, 3]])

    projection = eeg_getdatact(eeg, component=[1, 2], projchan=[1, 3])
    np.testing.assert_allclose(
        projection,
        np.einsum("ci,ist->cst", eeg["icawinv"][[0, 2], :2], expected[:2]),
        atol=1e-12,
    )
    labeled_projection = eeg_getdatact(eeg, component=[1, 2], projchan=["EEG Fz", "'Pz'"])
    np.testing.assert_allclose(labeled_projection, projection)
    flattened = eeg_getdatact(eeg, channel=[1, 3], reshape="2d")
    np.testing.assert_array_equal(flattened, eeg["data"][[0, 2]].reshape(2, -1, order="F"))


def test_eeg_getdatact_concatenates_datasets_and_reports_continuous_boundaries():
    first = _eeg()
    first["data"] = first["data"][:, :5]
    first["icaact"] = first["icaact"][:, :5]
    first["pnts"] = 5
    first["event"] = [{"type": "boundary", "latency": 3.5}]
    second = deepcopy(first)
    second["data"] = second["data"] + 100
    second["icaact"] = second["icaweights"] @ second["data"]

    combined, boundaries = eeg_getdatact([first, second], return_boundaries=True)
    np.testing.assert_array_equal(combined, np.concatenate([first["data"], second["data"]], axis=1)[:, :, None])
    np.testing.assert_array_equal(boundaries, [3.0, 5.0, 8.0])


def test_eeg_getdatact_rejects_ambiguous_or_out_of_range_selections():
    eeg = _eeg(epoched=True)
    with pytest.raises(ValueError, match="cannot be used together"):
        eeg_getdatact(eeg, component=[1], rmcomps=[2])
    with pytest.raises(ValueError, match="cannot be used together"):
        eeg_getdatact(eeg, component=[1], channel=[2])
    with pytest.raises(IndexError, match="outside"):
        eeg_getdatact(eeg, channel=[4])
    with pytest.raises(ValueError, match="projchan requires"):
        eeg_getdatact(eeg, projchan=[1])
    with pytest.raises(ValueError, match="Continuous and epoched"):
        eeg_getdatact([_eeg(), eeg])


def test_eeg_checkchanlocs_separates_fiducials_and_rotates_nose_direction():
    eeg = _eeg()
    eeg["chanlocs"] = [
        {"labels": "Cz", "type": "EEG", "X": 0.0, "Y": 1.0, "Z": 0.0, "theta": 0.0, "sph_theta": 90.0},
        {"labels": "Nz", "type": "FID", "X": 1.0, "Y": 0.0, "Z": 0.0, "theta": 90.0, "sph_theta": 0.0},
    ]
    eeg["chaninfo"] = {"nosedir": "+Y"}

    checked = eeg_checkchanlocs(eeg)

    assert [location["labels"] for location in checked["chanlocs"]] == ["Cz"]
    assert [location["labels"] for location in checked["chaninfo"]["nodatchans"]] == ["Nz"]
    assert checked["chaninfo"]["nosedir"] == "+X"
    assert checked["chaninfo"]["originalnosedir"] == "+Y"
    np.testing.assert_allclose(
        [checked["chanlocs"][0]["X"], checked["chanlocs"][0]["Y"]],
        [1.0, 0.0],
        atol=1e-12,
    )


def test_eeg_checkchanlocs_lower_form_preserves_datachan_flags_and_meg_defaults():
    locations = [
        {"labels": "MLC11", "type": "MEG", "datachan": 1},
        {"labels": "Explicit", "type": "FID", "datachan": 1},
        {"labels": "Fid", "type": "", "datachan": 0, "urchan": 4},
    ]

    data_locations, info, all_locations = eeg_checkchanlocs(locations)

    assert [location["labels"] for location in data_locations] == ["MLC11", "Explicit"]
    assert [location["labels"] for location in info["nodatchans"]] == ["Fid"]
    assert [location["datachan"] for location in all_locations] == [1, 1, 0]
    assert all("datachan" not in location for location in data_locations)
    assert info["nodatchans"][0]["datachan"] == 0
    assert info["topoplot"] == ["conv", "on", "headrad", 0.3]


def test_eeg_getdatact_reads_fdt_dat_and_info_only_set_storage(tmp_path):
    eeg = _eeg()
    expected = np.asarray(eeg["data"][:, :5], dtype=np.float32)
    metadata = {**eeg, "pnts": 5, "data": "samples.fdt", "filepath": str(tmp_path), "filename": "source.set"}
    expected.reshape(-1, order="F").tofile(tmp_path / "samples.fdt")
    np.testing.assert_array_equal(eeg_getdatact(metadata)[:, :, 0], expected)

    frames = expected.T
    frames.reshape(-1, order="F").tofile(tmp_path / "samples.dat")
    metadata["data"] = "samples.dat"
    np.testing.assert_array_equal(eeg_getdatact(metadata)[:, :, 0], expected)

    saved = {
        **eeg,
        "data": expected,
        "pnts": 5,
        "trials": 1,
        "xmax": 0.04,
        "times": np.arange(5, dtype=float) * 10.0,
        "icaact": np.array([]),
    }
    set_path = tmp_path / "embedded.set"
    pop_saveset(saved, set_path, savemode="onefile")
    info_only = pop_loadset(set_path, loadmode="info")
    np.testing.assert_array_equal(eeg_getdatact(info_only)[:, :, 0], expected)


def test_eeg_getdatact_accepts_cached_components_without_inverse_and_interpolates_target_locations():
    eeg = _eeg(epoched=True)
    expected = np.asarray(eeg["icaact"][[0], :, :1])
    eeg["icasphere"] = np.array([])
    eeg["icawinv"] = np.array([])
    np.testing.assert_array_equal(eeg_getdatact(eeg, component=[1], trialindices=1), expected)

    interpolation_eeg = {
        "data": np.arange(4 * 8, dtype=float).reshape(4, 8),
        "nbchan": 4,
        "pnts": 8,
        "trials": 1,
        "xmin": 0.0,
        "xmax": 0.07,
        "chanlocs": [
            {"labels": "X+", "X": 1.0, "Y": 0.0, "Z": 0.0},
            {"labels": "X-", "X": -1.0, "Y": 0.0, "Z": 0.0},
            {"labels": "Y+", "X": 0.0, "Y": 1.0, "Z": 0.0},
            {"labels": "Y-", "X": 0.0, "Y": -1.0, "Z": 0.0},
        ],
    }
    target = deepcopy(interpolation_eeg["chanlocs"])
    target.append({"labels": "Z+", "X": 0.0, "Y": 0.0, "Z": 1.0})
    interpolated = eeg_getdatact(interpolation_eeg, interp=target)
    assert interpolated.shape == (5, 8, 1)
    np.testing.assert_array_equal(interpolated[:4, :, 0], interpolation_eeg["data"])
    assert np.all(np.isfinite(interpolated[4]))


def test_eeg_getdatact_reads_external_component_activity(tmp_path):
    eeg = _eeg(epoched=True)
    activity = np.asarray(eeg["icaact"], dtype=np.float32)
    rows = np.stack([component.reshape(-1, order="F") for component in activity])
    rows.tofile(tmp_path / "external.icaact")
    eeg["data"] = np.array([])
    eeg["icaact"] = np.array([])
    eeg["filepath"] = str(tmp_path)
    eeg["filename"] = "external.set"

    np.testing.assert_array_equal(eeg_getdatact(eeg, component=[2]), activity[[1]])


def test_eeg_getdatact_rejects_ambiguous_projection_labels():
    eeg = _eeg()
    eeg["chanlocs"][1]["labels"] = "EEG Fz"

    with pytest.raises(ValueError, match="not unique"):
        eeg_getdatact(eeg, component=[1], projchan=["eeg fz"])
