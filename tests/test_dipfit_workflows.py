from __future__ import annotations

from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from eegprep.functions.adminfunc.console import _console_python_command
from eegprep.plugins.dipfit._mri import dipfit_mri_slice_indices, dipfit_mri_slices, load_standard_mri_volume
from eegprep.plugins.dipfit._coordinates import (
    electroderealign,
    headcoordinates,
    homogenous2traditional,
    mni2tal,
    traditionaldipfit,
    warp_apply,
)
from eegprep.plugins.dipfit._fitting import leadfield_matrix, prepare_forward_data
from eegprep.plugins.dipfit.load_afni_atlas import load_afni_atlas
from eegprep.plugins.dipfit.dipfit_reject import dipfit_reject
from eegprep.plugins.dipfit.pop_dipfit_batch import pop_dipfit_batch
from eegprep.plugins.dipfit.pop_dipfit_gridsearch import pop_dipfit_gridsearch
from eegprep.plugins.dipfit.pop_dipfit_manual import pop_dipfit_manual
from eegprep.plugins.dipfit.pop_dipfit_nonlinear import pop_dipfit_nonlinear
from eegprep.plugins.dipfit.pop_dipfit_settings import pop_dipfit_settings
from eegprep.plugins.dipfit.pop_dipplot import pop_dipplot
from eegprep.plugins.dipfit.pop_leadfield import pop_leadfield
from eegprep.plugins.dipfit.pop_multifit import pop_multifit
from tests.fixtures import create_test_eeg_with_ica


def _ica_eeg() -> dict:
    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=80, n_components=4)
    eeg["dipfit"] = {}
    return eeg


def _configured_ica_eeg() -> dict:
    eeg, _command = pop_dipfit_settings(_ica_eeg(), model="standardBEM", return_com=True)
    eeg["dipfit"]["model"] = [
        {"posxyz": [0, -20, 40], "momxyz": [1, 0, 0], "rv": 0.12, "component": 1},
        {"posxyz": [25, 10, 35], "momxyz": [0, 1, 0], "rv": 0.2, "component": 2},
    ]
    return eeg


def _known_dipole_eeg() -> tuple[dict, np.ndarray, np.ndarray]:
    eeg = create_test_eeg_with_ica(n_channels=18, n_samples=80, n_components=1)
    phi = np.linspace(0.35, np.pi - 0.35, 3)
    theta = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    points = []
    for polar in phi:
        for azimuth in theta:
            points.append(
                [
                    85.0 * np.sin(polar) * np.cos(azimuth),
                    85.0 * np.sin(polar) * np.sin(azimuth),
                    85.0 * np.cos(polar),
                ]
            )
    points = np.asarray(points, dtype=float)
    eeg["chanlocs"] = [
        {
            "labels": f"E{index + 1}",
            "type": "EEG",
            "X": float(point[0]),
            "Y": float(point[1]),
            "Z": float(point[2]),
        }
        for index, point in enumerate(points)
    ]
    true_pos = np.asarray([10.0, -20.0, 40.0])
    true_mom = np.asarray([0.6, -0.2, 0.8])
    topography = leadfield_matrix(points, true_pos)[0] @ true_mom
    eeg["icawinv"] = topography[:, np.newaxis]
    eeg["icaweights"] = np.linalg.pinv(eeg["icawinv"])
    eeg["icasphere"] = np.eye(eeg["nbchan"])
    eeg["icachansind"] = np.arange(eeg["nbchan"])
    eeg, _com = pop_dipfit_settings(eeg, model="standardBESA", return_com=True)
    return eeg, true_pos, true_mom


def test_pop_dipfit_settings_stores_standard_model_and_python_history():
    eeg = _ica_eeg()

    out, com = pop_dipfit_settings(eeg, model="standardBEM", chanomit=[2], return_com=True)

    assert out is not eeg
    assert out["dipfit"]["coordformat"] == "MNI"
    assert out["dipfit"]["hdmfile"].endswith("standard_vol.mat")
    assert out["dipfit"]["chansel"] == [1, 3, 4]
    assert _console_python_command(com) == "EEG = pop_dipfit_settings(EEG, model='standardBEM', chanomit=[2])"


def test_packaged_standard_mri_volume_supports_dipfit_viewprops_slices():
    volume = load_standard_mri_volume()

    assert volume.anatomy.shape == (181, 217, 181)
    assert volume.anatomy.dtype.name == "uint8"
    assert volume.transform.shape == (4, 4)
    assert dipfit_mri_slice_indices(volume, [[0, -20, 40]]) == (87, 108, 109)
    slices = dipfit_mri_slices(volume, [[0, -20, 40]])
    assert len(slices) == 3
    assert slices[0].image.shape == (181, 217)
    assert slices[1].image.shape == (181, 181)
    assert slices[2].image.shape == (217, 181)
    assert slices[0].extent == (-125.0, 91.0, -72.0, 108.0)


def test_pop_dipfit_settings_gui_uses_first_dataset_and_applies_to_all():
    first = _ica_eeg()
    second = _ica_eeg()
    second["setname"] = "second"

    with mock.patch(
        "eegprep.plugins.dipfit.pop_dipfit_settings.inputgui",
        return_value={
            "model": 2,
            "coordformat": 2,
            "hdmfile": "standard_BEM/standard_vol.mat",
            "mrifile": "standard_BEM/standard_mri.mat",
            "chanfile": "standard_BEM/elec/standard_1005.elc",
            "coord_transform": "0 0 0 0 0 -1.5708 1 1 1",
            "no_coreg": False,
            "chanomit": "2",
        },
    ) as inputgui:
        out, com = pop_dipfit_settings([first, second], gui=True, return_com=True)

    inputgui.assert_called_once()
    assert inputgui.call_args.args[0].function_name == "pop_dipfit_settings"
    assert [dataset["setname"] for dataset in out] == [first["setname"], "second"]
    assert [dataset["dipfit"]["chansel"] for dataset in out] == [[1, 3, 4], [1, 3, 4]]
    assert "chanomit=[2]" in com


def test_dipfit_native_gridsearch_and_nonlinear_fit_known_spherical_source():
    eeg, true_pos, _true_mom = _known_dipole_eeg()

    coarse, com = pop_dipfit_gridsearch(
        eeg,
        [1],
        [true_pos[0]],
        [true_pos[1]],
        [true_pos[2]],
        100,
        gui=False,
        return_com=True,
    )

    model = coarse["dipfit"]["model"][0]
    np.testing.assert_allclose(model["posxyz"], [true_pos], atol=1e-9)
    assert model["rv"] < 1e-10
    assert _console_python_command(com).startswith("EEG = pop_dipfit_gridsearch(EEG, select=[1]")

    coarse["dipfit"]["model"][0]["posxyz"] = [[5.0, -15.0, 35.0]]
    refined, com = pop_dipfit_nonlinear(coarse, component=1, gui=False, return_com=True)

    assert refined["dipfit"]["model"][0]["rv"] < 0.02
    assert np.linalg.norm(np.asarray(refined["dipfit"]["model"][0]["posxyz"])[0]) < 85.0
    assert _console_python_command(com) == "EEG = pop_dipfit_nonlinear(EEG, component=1, nonlinear='yes')"


def test_dipfit_fitting_aligns_ica_subset_maps_with_coordinate_chansel_superset():
    eeg, true_pos, _true_mom = _known_dipole_eeg()
    eeg["chanlocs"].extend(
        [
            {"labels": "EOG1", "type": "EOG", "X": 95.0, "Y": 0.0, "Z": 0.0},
            {"labels": "M1", "type": "REF", "X": -95.0, "Y": 0.0, "Z": 0.0},
        ]
    )
    eeg["data"] = np.vstack([eeg["data"], np.zeros((2, eeg["pnts"]))])
    eeg["nbchan"] = len(eeg["chanlocs"])
    eeg["dipfit"]["chansel"] = list(range(1, eeg["nbchan"] + 1))

    forward = prepare_forward_data(eeg, [1])

    assert forward.maps.shape == (18, 1)
    assert forward.positions.shape == (18, 3)
    assert forward.chansel == list(range(1, 19))

    coarse, _com = pop_dipfit_gridsearch(
        eeg,
        [1],
        [true_pos[0]],
        [true_pos[1]],
        [true_pos[2]],
        100,
        gui=False,
        return_com=True,
    )
    assert coarse["dipfit"]["model"][0]["rv"] < 1e-10

    coarse["dipfit"]["model"][0]["posxyz"] = [[5.0, -15.0, 35.0]]
    refined = pop_dipfit_nonlinear(coarse, component=1, gui=False)
    assert refined["dipfit"]["model"][0]["rv"] < 0.02


def test_pop_multifit_batch_manual_and_leadfield_use_native_backend():
    eeg, true_pos, _true_mom = _known_dipole_eeg()

    fitted, com = pop_multifit(eeg, [1], "threshold", 100, return_com=True)

    assert fitted["dipfit"]["model"][0]["rv"] < 0.05
    assert "pop_multifit" in com

    batched, batch_com = pop_dipfit_batch(eeg, [1], [true_pos[0]], [true_pos[1]], [true_pos[2]], 100, return_com=True)
    assert batched["dipfit"]["model"][0]["rv"] < 1e-10
    assert "pop_dipfit_gridsearch" in batch_com

    manual, manual_com = pop_dipfit_manual(batched, component=1, gui=False, return_com=True)
    assert manual["dipfit"]["model"][0]["rv"] < 0.02
    assert "pop_dipfit_nonlinear" in manual_com

    leadfield, leadfield_com = pop_leadfield(eeg, sourcemodel={"pos": [[0, 0, 30], [10, -20, 40]]}, return_com=True)
    assert np.asarray(leadfield["dipfit"]["sourcemodel"]["leadfield"][0]).shape == (18, 3)
    assert "pop_leadfield" in leadfield_com


def test_dipfit_reject_matches_eeglab_empty_model_contract():
    models = [
        {"posxyz": [1, 2, 3], "momxyz": [1, 0, 0], "rv": 0.2, "component": 1},
        {"posxyz": [4, 5, 6], "momxyz": [0, 1, 0], "rv": 0.8, "component": 2},
    ]

    out = dipfit_reject(models, 0.4)

    assert out[0]["posxyz"] == [1, 2, 3]
    assert out[1]["posxyz"] == []
    assert out[1]["momxyz"] == []
    assert out[1]["rv"] == 1.0
    assert out[1]["component"] == 2


def test_dipfit_coordinate_transform_and_realign_helpers_are_deterministic():
    np.testing.assert_allclose(mni2tal([[10, 12, 14]]), [[9.9, 12.270032, 12.282544]], atol=1e-6)
    np.testing.assert_allclose(
        headcoordinates([1, 0, 0], [0, 1, 0], [0, -1, 0]),
        np.eye(4),
        atol=1e-12,
    )
    transform = traditionaldipfit([1, 2, 3, 0, 0, 0, 2, 3, 4])
    np.testing.assert_allclose(warp_apply(transform, [[1, 1, 1]], "homogeneous"), [[3, 5, 7]])

    template = {
        "label": ["nasion", "lpa", "rpa", "cz"],
        "pnt": np.asarray([[1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]], dtype=float),
    }
    shifted = {"label": template["label"], "pnt": template["pnt"] + np.asarray([3, -2, 4])}
    aligned = electroderealign({"method": "realignfiducial", "elec": shifted, "template": template})
    np.testing.assert_allclose(aligned["pnt"], template["pnt"], atol=1e-10)


def test_homogenous2traditional_roundtrips_nonzero_rotation_and_scale():
    params = np.asarray([1.5, -2.0, 3.25, 0.2, -0.35, 0.45, 1.1, 0.8, 1.4], dtype=float)
    transform = traditionaldipfit(params)

    recovered = homogenous2traditional(transform)

    np.testing.assert_allclose(recovered, params, atol=1e-12)
    np.testing.assert_allclose(traditionaldipfit(recovered), transform, atol=1e-12)


def test_load_afni_atlas_uses_nibabel_zero_based_voxel_affine(tmp_path):
    nib = pytest.importorskip("nibabel")
    data = np.zeros((3, 3, 3), dtype=np.int16)
    data[1, 2, 0] = 7
    affine = np.asarray(
        [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 2.0, 0.0, 20.0],
            [0.0, 0.0, 3.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    atlas_path = tmp_path / "atlas.nii"
    nib.save(nib.Nifti1Image(data, affine), atlas_path)

    _atlas, xyz, labels, _labelsstr = load_afni_atlas(atlas_path, downsample=1)

    np.testing.assert_allclose(xyz, [[11.0, 24.0, 30.0]])
    np.testing.assert_array_equal(labels, [7])


def test_pop_dipplot_defaults_to_localized_components_when_some_models_are_empty():
    eeg = _configured_ica_eeg()
    eeg["dipfit"]["model"].append({"posxyz": [], "momxyz": [], "rv": 1.0, "component": 3})

    figures, com = pop_dipplot(eeg, "", gui=False, plot=True, return_com=True)

    assert len(figures) == 1
    assert _console_python_command(com) == "pop_dipplot(EEG, comps=[1, 2])"
    plt.close(figures[0])
