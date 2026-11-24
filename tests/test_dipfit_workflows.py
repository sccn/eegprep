from __future__ import annotations

import copy
import os
from unittest import mock

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pytest

from eegprep.functions.adminfunc.console import _console_python_command
from eegprep.plugins.dipfit._mri import (
    dipfit_mri_slice_indices,
    dipfit_mri_slices,
    load_besa_mri_volume,
    load_standard_mri_volume,
)
from eegprep.plugins.dipfit._coordinates import (
    electroderealign,
    headcoordinates,
    homogenous2traditional,
    mni2tal,
    traditionaldipfit,
    warp_apply,
)
from eegprep.plugins.dipfit.eeglab2fieldtrip import eeglab2fieldtrip
from eegprep.plugins.dipfit.dipfit_gridsearch import fitting_config
from eegprep.plugins.dipfit._utils import DIPFITUnavailableError, dipfit_history, normalize_model_list
from eegprep.plugins.dipfit.ft_dipolefitting import (
    compute_dipole_leadfield as ft_compute_leadfield,
    standard_besa_headmodel,
    prepare_headmodel_sens,
)
from eegprep.plugins.dipfit.load_afni_atlas import load_afni_atlas
from eegprep.plugins.dipfit.dipfit_reject import dipfit_reject
from eegprep.plugins.dipfit.pop_dipfit_batch import pop_dipfit_batch
from eegprep.plugins.dipfit.pop_dipfit_gridsearch import pop_dipfit_gridsearch
from eegprep.plugins.dipfit.pop_dipfit_loreta import pop_dipfit_loreta
from eegprep.plugins.dipfit.pop_dipfit_manual import pop_dipfit_manual
from eegprep.plugins.dipfit.pop_dipfit_nonlinear import (
    pop_dipfit_nonlinear,
    pop_dipfit_nonlinear_dialog_spec,
)
from eegprep.plugins.dipfit.pop_dipfit_settings import pop_dipfit_settings, pop_dipfit_settings_dialog_spec
from eegprep.plugins.dipfit.pop_dipplot import pop_dipplot, pop_dipplot_dialog_spec
from eegprep.plugins.dipfit.pop_leadfield import pop_leadfield, pop_leadfield_dialog_spec
from eegprep.plugins.dipfit.pop_multifit import pop_multifit
from tests.fixtures import create_test_eeg, create_test_eeg_with_ica


def _ica_eeg() -> dict:
    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=80, n_components=4)
    eeg["dipfit"] = {}
    return eeg


def _configured_ica_eeg() -> dict:
    eeg, _command = pop_dipfit_settings(_ica_eeg(), model="standardBEM", coord_transform=[], return_com=True)
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
    topography = ft_compute_leadfield(true_pos, {"elecpos": points}, standard_besa_headmodel()) @ true_mom
    eeg["icawinv"] = topography[:, np.newaxis]
    eeg["icaweights"] = np.linalg.pinv(eeg["icawinv"])
    eeg["icasphere"] = np.eye(eeg["nbchan"])
    eeg["icachansind"] = np.arange(eeg["nbchan"])
    eeg, _com = pop_dipfit_settings(eeg, model="standardBESA", coord_transform=[], return_com=True)
    return eeg, true_pos, true_mom


def test_pop_dipfit_settings_stores_standard_model_and_python_history():
    eeg = _ica_eeg()

    out, com = pop_dipfit_settings(eeg, model="standardBEM", chanomit=[2], coord_transform=[], return_com=True)

    assert out is not eeg
    assert out["dipfit"]["coordformat"] == "MNI"
    assert out["dipfit"]["hdmfile"].endswith("standard_vol.mat")
    assert out["dipfit"]["chansel"] == [1, 3, 4]
    assert _console_python_command(com) == "EEG = pop_dipfit_settings(EEG, model='standardBEM', chanomit=[2])"


def test_packaged_standard_mri_volume_supports_dipfit_viewprops_slices():
    volume = load_standard_mri_volume()

    assert volume["anatomy"].shape == (181, 217, 181)
    assert volume["anatomy"].dtype.name == "uint8"
    assert volume["transform"].shape == (4, 4)
    assert dipfit_mri_slice_indices(volume, [[0, -20, 40]]) == (87, 108, 109)
    slices = dipfit_mri_slices(volume, [[0, -20, 40]])
    assert len(slices) == 3
    assert slices[0]["image"].shape == (181, 217)
    assert slices[1]["image"].shape == (181, 181)
    assert slices[2]["image"].shape == (217, 181)
    assert slices[0]["extent"] == (-125.0, 91.0, -72.0, 108.0)


def test_standard_mri_slice_extents_match_the_world_bounds():
    volume = load_standard_mri_volume()
    sagittal, coronal, axial = dipfit_mri_slices(volume, [[0.0, 0.0, 0.0]])
    assert sagittal["extent"] == (-125.0, 91.0, -72.0, 108.0)
    assert coronal["extent"] == (-90.0, 90.0, -72.0, 108.0)
    assert axial["extent"] == (-90.0, 90.0, -125.0, 91.0)
    _assert_three_voxel_shift_follows_the_millimetre_step(volume)


def test_avg152_slice_extents_match_the_world_bounds():
    volume = load_besa_mri_volume()
    sagittal, coronal, axial = dipfit_mri_slices(volume, [[0.0, 0.0, 0.0]])
    assert sagittal["extent"] == (-126.0, 90.0, -72.0, 108.0)
    assert coronal["extent"] == (-90.0, 90.0, -72.0, 108.0)
    assert axial["extent"] == (-90.0, 90.0, -126.0, 90.0)
    _assert_three_voxel_shift_follows_the_millimetre_step(volume)


def _assert_three_voxel_shift_follows_the_millimetre_step(volume) -> None:
    scale = float(volume["transform"][0, 0])
    origin = float(volume["transform"][0, 3])
    world = origin + scale * 40.0
    before = dipfit_mri_slice_indices(volume, [[world, 0.0, 0.0]])
    after = dipfit_mri_slice_indices(volume, [[world + 3.0 * scale, 0.0, 0.0]])
    assert after[0] - before[0] == 3


def test_mri_slices_keep_the_component_3_anatomy_upright():
    """Air above the vertex belongs at the top of an origin='lower' image."""
    volume = load_standard_mri_volume()
    slices = dipfit_mri_slices(volume, [[-0.05, 61.6, -41.1]])
    sagittal, coronal, axial = (item["image"] for item in slices)

    assert coronal[-20:].mean() < 1.0
    assert coronal[:20].mean() > 30.0
    assert sagittal[-15:].mean() < sagittal[:15].mean()
    anterior = max(_tissue_span(axial[row]) for row in range(205, 212))
    posterior = max(_tissue_span(axial[row]) for row in range(25, 36))
    assert anterior < 30
    assert posterior > 80


def _tissue_span(row) -> int:
    columns = np.flatnonzero(np.asarray(row) > 20)
    if columns.size == 0:
        return 0
    return int(columns[-1] - columns[0])


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
    assert "'chanomit', [2]" in com


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


def test_fine_fit_dialog_reads_nested_autofit_coordinates():
    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=32, n_components=1)
    eeg["dipfit"] = {
        "model": [{"posxyz": [[-6.0, -21.0, 48.0]], "momxyz": [[0.1, 0.2, 0.3]], "rv": 0.026, "component": 1}]
    }

    spec = pop_dipfit_nonlinear_dialog_spec(eeg)
    values = {control.tag: control.value for control in spec.controls if control.tag}
    labels = {control.tag: control.string for control in spec.controls if control.tag}

    assert values["dip1pos"] == "-6 -21 48"
    assert values["dip1mom"] == "0.1 0.2 0.3"
    assert labels["relvar"] == "2.6%"


def test_mni_nonlinear_fit_uses_the_packaged_bem():
    eeg = create_test_eeg_with_ica(n_channels=18, n_samples=40, n_components=1)
    phi = np.linspace(0.35, np.pi - 0.35, 3)
    theta = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    points = [
        [85.0 * np.sin(polar) * np.cos(azimuth), 85.0 * np.sin(polar) * np.sin(azimuth), 85.0 * np.cos(polar)]
        for polar in phi
        for azimuth in theta
    ]
    eeg["chanlocs"] = [
        {"labels": f"E{index + 1}", "type": "EEG", "X": float(point[0]), "Y": float(point[1]), "Z": float(point[2])}
        for index, point in enumerate(points)
    ]
    topography = np.zeros(len(points))
    topography[0] = 1.0
    eeg["icawinv"] = topography[:, np.newaxis]
    eeg["icaweights"] = np.linalg.pinv(eeg["icawinv"])
    eeg["icasphere"] = np.eye(eeg["nbchan"])
    eeg["icachansind"] = np.arange(eeg["nbchan"])
    eeg, _com = pop_dipfit_settings(eeg, model="standardBEM", coord_transform=[], return_com=True)
    eeg["dipfit"]["model"] = [{"posxyz": [[0.0, 0.0, 40.0]], "momxyz": [[0.0, 0.0, 1.0]], "rv": 1.0, "component": 1}]

    refined = pop_dipfit_nonlinear(eeg, component=1, gui=False)

    position = np.asarray(refined["dipfit"]["model"][0]["posxyz"], dtype=float)[0]
    assert np.all(np.isfinite(position))
    comp = eeglab2fieldtrip(refined)
    vol, _sens = prepare_headmodel_sens(fitting_config(refined), comp["elec"], refined["chanlocs"])
    assert str(vol["type"]).lower() == "dipoli"


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

    comp = eeglab2fieldtrip(eeg)
    cfg = fitting_config(eeg, [1])
    cfg["channel"] = [label for label in cfg["channel"] if label in comp["topolabel"]]
    _vol, sens = prepare_headmodel_sens(cfg, comp["elec"], eeg["chanlocs"])
    assert comp["topo"].shape == (18, 1)
    assert sens["elecpos"].shape == (18, 3)
    assert sens["label"] == [f"E{index}" for index in range(1, 19)]

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
    assert fitted["dipfit"]["coord_transform"] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
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


def test_besa_dipole_plot_uses_the_dialog_mri_without_mni_slices():
    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=32, n_components=1)
    eeg["dipfit"] = {
        "coordformat": "spherical",
        "mrifile": "standard_BESA/avg152t1.mat",
        "hdmfile": "standard_BESA/standard_BESA.mat",
        "model": [{"posxyz": [[10.0, -20.0, 40.0]], "momxyz": [[0.0, 0.0, 1.0]], "rv": 0.02, "component": 1}],
    }
    mri = _dialog_control(pop_dipplot_dialog_spec(eeg), "mri").value

    figures, _command = pop_dipplot(eeg, [1], mri=mri, gui=False, plot=True, return_com=True)

    assert "avg152t1.mat" in str(mri)
    assert len(figures) == 1
    assert figures[0].eegprep_mri == "avg152t1"
    assert any(axis.images for axis in figures[0].axes)
    plt.close(figures[0])


def test_pop_dipplot_defaults_to_localized_components_when_some_models_are_empty():
    eeg = _configured_ica_eeg()
    eeg["dipfit"]["model"].append({"posxyz": [], "momxyz": [], "rv": 1.0, "component": 3})

    figures, com = pop_dipplot(eeg, "", gui=False, plot=True, return_com=True)

    assert len(figures) == 1
    assert _console_python_command(com) == "pop_dipplot(EEG, comps=[1, 2])"
    plt.close(figures[0])


def _dialog_control(spec, tag: str):
    return next(control for control in spec.controls if control.tag == tag)


def test_empty_dipfit_array_is_missing_settings():
    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=32, n_components=1)
    eeg["dipfit"] = np.array([])

    assert normalize_model_list(eeg) == []
    with pytest.raises(ValueError, match="No dipole information"):
        pop_dipplot(eeg)


def test_gui_dipfit_template_choice_updates_files_and_browse(qapp):
    from PySide6.QtWidgets import QApplication

    from eegprep.functions.guifunc.qt import QtDialogRenderer

    _app, dialog, widgets = QtDialogRenderer().build_dialog(
        pop_dipfit_settings_dialog_spec(create_test_eeg(n_channels=2)), {}
    )
    try:
        widgets["model"].setCurrentIndex(0)
        QApplication.processEvents()
        assert "standard_BESA.mat" in widgets["hdmfile"].text()
        assert "avg152t1.mat" in widgets["mrifile"].text()
        assert widgets["no_coreg"].isChecked() is False
        assert widgets["coregister"].isEnabled()
        assert not widgets["hdmfile_browse"].isEnabled()
        widgets["model"].setCurrentIndex(1)
        QApplication.processEvents()
        assert "standard_vol.mat" in widgets["hdmfile"].text()
        assert "standard_mri.mat" in widgets["mrifile"].text()
        assert widgets["no_coreg"].isChecked() is False
        widgets["model"].setCurrentIndex(2)
        QApplication.processEvents()
        assert widgets["hdmfile"].text() == ""
        assert widgets["hdmfile_browse"].isEnabled()
        assert widgets["mrifile_browse"].isEnabled()
        assert widgets["chanfile_browse"].isEnabled()
        assert widgets["coregister"].isEnabled()
    finally:
        dialog.close()


def test_besa_coarse_fit_and_autofit_keep_linspace_history():
    eeg = pop_dipfit_settings(
        create_test_eeg_with_ica(n_channels=4, n_samples=40, n_components=2),
        model="standardBESA",
        coord_transform=[],
    )
    fitted, command = pop_dipfit_gridsearch(
        eeg,
        [1],
        "linspace(-40,40,3)",
        "linspace(-40,40,3)",
        "linspace(0,40,2)",
        100,
        return_com=True,
    )

    assert np.asarray(fitted["dipfit"]["model"][0]["posxyz"]).size >= 3
    assert "linspace(-40,40,3)" in command
    translated = _console_python_command(command)
    assert "linspace(-40, 40, 3)" in translated
    from eegprep.functions.miscfunc.linspace import linspace

    namespace = {"EEG": eeg, "pop_dipfit_gridsearch": pop_dipfit_gridsearch, "linspace": linspace}
    exec(translated, namespace)
    assert np.asarray(namespace["EEG"]["dipfit"]["model"][0]["posxyz"]).size >= 3

    autofit, autofit_command = pop_multifit(fitted, [1], threshold=100, return_com=True)
    assert np.asarray(autofit["dipfit"]["model"][0]["posxyz"]).size >= 3
    assert "pop_multifit" in _console_python_command(autofit_command)


def test_loreta_key_leadfield_and_missing_source_analysis():
    eeg = pop_dipfit_settings(
        create_test_eeg_with_ica(n_channels=4, n_samples=32, n_components=1),
        model="standardBEM",
        coord_transform=[],
    )
    with pytest.raises(ValueError, match="pop_leadfield"):
        pop_dipfit_loreta(eeg, [1])

    leadfield, command = pop_leadfield(eeg, sourcemodel="loreta-key", return_com=True)
    points = np.asarray(leadfield["dipfit"]["sourcemodel"]["pos"], dtype=float)
    assert points.ndim == 2 and points.shape[1] == 3 and points.shape[0] > 0
    assert np.all(np.linalg.norm(points, axis=1) <= 75.0)
    assert "sourcemodel='loreta-key'" in _console_python_command(command)
    with pytest.raises(DIPFITUnavailableError, match="pop_dipfit_loreta"):
        pop_dipfit_loreta(leadfield, [1])


def test_gui_leadfield_loreta_key_choice_fills_the_source_field(qapp):
    from PySide6.QtWidgets import QApplication

    from eegprep.functions.guifunc.qt import QtDialogRenderer

    spec = pop_leadfield_dialog_spec(create_test_eeg_with_ica(n_channels=2, n_samples=16, n_components=1))
    assert _dialog_control(spec, "sourcemodel").value == "loreta-key"
    _app, dialog, widgets = QtDialogRenderer().build_dialog(spec, {})
    try:
        assert widgets["sourcemodel"].text() == "loreta-key"
        assert not widgets["source_browse"].isEnabled()
        widgets["source_choice"].setCurrentIndex(4)
        QApplication.processEvents()
        assert widgets["sourcemodel"].text() == ""
        assert widgets["source_browse"].isEnabled()
        widgets["source_choice"].setCurrentIndex(0)
        QApplication.processEvents()
        with pytest.raises(ValueError, match="pop_leadfield"):
            pop_leadfield(
                create_test_eeg_with_ica(n_channels=2, n_samples=16, n_components=1),
                gui=True,
                renderer=_LeadfieldChoiceRenderer(),
            )
    finally:
        dialog.close()


class _LeadfieldChoiceRenderer:
    def run(self, spec, initial_values=None):
        return {"source_choice": 1, "sourcemodel": "", "sourcemodel2mni": "", "downsample": ""}


def test_autofit_plot_accepts_quoted_normlen_and_shows_dipoles(monkeypatch):
    shown = []
    monkeypatch.setattr(matplotlib.figure.Figure, "show", lambda self: shown.append(self))
    monkeypatch.setattr("eegprep.functions.popfunc.plot_utils.backend_can_display", lambda: True)
    eeg = pop_dipfit_settings(
        create_test_eeg_with_ica(n_channels=4, n_samples=32, n_components=1),
        model="standardBESA",
        coord_transform=[],
    )
    coarse, _command = pop_dipfit_gridsearch(eeg, [1], [0], [0], [40], 100, return_com=True)
    try:
        pop_multifit(coarse, [1], threshold=100, dipplot="on", plotopt="'normlen' 'on'")
        assert shown
        assert shown[-1].axes
    finally:
        plt.close("all")


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        pytest.skip(f"PySide6 Qt libraries unavailable in this environment: {exc}")
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.mark.modal_dialogs
def test_default_bem_coregister_click_returns_a_transform(qapp):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QPushButton

    from eegprep.functions.guifunc.qt import QtDialogRenderer

    _app, dialog, widgets = QtDialogRenderer().build_dialog(
        pop_dipfit_settings_dialog_spec(create_test_eeg(n_channels=4)), {}
    )
    stage = {"name": ""}

    def accept_coregister() -> None:
        active = QApplication.activeModalWidget()
        if active is None or active is dialog:
            QTimer.singleShot(20, accept_coregister)
            return
        stage["name"] = active.objectName()
        ok = active.findChild(QPushButton, "ok")
        if ok is not None:
            ok.click()
        else:
            active.accept()

    try:
        QTimer.singleShot(20, accept_coregister)
        widgets["coregister"].click()
        values = [float(item) for item in widgets["coord_transform"].text().split()]
        assert stage["name"] == "coregister"
        assert len(values) == 9
        assert all(np.isfinite(value) for value in values)
    finally:
        dialog.close()


def test_besa_template_is_packaged_and_warps(qapp):
    from eegprep.functions.guifunc.coregister import load_coregister_mesh
    from eegprep.functions.sigprocfunc.coregister import coregister
    from eegprep.plugins.dipfit._utils import packaged_dipfit_file

    electrodes = packaged_dipfit_file("standard_BESA/standard-10-5-cap385.elp")
    mesh = load_coregister_mesh(str(packaged_dipfit_file("standard_BESA/standard_BESA.mat")))
    assert mesh is not None
    assert mesh.vertices.shape[1] == 3
    result = coregister(str(electrodes), str(electrodes), warp="auto")
    assert result.transform.shape == (9,)
    assert np.isfinite(result.transform).all()


def test_disabled_dipfit_menu_tooltips_reach_the_main_window(qapp):
    from eegprep.functions.guifunc.main_window import build_main_window
    from eegprep.functions.guifunc.session import EEGPrepSession

    window = build_main_window(EEGPrepSession(), all_menus=True)
    actions = {}

    def collect(action) -> None:
        menu = action.menu()
        if menu is not None:
            for child in menu.actions():
                collect(child)
        if action.data():
            actions[str(action.data())] = action

    try:
        for action in window.window.menuBar().actions():
            collect(action)
        headmodel = actions["pop_dipfit_headmodel"]
        loreta = actions["pop_dipfit_loreta"]
        assert not headmodel.isEnabled()
        assert not loreta.isEnabled()
        assert "boundary-element head model" in headmodel.toolTip()
        assert "eLORETA" in loreta.toolTip()
    finally:
        window.window.close()


def test_multifit_history_emits_a_matlab_plot_cell():
    command = dipfit_history("pop_multifit", [1], plotopt="'normlen' 'on'")
    assert "{'normlen' 'on'}" in command


def test_compatlas_labels_the_nearest_desikan_killiany_area():
    from eegprep.plugins.dipfit._coordinates import traditionaldipfit
    from eegprep.plugins.dipfit.eeg_compatlas import _atlas, eeg_compatlas

    vertices, colors, labels, _tree = _atlas()
    chosen = int(np.flatnonzero(colors > 0)[0])
    transformed = traditionaldipfit(
        [0.0, -26.604623, -46.0, 0.12346256, 0.0, -1.5707963, 1000.0, 1000.0, 1000.0]
    ) @ np.array([*vertices[chosen], 1.0])
    eeg = {
        "icaweights": np.eye(1),
        "dipfit": {
            "coordformat": "MNI",
            "model": [{"posxyz": [transformed[:3].tolist()], "component": 1}],
        },
    }

    labeled = eeg_compatlas(eeg, components=[1])

    assert labeled["dipfit"]["model"][0]["areadk"] == str(labels[int(colors[chosen]) - 1])


def test_dipole_plot_navigation_moves_between_components():
    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=32, n_components=2)
    eeg["dipfit"] = {
        "coordformat": "MNI",
        "mrifile": "standard_BEM/standard_mri.mat",
        "model": [
            {
                "posxyz": [[10.0, -20.0, 30.0]],
                "momxyz": [[0.0, 0.0, 1.0]],
                "rv": 0.02,
                "areadk": "bankssts L",
                "component": 1,
            },
            {
                "posxyz": [[-10.0, 15.0, 25.0]],
                "momxyz": [[0.0, 1.0, 0.0]],
                "rv": 0.03,
                "areadk": "bankssts R",
                "component": 2,
            },
        ],
    }

    figures, _command = pop_dipplot(eeg, [1, 2], gui=False, plot=True, return_com=True)
    try:
        navigation = figures[0].eegprep_dipplot
        assert navigation["area"].get_text() == "Comp: 1 bankssts L"
        before = np.array(figures[0].axes[1].images[0].get_array())
        navigation["plot_one"]()
        assert navigation["state"]["single"] is True
        navigation["next"]()
        assert navigation["state"]["index"] == 1
        assert navigation["area"].get_text() == "Comp: 2 bankssts R"
        after = np.array(figures[0].axes[1].images[0].get_array())
        assert not np.array_equal(before, after)
        navigation["prev"]()
        assert navigation["state"]["index"] == 0
    finally:
        plt.close(figures[0])


def test_nonlinear_ok_keeps_the_flipped_moment(qapp, monkeypatch):
    from eegprep.functions.guifunc import qt as qt_module
    from eegprep.plugins.dipfit import pop_dipfit_nonlinear as nonlinear_module

    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=32, n_components=1)
    eeg["dipfit"] = {
        "coordformat": "spherical",
        "hdmfile": "standard_BESA/standard_vol.mat",
        "model": [{"posxyz": [[1.0, 2.0, 3.0]], "momxyz": [[1.0, 2.0, 3.0]], "rv": 0.2, "component": 1}],
    }

    def refuse(*_args, **_kwargs):
        raise AssertionError("OK must store the dialog dipole without refitting")

    monkeypatch.setattr(nonlinear_module, "run_nonlinear", refuse)

    class Renderer:
        def run(self, spec, initial_values=None):
            renderer = qt_module.QtDialogRenderer()
            _app, dialog, widgets = renderer.build_dialog(spec)
            try:
                widgets["flip1"].click()
                assert widgets["dip1mom"].text().split() == ["-1", "-2", "-3"]
                widgets["dip1pos"].setText("4 5 6")
                dialog.findChild(qt_module.QtWidgets.QPushButton, "ok").click()
                return {tag: renderer._read_widget(widget) for tag, widget in widgets.items()}
            finally:
                dialog.close()

    out, command = pop_dipfit_nonlinear(copy.deepcopy(eeg), gui=True, renderer=Renderer(), return_com=True)
    np.testing.assert_allclose(out["dipfit"]["model"][0]["momxyz"][0], [-1.0, -2.0, -3.0])
    np.testing.assert_allclose(out["dipfit"]["model"][0]["posxyz"][0], [4.0, 5.0, 6.0])
    assert "'posxyz'" in command and "'momxyz'" in command
    replayed = _replay_nonlinear_command(eeg, command)
    np.testing.assert_allclose(replayed["dipfit"]["model"][0]["momxyz"][0], [-1.0, -2.0, -3.0])
    np.testing.assert_allclose(replayed["dipfit"]["model"][0]["posxyz"][0], [4.0, 5.0, 6.0])


def test_nonlinear_ok_keeps_a_moment_only_fit(qapp, monkeypatch):
    from eegprep.functions.guifunc import qt as qt_module
    from eegprep.plugins.dipfit import pop_dipfit_nonlinear as nonlinear_module

    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=32, n_components=1)
    eeg["dipfit"] = {
        "coordformat": "spherical",
        "hdmfile": "standard_BESA/standard_vol.mat",
        "model": [{"posxyz": [[1.0, 2.0, 3.0]], "momxyz": [[1.0, 2.0, 3.0]], "rv": 0.2, "component": 1}],
    }
    calls = []

    def fitted(dataset, **kwargs):
        calls.append(kwargs.get("nonlinear"))
        updated = copy.deepcopy(dataset)
        model = updated["dipfit"]["model"][0]
        if str(kwargs.get("nonlinear")).lower() == "no":
            model["momxyz"] = [[9.0, 8.0, 7.0]]
        else:
            model["posxyz"] = [[4.0, 5.0, 6.0]]
        return updated

    monkeypatch.setattr(nonlinear_module, "run_nonlinear", fitted)

    class Renderer:
        def run(self, spec, initial_values=None):
            renderer = qt_module.QtDialogRenderer()
            _app, dialog, widgets = renderer.build_dialog(spec)
            try:
                widgets["fitmoment"].click()
                assert widgets["dip1pos"].text().split()[:3] == ["1", "2", "3"]
                assert widgets["dip1mom"].text().split()[:3] == ["9", "8", "7"]
                dialog.findChild(qt_module.QtWidgets.QPushButton, "ok").click()
                return {tag: renderer._read_widget(widget) for tag, widget in widgets.items()}
            finally:
                dialog.close()

    out, command = pop_dipfit_nonlinear(copy.deepcopy(eeg), gui=True, renderer=Renderer(), return_com=True)
    np.testing.assert_allclose(out["dipfit"]["model"][0]["posxyz"][0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(out["dipfit"]["model"][0]["momxyz"][0], [9.0, 8.0, 7.0])
    assert calls == ["no"]
    assert "'posxyz'" in command and "'momxyz'" in command
    replayed = _replay_nonlinear_command(eeg, command)
    np.testing.assert_allclose(replayed["dipfit"]["model"][0]["posxyz"][0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(replayed["dipfit"]["model"][0]["momxyz"][0], [9.0, 8.0, 7.0])


def test_nonlinear_ok_keeps_typed_dipole_values(qapp, monkeypatch):
    from eegprep.functions.guifunc import qt as qt_module
    from eegprep.plugins.dipfit import pop_dipfit_nonlinear as nonlinear_module

    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=32, n_components=1)
    eeg["dipfit"] = {
        "coordformat": "spherical",
        "hdmfile": "standard_BESA/standard_vol.mat",
        "model": [{"posxyz": [[1.0, 2.0, 3.0]], "momxyz": [[1.0, 2.0, 3.0]], "rv": 0.2, "component": 1}],
    }

    def refuse(*_args, **_kwargs):
        raise AssertionError("OK must store the typed dipole without refitting")

    monkeypatch.setattr(nonlinear_module, "run_nonlinear", refuse)

    class Renderer:
        def run(self, spec, initial_values=None):
            renderer = qt_module.QtDialogRenderer()
            _app, dialog, widgets = renderer.build_dialog(spec)
            try:
                widgets["dip1pos"].setText("4 5 6")
                widgets["dip1mom"].setText("7 8 9")
                dialog.findChild(qt_module.QtWidgets.QPushButton, "ok").click()
                return {tag: renderer._read_widget(widget) for tag, widget in widgets.items()}
            finally:
                dialog.close()

    out, command = pop_dipfit_nonlinear(copy.deepcopy(eeg), gui=True, renderer=Renderer(), return_com=True)
    np.testing.assert_allclose(out["dipfit"]["model"][0]["posxyz"][0], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(out["dipfit"]["model"][0]["momxyz"][0], [7.0, 8.0, 9.0])
    assert "'posxyz'" in command and "'momxyz'" in command
    replayed = _replay_nonlinear_command(eeg, command)
    np.testing.assert_allclose(replayed["dipfit"]["model"][0]["posxyz"][0], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(replayed["dipfit"]["model"][0]["momxyz"][0], [7.0, 8.0, 9.0])


def test_fine_fit_menu_stores_the_flipped_moment(qapp, monkeypatch):
    from eegprep.functions.guifunc import qt as qt_module
    from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher
    from eegprep.functions.guifunc.session import EEGPrepSession
    from eegprep.plugins.dipfit import pop_dipfit_nonlinear as nonlinear_module

    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=32, n_components=1)
    eeg["dipfit"] = {
        "coordformat": "spherical",
        "hdmfile": "standard_BESA/standard_vol.mat",
        "model": [{"posxyz": [[1.0, 2.0, 3.0]], "momxyz": [[1.0, 2.0, 3.0]], "rv": 0.2, "component": 1}],
    }
    session = EEGPrepSession()
    session.store_current(eeg, new=True)
    real_inputgui = nonlinear_module.inputgui

    def drive(spec, renderer=None, **_kwargs):
        class Renderer:
            def run(self, dialog_spec, initial_values=None):
                qt_renderer = qt_module.QtDialogRenderer()
                _app, dialog, widgets = qt_renderer.build_dialog(dialog_spec)
                try:
                    widgets["flip1"].click()
                    dialog.findChild(qt_module.QtWidgets.QPushButton, "ok").click()
                    return {tag: qt_renderer._read_widget(widget) for tag, widget in widgets.items()}
                finally:
                    dialog.close()

        return real_inputgui(spec, renderer=Renderer())

    monkeypatch.setattr(nonlinear_module, "inputgui", drive)
    MenuActionDispatcher(session)._run_dipfit_function("pop_dipfit_nonlinear", None)
    np.testing.assert_allclose(session.EEG["dipfit"]["model"][0]["momxyz"][0], [-1.0, -2.0, -3.0])
    assert any("pop_dipfit_nonlinear" in command and "'momxyz'" in command for command in session.ALLCOM)


def _replay_nonlinear_command(eeg, command):
    from eegprep.functions.adminfunc.console import _console_python_command
    from eegprep.plugins.dipfit.pop_dipfit_nonlinear import pop_dipfit_nonlinear

    namespace = {"EEG": copy.deepcopy(eeg), "pop_dipfit_nonlinear": pop_dipfit_nonlinear}
    exec(_console_python_command(command), namespace)
    return namespace["EEG"]


def test_nonlinear_fit_button_updates_the_displayed_model(qapp, monkeypatch):
    from eegprep.functions.guifunc.qt import QtDialogRenderer
    from eegprep.plugins.dipfit import pop_dipfit_nonlinear as nonlinear_module

    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=32, n_components=1)
    eeg["dipfit"] = {
        "coordformat": "spherical",
        "model": [{"posxyz": [[1.0, 2.0, 3.0]], "momxyz": [[0.1, 0.2, 0.3]], "rv": 0.2, "component": 1}],
    }

    def fitted(dataset, **_kwargs):
        updated = copy.deepcopy(dataset)
        updated["dipfit"]["model"][0] = {
            "posxyz": [[9.0, 8.0, 7.0]],
            "momxyz": [[-1.0, 0.0, 0.0]],
            "rv": 0.01,
            "component": 1,
        }
        return updated

    monkeypatch.setattr(nonlinear_module, "run_nonlinear", fitted)
    _app, dialog, widgets = QtDialogRenderer().build_dialog(pop_dipfit_nonlinear_dialog_spec(eeg), {})
    try:
        assert widgets["plotmap"].isEnabled()
        assert widgets["fitposition"].isEnabled()
        widgets["dip1mom"].setText("1340.8900000000001 0 0")
        widgets["flip1"].click()
        assert widgets["dip1mom"].text().split()[0] == "-1340.89"
        widgets["fitposition"].click()
        assert widgets["dip1pos"].text().split()[:3] == ["9", "8", "7"]
        assert widgets["relvar"].text() == "1%"
    finally:
        dialog.close()


def test_spherical_dipoles_are_drawn_with_sph2spm():
    from eegprep.plugins.dipfit._coordinates import sph2spm

    raw = np.array([0.0, 0.0, 85.0])
    expected = (sph2spm() @ np.array([*raw, 1.0]))[:3]
    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=32, n_components=3)
    eeg["dipfit"] = {
        "coordformat": "spherical",
        "mrifile": "standard_BESA/avg152t1.mat",
        "model": [
            {"posxyz": [[0.0, 0.0, 0.0]], "component": 1},
            {"posxyz": [[0.0, 0.0, 0.0]], "component": 2},
            {"posxyz": [raw.tolist()], "momxyz": [[0.0, 0.0, 1.0]], "component": 3},
        ],
    }
    figures, _command = pop_dipplot(eeg, [3], gui=False, plot=True, return_com=True)
    try:
        shown = np.column_stack(figures[0].axes[0].collections[0]._offsets3d)
        np.testing.assert_allclose(shown[0], expected, atol=1e-6)
        assert figures[0].eegprep_mri == "avg152t1"
        assert figures[0].axes[1].images[0].get_array().shape == (91, 109)
    finally:
        plt.close(figures[0])


def test_mni_dipole_slices_use_the_colin_volume():
    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=32, n_components=1)
    eeg["dipfit"] = {
        "coordformat": "MNI",
        "mrifile": "standard_BEM/standard_mri.mat",
        "model": [{"posxyz": [[-0.05, 61.6, -41.1]], "momxyz": [[0.0, 0.0, 1.0]], "component": 1}],
    }
    figures = pop_dipplot(eeg, [1], gui=False, plot=True)
    try:
        assert figures[0].eegprep_mri == "standard_mri"
        assert figures[0].axes[1].images[0].get_array().shape == (181, 217)
    finally:
        plt.close(figures[0])


def test_pop_dipplot_draws_only_the_selected_dipole():
    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=32, n_components=1)
    eeg["dipfit"] = {
        "coordformat": "MNI",
        "model": [
            {
                "posxyz": [[4.0, 5.0, 6.0], [0.0, 0.0, 0.0]],
                "momxyz": [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                "select": [1],
                "component": 1,
            }
        ],
    }
    figures = pop_dipplot(eeg, [1], gui=False, plot=True, projlines="on", normlen="on", title="IC 3")
    try:
        shown = np.column_stack(figures[0].axes[0].collections[0]._offsets3d)
        assert shown.shape[0] == 1
        np.testing.assert_allclose(shown[0], [4.0, 5.0, 6.0])
        assert figures[0].axes[0].get_title() == "IC 3"
        assert figures[0].axes[0].lines
    finally:
        plt.close(figures[0])


def test_plot_dipole_omits_an_unchecked_dipole_even_when_its_position_is_nonzero():
    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=32, n_components=1)
    eeg["dipfit"] = {
        "coordformat": "MNI",
        "model": [
            {
                "posxyz": [[4.0, 5.0, 6.0], [10.0, 20.0, 30.0]],
                "momxyz": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                "select": [1],
                "component": 1,
            }
        ],
    }
    figures = pop_dipplot(eeg, [1], gui=False, plot=True)
    try:
        shown = np.column_stack(figures[0].axes[0].collections[0]._offsets3d)
        assert shown.shape[0] == 1
        np.testing.assert_allclose(shown[0], [4.0, 5.0, 6.0])
    finally:
        plt.close(figures[0])


def test_empty_besa_coord_transform_labels_component_1_superiorfrontal_r():
    from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
    from eegprep.plugins.dipfit.eeg_compatlas import _atlas, eeg_compatlas

    vertices, colors, labels, _tree = _atlas()
    chosen = next(
        index
        for index, color in enumerate(colors)
        if int(color) > 0 and str(labels[int(color) - 1]) == "superiorfrontal R"
    )
    point = traditionaldipfit([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]) @ np.array([*vertices[chosen], 1.0])
    eeg = {
        "data": np.zeros((1, 8)),
        "nbchan": 1,
        "pnts": 8,
        "trials": 1,
        "srate": 100.0,
        "icaweights": np.eye(1),
        "dipfit": {
            "coordformat": "spherical",
            "coord_transform": [],
            "model": [{"posxyz": [point[:3].tolist()], "component": 1}],
        },
    }

    labeled = eeg_compatlas(eeg, components=[1])
    assert labeled["dipfit"]["model"][0]["areadk"] == "superiorfrontal R"

    checked = eeg_checkset(copy.deepcopy(eeg))
    assert checked["dipfit"]["coord_transform"] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_malformed_coord_transform_names_the_field():
    from eegprep.plugins.dipfit.eeg_compatlas import eeg_compatlas

    eeg = {
        "dipfit": {
            "coordformat": "spherical",
            "coord_transform": "not-nine-numbers",
            "model": [{"posxyz": [[0.0, 0.0, 0.0]], "component": 1}],
        }
    }
    with pytest.raises(ValueError, match=r"eeg_compatlas: coord_transform must be 9 numbers"):
        eeg_compatlas(eeg, components=[1])


def test_nonlinear_plot_buttons_show_the_dialog_dipole(qapp, monkeypatch):
    from eegprep.functions.guifunc.qt import QtDialogRenderer

    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=32, n_components=1)
    eeg["dipfit"] = {
        "coordformat": "MNI",
        "model": [{"posxyz": [[1.0, 2.0, 3.0]], "momxyz": [[0.1, 0.2, 0.3]], "rv": 0.2, "component": 1}],
    }
    calls = {}

    def topo(dataset, typeplot, items, title, rowcols, plotdip, **kwargs):
        calls["topo"] = (typeplot, items, title, rowcols, plotdip)

    def dipoles(dataset, components, **kwargs):
        model = dataset["dipfit"]["model"][0]
        calls["dipole"] = np.asarray(model["posxyz"], dtype=float)
        calls["select"] = list(model.get("select", []))
        calls["kwargs"] = kwargs
        return [], ""

    monkeypatch.setattr("eegprep.functions.popfunc.pop_topoplot.pop_topoplot", topo)
    monkeypatch.setattr("eegprep.plugins.dipfit.pop_dipplot.pop_dipplot", dipoles)
    _app, dialog, widgets = QtDialogRenderer().build_dialog(pop_dipfit_nonlinear_dialog_spec(eeg), {})
    try:
        widgets["plotmap"].click()
        widgets["dip1pos"].setText("4 5 6")
        widgets["plotdip"].click()
    finally:
        dialog.close()
    assert calls["topo"] == (0, 1, "IC 1", [1, 1], 1)
    np.testing.assert_allclose(calls["dipole"].reshape(-1)[:3], [4.0, 5.0, 6.0])
    assert calls["select"] == [1]
    assert calls["kwargs"]["projlines"] == "on"
    assert calls["kwargs"]["normlen"] == "on"
    assert calls["kwargs"]["title"] == "IC 1"


def test_coregister_fields_and_writeback_use_four_significant_digits(qapp, monkeypatch):
    from PySide6.QtWidgets import QCheckBox, QLineEdit

    from eegprep.functions.guifunc.coregister import CoregisterDialog
    from eegprep.functions.guifunc.qt import _dipfit_coregister

    monkeypatch.setattr(
        "eegprep.functions.guifunc.qt.run_coregister_dialog",
        lambda *_args, **_kwargs: [1.2345678901234567, 0, 0, 0, 0, 0, 1, 1, 1],
    )
    widgets = {
        "chanfile": QLineEdit("standard_BESA/standard-10-5-cap385.elp"),
        "hdmfile": QLineEdit("standard_BESA/standard_BESA.mat"),
        "coord_transform": QLineEdit(),
        "nocoreg": QCheckBox(),
    }
    _dipfit_coregister(
        None,
        widgets,
        {
            "chanfile": "chanfile",
            "hdmfile": "hdmfile",
            "target": "coord_transform",
            "no_coreg": "nocoreg",
            "chanlocs": [],
        },
    )
    assert widgets["coord_transform"].text().split()[0] == "1.235"

    locs = [{"labels": "Fz", "X": 0.0, "Y": 80.0, "Z": 10.0}, {"labels": "Cz", "X": 0.0, "Y": 0.0, "Z": 90.0}]
    dialog = CoregisterDialog(
        locs,
        locs,
        chaninfo=None,
        meshfile=None,
        transform=[1.2345678901234567, 0, 0, 0, 0, 0, 1, 1, 1],
        parent=None,
        title="coregister",
    )
    try:
        assert dialog._fields["right"].text() == "1.235"
    finally:
        dialog.close()


def test_headless_coregister_resolves_a_stored_dipfit_template():
    from eegprep.functions.sigprocfunc.coregister import _resolve_coregistration_file

    resolved = _resolve_coregistration_file("standard_BESA/standard-10-5-cap385.elp")
    assert resolved.is_file()
    assert resolved.name == "standard-10-5-cap385.elp"
