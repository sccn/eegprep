"""Release-readiness coverage for ICLabel, component properties, and DIPFIT."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pytest

from eegprep.functions.adminfunc.console import _console_python_command
from eegprep.plugins.ICLabel.iclabel import iclabel
from eegprep.plugins.ICLabel.menu import iclabel_menu, viewprops_plot_menus
from eegprep.plugins.ICLabel.pop_icflag import pop_icflag
from eegprep.plugins.ICLabel.pop_viewprops import pop_viewprops
from eegprep.plugins.dipfit._utils import DIPFITUnavailableError, normalize_model_list
from eegprep.plugins.dipfit.pop_dipfit_gridsearch import pop_dipfit_gridsearch
from eegprep.plugins.dipfit.pop_dipfit_loreta import pop_dipfit_loreta
from eegprep.plugins.dipfit.pop_dipfit_settings import pop_dipfit_settings, pop_dipfit_settings_dialog_spec
from eegprep.plugins.dipfit.pop_dipplot import pop_dipplot
from eegprep.plugins.dipfit.pop_leadfield import pop_leadfield, pop_leadfield_dialog_spec
from eegprep.plugins.dipfit.pop_multifit import pop_multifit
from tests.fixtures import create_test_eeg, create_test_eeg_with_ica

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _classified_eeg(rows: int = 3) -> dict:
    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=32, n_components=rows)
    classifications = np.zeros((rows, 7))
    classifications[:, 0] = 1.0
    if rows:
        classifications[0] = [0.02, 0.96, 0.0, 0.0, 0.0, 0.0, 0.02]
    eeg["etc"] = {"ic_classification": {"ICLabel": {"classifications": classifications, "classes": ["Brain"] * 7}}}
    return eeg


def _control(spec, tag: str):
    return next(control for control in spec.controls if control.tag == tag)


def test_f53_gui_and_dev_installs_include_pytorch_for_iclabel():
    project = tomllib.loads(Path("pyproject.toml").read_text())
    extras = project["project"]["optional-dependencies"]

    assert "torch>=2.0" in extras["gui"]
    assert "torch>=2.0" in project["dependency-groups"]["dev"]
    assert "torch>=2.0" in extras["torch"]
    assert all("torch" not in item for item in project["project"]["dependencies"])


def test_f55_icflag_history_is_replayable_python():
    eeg = _classified_eeg()
    _out, command = pop_icflag(eeg, gui=False, return_com=True)

    assert "None None" not in command
    assert _console_python_command(command) == (
        "EEG = pop_icflag(EEG, thresholds=[[None, None], [0.9, 1], "
        "[0.9, 1], [None, None], [None, None], [None, None], [None, None]])"
    )
    namespace = {"EEG": _classified_eeg(), "pop_icflag": pop_icflag}
    exec(command, namespace)
    assert namespace["EEG"]["reject"]["gcompreject"][0] == 1


def test_f56_empty_dipfit_array_is_missing_settings():
    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=32, n_components=1)
    eeg["dipfit"] = np.array([])

    assert normalize_model_list(eeg) == []
    with pytest.raises(ValueError, match="No dipole information"):
        pop_dipplot(eeg)


def test_f57_f75_f122_property_windows_are_shown_and_maps_open_a_component(monkeypatch):
    shown = []

    def record_show(self):
        shown.append(self)

    monkeypatch.setattr(matplotlib.figure.Figure, "show", record_show)
    monkeypatch.setattr("eegprep.functions.popfunc.plot_utils.backend_can_display", lambda: True)
    try:
        channels = pop_viewprops(create_test_eeg(n_channels=3, n_samples=32), 1, [1, 2])
        channel_titles = {axis.get_title() for figure in channels for axis in figure.axes}
        assert "Channel Activity Power Spectrum" in channel_titles
        assert any(figure in shown for figure in channels)

        components = pop_viewprops(_classified_eeg(2), 0, [1, 2])
        component_titles = {axis.get_title() for figure in components for axis in figure.axes}
        assert "ICLabel" in component_titles
        assert "IC1 Activity Power Spectrum" in component_titles
        grid = next(figure for figure in components if hasattr(figure, "eegprep_open_property"))
        opened = grid.eegprep_open_property(2)
        assert "IC2" in {axis.get_title() for axis in opened.axes}
        assert opened in shown
    finally:
        plt.close("all")


def test_f65_f113_f114_settings_default_to_the_besa_sphere_and_fill_template_files():
    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=32, n_components=1)
    spec = pop_dipfit_settings_dialog_spec(eeg)

    assert _control(spec, "model").value == 1
    assert "standard_BESA.mat" in _control(spec, "hdmfile").value
    assert "avg152t1.mat" in _control(spec, "mrifile").value
    assert "standard-10-5-cap385.elp" in _control(spec, "chanfile").value

    class Renderer:
        def run(self, spec, initial_values=None):
            return {
                "model": 1,
                "hdmfile": "",
                "mrifile": "",
                "chanfile": "",
                "coordformat": 1,
                "no_coreg": True,
            }

    configured = pop_dipfit_settings(eeg, gui=True, renderer=Renderer())
    assert configured["dipfit"]["hdmfile"].endswith("standard_BESA.mat")
    assert configured["dipfit"]["vol"]["r"] == [71.0, 72.0, 79.0, 85.0]
    assert configured["dipfit"]["vol"]["c"] == [0.33, 1.0, 0.0042, 0.33]

    bem = pop_dipfit_settings(configured, model="standardBEM")
    assert "vol" not in bem["dipfit"]
    with pytest.raises(ValueError, match="pop_dipfit_gridsearch"):
        pop_dipfit_gridsearch(bem, [1], [0], [0], [40], 100)


def test_gui_dipfit_template_choice_updates_files_and_browse(qapp):
    from PySide6.QtWidgets import QApplication

    from eegprep.functions.guifunc.qt import QtDialogRenderer

    _app, dialog, widgets = QtDialogRenderer().build_dialog(
        pop_dipfit_settings_dialog_spec(create_test_eeg(n_channels=2)), {}
    )
    try:
        assert widgets["model"].currentIndex() == 0
        assert "standard_BESA.mat" in widgets["hdmfile"].text()
        assert not widgets["hdmfile_browse"].isEnabled()
        assert not widgets["coregister"].isEnabled()
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
        assert not widgets["coregister"].isEnabled()
    finally:
        dialog.close()


def test_f114_f115_besa_coarse_fit_and_autofit_keep_linspace_history():
    eeg = pop_dipfit_settings(
        create_test_eeg_with_ica(n_channels=4, n_samples=40, n_components=2), model="standardBESA"
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
    assert "xgrid='linspace(-40,40,3)'" in translated
    namespace = {"EEG": eeg, "pop_dipfit_gridsearch": pop_dipfit_gridsearch}
    exec(translated, namespace)
    assert np.asarray(namespace["EEG"]["dipfit"]["model"][0]["posxyz"]).size >= 3

    autofit, autofit_command = pop_multifit(fitted, [1], threshold=100, return_com=True)
    assert np.asarray(autofit["dipfit"]["model"][0]["posxyz"]).size >= 3
    assert "pop_multifit" in _console_python_command(autofit_command)


def test_f116_f117_loreta_key_leadfield_and_missing_source_analysis():
    eeg = pop_dipfit_settings(
        create_test_eeg_with_ica(n_channels=4, n_samples=32, n_components=1), model="standardBESA"
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
    assert _control(spec, "sourcemodel").value == "loreta-key"
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
                renderer=_ChoiceRenderer(),
            )
    finally:
        dialog.close()


class _ChoiceRenderer:
    def run(self, spec, initial_values=None):
        return {"source_choice": 1, "sourcemodel": "", "sourcemodel2mni": "", "downsample": ""}


def test_f118_autofit_plot_accepts_quoted_normlen_and_shows_dipoles(monkeypatch):
    shown = []
    monkeypatch.setattr(matplotlib.figure.Figure, "show", lambda self: shown.append(self))
    monkeypatch.setattr("eegprep.functions.popfunc.plot_utils.backend_can_display", lambda: True)
    eeg = pop_dipfit_settings(
        create_test_eeg_with_ica(n_channels=4, n_samples=32, n_components=1), model="standardBESA"
    )
    coarse, _command = pop_dipfit_gridsearch(eeg, [1], [0], [0], [40], 100, return_com=True)
    try:
        pop_multifit(coarse, [1], threshold=100, dipplot="on", plotopt="'normlen' 'on'")
        assert shown
        assert shown[-1].axes
    finally:
        plt.close("all")


def test_f6_plot_menu_keeps_extended_property_entries():
    channel, component = viewprops_plot_menus()
    tools = iclabel_menu()

    assert (channel.label, channel.action) == ("View extended channel properties", "pop_viewprops:channels")
    assert (component.label, component.action) == ("View extended component properties", "pop_viewprops:components")
    assert tools.label == "Classify components using ICLabel"
    assert any(child.action == "pop_viewprops:components" for child in tools.children)


def test_f7_stale_iclabel_results_are_refused_until_relabeled():
    removed = _classified_eeg()
    for key in ("icaweights", "icasphere", "icawinv", "icaact"):
        removed[key] = np.array([])
    with pytest.raises(ValueError, match="pop_runica"):
        pop_icflag(removed)
    with pytest.raises(ValueError, match="pop_runica"):
        pop_viewprops(removed, 0, [1])
    with pytest.raises(ValueError, match="Run pop_runica"):
        iclabel({"data": np.zeros((2, 8)), "icaweights": np.array([]), "etc": {}}, feature="pop_iclabel")

    mismatched = _classified_eeg(3)
    mismatched["etc"]["ic_classification"]["ICLabel"]["classifications"] = np.zeros((2, 7))
    with pytest.raises(ValueError, match="pop_iclabel"):
        pop_icflag(mismatched)

    kept = _classified_eeg(1)
    flagged = pop_icflag(kept, gui=False)
    assert flagged["reject"]["gcompreject"][0] == 1


def test_f75_component_properties_without_ica_name_the_missing_decomposition():
    eeg = create_test_eeg(n_channels=3, n_samples=32)
    eeg["etc"] = {"ic_classification": {"ICLabel": {"classifications": np.ones((2, 7))}}}
    with pytest.raises(ValueError, match="pop_viewprops"):
        pop_viewprops(eeg, 0)


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        pytest.skip(f"PySide6 Qt libraries unavailable in this environment: {exc}")
    app = QApplication.instance() or QApplication([])
    yield app
