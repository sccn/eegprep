from __future__ import annotations

from unittest import mock

import matplotlib.pyplot as plt
import pytest

from eegprep.functions.adminfunc.console import _console_python_command
from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher, action_kind
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.plugins.dipfit._utils import DIPFITUnavailableError
from eegprep.plugins.dipfit.pop_dipfit_gridsearch import pop_dipfit_gridsearch, pop_dipfit_gridsearch_dialog_spec
from eegprep.plugins.dipfit.pop_dipfit_headmodel import pop_dipfit_headmodel, pop_dipfit_headmodel_dialog_spec
from eegprep.plugins.dipfit.pop_dipfit_loreta import pop_dipfit_loreta, pop_dipfit_loreta_dialog_spec
from eegprep.plugins.dipfit.pop_dipfit_nonlinear import pop_dipfit_nonlinear, pop_dipfit_nonlinear_dialog_spec
from eegprep.plugins.dipfit.pop_dipfit_settings import pop_dipfit_settings, pop_dipfit_settings_dialog_spec
from eegprep.plugins.dipfit.pop_dipplot import pop_dipplot, pop_dipplot_dialog_spec
from eegprep.plugins.dipfit.pop_leadfield import pop_leadfield, pop_leadfield_dialog_spec
from eegprep.plugins.dipfit.pop_multifit import pop_multifit, pop_multifit_dialog_spec
from tests.fixtures import SAMPLE_DATASET_PATH, create_test_eeg, create_test_eeg_with_ica


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


def test_pop_dipfit_settings_stores_standard_model_and_python_history():
    eeg = _ica_eeg()

    out, com = pop_dipfit_settings(eeg, model="standardBEM", chanomit=[2], return_com=True)

    assert out is not eeg
    assert out["dipfit"]["coordformat"] == "MNI"
    assert out["dipfit"]["hdmfile"].endswith("standard_vol.mat")
    assert out["dipfit"]["chansel"] == [1, 3, 4]
    assert _console_python_command(com) == "EEG = pop_dipfit_settings(EEG, model='standardBEM', chanomit=[2])"


def test_pop_dipfit_settings_works_on_sample_data():
    eeg = pop_loadset(SAMPLE_DATASET_PATH)

    out, com = pop_dipfit_settings(eeg, model="standardBEM", return_com=True)

    assert out["dipfit"]["coordformat"] == "MNI"
    assert out["dipfit"]["chansel"]
    assert "pop_dipfit_settings" in com


def test_pop_dipfit_settings_requires_usable_channel_locations():
    eeg = create_test_eeg(n_channels=2, n_samples=20)
    eeg["chanlocs"] = [{"labels": "A"}, {"labels": "B"}]

    with pytest.raises(ValueError, match="No channel locations"):
        pop_dipfit_settings(eeg, model="standardBEM")


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


def test_dipfit_dialog_specs_keep_eeglab_source_and_key_defaults():
    eeg = _configured_ica_eeg()
    specs = [
        pop_dipfit_settings_dialog_spec(eeg),
        pop_dipfit_headmodel_dialog_spec(eeg, "subject_T1.nii"),
        pop_dipfit_gridsearch_dialog_spec(eeg),
        pop_dipfit_nonlinear_dialog_spec(eeg),
        pop_dipplot_dialog_spec(eeg),
        pop_multifit_dialog_spec(eeg),
        pop_leadfield_dialog_spec(eeg),
        pop_dipfit_loreta_dialog_spec(eeg),
    ]

    for spec in specs:
        assert spec.eeglab_source.startswith("plugins/dipfit/")
        assert spec.eeglab_source.endswith(".m")
        assert spec.size is not None

    assert controls_by_tag(specs[0])["model"].value == 2
    assert controls_by_tag(specs[2])["select"].value == "1:4"
    assert controls_by_tag(specs[4])["normlen"].value is True
    assert controls_by_tag(specs[5])["threshold"].value == "100"


def test_dipfit_fieldtrip_workflows_fail_clearly_after_prerequisites():
    eeg = _configured_ica_eeg()

    with pytest.raises(DIPFITUnavailableError, match="FieldTrip"):
        pop_dipfit_gridsearch(eeg, [1], [0], [0], [0], 40)
    with pytest.raises(DIPFITUnavailableError, match="FieldTrip"):
        pop_dipfit_nonlinear(eeg, gui=False)
    with pytest.raises(DIPFITUnavailableError, match="FieldTrip"):
        pop_multifit(eeg, [1])
    with pytest.raises(DIPFITUnavailableError, match="FieldTrip"):
        pop_dipfit_headmodel(eeg, "subject_T1.nii")
    with pytest.raises(DIPFITUnavailableError, match="FieldTrip"):
        pop_leadfield(eeg, sourcemodel="loreta.mat")
    eeg["dipfit"]["sourcemodel"] = {"pos": [[0, 0, 0]], "leadfield": []}
    with pytest.raises(DIPFITUnavailableError, match="FieldTrip"):
        pop_dipfit_loreta(eeg, [1], gui=False)


def test_dipfit_fieldtrip_workflows_report_missing_inputs_first():
    eeg = create_test_eeg(n_channels=2, n_samples=20)

    with pytest.raises(ValueError, match="No ICA components"):
        pop_dipfit_gridsearch(eeg, [1])

    eeg = _ica_eeg()
    with pytest.raises(ValueError, match="General dipolefit settings"):
        pop_dipfit_gridsearch(eeg, [1])


def test_pop_dipplot_plots_existing_models_and_records_replayable_command():
    eeg = _configured_ica_eeg()

    figures, com = pop_dipplot(eeg, [1], normlen="on", plot=True, return_com=True)

    assert len(figures) == 1
    assert _console_python_command(com) == "pop_dipplot(EEG, comps=[1], normlen='on')"
    plt.close(figures[0])


def test_pop_dipplot_errors_on_missing_or_unlocalized_models():
    eeg = _configured_ica_eeg()
    eeg["dipfit"]["model"][0]["posxyz"] = []

    with pytest.raises(ValueError, match="Localization not found"):
        pop_dipplot(eeg, [1], plot=False)

    eeg.pop("dipfit")
    with pytest.raises(ValueError, match="No dipole information"):
        pop_dipplot(eeg, plot=False)


def test_dipfit_menu_actions_are_implemented_and_dispatchable():
    implemented = [
        "pop_dipfit_settings",
        "pop_dipfit_headmodel",
        "pop_dipfit_gridsearch",
        "pop_dipfit_nonlinear",
        "pop_dipfit_loreta",
        "pop_dipplot",
        "pop_leadfield",
        "pop_multifit",
    ]
    for action in implemented:
        assert action_kind(action) == "implemented"

    session = EEGPrepSession()
    eeg = _configured_ica_eeg()
    session.store_current(eeg, new=True)
    dispatcher = MenuActionDispatcher(session)

    with mock.patch(
        "eegprep.plugins.dipfit.pop_dipplot.pop_dipplot",
        return_value=(["figure"], "pop_dipplot(EEG, [1])"),
    ) as dipplot:
        dispatcher.dispatch("pop_dipplot")

    dipplot.assert_called_once()
    assert session.ALLCOM[-1] == "pop_dipplot(EEG, [1])"
