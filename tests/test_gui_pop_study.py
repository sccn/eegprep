import ast

from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.studyfunc.pop_study import pop_study, pop_study_dialog_spec
from eegprep.functions.studyfunc.pop_studydesign import pop_studydesign, pop_studydesign_dialog_spec
from tests.fixtures import create_test_eeg


class _Renderer:
    def __init__(self, result):
        self.result = result

    def run(self, spec, initial_values=None):
        self.spec = spec
        return self.result


def _study_inputs():
    eeg = create_test_eeg(n_channels=2, n_samples=10)
    eeg.update({"setname": "one", "subject": "S01", "condition": "target", "filename": "one.set"})
    study, alleeg = pop_study(None, [eeg], name="Study")
    return study, alleeg


def test_pop_study_dialog_spec_exposes_loaded_dataset_metadata():
    study, alleeg = _study_inputs()

    spec = pop_study_dialog_spec(study, alleeg)
    controls = controls_by_tag(spec)

    assert spec.title == "Edit STUDY set information - pop_study()"
    assert spec.function_name == "pop_study"
    assert spec.eeglab_source == "functions/studyfunc/pop_study.m"
    assert spec.size == (850, 560)
    assert controls["name"].value == "Study"
    assert controls["dataset_1_filename"].value == "one.set"
    assert controls["dataset_1_subject"].value == "S01"
    assert controls["dataset_1_condition"].value == "target"
    assert controls["dataset_1_filename"].enabled is False


def test_pop_study_dialog_spec_exposes_all_dataset_rows():
    datasets = []
    for index in range(12):
        eeg = create_test_eeg(n_channels=2, n_samples=10)
        eeg.update({"setname": f"set{index + 1}", "subject": f"S{index + 1:02d}", "filename": f"set{index + 1}.set"})
        datasets.append(eeg)
    study, alleeg = pop_study(None, datasets, name="Study")

    spec = pop_study_dialog_spec(study, alleeg)
    controls = controls_by_tag(spec)

    assert controls["dataset_12_subject"].value == "S12"
    assert spec.scrollable is True


def test_pop_study_gui_updates_metadata_and_returns_python_history():
    study, alleeg = _study_inputs()
    renderer = _Renderer(
        {
            "name": "Edited",
            "task": "task",
            "notes": "notes",
            "dataset_1_subject": "S02",
            "dataset_1_session": "2",
            "dataset_1_run": "1",
            "dataset_1_condition": "standard",
            "dataset_1_group": "control",
        }
    )

    edited, edited_alleeg, command = pop_study(study, alleeg, gui=True, renderer=renderer, return_com=True)

    assert edited["name"] == "Edited"
    assert edited["datasetinfo"][0]["subject"] == "S02"
    assert edited["datasetinfo"][0]["session"] == 2
    assert edited_alleeg[0]["condition"] == "standard"
    ast.parse(command)


def test_pop_study_gui_cancel_is_noop():
    study, alleeg = _study_inputs()

    edited, edited_alleeg, command = pop_study(study, alleeg, gui=True, renderer=_Renderer(None), return_com=True)

    assert edited == study
    assert edited_alleeg == alleeg
    assert command == ""


def test_pop_studydesign_dialog_spec_lists_factors_and_current_design():
    study, alleeg = _study_inputs()

    spec = pop_studydesign_dialog_spec(study, alleeg)
    controls = controls_by_tag(spec)

    assert spec.title == "Edit STUDY design -- pop_studydesign()"
    assert spec.function_name == "pop_studydesign"
    assert spec.eeglab_source == "functions/studyfunc/pop_studydesign.m"
    assert controls["designind"].value == "1"
    assert "condition" in controls["available_factors"].value
    assert controls["variable1"].value == "condition"
    assert controls["available_factors"].enabled is False


def test_pop_studydesign_gui_updates_design_values():
    study, alleeg = _study_inputs()
    renderer = _Renderer(
        {
            "subjects": "S01",
            "name": "Targets",
            "designind": "1",
            "variable1": "condition",
            "values1": "target",
            "variable2": "",
            "values2": "",
        }
    )

    edited, _alleeg, command = pop_studydesign(study, alleeg, gui=True, renderer=renderer, return_com=True)

    assert edited["design"][0]["name"] == "Targets"
    assert edited["design"][0]["variable"][0]["value"] == ["target"]
    ast.parse(command)


def test_pop_studydesign_default_is_noninteractive_select():
    study, alleeg = _study_inputs()

    edited, edited_alleeg = pop_studydesign(study, alleeg, 1)

    assert edited["currentdesign"] == 1
    assert edited_alleeg == alleeg
