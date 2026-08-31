import ast

import numpy as np

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
    assert controls["name"].value == "Study"
    assert controls["dataset_1_filename"].value == "one.set"
    assert controls["dataset_1_subject"].value == "S01"
    assert controls["dataset_1_condition"].value == "target"
    assert controls["dataset_1_filename"].enabled is True
    assert controls["dataset_1_browse"].enabled is True
    assert controls["dataset_10_clear"].enabled is True


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
    assert "commands=" in command
    ast.parse(command)
    namespace = {"STUDY": study, "ALLEEG": alleeg, "pop_study": pop_study}
    exec(command, namespace)
    assert namespace["STUDY"]["datasetinfo"][0]["subject"] == "S02"
    assert namespace["ALLEEG"][0]["condition"] == "standard"


def test_pop_study_gui_ignores_untouched_components_button_label():
    study, alleeg = _study_inputs()
    renderer = _Renderer(
        {
            "name": "Edited",
            "task": "",
            "notes": "",
            "dataset_1_subject": "S02",
            "dataset_1_components": "All comp.",
        }
    )

    edited, _edited_alleeg, command = pop_study(study, alleeg, gui=True, renderer=renderer, return_com=True)

    assert edited["datasetinfo"][0]["subject"] == "S02"
    assert "comps" not in command
    ast.parse(command)


def test_pop_study_gui_records_selected_components():
    study, alleeg = _study_inputs()
    renderer = _Renderer(
        {
            "name": "Study",
            "task": "",
            "notes": "",
            "dataset_1_components": [1, 2],
        }
    )

    edited, _edited_alleeg, command = pop_study(study, alleeg, gui=True, renderer=renderer, return_com=True)

    assert edited["datasetinfo"][0]["comps"] == [1, 2]
    assert "'comps', [1, 2]" in command
    ast.parse(command)


def test_pop_study_gui_cancel_is_noop():
    study, alleeg = _study_inputs()

    edited, edited_alleeg, command = pop_study(study, alleeg, gui=True, renderer=_Renderer(None), return_com=True)

    assert edited == study
    assert edited_alleeg == alleeg
    assert command == ""


def test_pop_study_dialog_spec_handles_numpy_component_lists():
    study, alleeg = _study_inputs()
    study["datasetinfo"][0]["comps"] = np.array([1, 2, 3, 4])

    spec = pop_study_dialog_spec(study, alleeg)
    controls = controls_by_tag(spec)

    assert controls["dataset_1_components"].string == "Comp.: 1 2 ..."
    assert controls["dataset_1_components"].value == [1, 2, 3, 4]


def test_pop_studydesign_dialog_spec_lists_factors_and_current_design():
    first = create_test_eeg(n_channels=2, n_samples=10)
    first.update({"setname": "one", "subject": "S01", "condition": "target", "filename": "one.set"})
    second = create_test_eeg(n_channels=2, n_samples=10)
    second.update({"setname": "two", "subject": "S02", "condition": "standard", "filename": "two.set"})
    study, alleeg = pop_study(None, [first, second], name="Study")

    spec = pop_studydesign_dialog_spec(study, alleeg)
    controls = controls_by_tag(spec)

    assert spec.title == "Edit STUDY design -- pop_studydesign()"
    assert spec.function_name == "pop_studydesign"
    assert spec.eeglab_source == "functions/studyfunc/pop_studydesign.m"
    assert spec.help_label == "Web help"
    assert controls["designind"].value == 1
    assert controls["subjects"].value == ""
    assert "Categorical variable: condition" in controls["variable_summary"].string
    assert controls["new_design"].enabled is True
    assert controls["edit_variable"].enabled is True


def test_pop_studydesign_gui_updates_subject_selection():
    study, alleeg = _study_inputs()
    renderer = _Renderer(
        {
            "subjects": "S01",
            "designind": "1",
        }
    )

    edited, _alleeg, command = pop_studydesign(study, alleeg, gui=True, renderer=renderer, return_com=True)

    assert edited["design"][0]["cases"]["value"] == ["S01"]
    ast.parse(command)


def test_pop_studydesign_default_is_noninteractive_select():
    study, alleeg = _study_inputs()

    edited, edited_alleeg = pop_studydesign(study, alleeg, 1)

    assert edited["currentdesign"] == 1
    assert edited_alleeg == alleeg
