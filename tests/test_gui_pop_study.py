import ast
import copy
import os

import numpy as np
import pytest

from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.studyfunc.pop_addindepvar import (
    apply_addindepvar_combine,
    pop_addindepvar,
    pop_addindepvar_dialog_spec,
)
from eegprep.functions.studyfunc.pop_study import pop_study, pop_study_dialog_spec
from eegprep.functions.studyfunc.pop_studydesign import (
    apply_studydesign_add,
    apply_studydesign_delete,
    apply_studydesign_delete_variable,
    apply_studydesign_new_variable,
    apply_studydesign_rename,
    apply_studydesign_select,
    apply_studydesign_select_subjects,
    apply_studydesign_set_subjects,
    pop_studydesign,
    pop_studydesign_dialog_spec,
    studydesign_factor_lines,
)
from eegprep.functions.studyfunc.std_makedesign import std_makedesign
from eegprep.functions.studyfunc.std_maketrialinfo import std_maketrialinfo
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


def test_pop_study_dialog_spec_handles_numpy_component_lists():
    study, alleeg = _study_inputs()
    study["datasetinfo"][0]["comps"] = np.array([1, 2, 3, 4])

    spec = pop_study_dialog_spec(study, alleeg)
    controls = controls_by_tag(spec)

    assert controls["dataset_1_components"].string == "Comp.: 1 2 ..."
    assert controls["dataset_1_components"].value == [1, 2, 3, 4]


def test_pop_study_gui_clear_resets_components_to_all():
    # The Clear button leaves an empty component list on the row; OK must accept it
    # (comps=[] means all components, as in EEGLAB) instead of raising.
    study, alleeg = _study_inputs()
    study["datasetinfo"][0]["comps"] = [1, 2]
    renderer = _Renderer({"name": "Study", "task": "", "notes": "", "dataset_1_components": []})

    edited, _edited_alleeg, command = pop_study(study, alleeg, gui=True, renderer=renderer, return_com=True)

    assert edited["datasetinfo"][0]["comps"] == []
    assert "'comps', []" in command
    ast.parse(command)


def test_pop_study_component_selection_propagates_to_same_recording_rows():
    # EEGLAB pop_study applies a component selection to every dataset with the same
    # subject, session, and run (they share one ICA decomposition).
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from eegprep.functions.guifunc.qt import _VALUE_PROPERTY, QtDialogRenderer, _require_qt

    _qt_core, qt_widgets = _require_qt()
    qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    datasets = []
    for setname, subject, condition in (("a", "S01", "target"), ("b", "S01", "standard"), ("c", "S02", "target")):
        eeg = create_test_eeg(n_channels=2, n_samples=10)
        eeg.update(
            {"setname": setname, "subject": subject, "session": 1, "condition": condition, "filename": f"{setname}.set"}
        )
        datasets.append(eeg)
    study, alleeg = pop_study(None, datasets, name="Study")
    controls = controls_by_tag(pop_study_dialog_spec(study, alleeg))
    widgets = {}
    for tag, control in controls.items():
        if control.style == "edit":
            widgets[tag] = qt_widgets.QLineEdit(str(control.value or ""))
        elif control.style == "pushbutton":
            widgets[tag] = qt_widgets.QPushButton(control.string)

    QtDialogRenderer._apply_study_component_selection(widgets, controls["dataset_1_components"].callback.params, [1, 2])

    assert widgets["dataset_1_components"].property(_VALUE_PROPERTY) == [1, 2]
    assert widgets["dataset_2_components"].property(_VALUE_PROPERTY) == [1, 2]  # same subject/session/run
    assert widgets["dataset_2_components"].text() == "Comp.: 1 2"
    assert widgets["dataset_3_components"].property(_VALUE_PROPERTY) is None  # other subject: untouched
    assert widgets["dataset_3_components"].text() == "All comp."


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


def _reaction_time_study():
    """Epoched STUDY whose time-locking events carry a continuous ``rt`` field.

    This is the state ``event_processing_study`` reaches before
    ``STUDY > Select/Edit study design(s)`` > New > Add variable.
    """
    eeg = create_test_eeg(n_channels=2, n_samples=10, n_trials=3)
    for index, event in enumerate(eeg["event"]):
        event["epoch"] = index + 1
        event["rt"] = np.empty((0, 0)) if index == 2 else (300.0 if index == 0 else 450.0)
    eeg.update({"setname": "rt", "subject": "S01", "filename": "rt.set"})
    return pop_study(None, [eeg], name="RT study")


def test_pop_studydesign_new_button_opens_add_variable_for_rt():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from eegprep.functions.guifunc.qt import QtDialogRenderer, _require_qt

    try:
        qt_core, qt_widgets = _require_qt()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    study, alleeg = _reaction_time_study()
    spec = pop_studydesign_dialog_spec(study, alleeg)
    assert controls_by_tag(spec)["new_variable"].callback.name == "studydesign_new_variable"
    renderer = QtDialogRenderer()
    app, dialog, widgets = renderer.build_dialog(spec)
    observed: dict[str, object] = {"tries": 0}

    def choose_rt():
        observed["tries"] = int(observed["tries"]) + 1
        child = app.activeModalWidget()
        if child is None or child.windowTitle() != "Add variable":
            if int(observed["tries"]) > 50:
                for widget in app.topLevelWidgets():
                    if widget.windowTitle() == "Add variable":
                        widget.reject()
                return
            qt_core.QTimer.singleShot(10, choose_rt)
            return
        try:
            observed["title"] = child.windowTitle()
            factors = child.findChild(qt_widgets.QListWidget, "lbfact0")
            for row in range(factors.count()):
                if factors.item(row).text() == "rt":
                    factors.setCurrentRow(row)
                    break
            vartype = child.findChild(qt_widgets.QComboBox, "vartype")
            values = child.findChild(qt_widgets.QListWidget, "lbval0")
            observed["vartype"] = vartype.currentText()
            observed["values_enabled"] = values.isEnabled()
            observed["values"] = [values.item(row).text() for row in range(values.count())]
        finally:
            child.accept()

    qt_core.QTimer.singleShot(10, choose_rt)
    widgets["new_variable"].click()
    dialog.close()

    stored = widgets["variable_summary"].property("eegprep_value")
    assert observed["title"] == "Add variable"
    assert observed["vartype"] == "Continuous variable"
    assert observed["values_enabled"] is False
    assert observed["values"] == ["300", "450"]
    assert stored[-1]["label"] == "rt"
    assert stored[-1]["vartype"] == "continuous"
    assert stored[-1]["value"] == []


def test_pop_studydesign_new_variable_lists_and_stores_continuous_rt():
    study, alleeg = _reaction_time_study()
    before_study, before_alleeg = copy.deepcopy(study), copy.deepcopy(alleeg)
    controls = controls_by_tag(pop_studydesign_dialog_spec(study, alleeg))
    callback = controls["new_variable"].callback

    assert callback.name == "studydesign_new_variable"
    assert "rt" in callback.params["varlist"]["factors"]
    rt_index = callback.params["varlist"]["factors"].index("rt")
    assert callback.params["varlist"]["numerical"][rt_index] is True
    assert "trialinfo" not in study["datasetinfo"][0] or not any(
        "rt" in row for row in study["datasetinfo"][0].get("trialinfo") or []
    )

    renderer = _Renderer(
        {
            "subjects": "",
            "designind": 1,
            "variable_summary": [{"label": "rt", "value": [], "vartype": "continuous"}],
        }
    )
    edited, edited_alleeg, command = pop_studydesign(study, alleeg, gui=True, renderer=renderer, return_com=True)

    rt_variable = edited["design"][0]["variable"][0]
    assert rt_variable["label"] == "rt"
    assert rt_variable["vartype"] == "continuous"
    assert rt_variable["pairing"] == "off"
    assert rt_variable["level"] == "one"
    assert [float(value) for value in rt_variable["value"]] == [300.0, 450.0]
    trialinfo = edited["datasetinfo"][0]["trialinfo"]
    assert len(trialinfo) == 3
    assert [row.get("rt") for row in trialinfo if "rt" in row] == [300.0, 450.0]
    assert [row.get("rt") for row in edited_alleeg[0]["trialinfo"] if "rt" in row] == [300.0, 450.0]
    assert "std_maketrialinfo" in command
    assert "variable1='rt'" in command
    assert "vartype1='continuous'" in command
    ast.parse(command)
    namespace = {
        "STUDY": before_study,
        "ALLEEG": before_alleeg,
        "std_makedesign": std_makedesign,
        "std_maketrialinfo": std_maketrialinfo,
    }
    exec(command, namespace)
    assert namespace["STUDY"]["design"][0]["variable"][0]["label"] == "rt"
    assert namespace["STUDY"]["design"][0]["variable"][0]["vartype"] == "continuous"


def test_pop_addindepvar_dialog_selects_categorical_values_and_combines_them():
    varlist = {
        "factors": ["None", "condition"],
        "factorvals": [[], ["standard", "target", "oddball"]],
        "numerical": [True, False],
    }
    renderer = _Renderer({"lbfact0": 2, "lbval0": [2], "vartype": 1})

    label, values, categorical = pop_addindepvar(varlist, gui=True, renderer=renderer)

    assert label == "condition"
    assert values == ["target"]
    assert categorical == 1

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from eegprep.functions.guifunc.qt import QtDialogRenderer, _require_qt

    try:
        _require_qt()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    spec = pop_addindepvar_dialog_spec(varlist)
    state = controls_by_tag(spec)["lbfact0"].callback.params
    qt_renderer = QtDialogRenderer()
    _app, dialog, widgets = qt_renderer.build_dialog(spec)
    widgets["lbfact0"].setCurrentRow(1)
    for row in range(widgets["lbval0"].count()):
        widgets["lbval0"].item(row).setSelected(row < 2)
    warning = apply_addindepvar_combine(widgets, state)
    dialog.close()

    assert warning is None
    assert state["factorvals"][1][-1] == ["standard", "target"]


def _two_condition_study():
    datasets = []
    for subject, condition in (("S01", "target"), ("S02", "standard")):
        eeg = create_test_eeg(n_channels=2, n_samples=10)
        eeg.update({"setname": subject, "subject": subject, "condition": condition, "filename": f"{subject}.set"})
        datasets.append(eeg)
    return pop_study(None, datasets, name="Study")


def _editor_state(study, alleeg):
    return controls_by_tag(pop_studydesign_dialog_spec(study, alleeg))["new_design"].callback.params


def _commit_editor(study, alleeg, state, **result):
    payload = {"designind": state, "subjects": state.get("subject_text") or "", "resave": False}
    payload.update(result)
    return pop_studydesign(study, alleeg, gui=True, renderer=_Renderer(payload), return_com=True)


def test_pop_studydesign_controls_are_wired():
    study, alleeg = _two_condition_study()
    controls = controls_by_tag(pop_studydesign_dialog_spec(study, alleeg))
    expected = {
        "subjects": "studydesign_set_subjects",
        "select_subjects": "studydesign_select_subjects",
        "new_design": "studydesign_add_design",
        "rename_design": "studydesign_rename_design",
        "delete_design": "studydesign_delete_design",
        "designind": "studydesign_select_design",
        "new_variable": "studydesign_new_variable",
        "edit_variable": "studydesign_edit_variable",
        "delete_variable": "studydesign_delete_variable",
        "list_factors": "studydesign_list_factors",
    }
    for tag, name in expected.items():
        assert controls[tag].callback.name == name
    spec = pop_studydesign_dialog_spec(study, alleeg)
    assert spec.help_label == "Web help"
    assert spec.help_text == "pop_studydesign"
    assert controls["resave"].value is True
    assert "show_message" not in {control.callback.name for control in spec.controls if control.callback}


def test_pop_studydesign_add_rename_and_delete_match_std_makedesign():
    study, alleeg = _two_condition_study()
    state = _editor_state(study, alleeg)

    assert apply_studydesign_delete(state) == "The first STUDY design cannot be removed, only modified"
    apply_studydesign_add(state)
    apply_studydesign_rename(state, "Only S01")
    apply_studydesign_select_subjects(state, ["S01"])
    edited, _alleeg, command = _commit_editor(study, alleeg, state)

    reference = std_makedesign(
        copy.deepcopy(study),
        copy.deepcopy(alleeg),
        2,
        name="Only S01",
        defaultdesign="forceoff",
        subjselect=["S01"],
        return_com=True,
    )[0]
    assert [design["name"] for design in edited["design"]] == ["STUDY.design 1", "Only S01"]
    assert edited["design"][1]["cases"]["value"] == ["S01"]
    assert edited["design"][1]["variable"] == reference["design"][1]["variable"]
    assert "std_makedesign" in command
    assert "name='Only S01'" in command
    assert "subjselect=['S01']" in command
    assert "variable1='none'" in command
    assert "variable2='none'" in command
    assert "defaultdesign='off'" in command
    ast.parse(command)
    namespace = {"STUDY": copy.deepcopy(study), "ALLEEG": copy.deepcopy(alleeg), "std_makedesign": std_makedesign}
    exec(command, namespace)
    assert namespace["STUDY"]["design"][1]["variable"] == []

    state = _editor_state(edited, alleeg)
    apply_studydesign_select(state, 2)
    assert apply_studydesign_delete(state) is None
    deleted, _alleeg, delete_command = _commit_editor(edited, alleeg, state)
    assert len(deleted["design"]) == 1
    assert "del STUDY['design'][1]" in delete_command


def test_pop_studydesign_subject_text_rejects_unknown_names():
    study, alleeg = _two_condition_study()
    state = _editor_state(study, alleeg)
    warning = apply_studydesign_set_subjects(state, "S99")
    assert warning == "Unknown subject(s) - check syntax"
    assert state["designs"][0]["cases"]["value"] == ["S01", "S02"]
    assert apply_studydesign_set_subjects(state, "S02") is None
    edited, _alleeg, command = _commit_editor(study, alleeg, state)
    reference = std_makedesign(
        copy.deepcopy(study),
        copy.deepcopy(alleeg),
        1,
        name="STUDY.design 1",
        variable1="condition",
        values1=["standard", "target"],
        vartype1="categorical",
        subjselect=["S02"],
        defaultdesign="off",
        return_com=True,
    )[0]
    assert edited["design"][0]["cases"]["value"] == ["S02"]
    assert edited["design"][0]["variable"][0]["label"] == reference["design"][0]["variable"][0]["label"]
    assert edited["design"][0]["variable"][0]["value"] == reference["design"][0]["variable"][0]["value"]
    assert "subjselect=['S02']" in command


def test_pop_studydesign_edits_and_lists_factors_without_a_four_variable_cap():
    study, alleeg = _two_condition_study()
    state = _editor_state(study, alleeg)
    lines = studydesign_factor_lines(state["designs"][0], state["datasetinfo"])
    assert "2nd-level available statistics" in lines
    assert "* t-test/ANOVA using condition = target" in lines
    assert apply_studydesign_delete_variable(state) is True
    for _index in range(5):
        updated, warning = apply_studydesign_new_variable(
            state["variables"],
            state["varlist"],
            {"lbfact0": {"label": "condition", "values": ["target"], "categorical": 1}},
        )
        assert warning is None
        state["variables"] = updated
        state["designs"][0]["variable"] = updated
    edited, _alleeg, command = _commit_editor(study, alleeg, state)
    labels = [variable["label"] for variable in edited["design"][0]["variable"]]
    assert labels == ["condition", "condition", "condition", "condition"]
    assert "variable4='condition'" in command
    assert "variable5" not in command


def _gui():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from eegprep.functions.guifunc.qt import QtDialogRenderer, _require_qt

    try:
        qt_core, qt_widgets = _require_qt()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    return qt_core, qt_widgets, QtDialogRenderer


def _on_modal(qt_core, qt_widgets, predicate, action):
    def poll():
        app = qt_widgets.QApplication.instance()
        child = app.activeModalWidget()
        poll.tries += 1
        if child is not None and predicate(child):
            action(child)
            return
        if poll.tries > 50:
            if child is not None:
                child.reject()
            return
        qt_core.QTimer.singleShot(10, poll)

    poll.tries = 0
    qt_core.QTimer.singleShot(10, poll)


def test_pop_studydesign_offscreen_buttons_match_std_makedesign(tmp_path):
    qt_core, qt_widgets, renderer_type = _gui()
    study, alleeg = _two_condition_study()
    study["filename"] = "designs.study"
    study["filepath"] = str(tmp_path)
    before = copy.deepcopy(study)
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    _app, dialog, widgets = renderer_type().build_dialog(pop_studydesign_dialog_spec(study, alleeg))
    for tag in (
        "subjects",
        "select_subjects",
        "new_design",
        "rename_design",
        "delete_design",
        "designind",
        "new_variable",
        "edit_variable",
        "delete_variable",
        "list_factors",
        "variable_summary",
        "resave",
    ):
        assert widgets[tag].objectName() == tag
    assert dialog.findChild(qt_widgets.QPushButton, "ok").objectName() == "ok"
    assert dialog.findChild(qt_widgets.QPushButton, "cancel").objectName() == "cancel"
    assert dialog.findChild(qt_widgets.QPushButton, "help").objectName() == "help"

    def close_warning(child):
        assert "first STUDY design" in child.text()
        child.accept()

    _on_modal(qt_core, qt_widgets, lambda child: child.windowTitle() == "Warning", close_warning)
    widgets["delete_design"].click()
    assert widgets["designind"].count() == 1

    widgets["new_design"].click()
    assert [widgets["designind"].item(row).text() for row in range(widgets["designind"].count())] == [
        "STUDY.design 1",
        "Design 2",
    ]

    def rename(child):
        child.findChild(qt_widgets.QLineEdit, "answer0").setText("Renamed design")
        child.accept()

    _on_modal(qt_core, qt_widgets, lambda child: child.windowTitle() == "Rename Study Design", rename)
    widgets["rename_design"].click()
    assert widgets["designind"].currentItem().text() == "Renamed design"

    widgets["designind"].setCurrentRow(0)

    def read_factors(child):
        texts = [label.text() for label in child.findChildren(qt_widgets.QLabel)]
        assert "* t-test/ANOVA using condition = target" in texts
        child.accept()

    _on_modal(qt_core, qt_widgets, lambda child: child.windowTitle() == "List factors", read_factors)
    widgets["list_factors"].click()

    def keep_target(child):
        values = child.findChild(qt_widgets.QListWidget, "lbval0")
        for row in range(values.count()):
            values.item(row).setSelected(values.item(row).text() == "target")
        child.accept()

    _on_modal(qt_core, qt_widgets, lambda child: child.windowTitle() == "Add variable", keep_target)
    widgets["edit_variable"].click()
    assert "target" in widgets["variable_summary"].item(0).text()
    assert "standard" not in widgets["variable_summary"].item(0).text()

    def choose_s01(child):
        values = child.findChild(qt_widgets.QListWidget, "listboxvals")
        for row in range(values.count()):
            values.item(row).setSelected(values.item(row).text() == "S01")
        child.findChild(qt_widgets.QPushButton, "ok").click()

    _on_modal(qt_core, qt_widgets, lambda child: child.objectName() == "listdlg2", choose_s01)
    widgets["select_subjects"].click()
    assert widgets["subjects"].text().strip() == "S01"

    dialog.findChild(qt_widgets.QPushButton, "help").click()
    help_dialog = next(widget for widget in app.topLevelWidgets() if widget.objectName() == "pophelp")
    assert help_dialog.windowTitle().startswith("pop_studydesign")
    help_dialog.close()

    state = widgets["designind"].property("eegprep_value")
    edited, _alleeg, command = _commit_editor(study, alleeg, state, resave=True)
    reference = std_makedesign(
        copy.deepcopy(before),
        copy.deepcopy(alleeg),
        1,
        name="STUDY.design 1",
        variable1="condition",
        values1=["target"],
        vartype1="categorical",
        subjselect=["S01"],
        defaultdesign="off",
        return_com=True,
    )[0]
    assert edited["design"][0]["variable"][0]["value"] == reference["design"][0]["variable"][0]["value"]
    assert edited["design"][0]["cases"]["value"] == ["S01"]
    assert edited["design"][1]["name"] == "Renamed design"
    assert (tmp_path / "designs.study").is_file()
    assert "pop_savestudy" in command
    ast.parse(command)
    dialog.close()

    def cancel():
        active = qt_widgets.QApplication.instance().activeModalWidget()
        active.findChild(qt_widgets.QPushButton, "cancel").click()

    qt_core.QTimer.singleShot(20, cancel)
    cancelled, _alleeg, cancel_command = pop_studydesign(before, alleeg, gui=True, return_com=True)
    assert cancel_command == ""
    assert cancelled["design"][0]["cases"]["value"] == ["S01", "S02"]
