from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from eegprep.functions.guifunc.qt import QtDialogRenderer
from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset
from eegprep.functions.popfunc.pop_newset import pop_newset, pop_newset_dialog_spec


def _eeg(*, name: str = "demo") -> dict:
    pnts = 4
    eeg = eeg_emptyset()
    eeg.update(
        {
            "setname": name,
            "data": np.zeros((1, pnts), dtype=np.float32),
            "nbchan": 1,
            "pnts": pnts,
            "trials": 1,
            "srate": 100,
            "xmin": 0.0,
            "xmax": (pnts - 1) / 100,
            "times": np.arange(pnts, dtype=float),
            "chanlocs": [{"labels": "Cz", "theta": 0.0, "radius": 0.0, "ref": "common"}],
            "event": np.array([{"type": "stim", "latency": 1}], dtype=object),
            "saved": "no",
        }
    )
    return eeg


def test_pop_newset_stores_setname_and_retrieves_dataset():
    alleeg, current, current_set, command = pop_newset([], _eeg(), 0, "setname", "renamed")

    assert current_set == 1
    assert current["setname"] == "renamed"
    assert alleeg[0]["setname"] == "renamed"
    assert command == "[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'setname', 'renamed');"

    _alleeg, retrieved, retrieved_set, retrieve_command = pop_newset(alleeg, current, current_set, "retrieve", 1)

    assert retrieved_set == 1
    assert retrieved["setname"] == "renamed"
    assert retrieve_command == "[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'retrieve', 1);"


def test_pop_newset_rejects_unknown_keyword_options():
    with pytest.raises(ValueError, match="Unsupported pop_newset"):
        pop_newset([], _eeg(), 0, unknown=True)


def test_pop_newset_empty_retrieve_option_stores_current_dataset():
    alleeg, current, current_set, command = pop_newset([], _eeg(name="stored"), 0, "retrieve", [])

    assert current_set == 1
    assert current["setname"] == "stored"
    assert alleeg[0]["setname"] == "stored"
    assert command == "[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET);"


def test_pop_newset_overwrites_current_dataset_when_requested():
    alleeg, current, current_set, _command = pop_newset([], _eeg(name="original"), 0)
    updated = _eeg(name="updated")

    alleeg, current, current_set, command = pop_newset(alleeg, updated, current_set, "overwrite", "on")

    assert len(alleeg) == 1
    assert current_set == 1
    assert current["setname"] == "updated"
    assert alleeg[0]["setname"] == "updated"
    assert command == "[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'overwrite', 'on');"


def test_pop_newset_gui_choice_can_create_new_dataset():
    class Renderer:
        def run(self, _spec, initial_values=None):
            return {"setname": "processed", "comments": "new notes", "overwrite": 1}

    alleeg, current, current_set, _command = pop_newset([], _eeg(name="original"), 0)
    processed = _eeg(name="processed")

    alleeg, current, current_set, command = pop_newset(alleeg, processed, current_set, "gui", "on", renderer=Renderer())

    assert len(alleeg) == 2
    assert current_set == 2
    assert current["setname"] == "processed"
    assert current["comments"] == "new notes"
    assert command == (
        "[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, "
        "'setname', 'processed', 'comments', 'new notes', 'overwrite', 'off');"
    )


def test_pop_newset_dialog_old_dataset_prompt_hides_currentset_index():
    spec = pop_newset_dialog_spec(_eeg(name="processed"), 1)

    labels = [control.string for control in spec.controls]
    assert "What do you want to do with the old dataset (not modified since last saved)?" in labels
    assert "What do you want to do with the old dataset 1 (not modified since last saved)?" not in labels


def test_pop_newset_dialog_edit_description_opens_multiline_editor():
    eeg = _eeg(name="processed")
    eeg["comments"] = ["first line", "second line"]

    control = controls_by_tag(pop_newset_dialog_spec(eeg, 1))["editdescription"]

    assert control.callback is not None
    assert control.callback.name == "edit_text"
    assert control.callback.params == {
        "button": "editdescription",
        "target": "editdescription",
        "title": "Edit description",
        "label": "Dataset description:",
        "value": "first line\nsecond line",
    }


def test_qt_edit_text_callback_stores_accepted_text(monkeypatch):
    class QInputDialog:
        @staticmethod
        def getMultiLineText(_parent, title, label, text):
            calls.append((title, label, text))
            return "", True

    class Widget:
        def __init__(self):
            self.properties = {}

        def property(self, name):
            return self.properties.get(name)

        def setProperty(self, name, value):
            self.properties[name] = value

    calls = []
    target = Widget()
    QtWidgets = type("QtWidgets", (), {"QInputDialog": QInputDialog})
    monkeypatch.setattr("eegprep.functions.guifunc.qt._require_qt", lambda: (None, QtWidgets))

    QtDialogRenderer._edit_text(
        object(),
        target,
        {"title": "Edit description", "label": "Dataset description:", "value": "old notes"},
    )

    assert calls == [("Edit description", "Dataset description:", "old notes")]
    assert QtDialogRenderer._read_widget(target) == ""


def test_pop_newset_gui_description_button_value_updates_comments():
    class Renderer:
        def run(self, _spec, initial_values=None):
            return {"setname": "processed", "editdescription": "edited notes", "overwrite": 1}

    alleeg, current, current_set, _command = pop_newset([], _eeg(name="original"), 0)

    alleeg, current, current_set, command = pop_newset(
        alleeg, _eeg(name="processed"), current_set, "gui", "on", renderer=Renderer()
    )

    assert len(alleeg) == 2
    assert current_set == 2
    assert current["comments"] == "edited notes"
    assert command == (
        "[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, "
        "'setname', 'processed', 'comments', 'edited notes', 'overwrite', 'off');"
    )


def test_pop_newset_gui_choice_can_overwrite_current_dataset():
    class Renderer:
        def run(self, _spec, initial_values=None):
            return {"setname": "processed", "comments": "", "overwrite": 2}

    alleeg, current, current_set, _command = pop_newset([], _eeg(name="original"), 0)

    alleeg, current, current_set, command = pop_newset(
        alleeg, _eeg(name="processed"), current_set, "gui", "on", renderer=Renderer()
    )

    assert len(alleeg) == 1
    assert current_set == 1
    assert current["setname"] == "processed"
    assert command == (
        "[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'setname', 'processed', 'overwrite', 'on');"
    )


def test_pop_newset_gui_cancel_retrieves_original_without_history_command():
    class Renderer:
        def run(self, _spec, initial_values=None):
            return None

    alleeg, current, current_set, _command = pop_newset([], _eeg(name="original"), 0)

    alleeg, current, current_set, command = pop_newset(
        alleeg, _eeg(name="processed"), current_set, "gui", "on", renderer=Renderer()
    )

    assert len(alleeg) == 1
    assert current_set == 1
    assert current["setname"] == "original"
    assert command == ""


def test_pop_newset_saveold_and_savenew_call_pop_saveset():
    original = _eeg(name="original")
    original["filename"] = "original.set"
    original["filepath"] = "/tmp"
    processed = _eeg(name="processed")

    alleeg, current, current_set, _command = pop_newset([], original, 0)

    with mock.patch("eegprep.functions.popfunc.pop_newset.pop_saveset") as saveset:
        alleeg, current, current_set, _command = pop_newset(
            alleeg,
            processed,
            current_set,
            "overwrite",
            "on",
            "saveold",
            "on",
            "savenew",
            "/tmp/new.set",
        )

    assert [call.args[1] for call in saveset.call_args_list] == [str(Path("/tmp") / "original.set"), "/tmp/new.set"]
    assert current["saved"] == "yes"
    assert alleeg[current_set - 1]["saved"] == "yes"
