from unittest import mock
from eegprep.functions.guifunc.qt import file_dialog_kwargs, _select_file
from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS

def test_file_dialog_kwargs_option_persistence():
    original_option = EEG_OPTIONS.get("option_native_dialogs", 0)
    class FakeOption:
        DontUseNativeDialog = 4
        ShowDirsOnly = 1
    class FakeQFileDialog:
        Option = FakeOption
    qt_widgets = type("FakeQtWidgets", (), {"QFileDialog": FakeQFileDialog})

    try:
        # Default behavior: uses Qt non-native because option defaults to 0
        EEG_OPTIONS["option_native_dialogs"] = 0
        kwargs = file_dialog_kwargs(qt_widgets)
        assert kwargs == {"options": 4}

        kwargs_dirs = file_dialog_kwargs(qt_widgets, directories=True)
        assert kwargs_dirs == {"options": 5}

        # Option persistence: users opt-in
        EEG_OPTIONS["option_native_dialogs"] = 1
        kwargs_native = file_dialog_kwargs(qt_widgets)
        assert kwargs_native == {}

        # Constructor override takes precedence over option
        EEG_OPTIONS["option_native_dialogs"] = 0
        kwargs_override = file_dialog_kwargs(qt_widgets, native_file_dialogs=True)
        assert kwargs_override == {}
        
        # Override to False
        EEG_OPTIONS["option_native_dialogs"] = 1
        kwargs_override_false = file_dialog_kwargs(qt_widgets, native_file_dialogs=False)
        assert kwargs_override_false == {"options": 4}
    finally:
        EEG_OPTIONS["option_native_dialogs"] = original_option

def test_callback_driven_select_file(monkeypatch):
    captured = {}
    class FakeOption:
        DontUseNativeDialog = 4
        ShowDirsOnly = 1

    class FakeQFileDialog:
        Option = FakeOption
        @staticmethod
        def getOpenFileName(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return "test_file.txt", ""
            
        @staticmethod
        def getSaveFileName(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return "save_file.txt", ""

    qt_widgets = type("FakeQtWidgets", (), {"QFileDialog": FakeQFileDialog})
    monkeypatch.setattr("eegprep.functions.guifunc.qt._require_qt", lambda: (None, qt_widgets))
    
    class TargetWidget:
        def __init__(self):
            self.text_val = ""
        def setText(self, val):
            self.text_val = val

    original_option = EEG_OPTIONS.get("option_native_dialogs", 0)
    try:
        EEG_OPTIONS["option_native_dialogs"] = 0
        target = TargetWidget()
        params = {"caption": "My Open Dialog", "filter": "Text (*.txt)", "mode": "open"}
        _select_file(None, target, params, {})
        assert target.text_val == "test_file.txt"
        assert captured["args"][1] == "My Open Dialog"
        assert captured["args"][3] == "Text (*.txt)"
        assert captured["kwargs"] == {"options": 4}
        
        params_save = {"caption": "My Save Dialog", "mode": "save"}
        _select_file(None, target, params_save, {})
        assert target.text_val == "save_file.txt"
        assert captured["args"][1] == "My Save Dialog"
        assert captured["kwargs"] == {"options": 4}

        # Test persistence in select_file
        EEG_OPTIONS["option_native_dialogs"] = 1
        _select_file(None, target, params, {})
        assert captured["kwargs"] == {}

    finally:
        EEG_OPTIONS["option_native_dialogs"] = original_option
