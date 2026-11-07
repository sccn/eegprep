"""Offscreen GUI regression checks for the Stream 9 import flows."""

import os
import importlib

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest

pytest.importorskip('PySide6')
from PySide6 import QtCore, QtWidgets

from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher
from eegprep.functions.guifunc.eeglab_menu import eeglab_menus
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.functions.adminfunc.console import EEGPrepConsoleWorkspace
from tests.test_pop_loadset import _singleton_eeg
from tests.test_cnt_eeglab_tests import _write_cnt

pytestmark = [pytest.mark.gui, pytest.mark.modal_dialogs]


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def dialog_deadline(app):
    timed_out = []
    timer = QtCore.QTimer()

    def abort():
        timed_out.append(True)
        for widget in app.topLevelWidgets():
            if isinstance(widget, QtWidgets.QDialog):
                widget.reject()

    timer.timeout.connect(abort)
    timer.start(5000)
    yield
    timer.stop()
    assert not timed_out, 'Qt test exceeded its five-second dialog deadline'


def test_f18_mff_picker_selects_directory(app, tmp_path, monkeypatch):
    folder = tmp_path / 'recording.mff'
    folder.mkdir()
    calls = []

    def choose_directory(parent, title, directory, **kwargs):
        calls.append((title, kwargs))
        return str(folder)

    monkeypatch.setattr(QtWidgets.QFileDialog, 'getExistingDirectory', choose_directory)
    dispatcher = MenuActionDispatcher(EEGPrepSession(), native_file_dialogs=False)
    assert dispatcher._open_import_filename('pop_fileio_mff', None) == str(folder)
    assert calls[0][1]['options'] & QtWidgets.QFileDialog.Option.ShowDirsOnly


@pytest.mark.parametrize(
    'name,kwargs,required',
    [
        (
            'pop_importdata',
            {},
            {
                'data',
                'dataformat',
                'setname',
                'srate',
                'pnts',
                'xmin',
                'nbchan',
                'ref',
                'subject',
                'condition',
                'session',
                'group',
                'chanlocs',
                'icaweights',
                'icasphere',
            },
        ),
        (
            'pop_biosig',
            {'filename': 'unused.edf'},
            {'channels', 'blockrange', 'ref', 'bdfeventmode', 'importevent', 'rmeventchan'},
        ),
        ('pop_loadbv', {'path': 'unused.vhdr'}, {'chans', 'srange'}),
        ('pop_fileio', {'filename': 'unused.eeg'}, {'channels', 'samples', 'trials'}),
        ('pop_loadcnt', {'filename': 'unused.cnt'}, {'dataformat', 't1', 'lddur', 'keystroke'}),
        ('pop_readegi', {'filename': 'unused.raw'}, {'datachunks', 'forceversion', 'fileloc'}),
        (
            'pop_importevent',
            {'EEG': 'continuous'},
            {'event', 'indices', 'append', 'fields', 'skipline', 'timeunit', 'align', 'optimalign'},
        ),
        (
            'pop_importepoch',
            {'EEG': 'epoched'},
            {
                'filename',
                'fieldlist',
                'latencyfields',
                'durationfields',
                'typefield',
                'timeunit',
                'headerlines',
                'clearevents',
            },
        ),
        (
            'pop_chanevent',
            {'EEG': 'continuous'},
            {'chan', 'oper', 'edge', 'edgelen', 'duration', 'delchan', 'delevent', 'nbtype'},
        ),
    ],
)
def test_f10_import_dialogs_cancel_without_reading_or_mutating(app, name, kwargs, required):

    arguments = dict(kwargs)
    if 'EEG' in arguments:
        shape = (2, 10, 2) if arguments['EEG'] == 'epoched' else (2, 10)
        arguments['EEG'] = _singleton_eeg(np.zeros(shape))
    found = []

    def cancel():
        dialog = app.activeModalWidget()
        if dialog is not None:
            found.append({widget.objectName() for widget in dialog.findChildren(QtWidgets.QWidget)})
            dialog.reject()

    timer = QtCore.QTimer()
    timer.timeout.connect(cancel)
    timer.start(10)
    try:
        function = getattr(importlib.import_module(f'eegprep.functions.popfunc.{name}'), name)
        out, command = function(**arguments, gui=True, return_com=True)
    finally:
        timer.stop()
    assert found and required <= found[0]
    assert command == ''
    if 'EEG' in arguments:
        np.testing.assert_array_equal(out['data'], arguments['EEG']['data'])


@pytest.mark.parametrize(
    'action,shape,values,expected',
    [
        ('pop_importdata', None, {'srate': '128', 'nbchan': '2', 'setname': 'imported'}, 'pop_importdata'),
        (
            'pop_importevent',
            (2, 100),
            {'fields': 'type latency', 'skipline': '1', 'timeunit': '1', 'align': 'NaN'},
            'pop_importevent',
        ),
        ('pop_importepoch', (2, 100, 2), {'fieldlist': 'response', 'headerlines': '1'}, 'pop_importepoch'),
        ('pop_chanevent', (2, 100), {'chan': '2', 'delchan': False}, 'pop_chanevent'),
    ],
)
def test_f10_accept_menu_import_updates_shared_session(app, tmp_path, action, shape, values, expected):

    path = tmp_path / 'input.txt'
    if action == 'pop_importdata':
        np.savetxt(path, np.arange(40).reshape(2, 20))
        values = dict(values, data=str(path))
    elif action == 'pop_importevent':
        path.write_text('type latency\nstim 0.25\n')
        values = dict(values, event=str(path))
    elif action == 'pop_importepoch':
        path.write_text('response\nleft\nright\n')
        values = dict(values, filename=str(path))
    session = EEGPrepSession()
    bridge = EEGPrepConsoleWorkspace(session)
    if shape:
        data = np.zeros(shape)
        data[1, 10:20] = 2
        session.store_current(_singleton_eeg(data), new=True)
    seen = []

    def accept():
        dialog = app.activeModalWidget()
        if dialog is None:
            return
        seen.append(dialog.objectName())
        if dialog.objectName() == action:
            for tag, value in values.items():
                widget = dialog.findChild(QtWidgets.QWidget, tag)
                if isinstance(widget, QtWidgets.QCheckBox):
                    widget.setChecked(value)
                else:
                    widget.setText(value)
        dialog.accept()

    timer = QtCore.QTimer()
    timer.timeout.connect(accept)
    timer.start(10)
    try:
        MenuActionDispatcher(session, native_file_dialogs=False).dispatch(action)
    finally:
        timer.stop()
    assert bridge.namespace['EEG'] is session.EEG
    assert bridge.namespace['LASTCOM'] == session.LASTCOM
    bridge.close()
    assert action in seen
    assert expected in '\n'.join(session.ALLCOM)
    assert session.EEG is session.ALLEEG[session.CURRENTSET[0] - 1]
    if action == 'pop_importdata':
        assert session.EEG['srate'] == 128
        assert session.EEG['setname'] == 'imported'
    elif action == 'pop_importevent':
        assert session.EEG['event'][0]['latency'] == 26
    elif action == 'pop_importepoch':
        assert len(session.EEG['event']) == 2
    else:
        assert [event['latency'] for event in session.EEG['event']] == [10, 21]


def test_f31_cancel_erplab_picker_preserves_session(app, monkeypatch):
    session = EEGPrepSession()
    session.store_current(_singleton_eeg(np.zeros((2, 20))), new=True)
    before = list(session.ALLCOM)
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName', lambda *args, **kwargs: ('', ''))
    MenuActionDispatcher(session, native_file_dialogs=False).dispatch('pop_importerplab')
    assert session.ALLCOM == before
    assert session.CURRENTSET == [1]


def test_f23_cnt_menu_uses_native_reader_and_history(app, tmp_path, monkeypatch):

    path = tmp_path / 'recording.cnt'
    _write_cnt(path, np.arange(40).reshape(2, 20), dataformat='int16', rate=100)
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName', lambda *args, **kwargs: (str(path), ''))
    session = EEGPrepSession()
    seen = []

    def accept():
        dialog = app.activeModalWidget()
        if dialog is not None:
            seen.append(dialog.objectName())
            dialog.accept()

    timer = QtCore.QTimer()
    timer.timeout.connect(accept)
    timer.start(10)
    try:
        MenuActionDispatcher(session, native_file_dialogs=False).dispatch('pop_fileio_cnt')
    finally:
        timer.stop()
    assert 'pop_loadcnt' in seen
    assert session.EEG['data'].shape == (2, 20)
    assert 'pop_loadcnt(' in '\n'.join(session.ALLCOM)


def test_f32_import_picker_leaves_initial_directory_to_qt(app, monkeypatch):
    directories = []

    def cancel(parent, title, directory, filters, **kwargs):
        directories.append(directory)
        return '', ''

    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName', cancel)
    dispatcher = MenuActionDispatcher(EEGPrepSession(), native_file_dialogs=False)
    assert dispatcher._open_import_filename('pop_importdata', None) == ''
    assert directories == ['']


def test_f8_import_menu_keeps_pinned_labels_and_order():
    menus = eeglab_menus(all_menus=True)
    file_menu = next(item for item in menus if item.label == 'File')
    imported = next(item for item in file_menu.children if item.label == 'Import data')
    native = imported.children[0]
    labels = [item.label for item in native.children if item.action]
    assert labels[:3] == [
        'From ASCII/float file or MATLAB array',
        'From Biosemi BDF file (BIOSIG toolbox)',
        'From EDF/EDF+/GDF files (BIOSIG toolbox)',
    ]
    actions = [item.action for item in native.children]
    assert actions.index('pop_fileio_mff') < actions.index('pop_fileio_cnt') < actions.index('pop_fileio_brainvision')
