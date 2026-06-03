from __future__ import annotations

import ast
import io
import importlib
import logging
import sys
import textwrap
import warnings
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from eegprep.extension_runtime import ExtensionRuntime
from eegprep.extensions import (
    ExtensionPopFunction,
    ExtensionRecord,
    ExtensionSourceType,
    ExtensionSpec,
    ExtensionStatus,
    LazyImport,
)
from eegprep.functions.adminfunc import console as console_module
from eegprep.functions.adminfunc.console import EEGPrepConsoleWorkspace
from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher
from eegprep.functions.guifunc.session import EEGPrepSession


def _demo_eeg(setname: str = "demo"):
    return {
        "setname": setname,
        "data": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "nbchan": 2,
        "pnts": 2,
        "trials": 1,
        "srate": 100.0,
        "xmin": 0.0,
        "xmax": 0.01,
        "chanlocs": [{"labels": "Cz"}, {"labels": "Pz"}],
        "event": [],
        "urevent": [],
        "epoch": [],
    }


def _extension_runtime(spec: ExtensionSpec) -> ExtensionRuntime:
    return ExtensionRuntime.from_records(
        (
            ExtensionRecord(
                name=spec.name,
                status=ExtensionStatus.INSTALLED,
                spec=spec,
                source_type=ExtensionSourceType.INSTALLED,
            ),
        )
    )


def _write_extension_package(tmp_path, package: str, files: dict[str, str], monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(tmp_path))
    for module_name in list(sys.modules):
        if module_name == package or module_name.startswith(f"{package}."):
            del sys.modules[module_name]
    package_dir = tmp_path / package
    package_dir.mkdir(exist_ok=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    for relative_path, content in files.items():
        path = package_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    importlib.invalidate_caches()


def _fake_pop_reref(EEG, ref, *, return_com=False):
    output = dict(EEG, setname="reref", ref=list(ref))
    command = "EEG = pop_reref(EEG, []);"
    return (output, command) if return_com else output


def _fake_pop_without_command(eeg, *, return_com=False):
    output = dict(eeg, setname="no-history-command")
    return (output, "") if return_com else output


def _fake_pop_topoplot(eeg, *, return_com=False):
    command = "pop_topoplot(EEG, typeplot=1, items=[0])"
    return (["figure"], command) if return_com else ["figure"]


def _fake_pop_eegplot(
    eeg,
    icacomp=1,
    superpose=0,
    reject=1,
    *,
    command_callback=None,
    return_com=False,
):
    _fake_pop_eegplot.command_callback = command_callback
    command = f"pop_eegplot(EEG, {icacomp}, {superpose}, {reject})"
    return (None, command) if return_com else "window"


_fake_pop_eegplot.command_callback = None


def _fake_pop_jointprob_browser(eeg, *args, command_callback=None, return_com=False, **kwargs):
    del args, kwargs
    _fake_pop_jointprob_browser.command_callback = command_callback
    command = "EEG = pop_jointprob(EEG, 1, [1], 4, 4, 0, 1, 1);"
    return (eeg, command) if return_com else eeg


_fake_pop_jointprob_browser.command_callback = None


def _fake_pop_rejcont_browser(eeg, *args, command_callback=None, return_com=False, **kwargs):
    del args, kwargs
    _fake_pop_rejcont_browser.command_callback = command_callback
    return (eeg, "") if return_com else eeg


_fake_pop_rejcont_browser.command_callback = None


def _fake_pop_copyset(ALLEEG, set_in, set_out=None, *, return_com=False):
    copied = dict(ALLEEG[int(set_in) - 1], setname="copied")
    output = list(ALLEEG)
    output.append(copied)
    command = "[ALLEEG EEG CURRENTSET LASTCOM] = pop_copyset(ALLEEG, 1, 2);"
    return (output, copied, 2, command) if return_com else (output, copied, 2)


def _fake_pop_study(STUDY, ALLEEG, *, return_com=False):
    study = {"name": "console study", "datasetinfo": [{"index": 1, "subject": "S01"}], "design": []}
    command = "STUDY, ALLEEG = pop_study(STUDY, ALLEEG, name='console study')"
    return (study, ALLEEG, command) if return_com else (study, ALLEEG)


def _fake_pop_savestudy(STUDY, EEG=None, *, return_com=False):
    study = dict(STUDY, filename="console.study", filepath="/tmp", saved="yes")
    command = "STUDY = pop_savestudy(STUDY, ALLEEG, filename='console.study', filepath='/tmp')"
    return (study, command) if return_com else study


def _fake_pop_fresh_study(*, return_com=False):
    study = {"name": "fresh study", "design": []}
    command = "STUDY = pop_freshstudy()"
    return (study, command) if return_com else study


def test_workspace_starts_with_eeglab_style_names():
    session = EEGPrepSession()
    workspace = EEGPrepConsoleWorkspace(session, exports={})

    for name in ("EEG", "ALLEEG", "CURRENTSET", "ALLCOM", "LASTCOM", "STUDY", "CURRENTSTUDY"):
        assert name in workspace.namespace
    assert workspace.namespace["session"] is session
    assert callable(workspace.namespace["pop_newset"])


def test_session_changes_update_console_namespace():
    session = EEGPrepSession()
    workspace = EEGPrepConsoleWorkspace(session, exports={})
    eeg = _demo_eeg()

    session.store_current(eeg, new=True, command="EEG = demo;")

    assert workspace.namespace["EEG"] is session.EEG
    assert workspace.namespace["ALLEEG"] is session.ALLEEG
    assert workspace.namespace["CURRENTSET"] == 1
    assert workspace.namespace["LASTCOM"] == "EEG = demo;"


def test_console_pop_study_result_updates_shared_study_workspace():
    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    workspace = EEGPrepConsoleWorkspace(session, exports={"pop_study": _fake_pop_study})

    result = workspace.namespace["pop_study"](workspace.namespace["STUDY"], workspace.namespace["ALLEEG"])
    assigned_study, assigned_alleeg = result

    assert session.CURRENTSTUDY == 1
    assert session.STUDY["name"] == "console study"
    assert workspace.namespace["STUDY"] is session.STUDY
    assert session.ALLCOM[-1].startswith("STUDY, ALLEEG = pop_study(")
    assert assigned_study is session.STUDY
    assert assigned_alleeg is session.ALLEEG
    assert result.command == session.LASTCOM
    assert len(result) == 2


def test_console_pop_study_history_assignment_replays_as_written():
    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    workspace = EEGPrepConsoleWorkspace(session, exports={"pop_study": _fake_pop_study})
    source = "STUDY, ALLEEG = pop_study(STUDY, ALLEEG)"

    exec(source, workspace.namespace)
    workspace.after_execute(source)

    assert workspace.namespace["STUDY"] is session.STUDY
    assert workspace.namespace["ALLEEG"] is session.ALLEEG
    assert session.STUDY["name"] == "console study"
    assert session.ALLCOM[-1].startswith("STUDY, ALLEEG = pop_study(")


def test_console_pop_savestudy_result_updates_study_without_replacing_alleeg():
    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    session.STUDY = {"name": "console study", "datasetinfo": [], "design": []}
    session.CURRENTSTUDY = 1
    workspace = EEGPrepConsoleWorkspace(session, exports={"pop_savestudy": _fake_pop_savestudy})

    result = workspace.namespace["pop_savestudy"](workspace.namespace["STUDY"], workspace.namespace["EEG"])

    assert session.STUDY["filename"] == "console.study"
    assert session.ALLEEG[0]["setname"] == "demo"
    assert len(result) == 2
    assert session.ALLCOM[-1].startswith("STUDY = pop_savestudy(")


def test_console_pop_result_detects_fresh_study_without_datasetinfo():
    session = EEGPrepSession()
    workspace = EEGPrepConsoleWorkspace(session, exports={"pop_freshstudy": _fake_pop_fresh_study})

    result = workspace.namespace["pop_freshstudy"]()

    assert session.CURRENTSTUDY == 1
    assert session.STUDY["name"] == "fresh study"
    assert result.command == "STUDY = pop_freshstudy()"


def test_console_pop_precomp_result_updates_shared_study_history():
    from eegprep.functions.studyfunc.pop_precomp import pop_precomp
    from eegprep.functions.studyfunc.pop_study import pop_study

    session = EEGPrepSession()
    eeg = _demo_eeg()
    eeg["data"] = np.dstack([eeg["data"], eeg["data"] + 1.0])
    eeg["trials"] = 2
    session.store_current(eeg, new=True)
    session.STUDY, session.ALLEEG = pop_study(None, session.ALLEEG, name="console measures")
    session.CURRENTSTUDY = 1
    workspace = EEGPrepConsoleWorkspace(session, exports={"pop_precomp": pop_precomp})

    result = workspace.namespace["pop_precomp"](
        workspace.namespace["STUDY"], workspace.namespace["ALLEEG"], "channels", erp="on"
    )
    assigned_study, assigned_alleeg = result

    assert assigned_study is session.STUDY
    assert assigned_alleeg is session.ALLEEG
    assert session.STUDY["changrp"][0]["erpdata"]
    assert session.ALLCOM[-1].startswith("STUDY, ALLEEG = pop_precomp(")


def test_session_history_commands_do_not_echo_to_console():
    session = EEGPrepSession()
    writes = []
    workspace = EEGPrepConsoleWorkspace(session, command_echo=writes.append, exports={})

    session.store_current(_demo_eeg(), new=True, command="EEG = pop_loadset('demo.set');")

    assert workspace.namespace["LASTCOM"] == "EEG = pop_loadset('demo.set');"
    assert writes == []


def test_command_echo_is_separate_from_session_history():
    session = EEGPrepSession()
    writes = []
    workspace = EEGPrepConsoleWorkspace(session, command_echo=writes.append, exports={})

    session.echo_command("EEG = pop_resample( EEG, 64);")
    assert session.ALLCOM == []

    session.add_history("EEG = pop_resample( EEG, 64);")

    assert writes == ["EEG = pop_resample( EEG, 64);"]
    assert session.ALLCOM == ["EEG = pop_resample( EEG, 64);"]
    workspace.close()


def test_gui_action_buffers_output_until_command_echo():
    session = EEGPrepSession()
    stream = io.StringIO()
    workspace = EEGPrepConsoleWorkspace(
        session,
        command_echo=lambda command: console_module._terminal_write(f"In [1]: {command}\n", stream=stream, sync=True),
        exports={},
    )

    with session.gui_action("pop_demo"):
        console_module._terminal_write("WARNING before command\n", stream=stream)
        session.echo_command("EEG = pop_demo(EEG);")

    output = stream.getvalue()
    assert output.index("In [1]: EEG = pop_demo(EEG);") < output.index("WARNING before command")
    workspace.close()


def test_nested_workspace_gui_buffers_restore_previous_buffer():
    first_session = EEGPrepSession()
    second_session = EEGPrepSession()
    first_stream = io.StringIO()
    second_stream = io.StringIO()
    first_workspace = EEGPrepConsoleWorkspace(
        first_session,
        command_echo=lambda command: console_module._terminal_write(
            f"In [1]: {command}\n", stream=first_stream, sync=True
        ),
        exports={},
    )
    second_workspace = EEGPrepConsoleWorkspace(
        second_session,
        command_echo=lambda command: console_module._terminal_write(
            f"In [1]: {command}\n", stream=second_stream, sync=True
        ),
        exports={},
    )

    try:
        first_session.begin_gui_action("first")
        console_module._terminal_write("first warning before nested\n", stream=first_stream)
        second_session.begin_gui_action("second")
        console_module._terminal_write("second warning\n", stream=second_stream)
        second_session.echo_command("EEG = pop_second(EEG);")
        second_session.end_gui_action("second")
        console_module._terminal_write("first warning after nested\n", stream=first_stream)
        first_session.echo_command("EEG = pop_first(EEG);")
        first_session.end_gui_action("first")
    finally:
        first_workspace.close()
        second_workspace.close()

    first_output = first_stream.getvalue()
    second_output = second_stream.getvalue()
    assert first_output.index("In [1]: EEG = pop_first(EEG);") < first_output.index("first warning before nested")
    assert first_output.index("In [1]: EEG = pop_first(EEG);") < first_output.index("first warning after nested")
    assert "second warning" not in first_output
    assert second_output.index("In [1]: EEG = pop_second(EEG);") < second_output.index("second warning")
    assert "first warning" not in second_output


def test_gui_action_buffers_logger_warnings_until_command_echo():
    logger = logging.getLogger("eegprep.tests.console_gui")
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(logging.Formatter("WARNING (%(name)s) %(message)s"))
    logger.addHandler(handler)

    session = EEGPrepSession()
    workspace = EEGPrepConsoleWorkspace(
        session,
        command_echo=lambda command: console_module._terminal_write(f"In [1]: {command}\n", stream=output, sync=True),
        exports={},
    )
    restore = console_module._install_prompt_safe_logging()
    try:
        with session.gui_action("pop_demo"):
            logger.warning("logger warning before command")
            session.echo_command("EEG = pop_demo(EEG);")
    finally:
        restore()
        workspace.close()
        logger.removeHandler(handler)
        logger.propagate = True

    console_output = output.getvalue()
    assert console_output.index("In [1]: EEG = pop_demo(EEG);") < console_output.index(
        "WARNING (eegprep.tests.console_gui) logger warning before command"
    )


def test_gui_action_buffers_pop_interp_logger_until_command_echo():
    from eegprep.functions.popfunc.pop_interp import logger

    logger.setLevel(logging.WARNING)
    logger.propagate = False
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(logging.Formatter("WARNING (%(name)s) %(message)s"))
    logger.addHandler(handler)

    session = EEGPrepSession()
    workspace = EEGPrepConsoleWorkspace(
        session,
        command_echo=lambda command: console_module._terminal_write(f"In [1]: {command}\n", stream=output, sync=True),
        exports={},
    )
    restore = console_module._install_prompt_safe_logging()
    try:
        with session.gui_action("pop_interp"):
            logger.warning("interpolation can be done on the fly in studies")
            logger.warning("this function will actually create channels in the dataset")
            logger.warning("do not interpolate channels before running ICA")
            session.echo_command("EEG = pop_interp(EEG, bad_elec=[2], method='spherical', t_range=[2, 3])")
    finally:
        restore()
        workspace.close()
        logger.removeHandler(handler)
        logger.propagate = True

    console_output = output.getvalue()
    assert console_output.index("In [1]: EEG = pop_interp") < console_output.index(
        "WARNING (eegprep.functions.popfunc.pop_interp) interpolation can be done on the fly in studies"
    )


def test_gui_action_writes_logger_warnings_synchronously_after_command_echo():
    logger = logging.getLogger("eegprep.tests.console_gui_after")
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(logging.Formatter("WARNING (%(name)s) %(message)s"))
    logger.addHandler(handler)

    session = EEGPrepSession()
    workspace = EEGPrepConsoleWorkspace(
        session,
        command_echo=lambda command: console_module._terminal_write(f"In [1]: {command}\n", stream=output, sync=True),
        exports={},
    )
    restore = console_module._install_prompt_safe_logging()
    try:
        with session.gui_action("pop_demo"):
            session.echo_command("EEG = pop_demo(EEG);")
            with mock.patch.object(console_module.importlib, "import_module") as import_module:
                logger.warning("logger warning after command")
    finally:
        restore()
        workspace.close()
        logger.removeHandler(handler)
        logger.propagate = True

    import_module.assert_not_called()
    console_output = output.getvalue()
    assert console_output.index("In [1]: EEG = pop_demo(EEG);") < console_output.index(
        "WARNING (eegprep.tests.console_gui_after) logger warning after command"
    )


def test_gui_action_buffers_late_created_logger_handlers_until_command_echo():
    logger = logging.getLogger("eegprep.tests.console_gui_late")
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    output = io.StringIO()

    session = EEGPrepSession()
    workspace = EEGPrepConsoleWorkspace(
        session,
        command_echo=lambda command: console_module._terminal_write(f"In [1]: {command}\n", stream=output, sync=True),
        exports={},
    )
    restore = console_module._install_prompt_safe_logging()
    handler = logging.StreamHandler(output)
    handler.setFormatter(logging.Formatter("WARNING (%(name)s) %(message)s"))
    logger.addHandler(handler)
    try:
        with session.gui_action("pop_demo"):
            logger.warning("late handler warning before command")
            session.echo_command("EEG = pop_demo(EEG);")
    finally:
        logger.removeHandler(handler)
        logger.propagate = True
        restore()
        workspace.close()

    console_output = output.getvalue()
    assert console_output.index("In [1]: EEG = pop_demo(EEG);") < console_output.index(
        "WARNING (eegprep.tests.console_gui_late) late handler warning before command"
    )


def test_gui_action_releases_store_current_output_at_end_without_command_echo():
    session = EEGPrepSession()
    stream = io.StringIO()
    fake_module = SimpleNamespace(run_in_terminal=lambda callback: callback())
    workspace = EEGPrepConsoleWorkspace(
        session,
        command_echo=lambda command: console_module._terminal_write(f"In [1]: {command}\n", stream=stream, sync=True),
        exports={},
    )

    with mock.patch.object(console_module.importlib, "import_module", return_value=fake_module):
        with session.gui_action("pop_loadset"):
            console_module._terminal_write("WARNING before command\n", stream=stream)
            session.store_current(_demo_eeg(), new=True, command="EEG = pop_loadset('demo.set');")

    output = stream.getvalue()
    assert "In [1]:" not in output
    assert "WARNING before command" in output
    workspace.close()


def test_gui_action_releases_add_history_output_at_end_without_command_echo():
    session = EEGPrepSession()
    stream = io.StringIO()
    fake_module = SimpleNamespace(run_in_terminal=lambda callback: callback())
    workspace = EEGPrepConsoleWorkspace(
        session,
        command_echo=lambda command: console_module._terminal_write(f"In [1]: {command}\n", stream=stream, sync=True),
        exports={},
    )

    with mock.patch.object(console_module.importlib, "import_module", return_value=fake_module):
        with session.gui_action("pop_export"):
            console_module._terminal_write("WARNING before command\n", stream=stream)
            session.add_history("LASTCOM = pop_export(EEG, 'demo.tsv');")

    output = stream.getvalue()
    assert "In [1]:" not in output
    assert "WARNING before command" in output
    workspace.close()


def test_gui_action_without_command_releases_output_through_terminal_redraw():
    session = EEGPrepSession()
    stream = io.StringIO()
    calls = []

    def fake_run_in_terminal(callback):
        calls.append(callback)
        callback()

    fake_module = SimpleNamespace(run_in_terminal=fake_run_in_terminal)
    workspace = EEGPrepConsoleWorkspace(session, command_echo=mock.Mock(), exports={})

    with mock.patch.object(console_module.importlib, "import_module", return_value=fake_module) as import_module:
        with session.gui_action("pop_runica"):
            console_module._terminal_write("ERROR before prompt\n", stream=stream)

    import_module.assert_called_once_with("prompt_toolkit.application.run_in_terminal")
    assert len(calls) == 1
    assert stream.getvalue() == "ERROR before prompt\n"
    workspace.close()


@pytest.mark.parametrize(
    ("action", "patch_target"),
    [
        ("pop_adjustevents", "eegprep.functions.popfunc.pop_adjustevents.pop_adjustevents"),
        ("pop_chanedit", "eegprep.functions.popfunc.pop_chanedit.pop_chanedit"),
        ("pop_clean_rawdata", "eegprep.plugins.clean_rawdata.pop_clean_rawdata.pop_clean_rawdata"),
        ("pop_comments", "eegprep.functions.popfunc.pop_comments.pop_comments"),
        ("pop_editset", "eegprep.functions.popfunc.pop_editset.pop_editset"),
        ("pop_editeventfield", "eegprep.functions.popfunc.pop_editeventfield.pop_editeventfield"),
        ("pop_editeventvals", "eegprep.functions.popfunc.pop_editeventvals.pop_editeventvals"),
        ("pop_epoch", "eegprep.functions.popfunc.pop_epoch.pop_epoch"),
        ("pop_reref", "eegprep.functions.popfunc.pop_reref.pop_reref"),
        ("pop_interp", "eegprep.functions.popfunc.pop_interp.pop_interp"),
        ("pop_resample", "eegprep.functions.popfunc.pop_resample.pop_resample"),
        ("pop_rmdat", "eegprep.functions.popfunc.pop_rmdat.pop_rmdat"),
        ("pop_runica", "eegprep.functions.popfunc.pop_runica.pop_runica"),
        ("pop_select", "eegprep.functions.popfunc.pop_select.pop_select"),
        ("pop_selectevent", "eegprep.functions.popfunc.pop_selectevent.pop_selectevent"),
        ("pop_iclabel", "eegprep.plugins.ICLabel.pop_iclabel.pop_iclabel"),
        ("pop_icflag", "eegprep.plugins.ICLabel.pop_icflag.pop_icflag"),
        ("pop_subcomp", "eegprep.functions.popfunc.pop_subcomp.pop_subcomp"),
    ],
)
def test_gui_pop_action_warning_output_follows_echoed_command(action, patch_target):
    from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher

    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    stream = io.StringIO()
    line_numbers = {"next": 1}

    def command_echo(command):
        line_number = line_numbers["next"]
        line_numbers["next"] += 1
        console_module._terminal_write(f"In [{line_number}]: {command}\n", stream=stream, sync=True)

    workspace = EEGPrepConsoleWorkspace(session, command_echo=command_echo, exports={})
    dispatcher = MenuActionDispatcher(session)
    command = f"EEG = {action}(EEG);"

    def fake_pop(eeg, *args, **kwargs):
        assert kwargs["return_com"] is True
        warnings.warn("warning before command", RuntimeWarning, stacklevel=2)
        return dict(eeg, setname=action), command

    restore = console_module._install_prompt_safe_logging()
    try:
        with (
            mock.patch.object(console_module.sys, "stderr", stream),
            mock.patch(patch_target, side_effect=fake_pop),
        ):
            dispatcher.dispatch_gui(action)
    finally:
        restore()
        workspace.close()

    output = stream.getvalue()
    assert output.index(f"In [1]: {command}") < output.index("RuntimeWarning: warning before command")
    assert session.ALLCOM == [command]


def test_console_history_edits_do_not_echo_as_gui_commands():
    session = EEGPrepSession()
    writes = []
    workspace = EEGPrepConsoleWorkspace(session, command_echo=writes.append, exports={})

    workspace.namespace["LASTCOM"] = "EEG = custom_command(EEG);"
    workspace.after_execute("LASTCOM = 'EEG = custom_command(EEG);'")

    assert session.ALLCOM == ["EEG = custom_command(EEG);"]
    assert writes == []


def test_preexisting_history_is_not_echoed_when_console_workspace_starts():
    session = EEGPrepSession()
    session.add_history("EEG = before_console;")
    writes = []

    EEGPrepConsoleWorkspace(session, command_echo=writes.append, exports={})

    assert writes == []


def test_console_eeg_assignment_stores_current_dataset_and_refreshes():
    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    refresh = mock.Mock()
    workspace = EEGPrepConsoleWorkspace(session, refresh=refresh, exports={})
    edited = dict(session.EEG, setname="edited")

    workspace.namespace["EEG"] = edited
    workspace.after_execute("EEG = edited")

    assert session.EEG["setname"] == "edited"
    assert session.ALLEEG[0]["setname"] == "edited"
    assert session.ALLCOM[-1] == "EEG = edited"
    refresh.assert_called_once()


def test_console_lastcom_assignment_updates_unified_history_once():
    session = EEGPrepSession()
    workspace = EEGPrepConsoleWorkspace(session, exports={})

    workspace.namespace["LASTCOM"] = "EEG = custom_command(EEG);"
    workspace.after_execute("LASTCOM = 'EEG = custom_command(EEG);'")

    assert session.LASTCOM == "EEG = custom_command(EEG);"
    assert session.ALLCOM == ["EEG = custom_command(EEG);"]


def test_bare_pop_call_updates_session_and_returns_compact_unpackable_result():
    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    refresh = mock.Mock()
    writes = []
    workspace = EEGPrepConsoleWorkspace(
        session,
        refresh=refresh,
        command_echo=writes.append,
        exports={"pop_reref": _fake_pop_reref},
    )

    result = workspace.namespace["pop_reref"](workspace.namespace["EEG"], [])
    workspace.after_execute("pop_reref(EEG, [])")

    eeg, command = result
    assert eeg is session.EEG
    assert command == "EEG = pop_reref(EEG, []);"
    assert session.EEG["setname"] == "reref"
    assert session.ALLCOM == ["EEG = pop_reref(EEG, []);"]
    assert "array" not in repr(result)
    assert writes == []
    refresh.assert_called_once()


def test_pop_call_without_history_command_records_raw_console_source():
    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    workspace = EEGPrepConsoleWorkspace(session, exports={"pop_interp": _fake_pop_without_command})

    result = workspace.namespace["pop_interp"](workspace.namespace["EEG"])
    workspace.after_execute("pop_interp(EEG)")

    eeg, command = result
    assert eeg is session.EEG
    assert command == ""
    assert session.EEG["setname"] == "no-history-command"
    assert session.ALLCOM == ["pop_interp(EEG)"]


def test_non_mutating_pop_plot_call_records_history_without_storing_dataset():
    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    original_eeg = session.EEG
    workspace = EEGPrepConsoleWorkspace(session, exports={"pop_topoplot": _fake_pop_topoplot})

    result = workspace.namespace["pop_topoplot"](workspace.namespace["EEG"])

    assert result == (["figure"], "pop_topoplot(EEG, typeplot=1, items=[0])")
    assert session.EEG is original_eeg
    assert session.ALLCOM == ["pop_topoplot(EEG, typeplot=1, items=[0])"]


def test_gui_pop_eegplot_accept_updates_console_namespace_without_duplicate_history():
    session = EEGPrepSession()
    eeg = _demo_eeg()
    eeg["data"] = np.arange(8, dtype=float).reshape(2, 4)
    eeg["pnts"] = 4
    eeg["xmax"] = 0.03
    session.store_current(eeg, new=True)
    refresh = mock.Mock()
    workspace = EEGPrepConsoleWorkspace(session, refresh=refresh, exports={})
    dispatcher = MenuActionDispatcher(session)
    captured = {}
    command = "pop_eegplot(EEG, 1, 0, 1)"

    def fake_pop_eegplot(eeg_in, *, command_callback=None, return_com=False, **_kwargs):
        captured["callback"] = command_callback
        assert eeg_in is session.EEG
        assert return_com is True
        return eeg_in, command

    with mock.patch("eegprep.functions.popfunc.pop_eegplot.pop_eegplot", side_effect=fake_pop_eegplot):
        dispatcher.dispatch("pop_eegplot:data")

    assert workspace.namespace["LASTCOM"] == command
    assert workspace.namespace["CURRENTSET"] == 1
    assert session.ALLCOM == [command]

    out = dict(session.EEG)
    out["data"] = np.asarray(out["data"])[:, :2]
    out["pnts"] = 2
    out["xmax"] = 0.01
    captured["callback"](out, command)

    assert workspace.namespace["CURRENTSET"] == 2
    assert workspace.namespace["EEG"]["pnts"] == 2
    assert len(workspace.namespace["ALLEEG"]) == 2
    assert session.ALLCOM == [command]


def test_console_pop_eegplot_accept_callback_refreshes_session_after_browser_accept():
    session = EEGPrepSession()
    eeg = _demo_eeg()
    eeg["data"] = np.arange(8, dtype=float).reshape(2, 4)
    eeg["pnts"] = 4
    eeg["xmax"] = 0.03
    session.store_current(eeg, new=True)
    refresh = mock.Mock()
    workspace = EEGPrepConsoleWorkspace(session, refresh=refresh, exports={"pop_eegplot": _fake_pop_eegplot})

    result = workspace.namespace["pop_eegplot"](workspace.namespace["EEG"])
    workspace.after_execute("pop_eegplot(EEG)")

    eeg_out, command = result
    assert eeg_out is session.EEG
    assert command == "pop_eegplot(EEG, 1, 0, 1)"
    assert "no EEG change" in repr(result)
    assert session.ALLCOM == [command]
    assert callable(_fake_pop_eegplot.command_callback)

    out = dict(session.EEG)
    out["data"] = np.asarray(out["data"])[:, :2]
    out["pnts"] = 2
    out["xmax"] = 0.01
    _fake_pop_eegplot.command_callback(out, command)

    assert session.CURRENTSET == [2]
    assert session.EEG["pnts"] == 2
    assert len(session.ALLEEG) == 2
    assert session.ALLCOM == [command]
    refresh.assert_called_once()


def test_console_pop_eegplot_positional_reject_argument_controls_accept_storage():
    session = EEGPrepSession()
    eeg = _demo_eeg()
    eeg["data"] = np.arange(8, dtype=float).reshape(2, 4)
    eeg["pnts"] = 4
    eeg["xmax"] = 0.03
    session.store_current(eeg, new=True)
    workspace = EEGPrepConsoleWorkspace(session, exports={"pop_eegplot": _fake_pop_eegplot})

    result = workspace.namespace["pop_eegplot"](workspace.namespace["EEG"], 1, 0, 0)
    workspace.after_execute("pop_eegplot(EEG, 1, 0, 0)")

    _eeg_out, command = result
    assert command == "pop_eegplot(EEG, 1, 0, 0)"
    out = dict(session.EEG)
    out["data"] = np.asarray(out["data"])[:, :2]
    out["pnts"] = 2
    out["xmax"] = 0.01
    _fake_pop_eegplot.command_callback(out, command)

    assert session.CURRENTSET == [1]
    assert session.EEG["pnts"] == 2
    assert len(session.ALLEEG) == 1


def test_console_rejection_browser_accept_callback_refreshes_session_after_accept():
    session = EEGPrepSession()
    eeg = _demo_eeg()
    eeg["data"] = np.arange(24, dtype=float).reshape(2, 4, 3)
    eeg["pnts"] = 4
    eeg["trials"] = 3
    eeg["xmax"] = 0.03
    session.store_current(eeg, new=True)
    refresh = mock.Mock()
    workspace = EEGPrepConsoleWorkspace(
        session,
        refresh=refresh,
        exports={"pop_jointprob": _fake_pop_jointprob_browser},
    )

    result = workspace.namespace["pop_jointprob"](workspace.namespace["EEG"])
    workspace.after_execute("pop_jointprob(EEG)")

    _eeg_out, command = result
    assert command == "EEG = pop_jointprob(EEG, 1, [1], 4, 4, 0, 1, 1);"
    assert session.ALLCOM == [command]
    assert callable(_fake_pop_jointprob_browser.command_callback)

    accepted = dict(session.EEG)
    accepted["data"] = np.asarray(session.EEG["data"])[:, :, :2]
    accepted["trials"] = 2
    _fake_pop_jointprob_browser.command_callback(accepted, command)

    assert session.CURRENTSET == [2]
    assert session.EEG["trials"] == 2
    assert len(session.ALLEEG) == 2
    assert session.ALLCOM == [command]
    assert refresh.call_count >= 2


def test_console_rejcont_browser_defers_store_until_accept():
    session = EEGPrepSession()
    eeg = _demo_eeg()
    eeg["data"] = np.arange(8, dtype=float).reshape(2, 4)
    eeg["pnts"] = 4
    eeg["xmax"] = 0.03
    session.store_current(eeg, new=True)
    refresh = mock.Mock()
    workspace = EEGPrepConsoleWorkspace(
        session,
        refresh=refresh,
        exports={"pop_rejcont": _fake_pop_rejcont_browser},
    )

    result = workspace.namespace["pop_rejcont"](workspace.namespace["EEG"])
    workspace.after_execute("pop_rejcont(EEG)")

    eeg_out, command = result
    assert eeg_out is session.EEG
    assert command == ""
    assert session.ALLCOM == []
    assert len(session.ALLEEG) == 1
    assert callable(_fake_pop_rejcont_browser.command_callback)

    accepted = dict(session.EEG)
    accepted["data"] = np.asarray(session.EEG["data"])[:, :2]
    accepted["pnts"] = 2
    accept_command = "EEG = pop_rejcont(EEG, 'eegplot', 'on');"
    _fake_pop_rejcont_browser.command_callback(accepted, accept_command)

    assert session.CURRENTSET == [2]
    assert session.EEG["pnts"] == 2
    assert len(session.ALLEEG) == 2
    assert session.ALLCOM == [accept_command]
    assert refresh.call_count >= 1


def test_bare_dataset_pop_call_updates_alleeg_eeg_currentset_and_history():
    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    refresh = mock.Mock()
    workspace = EEGPrepConsoleWorkspace(session, refresh=refresh, exports={"pop_copyset": _fake_pop_copyset})

    result = workspace.namespace["pop_copyset"](workspace.namespace["ALLEEG"], 1, 2)
    workspace.after_execute("pop_copyset(ALLEEG, 1, 2)")

    alleeg, eeg, currentset, command = result
    assert alleeg is session.ALLEEG
    assert eeg is session.EEG
    assert currentset == 2
    assert command == "[ALLEEG EEG CURRENTSET LASTCOM] = pop_copyset(ALLEEG, 1, 2);"
    assert session.CURRENTSET == [2]
    assert session.EEG["setname"] == "copied"
    assert session.ALLCOM == [command]
    assert "array" not in repr(result)
    refresh.assert_called_once()


def test_single_assignment_pop_call_without_history_resets_namespace_to_session_eeg():
    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    workspace = EEGPrepConsoleWorkspace(session, exports={"pop_interp": _fake_pop_without_command})

    result = workspace.namespace["pop_interp"](workspace.namespace["EEG"])
    workspace.namespace["EEG"] = result
    workspace.after_execute("EEG = pop_interp(EEG)")

    assert workspace.namespace["EEG"] is session.EEG
    assert session.EEG["setname"] == "no-history-command"
    assert session.ALLCOM == ["EEG = pop_interp(EEG)"]


def test_eegprep_proxy_pop_call_updates_session_like_direct_pop_call():
    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    workspace = EEGPrepConsoleWorkspace(session, exports={"pop_reref": _fake_pop_reref})

    result = workspace.namespace["eegprep"].pop_reref(workspace.namespace["EEG"], [])
    workspace.after_execute("eegprep.pop_reref(EEG, [])")

    eeg, command = result
    assert eeg is session.EEG
    assert command == "EEG = pop_reref(EEG, []);"
    assert session.EEG["setname"] == "reref"
    assert session.ALLCOM == ["EEG = pop_reref(EEG, []);"]


def test_keyword_eeg_pop_call_updates_current_dataset_in_place():
    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    workspace = EEGPrepConsoleWorkspace(session, exports={"pop_reref": _fake_pop_reref})

    result = workspace.namespace["pop_reref"](EEG=workspace.namespace["EEG"], ref=[])
    workspace.after_execute("pop_reref(EEG=EEG, ref=[])")

    eeg, command = result
    assert eeg is session.EEG
    assert command == "EEG = pop_reref(EEG, []);"
    assert session.EEG["setname"] == "reref"
    assert session.CURRENTSET == [1]
    assert len(session.ALLEEG) == 1
    assert session.ALLCOM == ["EEG = pop_reref(EEG, []);"]


def test_console_restores_eegprep_proxy_after_user_imports_eegprep():
    session = EEGPrepSession()
    workspace = EEGPrepConsoleWorkspace(session, exports={})

    workspace.namespace["eegprep"] = console_module.eegprep
    workspace.after_execute("import eegprep")

    assert isinstance(workspace.namespace["eegprep"], console_module.ConsoleEEGPrepModule)


def test_console_restores_pop_wrappers_after_from_import():
    session = EEGPrepSession()
    workspace = EEGPrepConsoleWorkspace(session, exports={"pop_reref": _fake_pop_reref})
    original_wrapper = workspace.namespace["pop_reref"]

    workspace.namespace["pop_reref"] = _fake_pop_reref
    workspace.after_execute("from eegprep import pop_reref")

    assert workspace.namespace["pop_reref"] is original_wrapper


def test_console_restores_aliased_pop_wrapper_after_from_import():
    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    workspace = EEGPrepConsoleWorkspace(session, exports={"pop_reref": _fake_pop_reref})

    workspace.namespace["reref"] = _fake_pop_reref
    workspace.after_execute("from eegprep import pop_reref as reref")
    result = workspace.namespace["reref"](workspace.namespace["EEG"], [])
    workspace.after_execute("reref(EEG, [])")

    eeg, command = result
    assert eeg is session.EEG
    assert command == "EEG = pop_reref(EEG, []);"
    assert session.EEG["setname"] == "reref"
    assert session.ALLCOM == ["EEG = pop_reref(EEG, []);"]


def test_console_restores_pop_alias_imported_after_pop_call_in_same_cell():
    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    workspace = EEGPrepConsoleWorkspace(
        session,
        exports={"pop_reref": _fake_pop_reref, "pop_resample": _fake_pop_without_command},
    )

    result = workspace.namespace["pop_reref"](workspace.namespace["EEG"], [])
    workspace.namespace["resample"] = _fake_pop_without_command
    workspace.after_execute("pop_reref(EEG, []); from eegprep import pop_resample as resample")

    assert result[0] is session.EEG
    assert isinstance(workspace.namespace["resample"], console_module.ConsolePopFunction)
    second = workspace.namespace["resample"](workspace.namespace["EEG"])
    workspace.after_execute("resample(EEG)")

    assert second[0] is session.EEG
    assert session.EEG["setname"] == "no-history-command"
    assert session.ALLCOM == ["EEG = pop_reref(EEG, []);", "resample(EEG)"]


def test_console_restores_aliased_eegprep_import_proxy():
    session = EEGPrepSession()
    workspace = EEGPrepConsoleWorkspace(session, exports={})

    workspace.namespace["ep"] = console_module.eegprep
    workspace.after_execute("import eegprep as ep")

    assert isinstance(workspace.namespace["ep"], console_module.ConsoleEEGPrepModule)


def test_extension_pop_namespace_updates_session_and_remains_console_local(
    tmp_path,
    monkeypatch,
):
    package = "console_extension_namespace"
    _write_extension_package(
        tmp_path,
        package,
        {
            "pop_functions.py": """
                import numpy as np

                def pop_console_ext(EEG, gain=2.0, *, return_com=False):
                    output = dict(EEG)
                    output["data"] = np.asarray(EEG["data"], dtype=float) * float(gain)
                    output["setname"] = "extension-console"
                    command = f"EEG = pop_console_ext(EEG, gain={float(gain)!r});"
                    return (output, command) if return_com else output
            """,
        },
        monkeypatch,
    )
    runtime = _extension_runtime(
        ExtensionSpec(
            name="console_extension",
            pop_functions=(
                ExtensionPopFunction("pop_console_ext", LazyImport(f"{package}.pop_functions", "pop_console_ext")),
            ),
        )
    )
    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    workspace = EEGPrepConsoleWorkspace(session, exports={}, extension_runtime=runtime)

    result = workspace.namespace["pop_console_ext"](workspace.namespace["EEG"], gain=3.0)
    workspace.after_execute("pop_console_ext(EEG, gain=3.0)")

    assert result.command == "EEG = pop_console_ext(EEG, gain=3.0);"
    assert workspace.namespace["EEG"] is session.EEG
    assert workspace.namespace["eegprep"].pop_console_ext is workspace.namespace["pop_console_ext"]
    np.testing.assert_allclose(session.EEG["data"], np.array([[3.0, 6.0], [9.0, 12.0]]))
    assert session.ALLCOM == ["EEG = pop_console_ext(EEG, gain=3.0);"]

    with pytest.raises(ImportError):
        exec("from eegprep import pop_console_ext", workspace.namespace)
    workspace.after_execute("from eegprep import pop_console_ext", success=False)

    assert workspace.namespace["pop_console_ext"].name == "pop_console_ext"
    assert session.ALLCOM == ["EEG = pop_console_ext(EEG, gain=3.0);"]


def test_assignment_style_pop_call_does_not_duplicate_history():
    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    workspace = EEGPrepConsoleWorkspace(session, exports={"pop_reref": _fake_pop_reref})

    result = workspace.namespace["pop_reref"](workspace.namespace["EEG"], [])
    workspace.namespace["EEG"], workspace.namespace["LASTCOM"] = result
    workspace.after_execute("EEG, LASTCOM = pop_reref(EEG, [])")

    assert session.EEG["setname"] == "reref"
    assert session.ALLCOM == ["EEG = pop_reref(EEG, []);"]


def test_failed_pop_call_does_not_mutate_session_or_history():
    def failing_pop(eeg, ref, *, return_com=False):
        raise ValueError("bad ref")

    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    workspace = EEGPrepConsoleWorkspace(session, exports={"pop_reref": failing_pop})

    with pytest.raises(ValueError, match="bad ref"):
        workspace.namespace["pop_reref"](workspace.namespace["EEG"], [])

    assert session.EEG["setname"] == "demo"
    assert session.ALLCOM == []


def test_multiple_selected_datasets_stay_selected_after_pop_call():
    def fake_pop_select(eeg, *, return_com=False):
        output = [dict(item, setname=f"{item['setname']}-selected") for item in eeg]
        command = "EEG = pop_select(EEG);"
        return (output, command) if return_com else output

    session = EEGPrepSession()
    session.store_current(_demo_eeg("one"), new=True)
    session.store_current(_demo_eeg("two"), new=True)
    session.retrieve([1, 2])
    workspace = EEGPrepConsoleWorkspace(session, exports={"pop_select": fake_pop_select})

    workspace.namespace["pop_select"](workspace.namespace["EEG"])

    assert session.CURRENTSET == [1, 2]
    assert [item["setname"] for item in session.ALLEEG] == ["one-selected", "two-selected"]


def test_real_pop_epoch_updates_console_session_and_history():
    from eegprep.functions.popfunc.pop_epoch import pop_epoch

    eeg = {
        "setname": "continuous",
        "data": np.arange(400, dtype=np.float32).reshape(2, 200),
        "nbchan": 2,
        "pnts": 200,
        "trials": 1,
        "srate": 100.0,
        "xmin": 0.0,
        "xmax": 1.99,
        "event": [{"type": "stim", "latency": 100, "duration": 0}],
        "urevent": [],
        "epoch": [],
    }
    session = EEGPrepSession()
    session.store_current(eeg, new=True)
    refresh = mock.Mock()
    workspace = EEGPrepConsoleWorkspace(session, refresh=refresh, exports={"pop_epoch": pop_epoch})

    result = workspace.namespace["pop_epoch"](workspace.namespace["EEG"], ["stim"], [-0.1, 0.1])
    workspace.after_execute("pop_epoch(EEG, ['stim'], [-0.1, 0.1])")

    output, command = result
    assert output is session.EEG
    assert session.EEG["trials"] == 1
    assert session.EEG["pnts"] == 20
    assert command == "EEG = pop_epoch( EEG, { 'stim' }, [-0.1 0.1]);"
    assert session.ALLCOM == [command]
    refresh.assert_called_once()


def test_safe_after_execute_sync_error_recovers_namespace_without_crashing():
    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    workspace = EEGPrepConsoleWorkspace(session, exports={})
    writes = []

    workspace.namespace["EEG"] = 3
    console_module._safe_after_execute(workspace, "EEG = 3", success=True, write=writes.append)

    assert session.EEG["setname"] == "demo"
    assert workspace.namespace["EEG"] is session.EEG
    assert session.ALLCOM == []
    assert "EEGPrep workspace sync failed" in "".join(writes)


class _FakeEvents:
    def __init__(self):
        self.callbacks = {}

    def register(self, name, callback):
        self.callbacks[name] = callback


class _FakeHistoryManager:
    def __init__(self):
        self.inputs = []

    def store_inputs(self, line_number, source, source_raw=None):
        self.inputs.append((line_number, source, source_raw))


class _FakeShell:
    def __init__(self):
        self.events = _FakeEvents()
        self.history_manager = _FakeHistoryManager()
        self.execution_count = 1
        self.enabled_gui = None
        self.called = False

    def enable_gui(self, gui):
        self.enabled_gui = gui

    def __call__(self):
        self.called = True
        callback = self.events.callbacks["post_run_cell"]
        callback(SimpleNamespace(info=SimpleNamespace(raw_cell="EEG"), success=True))


class _FakePrompts:
    def __init__(self, shell):
        self.shell = shell

    def in_prompt_tokens(self):
        return [("Token.Prompt", f"In [{self.shell.execution_count}]: ")]


def test_run_console_forwards_cli_options_to_gui_launcher():
    shell = _FakeShell()
    captured = {}

    def shell_factory(namespace, banner):
        captured["namespace"] = namespace
        captured["banner"] = banner
        return shell

    def gui_launcher(*args, **kwargs):
        captured["gui_args"] = args
        captured["gui_kwargs"] = kwargs
        return SimpleNamespace(refresh=mock.Mock())

    assert (
        console_module.run_console(
            ["--full", "--no-plugins", "--window-menu-bar"],
            shell_factory=shell_factory,
            gui_launcher=gui_launcher,
        )
        == 0
    )

    assert captured["gui_args"] == ("full",)
    assert captured["gui_kwargs"]["include_plugins"] is False
    assert captured["gui_kwargs"]["native_menu_bar"] is False
    assert captured["gui_kwargs"]["native_file_dialogs"] is False
    assert "EEGPrep interactive console" in captured["banner"]
    assert shell.enabled_gui == "qt"
    assert shell.called is True
    assert "EEG" in captured["namespace"]


def test_run_console_native_file_dialogs_are_explicit_opt_in():
    shell = _FakeShell()
    captured = {}

    def shell_factory(namespace, banner):
        return shell

    def gui_launcher(*args, **kwargs):
        captured["gui_kwargs"] = kwargs
        return SimpleNamespace(refresh=mock.Mock())

    assert (
        console_module.run_console(
            ["--native-file-dialogs"],
            shell_factory=shell_factory,
            gui_launcher=gui_launcher,
        )
        == 0
    )

    assert captured["gui_kwargs"]["native_file_dialogs"] is True


def test_ipython_factory_error_is_user_facing_when_dependency_missing():
    with (
        mock.patch.object(console_module.importlib, "import_module", side_effect=ImportError("missing")),
        pytest.raises(RuntimeError, match="IPython is required for eegprep-console"),
    ):
        console_module._ipython_shell_factory({}, "")


def test_ipython_adapter_records_gui_command_as_input_and_advances_prompt():
    shell = _FakeShell()
    workspace = EEGPrepConsoleWorkspace(EEGPrepSession(), exports={})
    adapter = console_module._IPythonShellAdapter(shell, workspace)

    with mock.patch.object(console_module, "_terminal_write") as terminal_write:
        adapter.echo_gui_command("EEG = pop_fileio('demo.set');")

    assert shell.execution_count == 2
    assert shell.history_manager.inputs == [
        (1, "EEG = pop_fileio(filename='demo.set')", "EEG = pop_fileio(filename='demo.set')")
    ]
    terminal_write.assert_called_once_with(
        "In [1]: EEG = pop_fileio(filename='demo.set')\n",
        stream=console_module.sys.stderr,
        sync=True,
    )


def test_ipython_adapter_echoes_gui_commands_as_valid_python():
    shell = _FakeShell()
    workspace = EEGPrepConsoleWorkspace(EEGPrepSession(), exports={})
    adapter = console_module._IPythonShellAdapter(shell, workspace)

    with mock.patch.object(console_module, "_terminal_write") as terminal_write:
        adapter.echo_gui_command("EEG = pop_interp(EEG, [1], 'spherical', [5 10]);")

    echoed = shell.history_manager.inputs[0][1]
    assert echoed == "EEG = pop_interp(EEG, bad_elec=[0], method='spherical', t_range=[5, 10])"
    ast.parse(echoed)
    terminal_write.assert_called_once_with(
        "In [1]: EEG = pop_interp(EEG, bad_elec=[0], method='spherical', t_range=[5, 10])\n",
        stream=console_module.sys.stderr,
        sync=True,
    )


def test_ipython_adapter_echoes_pop_reref_with_parameter_names():
    shell = _FakeShell()
    workspace = EEGPrepConsoleWorkspace(EEGPrepSession(), exports={})
    adapter = console_module._IPythonShellAdapter(shell, workspace)

    with mock.patch.object(console_module, "_terminal_write") as terminal_write:
        adapter.echo_gui_command("EEG = pop_reref( EEG, [], 'keepref', 'on');")

    echoed = shell.history_manager.inputs[0][1]
    assert echoed == "EEG = pop_reref(EEG, ref=[], keepref='on')"
    ast.parse(echoed)
    terminal_write.assert_called_once_with(
        "In [1]: EEG = pop_reref(EEG, ref=[], keepref='on')\n",
        stream=console_module.sys.stderr,
        sync=True,
    )


def test_console_python_command_converts_common_eeglab_history_syntax():
    commands = [
        "[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'retrieve', 1);",
        "CURRENTSTUDY = 0;[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'retrieve', 2);",
        "EEG = pop_select(EEG, 'channel', [1 2], 'chantype', {'EEG' 'EOG'});",
        "LASTCOM = pop_export(EEG, '/tmp/demo''s data.tsv');",
        "EEG = pop_resample( EEG, 64);",
        "EEG = pop_comments(EEG, '', 'sample notes');",
        "comments = pop_comments(comments, '', {'second' 'third'}, 1);",
        "EEG = pop_editset(EEG, 'setname', 'edited', 'subject', 'S01');",
        "EEG = pop_reref( EEG, [1], 'exclude', [4]);",
        "EEG = pop_reref( EEG, [], 'huber', 25);",
        "STUDY, ALLEEG = pop_study(STUDY, ALLEEG, name='demo');",
        "(ALLEEG, EEG, CURRENTSET) = pop_newset(ALLEEG, EEG, CURRENTSET, retrieve=3)",
    ]

    converted = [console_module._console_python_command(command) for command in commands]

    assert converted == [
        "ALLEEG, EEG, CURRENTSET = pop_newset(ALLEEG, EEG, CURRENTSET, retrieve=1)",
        "CURRENTSTUDY = 0; ALLEEG, EEG, CURRENTSET = pop_newset(ALLEEG, EEG, CURRENTSET, retrieve=2)",
        "EEG = pop_select(EEG, channel=[1, 2], chantype=['EEG', 'EOG'])",
        'LASTCOM = pop_export(EEG, filename="/tmp/demo\'s data.tsv")',
        "EEG = pop_resample(EEG, freq=64)",
        "EEG = pop_comments(EEG, plottitle='', newcomments='sample notes')",
        "comments = pop_comments(EEG=comments, plottitle='', newcomments=['second', 'third'], concat=1)",
        "EEG = pop_editset(EEG, setname='edited', subject='S01')",
        "EEG = pop_reref(EEG, ref=[0], exclude=[3])",
        "EEG = pop_reref(EEG, ref=[], huber=25)",
        "STUDY, ALLEEG = pop_study(STUDY, ALLEEG, name='demo')",
        "ALLEEG, EEG, CURRENTSET = pop_newset(ALLEEG, EEG, CURRENTSET, retrieve=3)",
    ]
    for command in converted:
        ast.parse(command)


def test_ipython_adapter_keeps_prompt_message_dynamic():
    shell = _FakeShell()
    shell.prompts = _FakePrompts(shell)

    def extra_prompt_options():
        return {"message": "In [1]: "}

    shell._extra_prompt_options = extra_prompt_options

    console_module._make_shell_prompt_dynamic(shell)
    options = shell._extra_prompt_options()

    assert callable(options["message"])
    assert shell._eegprep_dynamic_prompt is True


def test_ipython_adapter_installs_dynamic_prompt_before_shell_starts():
    shell = _FakeShell()
    shell.prompts = _FakePrompts(shell)
    shell._extra_prompt_options = lambda: {"message": "In [1]: "}
    workspace = EEGPrepConsoleWorkspace(EEGPrepSession(), exports={})
    adapter = console_module._IPythonShellAdapter(shell, workspace)

    adapter()

    assert shell._eegprep_dynamic_prompt is True


def test_ipython_adapter_installs_prompt_safe_logging_during_shell_run():
    shell = _FakeShell()
    workspace = EEGPrepConsoleWorkspace(EEGPrepSession(), exports={})
    adapter = console_module._IPythonShellAdapter(shell, workspace)

    with (
        mock.patch.object(console_module, "_install_prompt_safe_logging") as install_logging,
        mock.patch.object(console_module, "_make_shell_prompt_dynamic"),
    ):
        restore_logging = mock.Mock()
        install_logging.return_value = restore_logging
        adapter()

    install_logging.assert_called_once()
    restore_logging.assert_called_once()


def test_prompt_safe_logging_routes_python_warnings_through_terminal_write():
    restore = console_module._install_prompt_safe_logging()
    try:
        with mock.patch.object(console_module, "_terminal_write") as terminal_write:
            warnings.warn("demo warning", RuntimeWarning, stacklevel=1)
    finally:
        restore()

    terminal_write.assert_called_once()
    message = terminal_write.call_args.args[0]
    assert "RuntimeWarning: demo warning" in message
    assert terminal_write.call_args.kwargs["stream"] is console_module.sys.stderr


def test_format_ipython_input_trims_extra_newlines():
    assert console_module._format_ipython_input("EEG = demo;\n", 3) == "In [3]: EEG = demo;\n"


def test_prompt_safe_logging_stream_uses_terminal_write():
    stream = io.StringIO()
    safe_stream = console_module._PromptSafeStream(stream)

    with mock.patch.object(console_module, "_terminal_write") as terminal_write:
        assert safe_stream.write("WARNING (demo) message\n") == len("WARNING (demo) message\n")

    terminal_write.assert_called_once_with("WARNING (demo) message\n", stream=stream)


def test_prompt_safe_logging_install_restores_stream_handlers():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        restore = console_module._install_prompt_safe_logging()

        assert isinstance(handler.stream, console_module._PromptSafeStream)

        restore()

        assert handler.stream is stream
    finally:
        root_logger.removeHandler(handler)


def test_terminal_write_prints_above_active_prompt():
    stream = io.StringIO()
    calls = []

    def fake_run_in_terminal(callback):
        calls.append(callback)
        callback()

    fake_module = SimpleNamespace(run_in_terminal=fake_run_in_terminal)

    with (
        mock.patch.object(console_module.importlib, "import_module", return_value=fake_module) as import_module,
        mock.patch.object(console_module.sys, "stdout", stream),
    ):
        console_module._terminal_write("In [1]: EEG = pop_fileio('demo.set');\n")

    import_module.assert_called_once_with("prompt_toolkit.application.run_in_terminal")
    assert len(calls) == 1
    assert stream.getvalue() == "In [1]: EEG = pop_fileio('demo.set');\n"


def test_terminal_write_fallback_starts_on_new_line():
    stream = io.StringIO()

    with (
        mock.patch.object(console_module.importlib, "import_module", side_effect=ImportError("missing")),
        mock.patch.object(console_module.sys, "stdout", stream),
    ):
        console_module._terminal_write("In [1]: EEG = pop_fileio('demo.set');\n")

    assert stream.getvalue() == "\nIn [1]: EEG = pop_fileio('demo.set');\n"


def test_terminal_write_sync_path_writes_immediately_without_prompt_toolkit():
    stream = io.StringIO()

    with mock.patch.object(console_module.importlib, "import_module") as import_module:
        console_module._terminal_write("In [2]: EEG = pop_interp(EEG, [1]);\n", stream=stream, sync=True)

    import_module.assert_not_called()
    assert stream.getvalue() == "\nIn [2]: EEG = pop_interp(EEG, [1]);\n"
