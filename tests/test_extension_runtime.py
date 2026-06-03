from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from eegprep.extension_runtime import ExtensionRuntime
from eegprep.extensions import (
    EXTENSION_ENTRY_POINT_GROUP,
    ExtensionAction,
    ExtensionMenu,
    ExtensionPopFunction,
    ExtensionRecord,
    ExtensionRegistry,
    ExtensionResource,
    ExtensionSourceType,
    ExtensionSpec,
    ExtensionStatus,
    LazyImport,
    validate_extension_spec,
)
from eegprep.functions.adminfunc.console import EEGPrepConsoleWorkspace
from eegprep.functions.guifunc.eeglab_menu import eeglab_menus, menu_actions
from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher, action_kind
from eegprep.functions.guifunc.menu_spec import menu_enabled
from eegprep.functions.guifunc.pophelp import pophelp_text
from eegprep.functions.guifunc.session import EEGPrepSession


class FakeDistribution:
    def __init__(self, name: str) -> None:
        self.metadata = {"Name": name}


class FakeEntryPoint:
    def __init__(self, name: str, value: str, *, group: str = EXTENSION_ENTRY_POINT_GROUP) -> None:
        self.name = name
        self.value = value
        self.group = group
        self.dist = FakeDistribution(name)

    def load(self):
        module_name, _, attr_name = self.value.partition(":")
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)


def test_extension_menu_enabled_state_matrix() -> None:
    runtime = _runtime(
        ExtensionSpec(
            name="state_extension",
            menus=(
                ExtensionMenu(
                    path=("tools",),
                    action="pop_continuous_only",
                    label="Continuous extension",
                    userdata="startup:off;epoch:off",
                ),
                ExtensionMenu(
                    path=("tools",),
                    action="pop_epoched_only",
                    label="Epoched extension",
                    userdata="startup:off;continuous:off",
                ),
                ExtensionMenu(
                    path=("tools",),
                    action="pop_requires_ica",
                    label="ICA extension",
                    userdata="startup:off;ica:on",
                ),
                ExtensionMenu(
                    path=("tools",),
                    action="pop_study_or_multi",
                    label="Study extension",
                    userdata="startup:off;study:on",
                ),
            ),
        )
    )

    tools = _child(eeglab_menus(all_menus=True, extension_runtime=runtime), "Tools")
    items = {item.label: item for item in tools.children}

    assert not menu_enabled(items["Continuous extension"], {"startup"})
    assert menu_enabled(items["Continuous extension"], {"continuous_dataset"})
    assert not menu_enabled(items["Continuous extension"], {"epoched_dataset"})
    assert menu_enabled(items["Epoched extension"], {"epoched_dataset"})
    assert not menu_enabled(items["Epoched extension"], {"continuous_dataset"})
    assert not menu_enabled(items["ICA extension"], {"continuous_dataset", "ica_absent"})
    assert menu_enabled(items["ICA extension"], {"continuous_dataset"})
    assert menu_enabled(items["Study extension"], {"study"})
    assert menu_enabled(items["Study extension"], {"multiple_datasets"})


def test_include_plugins_false_hides_extension_menu_contributions() -> None:
    runtime = _runtime(
        ExtensionSpec(
            name="hidden_extension",
            menus=(ExtensionMenu(path=("tools",), action="pop_hidden_extension", label="Hidden extension"),),
        )
    )

    actions = menu_actions(eeglab_menus(all_menus=True, include_plugins=False, extension_runtime=runtime))

    assert "pop_hidden_extension" not in actions


def test_include_plugins_false_hides_bundled_plugins_without_hiding_core_menus() -> None:
    menus = eeglab_menus(all_menus=False, include_plugins=False)
    tools = _child(menus, "Tools")
    file_menu = _child(menus, "File")
    import_menu = _child(file_menu.children, "Import data")
    import_functions = _child(import_menu.children, "Using EEGPrep functions and plugins")

    assert [menu.label for menu in menus] == ["File", "Edit", "Tools", "Plot", "Study", "Datasets", "Help"]
    assert "Change sampling rate" in [item.label for item in tools.children]
    assert "Reject data using Clean Rawdata and ASR" not in [item.label for item in tools.children]
    assert "From BIDS folder structure" not in [item.label for item in import_functions.children]


def test_bundled_plugin_menus_land_at_expected_anchors() -> None:
    menus = eeglab_menus(all_menus=True)
    default_menus = eeglab_menus(all_menus=False)
    tools = _child(menus, "Tools")
    default_tools = _child(default_menus, "Tools")
    file_menu = _child(menus, "File")
    plot_menu = _child(menus, "Plot")

    tools_labels = [item.label for item in tools.children]
    default_tools_labels = [item.label for item in default_tools.children]
    filter_labels = [item.label for item in _child(tools.children, "Filter the data").children]
    import_menu = _child(file_menu.children, "Import data")
    import_functions = _child(import_menu.children, "Using EEGPrep functions and plugins")
    export_menu = _child(file_menu.children, "Export")
    file_labels = [item.label for item in file_menu.children]
    plot_labels = [item.label for item in plot_menu.children]

    assert (
        tools_labels[tools_labels.index("Inspect/reject data by eye") + 1] == "Reject data using Clean Rawdata and ASR"
    )
    assert (
        default_tools_labels[default_tools_labels.index("Inspect/label components by map") + 1]
        == "Classify components using ICLabel"
    )
    assert tools_labels[tools_labels.index("Remove epoch baseline") + 1] == "Source localization using DIPFIT"
    assert filter_labels == [
        "Basic FIR filter (new, default)",
        "Windowed sinc FIR filter",
        "Parks-McClellan (equiripple) FIR filter",
        "Moving average FIR filter",
        "Basic FIR filter (legacy)",
    ]
    assert "From BIDS folder structure" in [item.label for item in import_functions.children]
    assert "To BIDS folder structure" in [item.label for item in export_menu.children]
    assert file_labels[file_labels.index("Export") + 1] == "BIDS tools"
    assert plot_labels[-2:] == ["View extended channel properties", "View extended component properties"]


def test_missing_extension_menu_path_is_logged_and_skipped(caplog: pytest.LogCaptureFixture) -> None:
    runtime = _runtime(
        ExtensionSpec(
            name="missing_path_extension",
            menus=(
                ExtensionMenu(
                    path=("missing menu",),
                    action="pop_missing_path",
                    label="Missing path extension",
                ),
            ),
        )
    )
    caplog.set_level("WARNING", logger="eegprep.extension_runtime")

    actions = menu_actions(eeglab_menus(all_menus=True, extension_runtime=runtime))

    assert "pop_missing_path" not in actions
    assert "menu path missing menu was not found" in caplog.text


def test_duplicate_extension_menu_names_mark_later_record_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_package(
        tmp_path,
        "menu_duplicate_a",
        {
            "register.py": """
                from eegprep.extensions import ExtensionMenu, ExtensionSpec

                def register():
                    return ExtensionSpec(
                        name="menu_duplicate_a",
                        menus=(ExtensionMenu(path=("tools",), action="pop_a", label="Shared menu"),),
                    )
            """,
        },
        monkeypatch,
    )
    _write_package(
        tmp_path,
        "menu_duplicate_b",
        {
            "register.py": """
                from eegprep.extensions import ExtensionMenu, ExtensionSpec

                def register():
                    return ExtensionSpec(
                        name="menu_duplicate_b",
                        menus=(ExtensionMenu(path=("tools",), action="pop_b", label="Shared menu"),),
                    )
            """,
        },
        monkeypatch,
    )
    registry = ExtensionRegistry(
        include_bundled=False,
        entry_points_provider=_provider(
            FakeEntryPoint("menu-a", "menu_duplicate_a.register:register"),
            FakeEntryPoint("menu-b", "menu_duplicate_b.register:register"),
        ),
    )

    records = registry.discover()

    assert [record.status for record in records] == [ExtensionStatus.INSTALLED, ExtensionStatus.INVALID_SPEC]
    assert "Duplicate menu 'Shared menu'" in records[1].errors[0]


def test_malformed_extension_help_resource_invalidates_spec() -> None:
    spec = ExtensionSpec(
        name="malformed_help_extension",
        help_resources=(ExtensionResource("malformed_help_extension", "help/pop_bad.txt"),),
    )

    result = validate_extension_spec(spec)

    assert result.invalid_spec
    assert "must be a Markdown file" in result.invalid_spec[0]


def test_extension_action_result_shapes_update_session_history_and_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = "extension_action_shapes"
    _write_package(
        tmp_path,
        package,
        {
            "actions.py": """
                def pop_eeg_only(EEG, *, return_com=False):
                    return dict(EEG, setname="eeg-only")

                def pop_eeg_command(EEG, *, return_com=False):
                    return dict(EEG, setname="eeg-command"), "EEG = pop_eeg_command(EEG);"

                def command_only(*, session=None, return_com=False):
                    return "LASTCOM = extension_command();"

                def no_mutation(*, session=None, return_com=False):
                    return None
            """,
        },
        monkeypatch,
    )
    runtime = _runtime(
        ExtensionSpec(
            name="action_shapes_extension",
            actions=(
                ExtensionAction("pop_eeg_only", LazyImport(f"{package}.actions", "pop_eeg_only")),
                ExtensionAction("pop_eeg_command", LazyImport(f"{package}.actions", "pop_eeg_command")),
                ExtensionAction("command_only", LazyImport(f"{package}.actions", "command_only")),
                ExtensionAction("no_mutation", LazyImport(f"{package}.actions", "no_mutation")),
            ),
        )
    )
    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    refresh = mock.Mock()
    dispatcher = MenuActionDispatcher(session, refresh=refresh, extension_runtime=runtime)

    dispatcher.dispatch("pop_eeg_only")
    dispatcher.dispatch("pop_eeg_command")
    dispatcher.dispatch("command_only")
    history_after_command = list(session.ALLCOM)
    dispatcher.dispatch("no_mutation")

    assert session.EEG["setname"] == "eeg-command"
    assert session.ALLCOM == [
        "EEG = pop_eeg_command(EEG);",
        "LASTCOM = extension_command();",
    ]
    assert session.ALLCOM == history_after_command
    assert refresh.call_count == 3


def test_registered_extension_pop_function_dispatches_as_gui_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = "extension_gui_pop"
    _write_package(
        tmp_path,
        package,
        {
            "pop_functions.py": """
                def pop_gui_ext(EEG, *, return_com=False):
                    output = dict(EEG, setname="gui-extension")
                    command = "EEG = pop_gui_ext(EEG);"
                    return (output, command) if return_com else output
            """,
        },
        monkeypatch,
    )
    runtime = _runtime(
        ExtensionSpec(
            name="gui_pop_extension",
            pop_functions=(
                ExtensionPopFunction(
                    "pop_gui_ext",
                    LazyImport(f"{package}.pop_functions", "pop_gui_ext"),
                ),
            ),
        )
    )
    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    echoed = []
    session.add_command_echo_listener(echoed.append)
    dispatcher = MenuActionDispatcher(session, extension_runtime=runtime)

    dispatcher.dispatch("pop_gui_ext")

    assert session.EEG["setname"] == "gui-extension"
    assert session.ALLCOM == ["EEG = pop_gui_ext(EEG);"]
    assert echoed == ["EEG = pop_gui_ext(EEG);"]


def test_extension_pop_function_uses_single_dataset_selection_rule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = "extension_single_selection"
    _write_package(
        tmp_path,
        package,
        {
            "pop_functions.py": """
                calls = []

                def pop_single_ext(EEG, *, return_com=False):
                    calls.append(EEG)
                    output = dict(EEG, setname="single-extension")
                    command = "EEG = pop_single_ext(EEG);"
                    return (output, command) if return_com else output
            """,
        },
        monkeypatch,
    )
    runtime = _runtime(
        ExtensionSpec(
            name="single_selection_extension",
            pop_functions=(
                ExtensionPopFunction(
                    "pop_single_ext",
                    LazyImport(f"{package}.pop_functions", "pop_single_ext"),
                ),
            ),
        )
    )
    session = EEGPrepSession()
    session.store_current(_demo_eeg("one"), new=True)
    session.store_current(_demo_eeg("two"), new=True)
    session.retrieve([1, 2])
    dispatcher = MenuActionDispatcher(session, extension_runtime=runtime)
    module = importlib.import_module(f"{package}.pop_functions")

    with mock.patch.object(dispatcher, "_warn") as warn:
        dispatcher.dispatch("pop_single_ext", parent="window")

    warn.assert_called_once_with("window", "This action is not available for multiple selected datasets")
    assert module.calls == []
    assert session.CURRENTSET == [1, 2]
    assert session.ALLCOM == []


def test_extension_action_error_after_menu_click_is_reported_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = "extension_action_error"
    _write_package(
        tmp_path,
        package,
        {
            "actions.py": """
                def pop_raises(EEG, *, return_com=False):
                    raise RuntimeError("extension boom")
            """,
            "pop_functions.py": """
                def pop_after_gui_failure(EEG, *, return_com=False):
                    output = dict(EEG, setname="console-still-usable")
                    command = "EEG = pop_after_gui_failure(EEG);"
                    return (output, command) if return_com else output
            """,
        },
        monkeypatch,
    )
    runtime = _runtime(
        ExtensionSpec(
            name="raising_extension",
            actions=(ExtensionAction("pop_raises", LazyImport(f"{package}.actions", "pop_raises")),),
            pop_functions=(
                ExtensionPopFunction(
                    "pop_after_gui_failure",
                    LazyImport(f"{package}.pop_functions", "pop_after_gui_failure"),
                ),
            ),
        )
    )
    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    dispatcher = MenuActionDispatcher(session, extension_runtime=runtime)

    with mock.patch.object(dispatcher, "_warn") as warn:
        dispatcher.dispatch_gui("pop_raises", parent="window")

    warn.assert_called_once_with("window", "extension boom")
    assert session.ALLCOM == []
    assert session.EEG["setname"] == "demo"

    workspace = EEGPrepConsoleWorkspace(session, exports={}, extension_runtime=runtime)
    result = workspace.namespace["pop_after_gui_failure"](workspace.namespace["EEG"])
    workspace.after_execute("pop_after_gui_failure(EEG)")

    assert result.command == "EEG = pop_after_gui_failure(EEG);"
    assert session.EEG["setname"] == "console-still-usable"
    assert session.ALLCOM == ["EEG = pop_after_gui_failure(EEG);"]


def test_extension_pop_function_console_bare_and_assigned_calls_store_history_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = "extension_console_pop"
    _write_package(
        tmp_path,
        package,
        {
            "pop_functions.py": """
                def pop_console_ext(EEG, *, return_com=False):
                    output = dict(EEG, setname="console-extension")
                    command = "EEG = pop_console_ext(EEG);"
                    return (output, command) if return_com else output
            """,
        },
        monkeypatch,
    )
    runtime = _runtime(
        ExtensionSpec(
            name="console_extension",
            pop_functions=(
                ExtensionPopFunction(
                    "pop_console_ext",
                    LazyImport(f"{package}.pop_functions", "pop_console_ext"),
                ),
            ),
        )
    )

    bare_session = EEGPrepSession()
    bare_session.store_current(_demo_eeg(), new=True)
    bare_workspace = EEGPrepConsoleWorkspace(bare_session, exports={}, extension_runtime=runtime)
    bare_result = bare_workspace.namespace["pop_console_ext"](bare_workspace.namespace["EEG"])
    bare_workspace.after_execute("pop_console_ext(EEG)")

    assert bare_result.command == "EEG = pop_console_ext(EEG);"
    assert bare_session.EEG["setname"] == "console-extension"
    assert bare_session.ALLCOM == ["EEG = pop_console_ext(EEG);"]

    assigned_session = EEGPrepSession()
    assigned_session.store_current(_demo_eeg(), new=True)
    assigned_workspace = EEGPrepConsoleWorkspace(assigned_session, exports={}, extension_runtime=runtime)
    exec("EEG, com = pop_console_ext(EEG)", assigned_workspace.namespace)
    assigned_workspace.after_execute("EEG, com = pop_console_ext(EEG)")

    assert assigned_workspace.namespace["EEG"] is assigned_session.EEG
    assert assigned_workspace.namespace["com"] == "EEG = pop_console_ext(EEG);"
    assert assigned_session.EEG["setname"] == "console-extension"
    assert assigned_session.ALLCOM == ["EEG = pop_console_ext(EEG);"]


def test_extension_pop_function_console_failure_does_not_mutate_session_or_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = "extension_console_failure"
    _write_package(
        tmp_path,
        package,
        {
            "pop_functions.py": """
                def pop_console_raises(EEG, *, return_com=False):
                    raise RuntimeError("console extension boom")
            """,
        },
        monkeypatch,
    )
    runtime = _runtime(
        ExtensionSpec(
            name="console_failure_extension",
            pop_functions=(
                ExtensionPopFunction(
                    "pop_console_raises",
                    LazyImport(f"{package}.pop_functions", "pop_console_raises"),
                ),
            ),
        )
    )
    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    workspace = EEGPrepConsoleWorkspace(session, exports={}, extension_runtime=runtime)

    with pytest.raises(RuntimeError, match="console extension boom"):
        workspace.namespace["pop_console_raises"](workspace.namespace["EEG"])
    workspace.after_execute("pop_console_raises(EEG)", success=False)

    assert session.EEG["setname"] == "demo"
    assert session.ALLCOM == []
    assert session.LASTCOM == ""
    assert workspace.namespace["EEG"] is session.EEG


def test_extension_help_resource_lookup_uses_packaged_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = "extension_help_resource"
    _write_package(
        tmp_path,
        package,
        {
            "help/pop_help_ext.md": "POP_HELP_EXT - extension help text",
        },
        monkeypatch,
    )
    runtime = _runtime(
        ExtensionSpec(
            name="help_extension",
            help_resources=(ExtensionResource(package, "help/pop_help_ext.md"),),
        )
    )

    text, source_path = pophelp_text("pop_help_ext", extension_runtime=runtime)

    assert "POP_HELP_EXT - extension help text" in text
    assert source_path == f"{package}:help/pop_help_ext.md"


def test_extension_action_kind_is_runtime_backed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = "extension_action_kind"
    _write_package(
        tmp_path,
        package,
        {
            "actions.py": """
                def run():
                    return None
            """,
        },
        monkeypatch,
    )
    runtime = _runtime(
        ExtensionSpec(
            name="kind_extension",
            actions=(ExtensionAction("kind_action", LazyImport(f"{package}.actions", "run")),),
        )
    )

    assert action_kind("kind_action") == "unknown"
    assert action_kind("kind_action", extension_runtime=runtime) == "implemented"


def _runtime(spec: ExtensionSpec) -> ExtensionRuntime:
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


def _child(items, label):
    for item in items:
        if item.label == label:
            return item
    raise AssertionError(f"missing menu item {label!r}")


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
        "chanlocs": [{"labels": "Cz", "theta": 0.0}, {"labels": "Pz", "theta": 180.0}],
        "event": [],
        "urevent": [],
        "epoch": [],
    }


def _provider(*entry_points: FakeEntryPoint):
    def select(*, group: str) -> tuple[FakeEntryPoint, ...]:
        return tuple(entry_point for entry_point in entry_points if entry_point.group == group)

    return select


def _write_package(
    tmp_path: Path,
    package: str,
    files: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
