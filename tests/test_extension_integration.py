"""End-to-end extension ecosystem integration tests."""

from __future__ import annotations

import importlib
from importlib import metadata
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eegprep.extension_catalog import load_extension_catalog
from eegprep.extension_runtime import ExtensionRuntime
from eegprep.extension_testing import ExtensionTestHarness
from eegprep.extensions import EXTENSION_ENTRY_POINT_GROUP, ExtensionRegistry, ExtensionStatus
from eegprep.functions.adminfunc.console import EEGPrepConsoleWorkspace
from eegprep.functions.adminfunc.plugin_menu import plugin_menu
from eegprep.functions.guifunc.eeglab_menu import eeglab_menus
from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher
from eegprep.functions.guifunc.pophelp import pophelp_text
from eegprep.functions.guifunc.session import EEGPrepSession


class _FakeDistribution:
    def __init__(self, name: str) -> None:
        self.metadata = {"Name": name}


class _FakeEntryPoint:
    def __init__(
        self,
        name: str,
        value: str,
        *,
        package_name: str,
        group: str = EXTENSION_ENTRY_POINT_GROUP,
    ) -> None:
        self.name = name
        self.value = value
        self.group = group
        self.dist = _FakeDistribution(package_name)

    def load(self) -> Any:
        module_name, _, attr_name = self.value.partition(":")
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)


def test_installed_extension_works_across_registry_manager_gui_help_and_console(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package = "lab_gain_extension"
    distribution_name = "eegprep-ext-lab-gain"
    _write_package(
        tmp_path,
        package,
        {
            "registration.py": """
                from eegprep import (
                    EXTENSION_API_VERSION,
                    ExtensionAction,
                    ExtensionMenu,
                    ExtensionPopFunction,
                    ExtensionResource,
                    ExtensionSpec,
                    LazyImport,
                )

                def register():
                    return ExtensionSpec(
                        name="lab_gain_extension",
                        display_name="Lab Gain Extension",
                        version="0.1.0",
                        api_version=EXTENSION_API_VERSION,
                        package_name="lab_gain_extension",
                        description="Example lab extension that scales EEG data.",
                        maintainer="Example Lab",
                        capabilities=("signal-transform", "history", "menu"),
                        menus=(
                            ExtensionMenu(
                                path=("Tools", "Lab extensions"),
                                action="pop_lab_gain",
                                label="Apply lab gain",
                            ),
                        ),
                        actions=(
                            ExtensionAction(
                                name="pop_lab_gain",
                                target=LazyImport("lab_gain_extension.pop_functions", "pop_lab_gain"),
                            ),
                        ),
                        pop_functions=(
                            ExtensionPopFunction(
                                name="pop_lab_gain",
                                target=LazyImport("lab_gain_extension.pop_functions", "pop_lab_gain"),
                            ),
                        ),
                        help_resources=(
                            ExtensionResource("lab_gain_extension", "resources/help/pop_lab_gain.md"),
                        ),
                        package_data_resources=(
                            ExtensionResource("lab_gain_extension", "resources/sample_data/demo.json"),
                        ),
                        eegprep_requires=">=0.2",
                    )
            """,
            "pop_functions.py": """
                import numpy as np

                def pop_lab_gain(EEG, gain=2.0, *, return_com=False):
                    output = dict(EEG)
                    output["data"] = np.asarray(EEG["data"], dtype=float) * float(gain)
                    output["setname"] = "lab-gain"
                    command = f"EEG = pop_lab_gain(EEG, gain={float(gain)!r});"
                    return (output, command) if return_com else output
            """,
            "resources/help/pop_lab_gain.md": "POP_LAB_GAIN - Scale EEG data for integration testing.",
            "resources/sample_data/demo.json": '{"gain": 2.0}',
        },
        monkeypatch,
    )
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        """
        {
          "schema_version": 1,
          "extensions": [
            {
              "name": "lab_gain_extension",
              "display_name": "Lab Gain Extension",
              "version": "0.1.0",
              "package_name": "eegprep-ext-lab-gain",
              "description": "Curated lab gain extension metadata.",
              "maintainer": "Example Lab",
              "docs_url": "https://example.org/eegprep-ext-lab-gain",
              "source": {"type": "pypi", "url": "https://pypi.org/project/eegprep-ext-lab-gain/"},
              "capabilities": ["signal-transform", "history", "menu"],
              "eegprep_requires": ">=0.2"
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    entry_point = _FakeEntryPoint(
        "lab_gain",
        f"{package}.registration:register",
        package_name=distribution_name,
    )
    provider = _provider(entry_point)
    registry = ExtensionRegistry(include_bundled=False, entry_points_provider=provider)

    records = registry.discover()
    assert len(records) == 1
    assert records[0].status == ExtensionStatus.INSTALLED
    assert records[0].package_name == distribution_name
    assert records[0].spec is not None
    assert records[0].spec.package_name == distribution_name

    harness = ExtensionTestHarness.from_entry_point("lab_gain", entry_points_provider=provider)
    harness.assert_all_static_contracts()
    _eeg, command = harness.assert_pop_function_history_result("pop_lab_gain", _demo_eeg(), gain=2.0)
    assert command == "EEG = pop_lab_gain(EEG, gain=2.0);"

    runtime = ExtensionRuntime.from_records(records)
    tools_menu = _child(eeglab_menus(all_menus=True, extension_runtime=runtime), "Tools")
    lab_menu = _child(tools_menu.children, "Lab extensions")
    assert _child(lab_menu.children, "Apply lab gain").action == "pop_lab_gain"

    catalog = load_extension_catalog(catalog_path)
    plugins = plugin_menu(registry=registry, catalog=catalog, include_bundled=False, show=False)
    assert len(plugins) == 1
    assert plugins[0]["plugin"] == "lab_gain_extension"
    assert plugins[0]["package_name"] == distribution_name
    assert plugins[0]["catalog_status"] == "curated"
    assert plugins[0]["catalog_conflicts"] == ()

    help_text, source_path = pophelp_text("pop_lab_gain", extension_runtime=runtime)
    assert "POP_LAB_GAIN" in help_text
    assert source_path == f"{package}:resources/help/pop_lab_gain.md"

    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    echoed: list[str] = []
    session.add_command_echo_listener(echoed.append)
    dispatcher = MenuActionDispatcher(session, extension_runtime=runtime)

    dispatcher.dispatch("pop_lab_gain")

    assert session.EEG["setname"] == "lab-gain"
    np.testing.assert_allclose(session.EEG["data"], np.array([[2.0, 4.0], [6.0, 8.0]]))
    assert session.ALLCOM == ["EEG = pop_lab_gain(EEG, gain=2.0);"]
    assert echoed == ["EEG = pop_lab_gain(EEG, gain=2.0);"]

    workspace = EEGPrepConsoleWorkspace(session, exports={}, extension_runtime=runtime)
    result = workspace.namespace["pop_lab_gain"](workspace.namespace["EEG"], gain=3.0)
    workspace.after_execute("pop_lab_gain(EEG, gain=3.0)")

    assert result.command == "EEG = pop_lab_gain(EEG, gain=3.0);"
    assert workspace.namespace["EEG"] is session.EEG
    np.testing.assert_allclose(session.EEG["data"], np.array([[6.0, 12.0], [18.0, 24.0]]))
    assert session.ALLCOM == [
        "EEG = pop_lab_gain(EEG, gain=2.0);",
        "EEG = pop_lab_gain(EEG, gain=3.0);",
    ]


def test_real_installed_entry_points_are_lazy_resource_safe_and_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_root = tmp_path / "Windows style lab path with spaces"
    alpha_package = "phase6_alpha_extension"
    beta_package = "phase6_beta_extension"
    _write_installed_extension(
        install_root,
        alpha_package,
        distribution_name="eegprep-ext-phase6-alpha",
        entry_point_name="phase6_alpha",
        files={
            "registration.py": """
                from eegprep import ExtensionAction, ExtensionMenu, ExtensionResource, ExtensionSpec, LazyImport

                def register():
                    return ExtensionSpec(
                        name="phase6_alpha_extension",
                        display_name="Phase 6 Alpha",
                        version="0.1.0",
                        package_name="wrong-local-name",
                        actions=(
                            ExtensionAction("phase6.alpha", LazyImport("phase6_alpha_extension.heavy_action", "run")),
                        ),
                        menus=(
                            ExtensionMenu(path=("Tools", "Phase 6"), action="phase6.alpha", label="Alpha action"),
                        ),
                        package_data_resources=(
                            ExtensionResource("phase6_alpha_extension", "resources/data/config value.json"),
                        ),
                    )
            """,
            "heavy_action.py": """
                def run():
                    return "alpha"
            """,
            "resources/data/config value.json": '{"scale": 2}',
        },
        monkeypatch=monkeypatch,
    )
    _write_installed_extension(
        install_root,
        beta_package,
        distribution_name="eegprep-ext-phase6-beta",
        entry_point_name="phase6_beta",
        files={
            "registration.py": """
                from eegprep import ExtensionAction, ExtensionMenu, ExtensionSpec, LazyImport

                def register():
                    return ExtensionSpec(
                        name="phase6_beta_extension",
                        display_name="Phase 6 Beta",
                        version="0.1.0",
                        actions=(
                            ExtensionAction("phase6.beta", LazyImport("phase6_beta_extension.heavy_action", "run")),
                        ),
                        menus=(
                            ExtensionMenu(path=("Tools", "Phase 6"), action="phase6.beta", label="Beta action"),
                        ),
                    )
            """,
            "heavy_action.py": """
                def run():
                    return "beta"
            """,
        },
        monkeypatch=monkeypatch,
    )
    registry = ExtensionRegistry(
        include_bundled=False,
        entry_points_provider=_installed_entry_points("phase6_beta", "phase6_alpha"),
    )

    records = registry.discover()
    runtime = ExtensionRuntime.from_records(records)
    menus = eeglab_menus(all_menus=True, extension_runtime=runtime)

    assert [record.name for record in records] == ["phase6_alpha_extension", "phase6_beta_extension"]
    assert [record.entry_point_name for record in records] == ["phase6_alpha", "phase6_beta"]
    assert [record.package_name for record in records] == [
        "eegprep-ext-phase6-alpha",
        "eegprep-ext-phase6-beta",
    ]
    assert all(record.status == ExtensionStatus.INSTALLED for record in records)
    assert f"{alpha_package}.heavy_action" not in sys.modules
    assert f"{beta_package}.heavy_action" not in sys.modules
    assert records[0].spec is not None
    assert records[0].spec.package_data_resources[0].read_text() == '{"scale": 2}\n'
    assert _child(_child(menus, "Tools").children, "Phase 6").children[0].action == "phase6.alpha"
    assert f"{alpha_package}.heavy_action" not in sys.modules

    assert runtime.action("phase6.alpha").load()() == "alpha"
    assert f"{alpha_package}.heavy_action" in sys.modules


def test_disabled_installed_extension_removes_previous_menu_contribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_root = tmp_path / "disabled extension path"
    _write_installed_extension(
        install_root,
        "phase6_disabled_extension",
        distribution_name="eegprep-ext-phase6-disabled",
        entry_point_name="phase6_disabled",
        files={
            "registration.py": """
                from eegprep import ExtensionAction, ExtensionMenu, ExtensionSpec, LazyImport

                def register():
                    return ExtensionSpec(
                        name="phase6_disabled_extension",
                        display_name="Phase 6 Disabled",
                        version="0.1.0",
                        actions=(
                            ExtensionAction(
                                "phase6.disabled",
                                LazyImport("phase6_disabled_extension.actions", "run"),
                            ),
                        ),
                        menus=(
                            ExtensionMenu(path=("Tools",), action="phase6.disabled", label="Disabled action"),
                        ),
                    )
            """,
            "actions.py": """
                def run():
                    return None
            """,
        },
        monkeypatch=monkeypatch,
    )
    provider = _installed_entry_points("phase6_disabled")
    enabled_records = ExtensionRegistry(include_bundled=False, entry_points_provider=provider).discover()
    enabled_runtime = ExtensionRuntime.from_records(enabled_records)
    disabled_registry = ExtensionRegistry(
        disabled_extensions={"phase6_disabled_extension"},
        include_bundled=False,
        entry_points_provider=provider,
    )

    disabled_records = disabled_registry.discover()
    disabled_runtime = ExtensionRuntime.from_records(disabled_records)

    assert "phase6.disabled" in _menu_actions(eeglab_menus(all_menus=True, extension_runtime=enabled_runtime))
    assert disabled_records[0].status == ExtensionStatus.DISABLED
    assert disabled_records[0].enabled is False
    assert "phase6.disabled" not in _menu_actions(eeglab_menus(all_menus=True, extension_runtime=disabled_runtime))


def test_no_plugins_mode_skips_installed_entry_point_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_root = tmp_path / "no plugins path"
    _write_installed_extension(
        install_root,
        "phase6_skip_extension",
        distribution_name="eegprep-ext-phase6-skip",
        entry_point_name="phase6_skip",
        files={
            "registration.py": """
                raise RuntimeError("entry point should not import in no-plugins mode")
            """,
        },
        monkeypatch=monkeypatch,
    )
    registry = ExtensionRegistry(
        include_bundled=False,
        entry_points_provider=_installed_entry_points("phase6_skip"),
    )

    assert registry.discover(include_plugins=False) == ()
    assert ExtensionRuntime.discover(include_plugins=False).menu_contributions == ()
    assert "phase6_skip_extension.registration" not in sys.modules


def test_startup_imports_do_not_load_installed_extension_entry_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_root = tmp_path / "startup path with spaces"
    _write_installed_extension(
        install_root,
        "phase6_startup_extension",
        distribution_name="eegprep-ext-phase6-startup",
        entry_point_name="phase6_startup",
        files={
            "registration.py": """
                raise RuntimeError("startup imports must not load extension entry points")
            """,
        },
        monkeypatch=monkeypatch,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(install_root), env.get("PYTHONPATH", "")))

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import eegprep; "
                "from eegprep.functions.adminfunc.eeglab import main as gui_main; "
                "from eegprep.functions.adminfunc.console import main as console_main; "
                "assert gui_main and console_main; "
                "assert 'phase6_startup_extension.registration' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 0, result.stderr


def _provider(*entry_points: _FakeEntryPoint):
    def select(*, group: str) -> tuple[_FakeEntryPoint, ...]:
        return tuple(entry_point for entry_point in entry_points if entry_point.group == group)

    return select


def _installed_entry_points(*names: str):
    expected = set(names)

    def select(*, group: str):
        return tuple(entry_point for entry_point in metadata.entry_points(group=group) if entry_point.name in expected)

    return select


def _write_installed_extension(
    install_root: Path,
    package: str,
    *,
    distribution_name: str,
    entry_point_name: str,
    files: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_root.mkdir(parents=True, exist_ok=True)
    _write_package(install_root, package, files, monkeypatch)
    dist_info = install_root / f"{distribution_name.replace('-', '_')}-0.1.0.dist-info"
    dist_info.mkdir(exist_ok=True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {distribution_name}\nVersion: 0.1.0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        f"[{EXTENSION_ENTRY_POINT_GROUP}]\n{entry_point_name} = {package}.registration:register\n",
        encoding="utf-8",
    )


def _write_package(
    tmp_path: Path,
    package: str,
    files: dict[str, str],
    monkeypatch,
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


def _child(items, label: str):
    for item in items:
        if item.label == label:
            return item
    raise AssertionError(f"missing menu item {label!r}")


def _menu_actions(items) -> set[str]:
    actions: set[str] = set()
    for item in items:
        if item.action:
            actions.add(item.action)
        actions.update(_menu_actions(item.children))
    return actions


def _demo_eeg() -> dict[str, Any]:
    return {
        "setname": "demo",
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
