"""Tests for reusable extension author assertions."""

from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from eegprep.extension_testing import ExtensionTestHarness, assert_extension_entry_point_loads
from eegprep.extensions import (
    EXTENSION_ENTRY_POINT_GROUP,
    ExtensionAction,
    ExtensionMenu,
    ExtensionSpec,
    LazyImport,
)


class FakeDistribution:
    def __init__(self, name: str) -> None:
        self.metadata = {"Name": name}


class FakeEntryPoint:
    def __init__(self, name: str, value: str, *, group: str = EXTENSION_ENTRY_POINT_GROUP) -> None:
        self.name = name
        self.value = value
        self.group = group
        self.dist = FakeDistribution(name)

    def load(self) -> Any:
        module_name, _, attr_name = self.value.partition(":")
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)


def test_extension_harness_rejects_menu_without_registered_action() -> None:
    spec = ExtensionSpec(
        name="bad_menu_extension",
        version="1.0.0",
        menus=(ExtensionMenu(("Tools", "Bad"), "missing_action"),),
    )
    harness = ExtensionTestHarness(spec)

    with pytest.raises(AssertionError, match="undeclared action"):
        harness.assert_menus_register()


def test_extension_harness_rejects_missing_history_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = "bad_history_extension_pkg"
    _write_package(
        tmp_path,
        package,
        {
            "actions.py": """
                def run(EEG):
                    return EEG
            """,
        },
        monkeypatch,
    )
    spec = ExtensionSpec(
        name="bad_history_extension",
        version="1.0.0",
        actions=(ExtensionAction("bad.run", LazyImport(f"{package}.actions", "run")),),
    )
    harness = ExtensionTestHarness(spec)

    with pytest.raises(AssertionError, match=r"\(EEG, com\)"):
        harness.assert_action_history_result("bad.run", {"data": []})


def test_assert_extension_entry_point_loads_reports_failed_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = "failed_entry_harness_extension_pkg"
    _write_package(
        tmp_path,
        package,
        {
            "register.py": """
                raise RuntimeError("boom")
            """,
        },
        monkeypatch,
    )

    with pytest.raises(AssertionError, match="boom"):
        assert_extension_entry_point_loads(
            "failed",
            entry_points_provider=_provider(FakeEntryPoint("failed", f"{package}.register:register")),
        )


def _provider(*entry_points: FakeEntryPoint) -> Any:
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
