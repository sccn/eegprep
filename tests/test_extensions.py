"""Tests for the external extension SDK and registry."""

from __future__ import annotations

import importlib
import logging
import sys
import textwrap
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from eegprep.extensions import (
    EXTENSION_ENTRY_POINT_GROUP,
    ExtensionDependency,
    ExtensionRegistry,
    ExtensionSpec,
    ExtensionStatus,
    validate_extension_spec,
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


def test_bundled_extension_records_match_plugin_inventory() -> None:
    registry = ExtensionRegistry(include_entry_points=False)

    records = registry.discover()

    assert [record.name for record in records] == ["clean_rawdata", "ICLabel", "firfilt", "dipfit", "EEG_BIDS"]
    assert [record.status for record in records] == [ExtensionStatus.BUNDLED] * 5
    assert all(record.spec is not None for record in records)


def test_entry_point_import_failure_is_isolated_and_logged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    package = "broken_import_extension_pkg"
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
    caplog.set_level(logging.WARNING, logger="eegprep.extensions")
    registry = ExtensionRegistry(
        include_bundled=False,
        entry_points_provider=_provider(FakeEntryPoint("broken", f"{package}.register:register")),
    )

    records = registry.discover()

    assert records[0].status == ExtensionStatus.FAILED_IMPORT
    assert "boom" in records[0].errors[0]
    assert "failed_import" in caplog.text


def test_entry_point_registration_failure_has_accurate_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = "broken_registration_extension_pkg"
    _write_package(
        tmp_path,
        package,
        {
            "register.py": """
                def register():
                    raise RuntimeError("bad config")
            """,
        },
        monkeypatch,
    )
    registry = ExtensionRegistry(
        include_bundled=False,
        entry_points_provider=_provider(FakeEntryPoint("broken-registration", f"{package}.register:register")),
    )

    records = registry.discover()

    assert records[0].status == ExtensionStatus.FAILED_IMPORT
    assert "failed during registration" in records[0].errors[0]
    assert "bad config" in records[0].errors[0]


@pytest.mark.parametrize(
    "spec_args",
    (
        'api_version="2"',
        'eegprep_requires=">=999.0"',
    ),
)
def test_unsupported_api_or_eegprep_version_is_incompatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    spec_args: str,
) -> None:
    package = "incompatible_extension_pkg"
    _write_package(
        tmp_path,
        package,
        {
            "register.py": f"""
                from eegprep.extensions import ExtensionSpec

                def register():
                    return ExtensionSpec(
                        name="incompatible_extension",
                        display_name="Incompatible extension",
                        version="1.0",
                        package_name="incompatible_extension_pkg",
                        {spec_args},
                    )
            """,
        },
        monkeypatch,
    )
    registry = ExtensionRegistry(
        include_bundled=False,
        entry_points_provider=_provider(FakeEntryPoint("incompatible", f"{package}.register:register")),
    )

    records = registry.discover()

    assert records[0].status == ExtensionStatus.INCOMPATIBLE


def test_missing_dependency_is_reported_without_crashing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = "missing_dependency_extension_pkg"
    _write_package(
        tmp_path,
        package,
        {
            "register.py": """
                from eegprep.extensions import ExtensionDependency, ExtensionSpec

                def register():
                    return ExtensionSpec(
                        name="missing_dependency_extension",
                        display_name="Missing dependency extension",
                        version="1.0",
                        package_name="missing_dependency_extension_pkg",
                        dependencies=(ExtensionDependency("missing-eegprep-test-package"),),
                    )
            """,
        },
        monkeypatch,
    )

    def version_provider(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    registry = ExtensionRegistry(
        include_bundled=False,
        entry_points_provider=_provider(FakeEntryPoint("missing-dep", f"{package}.register:register")),
        version_provider=version_provider,
    )

    records = registry.discover()

    assert records[0].status == ExtensionStatus.MISSING_DEPENDENCY
    assert "not installed" in records[0].errors[0]


def test_compatible_release_dependency_spec_enforces_upper_bound() -> None:
    spec = ExtensionSpec(
        name="compatible_dependency_extension",
        version="1.0",
        dependencies=(ExtensionDependency("example-dependency", "~=1.4"),),
    )
    patch_spec = ExtensionSpec(
        name="compatible_patch_dependency_extension",
        version="1.0",
        dependencies=(ExtensionDependency("example-dependency", "~=1.4.5"),),
    )

    assert validate_extension_spec(spec, version_provider=lambda name: "1.9").ok
    assert validate_extension_spec(patch_spec, version_provider=lambda name: "1.4.6").ok
    assert validate_extension_spec(spec, version_provider=lambda name: "2.0").missing_dependency
    assert validate_extension_spec(patch_spec, version_provider=lambda name: "1.5").missing_dependency


def test_duplicate_action_names_mark_later_record_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_basic_extension(tmp_path, monkeypatch, "action_a_pkg", "action_a", action="shared_action")
    _write_basic_extension(tmp_path, monkeypatch, "action_b_pkg", "action_b", action="shared_action")
    registry = ExtensionRegistry(
        include_bundled=False,
        entry_points_provider=_provider(
            FakeEntryPoint("action-a", "action_a_pkg.register:register"),
            FakeEntryPoint("action-b", "action_b_pkg.register:register"),
        ),
    )

    records = registry.discover()

    assert [record.status for record in records] == [ExtensionStatus.INSTALLED, ExtensionStatus.INVALID_SPEC]
    assert "Duplicate action name" in records[1].errors[0]


def test_missing_help_or_package_data_resource_invalidates_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = "missing_resource_extension_pkg"
    _write_package(
        tmp_path,
        package,
        {
            "register.py": """
                from eegprep.extensions import ExtensionResource, ExtensionSpec

                def register():
                    return ExtensionSpec(
                        name="missing_resource_extension",
                        display_name="Missing resource extension",
                        version="1.0",
                        package_name="missing_resource_extension_pkg",
                        help_resources=(
                            ExtensionResource("missing_resource_extension_pkg", "help/missing.md"),
                        ),
                        package_data_resources=(
                            ExtensionResource("missing_resource_extension_pkg", "data/missing.dat"),
                        ),
                    )
            """,
        },
        monkeypatch,
    )
    registry = ExtensionRegistry(
        include_bundled=False,
        entry_points_provider=_provider(FakeEntryPoint("missing-resource", f"{package}.register:register")),
    )

    records = registry.discover()

    assert records[0].status == ExtensionStatus.INVALID_SPEC
    assert any("help/missing.md" in error for error in records[0].errors)
    assert any("data/missing.dat" in error for error in records[0].errors)


def _provider(*entry_points: FakeEntryPoint) -> Any:
    def select(*, group: str) -> tuple[FakeEntryPoint, ...]:
        return tuple(entry_point for entry_point in entry_points if entry_point.group == group)

    return select


def _write_basic_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    package: str,
    extension_name: str,
    *,
    action: str | None = None,
    pop_function: str | None = None,
) -> None:
    register_lines = [
        "from eegprep.extensions import ExtensionAction, ExtensionPopFunction, ExtensionSpec, LazyImport",
        "",
        "def register():",
        "    return ExtensionSpec(",
        f'        name="{extension_name}",',
        f'        display_name="{extension_name}",',
        '        version="1.0",',
        f'        package_name="{package}",',
    ]
    if action:
        register_lines.extend(
            [
                "        actions=(",
                "            ExtensionAction(",
                f'                name="{action}",',
                f'                target=LazyImport("{package}.actions", "run"),',
                "            ),",
                "        ),",
            ]
        )
    if pop_function:
        register_lines.extend(
            [
                "        pop_functions=(",
                "            ExtensionPopFunction(",
                f'                name="{pop_function}",',
                f'                target=LazyImport("{package}.pop_functions", "{pop_function}"),',
                "            ),",
                "        ),",
            ]
        )
    register_lines.extend(["    )", ""])
    _write_package(
        tmp_path,
        package,
        {
            "register.py": "\n".join(register_lines),
            "actions.py": """
                def run():
                    return None
            """,
            "pop_functions.py": f"""
                def {pop_function or "pop_unused"}():
                    return None
            """,
        },
        monkeypatch,
    )


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
