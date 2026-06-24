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
    ExtensionAction,
    ExtensionDependency,
    ExtensionRegistry,
    ExtensionResource,
    ExtensionSpec,
    ExtensionStatus,
    LazyImport,
    extension_api_major_version,
    extension_entry_point_package_name,
    extension_status_is_active,
    extension_status_is_installed,
    extension_version_satisfies,
    select_extension_entry_points,
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

    assert [record.name for record in records] == ["firfilt", "dipfit", "EEG_BIDS"]
    assert [record.status for record in records] == [ExtensionStatus.BUNDLED] * 3
    assert all(record.spec is not None for record in records)


def test_no_extensions_installed_returns_empty_registry() -> None:
    registry = ExtensionRegistry(include_bundled=False, entry_points_provider=_provider())

    assert registry.discover() == ()


def test_include_plugins_false_skips_bundled_and_entry_point_discovery() -> None:
    def fail_provider(*, group: str) -> tuple[FakeEntryPoint, ...]:
        raise AssertionError(f"entry points should not be queried for {group}")

    registry = ExtensionRegistry(entry_points_provider=fail_provider)

    assert registry.discover(include_plugins=False) == ()


def test_entry_point_discovery_keeps_action_targets_lazy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = "lazy_extension_pkg"
    _write_package(
        tmp_path,
        package,
        {
            "register.py": """
                from eegprep.extensions import ExtensionAction, ExtensionSpec, LazyImport

                def register():
                    return ExtensionSpec(
                        name="lazy_extension",
                        display_name="Lazy extension",
                        version="1.0",
                        package_name="lazy_extension_pkg",
                        actions=(
                            ExtensionAction(
                                name="lazy_action",
                                target=LazyImport("lazy_extension_pkg.heavy", "run"),
                            ),
                        ),
                    )
            """,
            "heavy.py": """
                def run():
                    return "ran"
            """,
        },
        monkeypatch,
    )
    registry = ExtensionRegistry(
        include_bundled=False,
        entry_points_provider=_provider(FakeEntryPoint("lazy", f"{package}.register:register")),
    )

    records = registry.discover()

    assert [record.name for record in records] == ["lazy_extension"]
    assert records[0].status == ExtensionStatus.INSTALLED
    assert f"{package}.heavy" not in sys.modules
    assert records[0].spec is not None
    action = records[0].spec.actions[0].load()
    assert action() == "ran"
    assert f"{package}.heavy" in sys.modules


def test_entry_point_provider_without_group_keyword_uses_select_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_basic_extension(tmp_path, monkeypatch, "selectable_pkg", "selectable_extension", action="selectable_action")

    class SelectableEntryPoints:
        def select(self, *, group: str) -> tuple[FakeEntryPoint, ...]:
            return tuple(entry_point for entry_point in entry_points if entry_point.group == group)

    entry_points = (FakeEntryPoint("selectable", "selectable_pkg.register:register"),)

    def provider() -> SelectableEntryPoints:
        return SelectableEntryPoints()

    registry = ExtensionRegistry(include_bundled=False, entry_points_provider=provider)

    records = registry.discover()

    assert [record.name for record in records] == ["selectable_extension"]


def test_extension_metadata_helpers_are_shared_for_registry_and_validation() -> None:
    entry_point = FakeEntryPoint("helper", "helper_pkg.register:register")

    assert select_extension_entry_points(_provider(entry_point), EXTENSION_ENTRY_POINT_GROUP) == (entry_point,)
    assert extension_entry_point_package_name(entry_point) == "helper"
    assert extension_api_major_version("1.2.3") == 1
    assert extension_status_is_active(ExtensionStatus.INSTALLED)
    assert extension_status_is_active(ExtensionStatus.BUNDLED.value)
    assert not extension_status_is_active(ExtensionStatus.FAILED_IMPORT)
    assert extension_status_is_installed(ExtensionStatus.FAILED_IMPORT)
    assert not extension_status_is_installed(ExtensionStatus.CURATED)


def test_registry_records_are_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_basic_extension(tmp_path, monkeypatch, "beta_pkg", "beta_extension", action="beta_action")
    _write_basic_extension(tmp_path, monkeypatch, "alpha_pkg", "alpha_extension", action="alpha_action")
    registry = ExtensionRegistry(
        include_bundled=False,
        entry_points_provider=_provider(
            FakeEntryPoint("beta", "beta_pkg.register:register"),
            FakeEntryPoint("alpha", "alpha_pkg.register:register"),
        ),
    )

    records = registry.discover()

    assert [record.name for record in records] == ["alpha_extension", "beta_extension"]


def test_entry_point_returning_wrong_type_is_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = "wrong_type_extension_pkg"
    _write_package(
        tmp_path,
        package,
        {
            "register.py": """
                def register():
                    return {"name": "not-a-spec"}
            """,
        },
        monkeypatch,
    )
    registry = ExtensionRegistry(
        include_bundled=False,
        entry_points_provider=_provider(FakeEntryPoint("wrong-type", f"{package}.register:register")),
    )

    records = registry.discover()

    assert records[0].status == ExtensionStatus.INVALID_SPEC
    assert "not ExtensionSpec" in records[0].errors[0]


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


def test_version_specifiers_use_pep440_semantics() -> None:
    assert extension_version_satisfies("1.0.0", ">=1.0.0rc1")
    assert not extension_version_satisfies("1.0.0a1", ">=1.0.0")
    result = validate_extension_spec(ExtensionSpec(name="bad_spec", eegprep_requires="not-a-spec"))
    assert "not a valid version specifier" in result.invalid_spec[0]


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


def test_disabled_extension_stays_visible_without_dependency_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = "disabled_extension_pkg"
    _write_package(
        tmp_path,
        package,
        {
            "register.py": """
                from eegprep.extensions import ExtensionDependency, ExtensionSpec

                def register():
                    return ExtensionSpec(
                        name="disabled_extension",
                        display_name="Disabled extension",
                        version="1.0",
                        package_name="disabled_extension_pkg",
                        dependencies=(ExtensionDependency("disabled-dependency"),),
                    )
            """,
        },
        monkeypatch,
    )

    def version_provider(name: str) -> str:
        raise AssertionError(f"disabled dependency should not be checked: {name}")

    registry = ExtensionRegistry(
        disabled_extensions={"disabled_extension"},
        include_bundled=False,
        entry_points_provider=_provider(FakeEntryPoint("disabled", f"{package}.register:register")),
        version_provider=version_provider,
    )

    records = registry.discover()

    assert records[0].status == ExtensionStatus.DISABLED
    assert records[0].enabled is False


def test_disabled_extension_status_wins_over_structural_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = "disabled_invalid_extension_pkg"
    _write_package(
        tmp_path,
        package,
        {
            "register.py": """
                from eegprep.extensions import ExtensionAction, ExtensionSpec

                def register():
                    return ExtensionSpec(
                        name="disabled_invalid_extension",
                        display_name="Disabled invalid extension",
                        version="1.0",
                        actions=(
                            ExtensionAction(name="bad_action", target="not-a-lazy-import"),
                        ),
                    )
            """,
        },
        monkeypatch,
    )
    registry = ExtensionRegistry(
        disabled_extensions={"disabled_invalid_extension"},
        include_bundled=False,
        entry_points_provider=_provider(FakeEntryPoint("disabled-invalid", f"{package}.register:register")),
    )

    records = registry.discover()

    assert records[0].status == ExtensionStatus.DISABLED
    assert "target must be a LazyImport" in records[0].errors[0]


def test_unordered_sets_are_normalized_deterministically() -> None:
    action_a = ExtensionAction(name="action_a", target=LazyImport("pkg.actions", "a"))
    action_b = ExtensionAction(name="action_b", target=LazyImport("pkg.actions", "b"))

    spec = ExtensionSpec(
        name="set_order_extension",
        version="1.0",
        capabilities={"beta", "alpha"},
        actions={action_b, action_a},
    )

    assert spec.capabilities == ("alpha", "beta")
    assert [action.name for action in spec.actions] == ["action_a", "action_b"]


def test_duplicate_extension_names_mark_later_record_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_basic_extension(tmp_path, monkeypatch, "duplicate_a_pkg", "duplicate_extension", action="first_action")
    _write_basic_extension(tmp_path, monkeypatch, "duplicate_b_pkg", "duplicate_extension", action="second_action")
    registry = ExtensionRegistry(
        include_bundled=False,
        entry_points_provider=_provider(
            FakeEntryPoint("duplicate-a", "duplicate_a_pkg.register:register"),
            FakeEntryPoint("duplicate-b", "duplicate_b_pkg.register:register"),
        ),
    )

    records = registry.discover()

    assert [record.status for record in records] == [ExtensionStatus.INSTALLED, ExtensionStatus.INVALID_SPEC]
    assert "Duplicate extension name" in records[1].errors[0]


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


def test_duplicate_pop_function_names_mark_later_record_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_basic_extension(tmp_path, monkeypatch, "pop_a_pkg", "pop_a", pop_function="pop_shared")
    _write_basic_extension(tmp_path, monkeypatch, "pop_b_pkg", "pop_b", pop_function="pop_shared")
    registry = ExtensionRegistry(
        include_bundled=False,
        entry_points_provider=_provider(
            FakeEntryPoint("pop-a", "pop_a_pkg.register:register"),
            FakeEntryPoint("pop-b", "pop_b_pkg.register:register"),
        ),
    )

    records = registry.discover()

    assert [record.status for record in records] == [ExtensionStatus.INSTALLED, ExtensionStatus.INVALID_SPEC]
    assert "Duplicate pop function" in records[1].errors[0]


def test_lazy_action_import_failure_happens_after_valid_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = "lazy_failure_extension_pkg"
    _write_package(
        tmp_path,
        package,
        {
            "register.py": """
                from eegprep.extensions import ExtensionAction, ExtensionSpec, LazyImport

                def register():
                    return ExtensionSpec(
                        name="lazy_failure_extension",
                        display_name="Lazy failure extension",
                        version="1.0",
                        package_name="lazy_failure_extension_pkg",
                        actions=(
                            ExtensionAction(
                                name="lazy_failure_action",
                                target=LazyImport("lazy_failure_extension_pkg.missing", "run"),
                            ),
                        ),
                    )
            """,
        },
        monkeypatch,
    )
    registry = ExtensionRegistry(
        include_bundled=False,
        entry_points_provider=_provider(FakeEntryPoint("lazy-failure", f"{package}.register:register")),
    )

    records = registry.discover()

    assert records[0].status == ExtensionStatus.INSTALLED
    assert records[0].spec is not None
    with pytest.raises(RuntimeError, match="Could not load extension target"):
        records[0].spec.actions[0].load()


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


def test_extension_resource_readers_raise_for_missing_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = "resource_reader_extension_pkg"
    _write_package(
        tmp_path,
        package,
        {
            "data/message.txt": "hello resource",
        },
        monkeypatch,
    )
    resource = ExtensionResource(package, "data/message.txt")
    missing = ExtensionResource(package, "data/missing.txt")

    assert resource.read_text() == "hello resource\n"
    assert resource.read_bytes().decode("utf-8").splitlines() == ["hello resource"]
    with pytest.raises(FileNotFoundError, match="data/missing.txt"):
        missing.read_text()
    with pytest.raises(FileNotFoundError, match="data/missing.txt"):
        missing.read_bytes()


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
