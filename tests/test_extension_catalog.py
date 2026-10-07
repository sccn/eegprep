"""Tests for extension catalog metadata validation."""

from __future__ import annotations

import importlib
import json
import sys
import textwrap
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from eegprep.extension_catalog import (
    CATALOG_KIND_CURATION,
    CATALOG_SCHEMA_VERSION,
)
from eegprep.extension_catalog_validation import (
    CatalogValidationOptions,
    load_catalog_entries,
    main,
    validate_catalog_entries,
)
from eegprep.extensions import EXTENSION_ENTRY_POINT_GROUP


class FakeDistribution:
    def __init__(self, name: str) -> None:
        self.metadata = {"Name": name}


class FakeEntryPoint:
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
        self.dist = FakeDistribution(package_name)

    def load(self) -> Any:
        module_name, _, attr_name = self.value.partition(":")
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)


def test_static_catalog_entry_is_valid_without_installed_package() -> None:
    report = validate_catalog_entries([_catalog_entry()])

    assert report.ok
    assert report.warnings == ()


def test_load_catalog_entries_accepts_catalog_directories(tmp_path: Path) -> None:
    catalog_dir = tmp_path / "catalog-index"
    catalog_dir.mkdir()
    _write_catalog(catalog_dir / "catalog.json", _catalog_entry(id="canonical", extension_name="canonical_extension"))
    nested_dir = catalog_dir / "nested"
    nested_dir.mkdir()
    _write_entry(nested_dir / "ignored.json", _catalog_entry(id="ignored", extension_name="ignored_extension"))

    split_dir = tmp_path / "split-index"
    split_dir.mkdir()
    _write_entry(split_dir / "alpha.json", _catalog_entry(id="alpha", extension_name="alpha_extension"))
    split_nested_dir = split_dir / "nested"
    split_nested_dir.mkdir()
    _write_entry(split_nested_dir / "beta.json", _catalog_entry(id="beta", extension_name="beta_extension"))

    assert [entry["id"] for entry in load_catalog_entries(catalog_dir)] == ["canonical"]
    assert [entry["id"] for entry in load_catalog_entries(split_dir)] == ["alpha", "beta"]


def test_catalog_cli_emits_json_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    catalog = tmp_path / "catalog.json"
    _write_catalog(catalog, _catalog_entry())

    exit_code = main([str(catalog), "--json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out) == {"ok": True, "errors": [], "warnings": []}


def test_invalid_metadata_rejects_malicious_looking_fields() -> None:
    entry = _catalog_entry(
        id="../bad",
        package_name="bad package",
        entry_point="bad/entry",
        extension_name="1bad",
        docs_url="javascript:alert(1)",
    )

    report = validate_catalog_entries([entry])

    assert not report.ok
    messages = _messages(report)
    assert "id: Must start with a letter" in messages
    assert "package_name: Must contain only letters" in messages
    assert "entry_point: Must start with a letter" in messages
    assert "extension_name: Must start with a letter" in messages
    assert "docs_url: Must be an https:// or http:// URL" in messages


def test_catalog_conflicts_are_reported() -> None:
    first = _catalog_entry(id="shared", extension_name="shared_extension", display_name="Shared")
    second = _catalog_entry(
        id="shared",
        extension_name="shared_extension",
        display_name="Shared",
        package_name="eegprep-ext-other",
    )

    report = validate_catalog_entries([first, second])

    assert not report.ok
    messages = _messages(report)
    assert "shared.id: Conflicts with catalog entry 'shared'" in messages
    assert "shared.extension_name: Conflicts with catalog entry 'shared'" in messages
    assert "shared.display_name: Conflicts with catalog entry 'shared'" in messages


def test_dependency_mismatch_is_reported_when_installed_checks_are_enabled() -> None:
    entry = _catalog_entry(dependencies=[{"package": "example-dependency", "version_spec": ">=2.0"}])

    def version_provider(name: str) -> str:
        if name == "eegprep-ext-example":
            return "1.0.0"
        if name == "example-dependency":
            return "1.0"
        raise metadata.PackageNotFoundError(name)

    report = validate_catalog_entries(
        [entry],
        options=CatalogValidationOptions(
            check_installed=True,
            version_provider=version_provider,
            entry_points_provider=_provider(
                FakeEntryPoint("example", "example_pkg.register:register", package_name="eegprep-ext-example")
            ),
        ),
    )

    assert not report.ok
    assert "Dependency 'example-dependency' requires >=2.0; installed version is 1.0" in _messages(report)


def test_imported_spec_mismatch_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = "catalog_mismatch_extension_pkg"
    _write_package(
        tmp_path,
        package,
        {
            "register.py": """
                from eegprep.extensions import ExtensionSpec

                def register():
                    return ExtensionSpec(
                        name="other_extension",
                        version="2.0.0",
                        package_name="eegprep-ext-example",
                    )
            """,
        },
        monkeypatch,
    )

    report = validate_catalog_entries(
        [_catalog_entry()],
        options=CatalogValidationOptions(
            check_import=True,
            version_provider=lambda name: "1.0.0",
            entry_points_provider=_provider(
                FakeEntryPoint("example", f"{package}.register:register", package_name="eegprep-ext-example")
            ),
        ),
    )

    assert not report.ok
    messages = _messages(report)
    assert (
        "extension_name: Catalog value 'example_extension' does not match imported spec value 'other_extension'"
        in messages
    )
    assert "version: Catalog value '1.0.0' does not match imported spec value '2.0.0'" in messages


def test_imported_spec_matching_catalog_is_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = "catalog_matching_extension_pkg"
    _write_package(
        tmp_path,
        package,
        {
            "register.py": """
                from eegprep.extensions import ExtensionSpec

                def register():
                    return ExtensionSpec(
                        name="example_extension",
                        version="1.0.0",
                        api_version="1",
                        package_name="eegprep-ext-example",
                    )
            """,
        },
        monkeypatch,
    )

    report = validate_catalog_entries(
        [_catalog_entry()],
        options=CatalogValidationOptions(
            check_import=True,
            version_provider=lambda name: "1.0.0",
            entry_points_provider=_provider(
                FakeEntryPoint("example", f"{package}.register:register", package_name="eegprep-ext-example")
            ),
        ),
    )

    assert report.ok
    assert report.warnings == ()


def _catalog_entry(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": "example_extension",
        "package_name": "eegprep-ext-example",
        "entry_point": "example",
        "extension_name": "example_extension",
        "display_name": "Example Extension",
        "version": "1.0.0",
        "api_version": "1",
        "eegprep_requires": ">=0.2",
        # Deliberately far below any interpreter this suite runs on. The happy path
        # must not depend on the project's Python floor, or every floor bump breaks
        # an unrelated fixture. The floor itself is covered by the two tests below.
        "python_requires": ">=3.0",
        "license": "BSD-3-Clause",
        "maintainer": {"name": "SCCN", "email": "maintainers@example.org"},
        "docs_url": "https://example.org/docs",
        "source_url": "https://example.org/source",
        "description": "Example extension catalog metadata.",
        "dependencies": [],
        "curation": {"status": "submitted"},
    }
    entry.update(overrides)
    return entry


def _provider(*entry_points: FakeEntryPoint) -> Any:
    def select(*, group: str) -> tuple[FakeEntryPoint, ...]:
        return tuple(entry_point for entry_point in entry_points if entry_point.group == group)

    return select


def _messages(report: Any) -> str:
    return "\n".join(issue.format() for issue in (*report.errors, *report.warnings))


def _write_catalog(path: Path, *entries: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            {
                "catalog_kind": CATALOG_KIND_CURATION,
                "schema_version": CATALOG_SCHEMA_VERSION,
                "extensions": list(entries),
            }
        ),
        encoding="utf-8",
    )


def _write_entry(path: Path, entry: dict[str, Any]) -> None:
    path.write_text(json.dumps(entry), encoding="utf-8")


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
