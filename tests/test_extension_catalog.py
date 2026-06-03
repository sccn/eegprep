"""Tests for extension catalog metadata validation."""

from __future__ import annotations

import importlib
import sys
import textwrap
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from eegprep.extension_catalog import (
    CATALOG_SCHEMA_VERSION,
    CatalogValidationOptions,
    load_catalog_entries,
    validate_catalog_entries,
    validate_catalog_file,
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


class BrokenEntryPoint(FakeEntryPoint):
    def load(self) -> Any:
        raise RuntimeError("import failed")


def test_static_catalog_entry_is_valid_without_installed_package() -> None:
    report = validate_catalog_entries([_catalog_entry()])

    assert report.ok
    assert report.warnings == ()


def test_load_catalog_entries_accepts_future_index_payload(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        textwrap.dedent(
            """
            {
              "schema_version": 1,
              "extensions": [
                {
                  "id": "sample_extension",
                  "package_name": "eegprep-ext-sample",
                  "entry_point": "sample",
                  "extension_name": "sample_extension",
                  "display_name": "Sample Extension",
                  "version": "1.0.0",
                  "api_version": "1",
                  "eegprep_requires": ">=0.2",
                  "python_requires": ">=3.10",
                  "license": "BSD-3-Clause",
                  "maintainer": {"name": "SCCN", "email": "maintainers@example.org"},
                  "docs_url": "https://example.org/docs",
                  "source_url": "https://example.org/source",
                  "description": "Sample extension catalog metadata.",
                  "curation": {"status": "submitted"}
                }
              ]
            }
            """
        ).strip(),
        encoding="utf-8",
    )

    entries = load_catalog_entries(catalog)

    assert len(entries) == 1
    assert entries[0]["id"] == "sample_extension"
    assert validate_catalog_file(catalog).ok


def test_schema_version_mismatch_is_reported(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        '{"schema_version": 999, "extensions": []}',
        encoding="utf-8",
    )

    report = validate_catalog_file(catalog)

    assert not report.ok
    assert f"schema_version must be {CATALOG_SCHEMA_VERSION}" in report.errors[0].message


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


def test_non_object_catalog_entry_is_reported() -> None:
    report = validate_catalog_entries(["not-an-entry"])

    assert not report.ok
    assert "Catalog entries must be mapping objects" in _messages(report)


def test_missing_license_maintainer_and_docs_are_blocking() -> None:
    entry = _catalog_entry(license="unknown", maintainer={}, docs_url="")

    report = validate_catalog_entries([entry])

    assert not report.ok
    messages = _messages(report)
    assert "license: License must identify the extension license" in messages
    assert "maintainer: Required catalog metadata must not be empty" in messages
    assert "docs_url: Required catalog metadata must not be empty" in messages


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


def test_missing_installed_package_is_reported() -> None:
    def version_provider(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    report = validate_catalog_entries(
        [_catalog_entry()],
        options=CatalogValidationOptions(check_installed=True, version_provider=version_provider),
    )

    assert not report.ok
    assert "package_name: Package 'eegprep-ext-example' is not installed" in _messages(report)


def test_package_without_entry_point_is_reported() -> None:
    report = validate_catalog_entries(
        [_catalog_entry()],
        options=CatalogValidationOptions(
            check_installed=True,
            version_provider=lambda name: "1.0.0",
            entry_points_provider=_provider(),
        ),
    )

    assert not report.ok
    assert "does not expose 'example'" in _messages(report)


def test_entry_point_import_failure_is_reported() -> None:
    report = validate_catalog_entries(
        [_catalog_entry()],
        options=CatalogValidationOptions(
            check_import=True,
            version_provider=lambda name: "1.0.0",
            entry_points_provider=_provider(
                BrokenEntryPoint("example", "example_pkg.register:register", package_name="eegprep-ext-example")
            ),
        ),
    )

    assert not report.ok
    assert "failed to import" in _messages(report)
    assert "import failed" in _messages(report)


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


def test_catalog_version_mismatch_with_installed_package_is_reported() -> None:
    report = validate_catalog_entries(
        [_catalog_entry()],
        options=CatalogValidationOptions(
            check_installed=True,
            version_provider=lambda name: "2.0.0",
            entry_points_provider=_provider(
                FakeEntryPoint("example", "example_pkg.register:register", package_name="eegprep-ext-example")
            ),
        ),
    )

    assert not report.ok
    assert "version: Catalog version 1.0.0 does not match installed package version 2.0.0" in _messages(report)


def test_unsupported_eegprep_version_is_reported() -> None:
    report = validate_catalog_entries([_catalog_entry(eegprep_requires=">=999.0")])

    assert not report.ok
    assert "requires EEGPrep >=999.0" in _messages(report)


def test_private_internal_extension_is_not_publicly_curated_by_default() -> None:
    entry = _catalog_entry(private=True, curation={"status": "private"})

    public_report = validate_catalog_entries([entry])
    private_report = validate_catalog_entries(
        [entry],
        options=CatalogValidationOptions(allow_private=True),
    )

    assert not public_report.ok
    assert "Private/internal extensions are supported" in _messages(public_report)
    assert private_report.ok
    assert "does not carry curated status" in _messages(private_report)


def test_non_recommended_package_name_warns_but_does_not_drive_discovery() -> None:
    report = validate_catalog_entries([_catalog_entry(package_name="researchlab_example")])

    assert report.ok
    assert "Recommended package names start with 'eegprep-ext-'" in _messages(report)


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
        "python_requires": ">=3.10",
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
