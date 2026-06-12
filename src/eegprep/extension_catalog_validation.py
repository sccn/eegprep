"""Submission-curation validator for EEGPrep extension catalog metadata.

This module powers the ``eegprep-validate-extension-catalog`` curation CI command.
It is separate from the runtime Extension Manager catalog loader in
``eegprep.extension_catalog``; the two share only a handful of catalog
constants and the ``_is_web_url`` helper.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from eegprep.extensions import (
    EXTENSION_API_VERSION,
    EXTENSION_ENTRY_POINT_GROUP,
    EXTENSION_NAMING_PREFIX,
    ExtensionDependency,
    ExtensionRegistry,
    ExtensionSpec,
    ExtensionStatus,
    _entry_point_package_name,
    _major_version,
    _select_entry_points,
    check_extension_compatibility,
    extension_version_satisfies,
    extension_version_spec_is_valid,
)

CATALOG_CURATION_STATUSES = ("submitted", "curated", "private", "internal")
CATALOG_REQUIRED_FIELDS = (
    "id",
    "package_name",
    "entry_point",
    "extension_name",
    "version",
    "api_version",
    "eegprep_requires",
    "python_requires",
    "license",
    "maintainer",
    "docs_url",
    "source_url",
    "description",
    "curation",
)

_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_PACKAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class CatalogValidationIssue:
    """One catalog metadata validation issue."""

    message: str
    entry_id: str = ""
    field: str = ""

    def format(self) -> str:
        """Return a concise human-readable issue line."""
        location = self.entry_id or "<catalog>"
        if self.field:
            location = f"{location}.{self.field}"
        return f"{location}: {self.message}"


@dataclass(frozen=True)
class CatalogValidationReport:
    """Catalog validation result with blocking errors and non-blocking warnings."""

    errors: tuple[CatalogValidationIssue, ...] = field(default_factory=tuple)
    warnings: tuple[CatalogValidationIssue, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """Return whether the catalog metadata has no blocking errors."""
        return not self.errors

    def format(self) -> str:
        """Return a readable validation summary."""
        lines: list[str] = []
        if self.errors:
            lines.append("Errors:")
            lines.extend(f"- {issue.format()}" for issue in self.errors)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {issue.format()}" for issue in self.warnings)
        if not lines:
            return "Catalog metadata is valid."
        return "\n".join(lines)


@dataclass(frozen=True)
class CatalogValidationOptions:
    """Validation switches for local and future catalog-CI checks."""

    allow_private: bool = False
    check_installed: bool = False
    check_import: bool = False
    current_eegprep_version: str | None = None
    version_provider: Any = metadata.version
    entry_points_provider: Any = metadata.entry_points


def load_catalog_entries(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load catalog entries from a JSON file or a directory of JSON files."""
    catalog_path = Path(path)
    if catalog_path.is_dir():
        files = _catalog_files(catalog_path)
        entries: list[dict[str, Any]] = []
        for file_path in files:
            entries.extend(load_catalog_entries(file_path))
        return tuple(entries)

    with catalog_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return _entries_from_payload(payload, catalog_path)


def validate_catalog_file(
    path: str | Path,
    *,
    options: CatalogValidationOptions | None = None,
) -> CatalogValidationReport:
    """Validate catalog metadata loaded from ``path``."""
    try:
        entries = load_catalog_entries(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return CatalogValidationReport(errors=(CatalogValidationIssue(str(exc)),))
    return validate_catalog_entries(entries, options=options)


def validate_catalog_entries(
    entries: Iterable[Mapping[str, Any]],
    *,
    options: CatalogValidationOptions | None = None,
) -> CatalogValidationReport:
    """Validate catalog metadata entries.

    Static validation does not require an installed extension package. Set
    ``check_installed`` to verify distribution metadata and entry points, and
    ``check_import`` to import the declared entry point through the registry.
    """
    opts = options or CatalogValidationOptions()
    errors: list[CatalogValidationIssue] = []
    warnings: list[CatalogValidationIssue] = []
    normalized_entries: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            errors.append(CatalogValidationIssue("Catalog entries must be mapping objects", f"entry[{index}]"))
            continue
        normalized_entries.append(dict(entry))

    _validate_static_entries(normalized_entries, opts, errors, warnings)
    _validate_catalog_conflicts(normalized_entries, errors)
    if opts.check_installed or opts.check_import:
        _validate_installed_entries(normalized_entries, opts, errors)

    return CatalogValidationReport(errors=tuple(errors), warnings=tuple(warnings))


def _catalog_files(path: Path) -> tuple[Path, ...]:
    catalog_json = path / "catalog.json"
    if catalog_json.is_file():
        return (catalog_json,)
    return tuple(sorted(candidate for candidate in path.rglob("*.json") if candidate.is_file()))


def _entries_from_payload(payload: Any, path: Path) -> tuple[dict[str, Any], ...]:
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict) and "extensions" in payload:
        _validate_schema_version(payload, path)
        entries = payload["extensions"]
    elif isinstance(payload, dict) and "entries" in payload:
        _validate_schema_version(payload, path)
        entries = payload["entries"]
    elif isinstance(payload, dict) and "id" in payload:
        entries = [payload]
    else:
        raise ValueError(f"{path} must contain an extension entry or an 'extensions' list")

    if not isinstance(entries, list):
        raise ValueError(f"{path} extensions payload must be a list")
    if not all(isinstance(entry, dict) for entry in entries):
        raise ValueError(f"{path} extension entries must be JSON objects")
    return tuple(dict(entry) for entry in entries)


def _validate_schema_version(payload: Mapping[str, Any], path: Path) -> None:
    # Local import breaks the extension_catalog <-> extension_catalog_validation cycle.
    from eegprep.extension_catalog import CATALOG_KIND_CURATION, CATALOG_SCHEMA_VERSION

    catalog_kind = payload.get("catalog_kind", CATALOG_KIND_CURATION)
    if catalog_kind != CATALOG_KIND_CURATION:
        raise ValueError(f"{path} catalog_kind must be {CATALOG_KIND_CURATION!r}; got {catalog_kind!r}")
    schema_version = payload.get("schema_version", CATALOG_SCHEMA_VERSION)
    if schema_version != CATALOG_SCHEMA_VERSION:
        raise ValueError(f"{path} schema_version must be {CATALOG_SCHEMA_VERSION}; got {schema_version!r}")


def _validate_static_entries(
    entries: list[dict[str, Any]],
    options: CatalogValidationOptions,
    errors: list[CatalogValidationIssue],
    warnings: list[CatalogValidationIssue],
) -> None:
    for entry in entries:
        entry_id = _entry_id(entry)
        _validate_required_fields(entry, entry_id, errors)
        _validate_names(entry, entry_id, errors, warnings)
        _validate_urls(entry, entry_id, errors)
        _validate_text_metadata(entry, entry_id, errors)
        _validate_curation(entry, entry_id, options, errors, warnings)
        _validate_version_policy(entry, entry_id, options, errors)
        _validate_dependency_metadata(entry, entry_id, options, errors)


def _validate_required_fields(entry: Mapping[str, Any], entry_id: str, errors: list[CatalogValidationIssue]) -> None:
    for field_name in CATALOG_REQUIRED_FIELDS:
        if field_name not in entry:
            errors.append(CatalogValidationIssue("Required catalog metadata is missing", entry_id, field_name))
            continue
        value = entry[field_name]
        if value in ("", None, [], {}):
            errors.append(CatalogValidationIssue("Required catalog metadata must not be empty", entry_id, field_name))


def _validate_names(
    entry: Mapping[str, Any],
    entry_id: str,
    errors: list[CatalogValidationIssue],
    warnings: list[CatalogValidationIssue],
) -> None:
    for field_name in ("id", "entry_point", "extension_name"):
        value = entry.get(field_name)
        if isinstance(value, str) and _NAME_RE.match(value):
            continue
        errors.append(
            CatalogValidationIssue(
                "Must start with a letter and contain only letters, numbers, '.', '_', or '-'",
                entry_id,
                field_name,
            )
        )

    package_name = entry.get("package_name")
    if not isinstance(package_name, str) or not _PACKAGE_RE.match(package_name):
        errors.append(
            CatalogValidationIssue(
                "Must contain only letters, numbers, '.', '_', or '-' and start with a letter or number",
                entry_id,
                "package_name",
            )
        )
    elif not package_name.startswith(EXTENSION_NAMING_PREFIX):
        warnings.append(
            CatalogValidationIssue(
                f"Recommended package names start with {EXTENSION_NAMING_PREFIX!r}; discovery still uses entry points",
                entry_id,
                "package_name",
            )
        )


def _validate_urls(entry: Mapping[str, Any], entry_id: str, errors: list[CatalogValidationIssue]) -> None:
    # Local import breaks the extension_catalog <-> extension_catalog_validation cycle.
    from eegprep.extension_catalog import _is_web_url

    for field_name in ("docs_url", "source_url"):
        value = entry.get(field_name)
        if isinstance(value, str) and _is_web_url(value):
            continue
        errors.append(CatalogValidationIssue("Must be an https:// or http:// URL", entry_id, field_name))


def _validate_text_metadata(entry: Mapping[str, Any], entry_id: str, errors: list[CatalogValidationIssue]) -> None:
    license_value = entry.get("license")
    if isinstance(license_value, str) and license_value.strip().lower() in {"unknown", "none", "n/a"}:
        errors.append(CatalogValidationIssue("License must identify the extension license", entry_id, "license"))

    maintainer = entry.get("maintainer")
    if isinstance(maintainer, dict):
        maintainer_name = maintainer.get("name")
        maintainer_contact = maintainer.get("email") or maintainer.get("url")
        if not maintainer_name or not maintainer_contact:
            errors.append(
                CatalogValidationIssue(
                    "Maintainer metadata must include a name and email or URL",
                    entry_id,
                    "maintainer",
                )
            )
    elif not isinstance(maintainer, str) or not maintainer.strip():
        errors.append(CatalogValidationIssue("Maintainer metadata must be a string or object", entry_id, "maintainer"))

    description = entry.get("description")
    if isinstance(description, str) and len(description.strip()) < 20:
        errors.append(CatalogValidationIssue("Description must be at least 20 characters", entry_id, "description"))


def _validate_curation(
    entry: Mapping[str, Any],
    entry_id: str,
    options: CatalogValidationOptions,
    errors: list[CatalogValidationIssue],
    warnings: list[CatalogValidationIssue],
) -> None:
    curation = entry.get("curation")
    if not isinstance(curation, dict):
        errors.append(CatalogValidationIssue("Curation metadata must be an object", entry_id, "curation"))
        return

    status = str(curation.get("status") or "").strip().lower()
    if status not in CATALOG_CURATION_STATUSES:
        errors.append(
            CatalogValidationIssue(
                f"Curation status must be one of {', '.join(CATALOG_CURATION_STATUSES)}",
                entry_id,
                "curation.status",
            )
        )
        return

    private = bool(entry.get("private") or entry.get("internal") or status in {"private", "internal"})
    if private and not options.allow_private:
        errors.append(
            CatalogValidationIssue(
                "Private/internal extensions are supported by direct installation, not the public curated catalog",
                entry_id,
                "curation.status",
            )
        )
    elif private:
        warnings.append(
            CatalogValidationIssue(
                "Private/internal extension metadata is valid locally but does not carry curated status",
                entry_id,
                "curation.status",
            )
        )

    if status == "curated":
        if not curation.get("reviewed_by"):
            errors.append(CatalogValidationIssue("Curated entries must record reviewed_by", entry_id, "curation"))
        if not curation.get("reviewed_at"):
            errors.append(CatalogValidationIssue("Curated entries must record reviewed_at", entry_id, "curation"))


def _validate_version_policy(
    entry: Mapping[str, Any],
    entry_id: str,
    options: CatalogValidationOptions,
    errors: list[CatalogValidationIssue],
) -> None:
    api_version = str(entry.get("api_version") or "")
    if _major_version(api_version) != _major_version(EXTENSION_API_VERSION):
        errors.append(
            CatalogValidationIssue(
                f"Extension API version {api_version!r} is not supported by this EEGPrep extension API",
                entry_id,
                "api_version",
            )
        )

    eegprep_requires = str(entry.get("eegprep_requires") or "")
    eegprep_requires_valid = True
    for field_name in ("eegprep_requires", "python_requires"):
        value = entry.get(field_name)
        if not isinstance(value, str) or not extension_version_spec_is_valid(value):
            errors.append(CatalogValidationIssue("Must be a simple version specifier", entry_id, field_name))
            if field_name == "eegprep_requires":
                eegprep_requires_valid = False

    if eegprep_requires and eegprep_requires_valid:
        spec = ExtensionSpec(
            name=str(entry.get("extension_name") or entry.get("id") or "catalog_entry"),
            version=str(entry.get("version") or ""),
            api_version=EXTENSION_API_VERSION,
            package_name=str(entry.get("package_name") or ""),
            eegprep_requires=eegprep_requires,
            dependencies=tuple(_catalog_dependencies(entry)),
        )
        compatibility = check_extension_compatibility(
            spec,
            current_version=options.current_eegprep_version,
            version_provider=options.version_provider,
            check_dependencies=False,
        )
        for message in compatibility.incompatible:
            errors.append(CatalogValidationIssue(message, entry_id, "eegprep_requires"))

    python_requires = str(entry.get("python_requires") or "")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if python_requires and extension_version_spec_is_valid(python_requires):
        if not extension_version_satisfies(python_version, python_requires):
            errors.append(
                CatalogValidationIssue(
                    f"Extension requires Python {python_requires}; current version is {python_version}",
                    entry_id,
                    "python_requires",
                )
            )


def _validate_dependency_metadata(
    entry: Mapping[str, Any],
    entry_id: str,
    options: CatalogValidationOptions,
    errors: list[CatalogValidationIssue],
) -> None:
    if "dependencies" not in entry:
        return
    dependencies = entry.get("dependencies", ())
    if dependencies in ("", None):
        return
    if not isinstance(dependencies, list):
        errors.append(CatalogValidationIssue("Dependencies must be a list", entry_id, "dependencies"))
        return
    for index, dependency in enumerate(dependencies):
        field = f"dependencies[{index}]"
        if not isinstance(dependency, dict):
            errors.append(CatalogValidationIssue("Dependency entries must be objects", entry_id, field))
            continue
        package = dependency.get("package")
        if not isinstance(package, str) or not _PACKAGE_RE.match(package):
            errors.append(CatalogValidationIssue("Dependency package name is invalid", entry_id, f"{field}.package"))
        version_spec = dependency.get("version_spec", "")
        if version_spec and (not isinstance(version_spec, str) or not extension_version_spec_is_valid(version_spec)):
            errors.append(CatalogValidationIssue("Dependency version specifier is invalid", entry_id, field))
        if not options.check_installed:
            continue
        try:
            installed_version = options.version_provider(str(package))
        except metadata.PackageNotFoundError:
            if not dependency.get("optional", False):
                errors.append(
                    CatalogValidationIssue(f"Required dependency {package!r} is not installed", entry_id, field)
                )
            continue
        if version_spec and not extension_version_satisfies(installed_version, str(version_spec)):
            errors.append(
                CatalogValidationIssue(
                    f"Dependency {package!r} requires {version_spec}; installed version is {installed_version}",
                    entry_id,
                    field,
                )
            )


def _validate_catalog_conflicts(entries: list[dict[str, Any]], errors: list[CatalogValidationIssue]) -> None:
    seen: dict[tuple[str, str], str] = {}
    for entry in entries:
        entry_id = _entry_id(entry)
        for field_name in ("id", "extension_name", "display_name"):
            value = entry.get(field_name)
            if not isinstance(value, str) or not value:
                continue
            key = (field_name, _normalize(value))
            owner = seen.get(key)
            if owner is not None:
                errors.append(
                    CatalogValidationIssue(
                        f"Conflicts with catalog entry {owner!r}",
                        entry_id,
                        field_name,
                    )
                )
            seen[key] = entry_id

        package_name = entry.get("package_name")
        entry_point = entry.get("entry_point")
        if isinstance(package_name, str) and isinstance(entry_point, str):
            key = ("package-entry-point", f"{_normalize(package_name)}:{_normalize(entry_point)}")
            owner = seen.get(key)
            if owner is not None:
                errors.append(
                    CatalogValidationIssue(
                        f"Conflicts with catalog entry {owner!r}",
                        entry_id,
                        "entry_point",
                    )
                )
            seen[key] = entry_id


def _validate_installed_entries(
    entries: list[dict[str, Any]],
    options: CatalogValidationOptions,
    errors: list[CatalogValidationIssue],
) -> None:
    selected_entry_points = _select_entry_points(options.entry_points_provider, EXTENSION_ENTRY_POINT_GROUP)
    for entry in entries:
        entry_id = _entry_id(entry)
        package_name = str(entry.get("package_name") or "")
        entry_point_name = str(entry.get("entry_point") or "")
        if not package_name or not entry_point_name:
            continue

        try:
            installed_version = options.version_provider(package_name)
        except metadata.PackageNotFoundError:
            errors.append(
                CatalogValidationIssue(f"Package {package_name!r} is not installed", entry_id, "package_name")
            )
            continue

        expected_version = str(entry.get("version") or "")
        if expected_version and installed_version != expected_version:
            errors.append(
                CatalogValidationIssue(
                    f"Catalog version {expected_version} does not match installed package version {installed_version}",
                    entry_id,
                    "version",
                )
            )

        entry_point = _matching_entry_point(selected_entry_points, package_name, entry_point_name)
        if entry_point is None:
            errors.append(
                CatalogValidationIssue(
                    f"Package {package_name!r} does not expose {entry_point_name!r} in {EXTENSION_ENTRY_POINT_GROUP}",
                    entry_id,
                    "entry_point",
                )
            )
            continue

        if options.check_import:
            _validate_imported_entry(entry, entry_point, options, errors)


def _validate_imported_entry(
    entry: Mapping[str, Any],
    entry_point: Any,
    options: CatalogValidationOptions,
    errors: list[CatalogValidationIssue],
) -> None:
    entry_id = _entry_id(entry)

    def provider(*, group: str) -> tuple[Any, ...]:
        if group == EXTENSION_ENTRY_POINT_GROUP:
            return (entry_point,)
        return ()

    registry = ExtensionRegistry(
        include_bundled=False,
        entry_points_provider=provider,
        current_version=options.current_eegprep_version,
        version_provider=options.version_provider,
    )
    record = registry.discover()[0]
    if record.status in {ExtensionStatus.FAILED_IMPORT, ExtensionStatus.INVALID_SPEC, ExtensionStatus.INCOMPATIBLE}:
        for message in record.errors:
            errors.append(CatalogValidationIssue(message, entry_id, "entry_point"))
        return
    if record.status == ExtensionStatus.MISSING_DEPENDENCY:
        for message in record.errors:
            errors.append(CatalogValidationIssue(message, entry_id, "dependencies"))
        return
    if record.spec is None:
        errors.append(CatalogValidationIssue("Entry point did not return an extension spec", entry_id, "entry_point"))
        return

    _validate_imported_spec_matches_catalog(entry, record.spec, errors)


def _validate_imported_spec_matches_catalog(
    entry: Mapping[str, Any],
    spec: ExtensionSpec,
    errors: list[CatalogValidationIssue],
) -> None:
    entry_id = _entry_id(entry)
    checks = (
        ("extension_name", spec.name),
        ("version", spec.version),
        ("api_version", spec.api_version),
    )
    for field_name, actual in checks:
        expected = str(entry.get(field_name) or "")
        if expected and expected != actual:
            errors.append(
                CatalogValidationIssue(
                    f"Catalog value {expected!r} does not match imported spec value {actual!r}",
                    entry_id,
                    field_name,
                )
            )


def _catalog_dependencies(entry: Mapping[str, Any]) -> tuple[ExtensionDependency, ...]:
    dependencies = entry.get("dependencies", ())
    if not isinstance(dependencies, list):
        return ()
    parsed: list[ExtensionDependency] = []
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            continue
        package = dependency.get("package")
        if not package:
            continue
        parsed.append(
            ExtensionDependency(
                package=str(package),
                version_spec=str(dependency.get("version_spec") or ""),
                optional=bool(dependency.get("optional", False)),
            )
        )
    return tuple(parsed)


def _matching_entry_point(entry_points: tuple[Any, ...], package_name: str, entry_point_name: str) -> Any | None:
    normalized_package = _normalize(package_name)
    for entry_point in entry_points:
        if getattr(entry_point, "name", None) != entry_point_name:
            continue
        entry_point_package = _entry_point_package_name(entry_point)
        if entry_point_package and _normalize(entry_point_package) != normalized_package:
            continue
        return entry_point
    return None


def _entry_id(entry: Mapping[str, Any]) -> str:
    value = entry.get("id")
    if isinstance(value, str) and value:
        return value
    package_name = entry.get("package_name")
    if isinstance(package_name, str) and package_name:
        return package_name
    return "<entry>"


def _normalize(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate EEGPrep extension catalog metadata.")
    parser.add_argument("path", help="Catalog JSON file, entry JSON file, or directory of JSON files")
    parser.add_argument(
        "--allow-private", action="store_true", help="Allow private/internal metadata for local catalogs"
    )
    parser.add_argument(
        "--check-installed", action="store_true", help="Verify installed package metadata and entry points"
    )
    parser.add_argument("--check-import", action="store_true", help="Import declared entry points through the registry")
    parser.add_argument("--json", action="store_true", help="Emit JSON validation output")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the catalog validator command-line interface."""
    args = _build_arg_parser().parse_args(argv)
    options = CatalogValidationOptions(
        allow_private=args.allow_private,
        check_installed=args.check_installed or args.check_import,
        check_import=args.check_import,
    )
    report = validate_catalog_file(args.path, options=options)
    if args.json:
        print(
            json.dumps(
                {
                    "ok": report.ok,
                    "errors": [issue.format() for issue in report.errors],
                    "warnings": [issue.format() for issue in report.warnings],
                },
                indent=2,
            )
        )
    else:
        print(report.format())
    return 0 if report.ok else 1


__all__ = [
    "CATALOG_CURATION_STATUSES",
    "CATALOG_REQUIRED_FIELDS",
    "CatalogValidationIssue",
    "CatalogValidationOptions",
    "CatalogValidationReport",
    "load_catalog_entries",
    "main",
    "validate_catalog_entries",
    "validate_catalog_file",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
