"""Runtime Extension Manager catalog loading for EEGPrep.

This module loads the metadata-only catalog that the Extension Manager dialog and
console inventory display, and builds copyable (never executed) install/update
commands. The submission-curation CI validator lives in
``eegprep.extension_catalog_validation``.
"""

from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass, field
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

CATALOG_SCHEMA_VERSION = 1
CATALOG_KIND_MANAGER = "extension_manager"
CATALOG_KIND_CURATION = "extension_curation"
CATALOG_ENV_VAR = "EEGPREP_EXTENSION_CATALOG"
CATALOG_RESOURCE = "extension_catalog.json"
INSTALL_TRUST_WARNING = (
    "Installing Python packages executes third-party code. Review the package, maintainer, "
    "and source before running any install command."
)

_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".whl")


class CatalogSourceType(str, Enum):
    """Install/source categories accepted by the curated catalog."""

    PYPI = "pypi"
    GIT = "git"
    LOCAL = "local"
    PRIVATE = "private"


@dataclass(frozen=True)
class ExtensionCatalogEntry:
    """Curated metadata for an EEGPrep extension, without bundled code."""

    name: str
    display_name: str = ""
    version: str = ""
    package_name: str = ""
    description: str = ""
    maintainer: str = ""
    docs_url: str = ""
    source_type: CatalogSourceType = CatalogSourceType.PYPI
    source_url: str = ""
    repository_url: str = ""
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    eegprep_requires: str = ""

    @property
    def install_target(self) -> str:
        """Return the package-manager target for this catalog entry."""
        if self.source_type == CatalogSourceType.GIT:
            return _git_install_target(self.source_url or self.repository_url)
        if self.source_type == CatalogSourceType.LOCAL:
            return self.source_url
        return self.package_name

    @property
    def source_label(self) -> str:
        """Return user-facing source text for the catalog entry."""
        labels = {
            CatalogSourceType.PYPI: "PyPI package",
            CatalogSourceType.GIT: "Git repository",
            CatalogSourceType.LOCAL: "Local editable path",
            CatalogSourceType.PRIVATE: "Private package/index",
        }
        return labels[self.source_type]


@dataclass(frozen=True)
class ExtensionCatalog:
    """Loaded extension catalog plus source diagnostics."""

    entries: tuple[ExtensionCatalogEntry, ...] = field(default_factory=tuple)
    source: str = ""
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def available(self) -> bool:
        """Return whether the catalog was loaded without diagnostics."""
        return not self.errors

    def by_name(self) -> dict[str, ExtensionCatalogEntry]:
        """Return entries keyed by normalized extension name."""
        return {_catalog_normalize_name(entry.name): entry for entry in self.entries}

    def by_package(self) -> dict[str, ExtensionCatalogEntry]:
        """Return entries keyed by normalized Python package name."""
        return {_catalog_normalize_name(entry.package_name): entry for entry in self.entries if entry.package_name}


def load_extension_catalog(catalog_path: str | os.PathLike[str] | None = None) -> ExtensionCatalog:
    """Load the packaged or local metadata-only extension catalog.

    Args:
        catalog_path: Optional JSON catalog path. When omitted, the
            ``EEGPREP_EXTENSION_CATALOG`` environment variable is checked before
            the packaged ``resources/extension_catalog.json`` fallback.

    Returns:
        The parsed catalog. Invalid or unavailable catalogs return an empty
        catalog with ``errors`` populated; they do not trigger network access.
    """
    explicit_path = Path(catalog_path) if catalog_path is not None else None
    env_path = os.environ.get(CATALOG_ENV_VAR)
    if explicit_path is None and env_path:
        explicit_path = Path(env_path)
    if explicit_path is not None:
        return _load_catalog_path(explicit_path)
    return _load_packaged_catalog()


def parse_extension_catalog(data: Any, *, source: str = "inline") -> ExtensionCatalog:
    """Parse a catalog JSON object into validated metadata entries."""
    if not isinstance(data, dict):
        return ExtensionCatalog(source=source, errors=(f"{source}: catalog root must be a JSON object",))

    errors: list[str] = []
    catalog_kind = _text(data.get("catalog_kind") or CATALOG_KIND_MANAGER)
    if catalog_kind != CATALOG_KIND_MANAGER:
        errors.append(
            f"{source}: catalog_kind must be {CATALOG_KIND_MANAGER!r} for Extension Manager catalogs; "
            f"got {catalog_kind!r}"
        )
    schema_version = data.get("schema_version")
    if schema_version != CATALOG_SCHEMA_VERSION:
        errors.append(f"{source}: schema_version must be {CATALOG_SCHEMA_VERSION}; got {schema_version!r}")

    raw_entries = data.get("extensions", [])
    if not isinstance(raw_entries, list):
        return ExtensionCatalog(source=source, errors=(*errors, f"{source}: extensions must be a list"))

    entries: list[ExtensionCatalogEntry] = []
    seen_names: set[str] = set()
    for index, raw_entry in enumerate(raw_entries):
        entry, entry_errors = _parse_catalog_entry(raw_entry, source=f"{source}:extensions[{index}]")
        errors.extend(entry_errors)
        if entry is None:
            continue
        normalized_name = _catalog_normalize_name(entry.name)
        if normalized_name in seen_names:
            errors.append(f"{source}: duplicate catalog extension name {entry.name!r}")
            continue
        seen_names.add(normalized_name)
        entries.append(entry)

    return ExtensionCatalog(entries=tuple(entries), source=source, errors=tuple(errors))


def build_safe_install_commands(entry: ExtensionCatalogEntry) -> dict[str, str]:
    """Return copyable install commands for a catalog entry without executing them."""
    target = entry.install_target
    if not target:
        return {}
    quoted_target = shlex.quote(target)
    if entry.source_type == CatalogSourceType.LOCAL:
        return {
            "uv": f"uv add --editable {quoted_target}",
            "pip": f"pip install -e {quoted_target}",
        }
    if entry.source_type == CatalogSourceType.GIT:
        return {
            "uv": f"uv add {quoted_target}",
            "pip": f"pip install {quoted_target}",
        }
    return {
        "uv": f"uv add {quoted_target}",
        "pip": f"pip install {quoted_target}",
    }


def build_safe_update_commands(entry: ExtensionCatalogEntry) -> dict[str, str]:
    """Return copyable update commands for a catalog entry without executing them."""
    if entry.source_type in {CatalogSourceType.GIT, CatalogSourceType.LOCAL}:
        return build_safe_install_commands(entry)
    if not entry.package_name:
        return {}
    quoted_package = shlex.quote(entry.package_name)
    return {
        "uv": f"uv add --upgrade-package {quoted_package} {quoted_package}",
        "pip": f"pip install --upgrade {quoted_package}",
    }


def _load_catalog_path(path: Path) -> ExtensionCatalog:
    source = str(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return ExtensionCatalog(source=source, errors=(f"{source}: catalog unavailable: {exc}",))
    return _loads_catalog(text, source=source)


def _load_packaged_catalog() -> ExtensionCatalog:
    source = f"eegprep.resources/{CATALOG_RESOURCE}"
    try:
        path = resources.files("eegprep.resources").joinpath(CATALOG_RESOURCE)
        if not path.is_file():
            return ExtensionCatalog(source=source)
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return ExtensionCatalog(source=source, errors=(f"{source}: catalog unavailable: {exc}",))
    return _loads_catalog(text, source=source)


def _loads_catalog(text: str, *, source: str) -> ExtensionCatalog:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return ExtensionCatalog(source=source, errors=(f"{source}: invalid JSON: {exc}",))
    return parse_extension_catalog(data, source=source)


def _parse_catalog_entry(raw_entry: Any, *, source: str) -> tuple[ExtensionCatalogEntry | None, tuple[str, ...]]:
    if not isinstance(raw_entry, dict):
        return None, (f"{source}: entry must be a JSON object",)

    errors: list[str] = []
    name = _text(raw_entry.get("name"))
    if not name:
        errors.append(f"{source}: name is required")
        if any(field in raw_entry for field in ("id", "entry_point", "extension_name")):
            errors.append(
                f"{source}: this looks like {CATALOG_KIND_CURATION!r} metadata; "
                "load Extension Manager catalogs with name/package_name/source fields, "
                "and validate curation metadata with eegprep-validate-extension-catalog"
            )

    source_data = raw_entry.get("source", {})
    if not isinstance(source_data, dict):
        errors.append(f"{source}: source must be an object")
        source_data = {}

    source_type_text = _text(source_data.get("type") or raw_entry.get("source_type") or CatalogSourceType.PYPI.value)
    try:
        source_type = CatalogSourceType(source_type_text)
    except ValueError:
        source_type = CatalogSourceType.PYPI
        errors.append(f"{source}: source.type {source_type_text!r} is not supported")

    source_url = _text(source_data.get("url") or raw_entry.get("source_url"))
    repository_url = _text(raw_entry.get("repository_url") or source_data.get("repository_url"))
    docs_url = _text(raw_entry.get("docs_url"))
    package_name = _text(raw_entry.get("package_name"))

    if source_type in {CatalogSourceType.PYPI, CatalogSourceType.PRIVATE} and not package_name:
        errors.append(f"{source}: package_name is required for {source_type.value} catalog entries")
    if source_type in {CatalogSourceType.GIT, CatalogSourceType.LOCAL} and not (source_url or repository_url):
        errors.append(f"{source}: source.url is required for {source_type.value} catalog entries")
    for field_name, url in (("source.url", source_url), ("repository_url", repository_url), ("docs_url", docs_url)):
        if _looks_like_archive(url):
            errors.append(f"{source}: {field_name} must point to metadata, docs, or a repository, not an archive")
        if field_name == "source.url" and source_type == CatalogSourceType.LOCAL:
            continue
        if url and not _is_web_url(url):
            errors.append(f"{source}: {field_name} must be an https:// or http:// URL")

    if errors:
        return None, tuple(errors)

    return (
        ExtensionCatalogEntry(
            name=name,
            display_name=_text(raw_entry.get("display_name")),
            version=_text(raw_entry.get("version")),
            package_name=package_name,
            description=_text(raw_entry.get("description")),
            maintainer=_text(raw_entry.get("maintainer")),
            docs_url=docs_url,
            source_type=source_type,
            source_url=source_url,
            repository_url=repository_url,
            capabilities=_text_tuple(raw_entry.get("capabilities")),
            eegprep_requires=_text(raw_entry.get("eegprep_requires")),
        ),
        (),
    )


def _git_install_target(url: str) -> str:
    if not url:
        return ""
    if url.startswith("git+"):
        return url
    return f"git+{url}"


def _looks_like_archive(url: str) -> bool:
    lowered = url.lower().split("?", 1)[0].split("#", 1)[0]
    return bool(lowered) and lowered.endswith(_ARCHIVE_SUFFIXES)


def _is_web_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"https", "http"} and bool(parsed.netloc)


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _text_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (_text(value),) if _text(value) else ()
    if not isinstance(value, (list, tuple)):
        return (_text(value),) if _text(value) else ()
    return tuple(text for text in (_text(item) for item in value) if text)


def _catalog_normalize_name(name: str) -> str:
    return str(name).strip().lower()


__all__ = [
    "CATALOG_KIND_CURATION",
    "CATALOG_KIND_MANAGER",
    "CATALOG_ENV_VAR",
    "CATALOG_RESOURCE",
    "CATALOG_SCHEMA_VERSION",
    "INSTALL_TRUST_WARNING",
    "CatalogSourceType",
    "ExtensionCatalog",
    "ExtensionCatalogEntry",
    "build_safe_install_commands",
    "build_safe_update_commands",
    "load_extension_catalog",
    "parse_extension_catalog",
]
