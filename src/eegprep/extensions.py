"""External extension SDK and discovery registry for EEGPrep."""

from __future__ import annotations

import importlib
import importlib.resources as resources
import logging
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from importlib import metadata
from typing import Any, Callable

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from eegprep.plugins.EEG_BIDS.menu import (
    eeg_bids_export_items,
    eeg_bids_import_items,
    eeg_bids_tools_menu,
)
from eegprep.plugins.dipfit.menu import dipfit_menu
from eegprep.plugins.firfilt.menu import firfilt_filter_items

EXTENSION_API_VERSION = "1"
EXTENSION_ENTRY_POINT_GROUP = "eegprep.extensions"
EXTENSION_NAMING_PREFIX = "eegprep-ext-"
EXTENSION_CURATION_POLICY_URL = "user_guide/extension_curation.html"
EXTENSION_TRUST_MESSAGE = (
    "EEGPrep extensions are Python packages. Installing a package can run package build hooks, "
    "and importing or activating an extension executes third-party Python code."
)
EXTENSION_COMPATIBILITY_POLICY = (
    "Extension API versions are major-version compatible. EEGPrep supports extensions whose "
    "api_version has the same major version as EXTENSION_API_VERSION. Extensions should declare "
    "eegprep_requires and required Python dependencies; unsupported EEGPrep versions or missing "
    "required dependencies mark the extension incompatible or missing_dependency before activation."
)

logger = logging.getLogger(__name__)


class ExtensionStatus(str, Enum):
    """Registry status for an extension record."""

    BUNDLED = "bundled"
    INSTALLED = "installed"
    CURATED = "curated"
    DISABLED = "disabled"
    INCOMPATIBLE = "incompatible"
    FAILED_IMPORT = "failed_import"
    INVALID_SPEC = "invalid_spec"
    MISSING_DEPENDENCY = "missing_dependency"
    UNKNOWN = "unknown"


EXTENSION_ACTIVE_STATUSES = frozenset(
    {
        ExtensionStatus.BUNDLED.value,
        ExtensionStatus.INSTALLED.value,
        ExtensionStatus.CURATED.value,
        "ok",
    }
)
EXTENSION_INSTALLED_STATUSES = frozenset(
    {
        ExtensionStatus.BUNDLED.value,
        ExtensionStatus.INSTALLED.value,
        ExtensionStatus.DISABLED.value,
        ExtensionStatus.INCOMPATIBLE.value,
        ExtensionStatus.FAILED_IMPORT.value,
        ExtensionStatus.INVALID_SPEC.value,
        ExtensionStatus.MISSING_DEPENDENCY.value,
        ExtensionStatus.UNKNOWN.value,
        "ok",
    }
)


class ExtensionSourceType(str, Enum):
    """Source category declared by an extension."""

    BUNDLED = "bundled"
    INSTALLED = "installed"
    CURATED = "curated"
    UNKNOWN = "unknown"


class ExtensionLoadError(RuntimeError):
    """Raised when a lazily referenced extension object cannot be loaded."""


@dataclass(frozen=True)
class LazyImport:
    """Reference to an object that should be imported only when used."""

    module: str
    attr: str

    def load(self) -> Any:
        """Import and return the referenced object."""
        try:
            module = importlib.import_module(self.module)
            return getattr(module, self.attr)
        except Exception as exc:
            raise ExtensionLoadError(f"Could not load extension target {self.module}:{self.attr}") from exc


@dataclass(frozen=True)
class ExtensionResource:
    """Packaged resource declared by an extension."""

    package: str
    path: str

    def exists(self) -> bool:
        """Return whether the packaged resource exists."""
        try:
            return resources.files(self.package).joinpath(self.path).is_file()
        except Exception:
            return False

    def read_text(self, encoding: str = "utf-8") -> str:
        """Read this resource as text."""
        try:
            return resources.files(self.package).joinpath(self.path).read_text(encoding=encoding)
        except Exception as exc:
            raise FileNotFoundError(f"Extension resource {self.package}:{self.path} is not available") from exc

    def read_bytes(self) -> bytes:
        """Read this resource as bytes."""
        try:
            return resources.files(self.package).joinpath(self.path).read_bytes()
        except Exception as exc:
            raise FileNotFoundError(f"Extension resource {self.package}:{self.path} is not available") from exc


@dataclass(frozen=True)
class ExtensionDependency:
    """Python distribution required by an extension."""

    package: str
    version_spec: str = ""
    optional: bool = False


@dataclass(frozen=True)
class ExtensionAction:
    """Declarative action contributed by an extension."""

    name: str
    target: LazyImport
    display_name: str = ""
    description: str = ""
    capabilities: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "capabilities", _as_tuple(self.capabilities))

    def load(self) -> Any:
        """Load the callable target for this action."""
        return self.target.load()


@dataclass(frozen=True)
class ExtensionPopFunction:
    """Declarative ``pop_*`` function contributed by an extension."""

    name: str
    target: LazyImport
    description: str = ""

    def load(self) -> Any:
        """Load the callable target for this ``pop_*`` function."""
        return self.target.load()


@dataclass(frozen=True)
class ExtensionMenu:
    """Declarative menu contribution for an extension action."""

    path: tuple[str, ...] = field(default_factory=tuple)
    action: str = ""
    label: str = ""
    tag: str = ""
    userdata: str = ""
    separator: bool = False
    enabled: bool = True
    checked: bool = False
    visibility: str = "always"
    insert_after: str = ""
    insert_before: str = ""
    children: tuple["ExtensionMenu", ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _as_tuple(self.path))
        object.__setattr__(self, "children", _as_tuple(self.children))


@dataclass(frozen=True)
class ExtensionSpec:
    """Declarative metadata returned by an ``eegprep.extensions`` entry point."""

    name: str
    display_name: str = ""
    version: str = ""
    api_version: str = EXTENSION_API_VERSION
    package_name: str = ""
    entry_point_name: str = ""
    source_type: ExtensionSourceType | str = ExtensionSourceType.INSTALLED
    description: str = ""
    docs_url: str = ""
    maintainer: str = ""
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    dependencies: tuple[ExtensionDependency, ...] = field(default_factory=tuple)
    menus: tuple[ExtensionMenu, ...] = field(default_factory=tuple)
    actions: tuple[ExtensionAction, ...] = field(default_factory=tuple)
    pop_functions: tuple[ExtensionPopFunction, ...] = field(default_factory=tuple)
    help_resources: tuple[ExtensionResource, ...] = field(default_factory=tuple)
    package_data_resources: tuple[ExtensionResource, ...] = field(default_factory=tuple)
    eegprep_requires: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "capabilities", _as_tuple(self.capabilities))
        object.__setattr__(self, "dependencies", _as_tuple(self.dependencies))
        object.__setattr__(self, "menus", _as_tuple(self.menus))
        object.__setattr__(self, "actions", _as_tuple(self.actions))
        object.__setattr__(self, "pop_functions", _as_tuple(self.pop_functions))
        object.__setattr__(self, "help_resources", _as_tuple(self.help_resources))
        object.__setattr__(self, "package_data_resources", _as_tuple(self.package_data_resources))


@dataclass(frozen=True)
class ExtensionValidationResult:
    """Validation errors grouped by registry status category."""

    invalid_spec: tuple[str, ...] = field(default_factory=tuple)
    incompatible: tuple[str, ...] = field(default_factory=tuple)
    missing_dependency: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """Return whether validation found no blocking errors."""
        return not self.invalid_spec and not self.incompatible and not self.missing_dependency

    @property
    def errors(self) -> tuple[str, ...]:
        """Return all validation errors in status-priority order."""
        return (*self.invalid_spec, *self.incompatible, *self.missing_dependency)


@dataclass(frozen=True)
class ExtensionRecord:
    """A discovered extension plus status and validation details."""

    name: str
    status: ExtensionStatus
    spec: ExtensionSpec | None = None
    source_type: ExtensionSourceType = ExtensionSourceType.UNKNOWN
    package_name: str = ""
    entry_point_name: str = ""
    enabled: bool = True
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_active(self) -> bool:
        """Return whether this record can contribute runtime behavior."""
        return self.enabled and self.spec is not None and extension_status_is_active(self.status)


EntryPointsProvider = Callable[..., Any]
VersionProvider = Callable[[str], str]


class ExtensionRegistry:
    """Discover, validate, and report EEGPrep extensions."""

    def __init__(
        self,
        *,
        disabled_extensions: set[str] | list[str] | tuple[str, ...] | None = None,
        include_bundled: bool = True,
        include_entry_points: bool = True,
        entry_point_group: str = EXTENSION_ENTRY_POINT_GROUP,
        entry_points_provider: EntryPointsProvider = metadata.entry_points,
        current_version: str | None = None,
        version_provider: VersionProvider = metadata.version,
    ) -> None:
        self.disabled_extensions = {_normalize_name(name) for name in disabled_extensions or ()}
        self.include_bundled = include_bundled
        self.include_entry_points = include_entry_points
        self.entry_point_group = entry_point_group
        self._entry_points_provider = entry_points_provider
        self.current_version = current_version or _current_eegprep_version()
        self._version_provider = version_provider
        self._records: tuple[ExtensionRecord, ...] = ()

    @property
    def records(self) -> tuple[ExtensionRecord, ...]:
        """Return records from the latest discovery pass."""
        return self._records

    def discover(self, *, include_plugins: bool = True) -> tuple[ExtensionRecord, ...]:
        """Discover extensions and return deterministic registry records."""
        if not include_plugins:
            self._records = ()
            return self._records

        records: list[ExtensionRecord] = []
        if self.include_bundled:
            records.extend(self._bundled_records())
        if self.include_entry_points:
            records.extend(sorted(self._entry_point_records(), key=_record_sort_key))

        self._records = tuple(self._mark_duplicate_contributions(records))
        self._log_problem_records(self._records)
        return self._records

    def get(self, name: str) -> ExtensionRecord | None:
        """Return the latest record for ``name`` if present."""
        normalized = _normalize_name(name)
        for record in self._records:
            if _normalize_name(record.name) == normalized:
                return record
        return None

    def _bundled_records(self) -> list[ExtensionRecord]:
        specs = (
            ExtensionSpec(
                name="firfilt",
                display_name="firfilt",
                version="bundled",
                package_name="eegprep.plugins.firfilt",
                source_type=ExtensionSourceType.BUNDLED,
                description="Windowed-sinc, Parks-McClellan, moving-average, and new default FIR filtering.",
                capabilities=("filter", "preprocessing"),
                menus=tuple(
                    _extension_menu_from_spec(
                        item,
                        path=("tools", "filter"),
                        insert_before="pop_eegfilt",
                    )
                    for item in firfilt_filter_items()
                ),
                pop_functions=(
                    ExtensionPopFunction(
                        name="pop_eegfiltnew",
                        target=LazyImport("eegprep.plugins.firfilt.pop_eegfiltnew", "pop_eegfiltnew"),
                    ),
                    ExtensionPopFunction(
                        name="pop_firws",
                        target=LazyImport("eegprep.plugins.firfilt.pop_firws", "pop_firws"),
                    ),
                    ExtensionPopFunction(
                        name="pop_firpm",
                        target=LazyImport("eegprep.plugins.firfilt.pop_firpm", "pop_firpm"),
                    ),
                    ExtensionPopFunction(
                        name="pop_firma",
                        target=LazyImport("eegprep.plugins.firfilt.pop_firma", "pop_firma"),
                    ),
                ),
            ),
            ExtensionSpec(
                name="dipfit",
                display_name="DIPFIT",
                version="bundled",
                package_name="eegprep.plugins.dipfit",
                source_type=ExtensionSourceType.BUNDLED,
                description="Source-localization menu surfaces with EEGPrep-native spherical DIPFIT workflows.",
                capabilities=("source", "localization"),
                menus=(
                    _extension_menu_from_spec(
                        dipfit_menu(),
                        path=("tools",),
                        insert_after="pop_rmbase",
                    ),
                ),
                pop_functions=(
                    ExtensionPopFunction(
                        name="pop_dipfit_settings",
                        target=LazyImport("eegprep.plugins.dipfit.pop_dipfit_settings", "pop_dipfit_settings"),
                    ),
                    ExtensionPopFunction(
                        name="pop_dipfit_headmodel",
                        target=LazyImport("eegprep.plugins.dipfit.pop_dipfit_headmodel", "pop_dipfit_headmodel"),
                    ),
                    ExtensionPopFunction(
                        name="pop_dipfit_gridsearch",
                        target=LazyImport(
                            "eegprep.plugins.dipfit.pop_dipfit_gridsearch",
                            "pop_dipfit_gridsearch",
                        ),
                    ),
                    ExtensionPopFunction(
                        name="pop_dipfit_batch",
                        target=LazyImport("eegprep.plugins.dipfit.pop_dipfit_batch", "pop_dipfit_batch"),
                    ),
                    ExtensionPopFunction(
                        name="pop_dipfit_nonlinear",
                        target=LazyImport("eegprep.plugins.dipfit.pop_dipfit_nonlinear", "pop_dipfit_nonlinear"),
                    ),
                    ExtensionPopFunction(
                        name="pop_dipfit_manual",
                        target=LazyImport("eegprep.plugins.dipfit.pop_dipfit_manual", "pop_dipfit_manual"),
                    ),
                    ExtensionPopFunction(
                        name="pop_dipplot",
                        target=LazyImport("eegprep.plugins.dipfit.pop_dipplot", "pop_dipplot"),
                    ),
                    ExtensionPopFunction(
                        name="pop_multifit",
                        target=LazyImport("eegprep.plugins.dipfit.pop_multifit", "pop_multifit"),
                    ),
                    ExtensionPopFunction(
                        name="pop_leadfield",
                        target=LazyImport("eegprep.plugins.dipfit.pop_leadfield", "pop_leadfield"),
                    ),
                    ExtensionPopFunction(
                        name="pop_dipfit_loreta",
                        target=LazyImport("eegprep.plugins.dipfit.pop_dipfit_loreta", "pop_dipfit_loreta"),
                    ),
                ),
            ),
            ExtensionSpec(
                name="EEG_BIDS",
                display_name="EEG-BIDS",
                version="bundled",
                package_name="eegprep.plugins.EEG_BIDS",
                source_type=ExtensionSourceType.BUNDLED,
                description="BIDS import, export, validation, and metadata helpers for EEG datasets.",
                capabilities=("import", "export", "bids", "study"),
                menus=(
                    *(
                        _extension_menu_from_spec(
                            item,
                            path=("File", "Import data", "import data"),
                        )
                        for item in eeg_bids_import_items()
                    ),
                    *(
                        _extension_menu_from_spec(
                            item,
                            path=("File", "export"),
                        )
                        for item in eeg_bids_export_items()
                    ),
                    _extension_menu_from_spec(
                        eeg_bids_tools_menu(),
                        path=("File",),
                        insert_after="export",
                    ),
                ),
                pop_functions=(
                    ExtensionPopFunction(
                        name="pop_importbids",
                        target=LazyImport("eegprep.plugins.EEG_BIDS.pop_importbids", "pop_importbids"),
                    ),
                    ExtensionPopFunction(
                        name="pop_exportbids",
                        target=LazyImport("eegprep.plugins.EEG_BIDS.pop_exportbids", "pop_exportbids"),
                    ),
                ),
            ),
        )
        return [self._record_from_spec(spec) for spec in specs]

    def _entry_point_records(self) -> list[ExtensionRecord]:
        records = []
        for entry_point in select_extension_entry_points(self._entry_points_provider, self.entry_point_group):
            records.append(self._record_from_entry_point(entry_point))
        return records

    def _record_from_entry_point(self, entry_point: Any) -> ExtensionRecord:
        entry_point_name = str(getattr(entry_point, "name", "") or "<unknown>")
        package_name = extension_entry_point_package_name(entry_point)
        try:
            loaded = entry_point.load()
        except Exception as exc:
            message = f"Entry point {entry_point_name!r} failed to import: {exc}"
            return ExtensionRecord(
                name=entry_point_name,
                status=ExtensionStatus.FAILED_IMPORT,
                source_type=ExtensionSourceType.UNKNOWN,
                package_name=package_name,
                entry_point_name=entry_point_name,
                enabled=True,
                errors=(message,),
            )
        try:
            candidate = loaded() if callable(loaded) and not isinstance(loaded, ExtensionSpec) else loaded
        except Exception as exc:
            message = f"Entry point {entry_point_name!r} failed during registration: {exc}"
            return ExtensionRecord(
                name=entry_point_name,
                status=ExtensionStatus.FAILED_IMPORT,
                source_type=ExtensionSourceType.UNKNOWN,
                package_name=package_name,
                entry_point_name=entry_point_name,
                enabled=True,
                errors=(message,),
            )

        if not isinstance(candidate, ExtensionSpec):
            return ExtensionRecord(
                name=entry_point_name,
                status=ExtensionStatus.INVALID_SPEC,
                source_type=ExtensionSourceType.UNKNOWN,
                package_name=package_name,
                entry_point_name=entry_point_name,
                enabled=True,
                errors=(f"Entry point {entry_point_name!r} returned {type(candidate).__name__}, not ExtensionSpec",),
            )

        spec = replace(
            candidate,
            package_name=package_name or candidate.package_name,
            entry_point_name=candidate.entry_point_name or entry_point_name,
        )
        return self._record_from_spec(spec)

    def _record_from_spec(self, spec: ExtensionSpec) -> ExtensionRecord:
        source_type = _coerce_source_type(spec.source_type)
        enabled = _normalize_name(spec.name) not in self.disabled_extensions
        validation = validate_extension_spec(
            spec,
            current_version=self.current_version,
            version_provider=self._version_provider,
            check_compatibility=enabled,
            check_dependencies=enabled,
            check_resources=enabled,
        )
        status = _status_for_spec(source_type, enabled, validation)
        return ExtensionRecord(
            name=spec.name,
            status=status,
            spec=spec,
            source_type=source_type,
            package_name=spec.package_name,
            entry_point_name=spec.entry_point_name,
            enabled=enabled,
            errors=validation.errors,
        )

    def _mark_duplicate_contributions(self, records: list[ExtensionRecord]) -> list[ExtensionRecord]:
        extension_names: set[str] = set()
        action_names: dict[str, str] = {}
        pop_names: dict[str, str] = {}
        menu_names: dict[tuple[tuple[str, ...], str], str] = {}
        final_records: list[ExtensionRecord] = []

        for record in records:
            if not record.is_active:
                final_records.append(record)
                continue

            errors: list[str] = []
            extension_name = _normalize_name(record.name)
            if extension_name in extension_names:
                errors.append(f"Duplicate extension name {record.name!r}")

            spec = record.spec
            if spec is not None:
                for action in spec.actions:
                    action_name = _normalize_name(action.name)
                    owner = action_names.get(action_name)
                    if owner is not None:
                        errors.append(f"Duplicate action name {action.name!r} already provided by {owner!r}")
                for pop_function in spec.pop_functions:
                    pop_name = _normalize_name(pop_function.name)
                    owner = pop_names.get(pop_name)
                    if owner is not None:
                        errors.append(f"Duplicate pop function {pop_function.name!r} already provided by {owner!r}")
                for menu_key, menu_label in _extension_menu_keys(spec.menus):
                    owner = menu_names.get(menu_key)
                    if owner is not None:
                        errors.append(f"Duplicate menu {menu_label!r} already provided by {owner!r}")

            if errors:
                final_records.append(_invalid_record(record, tuple(errors)))
                continue

            extension_names.add(extension_name)
            if spec is not None:
                for action in spec.actions:
                    action_names[_normalize_name(action.name)] = record.name
                for pop_function in spec.pop_functions:
                    pop_names[_normalize_name(pop_function.name)] = record.name
                for menu_key, _menu_label in _extension_menu_keys(spec.menus):
                    menu_names[menu_key] = record.name
            final_records.append(record)
        return final_records

    def _log_problem_records(self, records: tuple[ExtensionRecord, ...]) -> None:
        for record in records:
            if not record.errors:
                continue
            logger.warning(
                "EEGPrep extension %s has status %s: %s",
                record.name,
                record.status.value,
                "; ".join(record.errors),
            )


def discover_extensions(
    *,
    disabled_extensions: set[str] | list[str] | tuple[str, ...] | None = None,
    include_plugins: bool = True,
    include_bundled: bool = True,
    include_entry_points: bool = True,
) -> tuple[ExtensionRecord, ...]:
    """Discover EEGPrep extensions with the default registry settings."""
    registry = ExtensionRegistry(
        disabled_extensions=disabled_extensions,
        include_bundled=include_bundled,
        include_entry_points=include_entry_points,
    )
    return registry.discover(include_plugins=include_plugins)


def validate_extension_spec(
    spec: ExtensionSpec,
    *,
    current_version: str | None = None,
    version_provider: VersionProvider = metadata.version,
    check_compatibility: bool = True,
    check_dependencies: bool = True,
    check_resources: bool = True,
) -> ExtensionValidationResult:
    """Validate an extension spec without importing its lazy action targets."""
    invalid: list[str] = []
    incompatible: list[str] = []
    missing_dependency: list[str] = []
    current_version = current_version or _current_eegprep_version()

    if not isinstance(spec, ExtensionSpec):
        return ExtensionValidationResult(invalid_spec=("Extension specs must be ExtensionSpec instances",))

    if not spec.name or not isinstance(spec.name, str):
        invalid.append("Extension name is required")
    elif not re.match(r"^[A-Za-z][A-Za-z0-9_.-]*$", spec.name):
        invalid.append(
            f"Extension name {spec.name!r} must start with a letter and contain only letters, numbers, ., _, -"
        )

    if not spec.api_version or not isinstance(spec.api_version, str):
        invalid.append("Extension API version is required")
    elif check_compatibility and not _api_version_supported(spec.api_version):
        incompatible.append(f"Extension API version {spec.api_version!r} is not supported by EEGPrep")

    if spec.source_type not in {source.value for source in ExtensionSourceType} and not isinstance(
        spec.source_type, ExtensionSourceType
    ):
        invalid.append(f"Extension source type {spec.source_type!r} is not recognized")

    _validate_actions(spec.actions, invalid)
    _validate_pop_functions(spec.pop_functions, invalid)
    _validate_menus(spec.menus, invalid)

    if check_compatibility and spec.eegprep_requires:
        if not extension_version_spec_is_valid(spec.eegprep_requires):
            invalid.append(f"Extension EEGPrep requirement {spec.eegprep_requires!r} is not a valid version specifier")
        elif not _version_satisfies(current_version, spec.eegprep_requires):
            incompatible.append(
                f"Extension requires EEGPrep {spec.eegprep_requires}; current version is {current_version}"
            )

    if check_dependencies:
        dependency_invalid, dependency_missing = _dependency_errors(spec.dependencies, version_provider)
        invalid.extend(dependency_invalid)
        missing_dependency.extend(dependency_missing)

    if check_resources:
        for resource in spec.help_resources:
            if not isinstance(resource, ExtensionResource):
                invalid.append(f"Extension resource {resource!r} is not an ExtensionResource")
                continue
            if not resource.package or not resource.path:
                invalid.append("Extension resources must include package and path")
                continue
            if not resource.path.endswith(".md"):
                invalid.append(f"Extension help resource {resource.package}:{resource.path} must be a Markdown file")
                continue
            if not resource.exists():
                invalid.append(f"Extension resource {resource.package}:{resource.path} is missing")
        for resource in spec.package_data_resources:
            if not isinstance(resource, ExtensionResource):
                invalid.append(f"Extension resource {resource!r} is not an ExtensionResource")
                continue
            if not resource.package or not resource.path:
                invalid.append("Extension resources must include package and path")
                continue
            if not resource.exists():
                invalid.append(f"Extension resource {resource.package}:{resource.path} is missing")

    return ExtensionValidationResult(
        invalid_spec=tuple(invalid),
        incompatible=tuple(incompatible),
        missing_dependency=tuple(missing_dependency),
    )


def check_extension_compatibility(
    spec: ExtensionSpec,
    *,
    current_version: str | None = None,
    version_provider: VersionProvider = metadata.version,
    check_dependencies: bool = True,
) -> ExtensionValidationResult:
    """Check extension API, EEGPrep version, and dependency compatibility.

    This is the compatibility-policy entry point for catalog validation,
    author tests, and manager status messaging. It does not verify packaged
    resources, so callers can run it before installing optional help or data
    files into a test fixture.
    """
    return validate_extension_spec(
        spec,
        current_version=current_version,
        version_provider=version_provider,
        check_compatibility=True,
        check_dependencies=check_dependencies,
        check_resources=False,
    )


def extension_version_satisfies(version: str, specifier: str) -> bool:
    """Return whether ``version`` satisfies a simple PEP 440-style specifier."""
    return _version_satisfies(version, specifier)


def extension_version_spec_is_valid(specifier: str) -> bool:
    """Return whether ``specifier`` is a valid EEGPrep extension version spec."""
    try:
        _normalized_version_specifier(specifier)
    except InvalidSpecifier:
        return False
    return True


def compare_extension_versions(left: str, right: str) -> int:
    """Compare two PEP 440 versions, returning ``-1``, ``0``, or ``1``."""
    try:
        left_version = Version(str(left))
        right_version = Version(str(right))
    except InvalidVersion:
        return 0
    if left_version < right_version:
        return -1
    if left_version > right_version:
        return 1
    return 0


def _validate_actions(actions: tuple[Any, ...], invalid: list[str]) -> None:
    names: set[str] = set()
    for action in actions:
        if not isinstance(action, ExtensionAction):
            invalid.append(f"Extension action {action!r} is not an ExtensionAction")
            continue
        if not action.name:
            invalid.append("Extension actions must include a name")
            continue
        normalized = _normalize_name(action.name)
        if normalized in names:
            invalid.append(f"Duplicate action name {action.name!r} within extension")
        names.add(normalized)
        if not isinstance(action.target, LazyImport):
            invalid.append(f"Extension action {action.name!r} target must be a LazyImport")


def _validate_pop_functions(pop_functions: tuple[Any, ...], invalid: list[str]) -> None:
    names: set[str] = set()
    for pop_function in pop_functions:
        if not isinstance(pop_function, ExtensionPopFunction):
            invalid.append(f"Extension pop function {pop_function!r} is not an ExtensionPopFunction")
            continue
        if not pop_function.name:
            invalid.append("Extension pop functions must include a name")
            continue
        if not pop_function.name.startswith("pop_"):
            invalid.append(f"Extension pop function {pop_function.name!r} must start with 'pop_'")
        normalized = _normalize_name(pop_function.name)
        if normalized in names:
            invalid.append(f"Duplicate pop function {pop_function.name!r} within extension")
        names.add(normalized)
        if not isinstance(pop_function.target, LazyImport):
            invalid.append(f"Extension pop function {pop_function.name!r} target must be a LazyImport")


def _validate_menus(menus: tuple[Any, ...], invalid: list[str]) -> None:
    names: set[tuple[tuple[str, ...], str]] = set()
    for menu in menus:
        _validate_menu(menu, invalid, names, root=True)


def _validate_menu(
    menu: Any,
    invalid: list[str],
    names: set[tuple[tuple[str, ...], str]],
    *,
    root: bool,
) -> None:
    if not isinstance(menu, ExtensionMenu):
        invalid.append(f"Extension menu {menu!r} is not an ExtensionMenu")
        return
    if root and not menu.path:
        invalid.append("Extension menus must include a path")
    if menu.insert_before and menu.insert_after:
        invalid.append("Extension menus cannot set both insert_before and insert_after")
    if not menu.action and not menu.children:
        invalid.append("Extension menus must reference an action or include children")
    label = menu.label or menu.action
    if label:
        key = (_normalize_menu_path(menu.path), _normalize_name(label))
        if root and key in names:
            invalid.append(f"Duplicate menu {label!r} within extension")
        names.add(key)
    for child in menu.children:
        _validate_menu(child, invalid, names, root=False)


def _dependency_errors(dependencies: tuple[Any, ...], version_provider: VersionProvider) -> tuple[list[str], list[str]]:
    invalid: list[str] = []
    missing: list[str] = []
    for dependency in dependencies:
        if not isinstance(dependency, ExtensionDependency):
            invalid.append(f"Extension dependency {dependency!r} is not an ExtensionDependency")
            continue
        if not dependency.package:
            invalid.append("Extension dependencies must include a package name")
            continue
        try:
            installed_version = version_provider(dependency.package)
        except metadata.PackageNotFoundError:
            if not dependency.optional:
                missing.append(f"Required dependency {dependency.package!r} is not installed")
            continue
        if dependency.version_spec and not _version_satisfies(installed_version, dependency.version_spec):
            missing.append(
                f"Dependency {dependency.package!r} requires {dependency.version_spec}; "
                f"installed version is {installed_version}"
            )
    return invalid, missing


def select_extension_entry_points(provider: EntryPointsProvider, group: str) -> tuple[Any, ...]:
    """Return entry points from providers with modern or legacy selection APIs."""
    try:
        selected = provider(group=group)
    except TypeError:
        entry_points = provider()
        if hasattr(entry_points, "select"):
            selected = entry_points.select(group=group)
        else:
            selected = [entry_point for entry_point in entry_points if getattr(entry_point, "group", None) == group]
    return tuple(selected or ())


def extension_status_is_active(status: ExtensionStatus | str) -> bool:
    """Return whether an extension status can contribute runtime behavior."""
    value = status.value if isinstance(status, ExtensionStatus) else str(status)
    return value in EXTENSION_ACTIVE_STATUSES


def extension_status_is_installed(status: ExtensionStatus | str) -> bool:
    """Return whether an extension status represents an installed package/port."""
    value = status.value if isinstance(status, ExtensionStatus) else str(status)
    return value in EXTENSION_INSTALLED_STATUSES


def _status_for_spec(
    source_type: ExtensionSourceType,
    enabled: bool,
    validation: ExtensionValidationResult,
) -> ExtensionStatus:
    if not enabled:
        return ExtensionStatus.DISABLED
    if validation.invalid_spec:
        return ExtensionStatus.INVALID_SPEC
    if validation.incompatible:
        return ExtensionStatus.INCOMPATIBLE
    if validation.missing_dependency:
        return ExtensionStatus.MISSING_DEPENDENCY
    if source_type == ExtensionSourceType.BUNDLED:
        return ExtensionStatus.BUNDLED
    if source_type == ExtensionSourceType.CURATED:
        return ExtensionStatus.CURATED
    if source_type == ExtensionSourceType.INSTALLED:
        return ExtensionStatus.INSTALLED
    return ExtensionStatus.UNKNOWN


def _invalid_record(record: ExtensionRecord, errors: tuple[str, ...]) -> ExtensionRecord:
    return replace(
        record,
        status=ExtensionStatus.INVALID_SPEC,
        enabled=False,
        errors=(*record.errors, *errors),
    )


def _record_sort_key(record: ExtensionRecord) -> tuple[int, str, str, str]:
    return (
        _source_rank(record.source_type),
        _normalize_name(record.name),
        _normalize_name(record.package_name),
        _normalize_name(record.entry_point_name),
    )


def _source_rank(source_type: ExtensionSourceType) -> int:
    ranks = {
        ExtensionSourceType.BUNDLED: 0,
        ExtensionSourceType.CURATED: 1,
        ExtensionSourceType.INSTALLED: 2,
        ExtensionSourceType.UNKNOWN: 3,
    }
    return ranks[source_type]


def _coerce_source_type(source_type: ExtensionSourceType | str) -> ExtensionSourceType:
    if isinstance(source_type, ExtensionSourceType):
        return source_type
    try:
        return ExtensionSourceType(source_type)
    except ValueError:
        return ExtensionSourceType.UNKNOWN


def extension_entry_point_package_name(entry_point: Any) -> str:
    """Return the distribution name associated with an extension entry point."""
    dist = getattr(entry_point, "dist", None)
    dist_metadata = getattr(dist, "metadata", None)
    if dist_metadata is None:
        return ""
    try:
        return str(dist_metadata.get("Name") or "")
    except AttributeError:
        return ""


def _current_eegprep_version() -> str:
    from eegprep import __version__

    return __version__


def _api_version_supported(api_version: str) -> bool:
    return extension_api_major_version(api_version) == extension_api_major_version(EXTENSION_API_VERSION)


def extension_api_major_version(version: str) -> int:
    """Return the integer major component for an extension API/version string."""
    text = str(version).strip()
    if not text:
        return -1
    token = text.split(".", 1)[0]
    return int(token) if token.isdigit() else -1


def _version_satisfies(version: str, specifier: str) -> bool:
    try:
        normalized = _normalized_version_specifier(specifier)
        parsed_version = Version(str(version))
    except (InvalidSpecifier, InvalidVersion):
        return False
    return parsed_version in SpecifierSet(normalized)


def _normalized_version_specifier(specifier: str) -> str:
    conditions: list[str] = []
    for raw_condition in str(specifier).split(","):
        condition = raw_condition.strip()
        if not condition:
            continue
        if condition.lower().startswith("eegprep"):
            condition = condition[len("eegprep") :].strip()
        if condition.startswith("=") and not condition.startswith("=="):
            condition = f"=={condition[1:].strip()}"
        elif not condition.startswith(("~=", "==", "!=", "<=", ">=", "<", ">")):
            condition = f"=={condition}"
        conditions.append(condition)
    normalized = ",".join(conditions)
    if not normalized:
        raise InvalidSpecifier("empty version specifier")
    SpecifierSet(normalized)
    return normalized


def _normalize_name(name: str) -> str:
    return str(name).strip().lower()


def _normalize_menu_path(path: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(_normalize_name(str(part)) for part in path)


def _extension_menu_from_spec(
    spec: Any,
    *,
    path: tuple[str, ...] = (),
    insert_after: str = "",
    insert_before: str = "",
) -> ExtensionMenu:
    return ExtensionMenu(
        path=path,
        action=spec.action or "",
        label=spec.label,
        tag=spec.tag or "",
        userdata=spec.userdata,
        separator=spec.separator,
        enabled=spec.enabled,
        checked=spec.checked,
        visibility=spec.visibility,
        insert_after=insert_after,
        insert_before=insert_before,
        children=tuple(_extension_menu_from_spec(child) for child in spec.children),
    )


def _extension_menu_keys(menus: tuple[ExtensionMenu, ...]) -> tuple[tuple[tuple[tuple[str, ...], str], str], ...]:
    keys = []
    for menu in menus:
        label = menu.label or menu.action
        if label:
            keys.append(((_normalize_menu_path(menu.path), _normalize_name(label)), label))
    return tuple(keys)


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, set):
        return tuple(sorted(value, key=repr))
    return (value,)


__all__ = [
    "EXTENSION_API_VERSION",
    "EXTENSION_ACTIVE_STATUSES",
    "EXTENSION_COMPATIBILITY_POLICY",
    "EXTENSION_CURATION_POLICY_URL",
    "EXTENSION_ENTRY_POINT_GROUP",
    "EXTENSION_INSTALLED_STATUSES",
    "EXTENSION_NAMING_PREFIX",
    "EXTENSION_TRUST_MESSAGE",
    "ExtensionAction",
    "ExtensionDependency",
    "ExtensionLoadError",
    "ExtensionMenu",
    "ExtensionPopFunction",
    "ExtensionRecord",
    "ExtensionRegistry",
    "ExtensionResource",
    "ExtensionSourceType",
    "ExtensionSpec",
    "ExtensionStatus",
    "ExtensionValidationResult",
    "LazyImport",
    "check_extension_compatibility",
    "compare_extension_versions",
    "discover_extensions",
    "extension_api_major_version",
    "extension_entry_point_package_name",
    "extension_status_is_active",
    "extension_status_is_installed",
    "extension_version_satisfies",
    "extension_version_spec_is_valid",
    "select_extension_entry_points",
    "validate_extension_spec",
]
