"""External extension SDK and discovery registry for EEGPrep."""

from __future__ import annotations

import importlib
import importlib.resources as resources
import logging
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from importlib import metadata
from itertools import zip_longest
from typing import Any, Callable

EXTENSION_API_VERSION = "1"
EXTENSION_ENTRY_POINT_GROUP = "eegprep.extensions"

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

    @classmethod
    def from_string(cls, value: str) -> "LazyImport":
        """Build a lazy reference from ``"module:attribute"`` text."""
        module, separator, attr = value.partition(":")
        if not separator or not module or not attr:
            raise ValueError("Lazy imports must use 'module:attribute' syntax")
        return cls(module=module, attr=attr)

    def load(self) -> Any:
        """Import and return the referenced object."""
        try:
            module = importlib.import_module(self.module)
            return getattr(module, self.attr)
        except Exception as exc:  # pragma: no cover - exact import failures vary by platform
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
        except Exception as exc:  # pragma: no cover - exact resource failures vary by loader
            raise FileNotFoundError(f"Extension resource {self.package}:{self.path} is not available") from exc

    def read_bytes(self) -> bytes:
        """Read this resource as bytes."""
        try:
            return resources.files(self.package).joinpath(self.path).read_bytes()
        except Exception as exc:  # pragma: no cover - exact resource failures vary by loader
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
    """Future menu placement for an extension action."""

    path: tuple[str, ...]
    action: str
    label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _as_tuple(self.path))


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
        return (
            self.enabled
            and self.spec is not None
            and self.status
            in {
                ExtensionStatus.BUNDLED,
                ExtensionStatus.INSTALLED,
                ExtensionStatus.CURATED,
            }
        )


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
        from eegprep.functions.adminfunc.plugin_menu import bundled_plugins

        records = []
        for plugin in bundled_plugins():
            plugin_name = str(plugin["plugin"])
            folder_name = str(plugin.get("foldername") or plugin_name)
            funcname = str(plugin.get("funcname") or "")
            pop_functions = ()
            if funcname:
                pop_functions = (
                    ExtensionPopFunction(
                        name=funcname,
                        target=LazyImport(f"eegprep.plugins.{folder_name}.{funcname}", funcname),
                        description=str(plugin.get("description") or ""),
                    ),
                )
            menus = ()
            if plugin.get("menu") and funcname:
                menus = (
                    ExtensionMenu(
                        path=tuple(part.strip() for part in str(plugin["menu"]).split(">")),
                        action=funcname,
                        label=str(plugin.get("name") or plugin_name),
                    ),
                )
            spec = ExtensionSpec(
                name=plugin_name,
                display_name=str(plugin.get("name") or plugin_name),
                version=str(plugin.get("version") or "bundled"),
                api_version=EXTENSION_API_VERSION,
                package_name=f"eegprep.plugins.{folder_name}",
                source_type=ExtensionSourceType.BUNDLED,
                description=str(plugin.get("description") or ""),
                capabilities=tuple(str(tag) for tag in plugin.get("tags", ())),
                menus=menus,
                pop_functions=pop_functions,
            )
            records.append(self._record_from_spec(spec))
        return records

    def _entry_point_records(self) -> list[ExtensionRecord]:
        records = []
        for entry_point in _select_entry_points(self._entry_points_provider, self.entry_point_group):
            records.append(self._record_from_entry_point(entry_point))
        return records

    def _record_from_entry_point(self, entry_point: Any) -> ExtensionRecord:
        entry_point_name = str(getattr(entry_point, "name", "") or "<unknown>")
        package_name = _entry_point_package_name(entry_point)
        try:
            loaded = entry_point.load()
            candidate = loaded() if callable(loaded) and not isinstance(loaded, ExtensionSpec) else loaded
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
            package_name=candidate.package_name or package_name,
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
        final_records: list[ExtensionRecord] = []

        for record in records:
            if not _can_contribute(record):
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

            if errors:
                final_records.append(_invalid_record(record, tuple(errors)))
                continue

            extension_names.add(extension_name)
            if spec is not None:
                for action in spec.actions:
                    action_names[_normalize_name(action.name)] = record.name
                for pop_function in spec.pop_functions:
                    pop_names[_normalize_name(pop_function.name)] = record.name
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

    if check_compatibility and spec.eegprep_requires and not _version_satisfies(current_version, spec.eegprep_requires):
        incompatible.append(f"Extension requires EEGPrep {spec.eegprep_requires}; current version is {current_version}")

    if check_dependencies:
        dependency_invalid, dependency_missing = _dependency_errors(spec.dependencies, version_provider)
        invalid.extend(dependency_invalid)
        missing_dependency.extend(dependency_missing)

    if check_resources:
        for resource in (*spec.help_resources, *spec.package_data_resources):
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
    for menu in menus:
        if not isinstance(menu, ExtensionMenu):
            invalid.append(f"Extension menu {menu!r} is not an ExtensionMenu")
            continue
        if not menu.path:
            invalid.append("Extension menus must include a path")
        if not menu.action:
            invalid.append("Extension menus must reference an action")


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


def _select_entry_points(provider: EntryPointsProvider, group: str) -> tuple[Any, ...]:
    try:
        selected = provider(group=group)
    except TypeError:
        entry_points = provider()
        if hasattr(entry_points, "select"):
            selected = entry_points.select(group=group)
        else:
            selected = [entry_point for entry_point in entry_points if getattr(entry_point, "group", None) == group]
    return tuple(selected or ())


def _status_for_spec(
    source_type: ExtensionSourceType,
    enabled: bool,
    validation: ExtensionValidationResult,
) -> ExtensionStatus:
    if validation.invalid_spec:
        return ExtensionStatus.INVALID_SPEC
    if not enabled:
        return ExtensionStatus.DISABLED
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


def _can_contribute(record: ExtensionRecord) -> bool:
    return (
        record.enabled
        and record.spec is not None
        and record.status
        in {
            ExtensionStatus.BUNDLED,
            ExtensionStatus.INSTALLED,
            ExtensionStatus.CURATED,
        }
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


def _entry_point_package_name(entry_point: Any) -> str:
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
    return _major_version(api_version) == _major_version(EXTENSION_API_VERSION)


def _major_version(version: str) -> int:
    text = str(version).strip()
    if not text:
        return -1
    token = text.split(".", 1)[0]
    return int(token) if token.isdigit() else -1


def _version_satisfies(version: str, specifier: str) -> bool:
    for raw_condition in specifier.split(","):
        condition = raw_condition.strip()
        if not condition:
            continue
        if condition.lower().startswith("eegprep"):
            condition = condition[len("eegprep") :].strip()
        operator, expected = _split_version_condition(condition)
        comparison = _compare_versions(version, expected)
        if operator == "==" and comparison != 0:
            return False
        if operator == "!=" and comparison == 0:
            return False
        if operator == ">=" and comparison < 0:
            return False
        if operator == ">" and comparison <= 0:
            return False
        if operator == "<=" and comparison > 0:
            return False
        if operator == "<" and comparison >= 0:
            return False
        if operator == "~=" and comparison < 0:
            return False
    return True


def _split_version_condition(condition: str) -> tuple[str, str]:
    for operator in (">=", "<=", "==", "!=", "~=", ">", "<"):
        if condition.startswith(operator):
            return operator, condition[len(operator) :].strip()
    if condition.startswith("="):
        return "==", condition[1:].strip()
    return "==", condition


def _compare_versions(left: str, right: str) -> int:
    left_parts = _version_parts(left)
    right_parts = _version_parts(right)
    for left_part, right_part in zip_longest(left_parts, right_parts, fillvalue=0):
        if left_part < right_part:
            return -1
        if left_part > right_part:
            return 1
    return 0


def _version_parts(version: str) -> tuple[int, ...]:
    parts = tuple(int(part) for part in re.findall(r"\d+", str(version)))
    return parts or (0,)


def _normalize_name(name: str) -> str:
    return str(name).strip().lower()


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, set):
        return tuple(value)
    return (value,)


__all__ = [
    "EXTENSION_API_VERSION",
    "EXTENSION_ENTRY_POINT_GROUP",
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
    "discover_extensions",
    "validate_extension_spec",
]
