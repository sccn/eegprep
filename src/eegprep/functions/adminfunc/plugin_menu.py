"""EEGLAB-style extension inventory, status checks, and manager dialog."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
import html
import importlib
import re
from typing import Any

from eegprep.extensions import ExtensionRecord, ExtensionRegistry, ExtensionSourceType, ExtensionStatus
from eegprep.extension_catalog import (
    INSTALL_TRUST_WARNING,
    ExtensionCatalog,
    ExtensionCatalogEntry,
    build_safe_install_commands,
    build_safe_update_commands,
    load_extension_catalog,
)

EXTERNAL_PLUGIN_NOTICE = (
    "EEGPrep discovers installed Python extension packages through entry points and shows curated catalog metadata. "
    "This manager never downloads, unzips, installs, updates, or removes extension code."
)

_ACTIVE_STATUSES = {
    ExtensionStatus.BUNDLED.value,
    ExtensionStatus.INSTALLED.value,
    ExtensionStatus.CURATED.value,
    "ok",
}
_INSTALLED_STATUSES = {
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
_STATUS_LABELS = {
    ExtensionStatus.BUNDLED.value: "Bundled",
    ExtensionStatus.INSTALLED.value: "Installed",
    ExtensionStatus.CURATED.value: "Curated",
    ExtensionStatus.DISABLED.value: "Disabled",
    ExtensionStatus.INCOMPATIBLE.value: "Incompatible",
    ExtensionStatus.FAILED_IMPORT.value: "Failed",
    ExtensionStatus.INVALID_SPEC.value: "Invalid",
    ExtensionStatus.MISSING_DEPENDENCY.value: "Missing dependency",
    ExtensionStatus.UNKNOWN.value: "Unknown",
    "ok": "Installed",
    "unavailable": "Unavailable",
}
_STATUS_COLORS = {
    ExtensionStatus.BUNDLED.value: "#dff0d8",
    ExtensionStatus.INSTALLED.value: "#e7f0fd",
    ExtensionStatus.CURATED.value: "#fff4ce",
    ExtensionStatus.DISABLED.value: "#eeeeee",
    ExtensionStatus.INCOMPATIBLE.value: "#fde7e9",
    ExtensionStatus.FAILED_IMPORT.value: "#f8d7da",
    ExtensionStatus.INVALID_SPEC.value: "#f8d7da",
    ExtensionStatus.MISSING_DEPENDENCY.value: "#ffe9cc",
    ExtensionStatus.UNKNOWN.value: "#eeeeee",
}

_BUNDLED_PLUGINS: tuple[dict[str, Any], ...] = (
    {
        "plugin": "clean_rawdata",
        "name": "clean_rawdata",
        "version": "bundled",
        "foldername": "clean_rawdata",
        "funcname": "pop_clean_rawdata",
        "status": "ok",
        "installed": True,
        "source": "bundled",
        "menu": "Tools > Reject data using Clean Rawdata and ASR",
        "description": "Artifact Subspace Reconstruction and related channel/window cleaning workflows.",
        "tags": ("artifact", "preprocessing"),
    },
    {
        "plugin": "ICLabel",
        "name": "ICLabel",
        "version": "bundled",
        "foldername": "ICLabel",
        "funcname": "pop_iclabel",
        "status": "ok",
        "installed": True,
        "source": "bundled",
        "menu": "Tools > Classify components using ICLabel",
        "description": "Independent-component classification, flagging, and extended component properties.",
        "tags": ("ica", "classification"),
    },
    {
        "plugin": "firfilt",
        "name": "firfilt",
        "version": "bundled",
        "foldername": "firfilt",
        "funcname": "pop_eegfiltnew",
        "status": "ok",
        "installed": True,
        "source": "bundled",
        "menu": "Tools > Filter the data",
        "description": "Windowed-sinc, Parks-McClellan, moving-average, and new default FIR filtering.",
        "tags": ("filter", "preprocessing"),
    },
    {
        "plugin": "dipfit",
        "name": "DIPFIT",
        "version": "bundled",
        "foldername": "dipfit",
        "funcname": "pop_dipfit_settings",
        "status": "ok",
        "installed": True,
        "source": "bundled",
        "menu": "Tools > Source localization using DIPFIT",
        "description": "Source-localization menu surfaces and FieldTrip-backed DIPFIT workflows.",
        "tags": ("source", "localization"),
    },
    {
        "plugin": "EEG_BIDS",
        "name": "EEG-BIDS",
        "version": "bundled",
        "foldername": "EEG_BIDS",
        "funcname": "pop_importbids",
        "status": "ok",
        "installed": True,
        "source": "bundled",
        "menu": "File > Import data / Export / BIDS tools",
        "description": "BIDS import, export, validation, and metadata helpers for EEG datasets.",
        "tags": ("import", "export", "bids", "study"),
    },
)


def bundled_plugins() -> tuple[dict[str, Any], ...]:
    """Return metadata for EEGPrep extensions bundled in the installed package."""
    return tuple(deepcopy(plugin) for plugin in _BUNDLED_PLUGINS)


def plugin_status(
    pluginname: str,
    *,
    exactmatch: bool = False,
    pluginlist: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    registry: ExtensionRegistry | None = None,
    catalog: ExtensionCatalog | None = None,
    catalog_path: str | None = None,
    include_bundled: bool = True,
    include_entry_points: bool = True,
    disabled_extensions: set[str] | list[str] | tuple[str, ...] | None = None,
) -> tuple[list[int], list[str], list[dict[str, Any]]]:
    """Return EEGLAB-style installed status for EEGPrep extensions.

    Args:
        pluginname: Plugin or extension name, package name, or substring to search.
        exactmatch: Require exact case-insensitive name matching.
        pluginlist: Optional precomputed extension inventory. Defaults to the
            registry plus the curated catalog.
        registry: Optional discovered registry for tests or callers that need
            explicit discovery control.
        catalog: Optional loaded catalog. Defaults to the packaged/local catalog.
        catalog_path: Optional JSON catalog path.
        include_bundled: Include bundled EEGPrep plugin ports in default discovery.
        include_entry_points: Include installed entry-point extensions in default discovery.
        disabled_extensions: Registry names to mark disabled during default discovery.

    Returns:
        A tuple ``(status, names, pluginstruct)`` where status values are ``1``
        for active installed/bundled extensions and ``0`` for curated-only,
        disabled, incompatible, failed, or missing-dependency matches.
    """
    needle = pluginname.lower()
    plugins, _catalog_info = _plugins_for_manager(
        pluginlist,
        registry=registry,
        catalog=catalog,
        catalog_path=catalog_path,
        include_bundled=include_bundled,
        include_entry_points=include_entry_points,
        disabled_extensions=disabled_extensions,
    )
    matches: list[dict[str, Any]] = []
    for plugin in plugins:
        names = {
            str(plugin.get("plugin", "")),
            str(plugin.get("name", "")),
            str(plugin.get("display_name", "")),
            str(plugin.get("foldername", "")),
            str(plugin.get("package_name", "")),
        }
        lowered = {name.lower() for name in names if name}
        if exactmatch and needle not in lowered:
            continue
        if not exactmatch and not any(needle in name for name in lowered):
            continue
        matches.append(plugin)
    return (
        [1 if _is_active(plugin) else 0 for plugin in matches],
        [str(plugin.get("plugin") or plugin.get("name") or "") for plugin in matches],
        [deepcopy(plugin) for plugin in matches],
    )


def plugin_menu(
    pluginlist: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    *,
    parent: Any | None = None,
    session: Any | None = None,
    show: bool = True,
    registry: ExtensionRegistry | None = None,
    catalog: ExtensionCatalog | None = None,
    catalog_path: str | None = None,
    include_bundled: bool = True,
    include_entry_points: bool = True,
    disabled_extensions: set[str] | list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Show or return the EEGPrep Extension Manager inventory.

    Args:
        pluginlist: Optional extension inventory to display. Defaults to the
            Phase 1 extension registry merged with the curated metadata catalog.
        parent: Optional Qt parent widget for the dialog.
        session: Optional :class:`EEGPrepSession`; its ``PLUGINLIST`` mirror is
            updated with the displayed inventory.
        show: Show the Qt dialog when ``True``. Use ``False`` for scripts,
            examples, tests, or console inventory checks.
        registry: Optional discovered registry for tests or explicit control.
        catalog: Optional loaded catalog. Defaults to the packaged/local catalog.
        catalog_path: Optional JSON catalog path.
        include_bundled: Include bundled EEGPrep plugin ports in default discovery.
        include_entry_points: Include installed entry-point extensions in default discovery.
        disabled_extensions: Registry names to mark disabled during default discovery.

    Returns:
        The normalized extension inventory as a mutable list of dictionaries.
        Records include install/update command strings but never execute them.
    """
    plugins, catalog_info = _plugins_for_manager(
        pluginlist,
        registry=registry,
        catalog=catalog,
        catalog_path=catalog_path,
        include_bundled=include_bundled,
        include_entry_points=include_entry_points,
        disabled_extensions=disabled_extensions,
    )
    if session is not None:
        session.PLUGINLIST = [deepcopy(plugin) for plugin in plugins]
    if show:
        _show_plugin_dialog(plugins, parent, catalog_info)
    return plugins


def format_plugin_menu(
    pluginlist: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    *,
    registry: ExtensionRegistry | None = None,
    catalog: ExtensionCatalog | None = None,
    catalog_path: str | None = None,
    include_bundled: bool = True,
    include_entry_points: bool = True,
    disabled_extensions: set[str] | list[str] | tuple[str, ...] | None = None,
) -> str:
    """Return a plain-text extension inventory for console display."""
    plugins, catalog_info = _plugins_for_manager(
        pluginlist,
        registry=registry,
        catalog=catalog,
        catalog_path=catalog_path,
        include_bundled=include_bundled,
        include_entry_points=include_entry_points,
        disabled_extensions=disabled_extensions,
    )
    lines = ["Available EEGPrep extensions", ""]
    if catalog_info["catalog_source"]:
        lines.append(
            f"Catalog: {catalog_info['catalog_source']} "
            f"({catalog_info['catalog_entry_count']} curated entr"
            f"{'y' if catalog_info['catalog_entry_count'] == 1 else 'ies'})"
        )
        for error in catalog_info["catalog_errors"]:
            lines.append(f"Catalog warning: {error}")
        lines.append("")

    if not plugins:
        lines.append("No extensions discovered.")
    for plugin in plugins:
        version = str(plugin.get("version") or plugin.get("catalog_version") or "unknown")
        installed = "installed" if plugin.get("installed", False) else "not installed"
        lines.append(f"- {plugin['name']} v{version} ({plugin['state_label']}; {installed})")
        if plugin.get("description"):
            lines.append(f"  {plugin['description']}")
        if plugin.get("package_name"):
            lines.append(f"  Package: {plugin['package_name']}")
        if plugin.get("docs_url"):
            lines.append(f"  Docs: {plugin['docs_url']}")
        if plugin.get("menu"):
            lines.append(f"  Menu: {plugin['menu']}")
        if plugin.get("errors"):
            lines.append(f"  Errors: {'; '.join(plugin['errors'])}")
        if plugin.get("catalog_conflicts"):
            lines.append(f"  Catalog conflict: {'; '.join(plugin['catalog_conflicts'])}")
        install_commands = plugin.get("install_commands") or {}
        if install_commands and not plugin.get("installed"):
            lines.append(f"  Install: {install_commands.get('uv') or next(iter(install_commands.values()))}")
        update_commands = plugin.get("update_commands") or {}
        if update_commands and _can_show_update_commands(plugin):
            lines.append(f"  Update: {update_commands.get('uv') or next(iter(update_commands.values()))}")
    lines.extend(["", EXTERNAL_PLUGIN_NOTICE, INSTALL_TRUST_WARNING])
    return "\n".join(lines)


def _plugins_for_manager(
    pluginlist: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    *,
    registry: ExtensionRegistry | None,
    catalog: ExtensionCatalog | None,
    catalog_path: str | None,
    include_bundled: bool,
    include_entry_points: bool,
    disabled_extensions: set[str] | list[str] | tuple[str, ...] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    loaded_catalog = catalog if catalog is not None else load_extension_catalog(catalog_path)
    catalog_info = _catalog_info(loaded_catalog)
    if pluginlist is not None:
        plugins = [_normalize_plugin(plugin, catalog_info=catalog_info) for plugin in pluginlist]
        return plugins, catalog_info

    if registry is None:
        registry = ExtensionRegistry(
            disabled_extensions=disabled_extensions,
            include_bundled=include_bundled,
            include_entry_points=include_entry_points,
        )
        records = registry.discover()
    else:
        records = registry.records or registry.discover()

    return _plugins_from_records(records, loaded_catalog, catalog_info), catalog_info


def _plugins_from_records(
    records: Sequence[ExtensionRecord],
    catalog: ExtensionCatalog,
    catalog_info: dict[str, Any],
) -> list[dict[str, Any]]:
    by_name = catalog.by_name()
    by_package = catalog.by_package()
    used_catalog_names: set[str] = set()
    plugins: list[dict[str, Any]] = []

    for order, record in enumerate(records):
        catalog_entry = _catalog_for_record(record, by_name, by_package)
        if catalog_entry is not None:
            used_catalog_names.add(_normalize_name(catalog_entry.name))
        plugin = _plugin_from_record(record, catalog_entry, catalog_info)
        plugin["_order"] = order
        plugins.append(plugin)

    for index, catalog_entry in enumerate(catalog.entries):
        if _normalize_name(catalog_entry.name) in used_catalog_names:
            continue
        plugin = _plugin_from_catalog(catalog_entry, catalog_info)
        plugin["_order"] = len(records) + index
        plugins.append(plugin)

    sorted_plugins = sorted(plugins, key=_plugin_sort_key)
    for plugin in sorted_plugins:
        plugin.pop("_order", None)
    return sorted_plugins


def _catalog_for_record(
    record: ExtensionRecord,
    by_name: dict[str, ExtensionCatalogEntry],
    by_package: dict[str, ExtensionCatalogEntry],
) -> ExtensionCatalogEntry | None:
    entry = by_name.get(_normalize_name(record.name))
    if entry is not None:
        return entry
    if record.package_name:
        return by_package.get(_normalize_name(record.package_name))
    return None


def _plugin_from_record(
    record: ExtensionRecord,
    catalog_entry: ExtensionCatalogEntry | None,
    catalog_info: dict[str, Any],
) -> dict[str, Any]:
    spec = record.spec
    spec_display_name = spec.display_name if spec is not None else ""
    display_name = spec_display_name or (catalog_entry.display_name if catalog_entry else "") or record.name
    description = (spec.description if spec is not None else "") or (catalog_entry.description if catalog_entry else "")
    version = (spec.version if spec is not None else "") or ""
    package_name = record.package_name or (spec.package_name if spec is not None else "")
    docs_url = (spec.docs_url if spec is not None else "") or (catalog_entry.docs_url if catalog_entry else "")
    maintainer = (spec.maintainer if spec is not None else "") or (catalog_entry.maintainer if catalog_entry else "")
    capabilities = _dedupe(
        (*(spec.capabilities if spec is not None else ()), *(catalog_entry.capabilities if catalog_entry else ()))
    )
    install_commands = build_safe_install_commands(catalog_entry) if catalog_entry is not None else {}
    update_commands = build_safe_update_commands(catalog_entry) if catalog_entry is not None else {}
    conflicts = _catalog_conflicts(record, catalog_entry)
    status = record.status.value
    installed = status in _INSTALLED_STATUSES
    active = installed and status in _ACTIVE_STATUSES
    menu = _menu_text(record)
    plugin = {
        "plugin": record.name,
        "name": display_name,
        "display_name": display_name,
        "version": version,
        "catalog_version": catalog_entry.version if catalog_entry is not None else "",
        "foldername": _folder_name(record, package_name),
        "funcname": _first_pop_function(record),
        "status": status,
        "state_label": _STATUS_LABELS.get(status, status.replace("_", " ").title()),
        "installed": installed,
        "enabled": record.enabled,
        "curated": catalog_entry is not None,
        "source": record.source_type.value,
        "source_detail": _source_detail(record, catalog_entry),
        "source_url": catalog_entry.source_url if catalog_entry is not None else "",
        "repository_url": catalog_entry.repository_url if catalog_entry is not None else "",
        "package_name": package_name,
        "entry_point_name": record.entry_point_name,
        "menu": menu,
        "description": description,
        "maintainer": maintainer,
        "docs_url": docs_url,
        "capabilities": capabilities,
        "tags": capabilities,
        "errors": tuple(record.errors),
        "catalog_status": _catalog_status(catalog_entry, catalog_info, conflicts),
        "catalog_conflicts": conflicts,
        "install_commands": install_commands,
        "update_commands": update_commands,
        "update_available": _catalog_version_is_newer(version, catalog_entry.version if catalog_entry else ""),
        "install_guidance": _install_guidance(status, installed, active, catalog_entry, conflicts),
        "trust_warning": INSTALL_TRUST_WARNING,
        **catalog_info,
    }
    return _normalize_plugin(plugin, catalog_info=catalog_info)


def _plugin_from_catalog(catalog_entry: ExtensionCatalogEntry, catalog_info: dict[str, Any]) -> dict[str, Any]:
    install_commands = build_safe_install_commands(catalog_entry)
    update_commands = build_safe_update_commands(catalog_entry)
    plugin = {
        "plugin": catalog_entry.name,
        "name": catalog_entry.display_name or catalog_entry.name,
        "display_name": catalog_entry.display_name or catalog_entry.name,
        "version": catalog_entry.version,
        "catalog_version": catalog_entry.version,
        "foldername": catalog_entry.package_name or catalog_entry.name,
        "funcname": "",
        "status": ExtensionStatus.CURATED.value,
        "state_label": _STATUS_LABELS[ExtensionStatus.CURATED.value],
        "installed": False,
        "active": False,
        "enabled": False,
        "curated": True,
        "source": ExtensionSourceType.CURATED.value,
        "source_detail": catalog_entry.source_label,
        "source_url": catalog_entry.source_url,
        "repository_url": catalog_entry.repository_url,
        "package_name": catalog_entry.package_name,
        "entry_point_name": "",
        "menu": "",
        "description": catalog_entry.description,
        "maintainer": catalog_entry.maintainer,
        "docs_url": catalog_entry.docs_url,
        "capabilities": catalog_entry.capabilities,
        "tags": catalog_entry.capabilities,
        "errors": (),
        "catalog_status": "catalog_only",
        "catalog_conflicts": (),
        "install_commands": install_commands,
        "update_commands": update_commands,
        "update_available": False,
        "install_guidance": "Install with a package manager, then restart EEGPrep so entry-point discovery can load it.",
        "trust_warning": INSTALL_TRUST_WARNING,
        **catalog_info,
    }
    return _normalize_plugin(plugin, catalog_info=catalog_info)


def _normalize_plugin(plugin: dict[str, Any], *, catalog_info: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = dict(plugin)
    name = str(normalized.get("name") or normalized.get("display_name") or normalized.get("plugin") or "")
    plugin_name = str(normalized.get("plugin") or normalized.get("foldername") or name)
    if not name and not plugin_name:
        raise ValueError("Plugin entries must include a name, display_name, plugin, or foldername")
    normalized.setdefault("plugin", plugin_name or name)
    normalized.setdefault("name", name or plugin_name)
    normalized.setdefault("display_name", normalized["name"])
    normalized.setdefault("version", "bundled")
    normalized.setdefault("catalog_version", "")
    normalized.setdefault("foldername", normalized["plugin"])
    normalized.setdefault("funcname", "")
    normalized.setdefault("status", "ok" if normalized.get("installed", True) else "unavailable")
    normalized.setdefault("state_label", _STATUS_LABELS.get(str(normalized["status"]), str(normalized["status"])))
    normalized.setdefault("installed", str(normalized.get("status")) in _INSTALLED_STATUSES)
    normalized["active"] = _is_active(normalized)
    normalized.setdefault("enabled", normalized["active"])
    normalized.setdefault("curated", False)
    normalized.setdefault("source", "bundled")
    normalized.setdefault("source_detail", normalized["source"])
    normalized.setdefault("source_url", "")
    normalized.setdefault("repository_url", "")
    normalized.setdefault("package_name", "")
    normalized.setdefault("entry_point_name", "")
    normalized.setdefault("menu", "")
    normalized.setdefault("description", "")
    normalized.setdefault("maintainer", "")
    normalized.setdefault("docs_url", "")
    normalized.setdefault("capabilities", tuple(normalized.get("tags", ())))
    normalized.setdefault("tags", tuple(normalized.get("capabilities", ())))
    normalized.setdefault("errors", ())
    normalized.setdefault("catalog_status", "")
    normalized.setdefault("catalog_conflicts", ())
    normalized.setdefault("install_commands", {})
    normalized.setdefault("update_commands", {})
    normalized.setdefault("update_available", False)
    normalized.setdefault("install_guidance", "")
    normalized.setdefault("trust_warning", INSTALL_TRUST_WARNING)
    if catalog_info:
        normalized.update({key: deepcopy(value) for key, value in catalog_info.items() if key not in normalized})
    normalized["capabilities"] = _dedupe(normalized.get("capabilities", ()))
    normalized["tags"] = _dedupe(normalized.get("tags", normalized["capabilities"]))
    normalized["errors"] = tuple(normalized.get("errors") or ())
    normalized["catalog_conflicts"] = tuple(normalized.get("catalog_conflicts") or ())
    normalized["install_commands"] = dict(normalized.get("install_commands") or {})
    normalized["update_commands"] = dict(normalized.get("update_commands") or {})
    return normalized


def _catalog_conflicts(
    record: ExtensionRecord,
    catalog_entry: ExtensionCatalogEntry | None,
) -> tuple[str, ...]:
    if catalog_entry is None:
        return ()
    conflicts: list[str] = []
    spec = record.spec
    spec_display_name = spec.display_name if spec is not None else ""
    spec_docs_url = spec.docs_url if spec is not None else ""
    package_name = record.package_name or (spec.package_name if spec is not None else "")
    if package_name and catalog_entry.package_name and package_name != catalog_entry.package_name:
        conflicts.append(
            f"Catalog package name {catalog_entry.package_name!r} differs from installed package {package_name!r}"
        )
    if spec_display_name and catalog_entry.display_name and spec_display_name != catalog_entry.display_name:
        conflicts.append(
            f"Catalog display name {catalog_entry.display_name!r} differs from installed display name {spec_display_name!r}"
        )
    if spec_docs_url and catalog_entry.docs_url and spec_docs_url != catalog_entry.docs_url:
        conflicts.append(
            f"Catalog docs URL {catalog_entry.docs_url!r} differs from installed docs URL {spec_docs_url!r}"
        )
    return tuple(conflicts)


def _catalog_status(
    catalog_entry: ExtensionCatalogEntry | None,
    catalog_info: dict[str, Any],
    conflicts: tuple[str, ...],
) -> str:
    if catalog_info["catalog_errors"]:
        return "catalog_unavailable"
    if catalog_entry is None:
        return "not_curated"
    if conflicts:
        return "conflict"
    return "curated"


def _catalog_info(catalog: ExtensionCatalog) -> dict[str, Any]:
    return {
        "catalog_source": catalog.source,
        "catalog_available": catalog.available,
        "catalog_entry_count": len(catalog.entries),
        "catalog_errors": tuple(catalog.errors),
    }


def _show_plugin_dialog(plugins: list[dict[str, Any]], parent: Any | None, catalog_info: dict[str, Any]) -> None:
    try:
        QtWidgets = importlib.import_module("PySide6.QtWidgets")
    except ImportError as exc:  # pragma: no cover - optional GUI dependency
        raise RuntimeError(
            "PySide6 is required to show the EEGPrep Extension Manager. "
            "Call plugin_menu(show=False) to inspect extensions without a GUI."
        ) from exc

    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = _build_plugin_dialog(plugins, parent=parent, catalog_info=catalog_info)
    dialog.exec()


def _build_plugin_dialog(
    plugins: list[dict[str, Any]],
    *,
    parent: Any | None = None,
    catalog_info: dict[str, Any] | None = None,
) -> Any:
    QtCore = importlib.import_module("PySide6.QtCore")
    QtGui = importlib.import_module("PySide6.QtGui")
    QtWidgets = importlib.import_module("PySide6.QtWidgets")

    catalog_info = catalog_info or _catalog_info(ExtensionCatalog())
    dialog = QtWidgets.QDialog(parent)
    dialog.setObjectName("extension_manager")
    dialog.setWindowTitle("EEGPrep Extension Manager")
    dialog.resize(1120, 620)

    layout = QtWidgets.QVBoxLayout(dialog)
    layout.setContentsMargins(14, 12, 14, 12)
    layout.setSpacing(8)

    title = QtWidgets.QLabel("EEGPrep Extension Manager")
    title.setObjectName("extension_manager_title")
    title_font = title.font()
    title_font.setPointSize(title_font.pointSize() + 3)
    title_font.setBold(True)
    title.setFont(title_font)
    layout.addWidget(title)

    subtitle = QtWidgets.QLabel(
        "Installed Python packages are discovered through entry points. Curated entries are metadata only."
    )
    subtitle.setObjectName("extension_manager_subtitle")
    subtitle.setWordWrap(True)
    layout.addWidget(subtitle)

    splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
    splitter.setObjectName("extension_manager_splitter")
    layout.addWidget(splitter, 1)

    table = QtWidgets.QTableWidget(len(plugins), 4, dialog)
    table.setObjectName("extension_table")
    table.setHorizontalHeaderLabels(["Extension", "State", "Version", "Source"])
    table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
    table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setAlternatingRowColors(True)
    table.setWordWrap(False)
    for row, plugin in enumerate(plugins):
        values = (
            plugin["name"],
            plugin["state_label"],
            plugin.get("version") or plugin.get("catalog_version") or "",
            plugin.get("source_detail") or plugin.get("source") or "",
        )
        for column, value in enumerate(values):
            item = QtWidgets.QTableWidgetItem(str(value))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, row)
            if column == 1:
                color = _STATUS_COLORS.get(str(plugin.get("status")), "#ffffff")
                item.setBackground(QtGui.QColor(color))
            if column == 0 and plugin.get("active"):
                item_font = item.font()
                item_font.setBold(True)
                item.setFont(item_font)
            table.setItem(row, column, item)
    table.verticalHeader().setDefaultSectionSize(28)
    table.verticalHeader().setVisible(False)
    header = table.horizontalHeader()
    table.setColumnWidth(0, 210)
    header.setMinimumSectionSize(72)
    header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Interactive)
    header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
    header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
    header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)
    splitter.addWidget(table)

    detail_widget = QtWidgets.QWidget(dialog)
    detail_layout = QtWidgets.QVBoxLayout(detail_widget)
    detail_layout.setContentsMargins(10, 0, 0, 0)
    detail_layout.setSpacing(8)

    details = QtWidgets.QTextBrowser(detail_widget)
    details.setObjectName("extension_details")
    details.setOpenExternalLinks(True)
    detail_layout.addWidget(details, 1)

    guidance = QtWidgets.QLabel(detail_widget)
    guidance.setObjectName("extension_install_guidance")
    guidance.setWordWrap(True)
    detail_layout.addWidget(guidance)

    command_box = QtWidgets.QPlainTextEdit(detail_widget)
    command_box.setObjectName("extension_install_commands")
    command_box.setReadOnly(True)
    command_box.setMaximumHeight(120)
    detail_layout.addWidget(command_box)

    copy_button = QtWidgets.QPushButton("Copy commands", detail_widget)
    copy_button.setObjectName("copy_extension_commands")
    detail_layout.addWidget(copy_button)
    splitter.addWidget(detail_widget)
    splitter.setSizes([520, 580])

    warning = QtWidgets.QLabel(INSTALL_TRUST_WARNING)
    warning.setObjectName("extension_trust_warning")
    warning.setWordWrap(True)
    warning.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    layout.addWidget(warning)

    notice = QtWidgets.QLabel(_catalog_notice(catalog_info))
    notice.setObjectName("extension_catalog_notice")
    notice.setWordWrap(True)
    notice.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    layout.addWidget(notice)

    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
    help_button = buttons.addButton("Help", QtWidgets.QDialogButtonBox.ButtonRole.HelpRole)
    help_button.setObjectName("extension_manager_help")
    buttons.rejected.connect(dialog.reject)
    buttons.helpRequested.connect(lambda: _open_plugin_help(dialog))
    layout.addWidget(buttons)

    def selected_plugin() -> dict[str, Any] | None:
        selected = table.selectedItems()
        if not selected:
            return plugins[0] if plugins else None
        row = selected[0].row()
        return plugins[row] if 0 <= row < len(plugins) else None

    def update_details() -> None:
        plugin = selected_plugin()
        if plugin is None:
            details.setHtml(_empty_details_html())
            guidance.setText("No extensions were discovered.")
            command_box.setPlainText("")
            copy_button.setEnabled(False)
            return
        details.setHtml(_details_html(plugin))
        guidance.setText(plugin.get("install_guidance") or "")
        commands = _command_text(plugin)
        command_box.setPlainText(commands)
        copy_button.setEnabled(bool(commands))

    def copy_commands() -> None:
        text = command_box.toPlainText().strip()
        if not text:
            return
        QtWidgets.QApplication.clipboard().setText(text)

    table.itemSelectionChanged.connect(update_details)
    copy_button.clicked.connect(copy_commands)
    if plugins:
        table.selectRow(0)
    update_details()
    return dialog


def _open_plugin_help(parent: Any) -> None:
    from eegprep.functions.guifunc.pophelp import pophelp

    parent._eegprep_extension_help_dialog = pophelp("plugin_menu", parent=parent)


def _details_html(plugin: dict[str, Any]) -> str:
    rows = [
        ("State", plugin.get("state_label", "")),
        ("Version", plugin.get("version") or plugin.get("catalog_version") or ""),
        ("Catalog version", plugin.get("catalog_version", "")),
        ("Package", plugin.get("package_name", "")),
        ("Maintainer", plugin.get("maintainer", "")),
        ("Source", plugin.get("source_detail", "")),
        ("Source URL", _link(plugin.get("source_url", ""))),
        ("Repository", _link(plugin.get("repository_url", ""))),
        ("Docs", _link(plugin.get("docs_url", ""))),
        ("Entry point", plugin.get("entry_point_name", "")),
        ("Menu", plugin.get("menu", "")),
        ("Capabilities", ", ".join(plugin.get("capabilities", ()))),
        ("Catalog status", plugin.get("catalog_status", "")),
    ]
    problem_blocks = []
    if plugin.get("errors"):
        problem_blocks.append(_list_block("Errors", plugin["errors"]))
    if plugin.get("catalog_conflicts"):
        problem_blocks.append(_list_block("Catalog metadata conflicts", plugin["catalog_conflicts"]))
    description = plugin.get("description") or "No description provided."
    table_rows = "\n".join(
        f"<tr><th>{html.escape(label)}</th><td>{value if label in {'Source URL', 'Repository', 'Docs'} else html.escape(str(value))}</td></tr>"
        for label, value in rows
        if value
    )
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
body {{ font-family: "Helvetica Neue", Segoe UI, Roboto, Helvetica, Arial, sans-serif; font-size: 12px; }}
h2 {{ margin: 0 0 8px 0; font-size: 17px; }}
p {{ margin: 0 0 10px 0; line-height: 1.35; }}
table {{ border-collapse: collapse; width: 100%; }}
th {{ text-align: left; color: #555; padding: 3px 10px 3px 0; white-space: nowrap; vertical-align: top; }}
td {{ padding: 3px 0; vertical-align: top; }}
.problem {{ margin-top: 10px; color: #7a1f12; }}
</style>
</head>
<body>
<h2>{html.escape(str(plugin.get("name") or plugin.get("plugin") or "Extension"))}</h2>
<p>{html.escape(str(description))}</p>
<table>{table_rows}</table>
{''.join(problem_blocks)}
</body>
</html>"""


def _empty_details_html() -> str:
    return """<!doctype html><html><body><h2>No extensions discovered</h2></body></html>"""


def _link(url: Any) -> str:
    text = str(url or "").strip()
    if not text:
        return ""
    escaped = html.escape(text)
    return f'<a href="{escaped}">{escaped}</a>'


def _list_block(title: str, values: Sequence[str]) -> str:
    items = "".join(f"<li>{html.escape(str(value))}</li>" for value in values)
    return f'<div class="problem"><b>{html.escape(title)}</b><ul>{items}</ul></div>'


def _command_text(plugin: dict[str, Any]) -> str:
    lines: list[str] = []
    install_commands = plugin.get("install_commands") or {}
    update_commands = plugin.get("update_commands") or {}
    if install_commands and not plugin.get("installed"):
        lines.append("Install")
        lines.extend(f"{name}: {command}" for name, command in install_commands.items())
    if update_commands and _can_show_update_commands(plugin):
        if lines:
            lines.append("")
        lines.append("Update")
        lines.extend(f"{name}: {command}" for name, command in update_commands.items())
    return "\n".join(lines)


def _can_show_update_commands(plugin: dict[str, Any]) -> bool:
    return bool(plugin.get("update_available") and plugin.get("active"))


def _catalog_notice(catalog_info: dict[str, Any]) -> str:
    source = catalog_info.get("catalog_source") or "no catalog"
    errors = catalog_info.get("catalog_errors") or ()
    if errors:
        return f"Catalog unavailable from {source}: {'; '.join(errors)}"
    return (
        f"Catalog source: {source}. "
        f"{catalog_info.get('catalog_entry_count', 0)} curated entr"
        f"{'y' if catalog_info.get('catalog_entry_count', 0) == 1 else 'ies'} loaded."
    )


def _install_guidance(
    status: str,
    installed: bool,
    active: bool,
    catalog_entry: ExtensionCatalogEntry | None,
    conflicts: tuple[str, ...],
) -> str:
    if conflicts:
        return "Catalog metadata differs from the installed extension. Review details before updating or reinstalling."
    if catalog_entry is None and installed:
        if active:
            return "This installed extension is not in the curated catalog. Manage it with the package source you used."
        return "This installed extension is not active. Review errors and manage it with the package source you used."
    if catalog_entry is None:
        return "No curated install guidance is available for this extension."
    if not installed:
        return "Install with a package manager, then restart EEGPrep so entry-point discovery can load it."
    if status in {
        ExtensionStatus.FAILED_IMPORT.value,
        ExtensionStatus.INVALID_SPEC.value,
        ExtensionStatus.MISSING_DEPENDENCY.value,
        ExtensionStatus.INCOMPATIBLE.value,
    }:
        return "Review the errors above. Updating or reinstalling may help after you verify the package source."
    if status == ExtensionStatus.DISABLED.value:
        return "This extension is installed but disabled. Re-enable it in registry configuration before use."
    return "This extension is installed. Use the update commands only after reviewing the curated metadata."


def _menu_text(record: ExtensionRecord) -> str:
    if record.spec is None or not record.spec.menus:
        return ""
    menu = record.spec.menus[0]
    return " > ".join(part for part in (*menu.path, menu.label) if part)


def _first_pop_function(record: ExtensionRecord) -> str:
    if record.spec is None or not record.spec.pop_functions:
        return ""
    return record.spec.pop_functions[0].name


def _folder_name(record: ExtensionRecord, package_name: str) -> str:
    if record.source_type == ExtensionSourceType.BUNDLED and package_name:
        return package_name.rsplit(".", 1)[-1]
    return package_name or record.name


def _source_detail(record: ExtensionRecord, catalog_entry: ExtensionCatalogEntry | None) -> str:
    if record.source_type == ExtensionSourceType.BUNDLED:
        return "Bundled with EEGPrep"
    if catalog_entry is not None:
        return f"{record.source_type.value}; curated as {catalog_entry.source_label}"
    if record.entry_point_name:
        return f"{record.source_type.value}; entry point {record.entry_point_name}"
    return record.source_type.value


def _is_active(plugin: dict[str, Any]) -> bool:
    status = str(plugin.get("status", ""))
    return bool(plugin.get("installed", status in _INSTALLED_STATUSES)) and status in _ACTIVE_STATUSES


def _catalog_version_is_newer(installed_version: str, catalog_version: str) -> bool:
    installed_parts = _version_parts(installed_version)
    catalog_parts = _version_parts(catalog_version)
    if not installed_parts or not catalog_parts:
        return False
    return _compare_version_parts(catalog_parts, installed_parts) > 0


def _compare_version_parts(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    max_length = max(len(left), len(right))
    padded_left = left + (0,) * (max_length - len(left))
    padded_right = right + (0,) * (max_length - len(right))
    if padded_left > padded_right:
        return 1
    if padded_left < padded_right:
        return -1
    return 0


def _version_parts(version: str) -> tuple[int, ...]:
    if not version or not re.search(r"\d", str(version)):
        return ()
    return tuple(int(part) for part in re.findall(r"\d+", str(version)))


def _plugin_sort_key(plugin: dict[str, Any]) -> tuple[int, int, str]:
    status_rank = {
        ExtensionStatus.BUNDLED.value: 0,
        ExtensionStatus.INSTALLED.value: 1,
        ExtensionStatus.DISABLED.value: 2,
        ExtensionStatus.INCOMPATIBLE.value: 3,
        ExtensionStatus.MISSING_DEPENDENCY.value: 4,
        ExtensionStatus.FAILED_IMPORT.value: 5,
        ExtensionStatus.INVALID_SPEC.value: 6,
        ExtensionStatus.CURATED.value: 7,
        ExtensionStatus.UNKNOWN.value: 8,
    }
    return (
        status_rank.get(str(plugin.get("status")), 9),
        int(plugin.get("_order", 9999)),
        _normalize_name(str(plugin.get("name") or "")),
    )


def _dedupe(values: Any) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    if values is None:
        return ()
    if isinstance(values, str):
        iterable = (values,)
    else:
        iterable = values
    for value in iterable:
        text = str(value).strip()
        if not text:
            continue
        key = _normalize_name(text)
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return tuple(result)


def _normalize_name(name: str) -> str:
    return str(name).strip().lower()


__all__ = [
    "EXTERNAL_PLUGIN_NOTICE",
    "bundled_plugins",
    "format_plugin_menu",
    "plugin_menu",
    "plugin_status",
]
