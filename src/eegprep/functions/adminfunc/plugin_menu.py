"""Bundled EEGPrep extension inventory and manager dialog."""

from __future__ import annotations

import importlib
from copy import deepcopy
from typing import Any

from eegprep.extensions import ExtensionRecord, ExtensionRegistry, ExtensionStatus

EXTERNAL_PLUGIN_NOTICE = (
    "EEGPrep currently manages only extensions bundled inside this Python package. "
    "External EEGLAB plugin repositories are not installed, updated, or removed from this dialog."
)

_MENU_SUMMARIES = {
    "clean_rawdata": "Tools > Reject data using Clean Rawdata and ASR",
    "ICLabel": "Tools > Classify components using ICLabel",
    "firfilt": "Tools > Filter the data",
    "dipfit": "Tools > Source localization using DIPFIT",
    "EEG_BIDS": "File > Import data / Export / BIDS tools",
}


def bundled_plugins() -> tuple[dict[str, Any], ...]:
    """Return metadata for EEGPrep extensions bundled in the installed package."""
    records = ExtensionRegistry(include_entry_points=False).discover()
    return tuple(_plugin_from_record(record) for record in records)


def _plugin_from_record(record: ExtensionRecord) -> dict[str, Any]:
    spec = record.spec
    name = record.name if spec is None else spec.name
    display_name = name if spec is None else spec.display_name or spec.name
    package_name = "" if spec is None else spec.package_name
    foldername = package_name.rsplit(".", 1)[-1] if package_name else name
    funcname = "" if spec is None or not spec.pop_functions else spec.pop_functions[0].name
    source_type = record.source_type.value if hasattr(record.source_type, "value") else str(record.source_type)
    return {
        "plugin": name,
        "name": display_name,
        "version": "bundled" if spec is None else spec.version or "bundled",
        "foldername": foldername,
        "funcname": funcname,
        "status": "ok" if record.status == ExtensionStatus.BUNDLED else record.status.value,
        "installed": record.is_active,
        "source": source_type,
        "menu": _MENU_SUMMARIES.get(name, _menu_summary(record)),
        "description": "" if spec is None else spec.description,
        "tags": () if spec is None else tuple(spec.capabilities),
    }


def _menu_summary(record: ExtensionRecord) -> str:
    spec = record.spec
    if spec is None or not spec.menus:
        return ""
    menu = spec.menus[0]
    label = menu.label or menu.action
    return " > ".join((*menu.path, label))


def plugin_status(
    pluginname: str,
    *,
    exactmatch: bool = False,
    pluginlist: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
) -> tuple[list[int], list[str], list[dict[str, Any]]]:
    """Return EEGLAB-style installed status for bundled EEGPrep plugins.

    Args:
        pluginname: Plugin name or substring to search.
        exactmatch: Require exact case-insensitive name matching.
        pluginlist: Optional plugin inventory. Defaults to bundled plugins.

    Returns:
        A tuple ``(status, names, pluginstruct)`` where status values are ``1``
        for available bundled plugins and ``0`` for unavailable matches.
    """
    needle = pluginname.lower()
    matches: list[dict[str, Any]] = []
    for plugin in _normalized_plugins(pluginlist):
        names = {str(plugin.get("plugin", "")), str(plugin.get("name", "")), str(plugin.get("foldername", ""))}
        lowered = {name.lower() for name in names if name}
        if exactmatch and needle not in lowered:
            continue
        if not exactmatch and not any(needle in name for name in lowered):
            continue
        matches.append(plugin)
    return (
        [1 if plugin.get("status") == "ok" else 0 for plugin in matches],
        [str(plugin.get("plugin") or plugin.get("name") or "") for plugin in matches],
        [deepcopy(plugin) for plugin in matches],
    )


def plugin_menu(
    pluginlist: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    *,
    parent: Any | None = None,
    session: Any | None = None,
    show: bool = True,
) -> list[dict[str, Any]]:
    """Show or return the read-only EEGPrep bundled extension manager.

    Args:
        pluginlist: Optional plugin inventory to display. Defaults to bundled
            in-repo EEGPrep extensions.
        parent: Optional Qt parent widget for the dialog.
        session: Optional :class:`EEGPrepSession`; its ``PLUGINLIST`` mirror is
            updated with the displayed inventory.
        show: Show the Qt dialog when ``True``. Use ``False`` for headless
            Python, examples, tests, or console inventory checks.

    Returns:
        The normalized plugin inventory as a mutable list of dictionaries.
    """
    plugins = [deepcopy(plugin) for plugin in _normalized_plugins(pluginlist)]
    if session is not None:
        session.PLUGINLIST = [deepcopy(plugin) for plugin in plugins]
    if show:
        _show_plugin_dialog(plugins, parent)
    return plugins


def format_plugin_menu(
    pluginlist: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
) -> str:
    """Return a plain-text bundled plugin inventory for console display."""
    lines = ["Available EEGPrep extensions", ""]
    for plugin in _normalized_plugins(pluginlist):
        version = str(plugin.get("version") or "bundled")
        status = "installed" if plugin.get("installed", False) else "unavailable"
        lines.append(f"- {plugin['name']} v{version} ({status})")
        lines.append(f"  {plugin['description']}")
        lines.append(f"  Menu: {plugin['menu']}")
    lines.extend(["", EXTERNAL_PLUGIN_NOTICE])
    return "\n".join(lines)


def _normalized_plugins(
    pluginlist: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
) -> tuple[dict[str, Any], ...]:
    source = bundled_plugins() if pluginlist is None else tuple(deepcopy(plugin) for plugin in pluginlist)
    return tuple(_normalize_plugin(plugin) for plugin in source)


def _normalize_plugin(plugin: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(plugin)
    name = str(normalized.get("name") or normalized.get("plugin") or normalized.get("foldername") or "")
    if not name:
        raise ValueError("Plugin entries must include a name, plugin, or foldername")
    normalized.setdefault("plugin", name)
    normalized.setdefault("name", name)
    normalized.setdefault("version", "bundled")
    normalized.setdefault("foldername", normalized["plugin"])
    normalized.setdefault("funcname", "")
    normalized.setdefault("status", "ok" if normalized.get("installed", True) else "unavailable")
    normalized.setdefault("installed", normalized.get("status") == "ok")
    normalized.setdefault("source", "bundled")
    normalized.setdefault("menu", "")
    normalized.setdefault("description", "")
    normalized.setdefault("tags", ())
    return normalized


def _show_plugin_dialog(plugins: list[dict[str, Any]], parent: Any | None) -> None:
    try:
        QtCore = importlib.import_module("PySide6.QtCore")
        QtWidgets = importlib.import_module("PySide6.QtWidgets")
    except ImportError as exc:  # pragma: no cover - optional GUI dependency
        raise RuntimeError(
            "PySide6 is required to show the EEGPrep extension manager. "
            "Call plugin_menu(show=False) to inspect bundled plugins without a GUI."
        ) from exc

    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Manage EEGPrep extensions")
    layout = QtWidgets.QVBoxLayout(dialog)

    title = QtWidgets.QLabel("List of bundled plugins")
    font = title.font()
    font.setBold(True)
    title.setFont(font)
    layout.addWidget(title)

    table = QtWidgets.QTableWidget(len(plugins), 5, dialog)
    table.setHorizontalHeaderLabels(["Plugin", "Version", "Status", "Menu", "Description"])
    table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
    table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setWordWrap(False)
    for row, plugin in enumerate(plugins):
        values = (
            plugin["name"],
            plugin["version"],
            "Installed" if plugin["installed"] else "Unavailable",
            plugin["menu"],
            plugin["description"],
        )
        for column, value in enumerate(values):
            item = QtWidgets.QTableWidgetItem(str(value))
            if column == 0 and plugin["installed"]:
                item_font = item.font()
                item_font.setBold(True)
                item.setFont(item_font)
            table.setItem(row, column, item)
    table.verticalHeader().setDefaultSectionSize(30)
    header = table.horizontalHeader()
    header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
    header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
    header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
    header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
    header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Stretch)
    layout.addWidget(table)

    notice = QtWidgets.QLabel(EXTERNAL_PLUGIN_NOTICE)
    notice.setWordWrap(True)
    notice.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    layout.addWidget(notice)

    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    dialog.resize(1000, 440)
    dialog.exec()


__all__ = [
    "EXTERNAL_PLUGIN_NOTICE",
    "bundled_plugins",
    "format_plugin_menu",
    "plugin_menu",
    "plugin_status",
]
