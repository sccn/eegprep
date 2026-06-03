"""Tests for bundled plugin inventory and manager behavior."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from eegprep.extensions import ExtensionRecord, ExtensionSourceType, ExtensionSpec, ExtensionStatus
from eegprep.functions.adminfunc.extension_catalog import (
    INSTALL_TRUST_WARNING,
    ExtensionCatalog,
    ExtensionCatalogEntry,
    load_extension_catalog,
)
from eegprep.functions.adminfunc.plugin_menu import (
    EXTERNAL_PLUGIN_NOTICE,
    _build_plugin_dialog,
    _command_text,
    bundled_plugins,
    format_plugin_menu,
    plugin_menu,
    plugin_status,
)
from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher
from eegprep.functions.guifunc.pophelp import pophelp_text
from eegprep.functions.guifunc.session import EEGPrepSession


def test_bundled_plugins_describe_in_repo_extensions() -> None:
    plugins = bundled_plugins()
    names = [plugin["plugin"] for plugin in plugins]

    assert names == ["clean_rawdata", "ICLabel", "firfilt", "dipfit", "EEG_BIDS"]
    assert all(plugin["installed"] is True for plugin in plugins)
    assert all(plugin["status"] == "ok" for plugin in plugins)
    assert all(plugin["source"] == "bundled" for plugin in plugins)
    assert all(plugin["menu"] for plugin in plugins)


def test_bundled_plugins_returns_copies() -> None:
    plugins = bundled_plugins()
    plugins[0]["status"] = "changed"

    assert bundled_plugins()[0]["status"] == "ok"


def test_plugin_status_supports_partial_and_exact_matches() -> None:
    partial_status, partial_names, partial_struct = plugin_status("label")
    exact_status, exact_names, exact_struct = plugin_status("ICLabel", exactmatch=True)
    missing_status, missing_names, missing_struct = plugin_status("external")

    assert partial_status == [1]
    assert partial_names == ["ICLabel"]
    assert partial_struct[0]["name"] == "ICLabel"
    assert exact_status == [1]
    assert exact_names == ["ICLabel"]
    assert exact_struct[0]["foldername"] == "ICLabel"
    assert missing_status == []
    assert missing_names == []
    assert missing_struct == []


def test_plugin_menu_updates_session_without_gui() -> None:
    session = EEGPrepSession()
    plugins = plugin_menu(session=session, show=False)

    assert session.PLUGINLIST == plugins
    assert [plugin["plugin"] for plugin in session.PLUGINLIST] == [
        "clean_rawdata",
        "ICLabel",
        "firfilt",
        "dipfit",
        "EEG_BIDS",
    ]
    assert [plugin["status"] for plugin in session.PLUGINLIST] == ["bundled"] * 5


def test_format_plugin_menu_includes_external_plugin_exclusion() -> None:
    text = format_plugin_menu()

    assert "Available EEGPrep extensions" in text
    assert "ICLabel" in text
    assert "File > Import data / Export / BIDS tools" in text
    assert EXTERNAL_PLUGIN_NOTICE in text
    assert INSTALL_TRUST_WARNING in text


def test_file_menu_plugin_action_uses_bundled_inventory_headlessly() -> None:
    session = EEGPrepSession()
    dispatcher = MenuActionDispatcher(session)

    dispatcher.dispatch("plugin_menu", parent=None)

    assert [plugin["plugin"] for plugin in session.PLUGINLIST] == [
        "clean_rawdata",
        "ICLabel",
        "firfilt",
        "dipfit",
        "EEG_BIDS",
    ]


def test_plugin_menu_help_resource_is_packaged() -> None:
    text, source_path = pophelp_text("plugin_menu")

    assert "EEGPrep Extension Manager" in text
    assert source_path == "eegprep/resources/help/plugin_menu.md"


def test_catalog_only_extension_gets_safe_install_guidance() -> None:
    catalog = _catalog(
        ExtensionCatalogEntry(
            name="asr_tools",
            display_name="ASR Tools",
            version="1.2.0",
            package_name="eegprep-ext-asr-tools",
            description="Additional ASR reports.",
            maintainer="SCCN",
            docs_url="https://example.org/asr-tools",
            capabilities=("artifact", "report"),
        )
    )

    plugins = plugin_menu(registry=_registry(), catalog=catalog, include_bundled=False, show=False)

    assert len(plugins) == 1
    plugin = plugins[0]
    assert plugin["status"] == "curated"
    assert plugin["installed"] is False
    assert plugin["catalog_status"] == "catalog_only"
    assert plugin["install_commands"]["uv"] == "uv add eegprep-ext-asr-tools"
    assert plugin["install_commands"]["pip"] == "pip install eegprep-ext-asr-tools"
    assert plugin["trust_warning"] == INSTALL_TRUST_WARNING
    status, names, structs = plugin_status("asr", registry=_registry(), catalog=catalog, include_bundled=False)
    assert status == [0]
    assert names == ["asr_tools"]
    assert structs[0]["install_commands"]["uv"] == "uv add eegprep-ext-asr-tools"


def test_installed_only_private_extension_stays_visible_without_catalog() -> None:
    record = _record(
        "private_lab_extension",
        package_name="/Users/lab/src/private_lab_extension",
        spec_kwargs={
            "display_name": "Private Lab Extension",
            "description": "Local lab QA tools.",
            "maintainer": "Local EEG lab",
        },
    )

    plugin = plugin_menu(registry=_registry(record), catalog=_catalog(), include_bundled=False, show=False)[0]

    assert plugin["status"] == "installed"
    assert plugin["installed"] is True
    assert plugin["active"] is True
    assert plugin["curated"] is False
    assert plugin["catalog_status"] == "not_curated"
    assert plugin["install_commands"] == {}
    assert "not in the curated catalog" in plugin["install_guidance"]


def test_installed_plus_catalog_entry_merges_metadata_and_update_guidance() -> None:
    record = _record(
        "erp_reports",
        package_name="eegprep-ext-erp-reports",
        spec_kwargs={
            "display_name": "ERP Reports",
            "version": "1.0.0",
            "description": "Installed description.",
            "capabilities": ("erp",),
        },
    )
    catalog = _catalog(
        ExtensionCatalogEntry(
            name="erp_reports",
            display_name="ERP Reports",
            version="1.1.0",
            package_name="eegprep-ext-erp-reports",
            maintainer="SCCN",
            docs_url="https://example.org/erp-reports",
            capabilities=("report",),
        )
    )

    plugin = plugin_menu(registry=_registry(record), catalog=catalog, include_bundled=False, show=False)[0]

    assert plugin["status"] == "installed"
    assert plugin["catalog_status"] == "curated"
    assert plugin["version"] == "1.0.0"
    assert plugin["catalog_version"] == "1.1.0"
    assert plugin["update_available"] is True
    assert plugin["maintainer"] == "SCCN"
    assert plugin["docs_url"] == "https://example.org/erp-reports"
    assert plugin["capabilities"] == ("erp", "report")
    assert plugin["update_commands"]["pip"] == "pip install --upgrade eegprep-ext-erp-reports"


def test_update_available_uses_version_order_not_raw_difference() -> None:
    older_catalog = _catalog(
        ExtensionCatalogEntry(
            name="versioned_extension",
            version="1.9.0",
            package_name="eegprep-ext-versioned",
        )
    )
    newer_catalog = _catalog(
        ExtensionCatalogEntry(
            name="versioned_extension",
            version="1.10.0",
            package_name="eegprep-ext-versioned",
        )
    )
    equal_catalog = _catalog(
        ExtensionCatalogEntry(
            name="versioned_extension",
            version="1.10.0",
            package_name="eegprep-ext-versioned",
        )
    )

    installed_newer = _record(
        "versioned_extension",
        package_name="eegprep-ext-versioned",
        spec_kwargs={"version": "1.10.0"},
    )
    installed_older = _record(
        "versioned_extension",
        package_name="eegprep-ext-versioned",
        spec_kwargs={"version": "1.9.0"},
    )

    downgrade_plugin = plugin_menu(
        registry=_registry(installed_newer), catalog=older_catalog, include_bundled=False, show=False
    )[0]
    update_plugin = plugin_menu(
        registry=_registry(installed_older), catalog=newer_catalog, include_bundled=False, show=False
    )[0]
    equal_plugin = plugin_menu(
        registry=_registry(installed_newer), catalog=equal_catalog, include_bundled=False, show=False
    )[0]

    assert downgrade_plugin["update_available"] is False
    assert "Update" not in _command_text(downgrade_plugin)
    assert update_plugin["update_available"] is True
    assert "Update" in _command_text(update_plugin)
    assert equal_plugin["update_available"] is False


def test_problem_states_do_not_show_update_commands_before_errors_are_resolved() -> None:
    record = _record(
        "incompatible_extension",
        status=ExtensionStatus.INCOMPATIBLE,
        package_name="eegprep-ext-incompatible",
        errors=("Extension requires EEGPrep >=999.0; current version is 0.2.23",),
        spec_kwargs={"version": "1.0.0"},
    )
    catalog = _catalog(
        ExtensionCatalogEntry(
            name="incompatible_extension",
            version="2.0.0",
            package_name="eegprep-ext-incompatible",
        )
    )

    plugin = plugin_menu(registry=_registry(record), catalog=catalog, include_bundled=False, show=False)[0]
    text = format_plugin_menu(registry=_registry(record), catalog=catalog, include_bundled=False)

    assert plugin["update_available"] is True
    assert plugin["active"] is False
    assert "Update" not in _command_text(plugin)
    assert "  Update:" not in text


def test_problem_registry_states_are_reported_headlessly() -> None:
    records = (
        _record("disabled_extension", status=ExtensionStatus.DISABLED, enabled=False),
        _record(
            "incompatible_extension",
            status=ExtensionStatus.INCOMPATIBLE,
            errors=("Extension requires EEGPrep >=999.0; current version is 0.2.23",),
        ),
        _record(
            "missing_dependency_extension",
            status=ExtensionStatus.MISSING_DEPENDENCY,
            errors=("Required dependency 'foo' is not installed",),
        ),
        ExtensionRecord(
            name="failed_extension",
            status=ExtensionStatus.FAILED_IMPORT,
            spec=None,
            source_type=ExtensionSourceType.UNKNOWN,
            package_name="eegprep-ext-failed",
            entry_point_name="failed",
            enabled=True,
            errors=("Entry point 'failed' failed to import: boom",),
        ),
    )
    catalog = _catalog(
        ExtensionCatalogEntry(
            name="failed_extension",
            display_name="Failed Extension",
            version="0.1.0",
            package_name="eegprep-ext-failed",
            description="Catalog metadata remains visible.",
        )
    )

    plugins = plugin_menu(registry=_registry(*records), catalog=catalog, include_bundled=False, show=False)
    by_plugin = {plugin["plugin"]: plugin for plugin in plugins}

    assert by_plugin["disabled_extension"]["state_label"] == "Disabled"
    assert by_plugin["incompatible_extension"]["state_label"] == "Incompatible"
    assert by_plugin["missing_dependency_extension"]["state_label"] == "Missing dependency"
    assert by_plugin["failed_extension"]["state_label"] == "Failed"
    assert by_plugin["failed_extension"]["description"] == "Catalog metadata remains visible."
    assert by_plugin["failed_extension"]["install_commands"]["uv"] == "uv add eegprep-ext-failed"
    status, names, _structs = plugin_status(
        "extension", registry=_registry(*records), catalog=catalog, include_bundled=False
    )
    assert names == [
        "disabled_extension",
        "incompatible_extension",
        "missing_dependency_extension",
        "failed_extension",
    ]
    assert status == [0, 0, 0, 0]


def test_catalog_metadata_conflicts_are_preserved() -> None:
    record = _record(
        "conflicting_extension",
        package_name="eegprep-ext-installed",
        spec_kwargs={
            "display_name": "Installed Name",
            "docs_url": "https://installed.example/docs",
        },
    )
    catalog = _catalog(
        ExtensionCatalogEntry(
            name="conflicting_extension",
            display_name="Catalog Name",
            package_name="eegprep-ext-catalog",
            docs_url="https://catalog.example/docs",
        )
    )

    plugin = plugin_menu(registry=_registry(record), catalog=catalog, include_bundled=False, show=False)[0]

    assert plugin["catalog_status"] == "conflict"
    assert len(plugin["catalog_conflicts"]) == 3
    assert "eegprep-ext-catalog" in plugin["catalog_conflicts"][0]
    assert "Review details before updating" in plugin["install_guidance"]


def test_catalog_unavailable_path_does_not_hide_installed_extensions() -> None:
    record = _record("installed_extension", package_name="eegprep-ext-installed")
    catalog = ExtensionCatalog(source="/tmp/missing-catalog.json", errors=("catalog unavailable",))

    plugin = plugin_menu(registry=_registry(record), catalog=catalog, include_bundled=False, show=False)[0]
    text = format_plugin_menu(registry=_registry(record), catalog=catalog, include_bundled=False)

    assert plugin["catalog_status"] == "catalog_unavailable"
    assert plugin["catalog_errors"] == ("catalog unavailable",)
    assert "Catalog warning: catalog unavailable" in text
    assert "installed_extension" in text


def test_local_catalog_json_supports_git_and_rejects_archives(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        """
        {
          "schema_version": 1,
          "extensions": [
            {
              "name": "git_extension",
              "display_name": "Git Extension",
              "package_name": "eegprep-ext-git",
              "version": "0.3.0",
              "source": {"type": "git", "url": "https://github.com/sccn/eegprep-ext-git.git"},
              "docs_url": "https://example.org/git-extension",
              "capabilities": ["qa"]
            },
            {
              "name": "zip_extension",
              "package_name": "eegprep-ext-zip",
              "source": {"type": "git", "url": "https://example.org/ext.zip"}
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    catalog = load_extension_catalog(catalog_path)

    assert [entry.name for entry in catalog.entries] == ["git_extension"]
    assert "not an archive" in catalog.errors[0]
    assert plugin_menu(registry=_registry(), catalog=catalog, include_bundled=False, show=False)[0][
        "install_commands"
    ] == {
        "uv": "uv add git+https://github.com/sccn/eegprep-ext-git.git",
        "pip": "pip install git+https://github.com/sccn/eegprep-ext-git.git",
    }


def test_catalog_rejects_local_archive_paths(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        """
        {
          "schema_version": 1,
          "extensions": [
            {
              "name": "local_wheel_extension",
              "source": {"type": "local", "url": "/tmp/eegprep_ext_local-1.0.0-py3-none-any.whl"}
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    catalog = load_extension_catalog(catalog_path)

    assert catalog.entries == ()
    assert "source.url must point to metadata, docs, or a repository, not an archive" in catalog.errors[0]


def test_environment_catalog_path_is_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        '{"schema_version": 1, "extensions": [{"name": "env_extension", "package_name": "eegprep-ext-env"}]}',
        encoding="utf-8",
    )
    monkeypatch.setenv("EEGPREP_EXTENSION_CATALOG", str(catalog_path))

    plugins = plugin_menu(registry=_registry(), include_bundled=False, show=False)

    assert [plugin["plugin"] for plugin in plugins] == ["env_extension"]
    assert plugins[0]["install_commands"]["uv"] == "uv add eegprep-ext-env"


def test_gui_extension_manager_dialog_renders_details_and_commands() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    catalog = _catalog(
        ExtensionCatalogEntry(
            name="catalog_extension",
            display_name="Catalog Extension",
            package_name="eegprep-ext-catalog",
            description="Catalog-only EEGPrep tooling.",
            maintainer="SCCN",
            docs_url="https://example.org/catalog-extension",
            capabilities=("visual",),
        )
    )
    plugins = plugin_menu(registry=_registry(), catalog=catalog, include_bundled=False, show=False)
    catalog_info = {
        "catalog_source": catalog.source,
        "catalog_available": catalog.available,
        "catalog_entry_count": len(catalog.entries),
        "catalog_errors": catalog.errors,
    }

    dialog = _build_plugin_dialog(plugins, catalog_info=catalog_info)
    app.processEvents()
    table = dialog.findChild(qt_widgets.QTableWidget, "extension_table")
    details = dialog.findChild(qt_widgets.QTextBrowser, "extension_details")
    commands = dialog.findChild(qt_widgets.QPlainTextEdit, "extension_install_commands")
    copy_button = dialog.findChild(qt_widgets.QPushButton, "copy_extension_commands")

    assert table is not None
    assert details is not None
    assert commands is not None
    assert copy_button is not None
    assert table.rowCount() == 1
    assert "Catalog Extension" in details.toPlainText()
    assert "Catalog-only EEGPrep tooling." in details.toPlainText()
    assert "uv add eegprep-ext-catalog" in commands.toPlainText()
    copy_button.click()
    assert "uv add eegprep-ext-catalog" in qt_widgets.QApplication.clipboard().text()
    dialog.close()


class _Registry:
    def __init__(self, *records: ExtensionRecord) -> None:
        self.records = tuple(records)

    def discover(self) -> tuple[ExtensionRecord, ...]:
        return self.records


def _registry(*records: ExtensionRecord) -> _Registry:
    return _Registry(*records)


def _catalog(*entries: ExtensionCatalogEntry) -> ExtensionCatalog:
    return ExtensionCatalog(entries=tuple(entries), source="test catalog")


def _record(
    name: str,
    *,
    status: ExtensionStatus = ExtensionStatus.INSTALLED,
    source_type: ExtensionSourceType = ExtensionSourceType.INSTALLED,
    package_name: str = "",
    enabled: bool = True,
    errors: tuple[str, ...] = (),
    spec_kwargs: dict[str, Any] | None = None,
) -> ExtensionRecord:
    spec_kwargs = dict(spec_kwargs or {})
    package_name = package_name or spec_kwargs.get("package_name") or f"eegprep-ext-{name.replace('_', '-')}"
    spec = ExtensionSpec(
        name=name,
        display_name=spec_kwargs.pop("display_name", name),
        version=spec_kwargs.pop("version", "1.0.0"),
        package_name=package_name,
        source_type=source_type,
        description=spec_kwargs.pop("description", ""),
        docs_url=spec_kwargs.pop("docs_url", ""),
        maintainer=spec_kwargs.pop("maintainer", ""),
        capabilities=spec_kwargs.pop("capabilities", ()),
        **spec_kwargs,
    )
    return ExtensionRecord(
        name=name,
        status=status,
        spec=spec,
        source_type=source_type,
        package_name=package_name,
        entry_point_name=name,
        enabled=enabled,
        errors=errors,
    )
