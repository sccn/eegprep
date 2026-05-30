"""Tests for bundled plugin inventory and manager behavior."""

from __future__ import annotations

from eegprep.functions.adminfunc.plugin_menu import (
    EXTERNAL_PLUGIN_NOTICE,
    bundled_plugins,
    format_plugin_menu,
    plugin_menu,
    plugin_status,
)
from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher
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


def test_format_plugin_menu_includes_external_plugin_exclusion() -> None:
    text = format_plugin_menu()

    assert "Available EEGPrep extensions" in text
    assert "ICLabel" in text
    assert "File > Import data / Export / BIDS tools" in text
    assert EXTERNAL_PLUGIN_NOTICE in text


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
