"""Declarative EEGLAB-style menu specifications."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


Visibility = str


@dataclass(frozen=True)
class MenuItemSpec:
    """A menu or menu action in the EEGLAB main window."""

    label: str
    action: str | None = None
    tag: str | None = None
    userdata: str = ""
    separator: bool = False
    enabled: bool = True
    checked: bool = False
    visibility: Visibility = "always"
    origin: str = "core"
    children: tuple["MenuItemSpec", ...] = field(default_factory=tuple)

    def with_children(self, children: tuple["MenuItemSpec", ...]) -> "MenuItemSpec":
        """Return a copy with replaced children."""
        return replace(self, children=children)


def menu_item(
    label: str,
    *,
    action: str | None = None,
    tag: str | None = None,
    userdata: str = "",
    separator: bool = False,
    enabled: bool = True,
    checked: bool = False,
    visibility: Visibility = "always",
    origin: str = "core",
    children: tuple[MenuItemSpec, ...] | list[MenuItemSpec] = (),
) -> MenuItemSpec:
    """Create a menu item using compact call sites."""
    return MenuItemSpec(
        label=label,
        action=action,
        tag=tag,
        userdata=userdata,
        separator=separator,
        enabled=enabled,
        checked=checked,
        visibility=visibility,
        origin=origin,
        children=tuple(children),
    )


def visible_menu_items(items: tuple[MenuItemSpec, ...], *, all_menus: bool) -> tuple[MenuItemSpec, ...]:
    """Filter menu items using EEGLAB default/full menu visibility rules."""
    visible: list[MenuItemSpec] = []
    for item in items:
        if item.visibility == "allmenus" and not all_menus:
            continue
        if item.visibility == "default" and all_menus:
            continue
        visible.append(item.with_children(visible_menu_items(item.children, all_menus=all_menus)))
    return tuple(visible)


def menu_enabled(item: MenuItemSpec, statuses: set[str]) -> bool:
    """Return whether a menu item should be enabled for the current session."""
    if not item.enabled:
        return False
    userdata = item.userdata or ""
    if "enable:off" in userdata:
        return False
    if "startup" in statuses and "startup:off" in userdata:
        return False
    if "study" in statuses:
        return "study:on" in userdata
    if "multiple_datasets" in statuses:
        if item.tag == "study":
            return False
        return "study:on" in userdata
    if "epoched_dataset" in statuses and "epoch:off" in userdata:
        return False
    if "continuous_dataset" in statuses and "continuous:off" in userdata:
        return False
    if "chanloc_absent" in statuses and "chanloc:on" in userdata:
        return False
    if "ica_absent" in statuses and "ica:on" in userdata:
        return False
    if "roi_connect" in statuses and "roi:off" in userdata:
        return False
    return True


def menu_to_inventory(items: tuple[MenuItemSpec, ...], statuses: set[str] | None = None) -> list[dict[str, Any]]:
    """Convert menu specs to the JSON tree used by visual parity tooling."""
    statuses = statuses or {"startup"}
    return [
        {
            "label": item.label,
            "enabled": menu_enabled(item, statuses),
            "separator": item.separator,
            "checked": item.checked,
            "tag": item.tag or "",
            "children": menu_to_inventory(item.children, statuses),
        }
        for item in items
    ]
