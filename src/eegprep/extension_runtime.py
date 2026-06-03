"""Runtime adapters for registered EEGPrep extensions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Mapping

from eegprep.extensions import (
    ExtensionAction,
    ExtensionMenu,
    ExtensionPopFunction,
    ExtensionRecord,
    ExtensionRegistry,
    ExtensionResource,
)
from eegprep.functions.guifunc.menu_spec import MenuItemSpec, menu_item

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtensionMenuContribution:
    """A resolved extension menu item plus its insertion target."""

    owner: str
    path: tuple[str, ...]
    item: MenuItemSpec
    insert_after: str = ""
    insert_before: str = ""


@dataclass(frozen=True)
class ExtensionRuntime:
    """Active extension contributions prepared for GUI, help, and console use."""

    records: tuple[ExtensionRecord, ...] = ()
    actions: Mapping[str, ExtensionAction] | None = None
    pop_functions: Mapping[str, ExtensionPopFunction] | None = None
    help_resources: Mapping[str, ExtensionResource] | None = None
    menu_contributions: tuple[ExtensionMenuContribution, ...] = ()

    @classmethod
    def discover(
        cls,
        *,
        include_plugins: bool = True,
        registry: ExtensionRegistry | None = None,
    ) -> "ExtensionRuntime":
        """Discover active extensions and prepare their runtime contributions."""
        if not include_plugins:
            return cls.empty()
        registry = registry or ExtensionRegistry()
        return cls.from_records(registry.discover(include_plugins=True))

    @classmethod
    def from_records(cls, records: tuple[ExtensionRecord, ...]) -> "ExtensionRuntime":
        """Build runtime maps from registry records."""
        actions: dict[str, ExtensionAction] = {}
        pop_functions: dict[str, ExtensionPopFunction] = {}
        help_resources: dict[str, ExtensionResource] = {}
        menu_contributions: list[ExtensionMenuContribution] = []

        for record in records:
            if not record.is_active or record.spec is None:
                continue
            origin = f"extension:{record.name}"
            for action in record.spec.actions:
                actions[_key(action.name)] = action
            for pop_function in record.spec.pop_functions:
                pop_functions[_key(pop_function.name)] = pop_function
            for resource in record.spec.help_resources:
                stem = _resource_stem(resource)
                if stem in help_resources:
                    logger.warning(
                        "EEGPrep extension %s help resource %s duplicates an earlier help topic",
                        record.name,
                        resource.path,
                    )
                    continue
                help_resources[stem] = resource
            for menu in record.spec.menus:
                menu_contributions.append(_menu_contribution(record.name, origin, menu))

        return cls(
            records=records,
            actions=actions,
            pop_functions=pop_functions,
            help_resources=help_resources,
            menu_contributions=tuple(menu_contributions),
        )

    @classmethod
    def empty(cls) -> "ExtensionRuntime":
        """Return a runtime with no active extension contributions."""
        return cls(actions={}, pop_functions={}, help_resources={})

    def action(self, name: str) -> ExtensionAction | ExtensionPopFunction | None:
        """Return the registered action or pop callable descriptor for ``name``."""
        key = _key(name)
        actions = self.actions or {}
        pop_functions = self.pop_functions or {}
        return actions.get(key) or pop_functions.get(key)

    def pop_function(self, name: str) -> ExtensionPopFunction | None:
        """Return a registered extension ``pop_*`` descriptor."""
        return (self.pop_functions or {}).get(_key(name))

    def pop_function_resolvers(self) -> dict[str, Callable[[], object]]:
        """Return lazy callables keyed by extension ``pop_*`` name."""
        return {pop_function.name: pop_function.load for pop_function in (self.pop_functions or {}).values()}

    def help_resource(self, function_name: str) -> ExtensionResource | None:
        """Return an extension help resource for ``function_name`` if one exists."""
        return (self.help_resources or {}).get(_key(function_name))

    def has_action(self, name: str) -> bool:
        """Return whether an action is registered by an active extension."""
        return self.action(name) is not None


def apply_extension_menus(
    menus: tuple[MenuItemSpec, ...],
    runtime: ExtensionRuntime,
) -> tuple[MenuItemSpec, ...]:
    """Insert registered extension menu items into the core menu tree."""
    output = menus
    for contribution in runtime.menu_contributions:
        output, inserted = _insert_contribution(output, contribution)
        if not inserted:
            logger.warning(
                "EEGPrep extension %s menu path %s was not found for %s",
                contribution.owner,
                " > ".join(contribution.path),
                contribution.item.label,
            )
    return output


def _menu_contribution(owner: str, origin: str, menu: ExtensionMenu) -> ExtensionMenuContribution:
    return ExtensionMenuContribution(
        owner=owner,
        path=tuple(str(part) for part in menu.path),
        item=_menu_item_from_extension_menu(menu, origin),
        insert_after=menu.insert_after,
        insert_before=menu.insert_before,
    )


def _menu_item_from_extension_menu(menu: ExtensionMenu, origin: str) -> MenuItemSpec:
    label = menu.label or menu.action
    return menu_item(
        label,
        action=menu.action or None,
        tag=menu.tag or None,
        userdata=menu.userdata,
        separator=menu.separator,
        enabled=menu.enabled,
        checked=menu.checked,
        visibility=menu.visibility,
        origin=origin,
        children=tuple(_menu_item_from_extension_menu(child, origin) for child in menu.children),
    )


def _insert_contribution(
    items: tuple[MenuItemSpec, ...],
    contribution: ExtensionMenuContribution,
) -> tuple[tuple[MenuItemSpec, ...], bool]:
    return _insert_at_path(items, contribution.path, contribution)


def _insert_at_path(
    items: tuple[MenuItemSpec, ...],
    path: tuple[str, ...],
    contribution: ExtensionMenuContribution,
) -> tuple[tuple[MenuItemSpec, ...], bool]:
    if not path:
        return _insert_child(items, contribution), True
    updated: list[MenuItemSpec] = []
    inserted = False
    for item in items:
        if not inserted and _matches_menu_token(item, path[0]):
            if len(path) == 1:
                children = _insert_child(item.children, contribution)
                updated.append(item.with_children(children))
                inserted = True
                continue
            children, inserted = _insert_at_path(item.children, path[1:], contribution)
            updated.append(item.with_children(children) if inserted else item)
            continue
        updated.append(item)
    return tuple(updated), inserted


def _insert_child(
    children: tuple[MenuItemSpec, ...],
    contribution: ExtensionMenuContribution,
) -> tuple[MenuItemSpec, ...]:
    item = contribution.item
    if any(_key(child.label) == _key(item.label) for child in children):
        logger.warning(
            "EEGPrep extension %s menu %s duplicates an existing item",
            contribution.owner,
            item.label,
        )
        return children
    output = list(children)
    if contribution.insert_before:
        index = _anchor_index(children, contribution.insert_before)
        if index is not None:
            output.insert(index, item)
            return tuple(output)
        _log_missing_anchor(contribution)
    if contribution.insert_after:
        index = _anchor_index(children, contribution.insert_after)
        if index is not None:
            output.insert(index + 1, item)
            return tuple(output)
        _log_missing_anchor(contribution)
    output.append(item)
    return tuple(output)


def _anchor_index(children: tuple[MenuItemSpec, ...], anchor: str) -> int | None:
    for index, child in enumerate(children):
        if _matches_menu_token(child, anchor):
            return index
    return None


def _matches_menu_token(item: MenuItemSpec, token: str) -> bool:
    normalized = _key(token)
    return normalized in {
        _key(item.label),
        _key(item.tag or ""),
        _key(item.action or ""),
    }


def _log_missing_anchor(contribution: ExtensionMenuContribution) -> None:
    anchor = contribution.insert_before or contribution.insert_after
    logger.warning(
        "EEGPrep extension %s menu anchor %s was not found under %s for %s",
        contribution.owner,
        anchor,
        " > ".join(contribution.path),
        contribution.item.label,
    )


def _resource_stem(resource: ExtensionResource) -> str:
    name = resource.path.rsplit("/", 1)[-1]
    return _key(name.rsplit(".", 1)[0])


def _key(value: str) -> str:
    return str(value).strip().lower()


__all__ = [
    "ExtensionMenuContribution",
    "ExtensionRuntime",
    "apply_extension_menus",
]
