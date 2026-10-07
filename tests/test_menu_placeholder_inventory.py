from __future__ import annotations

from eegprep.functions.guifunc.eeglab_menu import eeglab_menus, menu_actions
from eegprep.functions.guifunc.menu_actions import IMPLEMENTED_ACTIONS, action_kind
from eegprep.functions.guifunc.menu_placeholders import (
    PLACEHOLDER_ACTIONS,
    placeholder_inventory,
)


def _all_menu_actions() -> set[str]:
    default_actions = menu_actions(eeglab_menus(all_menus=False, include_plugins=True))
    full_actions = menu_actions(eeglab_menus(all_menus=True, include_plugins=True))
    return default_actions | full_actions


def test_every_menu_action_is_implemented_placeholder_or_explicit_exclusion():
    unknown = sorted(action for action in _all_menu_actions() if action_kind(action) == "unknown")

    assert unknown == []


def test_placeholder_inventory_has_phase_or_exclusion_metadata():
    inventory = placeholder_inventory()

    assert set(inventory) == PLACEHOLDER_ACTIONS
    assert not any(metadata.phase == "2" for metadata in inventory.values())
    assert not any(metadata.excluded_reason == "eegbrowser" for metadata in inventory.values())
    for action, metadata in inventory.items():
        assert bool(metadata.phase) ^ bool(metadata.excluded_reason), action


def test_no_implemented_action_remains_marked_as_placeholder():
    overlap = sorted(IMPLEMENTED_ACTIONS & PLACEHOLDER_ACTIONS)

    assert overlap == []
