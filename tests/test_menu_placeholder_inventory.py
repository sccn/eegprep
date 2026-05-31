from __future__ import annotations

from eegprep.functions.guifunc.eeglab_menu import eeglab_menus, menu_actions
from eegprep.functions.guifunc.menu_actions import IMPLEMENTED_ACTIONS, action_kind
from eegprep.functions.guifunc.menu_placeholders import (
    PLACEHOLDER_ACTIONS,
    placeholder_inventory,
    placeholder_metadata,
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
    for action, metadata in inventory.items():
        assert bool(metadata.phase) ^ bool(metadata.excluded_reason), action


def test_no_implemented_action_remains_marked_as_placeholder():
    overlap = sorted(IMPLEMENTED_ACTIONS & PLACEHOLDER_ACTIONS)

    assert overlap == []


def test_placeholder_inventory_classifies_representative_phase_work():
    assert placeholder_metadata("pop_editeventfield") is None
    assert placeholder_metadata("pop_eegfilt") is None
    assert placeholder_metadata("pop_dipfit_settings") is None
    assert action_kind("pop_dipfit_settings") == "implemented"
    assert action_kind("pop_dipfit_gridsearch") == "implemented"
    assert not any(metadata.phase == "3" for metadata in placeholder_inventory().values())
    assert action_kind("pop_rejchan") == "implemented"
    assert placeholder_metadata("pop_spectopo") is None
    assert action_kind("pop_spectopo") == "implemented"
    assert placeholder_metadata("pop_signalstat") is None
    assert action_kind("pop_signalstat") == "implemented"
    assert action_kind("pop_newtimef") == "implemented"
    assert action_kind("pop_newcrossf") == "implemented"
    assert action_kind("pop_eventstat") == "implemented"
    assert placeholder_metadata("pop_viewprops:channels") is None
    assert action_kind("pop_viewprops:channels") == "implemented"
    assert action_kind("pop_viewprops:components") == "implemented"
    assert placeholder_metadata("eeglab_update") is None
    assert action_kind("updates") == "implemented"
    assert not any(metadata.phase == "6" for metadata in placeholder_inventory().values())
    assert placeholder_metadata("pop_studydesign") is None
    assert action_kind("pop_studydesign") == "implemented"
    assert action_kind("select_study_set") == "implemented"


def test_phase1b_placeholders_are_removed_after_file_edit_completion():
    assert not any(metadata.phase == "1b" for metadata in placeholder_inventory().values())


def test_eegbrowser_scrolling_actions_are_implemented():
    assert placeholder_metadata("pop_eegplot:data") is None
    assert action_kind("pop_eegplot:data") == "implemented"
    assert action_kind("pop_eegplot:channels") == "implemented"
    assert action_kind("pop_eegplot:components") == "implemented"
