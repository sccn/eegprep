"""Export EEGPrep's EEGLAB-style menu tree as JSON."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eegprep.functions.guifunc.eeglab_menu import eeglab_menus
from eegprep.functions.guifunc.menu_spec import MenuItemSpec, menu_to_inventory


STATES = {
    "startup": {"startup"},
    "continuous": {"continuous_dataset"},
    "epoched": {"epoched_dataset"},
    "multiple": {"multiple_datasets"},
    "study": {"study"},
}


def export_inventory(
    output: pathlib.Path,
    *,
    all_menus: bool = False,
    include_plugins: bool = True,
    state: str = "startup",
) -> None:
    """Write EEGPrep menu inventory JSON."""
    if state not in STATES:
        raise ValueError(f"unknown state: {state}")
    menus = _menus_for_state(all_menus=all_menus, include_plugins=include_plugins, state=state)
    payload = {"menus": menu_to_inventory(menus, STATES[state])}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")


def _menus_for_state(*, all_menus: bool, include_plugins: bool, state: str):
    menus = []
    for menu in eeglab_menus(all_menus=all_menus, include_plugins=include_plugins):
        if menu.label == "Datasets":
            menu = menu.with_children(_dataset_items_for_state(state))
        menus.append(menu)
    return tuple(menus)


def _dataset_items_for_state(state: str) -> tuple[MenuItemSpec, ...]:
    if state == "continuous":
        return (
            MenuItemSpec("Dataset 1:menu continuous", action="retrieve_dataset:1", userdata="study:on"),
            _select_multiple_item(),
        )
    if state == "epoched":
        return (
            MenuItemSpec("Dataset 1:menu epoched", action="retrieve_dataset:1", userdata="study:on"),
            _select_multiple_item(),
        )
    if state == "multiple":
        return (
            MenuItemSpec("Dataset 1:menu one", action="retrieve_dataset:1", userdata="study:on"),
            MenuItemSpec("Dataset 2:menu two", action="retrieve_dataset:2", userdata="study:on"),
            _select_multiple_item(),
        )
    return (MenuItemSpec("Select multiple datasets", action="select_multiple_datasets", separator=True),)


def _select_multiple_item() -> MenuItemSpec:
    return MenuItemSpec(
        "Select multiple datasets",
        action="select_multiple_datasets",
        userdata="study:on",
        separator=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--all-menus", action="store_true")
    parser.add_argument("--no-plugins", action="store_true")
    parser.add_argument("--state", choices=sorted(STATES), default="startup")
    args = parser.parse_args(argv)
    export_inventory(
        args.output,
        all_menus=args.all_menus,
        include_plugins=not args.no_plugins,
        state=args.state,
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
