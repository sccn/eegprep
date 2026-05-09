"""Compare EEGLAB and EEGPrep menu inventories."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


def _label(node: dict[str, Any]) -> str:
    return str(node.get("label", ""))


def _enabled(node: dict[str, Any]) -> bool:
    value = node.get("enabled", True)
    if isinstance(value, str):
        return value.lower() == "on"
    return bool(value)


def _separator(node: dict[str, Any]) -> bool:
    value = node.get("separator", False)
    if isinstance(value, str):
        return value.lower() == "on"
    return bool(value)


def _children(node: dict[str, Any]) -> list[dict[str, Any]]:
    value = node.get("children", [])
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return []


def compare_menu_trees(reference: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> list[str]:
    """Return structural differences between two menu trees."""
    differences: list[str] = []
    max_len = max(len(reference), len(candidate))
    for index in range(max_len):
        if index >= len(reference):
            differences.append(f"extra candidate menu at index {index}: {_label(candidate[index])}")
            continue
        if index >= len(candidate):
            differences.append(f"missing candidate menu at index {index}: {_label(reference[index])}")
            continue

        ref_node = reference[index]
        cand_node = candidate[index]
        path = _label(ref_node) or f"index {index}"
        if _label(ref_node) != _label(cand_node):
            differences.append(
                f"{path}: label mismatch, expected {_label(ref_node)!r}, got {_label(cand_node)!r}"
            )
        if _enabled(ref_node) != _enabled(cand_node):
            differences.append(
                f"{path}: enabled mismatch, expected {_enabled(ref_node)}, got {_enabled(cand_node)}"
            )
        if _separator(ref_node) != _separator(cand_node):
            differences.append(
                f"{path}: separator mismatch, expected {_separator(ref_node)}, got {_separator(cand_node)}"
            )

        child_differences = compare_menu_trees(
            _children(ref_node),
            _children(cand_node),
        )
        for child in child_differences:
            differences.append(f"{path} > {child}")
    return differences


def load_menu_tree(path: pathlib.Path) -> list[dict[str, Any]]:
    """Load a menu tree JSON file."""
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "menus" in raw:
        raw = raw["menus"]
    if not isinstance(raw, list):
        raise ValueError("menu inventory must be a list or an object with a menus list")
    return raw


def write_report(differences: list[str], report_path: pathlib.Path) -> None:
    """Write a menu inventory report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if not differences:
        report_path.write_text("Menu inventories match.\n")
        return
    report_path.write_text(
        "Menu inventory differences:\n\n"
        + "\n".join(f"- {difference}" for difference in differences)
        + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=pathlib.Path, required=True)
    parser.add_argument("--candidate", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, default=pathlib.Path(".visual-parity/menu_report.md"))
    parser.add_argument("--advisory", action="store_true", help="Always exit 0 after writing the report")
    args = parser.parse_args(argv)

    try:
        differences = compare_menu_trees(load_menu_tree(args.reference), load_menu_tree(args.candidate))
    except Exception as error:
        parser.exit(1, f"menu inventory error: {error}\n")
    write_report(differences, args.report)
    print(f"report: {args.report}")
    if differences:
        print(f"differences: {len(differences)}")
    return 0 if args.advisory or not differences else 1


if __name__ == "__main__":
    sys.exit(main())
