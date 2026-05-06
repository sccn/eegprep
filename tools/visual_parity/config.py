"""Manifest loading for EEGPrep visual parity cases."""

from __future__ import annotations

import json
import pathlib
import shlex
from dataclasses import dataclass, field
from typing import Any


VALID_TARGETS = {"eeglab", "eegprep"}
DEFAULT_MANIFEST = pathlib.Path(__file__).with_name("cases.json")


@dataclass(frozen=True)
class TargetSpec:
    """Capture instructions for one UI target."""

    type: str = "command"
    command: tuple[str, ...] = ()
    action: str = ""
    matlab_command: str = ""
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class VisualCase:
    """A UI state that can be captured for visual parity review."""

    id: str
    description: str
    window_size: tuple[int, int]
    timeout_seconds: int
    targets: dict[str, TargetSpec]


def _parse_command(raw_command: Any) -> tuple[str, ...]:
    if not raw_command:
        return ()
    if isinstance(raw_command, str):
        return tuple(shlex.split(raw_command))
    if isinstance(raw_command, list) and all(isinstance(part, str) for part in raw_command):
        return tuple(raw_command)
    raise ValueError("target command must be a string or a list of strings")


def _parse_target(raw_target: dict[str, Any]) -> TargetSpec:
    if not isinstance(raw_target, dict):
        raise ValueError("target spec must be an object")
    raw_env = raw_target.get("env", {})
    if not isinstance(raw_env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in raw_env.items()):
        raise ValueError("target env must be an object of string keys and values")
    return TargetSpec(
        type=str(raw_target.get("type", "command")),
        command=_parse_command(raw_target.get("command")),
        action=str(raw_target.get("action", "")),
        matlab_command=str(raw_target.get("matlab_command", "")),
        env=dict(raw_env),
    )


def load_manifest(path: pathlib.Path | str = DEFAULT_MANIFEST) -> dict[str, VisualCase]:
    """Load visual parity cases from a JSON manifest."""
    manifest_path = pathlib.Path(path)
    raw = json.loads(manifest_path.read_text())
    defaults = raw.get("defaults", {})
    cases = raw.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError("visual parity manifest must contain a cases list")

    parsed: dict[str, VisualCase] = {}
    for raw_case in cases:
        if not isinstance(raw_case, dict):
            raise ValueError("each visual parity case must be an object")
        case_id = str(raw_case["id"])
        raw_size = raw_case.get("window_size", defaults.get("window_size", [1100, 750]))
        if not isinstance(raw_size, list) or len(raw_size) != 2:
            raise ValueError(f"{case_id}: window_size must be [width, height]")
        raw_targets = raw_case.get("targets", {})
        if not isinstance(raw_targets, dict):
            raise ValueError(f"{case_id}: targets must be an object")
        targets = {name: _parse_target(value) for name, value in raw_targets.items()}
        unknown_targets = set(targets) - VALID_TARGETS
        if unknown_targets:
            names = ", ".join(sorted(unknown_targets))
            raise ValueError(f"{case_id}: unknown target(s): {names}")
        parsed[case_id] = VisualCase(
            id=case_id,
            description=str(raw_case.get("description", "")),
            window_size=(int(raw_size[0]), int(raw_size[1])),
            timeout_seconds=int(raw_case.get("timeout_seconds", defaults.get("timeout_seconds", 90))),
            targets=targets,
        )

    return parsed


def format_command(command: tuple[str, ...], values: dict[str, str]) -> list[str]:
    """Apply manifest placeholders to a subprocess command."""
    try:
        return [part.format(**values) for part in command]
    except KeyError as exc:
        missing = exc.args[0]
        raise ValueError(f"unknown command placeholder: {missing}") from exc
