"""Agent-facing Model Context Protocol integration for EEGPrep."""

from __future__ import annotations

from .tools import (
    agent_guide,
    capabilities,
    command_examples,
    command_schema,
    inspect_eeg_dataset,
    plan_cli_command,
    run_cli_command,
    validate_eeg_dataset,
)

__all__ = [
    "agent_guide",
    "capabilities",
    "command_examples",
    "command_schema",
    "inspect_eeg_dataset",
    "plan_cli_command",
    "run_cli_command",
    "validate_eeg_dataset",
]
