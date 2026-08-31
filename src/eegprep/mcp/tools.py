"""Reusable EEGPrep MCP tool implementations.

The functions in this module intentionally avoid importing the MCP SDK. That
keeps the agent contract unit-testable and lets ``eegprep-mcp`` fail with a
clear install hint when the optional protocol dependency is missing.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import io
import json
from typing import Any

import eegprep

from eegprep.cli import discovery
from eegprep.cli.core import EEGPrepCLIError, json_safe, ok
from eegprep.cli.dataset import inspect_channels, inspect_dataset, inspect_events, inspect_ica, validate_dataset
from eegprep.cli.main import main as cli_main


MCP_SKILL_NAME = "eegprep-mcp"

READ_ONLY_COMMANDS = {
    "capabilities",
    "schema",
    "examples",
    "skills",
    "inspect",
    "validate",
    "qc",
    "migrate",
}

WRITE_COMMANDS = {
    "batch",
    "bids",
    "clean",
    "epoch",
    "filter",
    "ica",
    "pipeline",
    "report",
    "rereference",
    "resample",
}

ALLOWED_COMMANDS = READ_ONLY_COMMANDS | WRITE_COMMANDS

WRITE_FLAGS = {
    "--bids-root",
    "--html",
    "--manifest",
    "--output",
    "--output-dir",
}


@dataclass(frozen=True)
class CommandPlan:
    """Agent-readable plan for an EEGPrep CLI command."""

    command_line: list[str]
    root_command: str
    writes_files: bool
    overwrites_files: bool
    requires_allow_write: bool
    requires_allow_overwrite: bool

    def to_response(self) -> dict[str, Any]:
        return ok(
            "eegprep.mcp.command_plan.v1",
            command_line=self.command_line,
            root_command=self.root_command,
            writes_files=self.writes_files,
            overwrites_files=self.overwrites_files,
            requires_allow_write=self.requires_allow_write,
            requires_allow_overwrite=self.requires_allow_overwrite,
            suggestion=_plan_suggestion(self),
        )


def capabilities() -> dict[str, Any]:
    """Return MCP and CLI capabilities for AI agents."""
    cli_capabilities = discovery.capabilities()
    return ok(
        "eegprep.mcp.capabilities.v1",
        eegprep_version=eegprep.__version__,
        transport="mcp",
        tools=[
            "eegprep_capabilities",
            "eegprep_agent_guide",
            "eegprep_inspect_dataset",
            "eegprep_validate_dataset",
            "eegprep_command_schema",
            "eegprep_command_examples",
            "eegprep_plan_cli_command",
            "eegprep_run_cli_command",
        ],
        command_policy={
            "allowed_commands": sorted(ALLOWED_COMMANDS),
            "read_only_commands": sorted(READ_ONLY_COMMANDS),
            "write_commands": sorted(WRITE_COMMANDS),
            "writes_require_allow_write": True,
            "overwrite_requires_allow_overwrite": True,
            "json_required_for_execution": True,
            "shell_execution": False,
        },
        cli=cli_capabilities,
    )


def agent_guide(*, full: bool = False) -> dict[str, Any]:
    """Return bundled MCP-specific agent guidance."""
    try:
        content = discovery.skill_text(MCP_SKILL_NAME, full=full)
        name = MCP_SKILL_NAME
    except EEGPrepCLIError:
        content = discovery.skill_text(discovery.CLI_SKILL_NAME, full=full)
        name = discovery.CLI_SKILL_NAME
    return ok("eegprep.mcp.agent_guide.v1", name=name, content=content)


def inspect_eeg_dataset(path: str, *, section: str = "summary", limit: int = 50) -> dict[str, Any]:
    """Inspect an EEGLAB ``.set`` dataset with bounded event/channel output."""
    if limit < 1:
        raise EEGPrepCLIError("CONFIG_SCHEMA_ERROR", "limit must be at least 1.", exit_code=2)
    if section == "summary":
        return inspect_dataset(path)
    if section == "events":
        return _bounded_records(inspect_events(path), key="events", limit=limit)
    if section == "channels":
        return _bounded_records(inspect_channels(path), key="channels", limit=limit)
    if section == "ica":
        return inspect_ica(path)
    raise EEGPrepCLIError(
        "CONFIG_SCHEMA_ERROR",
        "section must be one of: summary, events, channels, ica.",
        suggestion="Use eegprep_command_schema('inspect') before calling this tool.",
        exit_code=2,
    )


def validate_eeg_dataset(path: str) -> dict[str, Any]:
    """Validate an EEGLAB ``.set`` dataset."""
    return validate_dataset(path)


def command_schema(command: str) -> dict[str, Any]:
    """Return a machine-readable CLI command schema."""
    return discovery.command_schema(command)


def command_examples(command: str) -> dict[str, Any]:
    """Return copy-pasteable examples for a CLI command."""
    return discovery.examples(command)


def plan_cli_command(arguments: list[str]) -> dict[str, Any]:
    """Plan an allowlisted EEGPrep CLI command without executing it."""
    plan = _build_plan(arguments)
    return plan.to_response()


def run_cli_command(
    arguments: list[str],
    *,
    allow_write: bool = False,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Execute an allowlisted EEGPrep CLI command and return captured JSON.

    Args:
        arguments: CLI arguments after the ``eegprep`` executable name.
        allow_write: Required when the command may write files.
        allow_overwrite: Required in addition to ``allow_write`` when the
            command contains ``--overwrite``.
    """
    plan = _build_plan(arguments)
    if plan.requires_allow_write and not allow_write:
        raise EEGPrepCLIError(
            "WRITE_CONFIRMATION_REQUIRED",
            "This command may write files. Call again with allow_write=True after the user approves.",
            details=plan.to_response(),
            exit_code=2,
        )
    if plan.requires_allow_overwrite and not allow_overwrite:
        raise EEGPrepCLIError(
            "OVERWRITE_CONFIRMATION_REQUIRED",
            "This command contains --overwrite. Call again with allow_overwrite=True after explicit approval.",
            details=plan.to_response(),
            exit_code=2,
        )
    if "--json" not in arguments:
        raise EEGPrepCLIError(
            "JSON_REQUIRED",
            "MCP command execution requires --json so stdout remains machine-readable.",
            suggestion="Add --json to the EEGPrep CLI arguments.",
            exit_code=2,
        )

    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        exit_code = cli_main(arguments)
    stdout_text = stdout.getvalue()
    stderr_text = stderr.getvalue()
    payload = _parse_json_stdout(stdout_text)
    return ok(
        "eegprep.mcp.command_result.v1",
        command_line=plan.command_line,
        exit_code=exit_code,
        result=payload,
        stdout=stdout_text,
        stderr=stderr_text,
    )


def _build_plan(arguments: list[str]) -> CommandPlan:
    args = _normalize_arguments(arguments)
    root = _root_command(args)
    writes = _command_writes(root, args)
    overwrites = "--overwrite" in args
    return CommandPlan(
        command_line=["eegprep", *args],
        root_command=root,
        writes_files=writes,
        overwrites_files=overwrites,
        requires_allow_write=writes,
        requires_allow_overwrite=overwrites,
    )


def _normalize_arguments(arguments: list[str]) -> list[str]:
    if not isinstance(arguments, list) or not all(isinstance(item, str) for item in arguments):
        raise EEGPrepCLIError("CONFIG_SCHEMA_ERROR", "arguments must be a list of strings.", exit_code=2)
    if not arguments:
        raise EEGPrepCLIError("COMMAND_REQUIRED", "Pass EEGPrep CLI arguments after the executable name.", exit_code=2)
    if arguments[0] == "eegprep":
        return arguments[1:]
    return list(arguments)


def _root_command(arguments: list[str]) -> str:
    if not arguments:
        raise EEGPrepCLIError("COMMAND_REQUIRED", "Pass an EEGPrep CLI command.", exit_code=2)
    if arguments[0].startswith("-"):
        raise EEGPrepCLIError(
            "COMMAND_REQUIRED",
            "MCP command execution requires an explicit EEGPrep CLI subcommand.",
            exit_code=2,
        )
    root = arguments[0]
    if root not in ALLOWED_COMMANDS:
        raise EEGPrepCLIError(
            "COMMAND_NOT_ALLOWED",
            f"Command is not exposed through EEGPrep MCP: {root}",
            suggestion="Run eegprep_capabilities to inspect the allowed command list.",
            exit_code=2,
        )
    return root


def _has_write_flags(arguments: list[str]) -> bool:
    for item in arguments:
        if item in WRITE_FLAGS:
            return True
        if any(item.startswith(flag + "=") for flag in WRITE_FLAGS):
            return True
    return False


def _command_writes(root: str, arguments: list[str]) -> bool:
    if _has_write_flags(arguments):
        return True
    if root == "bids":
        return len(arguments) > 1 and arguments[1] in {"import", "export"}
    if root == "batch":
        return "--dry-run" not in arguments
    if root == "pipeline":
        return len(arguments) > 1 and arguments[1] == "run" and "--dry-run" not in arguments
    return root in WRITE_COMMANDS


def _parse_json_stdout(stdout_text: str) -> Any:
    text = stdout_text.strip()
    if not text:
        return None
    try:
        return json_safe(json.loads(text))
    except json.JSONDecodeError:
        return {"status": "warning", "schema_version": "eegprep.mcp.raw_stdout.v1", "text": text}


def _bounded_records(payload: dict[str, Any], *, key: str, limit: int) -> dict[str, Any]:
    records = list(payload.get(key, []))
    payload = dict(payload)
    payload[key] = records[:limit]
    payload["returned"] = len(payload[key])
    payload["truncated"] = len(records) > limit
    return payload


def _plan_suggestion(plan: CommandPlan) -> str:
    if plan.overwrites_files:
        return "Confirm overwrite intent with the user, then call eegprep_run_cli_command with allow_write=True and allow_overwrite=True."
    if plan.writes_files:
        return "Review output paths, then call eegprep_run_cli_command with allow_write=True."
    return "Safe to run with eegprep_run_cli_command after checking arguments."
