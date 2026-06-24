"""Top-level EEGPrep command-line interface."""

from __future__ import annotations

import argparse
import sys
from typing import Any, NoReturn

import eegprep

from eegprep.cli import discovery
from eegprep.cli.core import EEGPrepCLIError, print_result
from eegprep.cli.dataset import inspect_channels, inspect_dataset, inspect_events, inspect_ica, validate_dataset
from eegprep.cli.commands import batch as batch_commands
from eegprep.cli.commands import bids as bids_commands
from eegprep.cli.commands import migrate as migrate_commands
from eegprep.cli.commands import pipeline as pipeline_commands
from eegprep.cli.commands import qc as qc_commands
from eegprep.cli.commands import report as report_commands
from eegprep.cli.commands.transforms import register_subcommands


AGENT_EPILOG = """\
Start here (for AI agents):
  eegprep skills get eegprep-cli

Skills ship with the CLI, stay version-matched, and include workflow patterns,
machine-readable output rules, safety boundaries, and copy-paste examples.
Prefer this over guessing commands from flag docs alone.

Core agent commands:
  capabilities --json              List supported commands and contracts
  schema pipeline --json           Print the pipeline config schema
  schema command <name> --json      Print a command schema
  skills list|get|path              List or retrieve bundled agent guides
"""


class EEGPrepArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that can emit structured errors for JSON-mode agent calls."""

    def __init__(self, *args: Any, json_requested: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.json_requested = json_requested

    def add_subparsers(self, *args: Any, **kwargs: Any) -> argparse._SubParsersAction:
        kwargs.setdefault("parser_class", self._child_parser_factory())
        return super().add_subparsers(*args, **kwargs)

    def error(self, message: str) -> NoReturn:
        if self.json_requested:
            raise EEGPrepCLIError(
                "CONFIG_SCHEMA_ERROR",
                message,
                suggestion="Run the command with --help or inspect schema command <name>.",
                exit_code=2,
            )
        super().error(message)

    def _child_parser_factory(self) -> Any:
        json_requested = self.json_requested

        def factory(*args: Any, **kwargs: Any) -> "EEGPrepArgumentParser":
            return type(self)(*args, json_requested=json_requested, **kwargs)

        return factory


def main(argv: list[str] | None = None) -> int:
    """Run the EEGPrep CLI."""
    from eegprep.cli.cloud import managed_cache, CloudSyncError
    
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    requested_json = _argv_requests_json(raw_argv)
    parser = build_parser(json_requested=requested_json)
    args: argparse.Namespace | None = None
    
    try:
        with managed_cache():
            try:
                args = parser.parse_args(raw_argv)
                if not hasattr(args, "handler"):
                    parser.print_help()
                    return 0
                result = args.handler(args)
            except EEGPrepCLIError as exc:
                print_result(exc.to_response(), as_json=requested_json)
                return exc.exit_code
    except CloudSyncError as exc:
        payload = {
            "status": "error",
            "schema_version": "eegprep.error.v1",
            "code": "CLOUD_SYNC_ERROR",
            "message": str(exc),
            "suggestion": "Check network, credentials, and URI validity.",
        }
        print_result(payload, as_json=requested_json)
        return 1
    except Exception as exc:
        code = getattr(exc, "code", None)
        message = getattr(exc, "message", None)
        if code and message:
            payload: dict[str, Any] = {
                "status": "error",
                "schema_version": "eegprep.error.v1",
                "code": code,
                "message": message,
            }
            if getattr(exc, "path", None) is not None:
                payload["path"] = str(getattr(exc, "path"))
            if getattr(exc, "suggestion", None) is not None:
                payload["suggestion"] = getattr(exc, "suggestion")
            print_result(payload, as_json=_json_requested(args, requested_json=requested_json))
            return int(getattr(exc, "exit_code", 1) or 1)
        payload = {
            "status": "error",
            "schema_version": "eegprep.error.v1",
            "code": "UNEXPECTED_ERROR",
            "message": str(exc),
            "suggestion": "Rerun with --verbose or file an issue if this is reproducible.",
        }
        print_result(payload, as_json=_json_requested(args, requested_json=requested_json))
        if args is not None and bool(getattr(args, "verbose", False)):
            raise
        return 1

    if isinstance(result, dict) and "_raw_text" in result and not _json_requested(args, requested_json=requested_json):
        print(result["_raw_text"], end="" if result["_raw_text"].endswith("\n") else "\n")
    else:
        print_result(result, as_json=_json_requested(args, requested_json=requested_json))
    if isinstance(result, dict) and result.get("status") == "error":
        return int(result.get("exit_code", 1) or 1)
    if isinstance(result, dict):
        return int(result.get("exit_code", 0) or 0)
    return 0


def build_parser(*, json_requested: bool = False) -> EEGPrepArgumentParser:
    """Build the root parser."""
    parser = EEGPrepArgumentParser(
        prog="eegprep",
        description="EEGLAB-compatible, Python-native EEG preprocessing CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=AGENT_EPILOG,
        json_requested=json_requested,
    )
    parser.add_argument("--version", action="store_true", help="Show EEGPrep version and exit.")
    subparsers = parser.add_subparsers(dest="command")

    _register_discovery(subparsers)
    _register_inspect(subparsers)
    _register_validate(subparsers)
    register_subcommands(subparsers)
    pipeline_commands.register(subparsers)
    qc_commands.register(subparsers)
    report_commands.register(subparsers)
    batch_commands.register(subparsers)
    bids_commands.register(subparsers)
    migrate_commands.register(subparsers)

    parser.set_defaults(handler=_handle_root)
    return parser


def _register_discovery(subparsers: argparse._SubParsersAction) -> None:
    capabilities = subparsers.add_parser("capabilities", help="List agent-readable CLI capabilities.")
    capabilities.add_argument("--json", action="store_true", help="Emit structured JSON.")
    capabilities.set_defaults(handler=lambda _args: discovery.capabilities())

    schema = subparsers.add_parser("schema", help="Print command or pipeline schemas.")
    schema_sub = schema.add_subparsers(dest="schema_kind", required=True)
    pipeline = schema_sub.add_parser("pipeline", help="Print pipeline config schema.")
    pipeline.add_argument("--json", action="store_true")
    pipeline.set_defaults(handler=lambda _args: discovery.pipeline_schema())
    command = schema_sub.add_parser("command", help="Print a command schema.")
    command.add_argument("name")
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=lambda args: discovery.command_schema(args.name))

    examples = subparsers.add_parser("examples", help="Print command examples.")
    examples.add_argument("name")
    examples.add_argument("--json", action="store_true")
    examples.set_defaults(handler=lambda args: discovery.examples(args.name))

    skills = subparsers.add_parser("skills", help="List and retrieve bundled agent skill content.")
    skills_sub = skills.add_subparsers(dest="skills_command", required=True)
    skills_list = skills_sub.add_parser("list", help="List bundled CLI skills.")
    skills_list.add_argument("--json", action="store_true")
    skills_list.set_defaults(handler=lambda _args: discovery.skills_list())
    skills_get = skills_sub.add_parser("get", help="Print bundled skill content.")
    skills_get.add_argument("name")
    skills_get.add_argument("--full", action="store_true", help="Include extended reference content.")
    skills_get.add_argument("--json", action="store_true")
    skills_get.set_defaults(handler=_handle_skill_get)
    skills_path = skills_sub.add_parser("path", help="Print bundled skill path.")
    skills_path.add_argument("name")
    skills_path.add_argument("--json", action="store_true")
    skills_path.set_defaults(handler=_handle_skill_path)


def _register_inspect(subparsers: argparse._SubParsersAction) -> None:
    inspect = subparsers.add_parser("inspect", help="Inspect dataset metadata, events, channels, or ICA fields.")
    inspect.add_argument("target", nargs="+", help="[dataset|events|channels|ica] <path>, or just <path>.")
    inspect.add_argument("--json", action="store_true")
    inspect.set_defaults(handler=_handle_inspect)


def _register_validate(subparsers: argparse._SubParsersAction) -> None:
    validate = subparsers.add_parser("validate", help="Validate an EEG dataset.")
    validate.add_argument("path")
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(handler=lambda args: validate_dataset(args.path))


def _handle_root(args: argparse.Namespace) -> dict[str, Any]:
    if args.version:
        return {"status": "ok", "schema_version": "eegprep.version.v1", "version": eegprep.__version__}
    raise EEGPrepCLIError("COMMAND_REQUIRED", "Pass a command or --help.", exit_code=2)


def _handle_inspect(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.target) == 1:
        kind = "dataset"
        path = args.target[0]
    elif len(args.target) == 2 and args.target[0] in {"dataset", "events", "channels", "ica"}:
        kind, path = args.target
    else:
        raise EEGPrepCLIError(
            "CONFIG_SCHEMA_ERROR",
            "Use eegprep inspect <path> or eegprep inspect [dataset|events|channels|ica] <path>.",
            exit_code=2,
        )
    handlers = {
        "dataset": inspect_dataset,
        "events": inspect_events,
        "channels": inspect_channels,
        "ica": inspect_ica,
    }
    return handlers[kind](path)


def _handle_skill_get(args: argparse.Namespace) -> dict[str, Any]:
    text = discovery.skill_text(args.name, full=args.full)
    return {
        "status": "ok",
        "schema_version": "eegprep.skills.get.v1",
        "name": args.name,
        "content": text,
        "_raw_text": text,
    }


def _handle_skill_path(args: argparse.Namespace) -> dict[str, Any]:
    path = discovery.skill_path(args.name)
    return {"status": "ok", "schema_version": "eegprep.skills.path.v1", "name": args.name, "path": path}


def _argv_requests_json(argv: list[str]) -> bool:
    return "--json" in argv


def _json_requested(args: argparse.Namespace | None, *, requested_json: bool) -> bool:
    if requested_json:
        return True
    return bool(getattr(args, "json", False))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
