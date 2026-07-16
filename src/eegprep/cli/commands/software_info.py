"""Agent-friendly EEGPrep software_info command support."""

from __future__ import annotations

import argparse
from typing import Any

from eegprep.cli.core import software_info, command_ok


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    """Register ``software_info`` with an argparse dispatcher."""
    parser = subparsers.add_parser("software_info", help="Display software, active threading backend and core limits.")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON")
    parser.set_defaults(func=handle_registered, handler=handle_registered)
    return parser


def handle_registered(args: argparse.Namespace) -> dict[str, Any]:
    info = software_info()
    return command_ok("software_info", info=info)
