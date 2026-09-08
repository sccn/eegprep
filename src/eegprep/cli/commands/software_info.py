"""Agent-friendly EEGPrep software-info command support."""

from __future__ import annotations

import argparse
from typing import Any

from eegprep.cli.core import command_ok, software_info


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    """Register ``software-info`` with an argparse dispatcher."""
    parser = subparsers.add_parser("software-info", help="Display software, active threading backend and core limits.")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON")
    parser.set_defaults(func=handle_registered, handler=handle_registered)
    return parser


def handle_registered(args: argparse.Namespace) -> dict[str, Any]:
    return command_ok("software-info", **software_info())
