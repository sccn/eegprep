"""MCP server entrypoint for EEGPrep."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import eegprep

from . import tools


INSTALL_HINT = "Install the optional MCP dependencies with: pip install 'eegprep[mcp]' or uv sync --extra mcp"


def build_server(*, host: str = "127.0.0.1", port: int = 8000, log_level: str = "INFO") -> Any:
    """Build the EEGPrep FastMCP server."""
    FastMCP = _load_fastmcp()
    server = FastMCP(
        name="EEGPrep",
        instructions=(
            "Use EEGPrep MCP for EEGLAB-compatible EEG dataset inspection, validation, "
            "agent guidance, CLI schema discovery, and explicit JSON-safe CLI execution. "
            "Prefer inspect/validate/plan before mutating data."
        ),
        host=host,
        port=port,
        log_level=log_level,
    )

    @server.tool(
        name="eegprep_capabilities",
        description="List EEGPrep MCP tools, allowed CLI commands, and agent safety policy.",
    )
    def eegprep_capabilities() -> dict[str, Any]:
        return _call(tools.capabilities)

    @server.tool(
        name="eegprep_agent_guide",
        description="Return bundled version-matched guidance for agents using EEGPrep MCP and CLI.",
    )
    def eegprep_agent_guide(full: bool = False) -> dict[str, Any]:
        return _call(tools.agent_guide, full=full)

    @server.tool(
        name="eegprep_inspect_dataset",
        description="Inspect an EEGLAB .set dataset. section: summary, events, channels, or ica.",
    )
    def eegprep_inspect_dataset(path: str, section: str = "summary", limit: int = 50) -> dict[str, Any]:
        return _call(tools.inspect_eeg_dataset, path, section=section, limit=limit)

    @server.tool(name="eegprep_validate_dataset", description="Validate an EEGLAB .set dataset.")
    def eegprep_validate_dataset(path: str) -> dict[str, Any]:
        return _call(tools.validate_eeg_dataset, path)

    @server.tool(name="eegprep_command_schema", description="Return a machine-readable EEGPrep CLI command schema.")
    def eegprep_command_schema(command: str) -> dict[str, Any]:
        return _call(tools.command_schema, command)

    @server.tool(name="eegprep_command_examples", description="Return EEGPrep CLI examples for a command.")
    def eegprep_command_examples(command: str) -> dict[str, Any]:
        return _call(tools.command_examples, command)

    @server.tool(
        name="eegprep_plan_cli_command",
        description="Plan an allowlisted EEGPrep CLI command without executing it.",
    )
    def eegprep_plan_cli_command(arguments: list[str]) -> dict[str, Any]:
        return _call(tools.plan_cli_command, arguments)

    @server.tool(
        name="eegprep_run_cli_command",
        description=(
            "Execute an allowlisted EEGPrep CLI command through Python, not a shell. "
            "Arguments must include --json. File-writing commands require allow_write=True; "
            "--overwrite also requires allow_overwrite=True."
        ),
    )
    def eegprep_run_cli_command(
        arguments: list[str],
        allow_write: bool = False,
        allow_overwrite: bool = False,
    ) -> dict[str, Any]:
        return _call(
            tools.run_cli_command,
            arguments,
            allow_write=allow_write,
            allow_overwrite=allow_overwrite,
        )

    @server.resource(
        "eegprep://capabilities",
        name="EEGPrep MCP capabilities",
        description="Machine-readable EEGPrep MCP and CLI capabilities.",
        mime_type="application/json",
    )
    def capabilities_resource() -> str:
        return json.dumps(tools.capabilities(), sort_keys=True)

    @server.resource(
        "eegprep://agent-guide",
        name="EEGPrep MCP agent guide",
        description="Version-matched EEGPrep MCP guidance for AI agents.",
        mime_type="text/markdown",
    )
    def agent_guide_resource() -> str:
        return tools.agent_guide(full=True)["content"]

    @server.prompt(
        name="eegprep_preprocess_plan",
        description="Prompt template for planning a safe EEGPrep preprocessing workflow.",
    )
    def eegprep_preprocess_plan(dataset_path: str) -> str:
        return (
            "Use EEGPrep MCP to inspect and validate this dataset before proposing changes: "
            f"{dataset_path}\n"
            "Call eegprep_inspect_dataset(section='summary'), eegprep_validate_dataset, then "
            "eegprep_command_schema for each needed operation. Prefer non-destructive outputs, "
            "pipeline plan/dry-run before expensive work, and report stable error codes."
        )

    return server


def main(argv: list[str] | None = None) -> int:
    """Run the EEGPrep MCP server."""
    parser = argparse.ArgumentParser(
        prog="eegprep-mcp",
        description="Run the EEGPrep Model Context Protocol server for AI agents.",
        epilog=f"Agent start: configure your MCP client to run `eegprep-mcp`. {INSTALL_HINT}.",
    )
    parser.add_argument("--transport", choices=["stdio", "streamable-http", "sse"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP transports.")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP transports.")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    parser.add_argument("--version", action="store_true", help="Show EEGPrep version and exit.")
    args = parser.parse_args(argv)
    if args.version:
        print(f"eegprep-mcp {eegprep.__version__}")
        return 0
    try:
        server = build_server(host=args.host, port=args.port, log_level=args.log_level)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    server.run(transport=args.transport)
    return 0


def _load_fastmcp() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(f"The EEGPrep MCP server requires the optional 'mcp' extra. {INSTALL_HINT}.") from exc
    return FastMCP


def _call(func: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        code = getattr(exc, "code", "UNEXPECTED_ERROR")
        message = getattr(exc, "message", str(exc))
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
        if getattr(exc, "details", None) is not None:
            payload["details"] = getattr(exc, "details")
        return payload


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
