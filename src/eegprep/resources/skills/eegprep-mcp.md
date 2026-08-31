# EEGPrep MCP Agent Guide

Use `eegprep-mcp` when an AI agent needs a structured tool server for EEGPrep
dataset inspection, validation, workflow planning, and JSON-safe command
execution. Use `eegprep-gui` and `eegprep-console` for human interactive work.

## Agent Rules

- Start with `eegprep_capabilities` to learn the tool and command policy.
- Inspect and validate datasets before proposing processing steps.
- Use `eegprep_plan_cli_command` before `eegprep_run_cli_command`.
- Include `--json` in every CLI command executed through MCP.
- Treat write operations as explicit user decisions: pass `allow_write=True`
  only after reviewing output paths with the user.
- Treat `--overwrite` as a separate explicit decision: pass
  `allow_overwrite=True` only after the user approves overwriting data.
- Prefer non-destructive output files, manifests, and pipeline dry-runs.
- Do not shell out manually when an MCP tool can run the EEGPrep command
  in-process.

## Common Tool Flow

1. `eegprep_capabilities()`
2. `eegprep_agent_guide()`
3. `eegprep_inspect_dataset(path, section="summary")`
4. `eegprep_validate_dataset(path)`
5. `eegprep_command_schema("pipeline")` or another target command
6. `eegprep_plan_cli_command(["pipeline", "plan", "preprocess.yaml", "--json"])`
7. `eegprep_run_cli_command([...], allow_write=False)` for read-only commands

For write commands:

```text
eegprep_plan_cli_command([
  "resample", "input.set", "--freq", "128", "--output", "resampled.set", "--json"
])
eegprep_run_cli_command([...], allow_write=True)
```

## Client Configuration

Configure an MCP client to launch:

```bash
eegprep-mcp
```

For a source checkout:

```bash
uv run eegprep-mcp
```

If the command is unavailable, install the optional dependencies:

```bash
pip install "eegprep[mcp]"
```

## Extended Reference

The MCP server reuses the same command schemas, examples, JSON result format,
and stable error codes as the EEGPrep CLI. It does not depend on MATLAB or a
vendored EEGLAB checkout at runtime.

Available tools:

- `eegprep_capabilities`
- `eegprep_agent_guide`
- `eegprep_inspect_dataset`
- `eegprep_validate_dataset`
- `eegprep_command_schema`
- `eegprep_command_examples`
- `eegprep_plan_cli_command`
- `eegprep_run_cli_command`

Available resources:

- `eegprep://capabilities`
- `eegprep://agent-guide`
