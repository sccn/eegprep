#!/usr/bin/env python3
"""Create an EEGPrep autoreview campaign workflow directory."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict


class Packet(TypedDict):
    id: str
    branch: str
    paths: list[str]
    tests: list[str]


DEFAULT_PACKETS: list[Packet] = [
    {
        "id": "01-popfunc",
        "branch": "autoreview/popfunc",
        "paths": [
            "src/eegprep/functions/popfunc",
            "src/eegprep/resources/help/pop_*.md",
            "tests/test_pop_*.py",
        ],
        "tests": [
            "uv run --no-sync pytest tests/test_pop_utils.py tests/test_file_menu_pop_functions.py",
        ],
    },
    {
        "id": "02-sigproc",
        "branch": "autoreview/sigproc",
        "paths": [
            "src/eegprep/functions/sigprocfunc",
            "tests/test_*runica*.py",
            "tests/test_*resample*.py",
        ],
        "tests": [
            "uv run --no-sync pytest tests/test_pop_resample_python.py tests/test_eeg_runica.py tests/test_runica.py tests/test_gui_pop_runica.py",
        ],
    },
    {
        "id": "03-gui-session",
        "branch": "autoreview/gui-session",
        "paths": [
            "src/eegprep/functions/guifunc",
            "src/eegprep/functions/adminfunc",
            "tests/test_console_workspace.py",
            "tests/test_gui_*.py",
        ],
        "tests": [
            "uv run --no-sync pytest tests/test_console_workspace.py tests/test_gui_main_window.py",
        ],
    },
    {
        "id": "04-plugins",
        "branch": "autoreview/plugins",
        "paths": [
            "src/eegprep/plugins",
            "tests/test_*clean*.py",
            "tests/test_*iclabel*.py",
            "tests/test_*bids*.py",
        ],
        "tests": [
            "uv run --no-sync pytest tests/test_gui_pop_clean_rawdata.py tests/test_iclabel.py",
        ],
    },
    {
        "id": "05-io-bids-study",
        "branch": "autoreview/io-bids-study",
        "paths": [
            "src/eegprep/functions/popfunc/pop_fileio.py",
            "src/eegprep/functions/popfunc/pop_loadset.py",
            "src/eegprep/functions/popfunc/pop_saveset.py",
            "src/eegprep/plugins/EEG_BIDS",
            "src/eegprep/functions/studyfunc",
            "tests/test_*study*.py",
            "tests/test_*bids*.py",
            "tests/test_file_menu_pop_functions.py",
        ],
        "tests": [
            "uv run --no-sync pytest tests/test_file_menu_pop_functions.py tests/test_study_metadata.py tests/test_study_measures.py tests/test_study_clustering.py tests/test_study_end_to_end.py",
        ],
    },
    {
        "id": "06-cli-docs-tools",
        "branch": "autoreview/cli-docs-tools",
        "paths": [
            "src/eegprep/cli",
            "docs/source",
            ".agents/skills",
            "tools",
            "scripts",
            "tests/test_cli*.py",
        ],
        "tests": [
            "uv run --no-sync pytest tests/test_cli_main.py tests/test_cli_transforms.py tests/test_cli_pipeline_qc_report.py tests/test_cli_bids_eeglab_commands.py",
            "./pre-commit.py --changed-from origin/develop",
        ],
    },
]


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:64].strip("-") or "autoreview-campaign"


def write_new(path: Path, content: str) -> None:
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def packet_prompt(packet: Packet, base: str) -> str:
    paths = "\n".join(f"  --path {shlex.quote(path)} \\" for path in packet["paths"])
    tests = "\n".join(f"- `{test}`" for test in packet["tests"])
    return f"""# Packet {packet["id"]}: {packet["branch"]}

## Objective
Run a scoped EEGPrep autoreview loop for this codebase area, fix real findings from first principles, and open a PR to `{base}`.

## Scope
{chr(10).join(f"- `{path}`" for path in packet["paths"])}

Fixes may touch related helpers outside this scope when required by the root cause, but keep the PR conceptually tied to this packet.

## Command

```bash
.agents/skills/oc-autoreview-adapted/scripts/autoreview \\
  --mode codebase \\
{paths}
  --thinking codex=xhigh
```

## Verification
Run focused checks first:

{tests}

Then run broader checks if the fix affects shared behavior.

## PR Requirements
- Branch: `{packet["branch"]}`
- Target: `{base}`
- PR body must list every finding reviewed:
  - Fixed: finding, root cause, files changed, tests run.
  - Rejected: finding and why it is not real or not worth changing.
  - Follow-up: only when real but intentionally outside this PR.

## Do Not
- Do not auto-merge.
- Do not revert unrelated concurrent work.
- Do not report vague architecture preferences without concrete failure modes.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("title", nargs="?", default="EEGPrep autoreview campaign")
    parser.add_argument("--root", default=".workflow")
    parser.add_argument("--slug")
    parser.add_argument("--base", default="origin/develop")
    parser.add_argument("--max-concurrent", type=int, default=3)
    args = parser.parse_args()

    slug = slugify(args.slug or args.title)
    run_dir = Path(args.root) / slug
    packets_dir = run_dir / "packets"
    results_dir = run_dir / "results"
    packets_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(UTC).replace(microsecond=0).isoformat()
    packets = [
        {
            "id": packet["id"],
            "branch": packet["branch"],
            "paths": packet["paths"],
            "tests": packet["tests"],
            "status": "pending",
            "pr": None,
        }
        for packet in DEFAULT_PACKETS
    ]
    state = {
        "title": args.title,
        "slug": slug,
        "created_at": now,
        "status": "planned",
        "base": args.base,
        "max_concurrent_agents": args.max_concurrent,
        "packets": packets,
        "integration": {"status": "not_started", "notes": ""},
    }
    write_new(run_dir / "state.json", json.dumps(state, indent=2) + "\n")
    write_new(
        run_dir / "plan.md",
        f"""# {args.title}

## Goal
Run parallel scoped autoreview loops across EEGPrep, fix real bugs/parity/architecture issues, and open PRs for human review.

## Success Criteria
- Every packet has a PR or a recorded no-change result.
- Each PR body lists fixed, rejected, and follow-up findings.
- Each packet reruns autoreview after fixes.
- Integration checks pass after packet PRs merge.

## Constraints
- Keep AGENTS.md and EEGPrep's EEGLAB parity goal in force.
- Runtime code must remain standalone and not depend on vendored EEGLAB.
- Do not auto-merge packet PRs.
- Max concurrent agents: {args.max_concurrent}.

## Risks
- Concurrent work conflicts: keep packet ownership mostly disjoint and resolve against authoritative code.
- Noisy architecture findings: accept only findings with concrete failure modes.

## Work Packets
{chr(10).join(f"- `{packet['id']}` -> `{packet['branch']}`" for packet in packets)}

## Integration Policy
Parent agent tracks PRs, resolves conflicts after merges, runs broader checks, and updates final-report.md.
""",
    )
    write_new(
        run_dir / "orchestration.md",
        f"""# Orchestration: {args.title}

## Execution Rules
- Use available subagent/thread/worktree tools when exposed by the environment.
- Spawn at most {args.max_concurrent} packet agents at once.
- Each packet owns its branch and opens one PR before the parent starts further work in that area.
- If no subagent runner is available, execute packets sequentially and write notes in `results/`.
- Parent integrates packet results; do not paste raw worker dumps as final status.

## Packet Launch
Give each worker only its packet file plus AGENTS.md context. Workers must not revert unrelated edits and must adapt to concurrent changes.

## Completion Audit
- All packet PRs created or no-change results recorded.
- PR bodies include every finding reviewed.
- Final integration checks recorded in `final-report.md`.
""",
    )
    write_new(
        run_dir / "final-report.md",
        f"""# Final Report: {args.title}

## Outcome

## Packet PRs

## Findings Fixed

## Findings Rejected

## Follow-ups

## Integration Verification

## Remaining Risks
""",
    )
    for packet in DEFAULT_PACKETS:
        write_new(packets_dir / f"{packet['id']}.md", packet_prompt(packet, args.base))

    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
