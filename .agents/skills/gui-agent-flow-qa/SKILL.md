---
name: gui-agent-flow-qa
description: Exhaustive user-flow QA workflow using the Computer/GUI agent plus terminal for current PR or branch features. Use only when the user explicitly invokes `$gui-agent-flow-qa`, explicitly mentions `gui-agent-flow-qa`, or explicitly asks to use the Computer/GUI agent to simulate all user flows and produce a QA report. Do not use for ordinary testing, code review, implementation, or GUI parity work unless the user explicitly requests this workflow.
---

# GUI Agent Flow QA

Use this skill to run an end-to-end manual QA pass that simulates real users exercising every relevant terminal and GUI path introduced or changed by the current PR.

## Guardrails

- Invoke only when explicitly requested by the user. This workflow is intentionally heavy and should not run for ordinary test requests.
- Use the Computer/GUI agent or Computer Use tools for UI interactions. Do not replace requested GUI clicking with terminal-only tests.
- Do not edit code during the QA pass unless the user asks for fixes. If defects are found, report them clearly and stop at the report.
- Keep GUI sessions under control: start one flow at a time, capture terminal output, close/cancel dialogs, and verify no test GUI processes are left running before the final report.
- Preserve the working tree. Do not revert unrelated changes.

## Workflow

1. Orient to the PR.
   - Check branch and cleanliness with `git status --short --branch`.
   - Compare against the PR base, usually `origin/develop`, with `git diff --name-status origin/develop...HEAD`.
   - Identify the user-facing features, commands, dialogs, files, tests, resources, and sample data touched by the PR.

2. Build the flow matrix before clicking.
   - Include terminal/API paths, real sample-data paths, synthetic data paths, GUI happy paths, every visible action button, Help, OK, Cancel, close-window behavior, picker dialogs, dropdowns, checkboxes, and textbox edge cases.
   - Include negative cases: blank inputs, invalid labels/indices, out-of-range numeric values, missing optional datasets/resources, duplicate/no-op selections, and boundary values.
   - Define expected outputs before each run: command history strings, shape/channel-count changes, warnings, no-op behavior, GUI warnings, and whether data should change.

3. Run automated and CLI baselines.
   - Use the repo's dependency manager and test conventions, for example `uv run --no-sync pytest ...` or the repo-documented unittest commands.
   - Load any real sample data that a user would try first.
   - Exercise public APIs from terminal with realistic and minimal synthetic data. Record commands, key outputs, warnings, and expected-vs-actual results.

4. Drive the GUI like a user.
   - Launch each GUI flow from the terminal so stdout/stderr and returned values are visible.
   - Use the Computer/GUI agent to inspect the app state, click buttons, type into fields, select menu/dropdown choices, open Help, use pickers, and press OK/Cancel.
   - For each flow, record the visible UI state, the action taken, terminal output, returned command string, data mutation, and whether the dialog remained open or closed.
   - When a list/picker is hard to inspect via accessibility, use careful keyboard or coordinate interaction, then verify from the resulting UI text and terminal output.

5. Stress edge cases.
   - Textboxes: invalid text, empty text, malformed numeric ranges, boundary values, labels with spaces when relevant, and values outside known channel/data ranges.
   - Buttons: each action button, Help, Cancel, OK with incomplete selections, repeated picker use, and missing-resource states.
   - Modes: every checkbox/radio/dropdown combination that maps to a distinct code path.
   - Data: real sample data, small synthetic data, and any special structures needed by the feature such as alternate datasets, removed channels, events, epochs, or package resources.

6. Finish with cleanup and a report.
   - Verify no GUI test processes or terminal sessions remain running.
   - Report branch/base, commands run, tests run, data used, and each user flow tested.
   - Separate results into `Worked`, `Issues Found`, `Warnings/Residual Risk`, and `Suggested Fixes`.
   - For each issue, include reproduction steps, expected behavior, actual behavior, severity, and file/line pointers when known.
   - If everything works, say that clearly and list remaining coverage limits.

## Report Template

```markdown
**Scope**
- Branch/base:
- Features tested:
- Data used:

**Automated/CLI Checks**
- Command:
- Result:
- Expected:

**GUI Flow Results**
| Flow | Steps | Expected | Actual | Result |
| --- | --- | --- | --- | --- |

**Issues Found**
1. Severity: ...
   Repro:
   Expected:
   Actual:
   Evidence:

**Warnings/Residual Risk**
- ...

**Cleanup**
- Remaining processes:
- Working tree:
```
