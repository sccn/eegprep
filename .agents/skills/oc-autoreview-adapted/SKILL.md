---
name: oc-autoreview-adapted
description: Run an autonomous EEGPrep-focused structured autoreview on local changes, branches, commits, or PRs using the bundled Codex helper. Use when the user asks for autoreview, OC autoreview, closeout review, second-pass review, final review before commit/push/PR, or when non-trivial EEGPrep code changes need a high-signal correctness, EEGLAB parity, GUI/session, tests, and repo-instruction check.
---

# OC Autoreview Adapted

Run the bundled structured review helper as an autonomous closeout check for
EEGPrep. This skill adapts the OpenClaw autoreview principles to this project:
one frozen diff bundle, one structured JSON result, validated changed-file
findings, read-only inspection, heartbeat progress, optional parallel tests, and
repeat-until-clean behavior.

## Contract

- Run the helper for real unless the user explicitly asks for a plan or manual
  review only.
- Treat review output as advisory. Verify every accepted finding by reading the
  real code path and adjacent files before fixing or reporting it.
- Keep going until the helper exits cleanly with no accepted/actionable findings
  or until you consciously reject a remaining finding with a concrete reason.
- If a review-triggered fix changes code, rerun focused tests and rerun the
  helper on the same target.
- Do not run nested review tools from inside a review. The helper builds one
  bundle, calls Codex in read-only mode, validates the result, and exits.
- Do not push, stage, commit, or open a PR just to run autoreview. Do those only
  when the user requested that action.
- Be patient. The helper prints heartbeat lines such as
  `review still running: codex elapsed=... pid=...`; those are healthy progress.

## Helper

Use the repo-local helper:

```bash
.agents/skills/oc-autoreview-adapted/scripts/autoreview --help
```

The helper:

- defaults to Codex with read-only sandboxing and web search enabled;
- chooses dirty local changes first in `--mode auto`;
- otherwise uses the current PR base when discoverable, then `origin/develop`;
- accepts `--mode local`, `--mode branch --base origin/develop`, and
  `--mode commit --commit HEAD`;
- includes root/scoped `AGENTS.md` instructions in the review bundle;
- validates structured JSON against an EEGPrep-specific schema;
- filters findings to changed files only;
- exits nonzero when accepted/actionable findings remain;
- supports `--prompt`, `--prompt-file`, `--dataset`, `--json-output`,
  `--output`, `--parallel-tests`, `--require-finding`, `--expect-findings`,
  `--no-web-search`, `--model`, and `--thinking`.

The smoke harness creates a temporary EEG-style fixture repo:

```bash
.agents/skills/oc-autoreview-adapted/scripts/test-review-harness --dry-run
```

Run the full harness only when it is acceptable to spend a real Codex review:

```bash
.agents/skills/oc-autoreview-adapted/scripts/test-review-harness --fixture buggy
```

## Pick Target

Use the smallest target that covers the request.

Dirty local work:

```bash
.agents/skills/oc-autoreview-adapted/scripts/autoreview --mode local
```

Branch or PR work:

```bash
.agents/skills/oc-autoreview-adapted/scripts/autoreview --mode branch --base origin/develop
```

If an open PR exists, prefer its actual base:

```bash
base=$(gh pr view --json baseRefName --jq .baseRefName)
.agents/skills/oc-autoreview-adapted/scripts/autoreview --mode branch --base "origin/$base"
```

Committed single change:

```bash
.agents/skills/oc-autoreview-adapted/scripts/autoreview --mode commit --commit HEAD
```

Do not force local mode after committing. A clean local review only proves there
is no dirty patch.

## Parallel Closeout

It is OK to run focused tests concurrently with review after formatting-sensitive
work is done:

```bash
.agents/skills/oc-autoreview-adapted/scripts/autoreview \
  --parallel-tests "uv run pytest tests/test_pop_select.py"
```

If tests or review findings lead to edits, rerun the affected tests and rerun
autoreview. Stop when the final helper run exits 0 with no accepted/actionable
findings. Do not run another review only for cleaner wording.

## EEGPrep Review Surface

The helper prompt asks Codex to prioritize:

- correctness bugs, import/runtime failures, wrong numerical results, and broken
  common workflows;
- EEGLAB parity in APIs, `pop_*` wrappers, history commands, GUI behavior, event
  semantics, and expected data structures;
- EEG dict fields including `data`, `nbchan`, `pnts`, `trials`, `srate`,
  `xmin`, `xmax`, `times`, `chanlocs`, `event`, `urevent`, `epoch`, `history`,
  `icaact`, `icawinv`, `icasphere`, `icaweights`, and `icachansind`;
- MATLAB/Python indexing boundaries, especially 1-based EEGLAB latencies and
  user-facing indices versus 0-based Python arrays;
- channel-major shape assumptions: continuous `(nbchan, pnts)` and epoched
  `(nbchan, pnts, trials)`;
- GUI plus `eegprep-console` synchronization through `EEGPrepSession`;
- `return_com=True`, `(EEG, com)` returns, history strings, and session update
  paths for user-facing `pop_*` functions;
- runtime independence from `src/eegprep/eeglab/`;
- packaged Markdown help resources for GUI Help or `pophelp`;
- missing tests tied to changed behavior;
- concrete security, path, file I/O, and dependency risks;
- realistic EEG-size performance regressions.

## Triage Findings

Accept findings only when they are concrete and introduced or exposed by the
reviewed change. Reject:

- pre-existing issues outside the diff;
- generic linter/formatter comments;
- broad refactors and speculative abstractions;
- unlikely edge cases that would complicate the code without protecting real
  workflows;
- subjective MATLAB-vs-Python style preferences that do not break EEGPrep's
  parity contract.

For each accepted finding, fix the smallest ownership boundary that addresses
the bug. For each rejected finding, record the reason briefly in the final
report. Add an inline code comment only when it documents a real invariant that
future reviewers need to know.

## Final Report

Include:

- review command used;
- tests/proof run;
- findings accepted, fixed, or rejected, briefly why;
- the clean result from the final helper run, or the exact remaining risk if a
  finding was consciously left open.

If the final helper run exits 0 and prints
`autoreview clean: no accepted/actionable findings reported`, report that run as
clean and stop.
