---
name: oc-autoreview-adapted
description: Run autonomous EEGPrep-focused structured autoreview on dirty changes, branches, commits, PR stacks, or the whole EEGPrep-owned codebase; verify and fix real findings from first principles using AGENTS.md, EEGLAB parity, GUI/console, tests, docs, and security constraints.
---

# OC Autoreview Adapted

Use the bundled helper for high-signal closeout review or whole-codebase bug hunts. It builds one bounded review bundle, runs one or more read-only reviewer engines, validates structured JSON, prints heartbeats for long runs, and exits nonzero when actionable findings remain.

## Contract

- Run it for real unless the user asked only for a plan.
- Treat output as advisory. Verify every accepted finding in the real code path before fixing or reporting it.
- Accept concrete bugs, regressions, EEGLAB parity breaks, unsafe I/O/security risks, missing tests tied to behavior, and maintainability issues that cause real future defects.
- Accept structural findings when the code becomes harder to ship: spaghetti branching, wrong ownership layer, duplicate canonical helpers, non-atomic state updates, file sprawl, weak data boundaries, or indirection that hides EEG invariants.
- Reject speculative edge cases, broad rewrites, stale vendored/reference code, generic lint, subjective MATLAB/Python style comments, and "cleaner someday" feedback without a concrete failure mode.
- If a fix changes code, run focused tests and rerun autoreview on the same target. Stop when the final helper run exits 0 or when a remaining finding is consciously rejected with a concrete reason.
- Do not invoke nested review tools from inside review. The helper already runs one structured review path.
- Do not push/stage/commit/open PR unless the user requested that separately.
- If the user asks to review the whole codebase and wants fixes, default to the campaign workflow below: branch/worktree per codebase area, review/fix/test/rerun, then open a PR before moving to the next area.

## Commands

Set paths once:

```bash
export AUTOREVIEW=".agents/skills/oc-autoreview-adapted/scripts/autoreview"
export AUTOREVIEW_HARNESS=".agents/skills/oc-autoreview-adapted/scripts/test-review-harness"
export AUTOREVIEW_CAMPAIGN=".agents/skills/oc-autoreview-adapted/scripts/new-review-campaign.py"
```

Dirty local work:

```bash
"$AUTOREVIEW" --mode local
```

Branch or stacked PR work:

```bash
base=$(gh pr view --json baseRefName --jq .baseRefName 2>/dev/null || echo develop)
"$AUTOREVIEW" --mode branch --base "origin/$base"
```

Single committed change:

```bash
"$AUTOREVIEW" --mode commit --commit HEAD
```

Whole EEGPrep-owned codebase audit:

```bash
"$AUTOREVIEW" --mode codebase --thinking codex=xhigh
```

The codebase mode is not diff-limited. It lists tracked EEGPrep-owned files and excludes vendored EEGLAB/reference sample data by default; the reviewer may inspect files read-only and report real bugs anywhere in scope.

Scoped codebase audit:

```bash
"$AUTOREVIEW" --mode codebase \
  --path src/eegprep/functions/popfunc \
  --path tests/test_pop_utils.py \
  --thinking codex=xhigh
```

Use `--scope-file scopes.txt` when a slice has many paths.
Quote any `--path` value that contains shell globs, e.g. `--path 'tests/test_pop_*.py'`.

## Default Whole-Codebase Campaign

When asked to review the whole codebase hands-off, do not make one giant PR. Split work into PR-sized areas, usually:

- `popfunc`: `src/eegprep/functions/popfunc`, matching pop tests/help.
- `sigproc`: `src/eegprep/functions/sigprocfunc`, numerical/parity tests.
- `gui-session`: `src/eegprep/functions/guifunc`, `adminfunc`, console/session tests.
- `plugins`: `src/eegprep/plugins`, bundled plugin tests/resources.
- `io-bids-study`: file I/O, BIDS, STUDY, dataset/session persistence.
- `cli-docs-tools`: CLI, docs, skills, tools, workflows.

Start by scaffolding an orchestration artifact:

```bash
uv run python "$AUTOREVIEW_CAMPAIGN" "EEGPrep whole-codebase autoreview"
```

This creates `.workflow/<slug>/` with `plan.md`, `state.json`, `orchestration.md`, `packets/`, `results/`, and `final-report.md`. Keep `plan.md` human-readable, update `state.json` as packet status changes, and write integration evidence in `final-report.md`.

For each area:

1. Create a fresh worktree/branch from the requested base, e.g. `autoreview/popfunc`.
2. Run `autoreview --mode codebase --path ... --thinking codex=xhigh`.
3. Verify each finding from first principles. Fix real issues even when the fix touches a related helper outside the initial path scope; keep the PR conceptually tied to that area.
4. Run focused tests, lint/type checks when relevant, then rerun the same scoped autoreview until clean or until remaining findings are rejected with reasons.
5. Commit, push, and open a PR before starting the next area.

PR body must list every finding reviewed:

- **Fixed:** finding, root cause, files changed, tests run.
- **Rejected:** finding, why it is not real or not worth changing.
- **Follow-up:** only when real but intentionally outside this PR's area.

Do not auto-merge. The human reviews each PR normally.

## Parallel Subagents

Use parallel agents by default for whole-codebase campaigns when the environment exposes subagent/thread/worktree tools and the user has asked for hands-off or parallel work.

- Launch at most 3 packet agents at once unless the user approves more.
- Give each packet agent its `packets/<id>.md`, base branch, branch name, path scope, test commands, AGENTS.md constraints, and PR-body requirements.
- Packet agents may edit related files outside their path scope only when required by the verified root cause; they must explain that in the PR.
- Do not duplicate work across agents. If a packet blocks on another packet's result, keep it pending.
- Parent agent owns integration: track packet PR URLs, inspect conflicts, synthesize accepted/rejected findings, and run broader checks after packet PRs merge.
- If no subagent runner is available, simulate packets sequentially and write packet notes under `results/`.
- Do not claim that a script launched subagents. The campaign script only scaffolds orchestration; actual subagents require exposed agent/thread tools.

## Useful Options

- `--engine codex|claude|droid|copilot`; default is Codex.
- `--reviewers codex,claude` or `--panel` for a multi-reviewer pass.
- `--model codex=gpt-5.1 --thinking codex=xhigh`; Claude also accepts `max`.
- `--stream-engine-output` to see compact live engine activity.
- `--parallel-tests "uv run pytest tests/test_file.py"` to run tests while review runs.
- `--prompt` / `--prompt-file` / `--dataset` to add evidence.
- `--path` / `--scope-file` to constrain a codebase, branch, local, or commit review to a PR-sized area.
- `--json-output /tmp/review.json` and `--output /tmp/review.txt` for artifacts.
- `--mode uncommitted` is an alias for `local`; use branch/commit modes after committing.
- `--skip-fetch` avoids fetching before branch diffs.
- `--heartbeat-seconds 60` controls long-run heartbeat cadence.

Smoke check:

```bash
"$AUTOREVIEW_HARNESS" --dry-run
"$AUTOREVIEW_HARNESS" --fixture buggy --engine codex
```

On Windows, use:

```powershell
python .agents\skills\oc-autoreview-adapted\scripts\autoreview --help
.agents\skills\oc-autoreview-adapted\scripts\test-review-harness.ps1 -Fixture buggy -Engine codex
```

## EEGPrep Review Surface

Prioritize:

- correctness, runtime/import failures, bad numerical results, broken common workflows;
- EEGLAB parity in APIs, `pop_*` wrappers, history commands, GUI layout/behavior, events, and data structures;
- EEG dict invariants: `data`, `nbchan`, `pnts`, `trials`, `srate`, `xmin`, `xmax`, `times`, `chanlocs`, `event`, `urevent`, `epoch`, `history`, ICA fields;
- 1-based EEGLAB user indices/latencies versus 0-based Python array indices;
- channel-major continuous `(nbchan, pnts)` and epoched `(nbchan, pnts, trials)` data;
- GUI plus `eegprep-console` synchronization through `EEGPrepSession`;
- `return_com=True`, `(EEG, com)` returns, history replay, and session update paths;
- runtime independence from `src/eegprep/eeglab`;
- packaged Markdown help for GUI Help / `pophelp`;
- realistic EEG-size performance and concrete security/path/I/O risks.

## Architecture Bar

Use this only for code that will make EEGPrep less reliable or maintainable, not for taste.

- Look for a simpler "code judo" move that preserves behavior while deleting branches, modes, helper layers, or special cases.
- Flag spaghetti growth: ad-hoc conditionals in busy flows, scattered feature checks, one-off booleans, nullable modes, and partial updates.
- Keep logic in the canonical layer: signal processing in `sigprocfunc`, user wrappers in `popfunc`, GUI/session coordination in `guifunc`/`adminfunc`, plugin code in its plugin package, CLI orchestration outside core math.
- Prefer existing helpers/contracts over near-duplicates; remove thin wrappers or generic magic that hide simple EEG data shapes.
- Treat file-size growth past roughly 1000 lines as a warning in diff review; in whole-codebase audits, flag large modules only with a concrete bug-prone coupling or focused split.
- Prefer fixes that remove concepts, collapse duplicate branches, clarify data boundaries, or make state/session/history updates atomic.

## Loop

1. Format first if formatting can change line locations.
2. Run autoreview on the smallest sufficient target.
3. Verify each finding against code and AGENTS.md.
4. Fix accepted findings at the right ownership boundary.
5. Run focused tests, then broader tests if risk warrants.
6. Rerun the same autoreview target.
7. Final response: command used, tests run, findings fixed/rejected, and final clean result or remaining risk.

If the helper prints `autoreview clean: no accepted/actionable findings reported` and exits 0, report that as clean and stop.
