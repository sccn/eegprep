---
name: github-pr-review
description: Review a pull request for correctness, EEGLAB parity, and behavioral regressions in sccn/eegprep.
allowed-tools: Bash(gh issue view:*), Bash(gh search:*), Bash(gh issue list:*), Bash(gh pr comment:*), Bash(gh pr diff:*), Bash(gh pr view:*), Bash(gh pr list:*), Bash(git rev-parse:*), Bash(git show:*), Bash(git diff:*), mcp__github_inline_comment__create_inline_comment
---

# Review a pull request

Review the requested pull request at the exact head named by the caller. The
review is read-only except for comments posted when `--comment` is set.

Do not launch subagents or background agents. Do the whole review in this
session. Parallel agent fan-out is forbidden: headless claude-code-action
exits green while children are still running and posts nothing.

Use `gh` for GitHub reads and for the top-level comment. Do not use web fetch.
The pull request number is the one the caller named; `--comment` means post
that review.

## Review standard

A reviewable EEGPrep pull request:

- Implements the behavior claimed by its title and description without an
  introduced correctness, security, or error-handling defect.
- Preserves numerical parity with EEGLAB for changed signal-processing
  behavior. Where EEGLAB is mathematically wrong, EEGPrep does the correct
  thing and documents and tests the difference. A silent divergence, a wrong
  numerical result, or a divergence that is neither documented nor tested is a
  defect. A documented and tested correction is not.
- Preserves EEG dict invariants. Core fields include `data`, `nbchan`, `pnts`,
  `trials`, `srate`, `xmin`, `xmax`, `times`, `chanlocs`, `event`, `urevent`,
  `epoch`, `history`, `icaact`, `icawinv`, `icasphere`, `icaweights`, and
  `icachansind`. Continuous data is `(nbchan, pnts)`; epoched data is
  `(nbchan, pnts, trials)`. EEGLAB event latencies and many user-facing indices
  are 1-based; Python arrays are 0-based internally. Check the first sample,
  the last sample, and boundary events.
- Keeps GUI behavior predictable for EEGLAB users: alignment, arrangement,
  labels, buttons, text fields, control order, default values, enabled and
  disabled states, and workflow. GUI and `eegprep-console` share one
  `EEGPrepSession`, so dataset selection, history, and visible state stay
  synchronized in both directions.
- Follows the root and path-scoped `CLAUDE.md` and `AGENTS.md` instructions.
  Comments explain non-obvious intent or EEGLAB parity only. A comment that
  restates the code is a defect.
- Uses helpful errors. State what is wrong, which function failed, and the
  exact fix, such as `pip install eegprep[x]`.
- Tests changed behavior, including EEGLAB parity where that behavior is
  numerical. Name the missing case. Do not request tests for coverage. Do not
  accept loosened tolerances. `tests/fuzz/tolerances.toml` is human-owned;
  loosening it to hide a mismatch is a blocking defect.
- Meets the `AGENTS.md` testing-evidence bar. A pull request that changes core
  EEGLAB-ported processing must include the MATLAB reference-suite result in
  the description: the exact command or workflow run, the pinned EEGLAB and
  eeglab_tests commits, pass/fail/skip counts, runtime, and an explanation of
  any failures or skips. Tests, docs, CI, and tooling-only changes do not need
  that run. Console-visible session, history, import-alias, or bare `pop_*`
  auto-store changes need `tests/test_console_workspace.py` coverage.
  User-facing behavior needs the matching Sphinx update, and a new Help button
  needs `src/eegprep/resources/help/<function_name>.md`.
- Stays small. Match existing EEGPrep style. Do not request speculative
  abstractions, broad refactors, or unrelated cleanup. Flag unrelated dead code
  as out of scope. Prefer idiomatic Python where MATLAB style would make the
  code worse, and explain that tradeoff when it changes user-facing behavior.
- Considers realistic EEG data sizes when a changed path is actually expensive,
  and flags unsafe file, path, or input handling in changed code.

Report only actionable defects introduced by the pull request. Validate every
finding against the diff, the surrounding code, and the applicable
instructions. If the evidence is uncertain, omit the finding.

Review in this order: correctness and EEGLAB parity, data-structure
compatibility, bugs and invalid inputs, tests for changed behavior, unsafe
inputs, realistic performance, then simplicity.

## High-signal threshold

Flag an issue only when:

- The code fails to parse, import, or run on a common path.
- The code produces wrong results for realistic inputs.
- The change breaks expected EEGLAB numerical or GUI parity, or mishandles an
  EEGPrep data-structure invariant.
- Changed behavior lacks the test that would catch a likely regression.
- Changed code handles a file, path, or input unsafely.
- A `CLAUDE.md` or `AGENTS.md` rule scoped to the changed file is clearly
  broken, and you can quote the rule.

Do not flag pre-existing issues, pedantic nits, linter findings, generic
quality or security or performance warnings, speculative failures, subjective
style, or broad refactors. Do not flag an instruction that the changed code
explicitly silences. False positives waste reviewer time.

## Workflow

1. Inspect the pull request metadata and issue comments. Stop without
   commenting when it is closed, a draft, an automated pull request, or a
   trivial change that is obviously correct. Still review agent-generated and
   Claude-generated pull requests. Stop without commenting when an issue
   comment contains `<!-- eegprep-claude-review -->` unless this run was
   explicitly asked to review again. A comment body that contains
   `claude review this` is an explicit request.

2. Confirm the checked-out commit equals the pull request head. Compare
   `git rev-parse HEAD` with `gh pr view <PR> --json headRefOid`. If they
   differ, report the stale review and post nothing.

3. Read the root `CLAUDE.md` and `AGENTS.md`. Read every `CLAUDE.md` and
   `AGENTS.md` that sits in a changed file's directory or a parent of that
   directory. When judging a file, apply only the instructions that scope that
   file. Read the complete pull request diff and enough surrounding code to
   validate behavior and call sites. Use `gh pr diff` and `git show`.

4. Review the change against the standard above. For each finding, record the
   changed file and line, severity (`blocking`, `important`, or `nit`),
   concrete impact, and the evidence that makes it certain. Record test-gap
   and pull-request-description problems separately when they are not tied to
   one line.

5. Print the findings. For each one, include severity, file and line, impact,
   and evidence. If there are none, print:

   `No issues found. Checked for bugs, EEGLAB parity, data structure compatibility, tests, and CLAUDE.md/AGENTS.md compliance.`

   Without `--comment`, stop here. Do not post any GitHub comment.

6. With `--comment`, post one top-level summary, or update the progress
   comment, using:

   ```bash
   gh pr comment "$PR_NUMBER" --edit-last --create-if-none --body-file <file>
   ```

   The body uses the format below and ends with `<!-- eegprep-claude-review -->`.
   Then post one inline comment per unique high-signal finding through
   `mcp__github_inline_comment__create_inline_comment` with `confirmed: true`.
   Do not post an inline comment for a summary-only note, a general test gap,
   or a description problem unless a specific changed line is the defect.

## Summary comment

```markdown
## Code review

- Overall assessment: <safe to merge / needs changes / needs more context>
- Highest-risk area: <area>
- Merge recommendation: <safe to merge / needs changes / needs more context>

## Blocking

<findings or None.>

## Important

<findings or None.>

## Nits

<findings or None.>

## Test gaps

<findings or None.>

## EEGLAB parity notes

<findings or None.>

<!-- eegprep-claude-review -->
```

Write `None.` for an empty section. Each finding names the file and line, the
impact, and a concrete fix. Keep it short. Do not add generic praise. When no
issues were found, use:

```markdown
## Code review

- Overall assessment: Looks good.
- Highest-risk area: None identified.
- Merge recommendation: Safe to merge.

## Blocking

None.

## Important

None.

## Nits

None.

## Test gaps

None.

## EEGLAB parity notes

None.

Checked for correctness bugs, EEGLAB parity, data structure compatibility, changed-behavior tests, and CLAUDE.md/AGENTS.md compliance.

<!-- eegprep-claude-review -->
```

## Inline comments

Each inline comment states the severity, the impact, and a concrete fix. Include
a committable suggestion only when committing that suggestion fixes the issue
entirely and the fix is under six lines in one place. For a larger or
multi-location fix, describe the fix and do not include a suggestion block. Post
one comment per unique issue. Do not repeat a point the summary already covers
adequately.

When an inline comment cites code, `CLAUDE.md`, `AGENTS.md`, or EEGLAB
reference material, link the repository under review with the full head SHA:

```markdown
https://github.com/sccn/eegprep/blob/<full-sha>/<path>#L<start>-L<end>
```

Use the full git SHA. Do not use a command substitution inside the link. Put
`#` immediately after the path. Use at least one line of context before and
after the line you are discussing.
