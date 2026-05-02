---
name: github-pr-review
description: Review a pull request for correctness, EEGLAB parity, and behavioral regressions in sccn/eegprep.
allowed-tools: Bash(gh issue view:*), Bash(gh search:*), Bash(gh issue list:*), Bash(gh pr comment:*), Bash(gh pr diff:*), Bash(gh pr view:*), Bash(gh pr list:*), Bash(git rev-parse:*), mcp__github_inline_comment__create_inline_comment
---

Provide a code review for the given pull request.

You are reviewing this PR for EEGPrep, a Python library that aims to port core EEGLAB concepts, workflows, code organization, GUI patterns, and user experience from MATLAB to Python.

## Review goals

Prioritize the following:

1. Catch correctness bugs, API mismatches, broken assumptions, edge cases, and regressions.
2. Check whether the implementation preserves EEGLAB parity where relevant.
3. Encourage simple, maintainable Python code that follows this repository's existing conventions.
4. Avoid generic comments. Only leave feedback that is specific, actionable, and tied to changed code.

## Review priorities

Review in this order:

1. Correctness and behavioral parity with EEGLAB
2. Data structure compatibility with EEGLAB concepts
3. Bugs, edge cases, error handling, and invalid inputs
4. Test coverage for changed behavior
5. Security or unsafe file, path, or input handling
6. Performance issues that are realistic for EEG data sizes
7. Code simplicity, readability, and maintainability

## EEGLAB context

Use this context when reviewing:

- EEGLAB repo: https://github.com/sccn/eeglab
- EEGLAB data structures guide: https://eeglab.org/tutorials/ConceptsGuide/Data_Structures.html
- Prefer conventions that make EEGPrep feel familiar to EEGLAB users, especially around naming, code modularization, data structures, GUI behavior, and user experience.
- Do not force MATLAB style blindly where idiomatic Python or existing EEGPrep style is clearly better.
- If there is a tradeoff between EEGLAB parity and idiomatic Python, explain the tradeoff clearly.

## Repository conventions

- Read and follow CLAUDE.md and AGENTS.md if available before reviewing.
- Match existing style and architecture unless the changed code introduces a clear problem.
- Prefer minimal code that solves the actual PR goal.
- Do not suggest speculative abstractions, broad refactors, or unrelated cleanup.
- If you notice unrelated dead code or unrelated refactoring, flag it as out of scope rather than asking the author to fix everything in this PR.

## Feedback rules

- Do not give generic compliments or generic warnings.
- Do not comment on unchanged code unless it directly affects the PR.
- Prefer fewer, higher-signal comments over a long checklist.
- Each issue should include:
  - Severity: blocking, important, or nit
  - File/function reference when possible
  - Why it matters
  - A concrete suggested fix
- If the PR looks good, say so and mention the specific areas checked.

## Test review

When reviewing tests:

- Identify which changed behaviors need tests.
- Check for parity tests against expected EEGLAB behavior where feasible.
- Suggest exact missing test cases rather than saying "add more tests."
- Do not request broad test coverage unless the missing coverage directly affects changed behavior.

## Agent assumptions

These assumptions apply to all agents and subagents:

- All tools are functional and will work without error.
- Do not test tools or make exploratory tool calls.
- Only call a tool if it is required to complete the task.
- Every tool call should have a clear purpose.
- Use gh CLI to interact with GitHub.
- Do not use web fetch.
- Create a todo list before starting.

## Review workflow

Follow these steps precisely.

### 1. Pre-review gate

Launch a haiku agent to check if any of the following are true:

- The pull request is closed.
- The pull request is a draft.
- The pull request does not need code review, such as an automated PR or a trivial change that is obviously correct.
- Claude has already commented on this PR, checked with `gh pr view <PR> --comments`, and the review was not explicitly requested via comment, such as "claude review this."

When a maintainer explicitly requests a re-review, always proceed even if a prior review exists.

If any stop condition is true, stop and do not proceed.

Still review Claude-generated PRs.

### 2. Find repository instructions

Launch a haiku agent to return a list of file paths, not contents, for all relevant CLAUDE.md and AGENTS.md files, including:

- Root CLAUDE.md and AGENTS.md files, if they exist
- Any CLAUDE.md or AGENTS.md files in directories containing files modified by the pull request
- Any CLAUDE.md or AGENTS.md files in parent directories of modified files

When evaluating compliance for a file, only consider CLAUDE.md and AGENTS.md files that are in that file's directory or parent directories.

### 3. Summarize the PR

Launch a sonnet agent to view the pull request and return:

- PR title
- PR description
- Changed files
- High-level summary of the change
- Areas most likely to affect correctness, EEGLAB parity, data structure compatibility, GUI/user experience, or tests

### 4. Independent review agents

Launch 4 agents in parallel. Each agent should receive:

- PR title
- PR description
- Changed files
- Relevant CLAUDE.md and AGENTS.md paths
- EEGPrep and EEGLAB review context from this skill

Each agent should return a list of issues. Each issue must include:

- Description
- Severity: blocking, important, or nit
- File/function reference when possible
- Reason it was flagged, such as bug, EEGLAB parity, CLAUDE.md adherence, data structure compatibility, unsafe input handling, or test gap
- Confidence level

#### Agent 1: CLAUDE.md/AGENTS.md compliance sonnet agent

Audit changes for CLAUDE.md and AGENTS.md compliance.

Only flag clear, unambiguous violations where the relevant instruction is scoped to the changed file.

#### Agent 2: Repository convention and maintainability sonnet agent

Audit whether the PR follows existing EEGPrep style, architecture, and conventions.

Only flag maintainability issues that are specific to the changed code and likely to matter.

Do not request speculative abstractions, broad refactors, or unrelated cleanup.

#### Agent 3: Opus correctness and regression agent

Scan for correctness bugs in the diff itself.

Focus on:

- Syntax errors
- Type errors
- Missing imports
- Unresolved references
- Logic that definitely produces wrong behavior
- Broken assumptions
- Edge cases introduced by the changed code
- Regressions in changed behavior

Flag only significant bugs. Ignore nitpicks and likely false positives.

Do not flag issues that cannot be validated without looking at context outside of the diff.

#### Agent 4: Opus EEGLAB parity and data structure agent

Review the introduced code for EEGPrep-specific risks.

Focus on:

- Behavioral mismatches with EEGLAB concepts
- Incorrect or incomplete EEG structure handling
- Incompatible assumptions about EEGLAB-style fields, dimensions, channel data, events, epochs, or metadata
- API behavior that would surprise EEGLAB users
- GUI or workflow changes that break expected EEGLAB-like behavior
- Performance concerns that are realistic for EEG data sizes

Only flag concrete, actionable issues tied to changed code.

## High-signal threshold

We only want high-signal findings.

Flag issues where:

- The code will fail to compile, parse, import, or run in a common path.
- The code will produce wrong results for realistic inputs.
- The implementation clearly breaks expected EEGLAB parity.
- The implementation clearly mishandles EEGPrep or EEGLAB-like data structures.
- A changed behavior lacks a test that is necessary to catch a likely regression.
- There is unsafe file, path, or input handling in changed code.
- There is a clear, unambiguous CLAUDE.md or AGENTS.md violation where you can quote the exact rule being broken.

Do not flag:

- Pre-existing issues
- Pedantic nitpicks that a senior engineer would not flag
- Issues that a linter will catch
- General code quality concerns
- Generic security warnings
- Generic performance warnings
- Potential issues that depend on unlikely inputs or speculative state
- Subjective style preferences
- Issues mentioned in CLAUDE.md or AGENTS.md but explicitly silenced in the code, such as via a lint ignore comment
- Broad refactors unrelated to the PR

If you are not confident an issue is real, do not flag it. False positives erode trust and waste reviewer time.

### 5. Validate candidate issues

For each issue found in step 4, launch a parallel validation subagent.

The validation subagent should receive:

- PR title
- PR description
- Changed files
- Relevant repository instruction files
- Description of the issue
- Reason it was flagged
- EEGPrep and EEGLAB context

Use:

- Opus subagents for correctness, logic, regression, EEGLAB parity, data structure, security, and performance issues
- Sonnet subagents for CLAUDE.md, AGENTS.md, test coverage, and repository convention issues

The validation subagent must determine whether the issue is truly valid with high confidence.

For example:

- If the issue is "variable is not defined," validate that the variable is actually undefined in the changed code.
- If the issue is an EEGLAB parity mismatch, validate that the changed behavior conflicts with an expected EEGLAB concept or workflow.
- If the issue is a CLAUDE.md violation, validate that the rule is scoped to the changed file and is actually violated.
- If the issue is a test gap, validate that the missing test corresponds to changed behavior and is not merely a general coverage request.

### 6. Filter issues

Filter out any issues that were not validated in step 5.

The remaining validated findings form the final review.

### 7. Terminal summary

Output a summary of the review findings to the terminal.

If issues were found, list each issue with:

- Severity
- File/function reference
- Brief description
- Why it matters

If no issues were found, state:

`No issues found. Checked for bugs, EEGLAB parity, data structure compatibility, tests, and CLAUDE.md/AGENTS.md compliance.`

If `--comment` argument was not provided, stop here. Do not post any GitHub comments.

If `--comment` argument was provided, continue.

### 8. Prepare review comment

Create a review comment body file.

The comment must use this format:

```md
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
````

Rules for the review comment:

* If there are no findings in a section, write `None.`
* Findings must be specific, actionable, and tied to changed code.
* Each finding should explain why it matters and give a concrete suggested fix.
* Keep the review concise.
* Do not include generic praise.
* If the PR looks good, say so in the overall assessment and mention the specific areas checked.

If no issues were found, use this format:

```md
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
```

Post or update the PR review comment using:

```bash
gh pr comment "$PR_NUMBER" --edit-last --create-if-none --body-file <file>
```

### 9. Prepare inline comments

Create a list of all inline comments you plan to leave. This list is only for internal review. Do not post it anywhere.

Post inline comments only for validated issues that are best attached to a specific changed line.

Do not post inline comments for broad summary items, general test gaps, or high-level EEGLAB parity notes unless there is a specific changed line where the issue belongs.

### 10. Post inline comments

Post inline comments using `mcp__github_inline_comment__create_inline_comment` with `confirmed: true`.

For each inline comment:

* Provide a brief description of the issue.
* Include severity.
* Explain why it matters.
* Suggest a concrete fix.
* For small, self-contained fixes, include a committable suggestion block.
* For larger fixes, meaning 6 or more lines, structural changes, or changes spanning multiple locations, describe the issue and suggested fix without a suggestion block.
* Never post a committable suggestion unless committing the suggestion fixes the issue entirely.
* If follow-up steps are required, do not leave a committable suggestion.
* Only post one comment per unique issue.
* Do not duplicate content already handled better in the summary comment.

## Linking requirements

You must cite and link each issue in inline comments when referring to files, code, CLAUDE.md, AGENTS.md, or EEGLAB reference material.

When linking to code in inline comments, follow this format precisely:

```md
https://github.com/sccn/eegprep/blob/e2e3d3971874cbf0102fc0d3e2a189277f999a1a/README.md#L1-L5
```

Requirements:

* Use the full git SHA.
* Do not use command substitutions such as `$(git rev-parse HEAD)` in Markdown links.
* Repo name must match the repo being reviewed.
* Use `#` after the file name.
* Line range format must be `L[start]-L[end]`.
* Provide at least one line of context before and after, centered on the line you are commenting about when possible.