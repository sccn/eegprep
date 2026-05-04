---
name: fix-issue
description: End-to-end workflow for fixing a GitHub issue. Use when asked to fix, investigate, or resolve a GitHub issue in sccn/eegprep.
---

# Skill: Fix GitHub Issue

You are to fix the Github issue indicated by the user.

## Background

Read first:

@AGENTS.md

You may not proceed to a new task until you have completed all prior tasks. If
you cannot complete a task, write a Github comment with your last status and why
you failed. A task list is at the end of this document.

## Writing Style for All GitHub Comments

**Be terse.** Every sentence must convey information that the reader does not
already have. Follow these rules in all comments you post:

- No preamble, no filler, no editorializing ("I've thoroughly investigated...",
  "After careful analysis...", "This is an interesting problem...").
- No repeating information already in the issue body.
- No restating what code does when a link suffices.
- Max 3-4 sentences of prose per comment section. Use links and code, not words.
- Annotate code links, don't narrate them. Bad: "The key/value arg validation is
  in eegobj.py where it checks for paired inputs".
  Good: `src/eegprep/eegobj.py#L65-L71` - key/value arg validation.

## Research

Use `gh` to fetch the issue. Read the codebase to find all relevant source
files. Post a single comment combining your research and proposed fix.

Before you write code, run a duplicate-work preflight:

- Check open pull requests for the same issue or subsystem (`gh pr list --state open --search "<issue-id or keyword>"`).
- Check open issues for the same root cause (`gh issue list --state open --search "<keyword>"`).
- If overlapping work already exists, do not open a parallel implementation PR.
  Add a short issue comment summarizing what you found and either:
  - hand off to the existing PR, or
  - scope your change explicitly to non-overlapping follow-up work.

The comment has two sections: **Research** and **Proposed Fix**.

### Research section

Title: `# Research`

- 2-3 sentences: what's broken/missing and why (root cause, not symptoms).
- A `**Relevant code**` list: max 5 links, each with a short (<10 word) annotation.

Example:

> `EEGobj.__init__()` routes path strings through `pop_loadset()`. When
> `scipy.io.loadmat()` fails, `pop_loadset()` falls back to `pop_loadset_h5()`,
> which opens the HDF5 file without a context manager, so repeated loads can
> leak file handles. Root cause: `h5py.File(...)` is never closed.
>
> **Relevant code**
> - [`src/eegprep/eegobj.py#L17-L29`](src/eegprep/eegobj.py#L17-L29) - path to pop_loadset
> - [`src/eegprep/popfunc/pop_loadset.py#L80-L88`](src/eegprep/popfunc/pop_loadset.py#L80-L88) - loadmat fallback
> - [`src/eegprep/popfunc/pop_loadset_h5.py#L20-L26`](src/eegprep/popfunc/pop_loadset_h5.py#L20-L26) - h5py file open

### Proposed Fix section

Title: `# Proposed Fix`

There are 2 cases:

* **Bug fix**: Show the call chain that triggers the error, then state the fix
  in one sentence. Include a code snippet only if the fix is non-obvious.

  > `env.run()` returns a tuple; `foo()` assumes an object and calls `.abc()`.
  > Fix: unwrap the tuple in `RolloutWorker._step()` before passing to `foo()`.

* **Design change**: The design doc replaces the `# Proposed Fix` section.
  Write it directly in the issue comment (agents cannot commit files before the PR). The 3-4 sentence prose cap applies to `# Research`; the design doc can be around a thousand words. As required, Use an Explore subagent to gather relevant files/lines, related issues/PRs, and reusable utilities. Summarize findings, note gaps, and save a research note. Ask 3–6 targeted questions: scope boundaries, tests, tradeoffs, open questions. Avoid questions answered by code search. Then draft the design doc as a spec. Keep it concise, cite real files/lines, include non-empty Open Questions, and avoid backward-compat unless needed. Iterate until settled. Send design + spec to a reviewer subagent. Incorporate clear improvements, and ask the user only about ambiguous decisions. Keep everything in one comment.

In both cases: find the minimally disruptive fix. Do not over-engineer.

## Implementation

Once you have researched and prepared your design or planned fix, you may
proceed to implementation. You MUST implement your changes in a branch with the
following format `agent/{YYYYMMDD}-fix-{issue-id}`. You are allowed to create
branches on github as well as submit PRs from those branches.

You will implement a test which demonstrates your changed behavior before
beginning work. Your tests will be minimal and refrain from using mocks.

## Testing

* You write your test _before_ you make your fix, and validate your fix works.
* You use an existing test file in `tests/` if appropriate.
* You never use mocks when testing.
* You keep tests simple and minimal. You do not test obvious behavior like "object has an attr".
* You run `./pre-commit.py --fix` and resolve all reported issues before uploading.
* You will test using `python -m unittest discover -s tests` before uploading.
* You will ensure _all_ tests pass.

## Uploading

When all tests pass, you may proceed to upload your branch.
Open a pull request following `.agents/skills/pull-request/SKILL.md` **exactly**.
Read that file and use the plain-text format it specifies — no markdown headers,
no bullet lists, no `## Summary` sections. Violations will be rejected.
Attach it to the Github issue with a comment summarizing the fix.

## Verify CI Status

After opening the pull request, you must verify that CI checks pass:

* Monitor the PR using `gh pr view <number> --json statusCheckRollup`
* Wait for the unit tests to complete
* If tests fail, investigate and fix any issues
* Push additional commits to address CI failures
* Do not consider the PR complete until all relevant checks pass

Key checks to monitor:
- **unit tests**: Must pass - validates your changes don't break existing functionality
- **lint-and-format**: Must pass - ensures code style compliance
- **build-docs**: Should pass if you modified documentation

# Tasks

The tasks for this skill:

- [ ] Fetch issue information
- [ ] Research codebase for all relevant source files
- [ ] Run duplicate-work preflight against open PRs/issues
- [ ] Formulate fix and post Research + Proposed Fix comment
- [ ] Create branch for the changes
- [ ] Write a test case as needed for changes
- [ ] Implement changes until all tests pass
- [ ] Run `./pre-commit.py --fix` and resolve all issues
- [ ] Upload branch to github
- [ ] Open pull request
- [ ] Verify CI checks pass and address any failures (by polling gh pr view)
- [ ] Update comment with final status

If at any point you are unable to proceed, you must add a comment to the Github
issue with your last status.

# RULES

0. Never credit yourself in commits or comments.
