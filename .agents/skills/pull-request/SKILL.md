---
name: pull-request
description: Authoring pull requests in sccn/eegprep. Use when creating or updating a PR, and whenever changing a branch that is already associated with a PR.
---

# Skill: Author a Pull Request

This skill defines the exact output format for pull requests. Follow it
literally when creating or updating a PR. Also apply it whenever you make
changes to a branch that already has an associated PR.

When a branch is already associated with a PR, check the current PR title and
description after your code changes. Keep them aligned with the actual scope of
the branch, and update them if they are no longer correct or sufficient.

## PR Description Format

The PR description becomes the squash-merge commit message. Write it as
plain text — no markdown.

**Title:** Short imperative sentence. Optional scope tag in brackets.

**Body:** 1-3 sentences stating what changed and why. End with issue link if one exists.

**Hard rules — violations will be rejected:**

- No markdown: no headers (`##`), no bullet lists (`-`/`*`), no tables, no images, no `[text](url)` links.
- No checkboxes (`- [ ]`, `- [x]`).
- No section headers like `## Summary`, `## Test plan`, `## Changes`.
- No filler phrases: "This PR...", "I noticed...", "Summary of changes:".
- No emoji.
- Under ~80 words total.

### Example (follow this exactly)

```
Title: [EEG] Fix pop_loadset HDF5 path handling

Body:
Ensure pop_loadset resolves HDF5 sidecar paths relative to the .set file
location. This prevents missing file errors when datasets are moved between
folders. Adds a regression test covering relative paths.

Fixes #1234
```

## Issue Linking

If the work originated from a GitHub issue, reference it:

- `Fixes #NNNN` — auto-closes the issue on merge.
- `Part of #NNNN` — for partial work.

Do not create an issue solely to satisfy this rule. When there is no
pre-existing issue (e.g., user-directed work in a conversation), omit the
issue link.

## Pre-Push Checklist

Run these before pushing. Do not skip any step.

1. `./pre-commit.py --fix` — resolve all issues. Do not substitute `uv run pre-commit ...`.
2. `python -m unittest discover -s tests` — relevant test directories.

After pushing, monitor CI: `gh pr view <number> --json statusCheckRollup`.
Fix failures before considering the PR complete.

## Specifications (>500 LOC)

PRs over ~500 lines must include a specification. Put it in (preferred order):

1. The associated GitHub issue
2. First PR comment after the description

A specification contains:

1. **Problem** — what is broken or missing, with file/line references.
2. **Approach** — which modules change, what gets added/removed.
3. **Key code** — 10-30 line snippets for non-obvious logic.
4. **Tests** — what is tested, how, and why sufficient.

## Creating the PR

Unless the user says otherwise, and when permissions allow, push directly to a
branch on the main repository and open the PR from that branch. Do not default
to pushing to a fork. Use a fork only when direct push to the main repository
is not available or the user explicitly asks for it.

Use `gh pr create` with these flags:

```bash
gh pr create \
  --title "<title>" \
  --body "<plain text body>"
```

- Add the `agent-generated` label only when the PR is created by a repository
  automation workflow. Do not add it when a human asks an agent to create or
  update the PR from an interactive session.
- Never credit yourself in commits or PR descriptions.
- Include `Fixes #NNNN` when addressing a pre-existing issue.

## See Also

- `.agents/skills/github-pr-review/` — PR review skill (separate concern)
- `.agents/skills/fix-issue/` — end-to-end issue fix workflow
- `AGENTS.md` — coding guidelines
