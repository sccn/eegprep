---
name: eegprep-project-management
description: Plan EEGPrep multi-phase GitHub epics, phase issues, epic branches, merge order, and agent-ready /goal prompts for EEGLAB parity or major product work. Use when the user asks to organize EEGPrep work as an epic, create phase issues, plan parallel agent streams, or set up an epic branch.
---

# EEGPrep Project Management

Use this skill for EEGPrep project planning that is too large for one PR.
The output should make independent phase agents productive without requiring
extra product decisions.

## Grounding

Before planning, inspect:

- `AGENTS.md`
- relevant `.notes/` audits or implementation notes
- open issues/PRs for overlapping work
- current branches and the intended base branch
- relevant skills, especially `eegprep-feature-development`,
  `eeglab-gui-visual-parity`, `gui-agent-flow-qa`, and `pull-request`

Use EEGLAB source as the parity oracle, but keep EEGPrep standalone. Do not plan
runtime dependencies on `src/eegprep/eeglab`.

## Epic Shape

An epic is a multi-phase feature. Each phase must be independently reviewable
and mergeable into the epic branch.

For each epic, define:

- parent issue title and branch name
- final target branch, usually `origin/develop`
- phase PR target, always the epic branch
- scope and explicit non-goals
- phase list with acceptance criteria
- merge order and what can run in parallel
- epic-level test and integration evidence

Prefer 4-8 phases. Split a phase when it blocks parallel work or combines
unrelated failure modes. Do not split merely to make a checklist longer.

## Phase Issue Requirements

Each phase issue should include:

- summary and why it exists in the epic
- dependency/parallelization notes
- concrete scope, including examples of functions/workflows
- acceptance criteria covering API, GUI, console, docs, tests, and parity where relevant
- required references to `AGENTS.md` and applicable skills
- a ready-to-run `/goal` prompt

For EEGPrep parity work, phase agents must:

- inspect matching EEGLAB implementation first
- avoid stale or unused MATLAB bloat
- preserve GUI plus `eegprep-console` synchronization
- add MATLAB parity tests where deterministic
- attach GUI side-by-side images as GitHub user attachments for GUI changes
- update help Markdown for user-facing functions

## `/goal` Prompt Pattern

Write `/goal` prompts as outcome contracts, not task lists. Include:

- target outcome
- source of truth, such as EEGLAB files or an audit matrix
- implementation constraints, especially standalone runtime behavior
- verification surface: focused tests, MATLAB parity, visual parity, GUI flow QA,
  ruff, format, ty, and pre-commit
- iteration policy: review findings should be fixed unless they are scope creep,
  and unfixed findings must be reported

## GitHub Workflow

Use one tracker: GitHub issues and sub-issues. No separate planning document is
required unless the user asks for one.

Typical commands:

```bash
git fetch origin
git switch <base-branch>
git pull --ff-only
git switch -c <epic-branch>
git push -u origin <epic-branch>

gh label create epic --color 5319e7 --description "Multi-phase implementation epic" || true
gh label create phase --color 1d76db --description "Epic implementation phase" || true
gh label create P1 --color d93f0b --description "High priority" || true

gh issue create --title "<Epic title>" --label epic --label P1 --body-file /tmp/epic.md
gh issue create --title "<Phase title>" --label phase --body-file /tmp/phase.md
gh sub-issue add <EPIC_NUMBER> --sub-issue-number <PHASE_NUMBER>
```

After creating sub-issues, edit the parent issue to list phase issue numbers
and verify:

```bash
gh sub-issue list <EPIC_NUMBER>
```

Agent comments on issues or PRs must begin with `🤖`.

## Closeout

After all phase PRs merge into the epic branch:

- pull the latest epic branch
- resolve integration conflicts from first principles
- run the epic-level test plan
- run adversarial review when requested
- fix real findings
- open the final epic PR to `origin/develop`

The final PR should summarize phases, link phase issues/PRs, explain major
features, include integration evidence, and attach GUI side-by-side images as
GitHub user attachments when GUI changed.
