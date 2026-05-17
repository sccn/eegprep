---
name: eegprep-feature-development
description: Develop new EEGPrep features end to end with EEGLAB parity, AGENTS.md compliance, planning, implementation, tests, GUI visual parity, GUI user-flow QA, review, and PR creation. Use when a user asks Codex to build or port an EEGPrep feature, especially pop_* functions, GUI components, or EEGLAB-parity workflows.
---

# EEGPrep Feature Development

Use this workflow when building a new EEGPrep feature.

## Workflow

1. Switch to `origin/develop`, pull from `origin`, and create a new branch with
   an appropriate name.

2. For every feature to be developed, look at EEGLAB first and find equivalent
   implementations if they exist. Go through those implementations thoroughly.
   Match functionality and UX where relevant, and inspect every plausible user
   flow and edge case.

3. Plan how to build the features within EEGPrep's existing structure. Read and
   follow `AGENTS.md`. While planning, switch to plan mode, ask the user any
   non-obvious questions, and finish this step with a concrete plan for the
   requested features.

4. Execute the plan and write code. Maintain current coding conventions and
   follow `AGENTS.md`. Write tests as described there. Aim for more than 90%
   coverage for the changed feature code, and ensure the tests pass.

5. For features that involve a GUI component, use the
   [`eeglab-gui-visual-parity`](../eeglab-gui-visual-parity/SKILL.md) skill to
   iteratively develop the GUI so the UI/UX is familiar to EEGLAB users. Follow
   EEGPrep's current GUI conventions, keep performance and user experience high,
   and do not take shortcuts.

6. Simulate how users would exercise each feature with the
   [`gui-agent-flow-qa`](../gui-agent-flow-qa/SKILL.md) skill and Codex's GUI
   Agent. Cover typical user flows, including loading data from `sample_data`,
   and use every feature developed in the branch as a user would. Also cover
   edge cases that are likely from a user's point of view. Fix any bugs that
   surface.

7. After implementation and GUI parity/QA work, write additional regression
   tests and integration tests for behaviors discovered during testing.

8. Review the current feature branch against `origin/develop`. Use the
   [`github-pr-review`](../github-pr-review/SKILL.md) skill when appropriate.

9. Act on review findings when they make sense and are not unreasonable scope
   creep. Do not take shortcuts while addressing findings. If any findings are
   not fixed, report them to the user at the end. If GUI features are involved,
   return to Step 5, then repeat Step 8 until the branch looks good.

10. When implementation matches the plan and all required tests succeed, create
    a PR to `origin/develop` as described in `AGENTS.md`. The PR must include
    all features requested by the user.

11. After creating the PR, tell the user everything that was done, especially
    anything that should be flagged for their attention.
