---
name: thermo-nuclear-code-quality-review
description: Run an unusually strict EEGPrep code-quality review for architecture, maintainability, abstraction quality, file sprawl, spaghetti branching, and missed simplification. Use for thermo-nuclear review, thermonuclear review, deep maintainability audit, strict architecture review, or when code technically works but may make EEGPrep harder to ship.
---

# Thermo-Nuclear Code Quality Review

Use this for a demanding maintainability review, not a routine bug pass. The goal is to make EEGPrep shippable: standalone, EEGLAB-familiar, easy for EEG researchers, and structurally simple enough for future agents and humans to extend safely.

Inspired by Cursor's MIT-licensed `thermo-nuclear-code-quality-review` skill: https://github.com/cursor/plugins/blob/main/cursor-team-kit/skills/thermo-nuclear-code-quality-review/SKILL.md

## Contract

- Read `AGENTS.md` first and keep its EEGLAB parity, GUI/console, testing, docs, and style rules in force.
- Review the current diff, PR, branch, or named scope. Do not turn this into a whole-codebase rewrite unless asked.
- Be ambitious about simplification, but only flag structural issues with a concrete failure mode or clear maintainability cost.
- Prefer fewer high-conviction findings over a long list of taste comments.
- If asked to fix findings, verify each one from first principles, make the smallest durable change, run focused tests, and commit only when requested.
- Do not rubber-stamp code because tests pass. Passing behavior can still be architecturally wrong.

## Review Bar

Ask these questions for every meaningful change:

- Is there a simpler framing that deletes branches, modes, wrappers, flags, or helper layers?
- Did this add special cases to an already busy flow instead of moving logic to the owning module?
- Is the logic in the canonical EEGPrep layer?
- Does the code preserve EEG dict invariants and EEGLAB-facing semantics without hidden global state?
- Does GUI/menu/console code update `EEGPrepSession`, history, and visible state atomically?
- Is this abstraction earning its keep, or is it a pass-through wrapper?
- Did the change create duplicate helpers instead of reusing an existing contract?
- Did it make data boundaries weaker through unnecessary optionality, casts, duck typing, or silent fallbacks?
- Did a file cross or approach roughly 1000 lines because new concepts were not decomposed?
- Does the code remain understandable to an EEG researcher migrating from EEGLAB?

## EEGPrep Architecture Boundaries

Keep ownership clear:

- `popfunc`: user-facing `pop_*` wrappers, history strings, dialogs, `return_com=True`, and EEGLAB-compatible command surfaces.
- `sigprocfunc`: low-level signal processing and numerical transforms. No GUI/session orchestration here.
- `guifunc`: Qt/inputgui/menu rendering and GUI coordination. No low-level numerical algorithm ownership here.
- `adminfunc`: session, options, console, history, storage, and administrative runtime behavior.
- `plugins/*`: bundled plugin ports and plugin-owned helpers.
- `resources/help`: EEGPrep-owned Help Markdown for user-facing dialogs/functions.
- `eeglab/`: development reference only. Runtime code must not depend on it.

Flag layer leaks aggressively when they make future behavior harder to reason about.

## What To Flag

Flag issues such as:

- A "works but messy" implementation where a clear code-judo move would delete complexity.
- One-off booleans, nullable modes, or scattered feature checks.
- Repeated conditionals that indicate a missing helper, model, or dispatcher.
- Partial session/history/dataset updates that can leave GUI and console out of sync.
- EEGLAB user-facing indices mixed with Python 0-based indices without an explicit boundary.
- Channel-major EEG data assumptions hidden behind generic array handling.
- New runtime dependency on vendored EEGLAB, local paths, optional files, or unstated environment state.
- Thin wrappers, identity helpers, or generic magic that obscure simple EEG invariants.
- Copy-pasted parsing/history/dialog helpers when a canonical helper exists.
- Large-file growth that should be split into focused modules.
- Tests that only assert implementation details while missing user-observable behavior.

## Preferred Remedies

Prefer remedies that:

- Delete concepts rather than rename them.
- Collapse duplicate branches into one explicit flow.
- Move logic to the module that owns the concept.
- Extract small pure helpers for repeated parsing, shape, or indexing contracts.
- Make state transitions atomic through `EEGPrepSession` helpers.
- Make data boundaries explicit before converting between EEGLAB-facing and Python-facing indices.
- Split large modules by stable ownership, not by arbitrary line count.
- Replace loose optional/cast-heavy code with a concrete contract.
- Add focused tests for externally observable EEG dict, GUI/session, history, or file behavior.

## Tone And Output

Lead with findings. For each finding include:

- file and line;
- why it matters for correctness, parity, or maintainability;
- the concrete scenario or future failure mode;
- a preferred fix direction.

Order findings by severity:

1. Structural regressions that can create bugs or block maintainability.
2. Missed simplification that would remove significant complexity.
3. Wrong ownership/layering or canonical-helper duplication.
4. State/session/history atomicity risks.
5. File sprawl and decomposition concerns.
6. Lower-level legibility issues with real cost.

If there are no actionable issues, say so directly and mention any residual test or review limits.

Do not soften major structural problems into vague nits. Also do not invent architecture work without a real failure mode.
