# EEGPREP Agent Notes

- Goal: port EEGLAB to production Python without losing EEGLAB numerical, data-structure, workflow, or round-trip behavior.
- Conversion philosophy: preserve EEGLAB empirical/numerical behavior and familiar workflows for long-time EEGLAB users, while using Pythonic APIs, tooling, tests, packaging, and maintainable software design where they do not compromise parity.
- This is intended as a production research library for tens of thousands of EEG researchers; hold a high bar for correctness, tests, clear APIs, maintainability, and reviewable engineering decisions.
- Existing EEGPREP patterns win over new architecture ideas. Inspect current code/tests before changing conventions.
- Use EEGLAB data-structure docs as a reference when touching EEG/ALLEEG/STUDY-style fields: https://eeglab.org/tutorials/ConceptsGuide/Data_Structures.html
- When adding or changing public behavior, update Sphinx docs in the same pass unless there is a clear reason not to.
- Canonical EEG state is still dict-shaped for EEGLAB parity. `EEGobj` wraps that dict at `.EEG`; fields proxy via `obj.nbchan`/`obj.srate`, assignment writes into `.EEG`, and method calls deep-copy `.EEG`, call an eegprep function, then update `.EEG`. Do not treat `EEGobj` as a full `dict`/`MutableMapping` unless that is explicitly implemented.
- Preserve EEGLAB data axis order: channels x timepoints x epochs. Do not switch core code to epochs-first. Existing continuous 2-D data must remain accepted unless a tested migration changes it.
- Preserve separate `event` and `urevent` semantics. Event `latency` is an EEGLAB-style 1-based floating sample position by convention, not a Python array index.
- Use Python 0-based indices for internal object references where current code does so (`icachansind`, `chanlocs.urchan`, loaded `event.urevent`), and convert at EEGLAB import/export boundaries. Verify per function: older code/tests still mix conventions. Event latencies stay 1-based floats.
- Parity standard is function-specific: exact where realistic, tolerance-based for numerical algorithms, semantic/statistical for ICA, visualization, and workflows.
- Any new behavior needs parity tests. Use `HARNESS.md` and `eegprep.parity` manifest/deviation/oracle/reporting utilities rather than ad hoc comparisons.
- Do not newly drop EEGLAB/plugin metadata. Current save code preserves fixed EEGLAB fields plus `etc`/`dipfit`/`roi`, not arbitrary top-level fields; add parity tests before claiming lossless round-trip.
- Known gap: `pop_saveset` appears to handle event `urvent` instead of `urevent` during 0-to-1-based export conversion. Fix with round-trip parity tests before relying on event-urevent save parity.
- Before coding, state assumptions, surface ambiguity/tradeoffs, choose the simplest viable approach, and ask instead of hiding uncertainty.
- Keep changes surgical: no speculative features, no single-use abstractions, no adjacent refactors, no formatting churn, and every changed line should trace to the task.
- Define verifiable success criteria before implementation: new behavior gets tests first or alongside code; bug fixes need reproducing tests; refactors need before/after test confidence.
- Remove only unused code introduced by your own change. Mention unrelated dead code or risks; do not clean them up unless asked.
