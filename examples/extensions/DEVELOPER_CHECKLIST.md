# EEGPrep Extension Developer Checklist

Use this checklist before sharing an extension with another EEG researcher.

## GUI Parity

- Match EEGLAB label text, control order, defaults, enabled states, and button behavior for ported dialogs.
- Put renderer-independent dialog specs in code and test them with a fake renderer.
- Return the original EEG and an empty command when the user cancels a dialog.
- Keep Help buttons disabled until your extension runtime can route them to packaged Markdown resources.
- When Help routing is available, point Help buttons at packaged Markdown resources, not MATLAB files or Python docstrings.

## Console And History

- User-facing `pop_*` functions accept `return_com=True`.
- Mutating `pop_*` functions return `(EEG, com)` where `com` is an EEGLAB-style command string.
- Console wrappers can replay the command after MATLAB-to-Python conversion.
- GUI actions that mutate EEG update `LASTCOM` and `ALLCOM` through `EEGPrepSession`.
- Treat `ExtensionMenu` entries as declarative metadata unless your target EEGPrep runtime has installed-extension menu insertion enabled.

## EEG Data Semantics

- Treat continuous data as channel-major `(nbchan, pnts)`.
- Treat epoched data as channel-major `(nbchan, pnts, trials)`.
- Preserve EEGLAB-facing 1-based event latencies and document any Python-only 0-based index inputs.
- Deep-copy EEG dictionaries before mutating unless the function is explicitly documented as in-place.
- Update dependent fields such as `nbchan`, `pnts`, `trials`, `xmin`, `xmax`, `times`, `event`, `history`, `icaact`, `icawinv`, `icasphere`, `icaweights`, and `icachansind` when behavior affects them.

## Package Data

- Declare help Markdown, sample data, model weights, montages, or lookup tables in package data.
- Verify resources with `ExtensionResource.exists()` and `read_text()` or `read_bytes()`.
- Keep runtime behavior independent of `src/eegprep/eeglab`.
- Use tiny sample files in tests; do not check in large model weights unless they are essential.

## Dependencies

- Put required runtime dependencies in `project.dependencies`.
- Declare optional extension capabilities with `ExtensionDependency(..., optional=True)`.
- Fail clearly when an optional action is used without its optional dependency.
- Avoid dependencies for small helpers that standard library, NumPy, SciPy, or EEGPrep already cover.

## Tests

- Test `ExtensionSpec` validation and entry-point loading.
- Test every `pop_*` history command with `return_com=True`.
- Test GUI cancel and invalid-input paths.
- Test package data/model-resource availability.
- Test sample-data workflows with realistic EEG dicts.
- Add console/session tests when an action mutates EEG or depends on `LASTCOM`/`ALLCOM`.
- Test `--no-plugins` behavior through `ExtensionRegistry.discover(include_plugins=False)` when discovery is relevant.

## Version Compatibility

- Set `api_version` to the current EEGPrep extension API version.
- Use `eegprep_requires` for the minimum EEGPrep version your extension needs.
- Keep private extensions on the same contract as public packages so they can be installed editable, from GitHub, or from PyPI.
- For EEGLAB plugin ports with MATLAB-only pieces, mark unsupported paths explicitly and keep Python runtime code standalone.
