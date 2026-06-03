# EEGPrep Extension Authoring Examples

This directory contains concrete extension packages for researchers who want to build private,
GitHub-hosted, or PyPI-published EEGPrep extensions.

Start with `eegprep_ext_template`. It demonstrates:

- `pyproject.toml` with `[project.entry-points."eegprep.extensions"]`
- an `ExtensionSpec` registration function
- one `pop_*` function that preserves EEG dict semantics and returns `(EEG, com)`
- one declarative menu contribution for the extension runtime to place
- one packaged help Markdown resource
- one packaged sample-data resource
- tests that exercise SDK registration, package data, GUI cancel, and console/history behavior

Install modes:

```bash
uv add -e /path/to/eegprep_ext_template
uv add git+https://github.com/lab/eegprep-ext-template
uv add eegprep-ext-template
```

Run template tests after copying or adapting the package:

```bash
uv run pytest /path/to/eegprep_ext_template/tests
```

The smaller packages show focused variants:

- `eegprep_ext_signal_transform`: pure EEG signal transform
- `eegprep_ext_file_io`: file importer/exporter functions
- `eegprep_ext_gui_dialog`: renderer-independent `inputgui` dialog and cancel path
- `eegprep_ext_plot_browser`: plot/browser-style action with a callback boundary
- `eegprep_ext_optional_dependency`: optional dependency plus packaged model/data file

Private extensions do not need catalog submission. Keep the package in a private repository or
install it from a local path, and use the same entry-point registration contract.

The Phase 1 SDK validates menu metadata; installed-extension GUI insertion is owned by the Phase 2
runtime integration. Until that runtime hook is present, treat `ExtensionMenu` entries as declarative
metadata that tests can validate but the core GUI may not display.

`eegprep --no-plugins` and `eegprep-console --no-plugins` should not load extension entry points.
Use that mode when debugging core EEGPrep behavior without external contributions.
