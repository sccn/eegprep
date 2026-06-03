# EEGPrep Extension Template

Copy this package when starting a small EEGPrep extension.

Install locally while developing:

```bash
uv add -e /path/to/eegprep_ext_template
```

Install from GitHub:

```bash
uv add git+https://github.com/lab/eegprep-ext-template
```

Install from PyPI:

```bash
uv add eegprep-ext-template
```

Run the template tests:

```bash
uv run pytest tests
```

The package contributes `pop_template_gain`, a single `pop_*` function that
scales EEG data, records an EEGLAB-style history command, declares Tools menu
metadata for the extension runtime, and exposes packaged help plus sample data
resources. Current EEGPrep SDK tests can validate the menu metadata; actual GUI
insertion depends on the Phase 2 runtime integration.
