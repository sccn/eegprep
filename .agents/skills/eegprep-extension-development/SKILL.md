---
name: eegprep-extension-development
description: Build EEGPrep external extension packages with SDK registration, declarative menus, pop_* console/history behavior, packaged help/data, distribution choices, tests, GUI visual parity, and catalog-readiness. Use when Codex or a researcher needs to create, port, review, package, or submit an EEGPrep extension outside the core package.
---

# EEGPrep Extension Development

Use this skill to build extension packages that work as standalone Python
packages and feel natural to EEGLAB users inside EEGPrep.

## Start Here

1. Read `AGENTS.md` and follow `.agents/skills/eegprep-feature-development/SKILL.md`
   for repo workflow, notes, tests, and PR expectations.
2. Inspect `src/eegprep/extensions.py` and `docs/source/api/extensions.rst` before
   writing registration code. Use real SDK symbols, especially
   `ExtensionSpec`, `ExtensionAction`, `ExtensionMenu`, `ExtensionPopFunction`,
   `ExtensionResource`, `ExtensionDependency`, `LazyImport`,
   `validate_extension_spec`, and the `eegprep.extensions` entry-point group.
3. If porting an EEGLAB plugin or workflow, inspect the matching EEGLAB MATLAB
   source during development. Do not make runtime extension code depend on a
   local EEGLAB checkout.
4. Check `examples/extensions/` for templates, scaffolding, and focused example
   packages. Reference or use them where available; do not duplicate their full
   implementation inside custom extension code.

## Plan the Extension

Define:

- Distribution path: local/private editable, GitHub-only, PyPI, or private
  package index.
- Trust posture: experimental, private lab use, public uncurated, or catalog
  submission. Curated means catalog-reviewed and mechanically validated, not
  scientific endorsement.
- Contributions: actions, menus, `pop_*` functions, help resources, package
  data, console aliases, optional dependencies, and large model files.
- EEGPrep version contract through `eegprep_requires` and extension API version.
- Tests that prove discovery, validation, lazy imports, resources, history,
  console behavior, and GUI behavior where applicable.

Prefer the smallest SDK surface that expresses the feature. If the extension
needs to mutate EEGPrep internals directly, stop and add or request an SDK
surface instead.

## Register Through the SDK

Create a normal Python package with `pyproject.toml`:

```toml
[project]
name = "eegprep-ext-foo"
version = "0.1.0"
dependencies = ["eegprep"]

[project.entry-points."eegprep.extensions"]
foo = "eegprep_ext_foo.register:register"
```

Keep `register()` lightweight:

```python
from eegprep import ExtensionAction, ExtensionMenu, ExtensionPopFunction, ExtensionSpec, LazyImport


def register():
    return ExtensionSpec(
        name="foo",
        display_name="Foo",
        version="0.1.0",
        package_name="eegprep-ext-foo",
        eegprep_requires=">=0.2",
        actions=(
            ExtensionAction("foo.run", LazyImport("eegprep_ext_foo.actions", "run")),
        ),
        menus=(ExtensionMenu(("Tools", "Foo"), "foo.run", "Run Foo"),),
        pop_functions=(
            ExtensionPopFunction("pop_foo", LazyImport("eegprep_ext_foo.pop_foo", "pop_foo")),
        ),
    )
```

Use `LazyImport` for processing modules, dialogs, models, and optional
dependencies. A registration function may import the SDK and small metadata; it
must not import heavy code, start Qt, read data files, or call EEGLAB.

## Menus, Help, Console, and History

- Declare menus with `ExtensionMenu`; do not mutate Qt menu objects or patch
  EEGPrep menu builders.
- Keep user-facing actions compatible with `EEGPrepSession`. GUI actions should
  store datasets, add history, and notify changes through session helpers.
- Implement user-facing `pop_*` functions with `return_com=True`; return
  `(EEG, com)` when history should be recorded.
- Preserve EEGLAB-facing command strings and indexing semantics while using
  Python indexing internally.
- Package GUI Help and `pophelp` text as Markdown resources and declare them
  with `ExtensionResource`.
- Put data/model resources in the package and access them with
  `importlib.resources` or declared `ExtensionResource` records. Do not read
  from `src/eegprep/eeglab` at runtime.

## Dependencies and Large Files

- Declare required Python packages normally in `pyproject.toml`.
- Use `ExtensionDependency(package, version_spec, optional=True)` only for
  optional behavior. Required missing dependencies should produce
  `missing_dependency`, not a late crash.
- For large models, choose an explicit path: package extra, private index,
  Git LFS-backed package source, or an extension-owned first-run download.
  Document the choice and never imply EEGPrep hosts model artifacts.

## GUI Extensions

For dialogs, windows, or menu UX:

1. Use `.agents/skills/eeglab-gui-visual-parity/SKILL.md` to compare labels,
   alignment, order, defaults, enabled states, buttons, and workflow against
   EEGLAB where an EEGLAB reference exists.
2. Exercise mixed GUI plus `eegprep-console` flows: GUI action, inspect `EEG`,
   `ALLEEG`, `CURRENTSET`, `LASTCOM`, and `ALLCOM`; then run the `pop_*`
   function from the console and verify GUI/history refresh.
3. Use `.agents/skills/gui-agent-flow-qa/SKILL.md` only when explicitly asked
   for full GUI agent QA or when the project workflow requires it.

## Tests and Checks

Write focused tests for:

- Entry-point discovery through `ExtensionRegistry` or `discover_extensions`.
- `validate_extension_spec` success and failure cases.
- Lazy imports remaining unloaded during discovery.
- Missing help/data resources and missing required dependencies.
- Duplicate action or `pop_*` names when relevant.
- `pop_*` return values, `return_com=True`, history strings, and console
  auto-store behavior.
- GUI visual parity and user-flow behavior for GUI extensions.
- Wheel or source distribution contents when packaged resources matter.

Run the narrowest relevant pytest files first, then `./pre-commit.py --fix`,
focused tests, and broader ruff/format/ty checks when the change affects
imports, docs, packaging, or public APIs.

## Distribution Decision

- Local/private lab only: `uv add --editable /path/to/eegprep-ext-foo`.
- GitHub-only: `uv add git+https://github.com/lab/eegprep-ext-foo`; pin tags
  or commits for reproducibility.
- PyPI: `uv add eegprep-ext-foo` after normal package publishing.
- Private index: `uv add --index https://packages.lab.example/simple
  eegprep-ext-foo` or `uv add --default-index ...` when the private index
  replaces PyPI.

EEGPrep does not host extension zips, install arbitrary archives, or treat
curation as scientific endorsement.

## Catalog Submission

Before proposing curation, verify:

- The package installs from a standard path and the entry point returns a valid
  `ExtensionSpec`.
- The extension passes validation in a clean environment.
- Menus, help, package data, `pop_*` history, and console behavior are covered.
- Optional dependencies and large files are documented and do not block startup.
- Experimental status is explicit in docs and metadata.
- Governance/catalog checks pass when the extension is intended for catalog
  submission.
