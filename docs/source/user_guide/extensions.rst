.. _extensions:

External Extensions
===================

EEGPrep extensions are normal Python packages. Install them with standard
Python package tooling, then start ``eegprep-gui`` or ``eegprep-console`` in
the same environment. EEGPrep discovers installed extensions through the
``eegprep.extensions`` entry-point group and validates their declarative
``ExtensionSpec`` records before they contribute menus, actions, help, or
``pop_*`` functions.

The Extension Manager is available from ``File > Manage EEGPrep extensions`` or
from Python:

.. code-block:: python

   import eegprep

   plugins = eegprep.plugin_menu(show=False)
   print(eegprep.format_plugin_menu())

The Extension Manager separates installed runtime state from curated catalog
metadata. Installed state comes from the local Python environment. Catalog
entries are metadata-only records that point to a package name, repository,
documentation URL, maintainer, and capabilities. EEGPrep does not host extension
zips, does not host arbitrary extension zip files, and the manager does not
download, unzip, install, update, or remove extension code.

Installing Extensions
=====================

Install EEGPrep first, then add extension packages to the same project or
virtual environment. The manager shows copyable commands when catalog metadata
is available; it never runs them for you.

Local or private editable extension
-----------------------------------

Use an editable install while developing an extension or keeping a lab-only
extension on disk:

.. code-block:: bash

   uv add --editable /path/to/eegprep-ext-foo

This is the best path for private lab workflows, prototype methods, and
extensions that should never leave a controlled environment. If the extension
uses optional dependencies, install the extra declared by that package:

.. code-block:: bash

   uv add --editable "/path/to/eegprep-ext-foo[models]"

GitHub extension
----------------

A researcher does not need to publish to PyPI to share an extension. A GitHub
repository can be installed directly:

.. code-block:: bash

   uv add git+https://github.com/lab/eegprep-ext-foo

Pin a branch, tag, or commit when reproducibility matters:

.. code-block:: bash

   uv add git+https://github.com/lab/eegprep-ext-foo --tag v0.3.0

PyPI extension
--------------

Use the package name when the extension is published on PyPI:

.. code-block:: bash

   uv add eegprep-ext-foo

PyPI is useful when an extension is public, versioned, and meant for broad
reuse. It is not required for GitHub-only or private lab distribution.

Private package index
---------------------

For a private package index, point uv at the lab or institution index:

.. code-block:: bash

   uv add --index https://packages.lab.example/simple eegprep-ext-foo

When the private index should replace PyPI rather than supplement it, use:

.. code-block:: bash

   uv add --default-index https://packages.lab.example/simple eegprep-ext-foo

Configure authentication through uv-supported environment variables, keyring
configuration, or your institution's package-index instructions. Do not put
passwords in committed ``pyproject.toml`` files.

Using Installed Extensions
==========================

After installation, launch EEGPrep from the same environment:

.. code-block:: bash

   uv run eegprep-gui --full
   uv run eegprep-console --full

Extension menus and actions appear automatically when the extension record is
active. In ``eegprep-console``, extension ``pop_*`` functions should behave like
EEGPrep ``pop_*`` functions: GUI actions and console calls share one
``EEGPrepSession``, update ``EEG`` and ``ALLEEG`` when data changes, and append
the returned command to ``LASTCOM`` and ``ALLCOM``.

For example, a user-facing extension function should support:

.. code-block:: python

   EEG, LASTCOM = pop_my_extension(EEG, return_com=True)
   pop_my_extension(EEG)

Normal Python imports still use standard Python semantics. The automatic
session storage behavior is specific to ``eegprep-console`` and registered GUI
actions.

Required Package Format
=======================

An EEGPrep extension package must provide:

* A ``pyproject.toml`` project.
* A ``[project.entry-points."eegprep.extensions"]`` entry point.
* A lightweight ``register()`` function that returns ``ExtensionSpec``.
* Declarative ``ExtensionAction``, ``ExtensionMenu``, and
  ``ExtensionPopFunction`` records where applicable.
* ``LazyImport`` targets for heavy modules, GUI modules, processing code, and
  model code.
* Packaged Markdown help and package data declared through
  ``ExtensionResource`` when those files are user-facing or required at
  validation time.
* ``pop_*`` functions that support ``return_com=True`` when the call is
  user-facing or history-relevant.
* No runtime dependency on ``src/eegprep/eeglab`` or any local EEGLAB checkout.

Minimal ``pyproject.toml`` metadata:

.. code-block:: toml

   [project]
   name = "eegprep-ext-foo"
   version = "0.1.0"
   dependencies = ["eegprep"]

   [project.entry-points."eegprep.extensions"]
   foo = "eegprep_ext_foo.register:register"

Minimal registration:

.. code-block:: python

   from eegprep import (
       ExtensionAction,
       ExtensionMenu,
       ExtensionPopFunction,
       ExtensionResource,
       ExtensionSpec,
       LazyImport,
   )

   def register():
       return ExtensionSpec(
           name="foo",
           display_name="Foo",
           version="0.1.0",
           package_name="eegprep-ext-foo",
           eegprep_requires=">=0.2",
           actions=(
               ExtensionAction(
                   name="foo.run",
                   target=LazyImport("eegprep_ext_foo.actions", "run"),
               ),
           ),
           menus=(
               ExtensionMenu(path=("Tools", "Foo"), action="foo.run", label="Run Foo"),
           ),
           pop_functions=(
               ExtensionPopFunction(
                   name="pop_foo",
                   target=LazyImport("eegprep_ext_foo.pop_foo", "pop_foo"),
               ),
           ),
           help_resources=(
               ExtensionResource("eegprep_ext_foo", "help/pop_foo.md"),
           ),
       )

``register()`` should import only the SDK and small metadata. Import processing
functions, Qt dialogs, and model loaders lazily through ``LazyImport`` so a
broken optional module does not slow or break EEGPrep startup.

Menus must be declarative. Do not mutate EEGPrep Qt objects, monkeypatch menu
builders, or write directly to ``EEGPrepSession`` internals from an extension.
If an extension needs a new integration surface, add or request SDK support
instead of reaching into private EEGPrep modules.

``pop_*`` History and Console Rules
===================================

Extension ``pop_*`` functions should follow EEGPrep and EEGLAB-facing
conventions:

* Accept an EEG dict or EEG-like object explicitly.
* Return the updated EEG object for data-changing calls.
* Support ``return_com=True`` and return ``(EEG, com)`` when the call should be
  recorded in history.
* Use EEGLAB-facing 1-based values in command strings when users provide
  EEGLAB-style indices, and convert to Python 0-based indexing internally.
* Keep GUI and console workflows synchronized through the registered action and
  ``EEGPrepSession`` helpers instead of mutating GUI-only state.
* Add packaged help resources for GUI Help buttons and ``pophelp`` surfaces.

Package Data, Optional Dependencies, and Large Files
====================================================

Use package resources for help text, small calibration files, montages, and
other data that must ship with the extension. Declare required runtime files
with ``ExtensionResource`` so ``validate_extension_spec`` can catch missing
resources before users click the action.

For optional dependencies, declare ``ExtensionDependency(..., optional=True)``
when the dependency unlocks optional behavior. Declare required dependencies
without ``optional=True`` so missing packages appear as
``missing_dependency`` rather than a runtime crash.

Large model files need an explicit distribution choice. Prefer package extras,
an institution-controlled package index, or an explicit first-run download
implemented by the extension. Do not imply that a curated catalog entry hosts
or downloads large model artifacts for EEGPrep.

Trust Levels and States
=======================

Installing Python packages executes third-party code. Review the package,
maintainer, source repository, documentation, and your lab's software policy
before running any install command. Treat private package indexes and editable
local paths with the same care as public packages.

The Extension Manager may show installed packages that are not in the curated
catalog. Those are not automatically unsafe, but they are outside EEGPrep's
catalog metadata. Manage them through the package source you used to install
them.

The Extension Manager and registry use these user-facing meanings:

Bundled/Core
  Ships with EEGPrep and is covered by the EEGPrep CI and release process.
  Registry value: ``bundled``.

Curated
  Appears in an EEGPrep catalog that has been reviewed for metadata quality,
  package-format requirements, and mechanical validation. Curated does not mean
  SCCN, EEGLAB, or EEGPrep scientifically endorses the method, results, claims,
  or clinical suitability. Registry value: ``curated``.

Installed
  Present in the user's Python environment and discovered through
  ``eegprep.extensions``. Installed does not imply catalog review. Registry
  value: ``installed``.

Disabled
  Present but intentionally turned off by the user or configuration. Disabled
  extensions remain visible so users can re-enable them or inspect why they are
  inactive. Registry value: ``disabled``.

Failed or incompatible
  Present but not active because import, registration, validation, dependency,
  or version checks failed. Registry values include ``failed_import``,
  ``invalid_spec``, ``missing_dependency``, ``incompatible``, and ``unknown``.
  The user action is to update, reinstall with missing dependencies, disable,
  or contact the extension maintainer.

The important combinations are:

* Curated but not installed: visible as reviewed catalog metadata only; it does
  not run until the user installs the package.
* Installed but not curated: usable if valid, but not reviewed by the catalog.
* Curated and installed: both reviewed in the catalog and present locally.
* Installed then broken after an EEGPrep update: visible as incompatible or
  failed, with errors preserved so the user can update or disable it.
* Experimental: can be installed or curated as experimental, but the docs and
  UI must not imply scientific endorsement.

Catalog Schema
==============

EEGPrep ships a packaged ``resources/extension_catalog.json`` file and can load
a local JSON catalog for tests or future download integration. Set
``EEGPREP_EXTENSION_CATALOG`` or pass ``catalog_path=`` to ``plugin_menu``.
Catalog loading is local-file-only; it does not fetch URLs.

Minimal catalog:

.. code-block:: json

   {
     "schema_version": 1,
     "extensions": [
       {
         "name": "example_extension",
         "display_name": "Example Extension",
         "version": "1.0.0",
         "package_name": "eegprep-ext-example",
         "description": "Example EEGPrep extension metadata.",
         "maintainer": "Example Lab",
         "docs_url": "https://example.org/eegprep-ext-example",
         "source": {
           "type": "pypi",
           "url": "https://pypi.org/project/eegprep-ext-example/"
         },
         "repository_url": "https://github.com/example/eegprep-ext-example",
         "capabilities": ["preprocessing", "reporting"],
         "eegprep_requires": ">=0.2.23"
       }
     ]
   }

Supported ``source.type`` values are ``pypi``, ``git``, ``local``, and
``private``. Source URLs must point to package metadata, documentation, local
paths, or repositories, not zip archives or wheel files. The catalog is a guide
for users and tests; installed extension compatibility is still determined by
the extension registry.

Catalog Submission Checks
=========================

Catalog submission and governance automation are maintained separately from
runtime extension loading. Before submitting an extension for curation, check
that:

* The package installs from a normal Python distribution path.
* The entry point returns a valid ``ExtensionSpec``.
* ``validate_extension_spec`` passes in a clean environment.
* Menus and ``pop_*`` functions are declarative and use ``LazyImport`` targets.
* Help and package data are included in the built wheel or source distribution.
* GUI extensions pass the relevant visual parity and user-flow checks.
* The README or package metadata states whether the method is experimental.
* Large files, optional dependencies, and private indexes are documented.
* No runtime code imports from the vendored EEGLAB reference checkout.

See :ref:`api_extensions` for the SDK API reference.
