.. _api_extensions:

==========================
Extension SDK and Registry
==========================

EEGPrep discovers external Python extensions through the
``eegprep.extensions`` entry-point group. An entry point should load a lightweight
``register()`` function, and that function must return an
``eegprep.ExtensionSpec``. The register function may import the SDK, read small
metadata constants, and declare lazy targets, but it should not import heavy
processing, GUI, or machine-learning modules.

Extension specs are declarative. Actions and ``pop_*`` functions use
``LazyImport("package.module", "callable_name")`` so GUI startup and registry
inspection do not import the callable modules. Packaged help and data resources
are checked during validation because missing user-facing resources should be
visible before an extension is activated.

Minimal extension entry point
=============================

.. code-block:: python

   from eegprep import ExtensionAction, ExtensionSpec, LazyImport

   def register():
       return ExtensionSpec(
           name="my_extension",
           display_name="My Extension",
           version="1.0.0",
           api_version="1",
           package_name="my_extension",
           actions=(
               ExtensionAction(
                   name="my_extension.run",
                   target=LazyImport("my_extension.actions", "run"),
               ),
           ),
       )

The package advertises the function through standard Python package metadata:

.. code-block:: toml

   [project.entry-points."eegprep.extensions"]
   my_extension = "my_extension.register:register"

Status model
============

``ExtensionRegistry.discover()`` returns deterministic ``ExtensionRecord``
objects. Records remain visible when an extension fails so one broken package
does not crash EEGPrep startup. Status values are ``bundled``, ``installed``,
``curated``, ``disabled``, ``incompatible``, ``failed_import``,
``invalid_spec``, ``missing_dependency``, and ``unknown``.

Bundled EEGPrep plugin ports are exposed as bundled extension records for
inventory purposes. Their existing menus and direct public APIs are still
provided by the current bundled plugin modules; external menu/action/console
wiring belongs to later extension-ecosystem phases.

Catalog and Governance
======================

Catalog metadata validation lives in ``eegprep.extension_catalog`` and is also
available as the ``eegprep-validate-extension-catalog`` console script. Static
validation checks JSON schema version, required metadata, naming, URLs, license,
maintainer contact, docs, conflicts, curation status, and compatibility fields
without requiring the extension package to be installed.

Stricter validation can also check installed package versions, required
dependencies, the ``eegprep.extensions`` entry point, import failures, and
whether the imported ``ExtensionSpec`` matches the catalog metadata.

See :ref:`extension_curation` for the official curation policy, trust message,
compatibility rules, catalog submission format, and naming recommendations.

Author Test Harness
===================

``ExtensionTestHarness`` provides reusable assertions for extension authors. It
checks that a spec validates, menus reference declared actions or ``pop_*``
functions, help resources exist, lazy targets load, and callable actions or
``pop_*`` functions return EEGLAB-style ``(EEG, com)`` results when tested by
the author.

API Reference
=============

.. autosummary::
   :toctree: generated/

   eegprep.CATALOG_SCHEMA_VERSION
   eegprep.CatalogValidationIssue
   eegprep.CatalogValidationOptions
   eegprep.CatalogValidationReport
   eegprep.EXTENSION_COMPATIBILITY_POLICY
   eegprep.EXTENSION_CURATION_POLICY_URL
   eegprep.EXTENSION_NAMING_PREFIX
   eegprep.EXTENSION_TRUST_MESSAGE
   eegprep.ExtensionSpec
   eegprep.ExtensionRegistry
   eegprep.ExtensionRecord
   eegprep.ExtensionStatus
   eegprep.ExtensionSourceType
   eegprep.ExtensionAction
   eegprep.ExtensionPopFunction
   eegprep.ExtensionMenu
   eegprep.ExtensionDependency
   eegprep.ExtensionResource
   eegprep.ExtensionLoadError
   eegprep.ExtensionValidationResult
   eegprep.ExtensionTestHarness
   eegprep.LazyImport
   eegprep.assert_extension_entry_point_loads
   eegprep.check_extension_compatibility
   eegprep.discover_extensions
   eegprep.extension_version_satisfies
   eegprep.load_catalog_entries
   eegprep.validate_catalog_entries
   eegprep.validate_catalog_file
   eegprep.validate_extension_spec

.. automodule:: eegprep.extensions
   :members:
   :undoc-members:

.. automodule:: eegprep.extension_catalog
   :members:
   :undoc-members:

.. automodule:: eegprep.extension_testing
   :members:
   :undoc-members:
