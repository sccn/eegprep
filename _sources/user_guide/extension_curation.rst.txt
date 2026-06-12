.. _extension_curation:

=====================================
Extension Curation and Catalog Policy
=====================================

EEGPrep extensions are normal Python packages discovered through the
``eegprep.extensions`` entry-point group. The curated catalog is metadata only:
it does not host code archives, unzip plugin bundles, or make package names
load-bearing. Installing an extension still means installing third-party Python
code from PyPI, GitHub, a private package index, or a local checkout.

Trust and execution model
=========================

Extensions execute Python code. Installing a package can run package build
hooks, and importing or activating an extension executes third-party Python
code. EEGPrep isolates failed imports and marks broken extensions as
``failed_import``, ``invalid_spec``, ``incompatible``, or ``missing_dependency``,
but it cannot make arbitrary extension code safe.

Users should install extensions only from authors and package indexes they
trust. The Extension Manager and catalog docs should show the same warning:
curation is a review signal, not a sandbox.

What curation reviews
=====================

Maintainers review catalog submissions for:

- a valid ``eegprep.extensions`` entry point and an importable
  ``ExtensionSpec``;
- declared extension API version, EEGPrep version requirement, Python version
  requirement, and required dependencies;
- packaged help resources and documentation links;
- maintainer contact, source repository, license, and release version metadata;
- unique catalog id, extension name, display name, action names, and
  ``pop_*`` function names;
- clear install guidance that does not ask EEGPrep to download or execute
  arbitrary zip files.

Automated checks enforce schema, conflicts, version requirements, dependency
metadata, entry-point presence, import failures, and catalog/spec version
mismatches. Human review still covers scientific intent, documentation clarity,
maintainer identity, and whether the extension belongs in the public catalog.

What curation does not guarantee
================================

``curated`` does not mean:

- EEGPrep authored or maintains the extension;
- all code paths have been audited for security;
- scientific claims or preprocessing choices are endorsed;
- future releases will remain compatible;
- private or internal extensions become public.

Private/internal extensions remain supported through normal Python installation
and entry-point discovery, but they are not submitted to the public curated
catalog. Local organizations can validate private metadata with
``allow_private=True`` or ``--allow-private``.

Catalog submission format
=========================

The future ``eegprep-extension-index`` repository should use JSON metadata with
catalog kind ``extension_curation`` and schema version 1. A catalog file may
contain a top-level ``extensions`` list, or each extension can live in its own
JSON file in a directory.

.. code-block:: json

   {
     "catalog_kind": "extension_curation",
     "schema_version": 1,
     "extensions": [
       {
         "id": "example_extension",
         "package_name": "eegprep-ext-example",
         "entry_point": "example",
         "extension_name": "example_extension",
         "display_name": "Example Extension",
         "version": "1.0.0",
         "api_version": "1",
         "eegprep_requires": ">=0.2",
         "python_requires": ">=3.10",
         "license": "BSD-3-Clause",
         "maintainer": {"name": "SCCN", "email": "maintainers@example.org"},
         "docs_url": "https://example.org/eegprep-ext-example",
         "source_url": "https://github.com/example/eegprep-ext-example",
         "description": "Example extension catalog metadata.",
         "dependencies": [
           {"package": "mne", "version_spec": ">=1.10", "optional": false}
         ],
         "curation": {"status": "submitted"}
       }
     ]
   }

When maintainers mark an entry as curated, the metadata should also include
``curation.reviewed_by`` and ``curation.reviewed_at``. Submitted entries can be
validated before that human-review metadata is filled in.

Catalog CI validation
=====================

Run static validation in the catalog repository without installing extension
packages:

.. code-block:: bash

   eegprep-validate-extension-catalog catalog.json

or from an EEGPrep checkout:

.. code-block:: bash

   uv run python tools/validate_extension_catalog.py catalog.json

Use stricter checks after installing a candidate package into the CI
environment:

.. code-block:: bash

   eegprep-validate-extension-catalog catalog.json --check-installed --check-import

The stricter mode verifies installed package version metadata, the
``eegprep.extensions`` entry point, import failures, imported ``ExtensionSpec``
metadata, unsupported EEGPrep versions, and required dependency mismatches.

Compatibility policy
====================

Extension API versions are major-version compatible. EEGPrep accepts extensions
whose ``api_version`` has the same major version as
``eegprep.EXTENSION_API_VERSION``. A future EEGPrep release that changes the
major extension API version may mark older extensions as ``incompatible``.

Extensions should declare:

- ``api_version`` for the SDK contract;
- ``eegprep_requires`` for supported EEGPrep releases;
- ``python_requires`` for supported Python versions;
- required and optional dependencies with version specifiers.

Deprecations should be documented for at least one minor EEGPrep release before
removal when practical. Removal that changes the extension API major version is
allowed to mark older extensions incompatible at discovery time.

Naming recommendations
======================

Public packages should prefer names like ``eegprep-ext-clean-example``. The
validator warns when a package does not use the ``eegprep-ext-`` prefix, but it
does not reject the package. Discovery uses the ``eegprep.extensions`` entry
point, not package names.

Extension ``pop_*`` names should also be unique. If an extension declares a
``pop_*`` name that already exists in core EEGPrep, the core command remains the
console/default target; choose a lab- or method-specific name such as
``pop_labname_clean`` to avoid surprising users.

Author test harness
===================

Extension authors can use the reusable harness to catch common integration
mistakes before submitting catalog metadata.

.. code-block:: python

   from eegprep import ExtensionTestHarness
   from my_extension.register import register

   def test_extension_contract():
       harness = ExtensionTestHarness(register())
       harness.assert_all_static_contracts()

   def test_pop_function_history(sample_eeg):
       harness = ExtensionTestHarness(register())
       harness.assert_pop_function_history_result("pop_my_extension", sample_eeg)

The harness checks spec validation, menu references, help resources, lazy action
loads, ``pop_*`` loads, and history-aware ``(EEG, com)`` console results for
actions or pop functions that the author calls in tests.
