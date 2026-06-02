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

API Reference
=============

.. autosummary::
   :toctree: generated/

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
   eegprep.LazyImport
   eegprep.discover_extensions
   eegprep.validate_extension_spec

.. automodule:: eegprep.extensions
   :members:
   :undoc-members:
