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

Extension Manager and catalog
=============================

``plugin_menu(show=False)`` returns the Extension Manager inventory for scripts
and ``eegprep-console``. The inventory is built from registry records plus a
metadata-only curated catalog loaded from packaged resources, a local
``catalog_path=``, or ``EEGPREP_EXTENSION_CATALOG``. Catalog entries are advisory
metadata: package names, repository/documentation links, maintainers,
capabilities, and safe install-command strings. They never contain code zips,
and EEGPrep never executes install or update commands.

Use ``load_extension_catalog()`` to inspect a local catalog before passing it to
``plugin_menu``:

.. code-block:: python

   catalog = eegprep.load_extension_catalog("lab-extension-catalog.json")
   plugins = eegprep.plugin_menu(catalog=catalog, show=False)

``build_safe_install_commands()`` and ``build_safe_update_commands()`` return
copyable strings such as ``uv add eegprep-ext-foo`` or
``pip install git+https://github.com/example/eegprep-ext-foo.git``. They do not
run subprocesses.

API Reference
=============

.. autosummary::
   :toctree: generated/

   eegprep.ExtensionSpec
   eegprep.ExtensionRegistry
   eegprep.ExtensionRecord
   eegprep.ExtensionStatus
   eegprep.ExtensionSourceType
   eegprep.ExtensionCatalog
   eegprep.ExtensionCatalogEntry
   eegprep.CatalogSourceType
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
   eegprep.load_extension_catalog
   eegprep.build_safe_install_commands
   eegprep.build_safe_update_commands

.. automodule:: eegprep.extensions
   :members:
   :undoc-members:

.. automodule:: eegprep.functions.adminfunc.extension_catalog
   :members:
   :undoc-members:
