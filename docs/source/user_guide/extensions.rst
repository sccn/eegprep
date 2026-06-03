.. _extensions:

==================
EEGPrep Extensions
==================

EEGPrep extensions are normal Python packages. EEGPrep discovers installed
extensions through the ``eegprep.extensions`` entry-point group and shows them
in ``File > Manage EEGPrep extensions`` or from Python:

.. code-block:: python

   import eegprep

   plugins = eegprep.plugin_menu(show=False)
   print(eegprep.format_plugin_menu())

The Extension Manager separates installed runtime state from curated catalog
metadata. Installed state comes from the local Python environment. Catalog
entries are metadata-only records that point to a package name, repository,
documentation URL, maintainer, and capabilities. EEGPrep does not host extension
zip files and the manager does not download, unzip, install, update, or remove
extension code.

Installing Extensions
=====================

Use the package manager for the Python environment that runs EEGPrep. The
manager shows copyable commands when catalog metadata is available; it never
runs them for you.

Install from PyPI:

.. code-block:: bash

   uv add eegprep-ext-example
   # or
   pip install eegprep-ext-example

Install from GitHub:

.. code-block:: bash

   uv add git+https://github.com/example/eegprep-ext-example.git
   # or
   pip install git+https://github.com/example/eegprep-ext-example.git

Install from a local editable checkout:

.. code-block:: bash

   uv add --editable /path/to/eegprep-ext-example
   # or
   pip install -e /path/to/eegprep-ext-example

After installing or updating an extension, restart EEGPrep so entry-point
discovery can inspect the new package.

Trust and Safety
================

Installing Python packages executes third-party code. Review the package,
maintainer, source repository, documentation, and your lab's software policy
before running any install command. Treat private package indexes and editable
local paths with the same care as public packages.

The Extension Manager may show installed packages that are not in the curated
catalog. Those are not automatically unsafe, but they are outside EEGPrep's
catalog metadata. Manage them through the package source you used to install
them.

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
