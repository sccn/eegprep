.. _releasing:

=========
Releasing
=========

How to publish a new EEGPrep release to PyPI. For maintainers.

The version lives in one place
==============================

``src/eegprep/__init__.py``::

    __version__ = "0.3.0"

``pyproject.toml`` declares ``dynamic = ["version"]`` and reads that attribute, so
there is no version string to edit in ``pyproject.toml``.

Recommended: tag and let CI publish
===================================

``.github/workflows/release.yml`` publishes to PyPI when a ``v*`` tag is pushed,
using PyPI Trusted Publishing, so no API token is stored in the repository.

.. code-block:: bash

   # 1. bump __version__ in src/eegprep/__init__.py
   git commit -am "release: 0.3.0"

   # 2. land it on master
   git checkout master && git merge --no-ff develop && git push origin master

   # 3. tag — this triggers the release
   git tag -a v0.3.0 -m "Release version 0.3.0"
   git push origin v0.3.0

The workflow runs four jobs in order:

1. **verify** — ``ruff check``, ``ruff format --check``, ``ty check``, and the full
   test suite with ``EEGPREP_SKIP_MATLAB=1``.
2. **build** — ``uv build``, then checks that the tag matches ``__version__``, that
   the vendored EEGLAB checkout is absent from both artifacts, that packaged data
   (ICLabel weights, help resources) is present, that ``twine check --strict``
   passes, and that the built wheel installs and imports.
3. **publish** — uploads to PyPI via Trusted Publishing.
4. **github-release** — creates the GitHub release with the artifacts attached.

To exercise the build and all its checks without publishing, run the workflow
manually from the Actions tab; its ``dry_run`` input defaults to true.

.. note::

   Trusted Publishing needs one-time setup on PyPI under
   *Manage project → Publishing → Add a new publisher*: owner ``sccn``, repository
   ``eegprep``, workflow ``release.yml``, environment ``pypi``.

Alternative: the local script
=============================

``scripts/make_release.py`` does an interactive release from your machine, and is
the only path that also builds and pushes the Docker image.

.. code-block:: bash

   uv sync --group release
   uv run python scripts/make_release.py

It prompts for the release type (TestPyPI as ``eegprep_test``, production, or both)
and the new version, then rewrites ``__version__``, commits, builds, uploads, pushes,
tags ``v<version>``, and builds the Docker image. Credentials come from ``~/.pypirc``
or the ``PYPI_TOKEN`` / ``TESTPYPI_TOKEN`` environment variables.

Two things that will trip you up
================================

**Use** ``uv build``\ **, never** ``python -m build``\ **.** The repository contains a
``build/`` output directory that shadows the ``build`` package on ``sys.path``, so
``python -m build`` fails with "'build' is a package and cannot be directly executed"
no matter what is installed.

**Clear** ``dist/`` **before building by hand.** It can hold artifacts from an earlier
version, and ``twine upload dist/*`` would pick them up. Both the workflow and the
script clear it for you; only ad-hoc commands are exposed.

.. warning::

   PyPI uploads are immutable. A version number can never be reused, even after the
   release is deleted. Let the workflow's checks run before publishing.

After a release
===============

- Add a section to :ref:`changelog`.
- If the Docker image changed, update the ``eegprep:<version>`` pin in
  ``tools/hpc/main.pbs``. CI does not build Docker images; only the local script does.
- Update the default app option on brainlife if the release affects it.
