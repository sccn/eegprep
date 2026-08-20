.. _releasing:

=========
Releasing
=========

For maintainers. Releases are published by ``.github/workflows/release.yml``,
triggered by pushing a ``v*`` tag. That is the only path to PyPI; there is no
separate local publishing route. You drive it from your machine with ``git`` and
``gh``, as below.

The version lives in one place
==============================

``src/eegprep/__init__.py``::

    __version__ = "0.3.0"

``pyproject.toml`` declares ``dynamic = ["version"]`` and reads that attribute, so
there is no version string to edit in ``pyproject.toml``.

Release from your machine
=========================

**1. Dry run first.** This runs the whole pipeline except publishing, so you find
problems before a tag exists:

.. code-block:: bash

   gh workflow run release.yml --ref master -f dry_run=true
   gh run watch "$(gh run list --workflow=release.yml --limit 1 --json databaseId --jq '.[0].databaseId')"

**2. Bump the version** in ``src/eegprep/__init__.py``, then commit and land it on
``master``:

.. code-block:: bash

   git commit -am "release: 0.3.0"
   git checkout master && git merge --no-ff develop && git push origin master

.. tip::

   If ``master`` is strictly behind ``develop``, ``git push origin develop:master``
   fast-forwards it without checking out ``master``. Useful when you have local
   changes — for example a modified ``src/eegprep/eeglab`` submodule — that would
   block the checkout.

**3. Tag and push.** Pushing the tag starts the release:

.. code-block:: bash

   git tag -a v0.3.0 -m "Release version 0.3.0"
   git push origin v0.3.0

**4. Watch it.** Nothing reaches PyPI unless every check passes:

.. code-block:: bash

   gh run watch "$(gh run list --workflow=release.yml --limit 1 --json databaseId --jq '.[0].databaseId')" --exit-status

If it fails, inspect and fix:

.. code-block:: bash

   gh run view --log-failed

**5. Add a section to** :ref:`changelog`.

What the workflow does
======================

1. **verify** — ``ruff check``, ``ruff format --check``, ``ty check``, and the full
   test suite with ``EEGPREP_SKIP_MATLAB=1``.
2. **build** — ``uv build``, then checks that the tag matches ``__version__``, that
   the vendored EEGLAB checkout is absent from both artifacts, that packaged data
   (ICLabel weights, help resources) is present, that ``twine check --strict``
   passes, and that the built wheel installs and imports.
3. **publish** — uploads to PyPI using Trusted Publishing, so no API token is
   stored in the repository.
4. **github-release** — creates the GitHub release with the artifacts attached.

.. note::

   Trusted Publishing needs one-time setup on PyPI under
   *Manage project → Publishing → Add a new publisher*: owner ``sccn``, repository
   ``eegprep``, workflow ``release.yml``, environment ``pypi``.

If a tag was pushed at a bad commit
===================================

Fix the problem, then move the tag. This is safe only while the tag published
nothing; once a version is on PyPI it can never be replaced.

.. code-block:: bash

   git push origin :refs/tags/v0.3.0     # delete the remote tag
   git tag -f v0.3.0 && git push origin v0.3.0

Docker image
============

CI does not build Docker images. After the tag is published, build and push the
image, which also updates the pin in ``tools/hpc/main.pbs``:

.. code-block:: bash

   docker login
   uv run python scripts/build_docker.py

Commit the updated pin. Use ``--no-push`` to build without publishing.

.. warning::

   PyPI uploads are immutable. A version number can never be reused, even after the
   release is deleted. Always dry-run first.

.. note::

   Build with ``uv build``, never ``python -m build``. The repository contains a
   ``build/`` output directory that shadows the ``build`` package on ``sys.path``, so
   ``python -m build`` fails with "'build' is a package and cannot be directly
   executed" no matter what is installed.
