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

    __version__ = "0.5.0"

``pyproject.toml`` declares ``dynamic = ["version"]`` and reads that attribute, so
there is no version string to edit in the eegprep ``pyproject.toml``.

``eegprep-lean`` carries the same version in two places, and both must equal
``__version__``: ``packages/eegprep-lean/src/eegprep_lean/__init__.py`` and
``version`` in ``packages/eegprep-lean/pyproject.toml``. The joint release
below exits before it builds if those three strings differ.

Both packages
=============

``scripts/release_packages.py`` builds eegprep and eegprep-lean with that one
version. With no flags it builds both and does not upload::

    python scripts/release_packages.py

``--publish`` checks that both packages have built artifacts and that
``UV_PUBLISH_TOKEN`` is set, then uploads eegprep-lean and eegprep in that
order. It does not upload eegprep unless lean can be uploaded too. Lean has
no trusted publisher, so a GitHub Actions token is not enough for this
command. ``--check`` compares the versions and does not build. A ``v*`` tag
in ``GITHUB_REF``, or ``RELEASE_TAG``, must name the same version.

``.github/workflows/synced-release.yml`` runs the script without ``--publish``.
That job builds and verifies. It does not upload. ``release.yml`` still
publishes the eegprep project when a ``v*`` tag is pushed. Uploading
``eegprep-lean`` needs a trusted publisher for that project, which is not
configured; ``lean-wheel.yml`` builds the lean wheel and does not publish it.

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

   git commit -am "release: 0.5.0"
   git checkout master && git merge --no-ff develop && git push origin master

.. tip::

   If ``master`` is strictly behind ``develop``, ``git push origin develop:master``
   fast-forwards it without checking out ``master``. Useful when you have local
   changes — for example a modified ``src/eegprep/eeglab`` submodule — that would
   block the checkout.

**3. Tag and push.** Pushing the tag starts the release:

.. code-block:: bash

   git tag -a v0.5.0 -m "Release version 0.5.0"
   git push origin v0.5.0

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
   the vendored EEGLAB checkout is absent from both artifacts, that the wheel and
   sdist contain both packaged ICLabel networks (and no source ``netICL.mat``), and
   that the SHA-256 of the packaged 8-bit network equals ``packaged_model.sha256``
   and that of the full-precision network equals ``float32_model.sha256`` in
   ``tools/iclabel/parity_report.json``, and that the report's ``gate_pass`` is
   true. It also runs ``twine check --strict`` and installs both artifacts with
   the ICLabel extra for runtime classification and the ASR filter smoke test.
   The smoke tests also pin ``onnxruntime==1.32.0``, the version ``uv.lock``
   resolves on Linux and the one the test suite runs on, which is stricter than the
   ``iclabel`` extra's ``>=1.20`` floor. Bump the pin in ``release.yml`` together
   with ``uv.lock``; ``tests/test_ci_workflows.py`` fails when they differ.
3. **publish** — uploads to PyPI using Trusted Publishing, so no API token is
   stored in the repository.
4. **github-release** — creates the GitHub release with the artifacts attached.

.. note::

   Trusted Publishing needs one-time setup on PyPI under
   *Manage project → Publishing → Add a new publisher*: owner ``sccn``, repository
   ``eegprep``, workflow ``release.yml``, environment ``pypi``.

Regenerating the ICLabel network
================================

Two network files are built from the EEGLAB weights in ``netICL.mat``:
``src/eegprep/plugins/ICLabel/iclabel.onnx``, the full-precision (float32)
network that every native run loads, and ``iclabel_int8.onnx`` next to it, the
8-bit network the browser loads. Rebuild them only when the weights or the
quantization change; a normal release reuses the committed files. The sequence is:

1. Install the developer tools with the ``torch`` extra (very large); the ``dev``
   group already includes ONNX Runtime.
2. Export the full-precision (float32) network from ``netICL.mat``; it is written
   under ``build/iclabel/`` first.
3. Quantize its weights to 8-bit integers. The script compares the result with the
   frozen reference probabilities in ``tools/iclabel/`` and exits with an error,
   without writing a report, when it fails an acceptance criterion.
4. Copy the float32 export to ``src/eegprep/plugins/ICLabel/iclabel.onnx`` and the
   passing 8-bit network to ``src/eegprep/plugins/ICLabel/iclabel_int8.onnx``.
5. Commit both networks together with the updated
   ``tools/iclabel/parity_report.json``, so the report's ``packaged_model.sha256``
   matches the packaged 8-bit file and its ``float32_model.sha256`` matches the
   full-precision file.

The exact commands are in the docstring of ``tools/iclabel/quantize_iclabel_onnx.py``.
If ``netICL.mat`` itself changes, first refreeze the reference probabilities as
that docstring describes.

Continuous integration repeats steps 2 and 3 on every change. It requires the
regenerated float32 network to match the committed ``iclabel.onnx`` to 1e-5 on the
evaluation components and the regenerated 8-bit network to produce network outputs
identical to the committed ``iclabel_int8.onnx``; it does not compare bytes, because a
fresh PyTorch export need not be byte-identical across PyTorch versions. The build job
above fails the release when a packaged network differs from its source file, or
either packaged network and the report disagree, or the report records a failed
criterion. After a rebuild, update the measured agreement shown to users
(:ref:`iclabel_measured_agreement`) from the new report.

If a tag was pushed at a bad commit
===================================

Fix the problem, then move the tag. This is safe only while the tag published
nothing; once a version is on PyPI it can never be replaced.

.. code-block:: bash

   git push origin :refs/tags/v0.5.0     # delete the remote tag
   git tag -f v0.5.0 && git push origin v0.5.0

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
