.. _development:

==================
Development Setup
==================

This guide covers setting up a development environment for EEGPrep and contributing to the project.

Prerequisites
=============

System Requirements
-------------------

- **Python**: 3.11 or higher
- **Git**: For version control
- **uv**: Default package and environment manager

Check your Python version:

.. code-block:: bash

    python --version

Required Tools
--------------

- **Git**: `https://git-scm.com/ <https://git-scm.com/>`_
- **Python**: `https://www.python.org/ <https://www.python.org/>`_
- **uv**: `https://docs.astral.sh/uv/ <https://docs.astral.sh/uv/>`_

Optional Tools
--------------

- **Conda**: For environment management (`https://conda.io/ <https://conda.io/>`_)
- **Docker**: For containerized development
- **Make**: For running build commands

Installation from Source
========================

Clone the Repository
--------------------

.. code-block:: bash

    git clone https://github.com/sccn/eegprep.git
    cd eegprep

Create the uv Environment
-------------------------

Install the default development environment:

.. code-block:: bash

    uv python install 3.11
    uv sync --group dev

``uv sync`` creates ``.venv/`` and installs EEGPrep in editable mode from the
locked dependency set. The development environment includes the GUI and
``eegprep-console`` runtime dependencies so ``uv run eegprep-console --full``
works from a fresh checkout. Use ``uv run`` for commands so they execute inside
this environment.

Install Documentation Dependencies
----------------------------------

.. code-block:: bash

    uv sync --extra docs --group dev

This installs:

- The eegprep package in editable mode
- Development dependencies used by repo tooling
- Documentation dependencies

Running Tests
=============

Test Discovery
--------------

Tests are located in the ``tests/`` directory. Run all tests:

.. code-block:: bash

    uv run pytest tests

Run specific test file:

.. code-block:: bash

    uv run pytest tests/test_clean_artifacts.py

Run specific test function:

.. code-block:: bash

    uv run pytest tests/test_clean_artifacts.py::TestClassName::test_method_name

Run a marker subset:

.. code-block:: bash

    uv run pytest -m "not slow"

Markers include ``slow``, ``matlab``, ``octave``, ``gui``, ``visual``, and
``parity``. Legacy ``unittest`` tests are categorized during collection in
``tests/conftest.py`` so marker expressions work without rewriting the tests.

Continuous Integration
----------------------

Tests run automatically on:

- Every push to a branch
- Every pull request
- Scheduled nightly runs

Check CI status on GitHub Actions.

EEG And Session Contracts
=========================

EEGPrep follows EEGLAB's public data model while keeping Python internals
explicit and testable. Feature code should treat the following contracts as
shared foundations rather than per-function conventions.

EEG Dictionaries
----------------

Stored EEG dictionaries should be normalized through ``eeg_checkset`` or an
``EEGPrepSession`` storage helper before other code relies on shape or type
invariants. The core fields are ``data``, ``nbchan``, ``pnts``, ``trials``,
``srate``, ``xmin``, ``xmax``, ``times``, ``event``, ``urevent``, ``epoch``,
``chanlocs``, ``chaninfo``, ``history``, ``icaact``, ``icawinv``,
``icasphere``, ``icaweights``, and ``icachansind``.

Continuous data is channel-major with shape ``(nbchan, pnts)``. Epoched data
is channel-major with shape ``(nbchan, pnts, trials)``. Event latencies and
user-visible dataset, channel, epoch, and component indices are EEGLAB-style
1-based values. Python array indexing remains 0-based inside numerical code.

``event`` entries should keep EEGLAB-facing ``latency`` values and, when
available, ``urevent`` pointers back to ``urevent`` entries. ``urevent`` is the
original-event table; functions that create, delete, or reorder events must
state whether they preserve, extend, or rebuild it. ``epoch`` stores per-epoch
event metadata for epoched datasets. ``chanlocs`` entries use EEGLAB-style
channel dictionaries, and ``chaninfo`` stores global channel-location metadata.
ICA fields must be cleared or recomputed consistently when data, channel
order, or channel count changes.

Session Selection
-----------------

``EEGPrepSession.CURRENTSET`` is always a Python ``list[int]`` containing
EEGLAB-facing 1-based dataset indices. An empty selection is ``[]`` internally
and ``0`` in the console workspace. A single selected dataset is exposed as
``CURRENTSET == n`` in the console. Multiple selected datasets are exposed as
``CURRENTSET == [n, ...]``. Selection order is preserved and duplicate dataset
indices are invalid.

Read selection state through ``EEGPrepSession.selected_dataset_indices()`` when
future STUDY or group-level code needs the current dataset vector. Phase 1a
defines this read contract only; user-facing multi-selection mutation belongs
to later feature work.

History And Menu Inventory
--------------------------

User-facing ``pop_*`` functions should support ``return_com=True`` and return a
history command that can be converted to valid ``eegprep-console`` input. GUI
and console code should append each successful command once through
``EEGPrepSession.add_history`` or storage helpers.

GUI data-changing actions that produce a new EEG dataset, such as resampling,
filtering, cleaning, epoching, selecting data, rereferencing, interpolation,
or component removal, should commit through ``pop_newset`` so the user can
choose whether to overwrite the current dataset or keep the result as a new
dataset. Actions that only update metadata, marks, history, ICA fields, or
STUDY state may store directly when that matches EEGLAB's callback behavior.

``eegprep-console`` and the GUI share one session. GUI command echoes should
show replayable Python input before progress messages or warnings from the same
action. ``eegh`` presents history newest-first like EEGLAB while
``EEGPrepSession.ALLCOM`` remains chronological internally.

Do not fake EEGLAB's one-dataset-in-memory ``option_storedisk`` behavior.
Use ``eeg_store``/``eeg_retrieve`` or ``EEGPrepSession`` so saved non-current
datasets are represented by explicit offloaded disk handles and rehydrated
through the shared storage path. Unsaved resident datasets must stay resident
or fail clearly until the user saves them.

Menu placeholders are machine-readable. Each placeholder action has either a
target epic phase or an explicit exclusion reason for workflows that cannot be
packaged in EEGPrep. Runtime package code must not read, import, or shell out
to ``src/eegprep/eeglab``; that tree is only a development parity reference.

GUI Help buttons and Help-menu topics must resolve to packaged Markdown files
under ``src/eegprep/resources/help``. Do not fall back to the vendored EEGLAB
tree or Python docstrings at runtime. When adding a new implemented
GUI-reachable ``pop_*``/``eeg_*`` action, add its help resource and extend the
menu/help resource inventory tests.

EEGLAB Core Parity Matrix
=========================

The Phase 1 core parity epic uses a committed machine-readable matrix at
``docs/parity/eeglab_core_parity_matrix.json``. The matrix classifies the
EEGLAB public and semi-public functions in the scope categories recorded in
its metadata and source issue. It is a work contract for later phase agents,
not package runtime data.

Rows use these statuses:

- ``implemented``: EEGPrep already covers the behavior.
- ``partial``: EEGPrep has an implementation, but important EEGLAB behavior,
  options, or workflow paths remain.
- ``port``: the row should be ported or wrapped during the responsible phase.
- ``consolidated``: EEGPrep covers the behavior through another module or
  helper, and a duplicate same-name file is not needed.
- ``stale_skip``: the function is obsolete or stale enough to skip.
- ``matlab_runtime_skip``: the function is MATLAB-specific runtime, GUI shim,
  path, deployment, or compatibility behavior that should not exist in
  standalone EEGPrep package code.
- ``external_dependency_skip``: the function depends on external MATLAB
  toolboxes/plugins or web/path integration outside this epic.

Use ``stale_skip`` only when every stale-policy field in the matrix is false:
the function is not menu-reachable, not a documented user API, not called by an
in-scope workflow, not required by parity tests, not needed as a helper for the
remaining phases, and not a likely compatibility alias users type. When in
doubt, leave the row as ``port`` or ``partial`` and add notes for the
responsible phase.

When a later phase implements or intentionally skips a row:

1. Update ``status``, ``eegprep_equivalent``, ``rationale``,
   ``responsible_phase``, ``user_facing_surface``, and ``test_notes`` in the
   JSON row.
2. For a new ``stale_skip`` row, include the complete ``stale_policy`` object
   with all fields set to ``false`` and explain the evidence in ``rationale``.
3. For ``partial`` rows, keep the row ``partial`` until unsupported behavior is
   implemented or explicitly reclassified with a defensible limitation.
   Any row left as ``port`` or ``partial`` after a phase closes must cite a
   concrete follow-up issue in ``follow_up_issue`` or, when needed for prose
   context, in ``rationale`` or ``test_notes``.
4. Run the validator:

   .. code-block:: bash

      uv run --no-sync python -m tools.eeglab_parity_matrix

5. Run the focused matrix tests:

   .. code-block:: bash

      uv run --no-sync pytest tests/test_eeglab_parity_matrix.py

The validator may read ``src/eegprep/eeglab`` because it is development
tooling. Installed package code under ``src/eegprep`` must not read, import
from, or shell out to ``src/eegprep/eeglab``. If EEGLAB-like help text,
examples, options, or resources are needed at runtime, convert them into
EEGPrep-owned packaged resources instead of reaching into the vendored
reference tree.

EEGLAB Final Standalone Parity Matrix
=====================================

The final standalone parity epic uses a second machine-readable matrix at
``docs/parity/eeglab_final_parity_matrix.json``. It extends the core matrix into
the remaining product surfaces that are not simple core-function parity rows:
bundled plugin depth, MATLAB object/storage semantics, optional-toolbox
workflows, and documentation/tutorial coverage.

Rows group source files into user workflows, but every discovered final-epic
EEGLAB reference path must be covered exactly once. The validator discovers:

- ``plugins/clean_rawdata``, ``plugins/firfilt``, ``plugins/ICLabel``, and
  ``plugins/dipfit`` files, excluding vendored third-party MatConvNet and
  Manopt internals, examples, and tests;
- ``functions/@eegobj``, ``functions/@memmapdata``, and ``functions/@mmo``;
- EEGLAB tutorial scripts under ``tutorial_scripts``;
- selected optional-toolbox workflow rows that point back to the core matrix.

Final matrix statuses are ``implemented``, ``partial``, ``port``,
``consolidated``, ``stale_skip``, ``matlab_runtime_skip``,
``optional_dependency``, ``external_plugin``, and ``docs_gap``. Non-skip rows
must name a responsible Phase 2-8 issue. Skip rows must use
``responsible_phase: "none"``. ``optional_dependency`` rows must name the
backend decision, fallback behavior, user-facing message, and phase contract so
later agents do not silently fake external-toolbox behavior.

Validate the final matrix with:

.. code-block:: bash

   uv run --no-sync python -m tools.eeglab_final_parity_matrix --json

The docs architecture for the final epic is recorded directly in the matrix
metadata and its source issue. It should be useful to EEG researchers first:
describe EEGPrep's standalone Python package, Qt GUI, and ``eegprep-console``
behavior accurately, and use EEGLAB comparisons only where they help users
migrate or understand familiar concepts.

Building Documentation
======================

Build HTML Documentation
------------------------

Sync the docs extra, then build with the same command used by the Phase 7
acceptance criteria:

.. code-block:: bash

    uv sync --group dev --extra docs
    uv run --no-sync sphinx-build -b html docs/source docs/_build/html

The ``docs/Makefile`` target remains available for local iteration:

.. code-block:: bash

    uv run make -C docs html

The direct Sphinx command writes to ``docs/_build/html/``. The Makefile target
writes to ``docs/build/html/``.

View Documentation Locally
---------------------------

Open the built documentation in your browser:

.. code-block:: bash

    open docs/build/html/index.html  # macOS
    xdg-open docs/build/html/index.html  # Linux
    start docs/build/html/index.html  # Windows

Or use a local server:

.. code-block:: bash

    cd docs/build/html
    uv run python -m http.server 8000

Then visit ``http://localhost:8000`` in your browser.

Clean Build
-----------

Remove old build files and rebuild:

.. code-block:: bash

    uv run make -C docs clean
    uv run make -C docs html

Build Options
-------------

Build PDF documentation (requires LaTeX):

.. code-block:: bash

    uv run make -C docs latexpdf

Build EPUB documentation:

.. code-block:: bash

    uv run make -C docs epub

Debugging Tips
==============

Logging
-------

Enable debug logging in your code:

.. code-block:: python

    import logging

    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Use logging in your code
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

Breakpoints
-----------

Use Python's built-in debugger:

.. code-block:: python

    import pdb

    def my_function():
        x = 10
        pdb.set_trace()  # Execution pauses here
        y = x + 5
        return y

Or use the newer breakpoint() function (Python 3.7+):

.. code-block:: python

    def my_function():
        x = 10
        breakpoint()  # Execution pauses here
        y = x + 5
        return y

Profiling
---------

Profile code performance:

.. code-block:: python

    import cProfile
    import pstats

    # Profile a function
    profiler = cProfile.Profile()
    profiler.enable()

    # Your code here
    my_function()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Print top 10 functions

Memory Profiling
----------------

Install memory profiler:

.. code-block:: bash

    uv add --dev memory-profiler

Use it in your code:

.. code-block:: python

    from memory_profiler import profile

    @profile
    def my_function():
        large_list = [i for i in range(1000000)]
        return sum(large_list)

Run with:

.. code-block:: bash

    uv run python -m memory_profiler script.py

Release Process
===============

Version Numbering
-----------------

EEGPrep uses `Semantic Versioning <https://semver.org/>`_:

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

Example: ``1.2.3`` (Major.Minor.Patch)

Versioning Steps
----------------

1. Update version in ``src/eegprep/__init__.py``:

.. code-block:: python

    __version__ = "1.2.3"

2. Update version in ``pyproject.toml``:

.. code-block:: toml

    [project]
    version = "1.2.3"

3. Update ``docs/source/changelog.rst`` with release notes

4. Commit changes:

.. code-block:: bash

    git add .
    git commit -m "Release version 1.2.3"

Tagging
-------

Create a git tag for the release:

.. code-block:: bash

    git tag -a v1.2.3 -m "Release version 1.2.3"
    git push origin v1.2.3

PyPI Release
------------

Build distribution packages:

.. code-block:: bash

    uv run --group release python -m build

Upload to PyPI:

.. code-block:: bash

    uv run --group release python -m twine upload dist/*

Or upload to TestPyPI first:

.. code-block:: bash

    uv run --group release python -m twine upload --repository testpypi dist/*

Common Issues
=============

Import Errors
-------------

**Problem**: ``ModuleNotFoundError: No module named 'eegprep'``

**Solution**: Install the package in editable mode:

.. code-block:: bash

    uv sync --group dev

Test Failures
-------------

**Problem**: Tests fail with import errors

**Solution**: Ensure you're in the virtual environment and dependencies are installed:

.. code-block:: bash

    uv sync --group dev
    uv run pytest tests

Documentation Build Errors
---------------------------

**Problem**: Sphinx build fails with missing modules

**Solution**: Install documentation dependencies:

.. code-block:: bash

    uv sync --extra docs --group dev

Git Conflicts
-------------

**Problem**: Merge conflicts when pulling upstream changes

**Solution**: Resolve conflicts manually:

.. code-block:: bash

    git fetch upstream
    git rebase upstream/develop
    # Resolve conflicts in your editor
    git add .
    git rebase --continue

Virtual Environment Issues
---------------------------

**Problem**: Virtual environment not activating

**Solution**: Recreate the virtual environment:

.. code-block:: bash

    rm -rf .venv
    uv sync --group dev

Dependency Conflicts
--------------------

**Problem**: Dependency version conflicts

**Solution**: Refresh the locked environment:

.. code-block:: bash

    uv lock
    uv sync --group dev

Getting Help
============

- Check the :doc:`contributing` guide
- Review existing `GitHub Issues <https://github.com/sccn/eegprep/issues>`_
- Ask in GitHub Discussions
- Contact the maintainers

Happy developing!
