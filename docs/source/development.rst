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
locked dependency set. Use ``uv run`` for commands so they execute inside this
environment.

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

Building Documentation
======================

Build HTML Documentation
------------------------

Navigate to the docs directory and build:

.. code-block:: bash

    uv run make -C docs html

The built documentation is in ``docs/build/html/``.

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
    git rebase upstream/main
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
