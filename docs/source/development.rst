.. _development:

==================
Development Setup
==================

This guide covers setting up a development environment for EEGPrep and contributing to the project.

Prerequisites
=============

System Requirements
-------------------

- **Python**: 3.8 or higher
- **Git**: For version control
- **pip**: Python package manager
- **Virtual environment**: venv or conda

Check your Python version:

.. code-block:: bash

    python --version

Required Tools
--------------

- **Git**: `https://git-scm.com/ <https://git-scm.com/>`_
- **Python**: `https://www.python.org/ <https://www.python.org/>`_
- **pip**: Usually included with Python

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

    git clone https://github.com/NeuroTechX/eegprep.git
    cd eegprep

Create Virtual Environment
---------------------------

Using venv:

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

Using conda:

.. code-block:: bash

    conda create -n eegprep python=3.10
    conda activate eegprep

Install in Editable Mode
------------------------

Install the package with all development dependencies:

.. code-block:: bash

    pip install -e ".[dev]"

This installs:

- The eegprep package in editable mode (changes are reflected immediately)
- Development dependencies (testing, linting, formatting)
- Documentation dependencies

Install Documentation Dependencies
----------------------------------

.. code-block:: bash

    pip install -r requirements-docs.txt

This includes:

- Sphinx (documentation generator)
- sphinx-rtd-theme (Read the Docs theme)
- sphinx-autodoc-typehints (Type hints in documentation)
- sphinx-gallery (Example gallery)

Running Tests
=============

Test Discovery
--------------

Tests are located in the ``tests/`` directory. Run all tests:

.. code-block:: bash

    pytest

Run specific test file:

.. code-block:: bash

    pytest tests/test_clean_artifacts.py

Run specific test function:

.. code-block:: bash

    pytest tests/test_clean_artifacts.py::test_remove_artifacts

Pytest Options
--------------

Verbose output:

.. code-block:: bash

    pytest -v

Stop on first failure:

.. code-block:: bash

    pytest -x

Show print statements:

.. code-block:: bash

    pytest -s

Run only tests matching a pattern:

.. code-block:: bash

    pytest -k "artifact"

Test Coverage
-------------

Generate coverage report:

.. code-block:: bash

    pytest --cov=src/eegprep --cov-report=html

View HTML coverage report:

.. code-block:: bash

    open htmlcov/index.html  # macOS
    xdg-open htmlcov/index.html  # Linux
    start htmlcov/index.html  # Windows

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

    cd docs
    make html

The built documentation is in ``docs/_build/html/``.

View Documentation Locally
---------------------------

Open the built documentation in your browser:

.. code-block:: bash

    open docs/_build/html/index.html  # macOS
    xdg-open docs/_build/html/index.html  # Linux
    start docs/_build/html/index.html  # Windows

Or use a local server:

.. code-block:: bash

    cd docs/_build/html
    python -m http.server 8000

Then visit ``http://localhost:8000`` in your browser.

Clean Build
-----------

Remove old build files and rebuild:

.. code-block:: bash

    cd docs
    make clean
    make html

Build Options
-------------

Build PDF documentation (requires LaTeX):

.. code-block:: bash

    cd docs
    make latexpdf

Build EPUB documentation:

.. code-block:: bash

    cd docs
    make epub

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

    pip install memory-profiler

Use it in your code:

.. code-block:: python

    from memory_profiler import profile
    
    @profile
    def my_function():
        large_list = [i for i in range(1000000)]
        return sum(large_list)

Run with:

.. code-block:: bash

    python -m memory_profiler script.py

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

    pip install build twine
    python -m build

Upload to PyPI:

.. code-block:: bash

    python -m twine upload dist/*

Or upload to TestPyPI first:

.. code-block:: bash

    python -m twine upload --repository testpypi dist/*

Common Issues
=============

Import Errors
-------------

**Problem**: ``ModuleNotFoundError: No module named 'eegprep'``

**Solution**: Install the package in editable mode:

.. code-block:: bash

    pip install -e .

Test Failures
-------------

**Problem**: Tests fail with import errors

**Solution**: Ensure you're in the virtual environment and dependencies are installed:

.. code-block:: bash

    source venv/bin/activate
    pip install -e ".[dev]"
    pytest

Documentation Build Errors
---------------------------

**Problem**: Sphinx build fails with missing modules

**Solution**: Install documentation dependencies:

.. code-block:: bash

    pip install -r requirements-docs.txt

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

    rm -rf venv
    python -m venv venv
    source venv/bin/activate
    pip install -e ".[dev]"

Dependency Conflicts
--------------------

**Problem**: Dependency version conflicts

**Solution**: Update pip and reinstall:

.. code-block:: bash

    pip install --upgrade pip
    pip install -e ".[dev]" --force-reinstall

Getting Help
============

- Check the :doc:`contributing` guide
- Review existing `GitHub Issues <https://github.com/NeuroTechX/eegprep/issues>`_
- Ask in GitHub Discussions
- Contact the maintainers

Happy developing!
