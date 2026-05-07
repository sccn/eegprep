.. _installation:

============
Installation
============

This guide covers the installation of eegprep and its dependencies.

System Requirements
===================

Before installing eegprep, ensure your system meets the following requirements:

- **Python**: 3.10 or higher
- **uv**: Default package and environment manager for source installs and development
- **pip**: Optional fallback for published package installs
- **conda**: Optional environment manager when required by a local setup
- **Operating System**: Linux, macOS, or Windows
- **RAM**: Minimum 4GB (8GB+ recommended for large datasets)
- **Disk Space**: At least 500MB for installation and dependencies

Installation Methods
====================

Using uv (Recommended)
----------------------

For a project managed by uv, add eegprep with:

.. code-block:: bash

    uv add eegprep

To include all optional EEGPrep dependencies:

.. code-block:: bash

    uv add "eegprep[all]"

Using pip
---------

Published EEGPrep packages can also be installed with pip in non-uv
environments:

.. code-block:: bash

    pip install eegprep

Using conda
-----------

If you prefer conda, you can install eegprep from the conda-forge channel:

.. code-block:: bash

    conda install -c conda-forge eegprep

To create a new conda environment with eegprep:

.. code-block:: bash

    conda create -n eegprep-env python=3.10 eegprep
    conda activate eegprep-env

From Source
-----------

To install eegprep from source for development:

.. code-block:: bash

    git clone https://github.com/sccn/eegprep.git
    cd eegprep
    uv sync --group dev

``uv sync`` creates the project environment, installs EEGPrep in editable mode,
and uses ``uv.lock`` for reproducible dependency resolution.

Optional Dependencies
=====================

eegprep has several optional dependencies that enable additional functionality:

PyTorch (for GPU acceleration)
------------------------------

To use GPU-accelerated processing with PyTorch:

.. code-block:: bash

    uv add torch

For CUDA support (NVIDIA GPUs):

.. code-block:: bash

    uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

For CPU-only PyTorch:

.. code-block:: bash

    uv add torch --index-url https://download.pytorch.org/whl/cpu

EEGLAB I/O Support
------------------

To enable reading and writing EEGLAB .set files:

.. code-block:: bash

    uv add eeglabio

MNE-Python Integration
----------------------

For integration with MNE-Python:

.. code-block:: bash

    uv add mne

AMICA
-----

EEGPrep can run AMICA through ``eeg_amica`` and ``ICAAlgorithm='amica'``, but
published EEGPrep packages do not include AMICA binaries. This keeps Python
package downloads smaller and avoids shipping platform-specific executables.

To use AMICA, install or download an AMICA executable separately, then provide
it in one of these ways:

.. code-block:: bash

    export AMICA_BINARY=/path/to/amica15ub

or pass the path directly:

.. code-block:: python

    eeg = eeg_amica(eeg, amica_binary="/path/to/amica15ub")

Source checkouts may contain development binaries under ``src/eegprep/bin/``.
Those files are for local development and tests, not for package distribution.

Documentation Building
----------------------

To build the documentation locally:

.. code-block:: bash

    uv sync --extra docs

All Optional Dependencies
--------------------------

To install all optional dependencies at once:

.. code-block:: bash

    uv add "eegprep[all]"

Or with specific extras:

.. code-block:: bash

    uv add "eegprep[torch,eeglabio,gui,docs]"

Verification
============

After installation, verify that eegprep is correctly installed by running:

.. code-block:: python

    import eegprep
    print(eegprep.__version__)

You should see the version number printed without any errors.

To verify all core modules are available:

.. code-block:: python

    from eegprep import (
        pop_loadset,
        pop_saveset,
        clean_artifacts,
        iclabel,
        pop_resample,
        pop_reref,
        topoplot
    )
    print("All core modules imported successfully!")

Troubleshooting
===============

Import Errors
-------------

**Problem**: ``ModuleNotFoundError: No module named 'eegprep'``

**Solution**: Ensure eegprep is installed:

.. code-block:: bash

    uv add eegprep

If installing from source, ensure you're in the correct directory and use:

.. code-block:: bash

    uv sync --group dev

Version Conflicts
-----------------

**Problem**: Conflicts with NumPy, SciPy, or other dependencies

**Solution**: Create a fresh virtual environment:

.. code-block:: bash

    uv venv eegprep_env
    source eegprep_env/bin/activate  # On Windows: eegprep_env\Scripts\activate
    uv pip install eegprep

PyTorch Installation Issues
----------------------------

**Problem**: PyTorch installation fails or GPU not detected

**Solution**:

1. Check your CUDA version:

.. code-block:: bash

    nvidia-smi

2. Install the matching PyTorch version from https://pytorch.org/get-started/locally/

3. Verify PyTorch installation:

.. code-block:: python

    import torch
    print(torch.cuda.is_available())

EEGLAB File Format Issues
--------------------------

**Problem**: Cannot read .set files

**Solution**: Install eeglabio:

.. code-block:: bash

    uv add eeglabio

Then verify:

.. code-block:: python

    from eegprep import pop_loadset
    # Should work without errors

Memory Issues
-------------

**Problem**: Out of memory errors when processing large datasets

**Solution**:

1. Process data in chunks or epochs
2. Reduce the number of channels if possible
3. Increase available RAM or use a machine with more memory
4. Use GPU acceleration if available

Getting Help
============

If you encounter issues not covered here:

1. Check the `FAQ <faq.rst>`_ section
2. Review the `Common Issues <common_issues.rst>`_ guide
3. Visit the `GitHub Issues <https://github.com/sccn/eegprep/issues>`_ page
4. Check the `API Documentation <../api/index.rst>`_

Next Steps
==========

After successful installation, proceed to the :ref:`quickstart` guide to learn how to use eegprep.
