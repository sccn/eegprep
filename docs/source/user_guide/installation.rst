.. _installation:

============
Installation
============

This guide covers the installation of eegprep and its dependencies.

System Requirements
===================

Before installing eegprep, ensure your system meets the following requirements:

- **Python**: 3.10 or higher
- **pip**: Latest version (for pip installation)
- **conda**: Latest version (for conda installation, optional)
- **Operating System**: Linux, macOS, or Windows
- **RAM**: Minimum 4GB (8GB+ recommended for large datasets)
- **Disk Space**: At least 500MB for installation and dependencies

Installation Methods
====================

Using pip (Recommended)
-----------------------

The easiest way to install eegprep is using pip:

.. code-block:: bash

    pip install eegprep

To upgrade an existing installation:

.. code-block:: bash

    pip install --upgrade eegprep

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
    pip install -e .

The ``-e`` flag installs the package in editable mode, allowing you to modify the source code and see changes immediately.

Optional Dependencies
=====================

eegprep has several optional dependencies that enable additional functionality:

PyTorch (for GPU acceleration)
------------------------------

To use GPU-accelerated processing with PyTorch:

.. code-block:: bash

    pip install torch

For CUDA support (NVIDIA GPUs):

.. code-block:: bash

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

For CPU-only PyTorch:

.. code-block:: bash

    pip install torch --index-url https://download.pytorch.org/whl/cpu

EEGLAB I/O Support
------------------

To enable reading and writing EEGLAB .set files:

.. code-block:: bash

    pip install eeglabio

MNE-Python Integration
----------------------

For integration with MNE-Python:

.. code-block:: bash

    pip install mne

Documentation Building
----------------------

To build the documentation locally:

.. code-block:: bash

    pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints

All Optional Dependencies
--------------------------

To install all optional dependencies at once:

.. code-block:: bash

    pip install eegprep[all]

Or with specific extras:

.. code-block:: bash

    pip install eegprep[torch,mne,docs]

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

    pip install eegprep

If installing from source, ensure you're in the correct directory and use:

.. code-block:: bash

    pip install -e .

Version Conflicts
-----------------

**Problem**: Conflicts with NumPy, SciPy, or other dependencies

**Solution**: Create a fresh virtual environment:

.. code-block:: bash

    python -m venv eegprep_env
    source eegprep_env/bin/activate  # On Windows: eegprep_env\Scripts\activate
    pip install eegprep

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

    pip install eeglabio

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
