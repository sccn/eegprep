.. _installation:

============
Installation
============

This guide covers the installation of eegprep and its dependencies.

System Requirements
===================

Before installing eegprep, ensure your system meets the following requirements:

- **Python**: 3.12 or higher
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

For a project managed by uv, add the ``eegprep`` package with:

.. code-block:: bash

    uv add eegprep

This installs EEGPrep. It does not install the separate ``eegprep-lean``
distribution (import name ``eegprep_lean``). NumPy, SciPy, Matplotlib,
psutil, pyedflib, and h5py are part of this base install. Loading, cleaning,
plotting, and EDF, BDF, and GDF export work with no brackets.

Octave, MNE, PyBIDS, and Picard are extras. Install them together with:

.. code-block:: bash

    uv add "eegprep[full]"

To include every optional extra, including the GUI, console, docs, and
classifier dependencies:

.. code-block:: bash

    uv add "eegprep[all]"

Using pip
---------

Published EEGPrep packages can also be installed with pip in non-uv
environments:

.. code-block:: bash

    pip install eegprep

From Source
-----------

To install eegprep from source for development:

.. code-block:: bash

    git clone https://github.com/sccn/eegprep.git
    cd eegprep
    uv sync --group dev

``uv sync`` creates the project environment, installs EEGPrep in editable mode,
and uses ``uv.lock`` for reproducible dependency resolution. The development
environment includes the GUI and console runtime dependencies, so a fresh
checkout can immediately launch ``uv run eegprep-console --full``. It also
installs the ``eeglab``, ``mne``, ``pybids``, and ``picard`` packages so the
test suite can exercise Octave, MNE conversion, BIDS import, and Picard ICA.

To develop or build documentation from source, include the docs extra:

.. code-block:: bash

    uv sync --group dev --extra docs

Optional Dependencies
=====================

eegprep has several optional dependencies that enable additional functionality.
Calling a function whose extra is not installed raises ``ImportError``. The
message names the library, the function, and the command to install it, for
example ``pip install "eegprep[mne]"``, and it mentions ``eegprep[full]``.
When another path does not need that library, the message says what to do
instead, such as leaving ``engine`` unset or using ``runica``.

``dir(eegprep)`` lists every public name. ``eegprep.help()`` prints those
names with a one-line summary, read from the function's docstring without
importing its module, and names the extra when one path of the function needs
it. ``help`` is not exported by ``from eegprep import *``.

``full`` restores the previous base install
-------------------------------------------

.. code-block:: bash

    uv add "eegprep[full]"

This installs four extras:

- ``eeglab`` (``oct2py``), the Octave runtime for
  ``pop_resample(..., engine="octave")``, ``iclabel(..., engine="octave")``,
  the RANSAC octave subroutine used by ``clean_channels``, and
  ``eeglab_clean_artifacts``.
- ``mne``, conversion to and from MNE objects, and the EDF, BDF, GDF, and
  EGI ``.mff`` readers in ``pop_biosig`` and ``pop_fileio``. BrainVision
  ``.eeg`` files, and ``.vhdr`` files when a block range is set, also use it.
  EEGLAB ``.set``, CNT, BrainVision ``.vhdr`` without a block range, and
  BrainVision Analyzer ``.mat`` exports do not.
- ``pybids``, BIDS import and ``bids_preproc``.
- ``picard`` (``python-picard``), Picard ICA. The default ``runica`` path
  does not need it. This extra also installs scikit-learn.

``eegprep[all]`` includes these four extras plus the GUI, console, docs, and
PyTorch extras.

Riemannian ASR (``clean_asr(..., useriemannian=True)`` and
``Distance="Riemannian"``) is included. It uses NumPy and SciPy; there is no
separate manifold package to install.

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

    uv sync --group dev --extra docs
    uv run --no-sync sphinx-build -b html docs/source docs/_build/html

The ``docs/Makefile`` target is also available:

.. code-block:: bash

    uv run --no-sync make -C docs html

All Optional Dependencies
--------------------------

To install all optional dependencies at once:

.. code-block:: bash

    uv add "eegprep[all]"

Or with specific extras:

.. code-block:: bash

    uv add "eegprep[torch,gui,docs]"

Verification
============

After installation, verify that eegprep is correctly installed by running:

.. code-block:: python

    import eegprep
    print(eegprep.__version__)
    eegprep.help()

You should see the version number, then a list of public names, printed
without any errors.

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

EEGPrep reads ``.set`` files directly with SciPy (``scipy.io.loadmat``) and
``h5py`` for newer MATLAB v7.3 files, and writes them with
``scipy.io.savemat``. These libraries are installed automatically, so a load
failure is almost always a path or format problem rather than a missing
dependency.

**Solution**: Check that the path points at an existing ``.set`` file and that
any companion ``.fdt`` data file sits next to it:

.. code-block:: python

    from pathlib import Path
    from eegprep import pop_loadset

    path = Path("sub-01_task-rest_eeg.set")
    assert path.is_file(), f"no such file: {path}"
    EEG = pop_loadset(str(path))

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

1. Check the :doc:`../faq` section
2. Review the troubleshooting sections in this installation guide and FAQ
3. Visit the `GitHub Issues <https://github.com/sccn/eegprep/issues>`_ page
4. Check the :doc:`../api/index`

Next Steps
==========

After successful installation, proceed to the :ref:`quickstart` guide to learn how to use eegprep.
