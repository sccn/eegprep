.. _changelog:

=========
Changelog
=========

All notable changes to EEGPrep are documented in this file. The format is based on `Keep a Changelog <https://keepachangelog.com/>`_.

Version History
===============

For a complete list of releases and detailed release notes, see the `GitHub Releases <https://github.com/NeuroTechX/eegprep/releases>`_ page.

Release Notes
=============

Version 1.0.0 (Current)
-----------------------

**Release Date**: 2024

This is the first stable release of EEGPrep, featuring a comprehensive EEG preprocessing pipeline.

Major Features
~~~~~~~~~~~~~~

- **Comprehensive Preprocessing Pipeline**: Complete suite of preprocessing tools for EEG data
- **Artifact Removal**: Multiple algorithms for detecting and removing artifacts
  
  - ASR (Artifact Subspace Reconstruction)
  - ICA-based artifact removal
  - Automatic artifact detection

- **Channel Management**: Tools for channel interpolation and quality assessment
  
  - Flat-line detection and removal
  - Channel interpolation using spherical spline
  - Channel quality assessment

- **ICA and Component Classification**: Independent Component Analysis with automatic labeling
  
  - FastICA and Infomax ICA implementations
  - ICLabel for automatic component classification
  - Component visualization and inspection

- **BIDS Support**: Native support for Brain Imaging Data Structure format
  
  - Load BIDS-formatted datasets
  - Save processed data in BIDS format
  - BIDS validation

- **MNE Integration**: Seamless conversion between EEGPrep and MNE-Python
  
  - Convert MNE Raw objects to EEGPrep
  - Convert EEGPrep to MNE format
  - Compatible with MNE analysis tools

- **Data Format Support**: Multiple input/output formats
  
  - EEGLAB (.set, .fdt)
  - EDF (European Data Format)
  - BrainVision (.vhdr, .vmrk, .eeg)
  - Neuroscan (.cnt)
  - HDF5

- **Comprehensive Documentation**: Extensive user guides and API documentation
  
  - User guide with tutorials
  - API reference
  - Example scripts
  - Contributing guidelines

Bug Fixes
~~~~~~~~~

- Fixed channel interpolation accuracy
- Improved ICA convergence
- Enhanced BIDS compatibility
- Fixed memory leaks in large dataset processing

Performance Improvements
~~~~~~~~~~~~~~~~~~~~~~~~

- Optimized ASR algorithm for faster processing
- Improved memory efficiency for large datasets
- GPU acceleration support for ICA
- Parallel processing capabilities

Breaking Changes
================

None for version 1.0.0 (first stable release).

Deprecations
============

None for version 1.0.0.

Future Plans
============

Planned Features
----------------

**Version 1.1.0** (Planned)

- Enhanced visualization tools
- Additional artifact detection algorithms
- Improved GPU support
- Extended BIDS support

**Version 1.2.0** (Planned)

- Real-time preprocessing capabilities
- Advanced statistical analysis tools
- Machine learning integration
- Web-based interface

**Version 2.0.0** (Long-term)

- Major API improvements
- Advanced source localization
- Integration with other neuroimaging modalities
- Cloud-based processing

Roadmap
-------

**Short-term (Next 3 months)**

- [ ] Improve documentation with more examples
- [ ] Add more preprocessing algorithms
- [ ] Enhance error handling and validation
- [ ] Improve test coverage

**Medium-term (3-6 months)**

- [ ] Add real-time processing capabilities
- [ ] Implement advanced visualization
- [ ] Expand BIDS support
- [ ] Add machine learning integration

**Long-term (6+ months)**

- [ ] Major API redesign
- [ ] Multi-modal neuroimaging support
- [ ] Cloud-based processing
- [ ] Web interface

Contributing to Development
============================

We welcome contributions! See the :doc:`contributing` guide for details on:

- How to report bugs
- How to suggest features
- How to submit pull requests
- Code style guidelines

Development Setup
-----------------

To set up a development environment, see the :doc:`development` guide.

Reporting Issues
================

Found a bug? Please report it on `GitHub Issues <https://github.com/NeuroTechX/eegprep/issues>`_.

When reporting issues, please include:

- Python version
- EEGPrep version
- Operating system
- Minimal code to reproduce the issue
- Error message and traceback

Feature Requests
================

Have an idea for a new feature? Open a `GitHub Discussion <https://github.com/NeuroTechX/eegprep/discussions>`_ or create an issue with the "enhancement" label.

When requesting features, please include:

- Clear description of the feature
- Use cases and motivation
- Potential implementation approach
- Related issues or discussions

Version Numbering
=================

EEGPrep follows `Semantic Versioning <https://semver.org/>`_:

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

Example: ``1.2.3`` (Major.Minor.Patch)

Release Schedule
================

- **Patch releases**: As needed for bug fixes
- **Minor releases**: Approximately every 2-3 months
- **Major releases**: As needed for significant changes

Upgrade Guide
=============

Upgrading to Latest Version
---------------------------

To upgrade to the latest version:

.. code-block:: bash

    pip install --upgrade eegprep

Or from source:

.. code-block:: bash

    git pull origin main
    pip install -e .

Checking Your Version
---------------------

Check your installed version:

.. code-block:: python

    import eegprep
    print(eegprep.__version__)

Or from command line:

.. code-block:: bash

    pip show eegprep

Migration Guides
================

No migration guides needed for version 1.0.0 (first stable release).

For future major versions, migration guides will be provided here.

Acknowledgments
===============

We thank all contributors and the neuroscience community for their support and feedback.

Special thanks to:

- EEGLAB developers for pioneering EEG preprocessing
- MNE-Python team for excellent neuroimaging tools
- NeuroTechX community for support and contributions

Getting Help
============

- Check the :doc:`faq` for common questions
- Review the :doc:`user_guide/index` for usage information
- See :doc:`examples/index` for practical examples
- Check :doc:`development` for development setup
- Open an issue on `GitHub <https://github.com/NeuroTechX/eegprep/issues>`_

For more information, visit the `GitHub Repository <https://github.com/NeuroTechX/eegprep>`_.
