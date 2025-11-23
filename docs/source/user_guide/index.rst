.. _user_guide:

==========
User Guide
==========

Welcome to the eegprep User Guide! This comprehensive guide provides practical documentation for using eegprep in your EEG research and analysis workflows.

Whether you're just getting started with eegprep or looking to master advanced preprocessing techniques, this guide has you covered. We've organized the documentation into logical sections to help you find what you need quickly.

Learning Path
=============

We recommend following this learning path based on your experience level:

**Beginner**
  1. Start with :ref:`installation` to set up eegprep
  2. Follow the :ref:`quickstart` guide for a 5-minute introduction
  3. Read :ref:`preprocessing_pipeline` to understand the preprocessing workflow

**Intermediate**
  1. Explore :ref:`configuration` for parameter tuning
  2. Learn :ref:`bids_workflow` for batch processing
  3. Review :ref:`preprocessing_pipeline` for detailed step-by-step information

**Advanced**
  1. Master :ref:`advanced_topics` for custom pipelines
  2. Explore :ref:`configuration` for advanced settings
  3. Integrate with :ref:`advanced_topics` for MNE-Python and parallel processing

Getting Started
===============

.. toctree::
   :maxdepth: 2

   installation
   quickstart

Core Concepts
=============

.. toctree::
   :maxdepth: 2

   preprocessing_pipeline
   configuration

Data Workflows
==============

.. toctree::
   :maxdepth: 2

   bids_workflow

Advanced Topics
===============

.. toctree::
   :maxdepth: 2

   advanced_topics

Quick Reference
===============

**Common Tasks**

- :ref:`installation` - Install eegprep
- :ref:`quickstart` - Load, preprocess, and save EEG data
- :ref:`preprocessing_pipeline` - Understand preprocessing steps
- :ref:`bids_workflow` - Process BIDS datasets
- :ref:`configuration` - Configure preprocessing parameters
- :ref:`advanced_topics` - Create custom pipelines

**Key Functions**

- :func:`eegprep.pop_loadset` - Load EEG data
- :func:`eegprep.pop_saveset` - Save EEG data
- :func:`eegprep.clean_artifacts` - Comprehensive artifact removal
- :func:`eegprep.iclabel` - Classify ICA components
- :func:`eegprep.pop_resample` - Resample data
- :func:`eegprep.pop_eegfiltnew` - Filter data
- :func:`eegprep.bids_preproc` - Batch process BIDS datasets

**Configuration**

- :class:`eegprep.EEG_OPTIONS` - Configuration object
- Preprocessing parameters
- Custom preprocessing chains

**Integration**

- MNE-Python integration
- EEGLAB compatibility
- BIDS support

Documentation Structure
=======================

**Installation**
  Complete installation guide covering system requirements, installation methods, optional dependencies, verification, and troubleshooting.

**Quick Start**
  5-minute introduction to eegprep with practical examples covering loading data, preprocessing, saving results, and visualization.

**Preprocessing Pipeline**
  Detailed overview of the preprocessing pipeline including all steps, parameter tuning, quality control, and common issues.

**BIDS Workflow**
  Guide to working with BIDS-formatted datasets including loading, batch processing, output structure, and integration with other tools.

**Configuration**
  Comprehensive guide to configuring eegprep including EEG_OPTIONS, common parameters, custom preprocessing chains, and advanced settings.

**Advanced Topics**
  Advanced topics for experienced users including custom pipelines, extending the pipeline, MNE-Python integration, parallel processing, and performance optimization.

Key Concepts
============

**EEG Data Structure**
  eegprep uses the EEGobj class to represent EEG data, which is compatible with EEGLAB format.

**Preprocessing Pipeline**
  The preprocessing pipeline consists of sequential steps: channel selection, artifact removal, channel interpolation, resampling, filtering, ICA decomposition, and component classification.

**BIDS Format**
  Brain Imaging Data Structure (BIDS) is a standardized format for organizing neuroimaging data, enabling consistent and reproducible analysis.

**ICA Decomposition**
  Independent Component Analysis (ICA) decomposes EEG data into independent components, which can be classified as brain activity or artifacts.

**Component Classification**
  ICLabel automatically classifies ICA components into categories such as brain, muscle, eye, heart, line noise, and channel noise.

Getting Help
============

If you need help:

1. Check the relevant section in this guide
2. Review the :ref:`api_reference` documentation
3. Visit the `GitHub Issues <https://github.com/sccn/eegprep/issues>`_ page
4. Check the `GitHub Discussions <https://github.com/sccn/eegprep/discussions>`_ page

Contributing
============

We welcome contributions! If you find issues or have suggestions for improving the documentation, please:

1. Open an issue on `GitHub <https://github.com/sccn/eegprep/issues>`_
2. Submit a pull request with improvements
3. Share your feedback in `GitHub Discussions <https://github.com/sccn/eegprep/discussions>`_

License
=======

eegprep is released under the GNU General Public License v3.0. See the LICENSE file for details.

Citation
========

If you use eegprep in your research, please cite:

.. code-block:: bibtex

    @software{eegprep2024,
      title={eegprep: A Python package for EEG preprocessing},
      author={SCCN},
      year={2024},
      url={https://github.com/sccn/eegprep}
    }

Acknowledgments
===============

eegprep is built on the foundations of EEGLAB and incorporates algorithms and methods from the EEG research community. We acknowledge the contributions of all researchers and developers who have contributed to EEG analysis methods.
