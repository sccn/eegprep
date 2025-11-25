.. eegprep documentation master file, created by sphinx-quickstart on 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

========
eegprep
========

A comprehensive Python EEG preprocessing pipeline for neuroscience research.

.. raw:: html

   <div style="text-align: center; margin: 2rem 0;">
      <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">Get started with:</p>
      <div style="background-color: #f5f5f5; padding: 1rem; border-radius: 0.5rem; display: inline-block; border-left: 4px solid #0066cc;">
         <code style="font-size: 1.1rem; color: #0066cc; font-weight: 500;">pip install eegprep</code>
      </div>
   </div>

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/index
   user_guide/index
   examples/index
   contributing
   development
   faq
   references
   changelog
   glossary

Quick Start
===========

Installation
------------

Install eegprep using pip:

.. code-block:: bash

   pip install eegprep

Basic Usage
-----------

.. code-block:: python

   import eegprep
   from eegprep import EEGobj

   # Load EEG data
   eeg = EEGobj.load('data.set')

   # Apply preprocessing pipeline
   eeg = eegprep.clean_artifacts(eeg)
   eeg = eegprep.clean_flatlines(eeg)
   eeg = eegprep.clean_channels(eeg)

   # Save processed data
   eeg.save('data_processed.set')

Features
========

- **Comprehensive preprocessing**: Artifact removal, channel cleaning, and data quality assessment
- **ICA-based component classification**: Automatic IC labeling using ICLabel
- **BIDS compatibility**: Direct support for BIDS-formatted EEG datasets
- **MNE integration**: Seamless conversion between eegprep and MNE-Python formats
- **Flexible pipeline**: Mix and match preprocessing steps for your specific needs
- **Well-documented**: Extensive API documentation and user guides

Quick Links
===========

- :doc:`API Reference <api/index>` - Complete API documentation
- :doc:`User Guide <user_guide/index>` - Detailed usage guides and tutorials
- :doc:`Examples <examples/index>` - Example scripts and notebooks
- :doc:`Contributing <contributing>` - Contributing guidelines and code of conduct
- :doc:`Development <development>` - Development setup and debugging
- :doc:`FAQ <faq>` - Frequently asked questions
- :doc:`References <references>` - Key publications and related tools
- :doc:`Changelog <changelog>` - Version history and release notes
- :doc:`Glossary <glossary>` - EEG and signal processing terminology
- `GitHub Repository <https://github.com/NeuroTechX/eegprep>`_ - Source code and issue tracker

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
