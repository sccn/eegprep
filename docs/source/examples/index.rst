.. _examples:

========
Examples
========

This section contains executable example scripts demonstrating various eegprep workflows.
All examples are automatically executed by sphinx-gallery during documentation build,
generating a gallery with output plots and code.

.. toctree::
   :maxdepth: 2
   :caption: Example Gallery:

   ../auto_examples/index

Overview
========

The examples below demonstrate key eegprep functionality:

1. **Basic EEG Preprocessing Workflow** - A complete preprocessing pipeline including
   artifact cleaning and channel interpolation with visualization of results.

2. **BIDS Dataset Preprocessing** - Working with BIDS-formatted EEG datasets,
   demonstrating data loading and batch preprocessing workflows.

3. **Artifact Removal Comparison** - Comparing different artifact removal methods
   (clean_artifacts, ASR) with parameter effects and statistical analysis.

4. **ICA Decomposition and ICLabel Classification** - Independent Component Analysis
   with automatic component classification for artifact identification.

5. **Channel Interpolation** - Identifying bad channels and performing interpolation
   with quality assessment and visualization.

Example Categories
===================

Basic Examples
--------------

These examples demonstrate fundamental eegprep operations:

- **plot_basic_preprocessing.py** - Create synthetic EEG data, apply preprocessing steps,
  and visualize results. Covers artifact cleaning and channel interpolation.

Advanced Examples
-----------------

These examples show more sophisticated workflows:

- **plot_bids_pipeline.py** - Work with BIDS-formatted datasets, understand the BIDS
  structure, and apply preprocessing pipelines to multiple subjects.

- **plot_artifact_removal.py** - Compare different artifact removal methods, understand
  parameter effects, and analyze statistical properties of cleaned data.

- **plot_ica_and_iclabel.py** - Perform ICA decomposition, classify components using
  ICLabel, and identify artifacts for rejection.

Specialized Examples
--------------------

These examples focus on specific preprocessing tasks:

- **plot_channel_interpolation.py** - Identify bad channels using statistical criteria,
  perform interpolation, and assess interpolation quality.

Running the Examples
====================

All examples are designed to be self-contained and executable:

1. **Synthetic Data** - Examples use synthetic data to avoid external dependencies
2. **No Setup Required** - All necessary imports and data generation are included
3. **Visualization** - Examples generate plots showing preprocessing effects
4. **Documentation** - Each example includes detailed comments explaining each step

Example Structure
=================

Each example follows this structure:

1. **Title and Description** - Clear explanation of what the example demonstrates
2. **Imports** - Required libraries and eegprep modules
3. **Data Creation** - Generate synthetic EEG data with realistic characteristics
4. **Processing** - Apply eegprep functions with explanations
5. **Visualization** - Create plots showing results
6. **Analysis** - Print summary statistics and recommendations

Key Features
============

- **Executable Code** - All examples are runnable Python scripts
- **Matplotlib Plots** - Visualizations generated during execution
- **Print Output** - Summary statistics and results printed to console
- **Sphinx-Gallery Format** - Proper docstring format for auto-generation
- **Comments** - Detailed comments explaining each processing step
- **Realistic Data** - Synthetic data with realistic EEG characteristics

Learning Path
=============

We recommend exploring the examples in this order:

1. Start with **plot_basic_preprocessing.py** to understand the basic workflow
2. Move to **plot_artifact_removal.py** to learn about different cleaning methods
3. Explore **plot_channel_interpolation.py** for channel quality assessment
4. Study **plot_ica_and_iclabel.py** for advanced component analysis
5. Finally, check **plot_bids_pipeline.py** for working with real datasets

Tips for Using Examples
=======================

- **Modify Parameters** - Try changing preprocessing parameters to see effects
- **Inspect Plots** - Carefully examine generated plots to understand results
- **Read Comments** - Comments explain the reasoning behind each step
- **Check Output** - Print statements show important statistics and results
- **Adapt Code** - Use examples as templates for your own preprocessing pipelines

For more information, see the :ref:`user_guide` and :ref:`api_reference` documentation.
