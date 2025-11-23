================
Example Gallery
================

Comprehensive EEG Preprocessing Workflows with eegprep
======================================================

This gallery contains professional-grade example scripts demonstrating best practices for EEG preprocessing using the eegprep package. Each example is designed to be self-contained, executable, and educational, following the standards established by leading neuroimaging packages such as MNE-Python and scikit-learn.

**Key Features of These Examples:**

- **Realistic Workflows**: Examples demonstrate complete preprocessing pipelines from raw data to analysis-ready datasets
- **Best Practices**: Code follows professional standards with comprehensive documentation and error handling
- **Reproducibility**: All examples use fixed random seeds and synthetic data for consistent results
- **Visualization**: Extensive matplotlib visualizations help understand preprocessing effects
- **Educational**: Detailed comments explain key concepts and parameter choices
- **Modular Design**: Examples can be adapted and combined for custom workflows

**Example Categories:**

1. **Basic Preprocessing** - Fundamental EEG preprocessing workflow including artifact cleaning and channel interpolation
2. **BIDS Integration** - Working with standardized BIDS-formatted EEG datasets
3. **Artifact Removal** - Comparison of different artifact removal methods and their effects
4. **ICA and ICLabel** - Independent Component Analysis with automatic component classification
5. **Channel Interpolation** - Identifying and recovering data from bad channels

**Getting Started:**

Each example can be run independently. To execute an example:

.. code-block:: python

    # Run the example script directly
    python plot_basic_preprocessing.py

Or within a Jupyter notebook:

.. code-block:: python

    # Load and execute the example
    exec(open('plot_basic_preprocessing.py').read())

**Data Requirements:**

All examples use synthetic EEG data generated within the script, so no external datasets are required. This makes the examples:

- **Lightweight**: No large data downloads needed
- **Fast**: Examples complete in seconds to minutes
- **Reproducible**: Identical results across different systems
- **Educational**: Synthetic data allows clear visualization of preprocessing effects

**Customization:**

The examples are designed to be easily customizable:

- Modify parameters to see their effects
- Adapt data generation code for your specific needs
- Combine techniques from multiple examples
- Use as templates for your own preprocessing pipelines

**References:**

For more information about the techniques used in these examples, see:

- Delorme, A., & Makeig, S. (2004). EEGLAB: an open source toolbox for analysis of single-trial EEG dynamics. Journal of Neuroscience Methods, 134(1), 9-21.
- Jas, M., Engemann, D. A., Bekhti, Y., Raimondo, F., & Gramfort, A. (2017). Autoreject: Automated artifact rejection for MEG and EEG data. NeuroImage, 159, 417-429.
- Pion-Tonachini, L., Kreutz-Delgado, K., & Makeig, S. (2019). ICLabel: An automated electroencephalographic independent component classifier, dataset, and web interface. NeuroImage, 198, 181-197.

**Contributing:**

If you have suggestions for new examples or improvements to existing ones, please contribute to the eegprep project on GitHub.
