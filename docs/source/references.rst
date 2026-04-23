.. _references:

========================
References and Citations
========================

Key Publications
================

EEG Preprocessing Methods
-------------------------

The following papers describe key preprocessing methods implemented in EEGPrep:

**Artifact Removal and Cleaning**

- Delorme, A., & Makeig, S. (2004). EEGLAB: an open source toolbox for analysis of single-trial EEG dynamics including independent component analysis. *Journal of Neuroscience Methods*, 134(1), 9-21.
  
  - Foundational paper for EEGLAB and many preprocessing techniques

- Kothe, C. A., & Makeig, S. (2013). BCILAB: a platform for brain–computer interface development. *Journal of Neural Engineering*, 10(5), 056014.
  
  - Describes ASR (Artifact Subspace Reconstruction) algorithm

- Onton, J., Westerfield, M., Townsend, J., & Makeig, S. (2006). Imaging human EEG dynamics using independent component analysis. *Neuroscience & Biobehavioral Reviews*, 30(6), 808-822.
  
  - ICA for EEG analysis

**Independent Component Analysis (ICA)**

- Hyvärinen, A., & Oja, E. (2000). Independent component analysis: algorithms and applications. *Neural Networks*, 13(4-5), 411-430.
  
  - Comprehensive ICA review

- Bell, A. J., & Sejnowski, T. J. (1995). An information-maximization approach to blind separation and blind deconvolution. *Neural Computation*, 7(6), 1129-1159.
  
  - Infomax ICA algorithm

**ICLabel Component Classification**

- Pion-Tonachini, L., Kreutz-Delgado, K., & Makeig, S. (2019). ICLabel: Automated electroencephalographic independent component classification, labeling and brain source estimation. *NeuroImage*, 198, 181-197.
  
  - Deep learning-based IC classification

**BIDS Format**

- Gorgolewski, K. J., Auer, T., Calhoun, V. D., et al. (2016). The brain imaging data structure, a format for organizing and describing outputs of neuroimaging experiments. *Scientific Data*, 3, 160044.
  
  - BIDS specification paper

- Pernet, C. R., Appelhoff, S., Gorgolewski, K. J., et al. (2019). EEG-BIDS, an extension to the brain imaging data structure for electroencephalography. *Scientific Data*, 6, 103.
  
  - EEG-BIDS extension

**Signal Processing**

- Widmann, A., Schröger, E., & Maess, B. (2015). Digital filter design for electrophysiological data–a practical approach. *Journal of Neuroscience Methods*, 250, 34-46.
  
  - Filter design for EEG

Related Tools
=============

EEGLAB
------

**Website**: `https://sccn.ucsd.edu/eeglab/ <https://sccn.ucsd.edu/eeglab/>`_

**Description**: MATLAB-based EEG analysis toolbox with extensive preprocessing capabilities.

**Key Features**:

- Interactive GUI
- Comprehensive preprocessing tools
- ICA and component analysis
- Large plugin ecosystem
- Established in neuroscience community

**When to use**: Interactive exploration, MATLAB workflows, extensive plugin ecosystem

MNE-Python
----------

**Website**: `https://mne.tools/ <https://mne.tools/>`_

**Description**: Python package for MEG and EEG analysis.

**Key Features**:

- General neuroimaging (EEG, MEG, fMRI)
- Extensive analysis tools
- Source localization
- Time-frequency analysis
- Large community

**When to use**: Comprehensive analysis, source localization, Python workflows

Fieldtrip
---------

**Website**: `http://www.fieldtriptoolbox.org/ <http://www.fieldtriptoolbox.org/>`_

**Description**: MATLAB toolbox for MEG and EEG analysis.

**Key Features**:

- Comprehensive preprocessing
- Source analysis
- Statistical testing
- Good documentation
- Active community

**When to use**: MATLAB workflows, source analysis, statistical testing

Brainstorm
----------

**Website**: `https://neuroimage.usc.edu/brainstorm/ <https://neuroimage.usc.edu/brainstorm/>`_

**Description**: MATLAB-based neuroimaging software for MEG and EEG.

**Key Features**:

- Interactive visualization
- Source localization
- Preprocessing tools
- Good for clinical applications
- User-friendly interface

**When to use**: Interactive analysis, source localization, clinical applications

External Resources
===================

Tutorials and Documentation
---------------------------

**EEG Analysis Tutorials**

- `MNE-Python Tutorials <https://mne.tools/stable/auto_tutorials/index.html>`_ - Comprehensive EEG/MEG analysis tutorials
- `EEGLAB Wiki <https://sccn.ucsd.edu/wiki/EEGLAB>`_ - EEGLAB documentation and tutorials
- `Fieldtrip Tutorials <http://www.fieldtriptoolbox.org/tutorial/>`_ - Fieldtrip analysis tutorials

**Signal Processing**

- `Digital Signal Processing <https://en.wikipedia.org/wiki/Digital_signal_processing>`_ - Wikipedia overview
- `Scipy Signal Processing <https://docs.scipy.org/doc/scipy/reference/signal.html>`_ - Python signal processing library

**Machine Learning**

- `Scikit-learn Documentation <https://scikit-learn.org/>`_ - Machine learning in Python
- `PyTorch Documentation <https://pytorch.org/docs/>`_ - Deep learning framework

Forums and Communities
----------------------

**GitHub**

- `EEGPrep Issues <https://github.com/NeuroTechX/eegprep/issues>`_ - Report bugs and ask questions
- `EEGPrep Discussions <https://github.com/NeuroTechX/eegprep/discussions>`_ - Community discussions

**NeuroTalk**

- `NeuroTalk Forums <https://www.neurotalk.org/>`_ - Neuroscience discussion forums
- EEG and neuroimaging discussions

**Stack Overflow**

- `EEG Tag <https://stackoverflow.com/questions/tagged/eeg>`_ - EEG-related questions
- `Signal Processing Tag <https://stackoverflow.com/questions/tagged/signal-processing>`_ - Signal processing questions

**Reddit**

- `r/neuroscience <https://www.reddit.com/r/neuroscience/>`_ - Neuroscience community
- `r/MachineLearning <https://www.reddit.com/r/MachineLearning/>`_ - Machine learning discussions

Datasets
--------

**Public EEG Datasets**

- `OpenNeuro <https://openneuro.org/>`_ - Open neuroimaging datasets in BIDS format
- `PhysioNet <https://physionet.org/>`_ - Biomedical signal databases
- `EEG Motor Movement/Imagery Dataset <https://physionet.org/content/eegmmidb/1.0.0/>`_ - Motor imagery EEG data

Citation Information
====================

How to Cite EEGPrep
-------------------

If you use EEGPrep in your research, please cite it as:

**BibTeX**:

.. code-block:: bibtex

    @software{eegprep2024,
      title={EEGPrep: A comprehensive Python EEG preprocessing pipeline},
      author={EEGPrep Contributors},
      year={2024},
      url={https://github.com/NeuroTechX/eegprep}
    }

**APA Format**:

EEGPrep Contributors. (2024). EEGPrep: A comprehensive Python EEG preprocessing pipeline. Retrieved from https://github.com/NeuroTechX/eegprep

**Chicago Format**:

EEGPrep Contributors. "EEGPrep: A comprehensive Python EEG preprocessing pipeline." Accessed 2024. https://github.com/NeuroTechX/eegprep.

Citing Dependencies
-------------------

If you use specific algorithms, please also cite the original papers:

**For ASR (Artifact Subspace Reconstruction)**:

.. code-block:: bibtex

    @article{kothe2013bcilab,
      title={BCILAB: a platform for brain--computer interface development},
      author={Kothe, Christian A and Makeig, Scott},
      journal={Journal of Neural Engineering},
      volume={10},
      number={5},
      pages={056014},
      year={2013},
      publisher={IOP Publishing}
    }

**For ICLabel**:

.. code-block:: bibtex

    @article{pion2019iclabel,
      title={ICLabel: Automated electroencephalographic independent component classification, labeling and brain source estimation},
      author={Pion-Tonachini, Luca and Kreutz-Delgado, Kenneth and Makeig, Scott},
      journal={NeuroImage},
      volume={198},
      pages={181--197},
      year={2019},
      publisher={Elsevier}
    }

**For EEGLAB**:

.. code-block:: bibtex

    @article{delorme2004eeglab,
      title={EEGLAB: an open source toolbox for analysis of single-trial EEG dynamics including independent component analysis},
      author={Delorme, Arnaud and Makeig, Scott},
      journal={Journal of Neuroscience Methods},
      volume={134},
      number={1},
      pages={9--21},
      year={2004},
      publisher={Elsevier}
    }

Acknowledgments
===============

Contributors
------------

EEGPrep is developed and maintained by the NeuroTechX community. We thank all contributors who have helped improve the project through code contributions, bug reports, and feedback.

Funding
-------

EEGPrep development has been supported by:

- NeuroTechX community
- Open-source software initiatives
- Academic institutions

Inspiration and Acknowledgments
-------------------------------

EEGPrep builds upon the excellent work of:

- **EEGLAB**: For pioneering EEG preprocessing and analysis tools
- **MNE-Python**: For comprehensive neuroimaging analysis
- **Fieldtrip**: For robust signal processing methods
- **Brainstorm**: For user-friendly neuroimaging software

We acknowledge the neuroscience and signal processing communities for their contributions to EEG analysis methods.

Related Publications Using EEGPrep
==================================

If you've published research using EEGPrep, we'd love to hear about it! Please open an issue or discussion on GitHub to share your work.

Getting Help with References
=============================

- Check the :doc:`user_guide/index` for implementation details
- Review :doc:`examples/index` for practical examples
- Search `GitHub Issues <https://github.com/NeuroTechX/eegprep/issues>`_ for related discussions
- Contact the maintainers for citation questions

For more information about EEG analysis methods, see the :doc:`glossary` for terminology definitions.
