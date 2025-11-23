.. _glossary:

========
Glossary
========

This glossary defines key terms used in EEG analysis and signal processing.

EEG Terminology
===============

.. glossary::

    Electrode
        A conductor used to record electrical activity from the brain. Electrodes are placed on the scalp to measure voltage differences between different brain regions.

    Channel
        A single recording from one electrode. An EEG recording typically has multiple channels (e.g., 64 channels from 64 electrodes).

    Montage
        The arrangement and labeling of electrodes on the scalp. Common montages include 10-20, 10-10, and 10-5 systems.

    Artifact
        Unwanted electrical activity in the EEG signal that does not originate from brain activity. Common artifacts include eye movements (EOG), muscle activity (EMG), and electrical noise.

    Epoch
        A segment of EEG data, typically time-locked to a stimulus or event. Epochs are used for event-related potential (ERP) analysis.

    Trial
        A single experimental event or stimulus presentation. Multiple trials are typically averaged to improve signal-to-noise ratio.

    Baseline
        A reference period of EEG activity, typically before stimulus presentation. Baseline correction removes the average baseline activity from each epoch.

    Event
        A marker in the EEG data indicating when something occurred (e.g., stimulus presentation, button press). Events are used to segment data into epochs.

    Marker
        A label or timestamp indicating an event in the EEG recording. Markers are used to align EEG data with experimental events.

    Sampling Rate
        The number of times per second that the EEG signal is measured. Common sampling rates are 250 Hz, 500 Hz, and 1000 Hz.

    Hz (Hertz)
        Unit of frequency, representing cycles per second. EEG sampling rates and frequency bands are measured in Hz.

    Frequency Band
        A range of frequencies in the EEG signal. Common bands include:
        
        - Delta (0.5-4 Hz): Sleep and deep relaxation
        - Theta (4-8 Hz): Drowsiness and meditation
        - Alpha (8-12 Hz): Relaxation and idling
        - Beta (12-30 Hz): Active thinking and concentration
        - Gamma (30-100 Hz): High-level cognitive processing

    Power Spectral Density (PSD)
        The distribution of power across different frequencies in the EEG signal. Used to analyze frequency content and identify abnormalities.

    Coherence
        A measure of the correlation between EEG signals at different electrodes or frequencies. High coherence indicates synchronized activity.

    Phase
        The position of a wave in its cycle. Phase differences between channels can indicate functional connectivity.

    Amplitude
        The magnitude of the EEG signal, typically measured in microvolts (µV). Larger amplitudes indicate stronger electrical activity.

    Latency
        The time delay between a stimulus and a response in the EEG signal. Used to measure processing speed and neural efficiency.

    Component
        A distinct pattern or source of activity in the EEG signal. Components can be identified through ICA or other decomposition methods.

    Dipole
        A mathematical model of a neural source consisting of two opposite charges. Used to estimate the location of brain activity from EEG data.

Signal Processing Terms
=======================

.. glossary::

    Filter
        A mathematical operation that removes or attenuates certain frequencies from the signal. Common types include:
        
        - Highpass: Removes low frequencies
        - Lowpass: Removes high frequencies
        - Bandpass: Keeps frequencies within a range
        - Notch: Removes a specific frequency (e.g., 50/60 Hz line noise)

    Filtering
        The process of applying a filter to remove unwanted frequencies from the EEG signal.

    Cutoff Frequency
        The frequency at which a filter begins to attenuate the signal. For a highpass filter at 1 Hz, frequencies below 1 Hz are attenuated.

    Filter Order
        The steepness of the filter's frequency response. Higher order filters have steeper slopes but may introduce more distortion.

    Convolution
        A mathematical operation used to apply filters to signals. Convolution combines the signal with a filter kernel.

    Fourier Transform
        A mathematical operation that converts a signal from the time domain to the frequency domain. Used to analyze the frequency content of EEG signals.

    Fast Fourier Transform (FFT)
        An efficient algorithm for computing the Fourier Transform. Commonly used for frequency analysis of EEG data.

    Wavelet
        A small oscillating wave used for time-frequency analysis. Wavelets can represent both time and frequency information simultaneously.

    Spectrogram
        A visual representation of the frequency content of a signal over time. Shows how the power in different frequency bands changes over time.

    Resampling
        Changing the sampling rate of a signal. Downsampling reduces the sampling rate (and data size), while upsampling increases it.

    Downsampling
        Reducing the sampling rate of a signal by removing samples. Used to reduce data size and computation time.

    Interpolation
        Estimating values between known data points. Used in downsampling and for estimating missing data.

    Artifact Subspace Reconstruction (ASR)
        An algorithm for removing artifacts by identifying and removing the subspace containing artifact activity. Effective for removing large amplitude artifacts.

    Independent Component Analysis (ICA)
        A blind source separation technique that decomposes the EEG signal into independent components. Used to identify and remove artifacts and neural sources.

    Principal Component Analysis (PCA)
        A dimensionality reduction technique that identifies the directions of maximum variance in the data. Often used as a preprocessing step for ICA.

    Blind Source Separation
        A technique for separating mixed signals into their original sources without knowing the mixing process. ICA is a type of blind source separation.

    Whitening
        A preprocessing step that removes correlations and normalizes the variance of the data. Often used before ICA.

    Infomax ICA
        An ICA algorithm that maximizes information flow through a neural network. Commonly used for EEG analysis.

    FastICA
        An efficient ICA algorithm based on fixed-point iteration. Faster than Infomax ICA but may be less stable.

    Picard ICA
        A robust ICA algorithm that combines advantages of Infomax and FastICA. Often provides better results than other ICA algorithms.

Data Format Terms
=================

.. glossary::

    BIDS
        Brain Imaging Data Structure. A standardized format for organizing neuroimaging data. Ensures consistency and enables automated processing.

    EEGLAB
        A MATLAB toolbox for EEG analysis. EEGLAB format (.set and .fdt files) is widely used in neuroscience research.

    .set file
        EEGLAB header file containing metadata about the EEG recording (sampling rate, channel names, events, etc.).

    .fdt file
        EEGLAB data file containing the actual EEG signal data. Paired with a .set file.

    EDF
        European Data Format. A standard format for biomedical signals including EEG. Widely supported across different software packages.

    BrainVision
        A data format used by BrainVision Recorder software. Consists of three files: .vhdr (header), .vmrk (markers), and .eeg (data).

    MNE
        MNE-Python format for storing neuroimaging data. Includes Raw and Epochs objects for continuous and epoched data.

    HDF5
        Hierarchical Data Format 5. A flexible format for storing large amounts of data. Used by EEGPrep for efficient data storage.

    FIF
        Functional Image File. MNE-Python's native format for storing neuroimaging data.

    Neuroscan
        A data format used by Neuroscan software. Typically stored in .cnt files.

Statistical Terms
==================

.. glossary::

    Z-score
        A standardized score indicating how many standard deviations a value is from the mean. Used to identify outliers and normalize data.

    Threshold
        A cutoff value used to classify data points. Values above the threshold are classified as one category, below as another.

    Artifact Detection Threshold
        A threshold used to identify artifacts in the EEG signal. Data points exceeding this threshold are marked as artifacts.

    Variance
        A measure of how spread out data is from the mean. High variance indicates high variability in the signal.

    Standard Deviation
        The square root of variance. Indicates the typical deviation of data points from the mean.

    Mean
        The average value of a dataset. Calculated by summing all values and dividing by the number of values.

    Median
        The middle value in a sorted dataset. Less sensitive to outliers than the mean.

    Outlier
        A data point that is significantly different from other data points. Often indicates artifacts or errors.

    Correlation
        A measure of the linear relationship between two variables. Ranges from -1 (perfect negative correlation) to 1 (perfect positive correlation).

    Covariance
        A measure of how two variables change together. Related to correlation but not normalized.

    Signal-to-Noise Ratio (SNR)
        The ratio of signal power to noise power. Higher SNR indicates cleaner data.

    Noise
        Unwanted random fluctuations in the signal. Can come from electrical interference, electrode movement, or biological sources.

    Baseline Correction
        Subtracting the average baseline activity from each epoch to remove slow drifts and offsets.

    Normalization
        Scaling data to a standard range (e.g., 0-1 or -1 to 1). Used to make data comparable across different scales.

    Standardization
        Transforming data to have mean 0 and standard deviation 1. Also called z-score normalization.

Related Concepts
================

.. glossary::

    Preprocessing
        The process of cleaning and preparing raw EEG data for analysis. Includes filtering, artifact removal, and other data quality improvements.

    Pipeline
        A sequence of preprocessing and analysis steps applied to data in a specific order. Ensures reproducibility and consistency.

    Reproducibility
        The ability to obtain the same results when applying the same analysis to the same data. Important for scientific validity.

    Validation
        The process of checking that data meets quality criteria and that analysis methods are appropriate.

    Quality Assurance
        Systematic checking of data and analysis to ensure accuracy and reliability.

    Batch Processing
        Processing multiple datasets using the same pipeline. Efficient for analyzing large numbers of subjects.

    Parallel Processing
        Processing multiple datasets simultaneously using multiple processors or cores. Speeds up batch processing.

    Real-time Processing
        Processing data as it is being recorded, without waiting for the entire recording to complete.

    Offline Processing
        Processing data after it has been completely recorded. Allows for more sophisticated analysis but introduces latency.

Cross-References
================

For more information on specific topics, see:

- :doc:`user_guide/index` - Detailed usage guides
- :doc:`api/index` - API reference
- :doc:`examples/index` - Example scripts
- :doc:`references` - Key publications and resources
- :doc:`faq` - Frequently asked questions

Additional Resources
====================

- `EEGLAB Wiki <https://sccn.ucsd.edu/wiki/EEGLAB>`_ - EEGLAB documentation
- `MNE-Python Glossary <https://mne.tools/stable/glossary.html>`_ - MNE-Python glossary
- `Signal Processing Basics <https://en.wikipedia.org/wiki/Signal_processing>`_ - Wikipedia overview
- `EEG Analysis Tutorials <https://mne.tools/stable/auto_tutorials/index.html>`_ - MNE-Python tutorials
