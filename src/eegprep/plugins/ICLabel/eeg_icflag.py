"""
Flag independent components based on ICLabel classifications.

This is a Python equivalent of EEGLAB's pop_icflag function.
"""

import numpy as np


def eeg_icflag(EEG, thresholds):
    """
    Flag independent components based on ICLabel classification probabilities.

    Parameters
    ----------
    EEG : dict
        EEG structure with ICLabel classifications in EEG['etc']['ic_classification']['ICLabel']['classifications']
    thresholds : array-like, shape (7, 2)
        Threshold matrix where each row corresponds to an IC class:
        [Brain, Muscle, Eye, Heart, Line Noise, Channel Noise, Other]
        Each row contains [min_threshold, max_threshold].
        Use NaN to ignore a class.

    Returns
    -------
    EEG : dict
        EEG structure with added 'reject' field containing flags for each component

    Examples
    --------
    # Flag components with Muscle > 0.9 OR Eye > 0.9
    thresholds = np.array([
        [np.nan, np.nan],  # Brain
        [0.9, 1.0],        # Muscle
        [0.9, 1.0],        # Eye
        [np.nan, np.nan],  # Heart
        [np.nan, np.nan],  # Line Noise
        [np.nan, np.nan],  # Channel Noise
        [np.nan, np.nan],  # Other
    ])
    EEG = eeg_icflag(EEG, thresholds)
    """
    # Extract IC classifications
    if 'etc' not in EEG or 'ic_classification' not in EEG['etc']:
        raise ValueError("EEG structure does not contain ICLabel classifications")

    ic_class = EEG['etc']['ic_classification']['ICLabel']['classifications']
    n_comps = ic_class.shape[0]

    # Convert thresholds to numpy array if needed
    thresholds = np.array(thresholds)

    if thresholds.shape != (7, 2):
        raise ValueError("Thresholds must be a 7x2 array")

    # Initialize reject flags (False = keep, True = reject)
    reject = np.zeros(n_comps, dtype=bool)

    # Check each class
    for class_idx in range(7):
        min_thresh = thresholds[class_idx, 0]
        max_thresh = thresholds[class_idx, 1]

        # Skip if both thresholds are NaN
        if np.isnan(min_thresh) and np.isnan(max_thresh):
            continue

        # Get probabilities for this class
        probs = ic_class[:, class_idx]

        # Apply thresholds
        if not np.isnan(min_thresh) and not np.isnan(max_thresh):
            # Both thresholds specified: flag if within range
            reject |= (probs >= min_thresh) & (probs <= max_thresh)
        elif not np.isnan(min_thresh):
            # Only min threshold: flag if >= min
            reject |= (probs >= min_thresh)
        elif not np.isnan(max_thresh):
            # Only max threshold: flag if <= max
            reject |= (probs <= max_thresh)

    # Store reject flags in EEG structure
    EEG['reject'] = {
        'gcompreject': reject.astype(int)  # Convert bool to int (0/1)
    }

    return EEG
