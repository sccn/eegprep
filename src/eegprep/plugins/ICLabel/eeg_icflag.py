"""Flag independent components based on ICLabel classifications."""

import numpy as np


def eeg_icflag(EEG, thresholds):
    """Flag independent components based on ICLabel classification probabilities.

    Parameters
    ----------
    EEG : dict
        EEG structure with ICLabel classifications in EEG['etc']['ic_classification']['ICLabel']['classifications']
    thresholds : array-like, shape (7, 2)
        Threshold matrix where each row corresponds to an IC class:
        [Brain, Muscle, Eye, Heart, Line Noise, Channel Noise, Other]
        Each row contains [min_threshold, max_threshold].
        Use NaN in either column to ignore a class, matching EEGLAB's blank
        threshold fields.

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
    try:
        ic_class = EEG["etc"]["ic_classification"]["ICLabel"]["classifications"]
    except KeyError as exc:
        raise ValueError("EEG structure does not contain ICLabel classifications") from exc
    n_comps = ic_class.shape[0]

    thresholds = np.asarray(thresholds, dtype=float)

    if thresholds.shape != (7, 2):
        raise ValueError("Thresholds must be a 7x2 array")

    reject = np.zeros(n_comps, dtype=bool)

    for class_idx in range(7):
        min_thresh = thresholds[class_idx, 0]
        max_thresh = thresholds[class_idx, 1]
        if np.isnan(min_thresh) or np.isnan(max_thresh):
            continue
        probs = ic_class[:, class_idx]
        reject |= (probs > min_thresh) & (probs < max_thresh)

    EEG["reject"] = dict(EEG.get("reject") or {})
    EEG["reject"]["gcompreject"] = reject.astype(int)

    return EEG
