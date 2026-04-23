"""
Remove independent components from EEG data.

This is a Python equivalent of EEGLAB's pop_subcomp function.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def pop_subcomp(EEG, components=None, recompute=True):
    """
    Remove independent components from EEG data.

    Parameters
    ----------
    EEG : dict
        EEG structure with ICA decomposition
    components : array-like, optional
        Component indices to remove (0-based). If None, uses components flagged
        in EEG['reject']['gcompreject']
    recompute : bool, default=True
        If True (default), reconstruct data after component removal.
        If False, only update the ICA matrices (for MATLAB compatibility, but not used here).

    Returns
    -------
    EEG : dict
        EEG structure with components removed

    Notes
    -----
    This function removes components by:
    1. Identifying components to remove (from flags or explicit list)
    2. Zeroing out those components in the ICA mixing matrix
    3. Reconstructing the EEG data without those components
    4. Updating the ICA matrices to reflect the removal
    """
    # Check if ICA exists
    if 'icaweights' not in EEG or EEG['icaweights'] is None or EEG['icaweights'].size == 0:
        logger.warning("No ICA decomposition found. Skipping component removal.")
        return EEG

    if 'icasphere' not in EEG or EEG['icasphere'] is None or EEG['icasphere'].size == 0:
        logger.warning("No ICA sphere matrix found. Skipping component removal.")
        return EEG

    # Determine which components to remove
    if components is None:
        # Use flagged components from EEG.reject.gcompreject
        if 'reject' in EEG and 'gcompreject' in EEG['reject']:
            reject_flags = np.array(EEG['reject']['gcompreject'], dtype=bool)
            components = np.where(reject_flags)[0]
        else:
            logger.warning("No components specified and no reject flags found. No components removed.")
            return EEG
    else:
        components = np.array(components, dtype=int)

    if len(components) == 0:
        logger.info("No components to remove.")
        return EEG

    n_comps = EEG['icaweights'].shape[0]
    logger.info(f"Removing {len(components)} component(s) from {n_comps} total components.")

    # Create keep mask
    keep_comps = np.ones(n_comps, dtype=bool)
    keep_comps[components] = False

    # Reconstruct data without bad components
    W = EEG['icaweights'] @ EEG['icasphere']
    A = np.linalg.pinv(W)

    # Zero out bad components in mixing matrix
    A_clean = A.copy()
    A_clean[:, ~keep_comps] = 0

    # Reconstruct data
    original_shape = EEG['data'].shape
    data_2d = EEG['data'].reshape(EEG['nbchan'], -1)
    data_clean = A_clean @ W @ data_2d
    EEG['data'] = data_clean.reshape(original_shape)

    # Update ICA matrices to remove the components
    # Keep only the non-removed components
    EEG['icaweights'] = EEG['icaweights'][keep_comps, :]
    EEG['icawinv'] = EEG['icawinv'][:, keep_comps]

    if 'icaact' in EEG and EEG['icaact'] is not None and EEG['icaact'].size > 0:
        original_icaact_shape = EEG['icaact'].shape
        if len(original_icaact_shape) == 2:
            EEG['icaact'] = EEG['icaact'][keep_comps, :]
        elif len(original_icaact_shape) == 3:
            EEG['icaact'] = EEG['icaact'][keep_comps, :, :]

    # Update icachansind if present
    if 'icachansind' in EEG and EEG['icachansind'] is not None:
        # icachansind stays the same (it indicates which channels were used for ICA)
        pass

    # Clear reject flags since components have been removed
    if 'reject' in EEG and 'gcompreject' in EEG['reject']:
        del EEG['reject']['gcompreject']

    # Update etc.ic_classification if it exists
    if 'etc' in EEG and 'ic_classification' in EEG['etc']:
        if 'ICLabel' in EEG['etc']['ic_classification']:
            if 'classifications' in EEG['etc']['ic_classification']['ICLabel']:
                ic_class = EEG['etc']['ic_classification']['ICLabel']['classifications']
                EEG['etc']['ic_classification']['ICLabel']['classifications'] = ic_class[keep_comps, :]

    return EEG
