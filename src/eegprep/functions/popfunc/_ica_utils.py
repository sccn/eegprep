import numpy as np


def flatten_ica_data(data):
    """Flatten channel-major EEG data using EEGLAB/MATLAB epoch ordering."""
    array = np.asarray(data)
    return array.reshape(array.shape[0], -1, order="F")


def reshape_ica_activations(data, pnts, trials):
    """Reshape 2-D ICA activations back to EEGLAB's channel x point x trial form."""
    array = np.asarray(data)
    return array.reshape(array.shape[0], int(pnts), int(trials), order="F")


def finalize_ica_fields(EEG, *, sortcomps='off', posact='off'):
    """Apply optional component sorting and sign normalization to ICA fields.

    Operates in place on ``EEG['icaact']``, ``EEG['icaweights']``, and
    ``EEG['icawinv']`` and returns ``EEG``. Shared by the runica, AMICA, and
    Picard backends so the post-decomposition behavior stays identical.
    """
    # Optionally sort components by mean descending activation variance
    if sortcomps in ('on', True):
        # Flatten icaact to 2D for variance computation
        icaact_2d = flatten_ica_data(EEG['icaact'])
        # Compute variance metric: sum(icawinv^2) .* sum(icaact^2)
        variance_metric = np.sum(EEG['icawinv'] ** 2, axis=0) * np.sum(icaact_2d**2, axis=1)
        # Sort indices in descending order
        windex = np.argsort(variance_metric)[::-1]
        # Reorder components
        EEG['icaact'] = EEG['icaact'][windex, :, :]
        EEG['icaweights'] = EEG['icaweights'][windex, :]
        EEG['icawinv'] = EEG['icawinv'][:, windex]

    # Optionally normalize components using the same rule as runica()
    if posact in ('on', True):
        # Flatten icaact to 2D for finding max abs values
        icaact_2d = flatten_ica_data(EEG['icaact'])
        # Find indices of max absolute values for each component
        ix = np.argmax(np.abs(icaact_2d), axis=1)
        ncomps = EEG['icaact'].shape[0]

        for r in range(ncomps):
            if np.sign(icaact_2d[r, ix[r]]) < 0:
                # A sign flip commutes through the factorization, so negate the
                # matching row of icaweights and column of icawinv directly. This
                # preserves the invariants icawinv == pinv(icaweights @ icasphere)
                # and icaact == icaweights @ icasphere @ data, leaving icasphere
                # untouched.
                EEG['icaact'][r, :, :] = -EEG['icaact'][r, :, :]
                EEG['icawinv'][:, r] = -EEG['icawinv'][:, r]
                EEG['icaweights'][r, :] = -EEG['icaweights'][r, :]

    return EEG
