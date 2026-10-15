import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import option_scaleicarms
from eegprep.functions.miscfunc.misc import finite_matmul, finite_pinv
from eegprep.functions.miscfunc.pinv import pinv


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
    EEG['icawinv'] = finite_pinv(finite_matmul(EEG['icaweights'], EEG['icasphere']), solver=pinv)

    # Optionally sort components by mean descending activation variance
    if sortcomps in ('on', True):
        icaact_2d = flatten_ica_data(EEG['icaact'])
        # Compute variance metric: sum(icawinv^2) .* sum(icaact^2)
        variance_metric = np.sum(EEG['icawinv'] ** 2, axis=0) * np.sum(icaact_2d**2, axis=1)
        windex = np.argsort(variance_metric)[::-1]
        # Reorder components
        EEG['icaact'] = EEG['icaact'][windex, :, :]
        EEG['icaweights'] = EEG['icaweights'][windex, :]
        EEG['icawinv'] = EEG['icawinv'][:, windex]

    # Optionally normalize components using the same rule as runica()
    if posact in ('on', True):
        icaact_2d = flatten_ica_data(EEG['icaact'])
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

    EEG['icawinv'] = finite_pinv(finite_matmul(EEG['icaweights'], EEG['icasphere']), solver=pinv)
    return EEG


def scale_ica_to_rms_microvolt(EEG):
    """Scale each component so its mixing column has RMS 1.

    EEGLAB applies this in ``eeg_checkset`` when ``option_scaleicarms`` is on,
    including when a dataset is saved. ICLabel reads the scaled activations
    and topography, so the decomposition is scaled before it is returned.
    The pre-scale weights and sphere are kept in ``EEG['etc']``.
    """
    if not option_scaleicarms:
        return EEG
    weights = np.asarray(EEG.get("icaweights", []))
    sphere = np.asarray(EEG.get("icasphere", []))
    mixing = np.asarray(EEG.get("icawinv", []))
    if weights.ndim != 2 or sphere.ndim != 2 or mixing.ndim != 2:
        return EEG
    if weights.size == 0 or sphere.size == 0 or mixing.size == 0:
        return EEG
    if mixing.shape[1] != weights.shape[0]:
        return EEG

    scaling = np.sqrt(np.mean(mixing.astype(float) ** 2, axis=0))
    etc = EEG.get("etc")
    if not isinstance(etc, dict):
        etc = {}
        EEG["etc"] = etc
    etc["icaweights_beforerms"] = np.array(weights, copy=True)
    etc["icasphere_beforerms"] = np.array(sphere, copy=True)

    EEG["icaweights"] = weights.astype(float) * scaling[:, np.newaxis]
    EEG["icawinv"] = finite_pinv(finite_matmul(EEG["icaweights"], sphere), solver=pinv)
    if "data" not in EEG or "pnts" not in EEG or "trials" not in EEG:
        return EEG
    data = flatten_ica_data(np.asarray(EEG["data"], dtype=float))
    channel_index = np.asarray(EEG.get("icachansind", []))
    if channel_index.size:
        data = data[channel_index.astype(int)]
    activations = finite_matmul(finite_matmul(EEG["icaweights"], sphere), data)
    EEG["icaact"] = reshape_ica_activations(activations, EEG["pnts"], EEG["trials"])
    return EEG
