"""ICLabel module for classifying independent components in EEG data."""

from copy import deepcopy

import numpy as np

from eegprep.extras import missing_onnxruntime_message
from eegprep.plugins.ICLabel._prop_numerics import require_ica_weights
from eegprep.plugins.ICLabel.iclabel_net_onnx import run_iclabel_net


_SUPPORTED_ALGORITHMS = ('default', 'lite', 'beta')


def iclabel(EEG, algorithm='default', engine=None, *, feature='iclabel'):
    """Apply ICLabel to classify independent components.

    Parameters
    ----------
    EEG : dict
        EEGLAB EEG structure
    algorithm : str
        Algorithm to use for classification, passed to the MATLAB/Octave implementation.
        Default is 'default'.
    engine : str or None
        Engine to use for implementation. Options are:
        - None: Use the packaged float32 ONNX network (needs ``eegprep[iclabel]``)
        - 'matlab': Use MATLAB engine
        - 'octave': Use Octave engine

    Returns
    -------
    EEG : dict
        EEGLAB EEG structure with ICLabel classifications added
    """
    algorithm = _normalize_algorithm(algorithm)
    EEG = deepcopy(EEG)

    # Check if using MATLAB or Octave implementation
    if engine in ['matlab', 'octave']:
        from eegprep.functions.adminfunc.eeglabcompat import get_eeglab

        runtime = 'MAT' if engine == 'matlab' else 'OCT'
        eeglab = get_eeglab(runtime=runtime, feature=feature)

        # Run ICLabel using MATLAB/Octave, passing the algorithm parameter
        if algorithm == 'default':
            return eeglab.iclabel(EEG)
        else:
            return eeglab.iclabel(EEG, algorithm)

    # Default Python implementation
    elif engine is None:
        if algorithm != 'default':
            raise NotImplementedError(
                "EEGPrep standalone Python ICLabel only ships the default network (iclabel.onnx). "
                f"The '{algorithm}' network is available only with engine='matlab' or engine='octave' "
                "and an EEGLAB ICLabel checkout that provides that artifact."
            )
        try:
            import onnxruntime  # noqa: F401
        except ImportError as e:
            raise ImportError(missing_onnxruntime_message(feature)) from e
        require_ica_weights(EEG, feature)

        from eegprep import ICL_feature_extractor

        image, psdmed, autocorr = prepare_network_inputs(ICL_feature_extractor(EEG, True))
        output_np = postprocess_network_output(run_iclabel_net(image, psdmed, autocorr))

        if 'ic_classification' not in EEG['etc']:
            EEG['etc']['ic_classification'] = {}
        if 'ICLabel' not in EEG['etc']['ic_classification']:
            EEG['etc']['ic_classification']['ICLabel'] = {}

        EEG['etc']['ic_classification']['ICLabel']['classes'] = np.array(
            ['Brain', 'Muscle', 'Eye', 'Heart', 'Line Noise', 'Channel Noise', 'Other'], dtype=object
        )
        EEG['etc']['ic_classification']['ICLabel']['classifications'] = output_np
        EEG['etc']['ic_classification']['ICLabel']['version'] = algorithm

        return EEG
    else:
        raise ValueError(f"Unsupported engine: {engine}. Should be None, 'matlab', or 'octave'")


def prepare_network_inputs(features):
    """Augment ICLabel features four ways and convert them to network inputs.

    Args:
        features: ``(topo, psd, autocorr)`` from ``ICL_feature_extractor`` with
            shapes ``(32, 32, 1, n)``, ``(1, 100, 1, n)``, and ``(1, 100, 1, n)``.

    Returns:
        tuple: float32 NCHW ``image`` ``(4n, 1, 32, 32)``, ``psdmed``
        ``(4n, 1, 1, 100)``, and ``autocorr`` ``(4n, 1, 1, 100)``. Batch row
        ``k * n + j`` holds augmentation ``k`` (original, negated, flipped,
        negated and flipped topography) of component ``j``.
    """
    topo, psdmed, autocorr = features
    topo = np.single(np.concatenate([topo, -topo, topo[:, ::-1, :, :], -topo[:, ::-1, :, :]], axis=3))
    psdmed = np.single(np.tile(psdmed, (1, 1, 1, 4)))
    autocorr = np.single(np.tile(autocorr, (1, 1, 1, 4)))
    return (
        np.transpose(topo, (3, 2, 0, 1)),
        np.transpose(psdmed, (3, 2, 0, 1)),
        np.transpose(autocorr, (3, 2, 0, 1)),
    )


def postprocess_network_output(output_np):
    """Average the four augmented network outputs of each component.

    Args:
        output_np: Network output of shape ``(4n, 7, 1, 1)`` in the batch order
            of :func:`prepare_network_inputs`.

    Returns:
        numpy.ndarray: ``(n, 7)`` class probabilities.
    """
    output_np = output_np.T
    output_np = np.reshape(output_np, (-1, 4), order='F')
    output_np = np.mean(output_np, axis=1)
    return np.reshape(output_np, (7, -1), order='F').T


def _normalize_algorithm(algorithm):
    normalized = 'default' if algorithm is None else str(algorithm).lower()
    if normalized not in _SUPPORTED_ALGORITHMS:
        raise ValueError("algorithm must be one of 'default', 'lite', or 'beta'")
    return normalized
