"""ICLabel module for classifying independent components in EEG data."""

from copy import deepcopy
import sys

import numpy as np


_SUPPORTED_ALGORITHMS = ('default', 'lite', 'beta')
_IS_EMSCRIPTEN = sys.platform == 'emscripten'
SYNC_UNAVAILABLE_MESSAGE = (
    'ICLabel synchronous entry points are unavailable under Emscripten; '
    'use await iclabel_async(...) or await pop_iclabel_async(...).'
)


def iclabel(EEG, algorithm='default', engine=None):
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
        - None: Use the default Python implementation
        - 'matlab': Use MATLAB engine
        - 'octave': Use Octave engine

    Returns
    -------
    EEG : dict
        EEGLAB EEG structure with ICLabel classifications added
    """
    if _IS_EMSCRIPTEN:
        raise RuntimeError(SYNC_UNAVAILABLE_MESSAGE)
    return _iclabel_sync(EEG, algorithm=algorithm, engine=engine)


async def iclabel_async(EEG, algorithm='default', engine=None):
    """Apply ICLabel through an awaitable native or browser backend.

    The browser backend is selected once when the ICLabel module is loaded and
    awaits the ONNX Runtime Web Promise. Native callers may use this entry
    point as an asynchronous spelling of the same pure processing operation.
    """
    algorithm = _normalize_algorithm(algorithm)
    EEG = deepcopy(EEG)

    if engine in ['matlab', 'octave']:
        if _IS_EMSCRIPTEN:
            raise RuntimeError(SYNC_UNAVAILABLE_MESSAGE)
        return _iclabel_sync(EEG, algorithm=algorithm, engine=engine)
    if engine is not None:
        raise ValueError(f"Unsupported engine: {engine}. Should be None, 'matlab', or 'octave'")
    if algorithm != 'default':
        raise NotImplementedError(
            "EEGPrep standalone Python ICLabel only ships the default network (iclabel.onnx). "
            f"The '{algorithm}' network is available only with engine='matlab' or engine='octave' "
            "and an EEGLAB ICLabel checkout that provides that artifact."
        )

    from eegprep.plugins.ICLabel.iclabel_net_onnx import run_iclabel_net_async

    image, psdmed, autocorr = _prepare_features(EEG)
    output_np = await run_iclabel_net_async(image, psdmed, autocorr)
    return _attach_classification(EEG, output_np, algorithm)


def _iclabel_sync(EEG, algorithm='default', engine=None):
    algorithm = _normalize_algorithm(algorithm)
    EEG = deepcopy(EEG)

    if engine in ['matlab', 'octave']:
        from eegprep.functions.adminfunc.eeglabcompat import get_eeglab

        runtime = 'MAT' if engine == 'matlab' else 'OCT'
        eeglab = get_eeglab(runtime=runtime)
        if algorithm == 'default':
            return eeglab.iclabel(EEG)
        return eeglab.iclabel(EEG, algorithm)
    if engine is not None:
        raise ValueError(f"Unsupported engine: {engine}. Should be None, 'matlab', or 'octave'")
    if algorithm != 'default':
        raise NotImplementedError(
            "EEGPrep standalone Python ICLabel only ships the default network (iclabel.onnx). "
            f"The '{algorithm}' network is available only with engine='matlab' or engine='octave' "
            "and an EEGLAB ICLabel checkout that provides that artifact."
        )

    from eegprep.plugins.ICLabel.iclabel_net_onnx import run_iclabel_net

    image, psdmed, autocorr = _prepare_features(EEG)
    output_np = run_iclabel_net(image, psdmed, autocorr)
    return _attach_classification(EEG, output_np, algorithm)


def _prepare_features(EEG):
    from eegprep import ICL_feature_extractor

    features = ICL_feature_extractor(EEG, True)
    features[0] = np.single(
        np.concatenate([features[0], -features[0], features[0][:, ::-1, :, :], -features[0][:, ::-1, :, :]], axis=3)
    )
    features[1] = np.single(np.tile(features[1], (1, 1, 1, 4)))
    features[2] = np.single(np.tile(features[2], (1, 1, 1, 4)))

    return (
        np.transpose(features[0], (3, 2, 0, 1)),
        np.transpose(features[1], (3, 2, 0, 1)),
        np.transpose(features[2], (3, 2, 0, 1)),
    )


def _attach_classification(EEG, output_np, algorithm):
    output_np = output_np.T
    output_np = np.reshape(output_np, (-1, 4), order='F')
    output_np = np.mean(output_np, axis=1)
    output_np = np.reshape(output_np, (7, -1), order='F')
    output_np = output_np.T

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


def _normalize_algorithm(algorithm):
    normalized = 'default' if algorithm is None else str(algorithm).lower()
    if normalized not in _SUPPORTED_ALGORITHMS:
        raise ValueError("algorithm must be one of 'default', 'lite', or 'beta'")
    return normalized
