"""ICLabel module for classifying independent components in EEG data."""

from copy import deepcopy

import numpy as np

from eegprep.plugins.ICLabel.iclabel_net_onnx import IS_EMSCRIPTEN, run_iclabel_net, run_iclabel_net_async


_SUPPORTED_ALGORITHMS = ('default', 'lite', 'beta')
_SUPPORTED_ENGINES = (None, 'matlab', 'octave')
_SYNC_UNAVAILABLE_MESSAGE = (
    'ICLabel synchronous entry points are unavailable under Emscripten; '
    'use await iclabel_async(...) or await pop_iclabel_async(...).'
)
_ENGINE_UNAVAILABLE_MESSAGE = (
    'MATLAB and Octave engines are unavailable under Emscripten; use engine=None, '
    'which runs the packaged ICLabel network in the browser.'
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
    require_sync_backend(engine)
    algorithm = _validate_request(algorithm, engine)
    EEG = deepcopy(EEG)
    if engine is not None:
        return _run_eeglab_engine(EEG, algorithm, engine)
    return _attach_classification(EEG, run_iclabel_net(*_prepare_features(EEG)), algorithm)


async def iclabel_async(EEG, algorithm='default', engine=None):
    """Apply ICLabel through an awaitable native or browser backend.

    The browser backend is selected once when the ICLabel module is loaded and
    awaits the ONNX Runtime Web Promise. Native callers may use this entry
    point as an asynchronous spelling of the same pure processing operation.
    """
    algorithm = _validate_request(algorithm, engine)
    EEG = deepcopy(EEG)
    if engine is not None:
        return _run_eeglab_engine(EEG, algorithm, engine)
    return _attach_classification(EEG, await run_iclabel_net_async(*_prepare_features(EEG)), algorithm)


def require_sync_backend(engine=None):
    """Raise ``RuntimeError`` when ICLabel cannot run synchronously with ``engine`` here.

    Under Emscripten (Pyodide) the packaged network runs only through the
    asynchronous ONNX Runtime Web backend, and the MATLAB and Octave engines
    do not exist. Native platforms always pass.

    Args:
        engine: ``None`` for the packaged network, or ``'matlab'`` or ``'octave'``.

    Raises:
        RuntimeError: Under Emscripten. For ``engine=None`` the message points
            to the asynchronous entry points; for MATLAB or Octave it says to
            use ``engine=None``.
    """
    if IS_EMSCRIPTEN:
        raise RuntimeError(_SYNC_UNAVAILABLE_MESSAGE if engine is None else _ENGINE_UNAVAILABLE_MESSAGE)


def _validate_request(algorithm, engine):
    """Return the normalized algorithm after rejecting unsupported engine and algorithm pairs."""
    if engine not in _SUPPORTED_ENGINES:
        raise ValueError(f"Unsupported engine: {engine}. Should be None, 'matlab', or 'octave'")
    if engine is not None and IS_EMSCRIPTEN:
        raise RuntimeError(_ENGINE_UNAVAILABLE_MESSAGE)
    algorithm = _normalize_algorithm(algorithm)
    if engine is None and algorithm != 'default':
        raise NotImplementedError(
            "EEGPrep standalone Python ICLabel only ships the default network (iclabel.onnx). "
            f"The '{algorithm}' network is available only with engine='matlab' or engine='octave' "
            "and an EEGLAB ICLabel checkout that provides that artifact."
        )
    return algorithm


def _run_eeglab_engine(EEG, algorithm, engine):
    from eegprep.functions.adminfunc.eeglabcompat import get_eeglab

    eeglab = get_eeglab(runtime='MAT' if engine == 'matlab' else 'OCT')
    if algorithm == 'default':
        return eeglab.iclabel(EEG)
    return eeglab.iclabel(EEG, algorithm)


def _prepare_features(EEG):
    from eegprep import ICL_feature_extractor

    return prepare_network_inputs(ICL_feature_extractor(EEG, True))


def prepare_network_inputs(features):
    """Apply ICLabel augmentation and convert feature arrays to network inputs."""
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
    """Average the four augmented network outputs into component probabilities."""
    output_np = output_np.T
    output_np = np.reshape(output_np, (-1, 4), order='F')
    output_np = np.mean(output_np, axis=1)
    return np.reshape(output_np, (7, -1), order='F').T


def _attach_classification(EEG, output_np, algorithm):
    output_np = postprocess_network_output(output_np)

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
