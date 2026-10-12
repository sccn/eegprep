"""ICLabel classification of independent components in EEG data."""

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
    """Classify independent components with ICLabel.

    With ``engine=None`` this runs the packaged int8 ICLabel network
    (``iclabel.onnx``) through ONNX Runtime, which needs the ``iclabel`` extra
    (``pip install "eegprep[iclabel]"``). ``engine='matlab'`` or ``'octave'``
    instead calls ICLabel in an EEGLAB installation through that engine.

    Args:
        EEG: EEG dataset with an ICA decomposition. It is not modified.
        algorithm: ICLabel network: ``'default'``, ``'lite'``, or ``'beta'``
            (case-insensitive; ``None`` means ``'default'``). Only
            ``'default'`` ships with EEGPrep; ``'lite'`` and ``'beta'`` need a
            MATLAB or Octave engine.
        engine: ``None`` for the packaged network, or ``'matlab'`` or
            ``'octave'``.

    Returns:
        dict: A copy of ``EEG`` whose ``etc.ic_classification.ICLabel`` holds
        ``classes``, ``classifications`` (components x 7 probabilities), and
        ``version``.

    Raises:
        RuntimeError: Under Emscripten (Pyodide), where ICLabel runs only
            through :func:`iclabel_async`.
        ValueError: If ``engine`` or ``algorithm`` is not supported.
        NotImplementedError: If ``algorithm`` is ``'lite'`` or ``'beta'`` with
            ``engine=None``.
        ImportError: If ``engine=None`` and onnxruntime is not installed.
    """
    require_sync_backend(engine)
    algorithm = _validate_request(algorithm, engine)
    EEG = deepcopy(EEG)
    if engine is not None:
        return _run_eeglab_engine(EEG, algorithm, engine)
    return _attach_classification(EEG, run_iclabel_net(*_prepare_features(EEG)), algorithm)


async def iclabel_async(EEG, algorithm='default', engine=None):
    """Classify independent components with ICLabel from asynchronous code.

    This is the ICLabel entry point under Emscripten (Pyodide): the packaged
    network runs in the host's ONNX Runtime Web session, and this coroutine
    awaits its Promise. The backend is chosen when
    :mod:`eegprep.plugins.ICLabel.iclabel_net_onnx` is first imported. On
    native platforms it returns exactly what :func:`iclabel` returns, but it
    runs synchronously inside the coroutine and does not yield to the event
    loop; ``engine='matlab'`` or ``'octave'`` blocks the loop for the whole
    engine call.

    Args:
        EEG: EEG dataset with an ICA decomposition. It is not modified.
        algorithm: ICLabel network: ``'default'``, ``'lite'``, or ``'beta'``.
            Only ``'default'`` ships with EEGPrep.
        engine: ``None`` for the packaged network, or ``'matlab'`` or
            ``'octave'`` on native platforms.

    Returns:
        dict: A copy of ``EEG`` with ``etc.ic_classification.ICLabel`` filled
        in, as for :func:`iclabel`.

    Raises:
        RuntimeError: For ``engine='matlab'`` or ``'octave'`` under Emscripten.
        ValueError: If ``engine`` or ``algorithm`` is not supported.
        NotImplementedError: If ``algorithm`` is ``'lite'`` or ``'beta'`` with
            ``engine=None``.
        ImportError: On native platforms, if ``engine=None`` and onnxruntime
            is not installed.
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
