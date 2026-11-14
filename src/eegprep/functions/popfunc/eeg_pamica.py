"""Perform ICA decomposition using pAMICA, the PyTorch implementation of AMICA."""

import copy
from contextlib import contextmanager

import numpy as np
from threadpoolctl import threadpool_limits

from eegprep.extras import install_command
from ..miscfunc.misc import finite_matmul
from ..sigprocfunc.runamica import _AMICA_DEFAULTS
from ._ica_utils import finalize_ica_fields, flatten_ica_data, reshape_ica_activations

#: eegprep ``runamica15`` parameter, as ``_AMICA_DEFAULTS`` spells it, to pAMICA's name for it.
_PAMICA_NAMES = {
    'lrate': 'lrate',
    'minlrate': 'minlrate',
    'rho0': 'rho0',
    'do_newton': 'do_newton',
    'newt_start': 'newt_start',
    'use_min_dll': 'use_min_dll',
    'use_grad_norm': 'use_grad_norm',
    'do_mean': 'do_mean',
    'do_sphere': 'do_sphere',
    'max_decs': 'maxdecs',
    'newtrate': 'newtrate',
    'rholratefact': 'rholratefact',
    'invsigmin': 'invsigmin',
    'invsigmax': 'invsigmax',
}
_PAMICA_FLAGS = {'do_newton', 'use_min_dll', 'use_grad_norm', 'do_mean', 'do_sphere'}
_EEGLAB_DEFAULTS = {
    _PAMICA_NAMES[name]: bool(_AMICA_DEFAULTS[name]) if _PAMICA_NAMES[name] in _PAMICA_FLAGS else _AMICA_DEFAULTS[name]
    for name in _PAMICA_NAMES
}
#: pAMICA's ``fit`` takes these as named arguments; every other setting is a backend keyword.
_FIT_ARGUMENTS = ('lrate', 'do_mean', 'do_sphere', 'do_newton')


class PamicaFitError(RuntimeError):
    """pAMICA stopped on a degenerate reason, so the model holds non-finite parameters."""


def eeg_pamica(
    EEG,
    *,
    posact='off',
    sortcomps='off',
    n_models=1,
    n_mix=3,
    max_iter=2000,
    block_size=131072,
    seed=5489,
    device='cpu',
    threads=1,
    pcakeep=None,
    **amica_options,
):
    """Perform ICA decomposition using pAMICA's PyTorch engine, in float64.

    pAMICA is a Python implementation of AMICA (Adaptive Mixture ICA). Install it with
    ``pip install "eegprep[pamica]"``. Standard ICA fields are populated from model 0, and the
    run is recorded in ``EEG['etc']['pamica']``.

    A setting not passed takes eegprep's ``runamica15`` default, as ``runamica._AMICA_DEFAULTS``
    states it, for ``lrate``, ``minlrate``, ``rho0``, ``do_newton``, ``newt_start``, ``use_min_dll``,
    ``use_grad_norm``, ``do_mean``, ``do_sphere``, ``newtrate``, ``rholratefact``, ``invsigmin`` and
    ``invsigmax``. These follow eegprep's own table, which differs from EEGLAB's ``runamica15.m`` in
    places. eegprep's ``max_decs`` is pAMICA's ``maxdecs``. pAMICA's own value stands for
    every other setting. ``block_size`` is pAMICA's 131072: the 128 of ``runamica15`` is the Fortran
    program's block per thread and does not mean the same thing.

    ``icaweights @ icasphere`` is the unmixing pAMICA applies, from the channels to the components.
    pAMICA subtracts the channel mean before sphering and EEGLAB's activations do not, so
    ``icaact`` differs from pAMICA's ``transform`` by one constant per component. ``icasphere`` is
    the identity when ``do_sphere=False``. Components stay in pAMICA's order unless ``sortcomps``
    asks for descending activation variance.

    Parameters
    ----------
    EEG : dict
        EEGLAB-like data structure.
    posact : str | bool, optional
        If 'on' or True, normalize component signs so max(abs(activations)) is positive.
    sortcomps : str | bool, optional
        If 'on' or True, sort components by descending activation variance.
    n_models : int
        Number of ICA models to fit.
    n_mix : int
        Number of mixture components per source.
    max_iter : int
        Maximum number of training iterations.
    block_size : int
        Samples per block of pAMICA's iterations. Memory grows with it.
    seed : int | None
        Seed of pAMICA's initialization. The same seed, data and settings give identical
        weights in float64 on the CPU with one thread.
    device : str
        Torch device. Only ``'cpu'`` is covered by the reproducibility statement above.
    threads : int
        Threads for torch and the BLAS libraries during the run, restored afterwards. A larger
        count is reproducible only on the host where it was measured.
    pcakeep : int | None
        Number of principal components to keep. Default: every channel pAMICA finds full rank.
        pAMICA ignores it when ``do_sphere=False``. The record keeps the value requested in
        ``pcakeep`` and the number of components that ran in ``rank``.
    amica_options : dict
        Further pAMICA settings, which replace the defaults above, for example ``lrate=0.1``.

    Returns
    -------
    dict
        The updated EEG structure with ICA fields.

    Raises
    ------
    ImportError
        If ``pamica`` or ``torch`` is not installed.
    PamicaFitError
        If pAMICA stops on a degenerate reason. No ICA field is set.
    """
    pamica, torch, degenerate_stops = _import_pamica()
    EEG = copy.deepcopy(EEG)
    data = flatten_ica_data(np.asarray(EEG['data'], dtype=np.float64))

    settings = {**_EEGLAB_DEFAULTS, **amica_options}
    fit_arguments = {name: settings[name] for name in _FIT_ARGUMENTS}
    backend_options = {name: value for name, value in settings.items() if name not in _FIT_ARGUMENTS}
    with _limited_threads(torch, threads):
        model = pamica.AMICA(n_models=n_models, n_mix=n_mix, device=device, verbose=False)
        model.fit(
            data,
            max_iter=max_iter,
            dtype=torch.float64,
            block_size=block_size,
            seed=seed,
            pcakeep=pcakeep,
            **fit_arguments,
            **backend_options,
        )
        threads_used = torch.get_num_threads()

    stop_reason = model.stop_reason_
    history = np.asarray(model.ll_history_, dtype=np.float64)
    if stop_reason in degenerate_stops:
        raise PamicaFitError(
            f"pAMICA stopped on {stop_reason!r} after {len(history)} iterations and holds non-finite "
            "parameters. Lower lrate, disable do_newton, or check the data, then run it again."
        )

    EEG['icaweights'] = model.get_unmixing_matrix()
    EEG['icasphere'] = model.get_sphere()
    unmixing = finite_matmul(EEG['icaweights'], EEG['icasphere'])
    EEG['icaact'] = reshape_ica_activations(finite_matmul(unmixing, data), EEG['pnts'], EEG['trials'])
    EEG['icachansind'] = np.arange(EEG['nbchan'])

    best_iter = int(np.argmax(history))
    if not isinstance(EEG.get('etc'), dict):
        EEG['etc'] = {}
    EEG['etc']['pamica'] = {
        'version': pamica.__version__,
        'torch': torch.__version__,
        'device': str(model.model_.device),
        'dtype': 'float64',
        'seed': seed,
        'n_models': n_models,
        'n_mix': n_mix,
        'max_iter': max_iter,
        'block_size': block_size,
        'threads': threads_used,
        'pcakeep': pcakeep,
        'rank': EEG['icaweights'].shape[0],
        **settings,
        'iterations': len(history),
        'stop_reason': stop_reason,
        'did_converge': stop_reason != 'max_iter',
        'final_ll': float(model.final_ll_),
        'best_ll': float(history[best_iter]),
        'best_iter': best_iter,
    }

    return finalize_ica_fields(EEG, sortcomps=sortcomps, posact=posact)


def _import_pamica():
    """Import pAMICA and torch, and return pAMICA's degenerate stop reasons with them."""
    try:
        import pamica
        import torch
        from pamica.torch_impl import AMICATorchNG
    except ImportError as exc:
        missing = (exc.name or '').split('.', 1)[0]
        if missing not in ('pamica', 'torch'):
            raise
        raise ImportError(
            f"eeg_pamica needs {missing}, and it is not installed. "
            f"Install it with {install_command('pamica')}, which installs pamica and torch."
        ) from exc
    # pAMICA names its degenerate stops in this class attribute. Reading it keeps the list from drifting.
    return pamica, torch, AMICATorchNG._DEGENERATE_STOP_REASONS


@contextmanager
def _limited_threads(torch, threads):
    """Limit torch and the BLAS libraries to ``threads`` threads, and restore both afterwards."""
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        with threadpool_limits(limits=threads):
            yield
    finally:
        torch.set_num_threads(previous)
