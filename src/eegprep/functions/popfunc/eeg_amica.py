"""Perform ICA decomposition using the AMICA (Adaptive Mixture ICA) algorithm."""

import copy

import numpy as np
from ._ica_utils import finalize_ica_fields, flatten_ica_data, reshape_ica_activations
from ..sigprocfunc.runamica import runamica


def eeg_amica(
    EEG,
    posact='off',
    sortcomps='off',
    num_models=1,
    max_iter=2000,
    num_mix_comps=3,
    pcakeep=None,
    outdir=None,
    amica_binary=None,
    max_threads=4,
    **kwargs,
):
    """
    Perform ICA decomposition using AMICA (Adaptive Mixture ICA).

    AMICA fits one or more ICA models with generalized Gaussian mixture source
    distributions via an external Fortran binary. Standard ICA fields are
    populated from model 0. The full multi-model output is stored in
    EEG['etc']['amica'].

    EEGPrep wheels do not ship AMICA binaries. Provide the executable with the
    amica_binary argument, the AMICA_BINARY environment variable, or PATH.

    Parameters
    ----------
    EEG : dict
        EEGLAB-like data structure.
    posact : str | bool, optional
        If 'on' or True, normalize component signs so max(abs(activations))
        is positive. Default is 'off'.
    sortcomps : str | bool, optional
        If 'on' or True, sort components by descending activation variance.
        Default is 'off'.
    num_models : int
        Number of ICA models to fit.
    max_iter : int
        Maximum number of training iterations.
    num_mix_comps : int
        Number of mixture components per source.
    pcakeep : int or None
        Number of principal components to retain. Default: nbchan.
    outdir : str or None
        Output directory for AMICA. If None, a temporary directory is used.
    amica_binary : str or None
        Path to AMICA binary. If None, auto-detected from AMICA_BINARY, a
        source-tree development binary, EEGLAB's AMICA plugin, or PATH.
    max_threads : int
        Maximum number of threads for the binary.
    kwargs : dict
        Additional AMICA parameters passed through to the binary.

    Returns
    -------
    dict
        The updated EEG structure with ICA fields.
    """
    EEG = copy.deepcopy(EEG)

    # Extract data and reshape from 3D to 2D
    data = flatten_ica_data(EEG['data'].astype('float64'))

    # Run AMICA
    weights, sphere, mods = runamica(
        data,
        num_models=num_models,
        max_iter=max_iter,
        num_mix_comps=num_mix_comps,
        pcakeep=pcakeep,
        outdir=outdir,
        amica_binary=amica_binary,
        max_threads=max_threads,
        **kwargs,
    )

    # Store standard ICA fields from model 0
    num_pcs = mods['num_pcs']
    EEG['icaweights'] = mods['W'][:, :, 0]
    EEG['icasphere'] = mods['S'][:num_pcs, :]
    EEG['icawinv'] = mods['A'][:, :, 0]

    # Compute ICA activations
    EEG['icaact'] = (EEG['icaweights'] @ EEG['icasphere']) @ data
    # Reshape icaact back to 3D
    EEG['icaact'] = reshape_ica_activations(EEG['icaact'], EEG['pnts'], EEG['trials'])
    EEG['icachansind'] = np.arange(EEG['nbchan'])

    # Store full multi-model results
    if 'etc' not in EEG:
        EEG['etc'] = {}
    EEG['etc']['amica'] = mods

    return finalize_ica_fields(EEG, sortcomps=sortcomps, posact=posact)


def load_amica_model(EEG, mods, model_num=0):
    """Switch the active ICA model in an EEG structure.

    When AMICA is run with num_models > 1, this function loads a specific
    model's weights into the standard ICA fields (icaweights, icasphere,
    icawinv, icaact).

    Parameters
    ----------
    EEG : dict
        EEGLAB-like data structure with existing ICA fields.
    mods : dict
        Full AMICA model output (from EEG['etc']['amica'] or runamica()).
    model_num : int
        Model index to load (0-based). Default is 0.

    Returns
    -------
    dict
        The updated EEG structure with ICA fields from the specified model.
    """
    num_pcs = mods['num_pcs']
    num_models = mods['num_models']

    if model_num < 0 or model_num >= num_models:
        raise ValueError(f"model_num={model_num} out of range for {num_models} models")

    EEG = copy.deepcopy(EEG)

    EEG['icaweights'] = mods['W'][:, :, model_num]
    EEG['icasphere'] = mods['S'][:num_pcs, :]
    EEG['icawinv'] = mods['A'][:, :, model_num]

    # Recompute activations
    data = flatten_ica_data(EEG['data'].astype('float64'))
    EEG['icaact'] = (EEG['icaweights'] @ EEG['icasphere']) @ data
    EEG['icaact'] = reshape_ica_activations(EEG['icaact'], EEG['pnts'], EEG['trials'])

    return EEG
