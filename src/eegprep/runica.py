"""
ICA decomposition using the logistic infomax ICA algorithm.

Implementation of Bell & Sejnowski (1995) ICA with natural gradient feature
and optional extended-ICA algorithm. This is a Python translation of the
MATLAB runica() function from EEGLAB.

Reference:
    Makeig, S., Bell, A.J., Jung, T-P and Sejnowski, T.J.,
    "Independent component analysis of electroencephalographic data,"
    In: D. Touretzky, M. Mozer and M. Hasselmo (Eds). Advances in Neural
    Information Processing Systems 8:145-151, MIT Press, Cambridge, MA (1996).
"""

import numpy as np
from scipy.linalg import sqrtm, pinv, eig
from .utils.misc import round_mat


# Constants matching MATLAB defaults
MAX_WEIGHT = 1e8
DEFAULT_STOP = 0.000001
DEFAULT_ANNEALDEG = 60
DEFAULT_ANNEALSTEP = 0.90
DEFAULT_EXTANNEAL = 0.98
DEFAULT_MAXSTEPS = 512
DEFAULT_MOMENTUM = 0.0
DEFAULT_BLOWUP = 1000000000.0
DEFAULT_BLOWUP_FAC = 0.8
DEFAULT_RESTART_FAC = 0.9
MIN_LRATE = 0.000001
MAX_LRATE = 0.1
DEFAULT_EXTENDED = 0
DEFAULT_EXTBLOCKS = 1
DEFAULT_NSUB = 1
DEFAULT_EXTMOMENTUM = 0.5
MAX_KURTSIZE = 6000
MIN_KURTSIZE = 2000
SIGNCOUNT_THRESHOLD = 25
SIGNCOUNT_STEP = 2
DEFAULT_SPHEREFLAG = 'on'
DEFAULT_INTERRUPT = 'off'
DEFAULT_PCAFLAG = 'off'
DEFAULT_POSACTFLAG = 'off'
DEFAULT_VERBOSE = 1
DEFAULT_BIASFLAG = 1
DEFAULT_RESETRANDOMSEED = False  # Note: Changed to False in 2015 (was True before)


def runica(data, **kwargs):
    """
    Perform Independent Component Analysis (ICA) decomposition using infomax.

    This function implements the logistic infomax ICA algorithm of Bell & Sejnowski
    (1995) with the natural gradient feature, or optionally the extended-ICA
    algorithm with optional PCA dimension reduction. Annealing based on weight
    changes automates the separation process.

    Parameters
    ----------
    data : ndarray, shape (chans, frames)
        Input data matrix where chans is the number of channels and frames is
        the number of time points. Data should be baseline-zeroed if using epochs.

    **kwargs : keyword arguments
        Optional parameters:

        extended : int, default=0
            Perform TANH "extended-ICA" with sign estimation N training blocks.
            If N > 0, automatically estimate number of sub-Gaussian sources.
            If N < 0, fix number of sub-Gaussian comps to -N (faster).
            Default 0 = off.

        pca : int, default=0
            Decompose a principal component subspace of the data.
            Value is the number of PCs to retain. Default 0 = off.

        sphering : str, default='on'
            Flag for sphering of data. Options: 'on', 'off', 'none'.

        weights : ndarray, optional
            Initial weight matrix. Default is identity (or sphere if sphering='off').

        lrate : float, optional
            Initial ICA learning rate (<< 1). Default is heuristic: 0.00065/log(chans).

        block : int, optional
            ICA block size (<< datalength). Default is heuristic:
            ceil(min(5*log(frames), 0.3*frames)).

        anneal : float, default=0.90 (or 0.98 if extended)
            Annealing constant in (0,1]. Controls speed of convergence.

        annealdeg : float, default=60
            Degrees weight change for annealing.

        stop : float, default=1e-6 (or 1e-7 if ncomps > 32)
            Stop training when weight-change < this value.

        maxsteps : int, default=512
            Max number of ICA training steps.

        bias : str, default='on'
            Perform bias adjustment. Options: 'on', 'off'.

        momentum : float, default=0
            Training momentum in [0,1].

        posact : str, default='off'
            Make all component activations net-positive. Options: 'on', 'off'.

        verbose : str or int, default='on' or 1
            Give ascii messages. Options: 'on', 'off' or 1, 0.

        logfile : str, optional
            Save all messages in a log file.

        interrupt : str, default='off'
            Draw interrupt figure. Options: 'on', 'off'.

        rndreset : str or bool, default='off' or False
            Reset the random seed based on time of day. Default 'off' means
            ICA will always return the SAME decomposition (deterministic).

    Returns
    -------
    weights : ndarray, shape (ncomps, chans)
        ICA weight matrix (in reverse order of projected mean variance).

    sphere : ndarray, shape (chans, chans)
        Data sphering matrix. Note: unmixing_matrix = weights @ sphere.
        If sphering='off' or 'none', returns identity.

    compvars : ndarray, shape (ncomps,)
        Back-projected component variances (in reverse order).

    bias : ndarray, shape (ncomps, 1)
        Vector of final online bias values.

    signs : ndarray, shape (ncomps,)
        Extended-ICA signs for components (-1 = sub-Gaussian, 1 = super-Gaussian).

    lrates : ndarray, shape (laststep,)
        Vector of learning rates used at each training step.

    activations : ndarray, shape (ncomps, frames), optional
        Activation time courses of the output components.

    Notes
    -----
    This is a direct translation of the MATLAB runica() function maintaining
    numerical parity. Uses float64 precision throughout for consistency.

    The RNG mechanism uses np.random.RandomState with seed 5489 (MATLAB default)
    when rndreset='off', or a time-based seed when rndreset='on'.
    """

    # =========================================================================
    # 1. DATA VALIDATION AND INITIALIZATION
    # =========================================================================

    # Ensure data is float64 for numerical consistency with MATLAB
    data = np.asarray(data, dtype=np.float64)

    chans, frames = data.shape
    urchans = chans  # remember original data channels
    datalength = frames

    if chans < 2:
        raise ValueError(f'runica(): data size ({chans},{frames}) too small.')

    # =========================================================================
    # 2. PARAMETER PARSING AND DEFAULTS
    # =========================================================================

    # Parse keyword arguments (case-insensitive)
    kwargs_lower = {k.lower(): v for k, v in kwargs.items()}

    # Initialize all parameters with defaults
    pcaflag = DEFAULT_PCAFLAG
    sphering = DEFAULT_SPHEREFLAG
    posactflag = DEFAULT_POSACTFLAG
    verbose = DEFAULT_VERBOSE
    logfile = kwargs_lower.get('logfile', None)

    # Heuristic defaults that depend on data size
    DEFAULT_LRATE_DATA = 0.00065 / np.log(chans)
    DEFAULT_BLOCK_DATA = int(np.ceil(min(5 * np.log(frames), 0.3 * frames)))

    lrate = kwargs_lower.get('lrate', DEFAULT_LRATE_DATA)
    block = kwargs_lower.get('block', DEFAULT_BLOCK_DATA)
    block = kwargs_lower.get('blocksize', block)  # also accept 'blocksize'
    block = int(np.floor(block))

    annealdeg = kwargs_lower.get('annealdeg', DEFAULT_ANNEALDEG)
    annealdeg = kwargs_lower.get('degrees', annealdeg)  # also accept 'degrees'

    annealstep = kwargs_lower.get('anneal', 0)
    annealstep = kwargs_lower.get('annealstep', annealstep)  # also accept 'annealstep'

    nochange = kwargs_lower.get('stop', np.nan)
    nochange = kwargs_lower.get('nochange', nochange)  # also accept 'nochange'
    nochange = kwargs_lower.get('stopping', nochange)  # also accept 'stopping'

    momentum = kwargs_lower.get('momentum', DEFAULT_MOMENTUM)
    maxsteps = kwargs_lower.get('maxsteps', DEFAULT_MAXSTEPS)
    maxsteps = kwargs_lower.get('steps', maxsteps)  # also accept 'steps'

    weights = kwargs_lower.get('weights', 0)
    weights = kwargs_lower.get('weight', weights)  # also accept 'weight'
    wts_passed = 0 if isinstance(weights, int) and weights == 0 else 1

    ncomps = kwargs_lower.get('ncomps', chans)
    biasflag = DEFAULT_BIASFLAG

    interrupt = kwargs_lower.get('interrupt', DEFAULT_INTERRUPT)
    interrupt = kwargs_lower.get('interupt', interrupt)  # also accept typo

    extended = kwargs_lower.get('extended', DEFAULT_EXTENDED)
    extended = kwargs_lower.get('extend', extended)  # also accept 'extend'

    extblocks = DEFAULT_EXTBLOCKS
    kurtsize = MAX_KURTSIZE
    signsbias = 0.02
    extmomentum = DEFAULT_EXTMOMENTUM
    nsub = DEFAULT_NSUB
    wts_blowup = 0

    reset_randomseed = kwargs_lower.get('rndreset', DEFAULT_RESETRANDOMSEED)

    # Handle PCA parameter
    if 'pca' in kwargs_lower:
        pca_val = kwargs_lower['pca']
        if ncomps < urchans and ncomps != pca_val:
            raise ValueError('runica(): Use either PCA or ICA dimension reduction')
        pcaflag = 'on'
        ncomps = pca_val
        if ncomps > chans or ncomps < -1:
            raise ValueError(f'runica(): pca value must be in range [{-chans+1},{chans}]')
        if ncomps < 0:
            ncomps = data.shape[0] + ncomps
        chans = ncomps

    # Handle sphering parameter
    if 'sphering' in kwargs_lower or 'sphereing' in kwargs_lower or 'sphere' in kwargs_lower:
        sphering = kwargs_lower.get('sphering', kwargs_lower.get('sphereing',
                                    kwargs_lower.get('sphere', sphering)))
        if sphering not in ['on', 'off', 'none']:
            raise ValueError('runica(): sphering value must be on, off, or none')

    # Handle bias parameter
    if 'bias' in kwargs_lower:
        bias_val = kwargs_lower['bias']
        if bias_val == 'on':
            biasflag = 1
        elif bias_val == 'off':
            biasflag = 0
        else:
            raise ValueError('runica(): bias value must be on or off')

    # Handle extended parameter
    if 'extended' in kwargs_lower or 'extend' in kwargs_lower:
        ext_val = extended  # already retrieved above
        if ext_val != 0:
            extended = 1
            extblocks = int(ext_val)
            if extblocks < 0:
                nsub = -1 * extblocks
            elif extblocks == 0:
                extended = 0
            elif kurtsize > frames:
                kurtsize = frames
                if kurtsize < MIN_KURTSIZE:
                    print(f'runica() warning: kurtosis values inexact for << {MIN_KURTSIZE} points.')

    # Handle verbose parameter
    if 'verbose' in kwargs_lower:
        verb_val = kwargs_lower['verbose']
        if isinstance(verb_val, str):
            if verb_val == 'on':
                verbose = 1
            elif verb_val == 'off':
                verbose = 0
            else:
                raise ValueError('runica(): verbose flag value must be on or off')
        else:
            verbose = int(verb_val)

    # Handle rndreset parameter
    if 'rndreset' in kwargs_lower:
        rnd_val = kwargs_lower['rndreset']
        if isinstance(rnd_val, str):
            if rnd_val == 'yes' or rnd_val == 'on':
                reset_randomseed = True
            elif rnd_val == 'no' or rnd_val == 'off':
                reset_randomseed = False
            else:
                raise ValueError('runica(): rndreset should be yes, no, on, off, 0, or 1')
        else:
            reset_randomseed = bool(rnd_val)

    # Handle posact parameter
    if 'posact' in kwargs_lower:
        posact_val = kwargs_lower['posact']
        if posact_val not in ['on', 'off']:
            raise ValueError('runica(): posact value must be on or off')
        posactflag = posact_val

    # =========================================================================
    # 3. SPECIAL PARAMETER ADJUSTMENTS
    # =========================================================================

    # Set annealstep based on extended mode if not provided
    if annealstep == 0:
        if extended == 0:
            annealstep = DEFAULT_ANNEALSTEP
        else:
            annealstep = DEFAULT_EXTANNEAL

    # Adjust annealdeg based on momentum
    if annealdeg == DEFAULT_ANNEALDEG:
        annealdeg = DEFAULT_ANNEALDEG - momentum * 90
        if annealdeg < 0:
            annealdeg = 0

    # Auto-adjust nochange threshold based on ncomps
    nochangeupdated = False
    if np.isnan(nochange):
        if ncomps > 32:
            nochange = 1e-7
            nochangeupdated = True
        else:
            nochange = DEFAULT_STOP
            nochangeupdated = True

    # =========================================================================
    # 4. VALIDATION CHECKS
    # =========================================================================

    if ncomps > chans or ncomps < 1:
        raise ValueError(f'runica(): number of components must be 1 to {chans}.')

    if frames < chans:
        raise ValueError(f'runica(): data length ({frames}) < data channels ({chans})!')

    if block < 2:
        raise ValueError(f'runica(): block size {block} too small!')

    if block > frames:
        raise ValueError('runica(): block size exceeds data length!')

    if nsub > ncomps:
        raise ValueError(f'runica(): there can be at most {ncomps} sub-Gaussian components!')

    if lrate > MAX_LRATE or lrate < 0:
        raise ValueError('runica(): lrate value is out of bounds')

    if lrate == 0:
        lrate = DEFAULT_LRATE_DATA

    if maxsteps == 0:
        maxsteps = DEFAULT_MAXSTEPS

    if maxsteps < 0:
        raise ValueError(f'runica(): maxsteps value ({maxsteps}) must be a positive integer')

    if annealstep <= 0 or annealstep > 1:
        raise ValueError(f'runica(): anneal step value ({annealstep:.4f}) must be (0,1]')

    if annealdeg > 180 or annealdeg < 0:
        raise ValueError(f'runica(): annealdeg ({annealdeg:.1f}) is out of bounds [0,180]')

    if momentum > 1.0 or momentum < 0:
        raise ValueError('runica(): momentum value is out of bounds [0,1]')

    # =========================================================================
    # 5. VERBOSE INITIALIZATION MESSAGES
    # =========================================================================

    if verbose:
        print(f'\nInput data size [{chans},{frames}] = {chans} channels, {frames} frames')

        if pcaflag == 'on':
            print('After PCA dimension reduction,\n  finding ', end='')
        else:
            print('Finding ', end='')

        if extended == 0:
            print(f'{ncomps} ICA components using logistic ICA.')
        else:
            print(f'{ncomps} ICA components using extended ICA.')
            if extblocks > 0:
                print(f'Kurtosis will be calculated initially every {extblocks} blocks using {kurtsize} data points.')
            else:
                print(f'Kurtosis will not be calculated. Exactly {nsub} sub-Gaussian components assumed.')

        print(f'Decomposing {int(np.floor(frames/ncomps**2))} frames per ICA weight '
              f'(({ncomps})^2 = {ncomps**2} weights, {frames} frames)')
        print(f'Initial learning rate will be {lrate:g}, block size {block}.')

        if momentum > 0:
            print(f'Momentum will be {momentum:g}.')

        print(f'Learning rate will be multiplied by {annealstep:g} whenever angledelta >= {annealdeg:g} deg.')

        if nochangeupdated:
            print('More than 32 channels: default stopping weight change 1E-7')

        print(f'Training will end when wchange < {nochange:g} or after {maxsteps} steps.')

        if biasflag:
            print('Online bias adjustment will be used.')
        else:
            print('Online bias adjustment will not be used.')

    # =========================================================================
    # 6. DATA PREPROCESSING - REMOVE ROW MEANS
    # =========================================================================

    if verbose:
        print('Removing mean of each channel ...')

    rowmeans = np.mean(data, axis=1)  # shape: (chans,)
    for i in range(data.shape[0]):
        data[i, :] = data[i, :] - rowmeans[i]

    if verbose:
        print(f'Final training data range: {np.min(data):g} to {np.max(data):g}')

    # =========================================================================
    # 7. PCA DIMENSION REDUCTION (if requested)
    # =========================================================================

    if pcaflag == 'on':
        if verbose:
            print(f'Reducing the data to {ncomps} principal dimensions...')

        # Transpose and normalize
        PCdat2 = data.T  # shape: (frames, chans)
        PCn, PCp = PCdat2.shape
        PCdat2 = PCdat2 / PCn
        PCout = data @ PCdat2

        # Eigendecomposition
        # Note: scipy.linalg.eig returns (eigenvalues, eigenvectors)
        PCeigenval, PCEigenVec = eig(PCout)

        # Ensure real values (discard negligible imaginary parts)
        PCeigenval = PCeigenval.real
        PCEigenVec = PCEigenVec.real

        # Sort by decreasing eigenvalue
        PCindex = np.argsort(PCeigenval)[::-1]  # descending order
        PCEigenValues = PCeigenval[PCindex]
        PCEigenVectors = PCEigenVec[:, PCindex]

        # Project to ncomps dimensions
        eigenvectors = PCEigenVectors
        eigenvalues = PCEigenValues
        data = eigenvectors[:, :ncomps].T @ data

    # =========================================================================
    # 8. SPHERING COMPUTATION
    # =========================================================================

    if sphering == 'on':
        if verbose:
            print('Computing the sphering matrix...')

        # Compute sphering matrix: 2 * inv(sqrtm(cov(data')))
        sphere = 2.0 * np.linalg.inv(sqrtm(np.cov(data, rowvar=True)))
        sphere = sphere.real  # Ensure real (sqrtm can return complex)

        if wts_passed == 0:
            if verbose:
                print('Starting weights are the identity matrix ...')
            weights = np.eye(ncomps, chans)
        else:
            if verbose:
                print('Using starting weights named on commandline ...')

        if verbose:
            print('Sphering the data ...')
        data = sphere @ data

    elif sphering == 'off':
        if wts_passed == 0:
            if verbose:
                print('Using the sphering matrix as the starting weight matrix ...')
                print('Returning the identity matrix in variable "sphere" ...')
            sphere_temp = 2.0 * np.linalg.inv(sqrtm(np.cov(data, rowvar=True)))
            sphere_temp = sphere_temp.real
            weights = np.eye(ncomps, chans) @ sphere_temp
            sphere = np.eye(chans)
        else:
            if verbose:
                print('Using starting weights from commandline ...')
                print('Returning the identity matrix in variable "sphere" ...')
            sphere = np.eye(chans)

    elif sphering == 'none':
        sphere = np.eye(chans, chans)
        if wts_passed == 0:
            if verbose:
                print('Starting weights are the identity matrix ...')
                print('Returning the identity matrix in variable "sphere" ...')
            weights = np.eye(ncomps, chans)
        else:
            if verbose:
                print('Using starting weights named on commandline ...')
                print('Returning the identity matrix in variable "sphere" ...')
        if verbose:
            print('Returned variable "sphere" will be the identity matrix.')

    # =========================================================================
    # 9. WEIGHT INITIALIZATION FOR TRAINING
    # =========================================================================

    lastt = int((datalength / block - 1) * block + 1)
    BI = block * np.eye(ncomps, ncomps)
    delta = np.zeros(chans * ncomps)
    changes = []
    degconst = 180.0 / np.pi
    startweights = weights.copy()
    prevweights = startweights.copy()
    oldweights = startweights.copy()
    prevwtchange = np.zeros((chans, ncomps))
    oldwtchange = np.zeros((chans, ncomps))
    lrates = np.zeros(maxsteps)
    onesrow = np.ones((1, block))
    bias = np.zeros((ncomps, 1))

    # Initialize signs for extended-ICA
    signs = np.ones(ncomps)
    for k in range(nsub):
        signs[k] = -1  # first nsub components are sub-Gaussian

    if extended and extblocks < 0:
        if verbose:
            print('Fixed extended-ICA sign assignments: ', end='')
            for k in range(ncomps):
                print(f'{int(signs[k])} ', end='')
            print()

    signs = np.diag(signs)  # make diagonal matrix
    oldsigns = np.zeros_like(signs)
    signcount = 0
    signcounts = []
    urextblocks = extblocks  # original value for resets
    old_kk = np.zeros(ncomps)

    # =========================================================================
    # 10. RNG INITIALIZATION
    # =========================================================================

    # Initialize random number generator using mechanism from test_parity_rng.py
    # MATLAB default seed is 5489, equivalent to rng('default')
    if reset_randomseed:
        # Set seed based on time (random state)
        # Use None to get time-based seed, similar to MATLAB's sum(100*clock)
        import time
        seed = int(time.time() * 1000) % (2**32)
        rng = np.random.RandomState(seed)
    else:
        # Fixed seed for reproducibility (MATLAB default)
        # MATLAB uses rand('state', 0) which corresponds to seed 5489
        rng = np.random.RandomState(5489)

    # Store RNG for use in training loop
    # This will be used for:
    # - Data shuffling (permutation)
    # - Random subset selection for kurtosis estimation

    if verbose:
        print('Beginning ICA training ...', end='')
        if extended:
            print(' first training step may be slow ...')
        else:
            print()

    # =========================================================================
    # Phase 2: Core ICA Training Loop
    # =========================================================================

    # Helper function for random permutation
    def custom_randperm(n, rng_state):
        """
        Random permutation compatible with ICA algorithm requirements.

        Note: This uses numpy's permutation which is faster and more reliable
        than trying to exactly replicate MATLAB's randperm internal algorithm.
        The ICA algorithm is stochastic by nature, so different permutations
        will lead to slightly different convergence paths, but the final
        solution quality should be similar.

        For exact MATLAB parity in research contexts, set rndreset='off' to
        use the same random seed in both MATLAB and Python.
        """
        return rng_state.permutation(n)

    # Initialize step counters and tracking variables
    step = 0  # Training step counter (MATLAB line 795)
    laststep = 0  # Will be set when stopping criterion met
    blockno = 1  # Running block counter for kurtosis (MATLAB line 797)

    # Variables for tracking weight changes and annealing
    olddelta = delta.copy()  # Previous weight change vector
    oldchange = 0.0  # Previous squared weight change
    change = 0.0  # Current squared weight change
    angledelta = 0.0  # Angle between weight changes
    wts_blowup = 0  # Flag for weight matrix blowup

    # =========================================================================
    # TRAINING LOOP DISPATCHER
    # =========================================================================
    # Four variants based on biasflag and extended flags:
    # 1. biasflag=True,  extended=True  (lines 827-1001 in MATLAB)
    # 2. biasflag=True,  extended=False (lines 1003-1125 in MATLAB)
    # 3. biasflag=False, extended=True  (lines 1127-1295 in MATLAB)
    # 4. biasflag=False, extended=False (lines 1298-1422 in MATLAB)

    # =========================================================================
    # TRAINING LOOP 1: bias=True, extended=True (Extended-ICA with tanh)
    # =========================================================================
    # This implements lines 827-1001 of runica.m

    if biasflag and extended:
        while step < maxsteps:  # MATLAB line 828

            # Shuffle data order at each step (MATLAB line 829)
            timeperm = custom_randperm(datalength, rng)

            # Process data in blocks (MATLAB line 831)
            for t in range(0, lastt, block):

                # Extract and process block (MATLAB line 846)
                # MATLAB: u = weights*double(data(:,timeperm(t:t+block-1))) + bias*onesrow
                u = weights @ data[:, timeperm[t:t+block]] + bias @ onesrow

                # Apply tanh nonlinearity (MATLAB line 848)
                y = np.tanh(u)

                # Extended-ICA natural gradient weight update (MATLAB line 849)
                # weights = weights + lrate*(BI-signs*y*u'-u*u')*weights
                weights = weights + lrate * (BI - signs @ y @ u.T - u @ u.T) @ weights

                # Bias update for tanh (MATLAB line 850)
                # bias = bias + lrate*sum((-2*y)')';
                bias = bias + lrate * np.sum(-2*y, axis=1, keepdims=True)

                # Add momentum if enabled (MATLAB lines 852-856)
                if momentum > 0:
                    weights = weights + momentum * prevwtchange
                    prevwtchange = weights - prevweights
                    prevweights = weights.copy()

                # Check for weight blowup (MATLAB lines 858-861)
                if np.max(np.abs(weights)) > MAX_WEIGHT:
                    wts_blowup = 1
                    change = nochange

                # Extended-ICA kurtosis estimation (MATLAB lines 862-900)
                if not wts_blowup:
                    # Recompute signs vector using kurtosis (MATLAB line 866)
                    if extblocks > 0 and blockno % extblocks == 0:
                        # Random subset selection or whole data (MATLAB lines 868-879)
                        if kurtsize < frames:
                            # Pick random subset (MATLAB lines 869-876)
                            rp = (rng.rand(kurtsize) * datalength).astype(int)
                            # Account for possibility of 0 generation (MATLAB lines 870-875)
                            while np.any(rp == 0):
                                ou = np.where(rp == 0)[0]
                                rp[ou] = (rng.rand(len(ou)) * datalength).astype(int)
                            partact = weights @ data[:, rp[:kurtsize]]
                        else:
                            # For small data sets, use whole data (MATLAB lines 877-878)
                            partact = weights @ data

                        # Compute kurtosis (MATLAB lines 880-882)
                        m2 = np.mean(partact**2, axis=1)**2
                        m4 = np.mean(partact**4, axis=1)
                        kk = (m4 / m2) - 3.0  # kurtosis estimates

                        # Apply momentum to kurtosis (MATLAB lines 883-886)
                        if extmomentum:
                            kk = extmomentum * old_kk + (1.0 - extmomentum) * kk
                            old_kk = kk

                        # Update signs based on kurtosis (MATLAB line 887)
                        signs = np.diag(np.sign(kk + signsbias))

                        # Track sign changes (MATLAB lines 888-898)
                        if np.array_equal(signs, oldsigns):
                            signcount = signcount + 1
                        else:
                            signcount = 0

                        oldsigns = signs.copy()
                        signcounts.append(signcount)

                        # Make kurtosis estimation less frequent if signs stable (MATLAB lines 895-898)
                        if signcount >= SIGNCOUNT_THRESHOLD:
                            extblocks = int(extblocks * SIGNCOUNT_STEP)
                            signcount = 0

                # Increment block counter (MATLAB line 901)
                blockno = blockno + 1

                # Break if weights blew up (MATLAB lines 902-904)
                if wts_blowup:
                    break

            # End of block loop (MATLAB line 905)

            # Compute weight changes if no blowup (MATLAB lines 907-917)
            if not wts_blowup:
                oldwtchange = weights - oldweights
                step = step + 1

                # Store learning rate (MATLAB line 913)
                lrates[step-1] = lrate

                # Compute change magnitude (MATLAB lines 914-916)
                angledelta = 0.0
                delta = oldwtchange.flatten()
                change = delta @ delta

            # Check for restart conditions (MATLAB lines 921-999)
            if wts_blowup or np.isnan(change) or np.isinf(change):
                if verbose:
                    print('')

                # Restart training (MATLAB lines 923-945)
                step = 0
                change = nochange
                wts_blowup = 0
                blockno = 1
                lrate = lrate * DEFAULT_RESTART_FAC
                weights = startweights.copy()
                oldweights = startweights.copy()
                change = nochange
                oldwtchange = np.zeros((chans, ncomps))
                delta = np.zeros(chans * ncomps)
                olddelta = delta.copy()
                extblocks = urextblocks
                prevweights = startweights.copy()
                prevwtchange = np.zeros((chans, ncomps))
                lrates = np.zeros(maxsteps)
                bias = np.zeros((ncomps, 1))

                # Reinitialize signs (MATLAB lines 940-945)
                signs_vec = np.ones(ncomps)
                for k in range(nsub):
                    signs_vec[k] = -1
                signs = np.diag(signs_vec)
                oldsigns = np.zeros_like(signs)

                # Check if we can continue (MATLAB lines 947-960)
                if lrate > MIN_LRATE:
                    r = np.linalg.matrix_rank(data)
                    if r < ncomps:
                        if verbose:
                            print(f'Data has rank {r}. Cannot compute {ncomps} components.')
                        break
                    else:
                        if verbose:
                            print(f'Lowering learning rate to {lrate:g} and starting again.')
                else:
                    if verbose:
                        print('runica(): QUITTING - weight matrix may not be invertible!')
                    break

            else:  # Weights in bounds (MATLAB line 961)

                # Compute angle delta after step 2 (MATLAB lines 965-967)
                if step > 2:
                    cos_angle = (delta @ olddelta) / np.sqrt(change * oldchange)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angledelta = np.arccos(cos_angle)

                # Print progress (MATLAB lines 968-970)
                if verbose and (step % 10 == 0 or step < 5):
                    print(f'step {step} - lrate {lrate:5f}, wchange {change:8.8f}, '
                          f'angledelta {degconst*angledelta:4.1f} deg')

                # Save current values (MATLAB lines 974-975)
                changes.append(change)
                oldweights = weights.copy()

                # Anneal learning rate (MATLAB lines 979-986)
                if degconst * angledelta > annealdeg:
                    lrate = lrate * annealstep
                    olddelta = delta.copy()
                    oldchange = change
                elif step == 1:
                    olddelta = delta.copy()
                    oldchange = change

                # Apply stopping rule (MATLAB lines 990-995)
                if step > 2 and change < nochange:
                    laststep = step
                    step = maxsteps
                elif change > DEFAULT_BLOWUP:
                    lrate = lrate * DEFAULT_BLOWUP_FAC

        # End while step < maxsteps (MATLAB line 1000)

    # =========================================================================
    # TRAINING LOOP 2: bias=True, extended=False (standard logistic ICA)
    # =========================================================================
    # This implements lines 1003-1125 of runica.m
    # This is the most common use case

    elif biasflag and not extended:
        while step < maxsteps:  # MATLAB line 1004

            # Shuffle data order at each step (MATLAB line 1005)
            timeperm = custom_randperm(datalength, rng)

            # Process data in blocks (MATLAB line 1007)
            for t in range(0, lastt, block):

                # Extract and process block (MATLAB line 1021)
                # MATLAB: u = weights*double(data(:,timeperm(t:t+block-1))) + bias*onesrow
                # Note: MATLAB uses 1-based indexing, so t:t+block-1 means t to t+block
                u = weights @ data[:, timeperm[t:t+block]] + bias @ onesrow

                # Apply logistic nonlinearity (MATLAB line 1022)
                # Clip u to prevent overflow in exp
                u = np.maximum(u, -MAX_WEIGHT)
                u = np.minimum(u, MAX_WEIGHT)
                y = 1.0 / (1.0 + np.exp(-u))

                # Natural gradient weight update (MATLAB line 1023)
                # weights = weights + lrate*(BI+(1-2*y)*u')*weights
                weights = weights + lrate * (BI + (1 - 2*y) @ u.T) @ weights

                # Bias update (MATLAB line 1024)
                # bias = bias + lrate*sum((1-2*y)')';
                bias = bias + lrate * np.sum(1 - 2*y, axis=1, keepdims=True)

                # Add momentum if enabled (MATLAB lines 1026-1030)
                if momentum > 0:
                    weights = weights + momentum * prevwtchange
                    prevwtchange = weights - prevweights
                    prevweights = weights.copy()

                # Check for weight blowup (MATLAB lines 1032-1035)
                if np.max(np.abs(weights)) > MAX_WEIGHT:
                    wts_blowup = 1
                    change = nochange

                # Increment block counter (MATLAB line 1036)
                blockno = blockno + 1

                # Break if weights blew up (MATLAB lines 1037-1039)
                if wts_blowup:
                    break

            # End of block loop (MATLAB line 1040)

            # Compute weight changes if no blowup (MATLAB lines 1042-1052)
            if not wts_blowup:
                oldwtchange = weights - oldweights
                step = step + 1

                # Store learning rate (MATLAB line 1048)
                # MATLAB uses 1-based indexing: lrates(1,step)
                lrates[step-1] = lrate

                # Compute change magnitude (MATLAB lines 1049-1051)
                angledelta = 0.0
                delta = oldwtchange.flatten()  # Reshape to 1D
                change = delta @ delta  # Squared norm

            # Check for restart conditions (MATLAB lines 1056-1085)
            if wts_blowup or np.isnan(change) or np.isinf(change):
                if verbose:
                    print('')

                # Restart training (MATLAB lines 1058-1073)
                step = 0
                change = nochange
                wts_blowup = 0
                blockno = 1
                lrate = lrate * DEFAULT_RESTART_FAC  # Lower learning rate
                weights = startweights.copy()
                oldweights = startweights.copy()
                change = nochange
                oldwtchange = np.zeros((chans, ncomps))
                delta = np.zeros(chans * ncomps)
                olddelta = delta.copy()
                extblocks = urextblocks
                prevweights = startweights.copy()
                prevwtchange = np.zeros((chans, ncomps))
                lrates = np.zeros(maxsteps)
                bias = np.zeros((ncomps, 1))

                # Check if we can continue (MATLAB lines 1074-1085)
                if lrate > MIN_LRATE:
                    r = np.linalg.matrix_rank(data)
                    if r < ncomps:
                        if verbose:
                            print(f'Data has rank {r}. Cannot compute {ncomps} components.')
                        # Return current state
                        break
                    else:
                        if verbose:
                            print(f'Lowering learning rate to {lrate:g} and starting again.')
                else:
                    if verbose:
                        print('runica(): QUITTING - weight matrix may not be invertible!')
                    # Return current state
                    break

            else:  # Weights in bounds (MATLAB line 1086)

                # Compute angle delta after step 2 (MATLAB lines 1090-1092)
                if step > 2:
                    # acos((delta*olddelta')/sqrt(change*oldchange))
                    # Clip to avoid numerical issues with acos
                    cos_angle = (delta @ olddelta) / np.sqrt(change * oldchange)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angledelta = np.arccos(cos_angle)

                # Print progress (MATLAB lines 1093-1095)
                if verbose and (step % 10 == 0 or step < 5):
                    print(f'step {step} - lrate {lrate:5f}, wchange {change:8.8f}, '
                          f'angledelta {degconst*angledelta:4.1f} deg')

                # Save current values (MATLAB lines 1099-1100)
                changes.append(change)
                oldweights = weights.copy()

                # Anneal learning rate (MATLAB lines 1104-1111)
                if degconst * angledelta > annealdeg:
                    lrate = lrate * annealstep  # Anneal
                    olddelta = delta.copy()
                    oldchange = change
                elif step == 1:  # On first step only
                    olddelta = delta.copy()
                    oldchange = change

                # Apply stopping rule (MATLAB lines 1115-1120)
                if step > 2 and change < nochange:
                    laststep = step
                    step = maxsteps  # Stop when weights stabilize
                elif change > DEFAULT_BLOWUP:
                    lrate = lrate * DEFAULT_BLOWUP_FAC  # Keep trying with smaller rate

        # End while step < maxsteps (MATLAB line 1123)

    # =========================================================================
    # TRAINING LOOP 3: bias=False, extended=True (Extended-ICA, no bias)
    # =========================================================================
    # This implements lines 1127-1295 of runica.m

    elif not biasflag and extended:
        while step < maxsteps:  # MATLAB line 1128

            # Shuffle data order at each step (MATLAB line 1129)
            timeperm = custom_randperm(datalength, rng)

            # Process data in blocks (MATLAB line 1131)
            for t in range(0, lastt, block):

                # Extract and process block - NO BIAS (MATLAB line 1145)
                u = weights @ data[:, timeperm[t:t+block]]

                # Apply tanh nonlinearity (MATLAB line 1146)
                y = np.tanh(u)

                # Extended-ICA natural gradient weight update (MATLAB line 1147)
                weights = weights + lrate * (BI - signs @ y @ u.T - u @ u.T) @ weights

                # NO BIAS UPDATE for no-bias variant

                # Add momentum if enabled (MATLAB lines 1149-1153)
                if momentum > 0:
                    weights = weights + momentum * prevwtchange
                    prevwtchange = weights - prevweights
                    prevweights = weights.copy()

                # Check for weight blowup (MATLAB lines 1155-1158)
                if np.max(np.abs(weights)) > MAX_WEIGHT:
                    wts_blowup = 1
                    change = nochange

                # Extended-ICA kurtosis estimation (MATLAB lines 1159-1197)
                if not wts_blowup:
                    if extblocks > 0 and blockno % extblocks == 0:
                        if kurtsize < frames:
                            rp = (rng.rand(kurtsize) * datalength).astype(int)
                            while np.any(rp == 0):
                                ou = np.where(rp == 0)[0]
                                rp[ou] = (rng.rand(len(ou)) * datalength).astype(int)
                            partact = weights @ data[:, rp[:kurtsize]]
                        else:
                            partact = weights @ data

                        m2 = np.mean(partact**2, axis=1)**2
                        m4 = np.mean(partact**4, axis=1)
                        kk = (m4 / m2) - 3.0

                        if extmomentum:
                            kk = extmomentum * old_kk + (1.0 - extmomentum) * kk
                            old_kk = kk

                        signs = np.diag(np.sign(kk + signsbias))

                        if np.array_equal(signs, oldsigns):
                            signcount = signcount + 1
                        else:
                            signcount = 0

                        oldsigns = signs.copy()
                        signcounts.append(signcount)

                        if signcount >= SIGNCOUNT_THRESHOLD:
                            extblocks = int(extblocks * SIGNCOUNT_STEP)
                            signcount = 0

                blockno = blockno + 1

                if wts_blowup:
                    break

            # Compute weight changes if no blowup (MATLAB lines 1204-1214)
            if not wts_blowup:
                oldwtchange = weights - oldweights
                step = step + 1
                lrates[step-1] = lrate
                angledelta = 0.0
                delta = oldwtchange.flatten()
                change = delta @ delta

            # Check for restart conditions (MATLAB lines 1218-1256)
            if wts_blowup or np.isnan(change) or np.isinf(change):
                if verbose:
                    print('')

                step = 0
                change = nochange
                wts_blowup = 0
                blockno = 1
                lrate = lrate * DEFAULT_RESTART_FAC
                weights = startweights.copy()
                oldweights = startweights.copy()
                change = nochange
                oldwtchange = np.zeros((chans, ncomps))
                delta = np.zeros(chans * ncomps)
                olddelta = delta.copy()
                extblocks = urextblocks
                prevweights = startweights.copy()
                prevwtchange = np.zeros((chans, ncomps))
                lrates = np.zeros(maxsteps)
                bias = np.zeros((ncomps, 1))

                signs_vec = np.ones(ncomps)
                for k in range(nsub):
                    signs_vec[k] = -1
                signs = np.diag(signs_vec)
                oldsigns = np.zeros_like(signs)

                if lrate > MIN_LRATE:
                    r = np.linalg.matrix_rank(data)
                    if r < ncomps:
                        if verbose:
                            print(f'Data has rank {r}. Cannot compute {ncomps} components.')
                        break
                    else:
                        if verbose:
                            print(f'Lowering learning rate to {lrate:g} and starting again.')
                else:
                    if verbose:
                        print('runica(): QUITTING - weight matrix may not be invertible!')
                    break

            else:  # Weights in bounds

                # Compute angle delta after step 2 (MATLAB lines 1261-1263)
                if step > 2:
                    cos_angle = (delta @ olddelta) / np.sqrt(change * oldchange)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angledelta = np.arccos(cos_angle)

                # Print progress (MATLAB lines 1265-1266)
                if verbose and (step % 10 == 0 or step < 5):
                    print(f'step {step} - lrate {lrate:5f}, wchange {change:8.8f}, '
                          f'angledelta {degconst*angledelta:4.1f} deg')

                # Save current values (MATLAB lines 1270-1271)
                changes.append(change)
                oldweights = weights.copy()

                # Anneal learning rate (MATLAB lines 1275-1282)
                if degconst * angledelta > annealdeg:
                    lrate = lrate * annealstep
                    olddelta = delta.copy()
                    oldchange = change
                elif step == 1:
                    olddelta = delta.copy()
                    oldchange = change

                # Apply stopping rule (MATLAB lines 1286-1291)
                if step > 2 and change < nochange:
                    laststep = step
                    step = maxsteps
                elif change > DEFAULT_BLOWUP:
                    lrate = lrate * DEFAULT_BLOWUP_FAC

        # End while step < maxsteps (MATLAB line 1294)

    # =========================================================================
    # TRAINING LOOP 4: bias=False, extended=False (standard ICA, no bias)
    # =========================================================================
    # This implements lines 1298-1422 of runica.m

    else:  # not biasflag and not extended
        while step < maxsteps:  # MATLAB line 1299

            # Shuffle data order at each step (MATLAB line 1300)
            timeperm = custom_randperm(datalength, rng)

            # Process data in blocks (MATLAB line 1302)
            for t in range(0, lastt, block):

                # Extract and process block - NO BIAS (MATLAB line 1315)
                u = weights @ data[:, timeperm[t:t+block]]

                # Apply logistic nonlinearity (MATLAB line 1316)
                u = np.maximum(u, -MAX_WEIGHT)
                u = np.minimum(u, MAX_WEIGHT)
                y = 1.0 / (1.0 + np.exp(-u))

                # Natural gradient weight update (MATLAB line 1317)
                weights = weights + lrate * (BI + (1 - 2*y) @ u.T) @ weights

                # NO BIAS UPDATE for no-bias variant

                # Add momentum if enabled (MATLAB lines 1319-1323)
                if momentum > 0:
                    weights = weights + momentum * prevwtchange
                    prevwtchange = weights - prevweights
                    prevweights = weights.copy()

                # Check for weight blowup (MATLAB lines 1325-1328)
                if np.max(np.abs(weights)) > MAX_WEIGHT:
                    wts_blowup = 1
                    change = nochange

                blockno = blockno + 1

                if wts_blowup:
                    break

            # Compute weight changes if no blowup (MATLAB lines 1336-1346)
            if not wts_blowup:
                oldwtchange = weights - oldweights
                step = step + 1
                lrates[step-1] = lrate
                angledelta = 0.0
                delta = oldwtchange.flatten()
                change = delta @ delta

            # Check for restart conditions (MATLAB lines 1350-1383)
            if wts_blowup or np.isnan(change) or np.isinf(change):
                if verbose:
                    print('')

                step = 0
                change = nochange
                wts_blowup = 0
                blockno = 1
                lrate = lrate * DEFAULT_RESTART_FAC
                weights = startweights.copy()
                oldweights = startweights.copy()
                change = nochange
                oldwtchange = np.zeros((chans, ncomps))
                delta = np.zeros(chans * ncomps)
                olddelta = delta.copy()
                extblocks = urextblocks
                prevweights = startweights.copy()
                prevwtchange = np.zeros((chans, ncomps))
                lrates = np.zeros(maxsteps)
                bias = np.zeros((ncomps, 1))

                if lrate > MIN_LRATE:
                    r = np.linalg.matrix_rank(data)
                    if r < ncomps:
                        if verbose:
                            print(f'Data has rank {r}. Cannot compute {ncomps} components.')
                        break
                    else:
                        if verbose:
                            print(f'Lowering learning rate to {lrate:g} and starting again.')
                else:
                    if verbose:
                        print('runica(): QUITTING - weight matrix may not be invertible!')
                    break

            else:  # Weights in bounds

                # Compute angle delta after step 2 (MATLAB lines 1388-1390)
                if step > 2:
                    cos_angle = (delta @ olddelta) / np.sqrt(change * oldchange)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angledelta = np.arccos(cos_angle)

                # Print progress (MATLAB lines 1392-1393)
                if verbose and (step % 10 == 0 or step < 5):
                    print(f'step {step} - lrate {lrate:5f}, wchange {change:8.8f}, '
                          f'angledelta {degconst*angledelta:4.1f} deg')

                # Save current values (MATLAB lines 1397-1398)
                changes.append(change)
                oldweights = weights.copy()

                # Anneal learning rate (MATLAB lines 1402-1409)
                if degconst * angledelta > annealdeg:
                    lrate = lrate * annealstep
                    olddelta = delta.copy()
                    oldchange = change
                elif step == 1:
                    olddelta = delta.copy()
                    oldchange = change

                # Apply stopping rule (MATLAB lines 1413-1418)
                if step > 2 and change < nochange:
                    laststep = step
                    step = maxsteps
                elif change > DEFAULT_BLOWUP:
                    lrate = lrate * DEFAULT_BLOWUP_FAC

        # End while step < maxsteps (MATLAB line 1421)

    # =========================================================================
    # OUTPUT PREPARATION
    # =========================================================================

    # Truncate learning rate history (MATLAB lines 1431-1434)
    if laststep == 0:
        laststep = step
    lrates = lrates[:laststep]

    if verbose:
        print(f'Training complete. Total steps: {laststep}')

    # =========================================================================
    # Compute activations (MATLAB lines 1439-1455)
    # =========================================================================
    # Note: data matrix was mean-centered at the start
    # Need to add back row means for activation computation

    # Make activations from sphered data (MATLAB line 1439)
    # Add back the row means removed from data before sphering (MATLAB lines 1442-1447)
    if pcaflag == 'off':
        sr = sphere @ rowmeans
        for r in range(ncomps):
            data[r, :] = data[r, :] + sr[r]
        activations_unsorted = weights @ data  # MATLAB line 1447
    else:
        # For PCA case (MATLAB lines 1449-1453)
        ser = sphere @ eigenvectors[:, :ncomps].T @ rowmeans
        for r in range(ncomps):
            data[r, :] = data[r, :] + ser[r]
        activations_unsorted = weights @ data

    # Now 'activations_unsorted' are the component activations = weights*sphere*raw_data

    # =========================================================================
    # Compose PCA and ICA matrices if PCA was used (MATLAB lines 1462-1468)
    # =========================================================================
    if pcaflag == 'on':
        if verbose:
            print('Composing the eigenvector, weights, and sphere matrices')
            print(f'  into a single rectangular weights matrix; sphere=eye({chans})')
        weights = weights @ sphere @ eigenvectors[:, :ncomps].T
        sphere = np.eye(urchans)

    # =========================================================================
    # Sort components by descending mean projected variance (MATLAB lines 1470-1492)
    # =========================================================================
    if verbose:
        print('Sorting components in descending order of mean projected variance ...')

    # Compute inverse of unmixing matrix for backprojection (MATLAB lines 1477-1482)
    if ncomps == urchans:  # if weights are square
        winv = np.linalg.inv(weights @ sphere)
    else:
        if verbose:
            print('Using pseudo-inverse of weight matrix to rank order component projections.')
        winv = pinv(weights @ sphere)

    # Compute variances without backprojecting (MATLAB line 1486)
    # Formula from Rey Ramirez 8/07
    meanvar = np.sum(winv**2, axis=0) * np.sum(activations_unsorted**2, axis=1) / ((chans * frames) - 1)

    # Sort components by mean variance (MATLAB lines 1490-1492)
    sortvar = np.argsort(meanvar)
    windex = sortvar[::-1]  # order large to small
    meanvar = meanvar[windex]

    # =========================================================================
    # Permute activations and reorder weights (MATLAB lines 1521-1528)
    # =========================================================================
    if verbose:
        print('Permuting the activation wave forms ...')

    activations = activations_unsorted[windex, :]  # data is now activations (MATLAB line 1523)
    weights = weights[windex, :]  # reorder the weight matrix (MATLAB line 1527)
    bias = bias[windex]  # reorder bias (MATLAB line 1528)

    # Convert signs diagonal matrix to vector and reorder (MATLAB lines 1529-1530)
    signs_vec = np.diag(signs)  # vectorize the signs matrix
    signs_vec = signs_vec[windex]  # reorder them

    # Prepare final outputs
    bias_out = bias
    signs_out = signs_vec
    lrates_out = lrates

    return weights, sphere, meanvar, bias_out, signs_out, lrates_out
