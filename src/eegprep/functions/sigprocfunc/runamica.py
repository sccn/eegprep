"""
Low-level wrapper for the AMICA (Adaptive Mixture ICA) Fortran binary.

Handles binary discovery, parameter file I/O, subprocess execution, and
output loading. This is a Python port of the MATLAB runamica15.m / loadmodout15.m
functions from https://github.com/sccn/amica.

EEGPrep package distributions do not ship AMICA binaries. Source checkouts may
contain local development binaries under src/eegprep/bin, but installed users
should pass amica_binary, set AMICA_BINARY, or put the AMICA executable on PATH.

The binary communicates via a parameter file (input.param) and writes results
as raw binary files (float64) to an output directory.
"""

import logging
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.special import gamma

from ..miscfunc.pinv import pinv

logger = logging.getLogger(__name__)
PACKAGE_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# Default AMICA parameters (matching MATLAB runamica15.m defaults)
# --------------------------------------------------------------------------
_AMICA_DEFAULTS = {
    'block_size': 128,
    'do_opt_block': 0,
    'blk_min': 256,
    'blk_step': 256,
    'blk_max': 1024,
    'num_models': 1,
    'num_mix_comps': 3,
    'pdftype': 0,
    'max_iter': 2000,
    'lrate': 0.05,
    'minlrate': 1e-8,
    'lratefact': 0.5,
    'rholrate': 0.05,
    'rho0': 1.5,
    'minrho': 1.0,
    'maxrho': 2.0,
    'rholratefact': 0.5,
    'do_newton': 1,
    'newt_start': 50,
    'newt_ramp': 10,
    'newtrate': 1.0,
    'do_reject': 0,
    'numrej': 3,
    'rejsig': 3.0,
    'rejstart': 2,
    'rejint': 3,
    'writestep': 20,
    'write_nd': 1,
    'write_LLt': 1,
    'decwindow': 1,
    'max_decs': 3,
    'fix_init': 0,
    'update_A': 1,
    'update_c': 1,
    'update_gm': 1,
    'update_alpha': 1,
    'update_mu': 1,
    'update_beta': 1,
    'do_rho': 1,
    'do_mean': 1,
    'do_sphere': 1,
    'doPCA': 1,
    'doscaling': 1,
    'scalestep': 1,
    'do_history': 0,
    'histstep': 10,
    'share_comps': 0,
    'share_start': 100,
    'comp_thresh': 0.99,
    'share_iter': 100,
    'load_rej': 0,
    'load_W': 0,
    'load_c': 0,
    'load_gm': 0,
    'load_alpha': 0,
    'load_mu': 0,
    'load_beta': 0,
    'load_rho': 0,
    'load_comp_list': 0,
    'mineig': 1e-15,
    'invsigmax': 100.0,
    'invsigmin': 0.00001,
    'kurt_start': 3,
    'num_kurt': 5,
    'kurt_int': 1,
    'use_min_dll': 0,
    'min_dll': 1e-9,
    'use_grad_norm': 0,
    'min_grad_norm': 1e-7,
    'pcadb': 30.0,
    'byte_size': 4,
    'num_samples': 1,
    'field_blocksize': 1,
}

# Parameters that use integer format in the param file
_INT_PARAMS = {
    'block_size', 'do_opt_block', 'blk_min', 'blk_step', 'blk_max',
    'num_models', 'num_mix_comps', 'pdftype', 'max_iter', 'max_threads',
    'do_newton', 'newt_start', 'newt_ramp', 'do_reject', 'numrej',
    'rejstart', 'rejint', 'writestep', 'write_nd', 'write_LLt',
    'decwindow', 'max_decs', 'fix_init', 'update_A', 'update_c',
    'update_gm', 'update_alpha', 'update_mu', 'update_beta',
    'do_rho', 'do_mean', 'do_sphere', 'doPCA', 'doscaling', 'scalestep',
    'do_history', 'histstep', 'share_comps', 'share_start', 'share_iter',
    'load_rej', 'load_W', 'load_c', 'load_gm', 'load_alpha', 'load_mu',
    'load_beta', 'load_rho', 'load_comp_list', 'kurt_start', 'num_kurt',
    'kurt_int', 'use_min_dll', 'use_grad_norm', 'pcakeep', 'byte_size',
    'data_dim', 'field_dim', 'num_samples', 'field_blocksize',
    'min_dll', 'min_grad_norm',
}

# Parameters that use scientific notation format
_SCI_PARAMS = {
    'minlrate', 'mineig',
}

# Parameters that use string format
_STR_PARAMS = {
    'files', 'outdir', 'indir',
}


def _find_amica_binary(amica_binary=None):
    """Find the AMICA binary executable.

    Search order:
    1. Explicit amica_binary argument.
    2. AMICA_BINARY environment variable.
    3. Source-tree development binary in src/eegprep/bin/ if present.
    4. EEGLAB submodule plugins directory.
    5. System PATH.

    Parameters
    ----------
    amica_binary : str or None
        Explicit path to the AMICA binary.

    Returns
    -------
    str
        Path to the AMICA binary.

    Raises
    ------
    FileNotFoundError
        If the binary cannot be found.
    """
    # 1. Explicit argument
    if amica_binary is not None:
        if os.path.isfile(amica_binary) and os.access(amica_binary, os.X_OK):
            return amica_binary
        raise FileNotFoundError(
            f"Specified AMICA binary not found or not executable: {amica_binary}")

    # 2. Environment variable
    env_binary = os.environ.get('AMICA_BINARY')
    if env_binary:
        if os.path.isfile(env_binary) and os.access(env_binary, os.X_OK):
            return env_binary
        raise FileNotFoundError(
            f"AMICA_BINARY is set but not found or not executable: {env_binary}")

    # Determine platform-specific binary name
    system = platform.system()
    if system == 'Darwin':
        binary_name = 'amica15mac'
    elif system == 'Linux':
        binary_name = 'amica15ub'
    elif system == 'Windows':
        binary_name = 'amica15mkl.exe'
    else:
        binary_name = 'amica15'

    # 3. Source-tree development binary in src/eegprep/bin/
    bin_dir = os.path.join(PACKAGE_ROOT, 'bin')
    vendored = os.path.join(bin_dir, binary_name)
    if os.path.isfile(vendored) and os.access(vendored, os.X_OK):
        return vendored

    # 4. EEGLAB submodule plugins
    eeglab_dir = os.path.join(PACKAGE_ROOT, 'eeglab')
    if os.path.isdir(eeglab_dir):
        plugins_dir = os.path.join(eeglab_dir, 'plugins')
        if os.path.isdir(plugins_dir):
            for entry in os.listdir(plugins_dir):
                if entry.startswith('amica'):
                    candidate = os.path.join(plugins_dir, entry, binary_name)
                    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                        return candidate

    # 5. System PATH
    found = shutil.which(binary_name)
    if found is not None:
        return found

    raise FileNotFoundError(
        f"AMICA binary '{binary_name}' not found. EEGPrep wheels do not "
        f"ship AMICA binaries. Install options:\n"
        f"  1. Pass amica_binary=/path/to/{binary_name}\n"
        f"  2. Set AMICA_BINARY=/path/to/{binary_name}\n"
        f"  3. Add {binary_name} to your system PATH\n"
        f"  4. Download AMICA from https://github.com/sccn/amica/releases\n"
        f"Source checkouts may also use a development binary in {bin_dir}/."
    )


def _write_data_file(data, path):
    """Write data matrix as raw float32 little-endian .fdt file.

    Matches MATLAB's fwrite(fid, dat, 'float') behavior.

    Parameters
    ----------
    data : ndarray, shape (chans, frames)
        Data matrix.
    path : str
        Output file path (should end in .fdt).
    """
    data_f32 = np.asarray(data, dtype='<f4')  # little-endian float32
    data_f32.tofile(path)
    logger.info("Wrote data file: %s (%d bytes)", path, data_f32.nbytes)


def _write_param_file(outdir, params):
    """Write AMICA input.param file.

    Parameters
    ----------
    outdir : str
        Output directory (also where input.param is written).
    params : dict
        All parameters to write. Required keys: files, outdir, data_dim, field_dim.

    Returns
    -------
    str
        Path to the written param file.
    """
    param_path = os.path.join(outdir, 'input.param')

    # Merge defaults with user params (user params take precedence)
    merged = dict(_AMICA_DEFAULTS)
    merged.update(params)

    # AMICA binary requires 'files' to appear before 'num_samples' in the
    # param file. Write file/directory params first, then everything else.
    priority_keys = ['files', 'outdir', 'indir', 'data_dim', 'field_dim',
                     'num_samples', 'byte_size', 'field_blocksize']
    ordered_keys = [k for k in priority_keys if k in merged]
    ordered_keys += [k for k in merged if k not in priority_keys]

    with open(param_path, 'w') as f:
        for key in ordered_keys:
            val = merged[key]
            if key in _STR_PARAMS:
                f.write(f"{key} {val}\n")
            elif key in _SCI_PARAMS:
                f.write(f"{key} {val:e}\n")
            elif key in _INT_PARAMS:
                f.write(f"{key} {int(val):d}\n")
            else:
                # Default: use float format for numeric values, string otherwise
                if isinstance(val, (int, float, np.integer, np.floating)):
                    f.write(f"{key} {float(val):f}\n")
                else:
                    f.write(f"{key} {val}\n")

    logger.info("Wrote param file: %s", param_path)
    return param_path


def _amica_subprocess_kwargs():
    """Build env / preexec kwargs for invoking the AMICA binary.

    The Linux build of AMICA statically links MPICH and unconditionally
    calls MPI_Init at startup. Without an external launcher (mpiexec /
    hydra_pmi_proxy), MPICH falls back to "singleton init" which tries to
    spawn a hydra process and segfaults when it can't find one. Setting
    PMI_RANK=0 / PMI_SIZE=1 makes MPICH believe it is already running
    inside a 1-process job and skip the spawn entirely.

    OMP_STACKSIZE / KMP_STACKSIZE and a relaxed RLIMIT_STACK keep AMICA's
    OpenMP threads from running out of stack on hosts with a small default
    (e.g. 8 MiB on Linux CI runners).
    """
    env = {**os.environ}
    env.setdefault('OMP_STACKSIZE', '512M')
    env.setdefault('KMP_STACKSIZE', '512M')
    env.setdefault('PMI_RANK', '0')
    env.setdefault('PMI_SIZE', '1')

    kwargs = {'env': env}
    if os.name == 'posix':
        import resource
        def _raise_stack():
            try:
                resource.setrlimit(
                    resource.RLIMIT_STACK,
                    (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
            except (OSError, ValueError):
                pass
        kwargs['preexec_fn'] = _raise_stack
    return kwargs


_amica_works_cache = None


def is_amica_available():
    """Return True iff the AMICA binary can launch on this system.

    Verifies more than just file presence: the vendored Linux build uses
    AVX-512 instructions, and the Windows build requires the Intel Fortran
    runtime DLLs. On hosts that lack either (e.g. AMD EPYC without
    AVX-512, vanilla Windows runners), the binary segfaults on launch.
    Smoke-tests by invoking with a bogus param file: a healthy binary
    exits gracefully with a file-not-found error, while a broken one
    returns 174 (Intel Fortran severe) or is killed by signal.
    """
    global _amica_works_cache
    if _amica_works_cache is not None:
        return _amica_works_cache
    try:
        binary = _find_amica_binary()
    except FileNotFoundError:
        _amica_works_cache = False
        return False
    try:
        result = subprocess.run(
            [binary, '/nonexistent/eegprep/amica/probe.param'],
            timeout=15,
            capture_output=True,
            **_amica_subprocess_kwargs(),
        )
    except (subprocess.TimeoutExpired, OSError):
        _amica_works_cache = False
        return False
    rc = result.returncode
    # 174 = Intel Fortran "severe" (SIGSEGV/SIGILL/etc); 3221225781 =
    # 0xC0000135 STATUS_DLL_NOT_FOUND on Windows; negative rc = signal kill.
    if rc < 0 or rc == 174 or rc == 3221225781:
        _amica_works_cache = False
        return False
    _amica_works_cache = True
    return True


def _run_amica(binary, param_file):
    """Run the AMICA binary.

    Parameters
    ----------
    binary : str
        Path to the AMICA binary.
    param_file : str
        Path to the input.param file.

    Raises
    ------
    RuntimeError
        If the binary returns a non-zero exit code.
    """
    logger.info("Running AMICA: %s %s", binary, param_file)
    try:
        result = subprocess.run(
            [binary, param_file],
            check=True,
            capture_output=True,
            text=True,
            **_amica_subprocess_kwargs(),
        )
        if result.stdout:
            logger.info("AMICA stdout:\n%s", result.stdout)
        if result.stderr:
            logger.warning("AMICA stderr:\n%s", result.stderr)
    except subprocess.CalledProcessError as e:
        msg = f"AMICA binary failed with exit code {e.returncode}"
        if e.stderr:
            msg += f"\nstderr: {e.stderr}"
        if e.stdout:
            msg += f"\nstdout: {e.stdout}"
        raise RuntimeError(msg) from e


def _load_amica_output(outdir, num_models, num_pcs, data_dim, num_mix_comps,
                       max_iter, field_dim):
    """Load AMICA output files from disk.

    Port of MATLAB loadmodout15.m. Reads binary double-precision files and
    applies post-processing: model reordering, variance sorting, normalization.

    Parameters
    ----------
    outdir : str
        Directory containing AMICA output files.
    num_models : int
        Number of ICA models.
    num_pcs : int
        Number of principal components (PCA-reduced dimension).
    data_dim : int
        Original data dimensionality (number of channels).
    num_mix_comps : int
        Number of mixture components per source.
    max_iter : int
        Maximum iterations (for sizing LL and nd arrays).
    field_dim : int
        Number of data frames.

    Returns
    -------
    dict
        Dictionary with keys: W, S, A, mean, gm, c, alpha, mu, sbeta, rho,
        LL, LLt, Lht, Lt, nd, comp_list, svar, origord, mod_prob, v, num_pcs.
    """
    mods = {}
    nw = num_pcs

    # -- gm (model probabilities) --
    gm_path = os.path.join(outdir, 'gm')
    if os.path.isfile(gm_path):
        gm = np.fromfile(gm_path, dtype=np.float64)
        num_models = len(gm)
    else:
        logger.info("No gm file found, setting num_models to 1")
        gm = np.array([1.0])
        num_models = 1

    # -- W (unmixing weights) --
    w_path = os.path.join(outdir, 'W')
    if os.path.isfile(w_path):
        W_raw = np.fromfile(w_path, dtype=np.float64)
        nw2 = len(W_raw) // num_models
        nw = int(np.sqrt(nw2))
        W = W_raw.reshape(nw, nw, num_models, order='F')
    else:
        raise FileNotFoundError(f"AMICA output file 'W' not found in {outdir}")

    # -- mean (data mean) --
    mean_path = os.path.join(outdir, 'mean')
    nx = data_dim
    if os.path.isfile(mean_path):
        mn = np.fromfile(mean_path, dtype=np.float64)
        nx = len(mn)
    else:
        logger.info("No mean file found, setting mean to zeros")
        mn = np.zeros(nx)

    # -- S (sphering matrix) --
    s_path = os.path.join(outdir, 'S')
    if os.path.isfile(s_path):
        S_raw = np.fromfile(s_path, dtype=np.float64)
        if len(S_raw) == nx * nx:
            S = S_raw.reshape(nx, nx, order='F')
        else:
            # PCA-reduced sphere: num_pcs x data_dim
            S = S_raw.reshape(nw, nx, order='F') if len(S_raw) == nw * nx else S_raw.reshape(nx, nx, order='F')
    else:
        logger.info("No S file found, using identity")
        S = np.eye(nx)

    # -- comp_list --
    comp_list_path = os.path.join(outdir, 'comp_list')
    complistset = False
    if os.path.isfile(comp_list_path):
        comp_list = np.fromfile(comp_list_path, dtype=np.int32)
        expected = nw * num_models
        if len(comp_list) >= expected:
            comp_list = comp_list[:expected].reshape(nw, num_models, order='F')
            complistset = True
        else:
            logger.warning("comp_list has %d values, expected %d; ignoring",
                           len(comp_list), expected)
            comp_list = None
    else:
        comp_list = None

    # -- LLt (per-frame log-likelihood) --
    llt_path = os.path.join(outdir, 'LLt')
    LLtset = False
    if os.path.isfile(llt_path):
        LLt_raw = np.fromfile(llt_path, dtype=np.float64)
        n_rows = num_models + 1
        n_cols = len(LLt_raw) // n_rows
        if n_cols > 0:
            LLt = LLt_raw.reshape(n_rows, n_cols, order='F')
            Lht = LLt[:num_models, :]
            Lt = LLt[num_models, :]
            LLtset = True
        else:
            Lht = None
            Lt = None
    else:
        Lht = None
        Lt = None

    # -- LL (log-likelihood per iteration) --
    ll_path = os.path.join(outdir, 'LL')
    if os.path.isfile(ll_path):
        LL = np.fromfile(ll_path, dtype=np.float64)
    else:
        LL = np.zeros(0)

    # -- c (model centers) --
    c_path = os.path.join(outdir, 'c')
    if os.path.isfile(c_path):
        c = np.fromfile(c_path, dtype=np.float64).reshape(nw, num_models, order='F')
    else:
        c = np.zeros((nw, num_models))

    # -- alpha, mu, sbeta, rho (mixture parameters, indexed via comp_list) --
    alpha, mu, sbeta, rho = _load_mixture_params(
        outdir, nw, num_models, num_mix_comps, comp_list, complistset)

    # -- nd (weight change history) --
    nd_path = os.path.join(outdir, 'nd')
    ndset = False
    if os.path.isfile(nd_path):
        nd_raw = np.fromfile(nd_path, dtype=np.float64)
        actual_max_iter = len(nd_raw) // (nw * num_models)
        if actual_max_iter > 0:
            nd = nd_raw.reshape(actual_max_iter, nw, num_models, order='F')
            ndset = True
        else:
            nd = np.zeros(0)
    else:
        nd = np.zeros(0)

    # =====================================================================
    # Post-processing (matching loadmodout15.m)
    # =====================================================================

    # 1. Reorder models by descending gm
    gmord = np.argsort(gm)[::-1]
    gm_sorted = gm[gmord]
    W = W[:, :, gmord]
    c = c[:, gmord]
    alpha = alpha[:, :, gmord]
    mu = mu[:, :, gmord]
    sbeta = sbeta[:, :, gmord]
    rho = rho[:, :, gmord]
    if LLtset:
        Lht = Lht[gmord, :]
    if complistset:
        comp_list = comp_list[:, gmord]
    if ndset:
        nd = nd[:, :, gmord]

    # 2. Compute A = pinv(W @ S) per model
    n = nw
    A = np.zeros((nx, n, num_models))
    for h in range(num_models):
        A[:, :, h] = pinv(W[:, :, h] @ S[:nw, :])

    # 3. Compute source variance and sort components by descending variance
    # Count active mixture components (alpha > 0) per component
    num_mix_used = np.zeros((n, num_models), dtype=int)
    for h in range(num_models):
        for i in range(n):
            num_mix_used[i, h] = int(np.sum(alpha[:, i, h] > 0))

    svar = np.zeros((n, num_models))
    origord = np.zeros((n, num_models), dtype=int)
    for h in range(num_models):
        for i in range(n):
            nm = num_mix_used[i, h]
            if nm > 0:
                a_slice = alpha[:nm, i, h]
                mu_slice = mu[:nm, i, h]
                rho_slice = rho[:nm, i, h]
                sbeta_slice = sbeta[:nm, i, h]
                # Source variance: alpha * (mu^2 + gamma(3/rho)/gamma(1/rho) / sbeta^2)
                var_mix = a_slice * (
                    mu_slice ** 2
                    + gamma(3.0 / rho_slice) / gamma(1.0 / rho_slice) / sbeta_slice ** 2
                )
                svar[i, h] = np.sum(var_mix) * np.linalg.norm(A[:, i, h]) ** 2

        # Sort by descending variance
        order = np.argsort(svar[:, h])[::-1]
        origord[:, h] = order
        svar[:, h] = svar[order, h]
        A[:, :, h] = A[:, order, h]
        W[:, :, h] = W[order, :, h]
        alpha[:, :, h] = alpha[:, order, h]
        mu[:, :, h] = mu[:, order, h]
        sbeta[:, :, h] = sbeta[:, order, h]
        rho[:, :, h] = rho[:, order, h]
        if complistset:
            comp_list[:, h] = comp_list[order, h]
        if ndset:
            nd[:, :, h] = nd[:, order, h]

    # 4. Compute log10 posterior model odds
    v = None
    if LLtset:
        v = np.zeros((num_models, Lht.shape[1]))
        for h in range(num_models):
            v[h, :] = 0.4343 * (Lht[h, :] - Lt)

    # 5. Normalize: unit columns of A, scale W rows accordingly
    for h in range(num_models):
        for i in range(nw):
            na = np.linalg.norm(A[:, i, h])
            if na > 0:
                A[:, i, h] /= na
                W[i, :, h] *= na
                mu[:, i, h] *= na
                sbeta[:, i, h] /= na

    # Assemble output
    mods['num_models'] = num_models
    mods['num_pcs'] = nw
    mods['data_dim'] = nx
    mods['W'] = W
    mods['S'] = S
    mods['A'] = A
    mods['mean'] = mn
    mods['gm'] = gm_sorted
    mods['mod_prob'] = gm_sorted
    mods['c'] = c
    mods['alpha'] = alpha
    mods['mu'] = mu
    mods['sbeta'] = sbeta
    mods['rho'] = rho
    mods['LL'] = LL
    mods['svar'] = svar
    mods['origord'] = origord
    if LLtset:
        mods['Lht'] = Lht
        mods['Lt'] = Lt
    if v is not None:
        mods['v'] = v
    if complistset:
        mods['comp_list'] = comp_list
    if ndset:
        mods['nd'] = nd

    return mods


def _load_mixture_params(outdir, nw, num_models, num_mix_comps, comp_list,
                         complistset):
    """Load and remap alpha, mu, sbeta, rho via comp_list.

    Parameters
    ----------
    outdir : str
        Output directory.
    nw : int
        Number of components (num_pcs).
    num_models : int
        Number of models.
    num_mix_comps : int
        Expected number of mixture components.
    comp_list : ndarray or None
        Component index mapping, shape (nw, num_models).
    complistset : bool
        Whether comp_list was loaded.

    Returns
    -------
    tuple of ndarray
        (alpha, mu, sbeta, rho), each shape (num_mix, nw, num_models).
    """
    def _load_and_remap(filename, default_val, nw, num_models, num_mix_comps,
                        comp_list, complistset):
        fpath = os.path.join(outdir, filename)
        if os.path.isfile(fpath):
            raw = np.fromfile(fpath, dtype=np.float64)
            num_mix = len(raw) // (nw * num_models)
            if num_mix == 0:
                num_mix = num_mix_comps
                return np.full((num_mix, nw, num_models), default_val), num_mix
            tmp = raw.reshape(num_mix, nw * num_models, order='F')
            out = np.zeros((num_mix, nw, num_models))
            if complistset and comp_list is not None:
                for h in range(num_models):
                    for i in range(nw):
                        # comp_list uses 1-based MATLAB indexing
                        idx = int(comp_list[i, h]) - 1
                        # Clamp index to valid range
                        idx = max(0, min(idx, nw * num_models - 1))
                        out[:, i, h] = tmp[:, idx]
            else:
                out = tmp.reshape(num_mix, nw, num_models, order='F')
            return out, num_mix
        else:
            return np.full((num_mix_comps, nw, num_models), default_val), num_mix_comps

    alpha, num_mix = _load_and_remap('alpha', 1.0, nw, num_models, num_mix_comps,
                                     comp_list, complistset)
    mu, _ = _load_and_remap('mu', 0.0, nw, num_models, num_mix,
                            comp_list, complistset)
    sbeta, _ = _load_and_remap('sbeta', 1.0, nw, num_models, num_mix,
                               comp_list, complistset)
    rho, _ = _load_and_remap('rho', 2.0, nw, num_models, num_mix,
                             comp_list, complistset)

    return alpha, mu, sbeta, rho


def runamica(data, num_models=1, max_iter=2000, num_mix_comps=3, pcakeep=None,
             outdir=None, amica_binary=None, max_threads=4, cleanup=True,
             **kwargs):
    """Run AMICA binary on a data matrix.

    Parameters
    ----------
    data : ndarray, shape (chans, frames)
        Input data matrix (float64).
    num_models : int
        Number of ICA models to fit.
    max_iter : int
        Maximum number of training iterations.
    num_mix_comps : int
        Number of mixture components per source.
    pcakeep : int or None
        Number of principal components to retain. Default: chans.
    outdir : str or None
        Output directory. If None, a temporary directory is created.
    amica_binary : str or None
        Path to AMICA binary. If None, auto-detected.
    max_threads : int
        Maximum number of threads for the binary.
    cleanup : bool
        If True, remove temporary files after loading results.
    **kwargs
        Additional AMICA parameters written to the param file as-is.

    Returns
    -------
    weights : ndarray, shape (num_pcs, num_pcs)
        Unmixing weights for model 0 (post-sphering).
    sphere : ndarray, shape (num_pcs, chans)
        Sphering matrix.
    mods : dict
        Full model output with keys: W, S, A, mean, gm, c, alpha, mu, sbeta,
        rho, LL, LLt, nd, comp_list, svar, origord, mod_prob, num_pcs, etc.
    """
    data = np.asarray(data, dtype=np.float64)
    chans, frames = data.shape

    if pcakeep is None:
        pcakeep = chans

    binary = _find_amica_binary(amica_binary)
    logger.info("Using AMICA binary: %s", binary)

    # Create output directory
    tmp_created = False
    if outdir is None:
        outdir = tempfile.mkdtemp(prefix='amica_')
        tmp_created = True
    else:
        os.makedirs(outdir, exist_ok=True)

    # Write data file
    data_file = os.path.join(outdir, 'data.fdt')
    _write_data_file(data, data_file)

    # Build parameter dict
    params = {
        'files': data_file,
        'outdir': outdir + os.sep,
        'data_dim': chans,
        'field_dim': frames,
        'num_models': num_models,
        'max_iter': max_iter,
        'num_mix_comps': num_mix_comps,
        'pcakeep': pcakeep,
        'max_threads': max_threads,
    }
    params.update(kwargs)

    # Write parameter file
    param_file = _write_param_file(outdir, params)

    # Run the binary
    _run_amica(binary, param_file)

    # Load output
    mods = _load_amica_output(
        outdir,
        num_models=num_models,
        num_pcs=pcakeep,
        data_dim=chans,
        num_mix_comps=num_mix_comps,
        max_iter=max_iter,
        field_dim=frames,
    )

    # Extract model 0 weights and sphere
    weights = mods['W'][:, :, 0]
    sphere = mods['S'][:mods['num_pcs'], :]

    # Cleanup
    if cleanup:
        # Remove the data file
        if os.path.isfile(data_file):
            os.remove(data_file)
        # Remove temp directory if we created it
        if tmp_created:
            try:
                shutil.rmtree(outdir)
            except OSError:
                logger.warning("Could not remove temporary directory: %s", outdir)

    return weights, sphere, mods
