"""Test-only port of the pinned IRLS validation helper, not a LIMO runtime API.

The workflow and resampling/plotting loops remain Python-owned on both backends.
Only LIMO's actual cluster primitives are dispatched to the selected backend.
The source's five-dimensional branch and indexing quirks are retained too.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import shutil
import subprocess

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import beta

from tests.eeglab_tests import load_matlab_test_fixture
from tests.eeglab_tests.backend import _decode


IRLS_SOURCE = "unittesting_limo/limo_zIRLS_validation_4_Arno.m"
IRLS_SOURCE_SHA256 = "b68aa89cec465b696c86d4c6f077c5caf1d91603e4f59b0bfc337b22a1027cb8"
IRLS_HELPER_SHA256 = "105cf817ffe6cd60009a8cb80bf5fd03e8208964fcc8d72f46629eb81513c79d"
IRLS_CHANLOCS_ASSIGNMENT = "chanlocs = fullfile(STUDY.filepath, 'derivatives', 'limo_gp_level_chanlocs.mat');\n"
IRLS_PATCH = Path(__file__).parents[1] / "matlab" / "limo_zIRLS_validation_4_Arno.source-corrections.patch"
RESULT_FIELDS = ("errIRLS", "ci_errIRLS", "maxIRLS", "maxci_errIRLS", "clusterIRLS", "maxcic_errIRLS")


def prepare_irls_source_overlay(suite_root: Path, directory: Path) -> Path:
    """Copy the pinned script/helper and apply only the approved input repairs.

    Run the corrected native script from the returned directory with a writable
    ds002718 copy beside it. The patch supplies std_limo's channel-neighbour file
    and uses pop_limo's returned model paths. The helper's file pattern matches
    the generated H0 filenames. Neither the pinned source nor any scientific
    algorithm changes.
    """
    source_directory = directory / "unittesting_limo"
    source_directory.mkdir()
    for name, digest in (
        (Path(IRLS_SOURCE).name, IRLS_SOURCE_SHA256),
        ("limo_test_glmboot.m", IRLS_HELPER_SHA256),
    ):
        source = suite_root / "unittesting_limo" / name
        if sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError(f"IRLS source differs from the pinned original: {source}")
        shutil.copy2(source, source_directory / name)
    subprocess.run(["git", "apply", "--no-index", str(IRLS_PATCH)], cwd=directory, check=True)
    return source_directory


def load_irls_mat(call, native, filename):
    """Load MAT data without dropping singleton frequency/time/bootstrap axes."""
    if native:
        return call("load", str(filename))
    return {key: _decode(value, value) for key, value in load_matlab_test_fixture(filename).items()}


def matlab_binofit(successes, trials):
    """The source's default 95% Clopper–Pearson binomial interval."""
    successes, trials = int(successes), int(trials)
    low = 0.0 if successes == 0 else beta.ppf(0.025, successes, trials - successes + 1)
    high = 1.0 if successes == trials else beta.ppf(0.975, successes + 1, trials - successes)
    return successes / trials, np.array([low, high])


def _squeeze(values):
    if values.ndim <= 2:
        return values
    values = np.squeeze(values)
    return values.reshape(-1, 1) if values.ndim < 2 else values


def _sum(values, dimension):
    return values.sum(axis=dimension - 1) if dimension <= values.ndim else values


def _matlab_max(values):
    # MATLAB max omits NaNs and returns the first MATLAB-linear maximum index;
    # for an all-NaN vector it returns NaN at the first position.
    values = values.ravel(order="F")
    valid = np.flatnonzero(~np.isnan(values))
    index = valid[np.argmax(values[valid])] if valid.size else 0
    return values[index], index


def _plot(call, native, operation, *args, **kwargs):
    if native:
        call(operation, *args, **kwargs, nargout=0)
    elif operation == "figure":
        plt.figure(num=kwargs["Name"])
    elif operation == "subplot":
        plt.subplot(*(int(value) for value in args))
    elif operation == "imagesc":
        plt.imshow(args[0], aspect="auto")
    elif operation == "errorbar":
        x, y, low, high = args
        plt.errorbar(x, y, yerr=np.array([[low], [high]]), **{key.lower(): value for key, value in kwargs.items()})
    elif operation == "hold":
        pass  # Matplotlib already retains existing artists on the current axes.
    elif operation == "grid":
        plt.grid(True)
    elif operation == "plot":
        x, y, *style = args
        x, y = np.asarray(x).ravel(), np.asarray(y)
        if y.ndim == 2 and 1 in y.shape:
            y = y.ravel()
        plt.plot(x, y, *style, **{key.lower(): value for key, value in kwargs.items()})
    else:
        getattr(plt, operation)(*args, **{key.lower(): value for key, value in kwargs.items()})


def _maxima_and_clusters(call, data, neighbours, alpha, count):
    spatial_shape = data.shape[:-2]
    maxima = np.zeros(count)
    positions = np.zeros(spatial_shape)
    for boot in range(count - 1, -1, -1):
        statistic = _squeeze(data[..., 0, boot])
        maxima[boot], index = _matlab_max(statistic)
        positions[np.unravel_index(index, spatial_shape, order="F")] += 1
    cluster_positions = np.zeros(spatial_shape)
    cluster_maxima = np.zeros(count)
    for boot in range(count):
        labels, nclusters = call("limo_findcluster", _squeeze(data[..., 1, boot]) <= alpha, neighbours, 2.0, nargout=2)
        statistic = _squeeze(data[..., 0, boot])
        nclusters = int(np.asarray(nclusters).item())
        if nclusters:
            masses = np.array([statistic[labels == label].sum() for label in range(1, nclusters + 1)])
            cluster_maxima[boot], index = _matlab_max(masses)
            label = index + 1
            cluster_positions += labels == label
    return maxima, positions, cluster_maxima, cluster_positions


def _shuffled(values, rng):
    # The original removes +Inf only, then permutes all remaining values.
    return rng.permutation(values[values != np.inf])


def _threshold(values, alpha):
    rank = int(np.floor((1 - alpha) * len(values) + 0.5))
    return np.sort(values)[rank - 1]


def _error_rates_4d(call, data, neighbours, alpha, nboot, samples, maxima, cluster_maxima, rng):
    result = [np.zeros(len(samples)) if index % 2 == 0 else np.zeros((2, len(samples))) for index in range(6)]
    null_size = data.shape[-1] - nboot
    ntests = np.prod(data.shape[:2])
    for index in range(len(samples) - 1, -1, -1):
        count = samples[index]
        errors = _sum(_squeeze(data[:, :, -1, -count:]) < alpha, 3)
        result[0][index], result[1][:, index] = matlab_binofit(errors.sum(), count * ntests)
        # The source initially allocates Nboot entries, then may assign at b>
        # Nboot. Explicit full-size zero storage preserves those expanded sums.
        max_errors, cluster_errors = np.zeros(data.shape[-1]), np.zeros(data.shape[-1])
        for boot in range(data.shape[-1] - 1, null_size - 1, -1):
            selected = _shuffled(maxima, rng)[np.arange(count)]
            mask = (_squeeze(data[:, :, 0, boot]) >= _threshold(selected, alpha)).astype(np.float32)
            if mask.sum():
                max_errors[boot] = 1
            selected = _shuffled(cluster_maxima, rng)[np.arange(count)]
            mask = call(
                "limo_cluster_test",
                _squeeze(data[:, :, 0, boot]),
                _squeeze(data[:, :, 1, boot]),
                selected.reshape(1, -1),
                neighbours,
                2.0,
                alpha,
            )
            if mask.sum():
                cluster_errors[boot] = 1
        result[2][index], result[3][:, index] = matlab_binofit(max_errors.sum(), nboot)
        result[4][index], result[5][:, index] = matlab_binofit(cluster_errors.sum(), nboot)
    return result


def _error_rates_5d(call, data, neighbours, alpha, nboot, samples, cluster_maxima, rng):
    result = [np.zeros(len(samples)) if index % 2 == 0 else np.zeros((2, len(samples))) for index in range(6)]
    ntests = np.prod(data.shape[:3])
    for index in range(len(samples) - 1, -1, -1):
        count = samples[index]
        errors = _sum(_squeeze(data[:, :, :, -1, :count]) < alpha, 4)
        result[0][index], result[1][:, index] = matlab_binofit(errors.sum(), count * ntests)
        max_errors, cluster_errors = np.zeros(data.shape[-1]), np.zeros(data.shape[-1])
        for boot in range(nboot - 1, -1, -1):
            # Literal source: its 5-D maximum correction uses cluster maxima,
            # unlike the 4-D branch. Do not silently repair this distinction.
            selected = _shuffled(np.delete(cluster_maxima, boot), rng)[:count]
            mask = _squeeze(data[:, :, :, 0, boot]) >= _threshold(selected, alpha)
            if mask.sum():
                max_errors[boot] = 1
            selected = _shuffled(np.delete(cluster_maxima, boot), rng)[:count]
            mask = call(
                "limo_cluster_test",
                _squeeze(data[:, :, :, 0, boot]),
                _squeeze(data[:, :, :, 1, boot]),
                selected.reshape(1, -1),
                neighbours,
                2.0,
                alpha,
            )
            if mask.sum():
                cluster_errors[boot] = 1
        result[2][index], result[3][:, index] = matlab_binofit(max_errors.sum(), data.shape[-1])
        result[4][index], result[5][:, index] = matlab_binofit(cluster_errors.sum(), data.shape[-1])
    return result


def reference_limo_glmboot(call, native, chanlocs, h0, *, step_size=200, Nboot=None, MinSamp=600, figure="on"):
    """Translate limo_test_glmboot.m, retaining all six outputs and default plots."""
    alpha = 0.05
    neighbours = load_irls_mat(call, native, chanlocs)["channeighbstructmat"]
    rng = np.random.default_rng()
    results = [dict() for _ in range(6)]
    errors, max_positions, cluster_positions = {}, {}, {}
    for folder in range(len(h0) - 1, -1, -1):
        files = sorted(Path(h0[folder]).glob("*H0.mat"))
        content_names = [path.name for path in files]
        file_index = 0
        for source_index, filename in enumerate(files):
            if any(part in filename.name for part in ("Betas", "tfce", "R2")):
                continue
            loaded = load_irls_mat(call, native, filename)
            (data,) = loaded.values()
            if Nboot is None:
                Nboot = data.shape[-1]
            if data.ndim == 4:
                data = data[~np.isnan(data[:, 0, 0, 0]), ...]
                null_size = data.shape[-1] - Nboot
                if null_size < MinSamp:
                    raise ValueError(f"less than {MinSamp:g} samples available to estimate the null - this is too low")
                maxima, positions, cluster_maxima, cluster_position = _maxima_and_clusters(
                    call, data, neighbours, alpha, null_size
                )
                ntests = np.prod(data.shape[:2])
                samples = np.arange(null_size, MinSamp - 1, -step_size)
                errors[folder, file_index] = _sum(_squeeze(data[:, :, -1, null_size:]) < alpha, 3)
                computed = _error_rates_4d(call, data, neighbours, alpha, Nboot, samples, maxima, cluster_maxima, rng)
            elif data.ndim == 5:
                data = data[~np.isnan(data[:, 0, 0, 0, 0]), ...]
                ntests = np.prod(data.shape[:3])
                samples = np.arange(data.shape[-1], 199, -step_size)
                maxima, positions, cluster_maxima, cluster_position = _maxima_and_clusters(
                    call, data, neighbours, alpha, data.shape[-1]
                )
                errors[folder, file_index] = _sum(_squeeze(data[:, :, :, -1, :]) < alpha, 3)
                computed = _error_rates_5d(call, data, neighbours, alpha, Nboot, samples, cluster_maxima, rng)
            else:
                raise ValueError("null data files are expected to be 4 or 5 dimensionals")
            max_positions[folder, file_index], cluster_positions[folder, file_index] = positions, cluster_position
            for target, value in zip(results, computed, strict=True):
                target[folder, file_index] = value
            if folder == 0:
                content_names[source_index] = filename.name.replace("_", " ")
            file_index += 1
    nfiles = max(index for _, index in errors) + 1
    if figure.lower() == "on":
        _plot_results(
            call,
            native,
            results,
            errors,
            max_positions,
            cluster_positions,
            content_names,
            len(h0),
            nfiles,
            ntests,
            samples,
        )
    outputs = []
    for values in results:
        cells = np.empty((len(h0), nfiles), dtype=object)
        for index in np.ndindex(cells.shape):
            value = values.get(index, np.empty((0, 0)))
            cells[index] = value.reshape(1, -1) if value.ndim == 1 else value
        outputs.append(cells)
    return tuple(outputs)


def _plot_results(call, native, results, errors, maxima, clusters, names, nsubjects, nfiles, ntests, samples):
    for file_index in range(nfiles):
        name = f"Error bias and rate {names[2 + file_index][:-4]}"
        _plot(call, native, "figure", Name=name)
        for kind, positions in enumerate((errors, maxima, clusters)):
            _plot(call, native, "subplot", 3.0, 3.0, float(kind + 1))
            values = positions[0, file_index].copy()
            for folder in range(1, nsubjects):
                if (folder, file_index) in errors:
                    values += positions[folder, file_index]
            _plot(call, native, "imagesc", values if values.ndim == 2 else _squeeze(values.mean(axis=1)))
            _plot(
                call,
                native,
                "title",
                ("Cell-wise Error density bias", "Max value Error density bias", "Cluster max Error density bias")[
                    kind
                ],
            )
            if kind == 0:
                _plot(call, native, "ylabel", "channels")
        colors = call("limo_color_images", float(nsubjects))
        for kind in range(3):
            _plot(call, native, "subplot", 3.0, 3.0, float(4 + kind))
            _plot(call, native, "hold", "on")
            average, interval = results[2 * kind : 2 * kind + 2]
            for folder in range(nsubjects - 1, -1, -1):
                if (folder, file_index) in errors:
                    value, ci = average[folder, file_index][-1], interval[folder, file_index][:, -1]
                    _plot(
                        call,
                        native,
                        "errorbar",
                        float(folder + 1),
                        value,
                        value - ci[0],
                        ci[1] - value,
                        LineWidth=2.0,
                        Color=colors[folder, :],
                    )
            x = np.arange(nsubjects + 1, dtype=float).reshape(1, -1)
            _plot(call, native, "plot", x, np.full_like(x, 0.05), "k")
            _, ci = matlab_binofit(np.floor(ntests * 0.05 + 0.5), ntests)
            _plot(call, native, "plot", x, np.full_like(x, ci[0]), "k--")
            # The source's single replication argument creates a square matrix.
            _plot(call, native, "plot", x, np.full((nsubjects + 1, nsubjects + 1), ci[1]), "k--")
            _plot(call, native, "grid", "on")
            _plot(
                call,
                native,
                "title",
                (
                    "Average cell-wise Type 1 Error",
                    "Type 1 FWER with max value correction",
                    "Type 1 FWER with cluster mass correction",
                )[kind],
            )
            if kind == 0:
                _plot(call, native, "ylabel", "subject's type 1 error")
            _plot(call, native, "subplot", 3.0, 3.0, float(7 + kind))
            low = np.zeros(len(results[0][0, 0]))
            high = low.copy()
            for folder in range(nsubjects - 1, -1, -1):
                if (folder, file_index) in errors:
                    _plot(
                        call,
                        native,
                        "plot",
                        samples.reshape(1, -1),
                        average[folder, file_index].reshape(1, -1),
                        "--",
                        LineWidth=1.0,
                    )
                    _plot(call, native, "hold", "on")
                    low += interval[folder, file_index][0, :]
                    high += interval[folder, file_index][1, :]
                    if kind == 0:
                        _plot(call, native, "ylabel", "type 1 error")
            for values in (low, high):
                _plot(
                    call,
                    native,
                    "plot",
                    samples.reshape(1, -1),
                    (values / (nsubjects * nfiles)).reshape(1, -1),
                    "k",
                    LineWidth=2.0,
                )
            _plot(call, native, "grid", "on")
            _plot(call, native, "title", "Sample size tested" if kind == 0 else "Convergence rate per null sample size")
        if native:
            handle = call("eegprep_test_gui_handle", "gcf")
            call("saveas", handle, name, "fig", nargout=0)
        else:
            # Preserve the requested artifact. Unsupported .fig is a real
            # Python graphics capability gap, not permission to replace it.
            plt.savefig(f"{name}.fig", format="fig")
