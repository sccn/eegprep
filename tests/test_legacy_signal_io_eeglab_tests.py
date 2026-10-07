from __future__ import annotations

import logging
import struct
from pathlib import Path

import numpy as np
import pytest
import matplotlib.pyplot as plt
from scipy.signal import freqz

from eegprep import blockave, eegfilt, env, loadeeg, loadtxt, movav, readneurodat
from tests.eeglab_tests import assert_matlab_near, eeglab_test


SIGPROC = "unittesting_sigprocfunc"


@eeglab_test(f"{SIGPROC}/blockave/sigprocfunc_blockave_wrapperTest.m", "test_pass_equal_weights")
def test_reference_blockave(eeglab_backend):
    data = np.array([[-3, 0.5, 1, -3.9, 0.5, -1.1], [2, 2, 3, -1.5, 4, 4]], dtype=float)
    actual = eeglab_backend("blockave", data, 2.0)
    # The source uses exact isequal, not the near helper, for this case.
    np.testing.assert_array_equal(actual, [[-0.5, -1.5], [3.0, 1.5]])


@eeglab_test(f"{SIGPROC}/env/sigprocfunc_env_wrapperTest.m", "test_pass_general")
@eeglab_test(f"{SIGPROC}/env/sigprocfunc_env_wrapperTest.m", "test_pass_interpolate")
def test_reference_env(eeglab_backend):
    data = np.array([[2, 1, 3], [8, 2, 6], [4, 7, 3], [0, 1, -3], [-5, 3, 1]], dtype=float)
    expected = np.array([[8, 7, 6], [-5, 1, -3]], dtype=float)
    assert_matlab_near(eeglab_backend("env", data), expected)
    interpolated = eeglab_backend("env", data, np.array([[1.0, 3.0]]), np.arange(1, 3.5, 0.5)[None, :])
    assert_matlab_near(interpolated[:, [0, 2, 4]], expected)
    assert np.all(interpolated[0] >= interpolated[1])


@eeglab_test(f"{SIGPROC}/movav/sigprocfunc_movav_wrapperTest.m", "test_pass_data_column_vector")
@eeglab_test(f"{SIGPROC}/movav/sigprocfunc_movav_wrapperTest.m", "test_pass_five_frames")
@eeglab_test(f"{SIGPROC}/movav/sigprocfunc_movav_wrapperTest.m", "test_pass_four_frames")
@eeglab_test(f"{SIGPROC}/movav/sigprocfunc_movav_wrapperTest.m", "test_pass_general")
@eeglab_test(f"{SIGPROC}/movav/sigprocfunc_movav_wrapperTest.m", "test_pass_i_empty")
@eeglab_test(f"{SIGPROC}/movav/sigprocfunc_movav_wrapperTest.m", "test_pass_nonorm_one")
@eeglab_test(f"{SIGPROC}/movav/sigprocfunc_movav_wrapperTest.m", "test_pass_xwidth_high")
@eeglab_test(f"{SIGPROC}/movav/sigprocfunc_movav_wrapperTest.m", "test_pass_xwidth_low")
@eeglab_test(f"{SIGPROC}/movav/sigprocfunc_movav_wrapperTest.m", "test_pass_xwin_column")
@eeglab_test(f"{SIGPROC}/movav/sigprocfunc_movav_wrapperTest.m", "test_pass_xwin_near_zero")
def test_reference_movav(eeglab_backend):
    data = np.array([[1, 2, 5, 3, 2, -1], np.arange(1, 7), [-2, 0, 4, -6, 3, 1]], dtype=float)
    empty = np.empty((0, 0))
    expected_x = np.arange(1.0, 6.0)[None, :] + 0.625
    cases = [
        (np.array([[1], [2], [3], [0], [5], [-6]], dtype=float), (), [[1.5, 2.5, 1.5, 2.5, -0.5]], expected_x),
        (data[:, :5], (), data[:, :5], np.arange(1.0, 6.0)[None, :] + 0.5),
        (data[:, :4], (), data[:, :4], np.arange(1.0, 5.0)[None, :] + 0.375),
        (data, (), [[1.5, 3.5, 4, 2.5, 0.5], [1.5, 2.5, 3.5, 4.5, 5.5], [-1, 2, -1, -1.5, 2]], expected_x),
        (
            data,
            (np.array([[4, 5, 6, 6, 6, 6]], dtype=float), 0.0, 0.0, 1.0, 6.0),
            [[0, 0, 1, 1.5, 2.2], [0, 0, 1, 1.5, 4], [0, 0, -2, -1, 0.4]],
            expected_x,
        ),
        (
            data,
            (0.0, 0.0, 0.0, empty, empty, 0.0, 1.0),
            [[3, 7, 8, 5, 1], [3, 5, 7, 9, 11], [-2, 4, -2, -3, 4]],
            expected_x,
        ),
        (data, (0.0, 7.0), [[2.0], [3.5], [0.0]], [[4.5]]),
        (
            np.array(
                [[1, 2, 5, 3, 2, -1, 0, 1, 2, -20], np.arange(1, 11), [-2, 0, 4, -6, 3, 1, -32, 5, 7, 18]], dtype=float
            ),
            (0.0, 2.0),
            [
                [1.5, 3.5, 4, 2.5, 0.5, -0.5, 0.5, 1.5, -9],
                [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
                [-1, 2, -1, -1.5, 2, -15.5, -13.5, 6, 12.5],
            ],
            np.arange(2.0, 11.0)[None, :],
        ),
        (
            np.arange(1.0, 7.0)[None, :],
            (0.0, 0.0, 0.0, empty, empty, np.array([[3.0], [1.0]])),
            np.array([[5.0, 9.0, 13.0, 17.0, 21.0]]) / 4,
            expected_x,
        ),
        (
            np.arange(1.0, 7.0)[None, :],
            (0.0, 0.0, 0.0, empty, empty, np.array([[3.0], [-3.0]])),
            np.full((1, 5), -3.0),
            expected_x,
        ),
    ]
    for values, options, expected_data, times in cases:
        actual_data, actual_times = eeglab_backend("movav", values, *options, nargout=2)
        assert_matlab_near(actual_data, expected_data)
        assert_matlab_near(actual_times, times)


@eeglab_test(f"{SIGPROC}/parsetxt/sigprocfunc_parsetxt_wrapperTest.m", "test_pass_custom_delim")
@eeglab_test(f"{SIGPROC}/parsetxt/sigprocfunc_parsetxt_wrapperTest.m", "test_pass_general")
@eeglab_test(f"{SIGPROC}/parsetxt/sigprocfunc_parsetxt_wrapperTest.m", "test_pass_only_delim")
def test_reference_parsetxt(eeglab_backend):
    actual = eeglab_backend("parsetxt", '1:3.c : "a"..}::{ ', '.:')
    np.testing.assert_array_equal(actual, np.array([["1", "3", "c ", ' "a"', "}", "{ "]], dtype=object))
    actual = eeglab_backend("parsetxt", ' Hello, , my  dear "friend".It\'s great! ')
    np.testing.assert_array_equal(
        actual, np.array([["Hello", "my", "dear", "friend", ".It", "s", "great!"]], dtype=object)
    )
    actual = eeglab_backend("parsetxt", ' , " \' \t,\', ')
    assert actual.shape == (0, 0)
    assert actual.dtype == object


def test_python_regression_blockave_equal_weights_matches_current_suite() -> None:
    data = np.asarray([[-3, 0.5, 1, -3.9, 0.5, -1.1], [2, 2, 3, -1.5, 4, 4]])

    result = blockave(data, 2)

    np.testing.assert_allclose(result, [[-0.5, -1.5], [3, 1.5]])
    np.testing.assert_allclose(blockave(data, 2, epochs=[1, 3], weights=[1, 100, 3]), [[-0.375, -0.7], [3.5, 3.5]])


@eeglab_test(f"{SIGPROC}/eegfilt/sigprocfunc_eegfilt_wrapperTest.m", "test_pass_general")
@pytest.mark.gui
def test_reference_eegfilt(eeglab_backend, request):
    data = np.zeros((2, 48))
    data[0, 0] = 1
    data[1, ::2] = np.arange(1.0, 25.0)
    filtered = eeglab_backend("eegfilt", data, 1.0, 0.17, 0.4)
    times = np.arange(1.0, 49.0)[None, :]
    matlab = request.config.getoption("--eeglab-backend") == "matlab"
    try:
        for channel in (0, 1):
            signals = np.vstack((data[channel], filtered[channel]))
            if matlab:
                if channel:
                    eeglab_backend("figure", nargout=0)
                eeglab_backend("plot", times, signals, nargout=0)
            else:
                if channel:
                    plt.figure()
                plt.plot(times.ravel(), signals.T)
    finally:
        if matlab:
            eeglab_backend("close", "all", nargout=0)
        else:
            plt.close("all")


def test_python_regression_eegfilt_general_case_designs_and_applies_legacy_bandpass() -> None:
    data = np.asarray(
        [
            [1, *([0] * 47)],
            [value for number in range(1, 25) for value in (number, 0)],
        ],
        dtype=float,
    )

    filtered, coefficients = eegfilt(data, 1, 0.17, 0.4)

    assert filtered.shape == data.shape
    expected_coefficients = np.asarray(
        [
            -0.013087039375293805,
            -0.04027022359539272,
            0.07703267043672658,
            0.033350774093545726,
            0.03718929923999598,
            -0.03170244161332772,
            -0.3677328893223719,
            0.31398418218061297,
        ]
    )
    expected_coefficients = np.r_[expected_coefficients, expected_coefficients[::-1]]
    np.testing.assert_allclose(coefficients, expected_coefficients, rtol=1e-12, atol=1e-12)
    frequencies, response = freqz(coefficients, worN=8192, fs=1)
    pass_gain = abs(response[np.argmin(abs(frequencies - 0.25))])
    stop_gain = abs(response[np.argmin(abs(frequencies - 0.05))])
    assert pass_gain > 5 * stop_gain
    assert np.all(np.isfinite(filtered))
    assert not np.allclose(filtered, data)


def test_python_regression_env_interpolation_preserves_current_suite_anchors_and_envelope_order() -> None:
    data = np.asarray([[2, 1, 3], [8, 2, 6], [4, 7, 3], [0, 1, -3], [-5, 3, 1]])

    result = env(data, [1, 3], np.arange(1, 3.1, 0.5))

    expected = np.asarray([[8, 8.05305535, 7, 6.56306299, 6], [-5, -1.8174217, 1, -0.32742934, -3]])
    np.testing.assert_allclose(result, expected, atol=5e-9)
    assert np.all(result[0] >= result[1])


def _source_loadtxt(eeglab_backend, eeglab_suite_root, filename, *options):
    source = eeglab_suite_root / "unittesting_sigprocfunc/loadtxt" / filename
    return eeglab_backend("loadtxt", str(source), *options)


@eeglab_test(f"{SIGPROC}/loadtxt/sigprocfunc_loadtxt_wrapperTest.m", "test_pass_convert_force")
def test_upstream_loadtxt_force(eeglab_backend, eeglab_suite_root):
    result = _source_loadtxt(eeglab_backend, eeglab_suite_root, "convert_force.txt", "convert", "force")
    expected = np.arange(1.0, 31.0).reshape(5, 6).reshape(1, -1, order="F")
    assert_matlab_near(result, expected)


@eeglab_test(f"{SIGPROC}/loadtxt/sigprocfunc_loadtxt_wrapperTest.m", "test_pass_convert_off")
def test_upstream_loadtxt_conversion_off(eeglab_backend, eeglab_suite_root):
    result = _source_loadtxt(eeglab_backend, eeglab_suite_root, "convert_off.txt", "convert", "off")
    np.testing.assert_array_equal(result, np.arange(1, 31).astype(str).reshape(5, 6))


@eeglab_test(f"{SIGPROC}/loadtxt/sigprocfunc_loadtxt_wrapperTest.m", "test_pass_general")
def test_upstream_loadtxt_general(eeglab_backend, eeglab_suite_root):
    result = _source_loadtxt(eeglab_backend, eeglab_suite_root, "general.txt")
    np.testing.assert_array_equal(result, np.arange(1.0, 31.0).reshape(5, 6))


@eeglab_test(f"{SIGPROC}/loadtxt/sigprocfunc_loadtxt_wrapperTest.m", "test_pass_negative_skipline")
def test_upstream_loadtxt_negative_skipline(eeglab_backend, eeglab_suite_root):
    result = _source_loadtxt(eeglab_backend, eeglab_suite_root, "negative_skipline.txt", "skipline", -4.0)
    np.testing.assert_array_equal(result, np.arange(1.0, 31.0).reshape(5, 6))


@eeglab_test(f"{SIGPROC}/loadtxt/sigprocfunc_loadtxt_wrapperTest.m", "test_pass_nlines")
def test_upstream_loadtxt_nlines(eeglab_backend, eeglab_suite_root):
    result = _source_loadtxt(eeglab_backend, eeglab_suite_root, "nlines.txt", "nlines", 3.0)
    np.testing.assert_array_equal(result, np.arange(1.0, 19.0).reshape(3, 6))


@eeglab_test(f"{SIGPROC}/loadtxt/sigprocfunc_loadtxt_wrapperTest.m", "test_pass_skipline")
def test_upstream_loadtxt_skipline(eeglab_backend, eeglab_suite_root):
    result = _source_loadtxt(eeglab_backend, eeglab_suite_root, "skipline.txt", "skipline", 4.0)
    np.testing.assert_array_equal(result, np.arange(1.0, 31.0).reshape(5, 6))


@eeglab_test(f"{SIGPROC}/loadtxt/sigprocfunc_loadtxt_wrapperTest.m", "test_pass_text")
def test_upstream_loadtxt_text(eeglab_backend, eeglab_suite_root):
    result = _source_loadtxt(eeglab_backend, eeglab_suite_root, "text.txt")
    expected = np.arange(1.0, 31.0).reshape(5, 6).astype(object)
    expected[0, 2] = "three"
    np.testing.assert_array_equal(result, expected)


@eeglab_test(f"{SIGPROC}/loadtxt/sigprocfunc_loadtxt_wrapperTest.m", "test_pass_verbose")
def test_upstream_loadtxt_verbose(eeglab_backend, eeglab_suite_root):
    result = _source_loadtxt(eeglab_backend, eeglab_suite_root, "verbose.txt", "verbose", "on")
    np.testing.assert_array_equal(result, np.arange(1.0, 34.0).reshape(11, 3))


@eeglab_test(f"{SIGPROC}/readneurodat/sigprocfunc_readneurodat_wrapperTest.m", "test_pass_general")
def test_upstream_readneurodat_original_file(eeglab_backend, eeglab_suite_root):
    path = eeglab_suite_root / "unittesting_sigprocfunc/readneurodat/test.dat"
    locations, labels, _theta, _phi = eeglab_backend("readneurodat", str(path), nargout=4)
    np.testing.assert_array_equal(locations["labels"], labels)


@eeglab_test(f"{SIGPROC}/readtxtfile/sigprocfunc_readtxtfile_wrapperTest.m", "test_test_readtxtfile")
def test_upstream_readtxtfile_original_location_files(eeglab_backend, eeglab_suite_root):
    for path in (
        "sample_locs/Standard-10-10-Cap33.ced",
        "sample_locs/Standard-10-20-Cap25.locs",
        "sample_data/eeglab_chan32.locs",
    ):
        eeglab_backend("readtxtfile", str(eeglab_suite_root / "eeglab" / path))


def test_readneurodat_labels_and_coordinates_match_current_suite(tmp_path: Path) -> None:
    source = tmp_path / "test.dat"
    source.write_text(
        "1 FP1 -296.703 826.087\n5 FP2 -10.989 826.087\n4 FP3 296.703 826.087\n"
        "2 F7A -450.549 704.348\n3 F3A -252.747 652.174\n",
        encoding="utf-8",
    )
    locs, labels, theta, phi = readneurodat(source)
    assert labels == ["FP1", "F7A", "F3A", "FP3", "FP2"]
    assert [loc["labels"] for loc in locs] == labels
    np.testing.assert_allclose(np.abs([loc["sph_theta_besa"] for loc in locs]), theta)
    np.testing.assert_allclose([loc["sph_phi_besa"] for loc in locs], (phi + 90) % 180 - 90)
    assert all({"X", "Y", "Z", "theta", "radius"} <= set(loc) for loc in locs)


def _write_neuroscan_eeg(
    path: Path, *, truncate_second: bool = False, dtype: str = "short"
) -> tuple[np.ndarray, np.ndarray]:
    header = bytearray(900)
    header[:8] = b"VERSION3"
    struct.pack_into("<H", header, 362, 2)
    struct.pack_into("<H", header, 368, 3)
    struct.pack_into("<H", header, 370, 2)
    struct.pack_into("<H", header, 376, 250)
    struct.pack_into("<f", header, 505, -0.1)
    struct.pack_into("<f", header, 509, 0.1)
    electrode_headers = bytearray()
    if dtype == "int32":
        struct.pack_into("<i", header, 12, 100)
        header[152] = 1
    for label, baseline, sensitivity, calibration in [("Fz", 100, 2.0, 1.0), ("Cz", 200, 4.0, 2.0)]:
        record = bytearray(75)
        record[: len(label)] = label.encode("ascii")
        struct.pack_into("<H", record, 47, baseline)
        struct.pack_into("<f", record, 59, sensitivity)
        struct.pack_into("<f", record, 71, calibration)
        electrode_headers.extend(record)
    first = np.asarray([[110, 120, 130], [220, 230, 240]])
    second = np.asarray([[140, 150, 160], [250, 260, 270]])
    raw_dtype = np.dtype("<i2" if dtype == "short" else "<i4")
    output = bytearray(header + electrode_headers)
    output.extend(struct.pack("<BHHfHH", 1, 11, 1, 350.0, 21, 0))
    output.extend(first.astype(raw_dtype).reshape(-1, order="F").tobytes())
    output.extend(struct.pack("<BHHfHH", 0, 12, 0, 450.0, 22, 0))
    second_bytes = second.astype(raw_dtype).reshape(-1, order="F").tobytes()
    output.extend(second_bytes[: len(second_bytes) // 2] if truncate_second else second_bytes)
    path.write_bytes(output)
    return first, second


def test_loadeeg_discards_incomplete_truncated_sweep(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    source = tmp_path / "bugzilla_456.eeg"
    first, _second = _write_neuroscan_eeg(source, truncate_second=True)
    with caplog.at_level(logging.WARNING, logger="eegprep.functions.sigprocfunc.loadeeg"):
        result = loadeeg(source, format="short")
    signal, accept, types, reaction_times, responses, names, points, sweeps, srate, xmin, xmax = result
    factors = np.asarray([2.0 / 204.8, 8.0 / 204.8])
    expected = (first - np.asarray([100, 200])[:, None]) * factors[:, None]
    np.testing.assert_allclose(signal, expected)
    np.testing.assert_array_equal(accept, [1])
    np.testing.assert_array_equal(types, [11])
    np.testing.assert_allclose(reaction_times, [350])
    np.testing.assert_array_equal(responses, [21])
    assert names == ["Fz", "Cz"]
    assert (points, sweeps, srate) == (3, 1, 250)
    assert xmin == pytest.approx(-0.1)
    assert xmax == pytest.approx(0.1)
    assert "incomplete data were discarded" in caplog.text


def test_loadeeg_selects_one_based_channels_and_sweep_metadata(tmp_path: Path) -> None:
    source = tmp_path / "complete.eeg"
    _first, second = _write_neuroscan_eeg(source)
    signal, accept, types, reaction_times, responses, names, _points, sweeps, *_ = loadeeg(
        source,
        chanlist=[2],
        triallist=[2],
        typerange=[12],
        accepttype=[0],
        rtrange=[400, 500],
        responsetype=[22],
        format="short",
    )
    np.testing.assert_allclose(signal, (second[[1]] - 200) * (8.0 / 204.8))
    np.testing.assert_array_equal(accept, [0])
    np.testing.assert_array_equal(types, [12])
    np.testing.assert_allclose(reaction_times, [450])
    np.testing.assert_array_equal(responses, [22])
    assert names == ["Cz"]
    assert sweeps == 1


def test_loadeeg_auto_detects_32_bit_neuroscan_samples(tmp_path: Path) -> None:
    source = tmp_path / "int32.eeg"
    first, second = _write_neuroscan_eeg(source, dtype="int32")
    signal, _accept, _types, _reaction_times, _responses, _names, _points, sweeps, *_ = loadeeg(source)
    raw = np.concatenate([first, second], axis=1)
    expected = (raw - np.asarray([100, 200])[:, None]) * np.asarray([2.0 / 204.8, 8.0 / 204.8])[:, None]
    np.testing.assert_allclose(signal, expected)
    assert sweeps == 2


def test_loadtxt_preserves_tab_and_comma_blank_cells(tmp_path: Path) -> None:
    source = tmp_path / "blanks.csv"
    source.write_text("a,,c\n1,2,\n", encoding="utf-8")
    table = loadtxt(source, delim=",", convert="off", verbose="off")
    np.testing.assert_array_equal(table, [["a", "", "c"], ["1", "2", ""]])


def test_eegfilt_filters_epochs_independently() -> None:
    first = np.zeros(80)
    second = np.zeros(80)
    first[-1] = 100
    filtered, _coefficients = eegfilt(np.r_[first, second], 100, 0, 20, epochframes=80, filtorder=20)
    assert filtered.shape == (1, 160)
    np.testing.assert_allclose(filtered[0, 80:], 0, atol=1e-12)


def test_python_regression_movav_ignores_nans_but_preserves_all_nan_windows() -> None:
    data = np.asarray([[1, np.nan, 3, 4], [np.nan, np.nan, 2, 2]])
    result, _x = movav(data, xwidth=2)
    assert result[0, 0] == pytest.approx(1)
    assert np.isnan(result[1, 0])
