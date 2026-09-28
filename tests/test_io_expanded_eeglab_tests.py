"""Python-owned translations of the real-fixture native I/O expansion."""

from copy import deepcopy
from importlib import import_module

import h5py
import numpy as np
import pytest
from scipy.io import loadmat, savemat

from tests.eeglab_tests import expanded_matlab_test
from tests.eeglab_tests.backend import _load_matlab_variable


NATIVE_SOURCE = "tests/matlab/expanded/test_eegprep_io_expanded.m"
NATIVE_SHA256 = "203d5bb794a9160e2ccb4ff89881d878b164686fb9b18f9b2988387a8fded4a8"
BVA_LABELS = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T7",
    "T8",
    "P7",
    "P8",
    "Fz",
    "Cz",
    "Pz",
    "FC1",
    "FC2",
    "CP1",
    "CP2",
    "FC5",
    "FC6",
    "CP5",
    "CP6",
    "TP9",
    "TP10",
    "Eog",
    "Ekg1",
    "Ekg2",
]


def _native(request):
    return request.config.getoption("--eeglab-backend") == "matlab"


@pytest.fixture(autouse=True)
def _io_options(request, eeglab_options_directory):
    # Match the three explicit preconditions in native setupOnce. Python's
    # two-file default differs; isolate this test setup with the shared fixture.
    if _native(request):
        engine = request.getfixturevalue("eeglab_matlab_engine")
        engine.eval(
            "eeglab_options; assert(option_scaleicarms == 1); "
            "assert(option_storedisk == 0); assert(option_savetwofiles == 1);",
            nargout=0,
        )
    else:
        options = import_module("eegprep.functions.adminfunc.eeg_options").EEG_OPTIONS
        options.update(option_scaleicarms=1, option_storedisk=0, option_savetwofiles=1)


@pytest.fixture
def original_epochs(request, eeglab_suite_root):
    directory = eeglab_suite_root / "eeglab" / "sample_data"
    filename = directory / "eeglab_data_epochs_ica.set"
    if _native(request):
        source = _load_matlab_variable(filename, "EEG")
    else:
        source = loadmat(filename, simplify_cells=True)["EEG"]
        # MATLAB on-disk pointers are one-based; EEGPrep's in-memory pointers
        # are zero-based. Latencies and public channel options stay one-based.
        source["icachansind"] = np.asarray(source["icachansind"], dtype=int) - 1
        for location in source["chanlocs"]:
            location["urchan"] -= 1
        for event in source["event"]:
            event["urevent"] -= 1
        for epoch in source["epoch"]:
            epoch["event"] = np.asarray(epoch["event"]) - 1
    shape = tuple(int(np.asarray(source[key]).item()) for key in ("nbchan", "pnts", "trials"))
    source["data"] = np.fromfile(directory / source["data"], dtype="<f4").reshape(shape, order="F")
    return source


def _records(value):
    if isinstance(value, dict):
        return [value]
    return list(value.ravel(order="F")) if isinstance(value, np.ndarray) else value


def _fields(records, name):
    return [np.asarray(record[name]).item() for record in _records(records)]


def _equal_data(actual, expected):
    assert np.asarray(actual).dtype == np.asarray(expected).dtype
    np.testing.assert_array_equal(actual, expected)


def _metadata(actual, expected, *, disk=False, python=False):
    for field, value in zip(("nbchan", "pnts", "trials", "srate", "xmin", "xmax"), (32, 384, 80, 128, -1, 255 / 128)):
        assert np.asarray(actual[field]).item() == value
    np.testing.assert_array_equal(np.asarray(actual["times"]).ravel(), np.asarray(expected["times"]).ravel())
    assert _fields(actual["chanlocs"], "labels") == _fields(expected["chanlocs"], "labels")
    for field in ("latency", "type"):
        assert _fields(actual["event"], field) == _fields(expected["event"], field)
    pointers = np.asarray(_fields(expected["event"], "urevent")) + int(disk and python)
    np.testing.assert_array_equal(_fields(actual["event"], "urevent"), pointers)
    _equal_data(actual["icasphere"], expected["icasphere"])
    # Independent normalization/reconstruction invariants, not only a writer
    # return-versus-disk check. The same fixed double-roundoff bound is native.
    np.testing.assert_allclose(np.sqrt(np.mean(actual["icawinv"] ** 2, axis=0)), np.ones(32), rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        actual["icawinv"] @ (actual["icaweights"] @ actual["icasphere"]),
        expected["icawinv"] @ (expected["icaweights"] @ expected["icasphere"]),
        rtol=0,
        atol=1e-12,
    )


def _read_set(filename):
    """Inspect persisted fields independently of EEGPrep's dataset loaders."""
    if not h5py.is_hdf5(filename):
        disk = loadmat(filename, simplify_cells=True)
        return disk["EEG"] if "EEG" in disk else disk
    # MATLAB v7.3's reversed dimensions and object references are read directly
    # for exactly the numeric/char/struct fields asserted by this native suite.
    with h5py.File(filename) as file:
        group = file["EEG"] if "EEG" in file else file

        def value(dataset):
            array = dataset[()].T
            if dataset.attrs.get("MATLAB_class") == b"char":
                return "".join(chr(int(code)) for code in array.ravel(order="F"))
            return array

        fields = (
            "data",
            "datfile",
            "setname",
            "nbchan",
            "pnts",
            "trials",
            "srate",
            "xmin",
            "xmax",
            "times",
            "icasphere",
            "icaweights",
            "icawinv",
        )
        disk = {key: value(group[key]) for key in fields}
        for key, names in (("chanlocs", ("labels",)), ("event", ("latency", "type", "urevent"))):
            references = {name: group[key][name][()].ravel() for name in names}
            disk[key] = [
                {name: value(file[references[name][index]]) for name in names}
                for index in range(len(references[names[0]]))
            ]
        return disk


@expanded_matlab_test(NATIVE_SOURCE, "testBrainVisionMultiplexedCalibrationAndSubset", NATIVE_SHA256)
def test_brainvision_multiplexed_calibration_and_subset(eeglab_backend, eeglab_suite_root, eeglab_working_directory):
    _brainvision(eeglab_backend, eeglab_suite_root, "multiplexed")


@expanded_matlab_test(NATIVE_SOURCE, "testBrainVisionVectorizedCalibrationAndSubset", NATIVE_SHA256)
def test_brainvision_vectorized_calibration_and_subset(eeglab_backend, eeglab_suite_root, eeglab_working_directory):
    _brainvision(eeglab_backend, eeglab_suite_root, "vectorized")


def _brainvision(call, suite, orientation):
    directory = suite / "unittesting_binary" / "testfiles" / "BVA"
    stem = f"brainvision_genericdataformat_binary{orientation}_int16"
    codes = np.fromfile(directory / f"{stem}.dat", dtype="<i2")
    assert codes.size == 32 * 2112
    codes = codes.reshape((32, 2112), order="F") if orientation == "multiplexed" else codes.reshape((32, 2112))
    # MATLAB single times a double scalar rounds after the double product.
    expected = (codes.astype(np.float64) * 0.00045777764).astype(np.float32)
    eeg = call("pop_loadbv", str(directory), f"{stem}.vhdr")
    _equal_data(eeg["data"], expected)
    assert [np.asarray(eeg[key]).item() for key in ("nbchan", "pnts", "trials", "srate")] == [32, 2112, 1, 200]
    assert [np.asarray(eeg[key]).item() for key in ("xmin", "xmax")] == [0, 2111 / 200]
    np.testing.assert_array_equal(np.asarray(eeg["times"]).ravel(), np.arange(2112) * 5)
    assert _fields(eeg["chanlocs"], "labels") == BVA_LABELS
    stimuli = [event for event in _records(eeg["event"]) if event["code"] == "Stimulus"]
    assert _fields(stimuli, "latency") == [
        108,
        265,
        282,
        455,
        629,
        803,
        811,
        977,
        1151,
        1325,
        1357,
        1499,
        1673,
        1847,
        1903,
        2021,
    ]
    assert _fields(stimuli, "type") == [
        "S  4",
        "S  1",
        "S  4",
        "S  4",
        "S  4",
        "S  4",
        "S  2",
        "S  4",
        "S  4",
        "S  4",
        "S  3",
        "S  4",
        "S  4",
        "S  4",
        "S  1",
        "S  4",
    ]
    subset = call(
        "pop_loadbv", str(directory), f"{stem}.vhdr", np.array([[108.0, 811.0]]), np.array([[32.0, 1.0, 17.0]])
    )
    _equal_data(subset["data"], expected[[31, 0, 16], 107:811])
    assert [np.asarray(subset[key]).item() for key in ("nbchan", "pnts", "trials", "srate")] == [3, 704, 1, 200]
    np.testing.assert_array_equal(np.asarray(subset["times"]).ravel(), np.arange(704) * 5)
    assert _fields(subset["chanlocs"], "labels") == [BVA_LABELS[index] for index in (31, 0, 16)]
    assert _fields(subset["event"], "latency") == [1, 158, 175, 348, 522, 696, 704]
    assert _fields(subset["event"], "type") == ["S  4", "S  1", "S  4", "S  4", "S  4", "S  4", "S  2"]
    assert _fields(subset["urevent"], "latency") == _fields(subset["event"], "latency")


@expanded_matlab_test(NATIVE_SOURCE, "testImportFloat32LittleEndianEpochs", NATIVE_SHA256)
def test_import_float32_little_endian_epochs(eeglab_backend, original_epochs, eeglab_working_directory):
    _import(eeglab_backend, original_epochs, eeglab_working_directory, "float32le")


@expanded_matlab_test(NATIVE_SOURCE, "testImportFloat32BigEndianEpochs", NATIVE_SHA256)
def test_import_float32_big_endian_epochs(eeglab_backend, original_epochs, eeglab_working_directory):
    _import(eeglab_backend, original_epochs, eeglab_working_directory, "float32be")


@expanded_matlab_test(NATIVE_SOURCE, "testImportMatlabEpochs", NATIVE_SHA256)
def test_import_matlab_epochs(eeglab_backend, original_epochs, eeglab_working_directory):
    _import(eeglab_backend, original_epochs, eeglab_working_directory, "matlab")


@expanded_matlab_test(NATIVE_SOURCE, "testImportTransposedAsciiEpochs", NATIVE_SHA256)
def test_import_transposed_ascii_epochs(eeglab_backend, original_epochs, eeglab_working_directory):
    _import(eeglab_backend, original_epochs, eeglab_working_directory, "ascii")


def _import(call, source, output, format):
    filename = output / f"epochs-{format}.dat"
    if format == "matlab":
        savemat(filename, {"data": source["data"]}, appendmat=False)
    elif format == "ascii":
        np.savetxt(filename, source["data"].reshape(32, -1, order="F").T.astype(float), delimiter="\t", fmt="%.17g")
    else:
        source["data"].ravel(order="F").astype("<f4" if format == "float32le" else ">f4").tofile(filename)
    eeg = call(
        "pop_importdata",
        "dataformat",
        format,
        "nbchan",
        32.0,
        "data",
        str(filename),
        "pnts",
        384.0,
        "srate",
        128.0,
        "xmin",
        -1.0,
        "chanlocs",
        source["chanlocs"],
        "setname",
        "Real epoch import",
        "subject",
        "S01",
        "session",
        2.0,
        "condition",
        "targets",
        "group",
        "Control",
        "ref",
        "average",
        "comments",
        "Original EEGLAB sample epochs; no sample reduction",
    )
    np.testing.assert_array_equal(np.asarray(eeg["data"], dtype=float), source["data"].astype(float))
    assert [np.asarray(eeg[key]).item() for key in ("nbchan", "pnts", "trials", "srate")] == [32, 384, 80, 128]
    assert [np.asarray(eeg[key]).item() for key in ("xmin", "xmax")] == [-1, 255 / 128]
    np.testing.assert_array_equal(np.asarray(eeg["times"]).ravel(), np.arange(-128, 256) * 1000 / 128)
    assert _fields(eeg["chanlocs"], "labels") == _fields(source["chanlocs"], "labels")
    for key, expected in {
        "setname": "Real epoch import",
        "subject": "S01",
        "session": 2,
        "condition": "targets",
        "group": "Control",
        "ref": "average",
        "comments": "Original EEGLAB sample epochs; no sample reduction",
    }.items():
        assert np.asarray(eeg[key]).item() == expected


@expanded_matlab_test(NATIVE_SOURCE, "testSingleFileSaveVersionsAndReload", NATIVE_SHA256)
def test_single_file_save_versions_and_reload(eeglab_backend, original_epochs, eeglab_working_directory, request):
    source, output, call = original_epochs, eeglab_working_directory, eeglab_backend
    for version in ("6", "7", "7.3"):
        filename = f"inline-{version}.set"
        saved = call(
            "pop_saveset",
            deepcopy(source),
            "filename",
            filename,
            "filepath",
            str(output),
            "savemode",
            "onefile",
            "version",
            version,
            "check",
            "on",
        )
        assert h5py.is_hdf5(output / filename) == (version == "7.3")
        disk = _read_set(output / filename)
        _equal_data(disk["data"], source["data"])
        _equal_data(disk["icaweights"], saved["icaweights"])
        _equal_data(disk["icawinv"], saved["icawinv"])
        _metadata(disk, source, disk=True, python=not _native(request))
        assert not (output / f"inline-{version}.fdt").exists()
        assert saved["filename"] == filename
        loaded = call("pop_loadset", "filename", filename, "filepath", str(output))
        _equal_data(loaded["data"], source["data"])
        _metadata(loaded, source)
        reloaded = call("pop_loadset", "eeg", loaded)
        _equal_data(reloaded["data"], source["data"])
        _metadata(reloaded, source)


@expanded_matlab_test(NATIVE_SOURCE, "testTwoFileInfoChannelLoadAndMetadataResave", NATIVE_SHA256)
def test_two_file_info_channel_load_and_metadata_resave(
    eeglab_backend, original_epochs, eeglab_working_directory, request
):
    source, output, call = original_epochs, eeglab_working_directory, eeglab_backend
    saved = call(
        "pop_saveset",
        deepcopy(source),
        "filename",
        "external.set",
        "filepath",
        str(output),
        "savemode",
        "twofiles",
        "version",
        "7",
    )
    file = output / "external.set"
    disk = _read_set(file)
    assert disk["data"] == disk["datfile"] == "external.fdt"
    _equal_data(disk["icaweights"], saved["icaweights"])
    _equal_data(disk["icawinv"], saved["icawinv"])
    _metadata(disk, source, disk=True, python=not _native(request))
    before = np.fromfile(output / "external.fdt", dtype="<f4")
    _equal_data(before, source["data"].ravel(order="F"))
    info = call("pop_loadset", "filename", "external.set", "filepath", str(output), "loadmode", "info")
    assert info["data"] == "external.fdt"
    _metadata(info, source)
    subset = call(
        "pop_loadset", "filename", "external.set", "filepath", str(output), "loadmode", np.array([[1.0, 17.0, 32.0]])
    )
    _equal_data(subset["data"], source["data"][[0, 16, 31]])
    assert np.asarray(subset["nbchan"]).item() == 3
    assert _fields(subset["chanlocs"], "labels") == [
        _fields(source["chanlocs"], "labels")[index] for index in (0, 16, 31)
    ]
    for field in ("icaweights", "icasphere", "icawinv"):
        assert np.asarray(subset[field]).size == 0
    info["setname"], info["saved"] = "Metadata-only resave", "no"
    call("pop_saveset", info, "savemode", "resave", "version", "7", nargout=0)
    disk = _read_set(file)
    assert disk["setname"] == "Metadata-only resave"
    assert disk["data"] == "external.fdt"
    _equal_data(np.fromfile(output / "external.fdt", dtype="<f4"), before)
    loaded = call("pop_loadset", "filename", "external.set", "filepath", str(output))
    _equal_data(loaded["data"], source["data"])
    _metadata(loaded, source)


@expanded_matlab_test(NATIVE_SOURCE, "testNonmonotonicChannelLoadPreservesDataLabelOrder", NATIVE_SHA256)
def test_nonmonotonic_channel_load_preserves_data_label_order(
    eeglab_backend, original_epochs, eeglab_working_directory
):
    source, output, call = original_epochs, eeglab_working_directory, eeglab_backend
    call(
        "pop_saveset",
        deepcopy(source),
        "filename",
        "reordered.set",
        "filepath",
        str(output),
        "savemode",
        "twofiles",
        "version",
        "7",
        nargout=0,
    )
    subset = call(
        "pop_loadset", "filename", "reordered.set", "filepath", str(output), "loadmode", np.array([[32.0, 1.0, 17.0]])
    )
    _equal_data(subset["data"], source["data"][[31, 0, 16]])
    assert _fields(subset["chanlocs"], "labels") == [
        _fields(source["chanlocs"], "labels")[index] for index in (31, 0, 16)
    ]
    assert [np.asarray(subset[key]).item() for key in ("nbchan", "pnts", "trials")] == [3, 384, 80]


@expanded_matlab_test(NATIVE_SOURCE, "testMultipleDatasetLoadAndResave", NATIVE_SHA256)
def test_multiple_dataset_load_and_resave(eeglab_backend, original_epochs, eeglab_working_directory, request):
    source, output, call = original_epochs, eeglab_working_directory, eeglab_backend
    for name in ("first", "second"):
        source["setname"] = name
        call(
            "pop_saveset",
            deepcopy(source),
            "filename",
            f"{name}.set",
            "filepath",
            str(output),
            "savemode",
            "twofiles",
            "version",
            "7",
            nargout=0,
        )
    filenames = ["first.set", "second.set"]
    if _native(request):
        filenames = np.array([filenames], dtype=object)
    eeg = call("pop_loadset", "filename", filenames, "filepath", str(output))
    assert np.asarray(eeg, dtype=object).shape == ((1, 2) if _native(request) else (2,))
    assert _fields(eeg, "setname") == ["first", "second"]
    for index, dataset in enumerate(_records(eeg), 1):
        _equal_data(dataset["data"], source["data"])
        _metadata(dataset, source)
        dataset["setname"], dataset["saved"] = f"Updated {index}", "no"
    call("pop_saveset", eeg, "savemode", "resave", nargout=0)
    for index, dataset in enumerate(_records(eeg), 1):
        disk = _read_set(output / dataset["filename"])
        assert disk["setname"] == f"Updated {index}"
        _equal_data(disk["icaweights"], dataset["icaweights"])
        _equal_data(disk["icawinv"], dataset["icawinv"])
        _metadata(disk, source, disk=True, python=not _native(request))
        _equal_data(np.fromfile(output / disk["datfile"], dtype="<f4"), source["data"].ravel(order="F"))


@expanded_matlab_test(NATIVE_SOURCE, "testExportAllEpochSamplesCsv", NATIVE_SHA256)
def test_export_all_epoch_samples_csv(eeglab_backend, original_epochs, eeglab_working_directory):
    source = original_epochs
    filename = eeglab_working_directory / "epochs.csv"
    eeglab_backend(
        "pop_export",
        source,
        str(filename),
        "transpose",
        "on",
        "separator",
        ",",
        "precision",
        9.0,
        "timeunit",
        1.0,
        nargout=0,
    )
    with filename.open() as stream:
        assert stream.readline().rstrip("\r\n").split(",") == ["Time", *_fields(source["chanlocs"], "labels")]
    values = np.loadtxt(filename, delimiter=",", skiprows=1)
    expected = np.column_stack((np.tile(np.arange(-128, 256) / 128, 80), source["data"].reshape(32, -1, order="F").T))
    assert values.shape == (384 * 80, 33)
    np.testing.assert_allclose(values, expected, rtol=0, atol=5.1e-10)


@expanded_matlab_test(NATIVE_SOURCE, "testExportErpExpressionWithoutLabelsOrTime", NATIVE_SHA256)
def test_export_erp_expression_without_labels_or_time(
    eeglab_backend, original_epochs, eeglab_working_directory, request
):
    source = original_epochs
    filename = eeglab_working_directory / "erp.txt"
    eeglab_backend(
        "pop_export",
        source,
        str(filename),
        "erp",
        "on",
        "time",
        "off",
        "elec",
        "off",
        "expr",
        "x = 2*x",
        "precision",
        9.0,
        nargout=0,
    )
    values = np.loadtxt(filename)
    # Keep the native single-precision reduction primitive in the MATLAB lane;
    # Python still owns the expected expression, file inspection and assertion.
    mean = eeglab_backend("mean", source["data"], 3.0) if _native(request) else source["data"].mean(axis=2)
    expected = (2 * mean).astype(float)
    assert values.shape == (32, 384)
    np.testing.assert_allclose(values, expected, rtol=0, atol=5.1e-10)
