"""
Test suite for pop_reref.py with MATLAB parity validation.

This module tests the pop_reref function which re-references EEG data
to average reference or to explicit common-reference channels.
"""

import os
import unittest
import sys
import tempfile
import warnings
import pytest
import numpy as np
import scipy.io

# Add src to path for imports
sys.path.insert(0, 'src')
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_reref import pop_reref
from eegprep.functions.sigprocfunc.reref import reref
from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.utils.testing import DebuggableTestCase
from tests.eeglab_tests import assert_matlab_near, eeglab_test
import importlib

eeg_checkset_module = importlib.import_module('eegprep.functions.adminfunc.eeg_checkset')


@eeglab_test("unittesting_popfunc/pop_reref/popfunc_pop_reref_wrapperTest.m", "test_pass_bugzilla_270")
def test_reference_pop_reref_original_bugzilla_270_input(eeglab_backend):
    eeg = eeglab_backend("eeg_emptyset")
    eeg.update(
        nbchan=3.0,
        pnts=20.0,
        trials=1.0,
        srate=1.0,
        xmin=0.0,
        xmax=2.0,
        data=np.arange(1.0, 61.0).reshape(3, 20),
    )
    eeglab_backend("pop_reref", eeg, np.array([[1.0, 2.0]]), "method", "standard")


@eeglab_test("unittesting_popfunc/pop_reref/popfunc_pop_reref_wrapperTest.m", "test_test_pop_reref")
def test_reference_pop_reref_original_average_reference_recording(eeglab_backend, eeglab_suite_root):
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data.set"))
    eeglab_backend("pop_reref", eeg, np.empty((0, 0)))


def test_pop_reref_current_suite_standard_method_with_multiple_references():
    eeg = {
        "data": np.arange(1, 61, dtype=float).reshape(3, 20),
        "nbchan": 3,
        "pnts": 20,
        "trials": 1,
        "srate": 1.0,
        "xmin": 0.0,
        "xmax": 2.0,
        "times": np.arange(20, dtype=float),
        "chanlocs": [{"labels": f"Ch{index + 1}"} for index in range(3)],
        "event": [],
        "urevent": [],
        "epoch": [],
        "icaweights": np.array([]),
        "icasphere": np.array([]),
        "icawinv": np.array([]),
        "icaact": np.array([]),
        "icachansind": np.array([], dtype=int),
    }

    output = pop_reref(eeg, [1, 2], "method", "standard")

    expected_reference = eeg["data"][[0, 1]].mean(axis=0)
    np.testing.assert_allclose(output["data"], eeg["data"][[2]] - expected_reference)


def test_pop_reref_keeps_a_nan_on_its_channel():
    """Subtraction keeps a NaN on its channel. EEGLAB's refmatrix spreads 0*NaN."""
    data = np.array(
        [
            [1.0, np.nan, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    eeg = {
        "data": data,
        "nbchan": 3,
        "pnts": 3,
        "trials": 1,
        "srate": 100.0,
        "xmin": 0.0,
        "xmax": 0.02,
        "times": np.arange(3) / 100.0,
        "chanlocs": [{"labels": f"Ch{index + 1}"} for index in range(3)],
        "event": [],
        "urevent": [],
        "epoch": [],
        "icaweights": np.array([]),
        "icasphere": np.array([]),
        "icawinv": np.array([]),
        "icaact": np.array([]),
        "icachansind": np.array([], dtype=int),
    }
    output = pop_reref(eeg, [3])
    reference = data[2]
    np.testing.assert_allclose(output["data"], data[:2] - reference, equal_nan=True)
    assert np.isnan(output["data"][0, 1])
    assert np.isfinite(output["data"][1]).all()

    refmatrix = np.eye(3)
    refmatrix[:, 2] -= 1.0
    with np.errstate(invalid="ignore"):
        spread = refmatrix @ data
    assert np.isnan(spread[:, 1]).all()


@eeglab_test("unittesting_sigprocfunc/reref/sigprocfunc_reref_wrapperTest.m", "test_test_reref")
def test_reference_reref(eeglab_backend, eeglab_suite_root):
    references = np.array([[2, 5, 26]], dtype=float)
    excluded = np.array([[1, 32]], dtype=float)
    for filename in ("eeglab_data.set", "eeglab_data_epochs_ica.set"):
        eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data" / filename))
        data = eeg["data"]
        average = eeglab_backend("reref", data, np.empty((0, 0)), "keepref", "on")
        explicit = eeglab_backend("reref", data, np.arange(1.0, 33.0)[None, :], "keepref", "on")
        assert_matlab_near(average, explicit)
        output = eeglab_backend("reref", data, references, "keepref", "off")
        assert_matlab_near([output.shape], [[29, *data.shape[1:]]])
        output = eeglab_backend("reref", data, 5.0, "keepref", "on")
        assert_matlab_near([output.shape], [data.shape])
        assert_matlab_near(output[4:5], np.zeros((1, *data.shape[1:])))
        output = eeglab_backend("reref", data, references, "exclude", excluded, "keepref", "off")
        assert_matlab_near([output.shape], [[29, *data.shape[1:]]])
        assert_matlab_near(output[[0, 28]], data[[0, 31]])
    output = eeglab_backend("reref", data, references, "exclude", excluded, "keepref", "off", "elocs", eeg["chanlocs"])
    assert_matlab_near([output.shape], [[29, *data.shape[1:]]])
    assert_matlab_near(output[[0, 28]], data[[0, 31]])
    output = eeglab_backend(
        "reref",
        data,
        references,
        "exclude",
        excluded,
        "keepref",
        "off",
        "elocs",
        eeg["chanlocs"],
        "refloc",
        eeg["chanlocs"][:, 1:2],
    )
    assert_matlab_near([output.shape], [[30, data.shape[1] * data.shape[2]]])


def test_reref_nan_stays_on_channels_that_depend_on_it():
    data = np.array(
        [
            [1.0, 2.0, np.nan],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [3.0, np.nan, 1.0],
        ]
    )
    explicit, *_rest = reref(data, ref=[2], keepref="off")
    np.testing.assert_allclose(
        explicit,
        np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0], [3.0, np.nan, 1.0]]) - np.array([7.0, 8.0, 9.0]),
        equal_nan=True,
    )
    assert np.isnan(explicit[0, 2])
    assert np.isfinite(explicit[1]).all()
    assert np.isnan(explicit[2, 1]) and np.isfinite(explicit[2, [0, 2]]).all()

    referenced, *_rest = reref(data, ref=[0], keepref="off")
    assert np.isnan(referenced[:, 2]).all()
    assert np.isfinite(referenced[:, 0]).all()

    excluded, *_rest = reref(data, ref=[], exclude=[0])
    assert np.isnan(excluded[0, 2])
    assert np.isfinite(excluded[1:, 2]).all()
    assert np.isnan(excluded[1:, 1]).all()


def exact_float32_average_reference(data: np.ndarray) -> np.ndarray:
    """Return the average reference rounded once to float32.

    Each finite sample is an integer multiple of one power of two. Those
    integers form ``channels * sample - sum(sample)``, and long double divides
    by the channel count. The channel count must be a power of two so that
    division is exact, and the numerator must fit in the long double significand.
    """
    values = np.ascontiguousarray(np.asarray(data), dtype=np.float32)
    if values.ndim < 2:
        raise AssertionError("average reference expects channel-major data")
    channels = int(values.shape[0])
    if channels < 1 or channels & (channels - 1):
        raise AssertionError("exact float32 average reference expects a power-of-two channel count")
    flat = values.reshape(channels, -1)
    if not np.isfinite(flat).all():
        raise AssertionError("exact float32 average reference expects finite samples")
    nonzero = flat[flat != 0]
    if nonzero.size == 0:
        return np.zeros_like(values)

    exponent = int(np.frexp(np.min(np.abs(nonzero)))[1])
    shift = 24 - exponent
    scaled = np.ldexp(flat.astype(np.float64), shift)
    recovered = np.rint(scaled)
    if not np.array_equal(scaled, recovered):
        raise AssertionError("float32 samples are not integer multiples of the scale")
    ints = recovered.astype(np.int64)
    if not np.array_equal(np.ldexp(ints.astype(np.float64), -shift), flat):
        raise AssertionError("scaled integers do not recover the float32 samples")
    if int(np.max(np.abs(ints))) > np.iinfo(np.int64).max // (2 * channels):
        raise AssertionError("scaled samples do not fit the int64 numerator")
    numerators = channels * ints - ints.sum(axis=0, dtype=np.int64)
    peak = max(abs(int(np.max(numerators))), abs(int(np.min(numerators))))
    if peak.bit_length() > int(np.finfo(np.longdouble).nmant):
        raise AssertionError("numerator does not fit exactly in long double")
    places = channels.bit_length() - 1
    exact = np.ldexp(numerators.astype(np.longdouble), -shift - places)
    return np.asarray(exact, dtype=np.float32).reshape(values.shape)


def _adversarial_float32_channels(channels: int, samples: int) -> np.ndarray:
    """Large float32 values whose float32 channel sum is not the real sum."""
    data = np.full((channels, samples), np.float32(2**22 + 3), dtype=np.float32)
    steps = np.arange(1, samples + 1, dtype=np.float32)
    data[0, :] = np.float32(2**22 + 3) + steps
    data[1, :] = np.float32(2**22 + 3) - steps
    return data


def test_float32_average_reference_matches_exact_rounding():
    """Float32 average reference equals one rounding of the real mean."""
    known = np.array([[1.0, 4.0], [3.0, 8.0]], dtype=np.float32)
    known_oracle = exact_float32_average_reference(known)
    np.testing.assert_array_equal(known_oracle, np.array([[-1.0, -2.0], [1.0, 2.0]], dtype=np.float32))

    data = _adversarial_float32_channels(256, 8)
    oracle = exact_float32_average_reference(data)
    naive = (data - data.mean(axis=0, dtype=np.float32)).astype(np.float32)
    assert not np.array_equal(naive, oracle)

    result, *_rest = reref(data, [])
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, oracle)

    epoched, *_rest = reref(data.reshape(256, 2, 4), [])
    assert epoched.dtype == np.float32
    np.testing.assert_array_equal(epoched, oracle.reshape(256, 2, 4))

    excluded_source = np.vstack([np.full((1, data.shape[1]), np.float32(2**20), dtype=np.float32), data])
    excluded, *_rest = reref(excluded_source, [], exclude=[0])
    assert excluded.dtype == np.float32
    np.testing.assert_array_equal(excluded[0], excluded_source[0])
    np.testing.assert_array_equal(excluded[1:], oracle)

    restored_source = _adversarial_float32_channels(127, 4)
    restored, *_rest = reref(restored_source, [], refloc={"labels": "old"})
    augmented = np.vstack([restored_source, np.zeros((1, restored_source.shape[1]), dtype=np.float32)])
    assert restored.dtype == np.float32
    np.testing.assert_array_equal(restored, exact_float32_average_reference(augmented))


def test_python_regression_low_level_reref_continuous_and_epoched_contracts():
    rng = np.random.default_rng(42)
    locs = [{"labels": f"Ch{index + 1}"} for index in range(32)]

    for data in (rng.normal(size=(32, 200)), rng.normal(size=(32, 40, 5))):
        average, *_ = reref(data, [], keepref="on")
        explicit_average, *_ = reref(data, np.arange(32), keepref="on")
        np.testing.assert_allclose(average, explicit_average, rtol=1e-12, atol=1e-12)

        removed, *_ = reref(data, [1, 4, 25], keepref="off")
        assert removed.shape == (29, *data.shape[1:])

        kept, *_ = reref(data, [4], keepref="on")
        assert kept.shape == data.shape
        np.testing.assert_allclose(kept[4], 0, atol=1e-12)

        excluded, out_locs, *_ = reref(
            data,
            [1, 4, 25],
            exclude=[0, 31],
            keepref="off",
            elocs=locs,
        )
        assert excluded.shape == (29, *data.shape[1:])
        np.testing.assert_array_equal(excluded[[0, -1]], data[[0, 31]])
        assert len(out_locs) == 29

    restored, restored_locs, *_ = reref(
        data,
        [1, 4, 25],
        exclude=[0, 31],
        keepref="off",
        elocs=locs,
        refloc={"labels": "old-reference"},
    )
    assert restored.shape == (30, *data.shape[1:])
    assert len(restored_locs) == 30


class PopRerefIcaRegressionTests(unittest.TestCase):
    def test_float_icachansind_values_update_ica_without_list_index_error(self):
        """MATLAB-loaded float ICA channel indices should stay usable internally."""
        EEG = {
            'data': np.arange(120, dtype=np.float64).reshape(3, 40),
            'nbchan': 3,
            'pnts': 40,
            'trials': 1,
            'srate': 100,
            'xmin': 0,
            'xmax': 0.39,
            'chanlocs': [{'labels': 'Ch1'}, {'labels': 'Ch2'}, {'labels': 'Ch3'}],
            'icaweights': np.eye(3),
            'icawinv': np.eye(3),
            'icasphere': np.eye(3),
            'icaact': np.array([]),
            'icachansind': np.array([0.0, 1.0, 2.0], dtype=np.float64),
            'ref': 'common',
            'epoch': np.array([], dtype=object),
        }
        original_option = eeg_checkset_module.option_scaleicarms
        eeg_checkset_module.option_scaleicarms = False
        try:
            result = pop_reref(EEG, ref=[])
        finally:
            eeg_checkset_module.option_scaleicarms = original_option

        np.testing.assert_array_equal(result['icachansind'], np.array([0, 1, 2]))
        self.assertTrue(np.issubdtype(result['icachansind'].dtype, np.integer))
        np.testing.assert_allclose(result['icawinv'].mean(axis=0), 0, atol=1e-12)

    def test_sample_data_average_reference_emits_no_runtime_warning(self):
        """Average-referencing sample data should not leak BLAS warnings."""
        EEG = pop_loadset("sample_data/eeglab_data.set")

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = pop_reref(EEG, ref=[])

        self.assertEqual(result["ref"], "average")
        self.assertTrue(np.isfinite(result["data"]).all())


class TestPopReref(DebuggableTestCase):
    """Test cases for pop_reref function."""

    def setUp(self):
        """Set up test fixtures."""
        if os.getenv('EEGPREP_SKIP_MATLAB') == '1':
            self.matlab_available = False
            return
        try:
            self.eeglab = get_eeglab()
            self.matlab_available = True
        except Exception:
            self.matlab_available = False

    def create_test_eeg(self, nbchan=32, pnts=1000, trials=1, srate=256):
        """Create a test EEG structure."""
        np.random.seed(42)  # For reproducible tests

        chanlocs = []
        for i in range(nbchan):
            chanlocs.append(
                {
                    'labels': f'Ch{i + 1}',
                    'X': np.cos(2 * np.pi * i / nbchan),
                    'Y': np.sin(2 * np.pi * i / nbchan),
                    'Z': 0.0,
                    'theta': 2 * np.pi * i / nbchan,
                    'radius': 0.5,
                    'ref': 'common',  # Initial reference
                }
            )

        icaweights = np.random.randn(nbchan, nbchan).astype(np.float64)
        icawinv = np.linalg.pinv(icaweights).astype(np.float64)
        icasphere = np.eye(nbchan).astype(np.float64)

        return {
            'data': np.random.randn(nbchan, pnts, trials).astype(np.float64),
            'nbchan': nbchan,
            'pnts': pnts,
            'trials': trials,
            'srate': srate,
            'chanlocs': chanlocs,
            'icaweights': icaweights,
            'icawinv': icawinv,
            'icasphere': icasphere,
            'icachansind': list(range(nbchan)),  # All channels used for ICA
            'ref': 'common',
        }

    def create_simple_eeg(self, nbchan=4, pnts=40):
        """Create a small EEG structure with deterministic channel locations."""
        coords = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.5, 0.5, 0.7071),
        ]
        chanlocs = []
        for idx in range(nbchan):
            x, y, z = coords[idx]
            chanlocs.append(
                {
                    'labels': f'Ch{idx + 1}',
                    'X': x,
                    'Y': y,
                    'Z': z,
                    'theta': 0.0,
                    'radius': 0.5,
                    'type': 'EEG',
                    'ref': 'common',
                }
            )
        data = np.arange(nbchan * pnts, dtype=np.float64).reshape(nbchan, pnts) / 10.0
        return {
            'data': data,
            'nbchan': nbchan,
            'pnts': pnts,
            'trials': 1,
            'srate': 100,
            'xmin': 0,
            'xmax': (pnts - 1) / 100,
            'chanlocs': chanlocs,
            'urchanlocs': chanlocs.copy(),
            'chaninfo': {},
            'icaweights': np.array([]),
            'icawinv': np.array([]),
            'icasphere': np.array([]),
            'icaact': np.array([]),
            'icachansind': [],
            'ref': 'common',
        }

    @staticmethod
    def _matlab_string(value):
        return "'" + str(value).replace("'", "''") + "'"

    @staticmethod
    def _matlab_cellstr(values):
        return "{" + ", ".join(TestPopReref._matlab_string(value) for value in values) + "}"

    @staticmethod
    def _matlab_numvec(values):
        return "[" + " ".join(f"{float(value):.17g}" for value in values) + "]"

    def _matlab_locs_code(self, variable, locs):
        labels = [loc.get('labels', '') for loc in locs]
        types = [loc.get('type', '') for loc in locs]
        refs = [loc.get('ref', '') for loc in locs]
        xs = [loc.get('X', 0.0) for loc in locs]
        ys = [loc.get('Y', 0.0) for loc in locs]
        zs = [loc.get('Z', 0.0) for loc in locs]
        theta = [loc.get('theta', 0.0) for loc in locs]
        radius = [loc.get('radius', 0.5) for loc in locs]
        return f"""
        {variable} = struct('labels', {{}}, 'X', {{}}, 'Y', {{}}, 'Z', {{}}, ...
            'theta', {{}}, 'radius', {{}}, 'type', {{}}, 'ref', {{}});
        tmp_labels = {self._matlab_cellstr(labels)};
        tmp_types = {self._matlab_cellstr(types)};
        tmp_refs = {self._matlab_cellstr(refs)};
        tmp_x = {self._matlab_numvec(xs)};
        tmp_y = {self._matlab_numvec(ys)};
        tmp_z = {self._matlab_numvec(zs)};
        tmp_theta = {self._matlab_numvec(theta)};
        tmp_radius = {self._matlab_numvec(radius)};
        for k = 1:numel(tmp_labels)
            {variable}(k).labels = tmp_labels{{k}};
            {variable}(k).X = tmp_x(k);
            {variable}(k).Y = tmp_y(k);
            {variable}(k).Z = tmp_z(k);
            {variable}(k).theta = tmp_theta(k);
            {variable}(k).radius = tmp_radius(k);
            {variable}(k).type = tmp_types{{k}};
            {variable}(k).ref = tmp_refs{{k}};
        end
        clear tmp_labels tmp_types tmp_refs tmp_x tmp_y tmp_z tmp_theta tmp_radius k;
        """

    def _matlab_pop_reref(self, EEG, ref_expr, option_expr=""):
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        temp_input = tempfile.mktemp(suffix='.mat')
        temp_output = tempfile.mktemp(suffix='.mat')
        icachansind = np.asarray(EEG.get('icachansind', []), dtype=np.float64)
        if icachansind.size:
            icachansind = icachansind + 1
        scipy.io.savemat(
            temp_input,
            {
                'data_in': EEG['data'],
                'icaweights_in': EEG.get('icaweights', np.array([])),
                'icawinv_in': EEG.get('icawinv', np.array([])),
                'icasphere_in': EEG.get('icasphere', np.array([])),
                'icaact_in': EEG.get('icaact', np.array([])),
                'icachansind_in': icachansind,
            },
        )

        chanlocs_code = self._matlab_locs_code('EEG.chanlocs', EEG.get('chanlocs', []))
        urchanlocs_code = self._matlab_locs_code('EEG.urchanlocs', EEG.get('urchanlocs', []))
        removed_code = ""
        removedchans = EEG.get('chaninfo', {}).get('removedchans', [])
        if removedchans:
            removed_code = self._matlab_locs_code('EEG.chaninfo.removedchans', removedchans)
        nodatchans_code = ""
        nodatchans = EEG.get('chaninfo', {}).get('nodatchans', [])
        if nodatchans:
            nodatchans_code = self._matlab_locs_code('EEG.chaninfo.nodatchans', nodatchans)

        matlab_code = f"""
        load({self._matlab_string(temp_input)});
        EEG = eeg_emptyset;
        EEG.setname = 'pop_reref parity';
        EEG.data = data_in;
        EEG.nbchan = size(data_in, 1);
        EEG.pnts = size(data_in, 2);
        EEG.trials = 1;
        EEG.srate = {float(EEG.get('srate', 100)):.17g};
        EEG.xmin = {float(EEG.get('xmin', 0)):.17g};
        EEG.xmax = {float(EEG.get('xmax', 0)):.17g};
        EEG.ref = 'common';
        {chanlocs_code}
        {urchanlocs_code}
        EEG.icaweights = icaweights_in;
        EEG.icawinv = icawinv_in;
        EEG.icasphere = icasphere_in;
        EEG.icaact = icaact_in;
        EEG.icachansind = icachansind_in;
        {removed_code}
        {nodatchans_code}
        [EEG, com] = pop_reref(EEG, {ref_expr}{option_expr});
        data_ml = EEG.data;
        nbchan_ml = EEG.nbchan;
        labels_ml = {{EEG.chanlocs.labels}};
        if isempty(EEG.icawinv), icawinv_ml = []; else, icawinv_ml = EEG.icawinv; end
        if isempty(EEG.icaweights), icaweights_ml = []; else, icaweights_ml = EEG.icaweights; end
        if isempty(EEG.icasphere), icasphere_ml = []; else, icasphere_ml = EEG.icasphere; end
        if isempty(EEG.icaact), icaact_ml = []; else, icaact_ml = EEG.icaact; end
        if isempty(EEG.icachansind), icachansind_ml = []; else, icachansind_ml = EEG.icachansind; end
        save({self._matlab_string(temp_output)}, 'data_ml', 'nbchan_ml', 'labels_ml', ...
            'icawinv_ml', 'icaweights_ml', 'icasphere_ml', 'icaact_ml', 'icachansind_ml');
        """
        self.eeglab.engine.eval(matlab_code, nargout=0)
        out = scipy.io.loadmat(temp_output)
        os.remove(temp_input)
        os.remove(temp_output)
        return out

    def test_explicit_single_reference_removes_ref_by_default(self):
        """Test common reference to a channel index."""
        EEG = self.create_test_eeg(nbchan=4, pnts=20)
        original_data = EEG['data'].copy()

        result = pop_reref(EEG, ref=2)

        expected = original_data - original_data[1:2, :, :]
        expected = np.delete(expected, 1, axis=0)
        if expected.ndim == 3 and expected.shape[2] == 1:
            expected = np.squeeze(expected, axis=2)
        np.testing.assert_allclose(result['data'], expected, rtol=1e-6)
        self.assertEqual(result['nbchan'], 3)
        self.assertEqual([chan['labels'] for chan in result['chanlocs']], ['Ch1', 'Ch3', 'Ch4'])
        self.assertEqual(result['ref'], 'common')
        self.assertEqual(result['chaninfo']['nodatchans'][0]['labels'], 'Ch2')

    def test_explicit_reference_label_keepref(self):
        """Test common reference to a channel label while keeping the reference row."""
        EEG = self.create_test_eeg(nbchan=4, pnts=20)
        original_data = EEG['data'].copy()

        result = pop_reref(EEG, ref='Ch2', keepref='on')

        expected = original_data - original_data[1:2, :, :]
        if expected.ndim == 3 and expected.shape[2] == 1:
            expected = np.squeeze(expected, axis=2)
        np.testing.assert_allclose(result['data'], expected, rtol=1e-6)
        self.assertEqual(result['nbchan'], 4)
        np.testing.assert_allclose(result['data'][1], 0, atol=1e-6)
        for chan in result['chanlocs']:
            self.assertEqual(chan['ref'], 'Ch2')

    def test_explicit_reference_without_chanlocs_still_rereferences(self):
        """Numeric common reference should work when chanlocs are absent."""
        EEG = self.create_simple_eeg(nbchan=4, pnts=20)
        EEG['chanlocs'] = []
        EEG['urchanlocs'] = []

        result = pop_reref(EEG, ref=2)

        self.assertEqual(result['nbchan'], 3)
        self.assertEqual(result['data'].shape[0], 3)
        self.assertEqual(np.size(result['chanlocs']), 0)

    def test_refica_remove_and_off_modes(self):
        """Test refica options that intentionally do not re-reference ICA maps."""
        original_option = eeg_checkset_module.option_scaleicarms
        eeg_checkset_module.option_scaleicarms = False
        try:
            EEG = self.create_test_eeg(nbchan=4, pnts=20)
            original_icawinv = EEG['icawinv'].copy()
            original_icaweights = EEG['icaweights'].copy()
            EEG['icaact'] = np.ones((4, EEG['pnts']))

            removed = pop_reref(EEG, ref=[], refica='remove')
            self.assertEqual(removed['icawinv'].size, 0)
            self.assertEqual(removed['icaweights'].size, 0)
            self.assertEqual(removed['icasphere'].size, 0)

            off = pop_reref(EEG, ref=[], refica='off')
            np.testing.assert_allclose(off['icawinv'], original_icawinv)
            np.testing.assert_allclose(off['icaweights'], original_icaweights)
            self.assertIsInstance(off['icachansind'], np.ndarray)
        finally:
            eeg_checkset_module.option_scaleicarms = original_option

    def test_refica_clears_when_ica_channel_is_excluded(self):
        """Test EEGLAB edge case where excluded ICA channels invalidate ICA."""
        EEG = self.create_test_eeg(nbchan=4, pnts=20)

        result = pop_reref(EEG, ref=[], exclude=[1])

        self.assertEqual(result['icawinv'].size, 0)
        self.assertEqual(result['icaweights'].size, 0)

    def test_interpchan_clears_ica_when_channel_set_changes(self):
        """Interpolating channels before reref invalidates ICA decomposition."""
        EEG = self.create_simple_eeg(nbchan=3, pnts=20)
        removed = {
            'labels': 'Ch4',
            'X': 0.0,
            'Y': -1.0,
            'Z': 0.0,
            'theta': 90.0,
            'radius': 0.5,
            'type': 'EEG',
            'ref': 'common',
        }
        EEG['urchanlocs'] = EEG['chanlocs'] + [removed]
        EEG['chaninfo'] = {'removedchans': [removed]}
        EEG['icawinv'] = np.array(
            [
                [1.0, 0.2, 0.1],
                [0.1, 1.0, 0.2],
                [0.2, 0.1, 1.0],
            ]
        )
        EEG['icaweights'] = np.linalg.pinv(EEG['icawinv'])
        EEG['icasphere'] = np.eye(3)
        EEG['icaact'] = np.ones((3, EEG['pnts']))
        EEG['icachansind'] = [0, 1, 2]

        result = pop_reref(EEG, ref=[], interpchan=[])

        self.assertEqual(result['icawinv'].size, 0)
        self.assertEqual(result['icaweights'].size, 0)
        self.assertEqual(result['icasphere'].size, 0)
        self.assertEqual(result['icaact'].size, 0)

    def test_error_icachansind_mismatch(self):
        """Test behavior when icachansind length doesn't match nbchan."""
        EEG = self.create_test_eeg(nbchan=16, pnts=100)

        # Make icachansind have different length (e.g., after channel removal)
        EEG['icachansind'] = list(range(8))  # Only 8 channels instead of 16

        # The function should clear ICA fields instead of raising an error
        result = pop_reref(EEG, ref=None)

        self.assertEqual(result['icawinv'].size, 0)
        self.assertEqual(result['icaweights'].size, 0)
        self.assertEqual(result['icasphere'].size, 0)

    def test_history_command_formats_label_reference_like_eeglab(self):
        EEG = self.create_simple_eeg(nbchan=4, pnts=20)

        _out, com = pop_reref(EEG, ref='Ch2', keepref='on', return_com=True)

        self.assertEqual(com, "EEG = pop_reref( EEG, {'Ch2'}, 'keepref', 'on');")

    def test_history_command_formats_numeric_channels_as_matlab_indices(self):
        EEG = self.create_simple_eeg(nbchan=4, pnts=20)

        _out, com = pop_reref(EEG, ref=[1], exclude=[4], return_com=True)

        self.assertEqual(com, "EEG = pop_reref( EEG, [1], 'exclude', [4]);")

    def test_history_command_formats_refloc_struct_like_eeglab(self):
        EEG = self.create_simple_eeg(nbchan=2, pnts=20)
        old_ref = {
            'labels': 'M1',
            'X': 0.0,
            'Y': -1.0,
            'Z': 0.0,
            'theta': -90.0,
            'radius': 0.5,
            'type': 'REF',
            'ref': 'common',
        }
        EEG['chaninfo'] = {'nodatchans': [old_ref], 'removedchans': [old_ref]}

        _out, com = pop_reref(EEG, ref=[], refloc=old_ref, return_com=True)

        self.assertIn("'refloc', struct(", com)
        self.assertIn("'labels',{'M1'}", com)
        self.assertIn("'theta',-90", com)

    def test_multiple_dataset_gui_path_prompts_once_like_eeglab(self):
        class Renderer:
            def __init__(self):
                self.calls = 0

            def run(self, spec, initial_values=None):
                self.calls += 1
                return {"ave": True}

        renderer = Renderer()
        datasets = [self.create_simple_eeg(nbchan=2, pnts=20), self.create_simple_eeg(nbchan=2, pnts=20)]

        outputs, com = pop_reref(datasets, gui=True, renderer=renderer, return_com=True)

        self.assertEqual(renderer.calls, 1)
        self.assertEqual(len(outputs), 2)
        self.assertEqual([out["ref"] for out in outputs], ["average", "average"])
        self.assertEqual(com, "EEG = pop_reref( EEG, []);")

    def test_gui_huber_path_preserves_other_eeglab_gui_options(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {
                    "huberef": True,
                    "huberval": "25",
                    "exclude": "2",
                }

        EEG = self.create_simple_eeg(nbchan=2, pnts=20)
        original_excluded = EEG['data'][1].copy()

        out = pop_reref(EEG, gui=True, renderer=Renderer())

        np.testing.assert_allclose(out['data'][1], original_excluded)

    def test_gui_numeric_reference_text_is_one_based_like_python_api(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {
                    "rerefstr": True,
                    "reref": "1",
                    "keepref": True,
                }

        EEG = self.create_simple_eeg(nbchan=3, pnts=20)

        out, com = pop_reref(EEG, gui=True, renderer=Renderer(), return_com=True)

        self.assertEqual(com, "EEG = pop_reref( EEG, [1], 'keepref', 'on');")
        np.testing.assert_allclose(out['data'][0], 0, atol=1e-6)

    def test_parity_data_reref_with_matlab(self):
        """Test parity with MATLAB for data average re-referencing."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        import os
        import tempfile
        import scipy.io
        from eegprep import pop_loadset, pop_saveset

        local_url = os.path.join(os.path.dirname(__file__), '../sample_data/')
        test_file = os.path.join(local_url, 'eeglab_data_with_ica_tmp.set')
        EEG = pop_loadset(test_file)

        # Python result
        py_result = pop_reref(EEG.copy(), ref=None)

        # Save to file for MATLAB
        temp_file = tempfile.mktemp(suffix='.set')
        pop_saveset(EEG, temp_file)

        # MATLAB result
        matlab_code = f'''
        EEG = pop_loadset('{temp_file}');
        [~, EEG] = evalc('pop_reref(EEG, []);');
        data_ml = EEG.data;
        save('{temp_file}.mat', 'data_ml');
        '''
        self.eeglab.engine.eval(matlab_code, nargout=0)

        # Load MATLAB result
        mat_data = scipy.io.loadmat(temp_file + '.mat')
        data_ml = mat_data['data_ml']

        # Clean up
        os.remove(temp_file)
        os.remove(temp_file + '.mat')
        if os.path.exists(temp_file.replace('.set', '.fdt')):
            os.remove(temp_file.replace('.set', '.fdt'))

        # Compare data (should match closely for average reference)
        # Data re-referencing: max abs diff ~1e-4 due to float32/float64 precision
        np.testing.assert_allclose(
            py_result['data'], data_ml, rtol=1e-3, atol=1e-3, err_msg="Data re-referencing differs from MATLAB"
        )

    def test_parity_explicit_reference_keepref_with_matlab(self):
        """Test MATLAB parity for explicit common reference with keepref."""
        EEG = self.create_simple_eeg(nbchan=4, pnts=20)

        py_result = pop_reref(EEG, ref=2, keepref='on')
        ml_result = self._matlab_pop_reref(EEG, "2", ", 'keepref', 'on'")

        np.testing.assert_allclose(py_result['data'], ml_result['data_ml'], rtol=1e-6, atol=1e-6)
        self.assertEqual(py_result['nbchan'], int(ml_result['nbchan_ml'].squeeze()))

    def test_parity_explicit_reference_remove_with_matlab(self):
        """Test MATLAB parity for explicit common reference with default ref removal."""
        EEG = self.create_simple_eeg(nbchan=4, pnts=20)

        py_result = pop_reref(EEG, ref=2)
        ml_result = self._matlab_pop_reref(EEG, "2")

        np.testing.assert_allclose(py_result['data'], ml_result['data_ml'], rtol=1e-6, atol=1e-6)
        self.assertEqual(py_result['nbchan'], int(ml_result['nbchan_ml'].squeeze()))

    def test_parity_exclude_with_matlab(self):
        """Test MATLAB parity for average reference with excluded channels."""
        EEG = self.create_simple_eeg(nbchan=4, pnts=20)

        py_result = pop_reref(EEG, ref=[], exclude=[4])
        ml_result = self._matlab_pop_reref(EEG, "[]", ", 'exclude', 4")

        np.testing.assert_allclose(py_result['data'], ml_result['data_ml'], rtol=1e-6, atol=1e-6)

    def test_parity_refloc_with_matlab(self):
        """Test MATLAB parity for adding the old reference channel back."""
        EEG = self.create_simple_eeg(nbchan=2, pnts=20)
        old_ref = {
            'labels': 'M1',
            'X': 0.0,
            'Y': -1.0,
            'Z': 0.0,
            'theta': -90.0,
            'radius': 0.5,
            'type': 'REF',
            'ref': 'common',
        }
        EEG['chaninfo'] = {'nodatchans': [old_ref], 'removedchans': [old_ref]}

        py_result = pop_reref(EEG, ref=[], refloc=old_ref)
        ml_result = self._matlab_pop_reref(
            EEG,
            "[]",
            ", 'refloc', EEG.chaninfo.nodatchans",
        )

        np.testing.assert_allclose(py_result['data'], ml_result['data_ml'], rtol=1e-6, atol=1e-6)
        self.assertEqual(py_result['nbchan'], int(ml_result['nbchan_ml'].squeeze()))

    def test_parity_huber_with_matlab(self):
        """Test MATLAB parity for Huber average reference."""
        EEG = self.create_simple_eeg(nbchan=4, pnts=20)
        EEG['data'][0, :] += 50.0

        py_result = pop_reref(EEG, ref=[], huber=25)
        ml_result = self._matlab_pop_reref(EEG, "[]", ", 'huber', 25")

        np.testing.assert_allclose(py_result['data'], ml_result['data_ml'], rtol=1e-6, atol=1e-6)

    def test_parity_interpchan_with_matlab(self):
        """Test MATLAB parity for interpchan before average reference."""
        EEG = self.create_simple_eeg(nbchan=3, pnts=20)
        removed = {
            'labels': 'Ch4',
            'X': 0.0,
            'Y': -1.0,
            'Z': 0.0,
            'theta': 90.0,
            'radius': 0.5,
            'type': 'EEG',
            'ref': 'common',
        }
        EEG['urchanlocs'] = EEG['chanlocs'] + [removed]
        EEG['chaninfo'] = {'removedchans': [removed]}

        py_result = pop_reref(EEG, ref=[], interpchan=[])
        ml_result = self._matlab_pop_reref(EEG, "[]", ", 'interpchan', []")

        np.testing.assert_allclose(py_result['data'], ml_result['data_ml'], rtol=1e-3, atol=1e-3)
        self.assertEqual(py_result['nbchan'], int(ml_result['nbchan_ml'].squeeze()))

    def test_parity_refica_average_with_matlab(self):
        """Test MATLAB parity for average reference ICA map update."""
        EEG = self.create_simple_eeg(nbchan=4, pnts=20)
        EEG['icawinv'] = np.array(
            [
                [1.0, 0.2, 0.1, 0.0],
                [0.1, 1.0, 0.2, 0.1],
                [0.0, 0.1, 1.0, 0.2],
                [0.2, 0.0, 0.1, 1.0],
            ]
        )
        EEG['icaweights'] = np.linalg.pinv(EEG['icawinv'])
        EEG['icasphere'] = np.eye(4)
        EEG['icaact'] = np.ones((4, EEG['pnts']))
        EEG['icachansind'] = [0, 1, 2, 3]

        py_result = pop_reref(EEG, ref=[])
        ml_result = self._matlab_pop_reref(EEG, "[]")

        np.testing.assert_allclose(py_result['icawinv'], ml_result['icawinv_ml'], rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(py_result['icaweights'], ml_result['icaweights_ml'], rtol=1e-8, atol=1e-8)

    def test_parity_refica_backwardcomp_with_matlab(self):
        """Test MATLAB parity for backwardcomp ICA handling."""
        EEG = self.create_simple_eeg(nbchan=4, pnts=20)
        EEG['icawinv'] = np.array(
            [
                [1.0, 0.2, 0.1, 0.0],
                [0.1, 1.0, 0.2, 0.1],
                [0.0, 0.1, 1.0, 0.2],
                [0.2, 0.0, 0.1, 1.0],
            ]
        )
        EEG['icaweights'] = np.linalg.pinv(EEG['icawinv'])
        EEG['icasphere'] = np.eye(4)
        EEG['icaact'] = np.ones((4, EEG['pnts']))
        EEG['icachansind'] = [0, 1, 2, 3]

        py_result = pop_reref(EEG, ref=[], refica='backwardcomp')
        ml_result = self._matlab_pop_reref(EEG, "[]", ", 'refica', 'backwardcomp'")

        np.testing.assert_allclose(py_result['icawinv'], ml_result['icawinv_ml'], rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(py_result['icaweights'], ml_result['icaweights_ml'], rtol=1e-8, atol=1e-8)
        self.assertEqual(py_result['icaact'].size, ml_result['icaact_ml'].size)


if __name__ == '__main__':
    unittest.main()


def test_reref_gui_excludes_exactly_the_typed_channels_and_records_them():
    eeg = TestPopReref().create_simple_eeg()
    eeg["data"] = np.arange(320, dtype=float).reshape(8, 40)
    eeg["nbchan"] = 8
    eeg["chanlocs"] = [{"labels": f"Ch{i + 1}"} for i in range(8)]

    class Renderer:
        def run(self, spec, initial_values=None):
            return {"ave": True, "exclude": "2 6"}

    out, command = pop_reref(eeg, gui=True, renderer=Renderer(), return_com=True)
    included = [0, 2, 3, 4, 6, 7]
    expected = eeg["data"].copy()
    expected[included] -= expected[included].mean(axis=0)
    np.testing.assert_allclose(out["data"], expected)
    assert command == "EEG = pop_reref( EEG, [], 'exclude', [2 6]);"
    by_api = pop_reref(eeg, [], exclude=[2, 6])
    np.testing.assert_array_equal(out["data"], by_api["data"])


@pytest.mark.parametrize("empty", [None, "", [], np.array([])])
def test_reref_empty_reference_header_is_unknown(empty):
    eeg = TestPopReref().create_simple_eeg()
    eeg["ref"] = empty
    for loc in eeg["chanlocs"]:
        loc["ref"] = empty

    class Renderer:
        def run(self, spec, initial_values=None):
            assert spec.controls[0].string == "Current data reference state is: unknown"
            return None

    pop_reref(eeg, gui=True, renderer=Renderer())


@pytest.mark.parametrize("key", ["ref", "exclude", "refloc", "interpchan"])
@pytest.mark.parametrize("value", [0, 5, 1.5, np.nan])
def test_reref_rejects_invalid_one_based_identifiers(key, value):
    eeg = TestPopReref().create_simple_eeg()
    eeg["urchanlocs"] = eeg["chanlocs"]
    eeg["chaninfo"] = {"nodatchans": eeg["chanlocs"]}
    kwargs = {key: [value]}
    kwargs.setdefault("ref", [])
    with pytest.raises(ValueError, match=r"pop_reref:.*1\.\.4"):
        pop_reref(eeg, **kwargs)
