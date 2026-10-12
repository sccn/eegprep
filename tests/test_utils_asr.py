import sys
import unittest
import numpy as np
from unittest.mock import patch

from eegprep.plugins.clean_rawdata.asr_calibrate import asr_calibrate
from eegprep.plugins.clean_rawdata.asr_process import DEFAULT_MAX_MEM_MB, asr_process
from eegprep.plugins.clean_rawdata.clean_artifacts import clean_artifacts
from eegprep.plugins.clean_rawdata.clean_asr import clean_asr


class TestAsrCalibrate(unittest.TestCase):
    """Test the asr_calibrate function."""

    def setUp(self):
        """Set up test fixtures with synthetic EEG data."""
        np.random.seed(42)  # For reproducible tests
        self.n_channels = 8
        self.n_samples = 1000
        self.srate = 250.0

        # Create synthetic clean EEG data (zero-mean)
        self.clean_data = np.random.randn(self.n_channels, self.n_samples) * 0.5

        # Add some realistic structure (autocorrelation)
        for i in range(self.n_channels):
            # Simple AR(1) process for more realistic EEG-like data
            for j in range(1, self.n_samples):
                self.clean_data[i, j] += 0.8 * self.clean_data[i, j - 1]

    def test_arbitrary_sampling_rate_designs_filter(self):
        """Any sampling rate gets a shaping filter, as asr_calibrate.m's yulewalk does.

        Rates such as 258 Hz (a real clinical EDF rate) have no entry in the
        pre-computed table that clean_rawdata falls back on without MATLAB's Signal
        Processing Toolbox, and must still calibrate rather than raise.
        """
        for srate in (258.0, 512.5, 999.0, 1000.0, 1024.0):
            with self.subTest(srate=srate):
                data = np.random.randn(4, max(1000, int(srate * 2))) * 0.3
                state = asr_calibrate(data, srate)
                self.assertEqual(len(state['B']), 9)
                self.assertEqual(len(state['A']), 9)
                self.assertTrue(np.all(np.isfinite(state['B'])))
                self.assertTrue(np.all(np.isfinite(state['A'])))
                # The design must be stable or the IIR pass would diverge on real data.
                self.assertTrue(np.all(np.abs(np.roots(state['A'])) < 1.0))

    def test_shaping_filter_matches_matlab_reference(self):
        """The designed filter must match MATLAB's yulewalk at the documented rates.

        Reference coefficients are the pre-computed table in EEGLAB clean_rawdata's
        asr_calibrate.m (GPL), which MATLAB uses when yulewalk is unavailable. They
        agree with live yulewalk output to ~1e-11, so they pin the port at every rate
        the reference implementation tabulates.
        """
        expected = {
            100: (
                [
                    0.9314233528641650,
                    -1.0023683814963549,
                    -0.4125359862018213,
                    0.7631567476327510,
                    0.4160430392910331,
                    -0.6549131038692215,
                    -0.0372583518046807,
                    0.1916268458752655,
                    0.0462411971592346,
                ],
                [
                    1.0000000000000000,
                    -0.4544220180303844,
                    -1.0007038682936749,
                    0.5374925521337940,
                    0.4905013360991340,
                    -0.4861062879351137,
                    -0.1995986490699414,
                    0.1830048420730026,
                    0.0457678549234644,
                ],
            ),
            128: (
                [
                    1.1027301639165037,
                    -2.0025621813611867,
                    0.8942119516481342,
                    0.1549979524226999,
                    0.0192366904488084,
                    0.1782897770278735,
                    -0.5280306696498717,
                    0.2913540603407520,
                    -0.0262209802526358,
                ],
                [
                    1.0000000000000000,
                    -1.1042042046423233,
                    -0.3319558528606542,
                    0.5802946221107337,
                    -0.0010360013915635,
                    0.0382167091925086,
                    -0.2609928034425362,
                    0.0298719057761086,
                    0.0935044692959187,
                ],
            ),
            200: (
                [
                    1.4489483325802353,
                    -2.6692514764802775,
                    2.0813970620731115,
                    -0.9736678877049534,
                    0.1054605060352928,
                    -0.1889101692314626,
                    0.6111331636592364,
                    -0.3616483013075088,
                    0.1834313060776763,
                ],
                [
                    1.0000000000000000,
                    -0.9913236099393967,
                    0.3159563145469344,
                    -0.0708347481677557,
                    -0.0558793822071149,
                    -0.2539619026478943,
                    0.2473056615251193,
                    -0.0420478437473110,
                    0.0077455718334464,
                ],
            ),
            # 250 Hz has no entry in asr_calibrate.m's table; these come from EEGPrep's
            # own previous table and agree with MATLAB's live yulewalk to 2e-14.
            250: (
                [
                    1.7313331085426,
                    -4.168133532957,
                    5.37379900844173,
                    -5.57212564343886,
                    4.70122651316513,
                    -3.34208799655246,
                    1.95045488724908,
                    -0.766909658912065,
                    0.233281060974837,
                ],
                [
                    1.0,
                    -1.6384949276666,
                    1.73987814299055,
                    -1.83638657883456,
                    1.3924177536798,
                    -0.953780426622198,
                    0.505158779550745,
                    -0.159504514603055,
                    0.0545278399847978,
                ],
            ),
            256: (
                [
                    1.7587013141770287,
                    -4.3267624394458641,
                    5.7999880031015953,
                    -6.2396625463547508,
                    5.3768079046882207,
                    -3.7938218893374835,
                    2.1649108095226470,
                    -0.8591392569863763,
                    0.2569361125627988,
                ],
                [
                    1.0000000000000000,
                    -1.7008039639301735,
                    1.9232830391058724,
                    -2.0826929726929797,
                    1.5982638742557307,
                    -1.0735854183930011,
                    0.5679719225652651,
                    -0.1886181499768189,
                    0.0572954115997261,
                ],
            ),
            300: (
                [
                    1.9153920676433143,
                    -5.7748421104926795,
                    9.1864764859103936,
                    -10.7350356619363630,
                    9.6423672437729007,
                    -6.6181939699544277,
                    3.4219421494177711,
                    -1.2622976569994351,
                    0.2968423019363821,
                ],
                [
                    1.0000000000000000,
                    -2.3143703322055491,
                    3.2222567327379434,
                    -3.6030527704320621,
                    2.9645154844073698,
                    -1.8842615840684735,
                    0.9222455868758080,
                    -0.3103251703648485,
                    0.0634586449896364,
                ],
            ),
            500: (
                [
                    2.3133520086975823,
                    -11.9471223009159130,
                    29.1067166493384340,
                    -43.7550171007238190,
                    44.3385767452216370,
                    -30.9965523846388000,
                    14.6209883020737190,
                    -4.2743412400311449,
                    0.5982553583777899,
                ],
                [
                    1.0000000000000000,
                    -4.6893329084452580,
                    10.5989986701080210,
                    -14.9691518101365230,
                    14.3320358399731820,
                    -9.4924317069169977,
                    4.2425899618982656,
                    -1.1715600975178280,
                    0.1538048427717476,
                ],
            ),
            512: (
                [
                    2.3275475636130865,
                    -12.2166478485960430,
                    30.1632789058248850,
                    -45.8009842020820410,
                    46.7261263011068880,
                    -32.7796858196767220,
                    15.4623349612560630,
                    -4.5019779685307473,
                    0.6242733481676324,
                ],
                [
                    1.0000000000000000,
                    -4.7827378944258703,
                    10.9780696236622980,
                    -15.6795187888195360,
                    15.1281978667576310,
                    -10.0632079834518220,
                    4.5014690636505614,
                    -1.2394100873286753,
                    0.1614727510688058,
                ],
            ),
        }

        for srate, (ref_b, ref_a) in expected.items():
            with self.subTest(srate=srate):
                data = np.random.randn(4, max(1000, srate * 2)) * 0.3
                state = asr_calibrate(data, float(srate))
                # Tolerance covers the reference table's own ~1e-11 spread against live
                # yulewalk plus LAPACK differences; a design regression would be far larger.
                np.testing.assert_allclose(state['B'], ref_b, rtol=1e-8, atol=1e-9)
                np.testing.assert_allclose(state['A'], ref_a, rtol=1e-8, atol=1e-9)


class TestAsrProcess(unittest.TestCase):
    """Test the asr_process function."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(123)
        self.n_channels = 6
        self.n_samples = 500
        self.srate = 200.0

        # Create calibration data and state
        calib_data = np.random.randn(self.n_channels, 1000) * 0.3
        self.state = asr_calibrate(calib_data, self.srate)

        # Create test data with some artifacts
        self.test_data = np.random.randn(self.n_channels, self.n_samples) * 0.4
        # Add some artifacts to specific channels/times
        self.test_data[2, 100:150] += np.random.randn(50) * 2.0  # Large artifacts

    def test_default_max_mem_works_without_psutil(self):
        """max_mem=None must resolve to the fixed default without importing psutil.

        psutil has no WebAssembly build, so this path must not import it (e.g. under
        Pyodide). Block the real import (rather than mocking asr_process's logic) to prove
        no code path reaches for psutil.
        """
        with patch.dict(sys.modules, {'psutil': None}):
            cleaned_data, new_state = asr_process(self.test_data, self.srate, self.state, max_mem=None)

        self.assertEqual(cleaned_data.shape, self.test_data.shape)
        self.assertTrue(np.all(np.isfinite(cleaned_data)))

    def test_default_max_mem_equals_explicit_default(self):
        """max_mem=None is exactly the documented default budget, not merely a plausible one."""
        default_out, default_state = asr_process(self.test_data.copy(), self.srate, self.state)
        explicit_out, explicit_state = asr_process(
            self.test_data.copy(), self.srate, self.state, max_mem=DEFAULT_MAX_MEM_MB
        )

        np.testing.assert_array_equal(default_out, explicit_out)
        np.testing.assert_array_equal(default_state['carry'], explicit_state['carry'])
        np.testing.assert_array_equal(default_state['last_R'], explicit_state['last_R'])

    def test_explicit_max_mem_forces_blocks(self):
        """A small but sufficient explicit budget splits the work into blocks and still cleans."""
        large_data = np.random.randn(self.n_channels, 5000) * 0.5

        with self.assertLogs('eegprep.plugins.clean_rawdata.asr_process', level='INFO') as log:
            cleaned_data, _ = asr_process(large_data, self.srate, self.state, max_mem=10)

        self.assertTrue(any('blocks' in msg for msg in log.output))
        self.assertEqual(cleaned_data.shape, large_data.shape)
        self.assertTrue(np.all(np.isfinite(cleaned_data)))

    def test_explicit_budget_below_fixed_overhead_raises_with_remedy(self):
        """An explicit budget the fixed overhead cannot fit in must fail, naming the minimum and the fix."""
        with self.assertRaises(RuntimeError) as cm:
            asr_process(self.test_data, self.srate, self.state, max_mem=0.001)

        message = str(cm.exception)
        self.assertIn('Not enough memory', message)
        self.assertIn('max_mem=0.001', message)
        self.assertIn(f'{self.n_channels} channels', message)
        self.assertIn('max_mem=None', message)

    def test_default_budget_scales_with_channels_and_sampling_rate(self):
        """The default budget must cover the fixed overhead of high-density, high-rate data.

        128 channels at 1000 Hz and 256 channels at 512 Hz both need more than the 64 MB base
        budget for the lookahead overhead (C * C * P * 8 * 3 bytes), which used to raise
        'Not enough memory' even though EEGLAB's asr_process.m falls back to a larger budget.
        """
        for n_channels, srate in ((128, 1000.0), (256, 512.0)):
            with self.subTest(channels=n_channels, srate=srate):
                rng = np.random.default_rng(n_channels)
                state = asr_calibrate(rng.standard_normal((n_channels, int(3 * srate))), srate)
                chunk = rng.standard_normal((n_channels, int(srate)))
                lookahead_samples = int(round(0.25 * srate))
                overhead_mb = n_channels * n_channels * lookahead_samples * 8 * 3 / 1024**2
                self.assertGreater(overhead_mb, DEFAULT_MAX_MEM_MB)

                cleaned, _ = asr_process(chunk, srate, state)

                self.assertEqual(cleaned.shape, chunk.shape)
                self.assertTrue(np.all(np.isfinite(cleaned)))

    def test_rank_deficient_covariance_produces_sane_output(self):
        """Process genuinely rank-deficient data (singular covariance).

        Duplicate and zeroed channels make the per-window covariance singular,
        exercising the eigendecomposition and pseudo-inverse paths with a real
        degenerate input rather than monkeypatching numpy to raise. The cleaned
        output must stay finite and keep its shape.
        """
        degenerate = self.test_data.copy()
        degenerate[3, :] = degenerate[0, :]  # duplicate channel -> singular covariance
        degenerate[5, :] = 0.0  # flat channel -> singular covariance

        cleaned_data, new_state = asr_process(degenerate, self.srate, self.state)

        self.assertEqual(cleaned_data.shape, degenerate.shape)
        self.assertTrue(np.all(np.isfinite(cleaned_data)))


class TestCleanAsrNoMutation(unittest.TestCase):
    """Regression tests that clean_asr never mutates the caller's EEG."""

    def setUp(self):
        np.random.seed(7)
        n_channels = 8
        n_samples = 2500
        srate = 250.0
        data = np.random.randn(n_channels, n_samples) * 0.5
        for i in range(n_channels):
            for j in range(1, n_samples):
                data[i, j] += 0.8 * data[i, j - 1]
        # Inject a non-finite sample to exercise the in-place NaN-zeroing path
        # that asr_calibrate applies to whatever array it receives.
        data[0, 100] = np.nan
        self.EEG = {
            'data': data,
            'srate': srate,
            'nbchan': n_channels,
            'pnts': n_samples,
            'etc': {},
        }

    def test_does_not_mutate_input_data(self):
        """clean_asr must leave the caller's EEG['data'] (incl. NaNs) unchanged."""
        EEG_in = self.EEG
        original_data = EEG_in['data'].copy()

        EEG_out = clean_asr(EEG_in, ref_maxbadchannels='off')

        # The caller's data is byte-for-byte unchanged, including the NaN that
        # asr_calibrate would otherwise have zeroed in place.
        self.assertTrue(np.array_equal(original_data, EEG_in['data'], equal_nan=True))
        # Output is a distinct object with distinct data.
        self.assertIsNot(EEG_out, EEG_in)
        self.assertIsNot(EEG_out['data'], EEG_in['data'])


class TestCleanAsrDefaultMemory(unittest.TestCase):
    """clean_asr and clean_artifacts must run at high channel counts with their default memory budget."""

    @staticmethod
    def _eeg(n_channels, srate, seconds):
        rng = np.random.default_rng(n_channels)
        data = rng.standard_normal((n_channels, int(seconds * srate)))
        return {'data': data, 'srate': srate, 'nbchan': n_channels, 'pnts': data.shape[1], 'etc': {}}

    def test_clean_asr_default_budget_at_128_channels_1000_hz(self):
        EEG = self._eeg(128, 1000.0, 3)

        cleaned = clean_asr(EEG, ref_maxbadchannels='off')

        self.assertEqual(cleaned['data'].shape, EEG['data'].shape)
        self.assertTrue(np.all(np.isfinite(cleaned['data'])))

    def test_clean_artifacts_default_budget_at_128_channels_1000_hz(self):
        """Burst repair alone, so the test exercises only the MaxMem default flowing into ASR."""
        EEG = self._eeg(128, 1000.0, 3)

        cleaned, _, _, _ = clean_artifacts(
            EEG,
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            FlatlineCriterion='off',
            Highpass='off',
            WindowCriterion='off',
            BurstCriterionRefMaxBadChns='off',
        )

        self.assertEqual(cleaned['data'].shape, EEG['data'].shape)
        self.assertTrue(np.all(np.isfinite(cleaned['data'])))

    def test_clean_asr_explicit_tiny_budget_still_raises(self):
        EEG = self._eeg(8, 250.0, 6)

        with self.assertRaisesRegex(RuntimeError, 'Not enough memory'):
            clean_asr(EEG, ref_maxbadchannels='off', maxmem=0.001)


if __name__ == '__main__':
    unittest.main()
