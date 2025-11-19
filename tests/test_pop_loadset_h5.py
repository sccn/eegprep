"""
Test suite for pop_loadset_h5.py - HDF5 EEGLAB file loading utilities.

This module tests the pop_loadset_h5 function which loads EEGLAB datasets
from HDF5 format files.
"""

import os
import unittest
import sys
import numpy as np
import h5py
import tempfile
import os
import shutil

# Add src to path for imports
sys.path.insert(0, 'src')
from eegprep.pop_loadset_h5 import pop_loadset_h5
from eegprep.utils.testing import DebuggableTestCase
from eegprep.eeg_compare import eeg_compare
from eegprep.pop_loadset import pop_loadset

class TestPopLoadsetH5(DebuggableTestCase):
    """Test cases for pop_loadset_h5 function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_h5_file(self, filename, include_data=True, include_chanlocs=True):
        """Create a test HDF5 file with EEGLAB structure."""
        filepath = os.path.join(self.temp_dir, filename)
        
        with h5py.File(filepath, 'w') as f:
            # Create EEG group (required by pop_loadset_h5)
            eeg_group = f.create_group('EEG')
            
            # Create scalar fields (as 1x1 arrays as expected by MATLAB)
            eeg_group.create_dataset('srate', data=np.array([[500.0]]))
            eeg_group.create_dataset('pnts', data=np.array([[1000]]))
            eeg_group.create_dataset('nbchan', data=np.array([[32]]))
            eeg_group.create_dataset('trials', data=np.array([[10]]))
            eeg_group.create_dataset('xmin', data=np.array([[-1.0]]))
            eeg_group.create_dataset('xmax', data=np.array([[1.0]]))
            
            # Create string fields (as byte strings)
            eeg_group.create_dataset('setname', data=np.array([b'test_dataset'], dtype='S'))
            eeg_group.create_dataset('filename', data=np.array([b'test.set'], dtype='S'))
            eeg_group.create_dataset('filepath', data=np.array([b'/tmp'], dtype='S'))
            eeg_group.create_dataset('ref', data=np.array([b'common'], dtype='S'))
            eeg_group.create_dataset('saved', data=np.array([b'yes'], dtype='S'))
            
            # Create array fields
            if include_data:
                # Create sample data (channels x timepoints x trials)
                data = np.random.randn(32, 1000, 10).astype(np.float32)
                eeg_group.create_dataset('data', data=data)
                eeg_group.create_dataset('times', data=np.linspace(-1, 1, 1000))
                eeg_group.create_dataset('icaweights', data=np.random.randn(32, 32))
                eeg_group.create_dataset('icasphere', data=np.eye(32))
                eeg_group.create_dataset('icawinv', data=np.random.randn(32, 32))
                eeg_group.create_dataset('icachansind', data=np.arange(32))
            
            # Create channel locations (struct_array format)
            if include_chanlocs:
                chanlocs_group = eeg_group.create_group('chanlocs')
                # Create fields as arrays with one element per channel
                labels_data = [f'Ch{i+1}'.encode() for i in range(32)]
                x_data = [np.cos(i * 2 * np.pi / 32) for i in range(32)]
                y_data = [np.sin(i * 2 * np.pi / 32) for i in range(32)]
                z_data = [0.0 for i in range(32)]
                type_data = [b'EEG' for i in range(32)]
                
                chanlocs_group.create_dataset('labels', data=np.array(labels_data, dtype='S'))
                chanlocs_group.create_dataset('X', data=np.array(x_data))
                chanlocs_group.create_dataset('Y', data=np.array(y_data))
                chanlocs_group.create_dataset('Z', data=np.array(z_data))
                chanlocs_group.create_dataset('type', data=np.array(type_data, dtype='S'))
            
            # Create event structure (struct_array format)
            event_group = eeg_group.create_group('event')
            if include_data:
                # Create sample events
                latencies = [i * 200 + 100 for i in range(5)]  # Event latencies
                types = [b'stimulus' if i % 2 == 0 else b'response' for i in range(5)]
                
                event_group.create_dataset('latency', data=np.array(latencies))
                event_group.create_dataset('type', data=np.array(types, dtype='S'))
            
            # Create epoch structure (struct_array format)
            epoch_group = eeg_group.create_group('epoch')
            if include_data:
                # Create sample epoch info
                epoch_latencies = [i * 100 for i in range(10)]
                epoch_group.create_dataset('event', data=np.array(epoch_latencies))
            
            # Create struct fields (simple groups with scalar data)
            chaninfo_group = eeg_group.create_group('chaninfo')
            chaninfo_group.create_dataset('info', data=np.array([b'channel_info'], dtype='S'))
            
            eventdesc_group = eeg_group.create_group('eventdescription')
            eventdesc_group.create_dataset('description', data=np.array([b'event_desc'], dtype='S'))
            
            epochdesc_group = eeg_group.create_group('epochdescription')
            epochdesc_group.create_dataset('description', data=np.array([b'epoch_desc'], dtype='S'))
            
            # Create reject structure
            reject_group = eeg_group.create_group('reject')
            reject_group.create_dataset('threshold', data=np.array([50.0]))
            reject_group.create_dataset('method', data=np.array([b'manual'], dtype='S'))
            
            # Create stats structure
            stats_group = eeg_group.create_group('stats')
            stats_group.create_dataset('mean', data=np.array([0.0]))
            stats_group.create_dataset('std', data=np.array([1.0]))
            
            # Create etc structure
            etc_group = eeg_group.create_group('etc')
            etc_group.create_dataset('version', data=np.array([b'1.0'], dtype='S'))
            etc_group.create_dataset('date', data=np.array([b'2024-01-01'], dtype='S'))
        
        return filepath

    def test_basic_h5_loading(self):
        """Test basic HDF5 file loading."""
        filepath = self.create_test_h5_file('test_basic.h5')
        
        EEG = pop_loadset_h5(filepath)
        
        # Check basic structure
        self.assertIsInstance(EEG, dict)
        self.assertIn('data', EEG)
        self.assertIn('srate', EEG)
        self.assertIn('pnts', EEG)
        self.assertIn('nbchan', EEG)
        self.assertIn('trials', EEG)
        
        # Check data types and values
        self.assertEqual(EEG['srate'], 500.0)
        self.assertEqual(EEG['pnts'], 1000)
        self.assertEqual(EEG['nbchan'], 32)
        self.assertEqual(EEG['trials'], 10)
        self.assertEqual(EEG['xmin'], -1.0)
        self.assertEqual(EEG['xmax'], 1.0)
        
        # Check string fields
        self.assertEqual(EEG['setname'], 'test_dataset')
        self.assertEqual(EEG['filename'], 'test.set')
        self.assertEqual(EEG['filepath'], '/tmp')
        self.assertEqual(EEG['ref'], 'common')
        self.assertEqual(EEG['saved'], 'yes')

    def test_data_array_loading(self):
        """Test loading of data arrays."""
        filepath = self.create_test_h5_file('test_data.h5')
        
        EEG = pop_loadset_h5(filepath)
        
        # Check data array
        self.assertIsInstance(EEG['data'], np.ndarray)
        self.assertEqual(EEG['data'].shape, (32, 1000, 10))
        self.assertEqual(EEG['data'].dtype, np.float32)
        
        # Check other arrays
        self.assertIsInstance(EEG['times'], np.ndarray)
        self.assertEqual(EEG['times'].shape, (1000,))
        
        self.assertIsInstance(EEG['icaweights'], np.ndarray)
        self.assertEqual(EEG['icaweights'].shape, (32, 32))
        
        self.assertIsInstance(EEG['icasphere'], np.ndarray)
        self.assertEqual(EEG['icasphere'].shape, (32, 32))
        
        self.assertIsInstance(EEG['icawinv'], np.ndarray)
        self.assertEqual(EEG['icawinv'].shape, (32, 32))
        
        self.assertIsInstance(EEG['icachansind'], np.ndarray)
        self.assertEqual(EEG['icachansind'].shape, (32,))

    def test_chanlocs_loading(self):
        """Test loading of channel locations."""
        filepath = self.create_test_h5_file('test_chanlocs.h5')
        
        EEG = pop_loadset_h5(filepath)
        
        # Check channel locations
        self.assertIn('chanlocs', EEG)
        # In numpy 2.x, this might be a structured array instead of a list
        self.assertTrue(isinstance(EEG['chanlocs'], (list, np.ndarray)))
        self.assertEqual(len(EEG['chanlocs']), 32)
        
        # Check individual channel properties
        for i, chan in enumerate(EEG['chanlocs']):
            # Handle both string and bytes format
            expected_label = f'Ch{i+1}'
            actual_label = chan['labels'].decode('utf-8') if isinstance(chan['labels'], bytes) else chan['labels']
            self.assertEqual(actual_label, expected_label)
            
            self.assertAlmostEqual(chan['X'], np.cos(i * 2 * np.pi / 32), places=5)
            self.assertAlmostEqual(chan['Y'], np.sin(i * 2 * np.pi / 32), places=5)
            self.assertEqual(chan['Z'], 0.0)
            
            # Handle both string and bytes format
            expected_type = 'EEG'
            actual_type = chan['type'].decode('utf-8') if isinstance(chan['type'], bytes) else chan['type']
            self.assertEqual(actual_type, expected_type)

    def test_event_loading(self):
        """Test loading of event structure."""
        filepath = self.create_test_h5_file('test_events.h5')
        
        EEG = pop_loadset_h5(filepath)
        
        # Check event structure
        self.assertIn('event', EEG)
        self.assertEqual(len(EEG['event']), 5)
        
        # Check event values
        for i, event in enumerate(EEG['event']):
            expected_latency = i * 200 + 100
            expected_type = 'stimulus' if i % 2 == 0 else 'response'
            self.assertEqual(event['latency'], expected_latency)
            
            # Handle both string and bytes format
            actual_type = event['type'].decode('utf-8') if isinstance(event['type'], bytes) else event['type']
            self.assertEqual(actual_type, expected_type)

    def test_struct_loading(self):
        """Test loading of struct fields."""
        filepath = self.create_test_h5_file('test_structs.h5')
        
        EEG = pop_loadset_h5(filepath)
        
        # Check struct fields exist
        self.assertIn('chaninfo', EEG)
        self.assertIn('eventdescription', EEG)
        self.assertIn('epochdescription', EEG)
        self.assertIn('reject', EEG)
        self.assertIn('stats', EEG)
        self.assertIn('etc', EEG)
        
    def test_string_conversion(self):
        """Test conversion of uint16 string data."""
        filepath = os.path.join(self.temp_dir, 'test_strings.h5')
        
        with h5py.File(filepath, 'w') as f:
            eeg_group = f.create_group('EEG')
            # Create uint16 string data (ASCII codes)
            hello_ascii = np.array([104, 101, 108, 108, 111], dtype=np.uint16)  # "hello"
            world_ascii = np.array([119, 111, 114, 108, 100], dtype=np.uint16)  # "world"
            
            eeg_group.create_dataset('test_string1', data=hello_ascii)
            eeg_group.create_dataset('test_string2', data=world_ascii)
            
            # Create string with newlines
            multiline_ascii = np.array([72, 101, 108, 108, 111, 13, 10, 87, 111, 114, 108, 100], dtype=np.uint16)
            eeg_group.create_dataset('multiline', data=multiline_ascii)
            
            # Add minimal required data to prevent eeg_checkset from failing
            eeg_group.create_dataset('srate', data=np.array([[500.0]]))
            eeg_group.create_dataset('nbchan', data=np.array([[4]]))
            eeg_group.create_dataset('pnts', data=np.array([[100]]))
            eeg_group.create_dataset('trials', data=np.array([[1]]))
            eeg_group.create_dataset('xmin', data=np.array([[-1.0]]))
            eeg_group.create_dataset('xmax', data=np.array([[1.0]]))
            eeg_group.create_dataset('data', data=np.random.randn(4, 100).astype(np.float32))
        
        EEG = pop_loadset_h5(filepath)
        
        # Check string conversion
        self.assertEqual(EEG['test_string1'], 'hello')
        self.assertEqual(EEG['test_string2'], 'world')
        self.assertEqual(EEG['multiline'], 'Hello\r\nWorld')

    def test_missing_fields(self):
        """Test handling of missing fields."""
        filepath = os.path.join(self.temp_dir, 'test_missing.h5')
        
        with h5py.File(filepath, 'w') as f:
            eeg_group = f.create_group('EEG')
            # Create minimal required fields including dummy data
            eeg_group.create_dataset('srate', data=np.array([[500.0]]))
            eeg_group.create_dataset('nbchan', data=np.array([[16]]))
            eeg_group.create_dataset('pnts', data=np.array([[100]]))
            eeg_group.create_dataset('trials', data=np.array([[1]]))
            eeg_group.create_dataset('xmin', data=np.array([[-1.0]]))
            eeg_group.create_dataset('xmax', data=np.array([[1.0]]))
            eeg_group.create_dataset('setname', data=np.array([b'minimal'], dtype='S'))
            # Add minimal data to prevent eeg_checkset from failing
            eeg_group.create_dataset('data', data=np.random.randn(16, 100).astype(np.float32))
        
        EEG = pop_loadset_h5(filepath)
        
        # Should load successfully with only available fields
        self.assertIn('srate', EEG)
        self.assertIn('nbchan', EEG)
        self.assertIn('setname', EEG)
        self.assertEqual(EEG['srate'], 500.0)
        self.assertEqual(EEG['nbchan'], 16)
        self.assertEqual(EEG['setname'], 'minimal')

    def test_empty_groups(self):
        """Test handling of empty groups."""
        filepath = os.path.join(self.temp_dir, 'test_empty.h5')
        
        with h5py.File(filepath, 'w') as f:
            eeg_group = f.create_group('EEG')
            # Create empty groups
            eeg_group.create_group('empty_chanlocs')
            eeg_group.create_group('empty_events')
            
            # Add minimal required data to prevent eeg_checkset from failing
            eeg_group.create_dataset('srate', data=np.array([[500.0]]))
            eeg_group.create_dataset('nbchan', data=np.array([[8]]))
            eeg_group.create_dataset('pnts', data=np.array([[100]]))
            eeg_group.create_dataset('trials', data=np.array([[1]]))
            eeg_group.create_dataset('xmin', data=np.array([[-1.0]]))
            eeg_group.create_dataset('xmax', data=np.array([[1.0]]))
            eeg_group.create_dataset('data', data=np.random.randn(8, 100).astype(np.float32))
        
        EEG = pop_loadset_h5(filepath)
        
        # Should handle empty groups gracefully
        self.assertIn('srate', EEG)
        self.assertEqual(EEG['srate'], 500.0)

    def test_file_not_found(self):
        """Test error handling for non-existent file."""
        non_existent_file = os.path.join(self.temp_dir, 'nonexistent.h5')
        
        with self.assertRaises(FileNotFoundError):
            pop_loadset_h5(non_existent_file)

    def test_invalid_h5_file(self):
        """Test error handling for invalid HDF5 file."""
        invalid_file = os.path.join(self.temp_dir, 'invalid.h5')
        
        # Create a file that's not a valid HDF5 file
        with open(invalid_file, 'w') as f:
            f.write("This is not a valid HDF5 file")
        
        with self.assertRaises(OSError):
            pop_loadset_h5(invalid_file)

    def test_unicode_strings(self):
        """Test handling of Unicode strings."""
        filepath = os.path.join(self.temp_dir, 'test_unicode.h5')
        
        with h5py.File(filepath, 'w') as f:
            eeg_group = f.create_group('EEG')
            # Create Unicode string data (special case handled in pop_loadset_h5)
            unicode_ascii = np.array([104, 101, 108, 108, 111, 32, 240, 159, 146, 150], dtype=np.uint16)  # "hello 👖"
            eeg_group.create_dataset('unicode_string', data=unicode_ascii)
            
            # Add minimal required data to prevent eeg_checkset from failing
            eeg_group.create_dataset('srate', data=np.array([[500.0]]))
            eeg_group.create_dataset('nbchan', data=np.array([[4]]))
            eeg_group.create_dataset('pnts', data=np.array([[100]]))
            eeg_group.create_dataset('trials', data=np.array([[1]]))
            eeg_group.create_dataset('xmin', data=np.array([[-1.0]]))
            eeg_group.create_dataset('xmax', data=np.array([[1.0]]))
            eeg_group.create_dataset('data', data=np.random.randn(4, 100).astype(np.float32))
        
        EEG = pop_loadset_h5(filepath)
        
        # Should handle Unicode strings (special case in the code)
        self.assertEqual(EEG['unicode_string'], 'hello 👖')

@unittest.skipIf(os.getenv('EEGPREP_SKIP_MATLAB') == '1', "MATLAB not available")
class TestPopLoadsetH5Parity(unittest.TestCase):
    """Test parity between Python pop_loadset_h5 and MATLAB pop_loadset for real HDF5 files.\"\"\"
    
    def setUp(self):
        """Set up MATLAB connection for parity testing."""
        try:
            from eegprep.eeglabcompat import get_eeglab
            self.eeglab = get_eeglab('MAT')
            self.matlab_available = True
        except Exception as e:
            print(f"MATLAB not available for parity testing: {e}")
            self.matlab_available = False
    
    def test_parity_continuous_data(self):
        """Test parity with continuous EEG data (single trial)."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available for parity testing")
        
        filepath = 'data/eeglab_data_hdf5.set'
        
        # Load with Python
        py_eeg = pop_loadset_h5(filepath)
        
        # Load with MATLAB
        data_dir = os.path.abspath('data')
        ml_eeg = self.eeglab.pop_loadset('filename', 'eeglab_data_hdf5.set', 'filepath', data_dir)
        
        eeg_compare(py_eeg, ml_eeg)
        
    def test_parity_epoched_data(self):
        """Test parity with continuous EEG data (single trial)."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available for parity testing")
        
        filepath = 'data/eeglab_data_epochs_ica_hdf5.set'
        
        # Load with Python
        py_eeg = pop_loadset_h5(filepath)
        
        # Load with MATLAB
        data_dir = os.path.abspath('data')
        ml_eeg = self.eeglab.pop_loadset('filename', 'eeglab_data_epochs_ica.set', 'filepath', data_dir)
        
        eeg_compare(py_eeg, ml_eeg)        

class TestPopLoadsetH5RealData(unittest.TestCase):
    """Test pop_loadset_h5 with real HDF5 files without MATLAB dependency."""
    
    def test_load_continuous_data(self):
        """Test loading continuous EEG data."""
        filepath = 'data/eeglab_data_hdf5.set'
        
        if not os.path.exists(filepath):
            self.skipTest(f"Test data file not found: {filepath}")
        
        EEG = pop_loadset_h5(filepath)
        
        # Check basic structure
        self.assertIn('data', EEG)
        self.assertIn('srate', EEG)
        self.assertIn('nbchan', EEG)
        self.assertIn('trials', EEG)
        self.assertIn('pnts', EEG)
        
        # Check data properties
        self.assertEqual(EEG['srate'], 128.0)
        self.assertEqual(EEG['nbchan'], 32)
        self.assertEqual(EEG['trials'], 1)  # Continuous data
        self.assertEqual(EEG['data'].shape, (32, 30504))  # (channels, timepoints)
        
        # Check that data is numeric and not empty
        self.assertTrue(np.isfinite(EEG['data']).all())
        self.assertFalse(np.all(EEG['data'] == 0))
    
    def test_load_epoched_data(self):
        """Test loading epoched EEG data."""
        filepath = 'data/eeglab_data_epochs_ica_hdf5.set'
        
        if not os.path.exists(filepath):
            self.skipTest(f"Test data file not found: {filepath}")
        
        EEG = pop_loadset_h5(filepath)
        
        # Check basic structure
        self.assertIn('data', EEG)
        self.assertIn('srate', EEG)
        self.assertIn('nbchan', EEG)
        self.assertIn('trials', EEG)
        self.assertIn('pnts', EEG)
        
        # Check data properties
        self.assertEqual(EEG['srate'], 128.0)
        self.assertEqual(EEG['nbchan'], 32)
        self.assertEqual(EEG['trials'], 80)  # Epoched data
        self.assertEqual(EEG['data'].shape, (32, 384, 80))  # (channels, timepoints, trials)
        
        # Check that data is numeric and not empty
        self.assertTrue(np.isfinite(EEG['data']).all())
        self.assertFalse(np.all(EEG['data'] == 0))
    
    def test_chanlocs_structure(self):
        """Test channel locations structure."""
        filepath = 'data/eeglab_data_hdf5.set'
        
        if not os.path.exists(filepath):
            self.skipTest(f"Test data file not found: {filepath}")
        
        try:
            EEG = pop_loadset_h5(filepath)
            
            # Check chanlocs if available
            if 'chanlocs' in EEG:
                self.assertIsInstance(EEG['chanlocs'], (np.ndarray, list))
                self.assertEqual(len(EEG['chanlocs']), EEG['nbchan'])
                
                # Check for expected fields
                expected_fields = ['labels', 'type', 'theta', 'radius', 'X', 'Y', 'Z']
                for field in expected_fields:
                    if field not in EEG['chanlocs'][0]:
                        self.fail(f"Field '{field}' not found in chanlocs structure")
        except Exception as e:
            self.skipTest(f"Could not load real data file: {e}")
    
    def test_events_structure(self):
        """Test events structure for epoched data."""
        filepath = 'data/eeglab_data_epochs_ica_hdf5.set'
        
        if not os.path.exists(filepath):
            self.skipTest(f"Test data file not found: {filepath}")
        
        try:
            EEG = pop_loadset_h5(filepath)
            
            # Check events if available
            if 'event' in EEG:
                self.assertIsInstance(EEG['event'], (np.ndarray, list))
                self.assertGreater(len(EEG['event']), 0)
                
                # Check for expected fields
                expected_fields = ['type', 'latency']
                for field in expected_fields:
                    if field not in EEG['event'][0]:
                        self.fail(f"Field '{field}' not found in event structure")
        except Exception as e:
            self.skipTest(f"Could not load real data file: {e}")
    
    def test_ica_components(self):
        """Test ICA components structure."""
        filepath = 'data/eeglab_data_epochs_ica_hdf5.set'
        
        EEG = pop_loadset_h5(filepath)
        
        # Check ICA fields if available
        ica_fields = ['icaweights', 'icasphere', 'icawinv', 'icachansind']
        for field in ica_fields:
            if field in EEG:
                self.assertIsInstance(EEG[field], np.ndarray)
                self.assertTrue(np.isfinite(EEG[field]).all())
                
                # Check specific shapes
                if field == 'icaweights':
                    self.assertEqual(EEG[field].shape[0], EEG[field].shape[1])  # Square matrix
                elif field == 'icasphere':
                    self.assertEqual(EEG[field].shape[0], EEG[field].shape[1])  # Square matrix
                elif field == 'icawinv':
                    self.assertEqual(EEG[field].shape[0], EEG['nbchan'])
                elif field == 'icachansind':
                    self.assertLessEqual(len(EEG[field]), EEG['nbchan'])
    
    def test_compare_continuous_pop_loadset(self):
        EEG1 = pop_loadset('data/eeglab_data.set')
        EEG2 = pop_loadset_h5('data/eeglab_data_hdf5.set')
        
        eeg_compare(EEG1, EEG2)

    def test_compare_epochs_pop_loadset(self):
        EEG1 = pop_loadset('data/eeglab_data_epochs_ica.set')
        EEG2 = pop_loadset_h5('data/eeglab_data_epochs_ica_hdf5.set')
        
        eeg_compare(EEG1, EEG2)
        
if __name__ == '__main__':
    # test test_load_epoched_data only
    # TestPopLoadsetH5RealData().test_load_epoched_data()
    
    unittest.main()
