"""
Test suite for pop_loadset_h5.py - HDF5 EEGLAB file loading utilities.

This module tests the pop_loadset_h5 function which loads EEGLAB datasets
from HDF5 format files.
"""

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
            # Create basic EEG structure
            if include_data:
                # Create sample data (channels x timepoints x trials)
                data = np.random.randn(32, 1000, 10).astype(np.float32)
                f.create_dataset('data', data=data)
            
            # Create scalar fields
            f.create_dataset('srate', data=np.array([[500.0]]))
            f.create_dataset('pnts', data=np.array([[1000]]))
            f.create_dataset('nbchan', data=np.array([[32]]))
            f.create_dataset('trials', data=np.array([[10]]))
            f.create_dataset('xmin', data=np.array([[-1.0]]))
            f.create_dataset('xmax', data=np.array([[1.0]]))
            
            # Create string fields
            f.create_dataset('setname', data=np.array([b'test_dataset'], dtype='S'))
            f.create_dataset('filename', data=np.array([b'test.set'], dtype='S'))
            f.create_dataset('filepath', data=np.array([b'/tmp'], dtype='S'))
            f.create_dataset('ref', data=np.array([b'common'], dtype='S'))
            f.create_dataset('saved', data=np.array([b'yes'], dtype='S'))
            
            # Create array fields
            if include_data:
                f.create_dataset('times', data=np.linspace(-1, 1, 1000))
                f.create_dataset('icaweights', data=np.random.randn(32, 32))
                f.create_dataset('icasphere', data=np.eye(32))
                f.create_dataset('icawinv', data=np.random.randn(32, 32))
                f.create_dataset('icachansind', data=np.arange(32))
            
            # Create channel locations
            if include_chanlocs:
                chanlocs_group = f.create_group('chanlocs')
                for i in range(32):
                    chan_group = chanlocs_group.create_group(str(i))
                    chan_group.create_dataset('labels', data=np.array([f'Ch{i+1}'.encode()], dtype='S'))
                    chan_group.create_dataset('X', data=np.array([np.cos(i * 2 * np.pi / 32)]))
                    chan_group.create_dataset('Y', data=np.array([np.sin(i * 2 * np.pi / 32)]))
                    chan_group.create_dataset('Z', data=np.array([0.0]))
                    chan_group.create_dataset('type', data=np.array([b'EEG'], dtype='S'))
            
            # Create event structure
            event_group = f.create_group('event')
            for i in range(5):
                event_group.create_dataset(str(i), data=np.array([i * 200]))  # Event latencies
            
            # Create epoch structure
            epoch_group = f.create_group('epoch')
            for i in range(10):
                epoch_group.create_dataset(str(i), data=np.array([i * 100]))  # Epoch latencies
            
            # Create other structures
            f.create_dataset('chaninfo', data=np.array([b'info'], dtype='S'))
            f.create_dataset('eventdescription', data=np.array([b'description'], dtype='S'))
            f.create_dataset('epochdescription', data=np.array([b'epoch_desc'], dtype='S'))
            
            # Create reject structure
            reject_group = f.create_group('reject')
            reject_group.create_dataset('threshold', data=np.array([50.0]))
            reject_group.create_dataset('method', data=np.array([b'manual'], dtype='S'))
            
            # Create stats structure
            stats_group = f.create_group('stats')
            stats_group.create_dataset('mean', data=np.array([0.0]))
            stats_group.create_dataset('std', data=np.array([1.0]))
            
            # Create etc structure
            etc_group = f.create_group('etc')
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
        self.assertIsInstance(EEG['chanlocs'], np.ndarray)
        self.assertEqual(len(EEG['chanlocs']), 32)
        
        # Check individual channel properties
        for i, chan in enumerate(EEG['chanlocs']):
            self.assertIn('labels', chan.dtype.names)
            self.assertIn('X', chan.dtype.names)
            self.assertIn('Y', chan.dtype.names)
            self.assertIn('Z', chan.dtype.names)
            self.assertIn('type', chan.dtype.names)
            
            self.assertEqual(chan['labels'], f'Ch{i+1}')
            self.assertAlmostEqual(chan['X'], np.cos(i * 2 * np.pi / 32), places=5)
            self.assertAlmostEqual(chan['Y'], np.sin(i * 2 * np.pi / 32), places=5)
            self.assertEqual(chan['Z'], 0.0)
            self.assertEqual(chan['type'], 'EEG')

    def test_event_loading(self):
        """Test loading of event structure."""
        filepath = self.create_test_h5_file('test_events.h5')
        
        EEG = pop_loadset_h5(filepath)
        
        # Check event structure
        self.assertIn('event', EEG)
        self.assertIsInstance(EEG['event'], np.ndarray)
        self.assertEqual(len(EEG['event']), 5)
        
        # Check event values
        for i, event in enumerate(EEG['event']):
            self.assertEqual(event, i * 200)

    def test_epoch_loading(self):
        """Test loading of epoch structure."""
        filepath = self.create_test_h5_file('test_epochs.h5')
        
        EEG = pop_loadset_h5(filepath)
        
        # Check epoch structure
        self.assertIn('epoch', EEG)
        self.assertIsInstance(EEG['epoch'], np.ndarray)
        self.assertEqual(len(EEG['epoch']), 10)
        
        # Check epoch values
        for i, epoch in enumerate(EEG['epoch']):
            self.assertEqual(epoch, i * 100)

    def test_struct_loading(self):
        """Test loading of struct fields."""
        filepath = self.create_test_h5_file('test_structs.h5')
        
        EEG = pop_loadset_h5(filepath)
        
        # Check struct fields
        self.assertIn('chaninfo', EEG)
        self.assertIn('eventdescription', EEG)
        self.assertIn('epochdescription', EEG)
        self.assertIn('reject', EEG)
        self.assertIn('stats', EEG)
        self.assertIn('etc', EEG)
        
        # Check values
        self.assertEqual(EEG['chaninfo'], 'info')
        self.assertEqual(EEG['eventdescription'], 'description')
        self.assertEqual(EEG['epochdescription'], 'epoch_desc')

    def test_reject_structure(self):
        """Test loading of reject structure."""
        filepath = self.create_test_h5_file('test_reject.h5')
        
        EEG = pop_loadset_h5(filepath)
        
        # Check reject structure
        self.assertIn('reject', EEG)
        reject = EEG['reject']
        
        self.assertIn('threshold', reject.dtype.names)
        self.assertIn('method', reject.dtype.names)
        
        self.assertEqual(reject['threshold'], 50.0)
        self.assertEqual(reject['method'], 'manual')

    def test_stats_structure(self):
        """Test loading of stats structure."""
        filepath = self.create_test_h5_file('test_stats.h5')
        
        EEG = pop_loadset_h5(filepath)
        
        # Check stats structure
        self.assertIn('stats', EEG)
        stats = EEG['stats']
        
        self.assertIn('mean', stats.dtype.names)
        self.assertIn('std', stats.dtype.names)
        
        self.assertEqual(stats['mean'], 0.0)
        self.assertEqual(stats['std'], 1.0)

    def test_etc_structure(self):
        """Test loading of etc structure."""
        filepath = self.create_test_h5_file('test_etc.h5')
        
        EEG = pop_loadset_h5(filepath)
        
        # Check etc structure
        self.assertIn('etc', EEG)
        etc = EEG['etc']
        
        self.assertIn('version', etc.dtype.names)
        self.assertIn('date', etc.dtype.names)
        
        self.assertEqual(etc['version'], '1.0')
        self.assertEqual(etc['date'], '2024-01-01')

    def test_string_conversion(self):
        """Test conversion of uint16 string data."""
        filepath = os.path.join(self.temp_dir, 'test_strings.h5')
        
        with h5py.File(filepath, 'w') as f:
            # Create uint16 string data (ASCII codes)
            hello_ascii = np.array([104, 101, 108, 108, 111], dtype=np.uint16)  # "hello"
            world_ascii = np.array([119, 111, 114, 108, 100], dtype=np.uint16)  # "world"
            
            f.create_dataset('test_string1', data=hello_ascii)
            f.create_dataset('test_string2', data=world_ascii)
            
            # Create string with newlines
            multiline_ascii = np.array([72, 101, 108, 108, 111, 13, 10, 87, 111, 114, 108, 100], dtype=np.uint16)
            f.create_dataset('multiline', data=multiline_ascii)
        
        EEG = pop_loadset_h5(filepath)
        
        # Check string conversion
        self.assertEqual(EEG['test_string1'], 'hello')
        self.assertEqual(EEG['test_string2'], 'world')
        self.assertEqual(EEG['multiline'], 'Hello\r\nWorld')

    def test_missing_fields(self):
        """Test handling of missing fields."""
        filepath = os.path.join(self.temp_dir, 'test_missing.h5')
        
        with h5py.File(filepath, 'w') as f:
            # Only create some basic fields
            f.create_dataset('srate', data=np.array([[500.0]]))
            f.create_dataset('nbchan', data=np.array([[16]]))
            f.create_dataset('setname', data=np.array([b'minimal'], dtype='S'))
        
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
            # Create empty groups
            f.create_group('empty_chanlocs')
            f.create_group('empty_events')
            
            # Add some basic data
            f.create_dataset('srate', data=np.array([[500.0]]))
        
        EEG = pop_loadset_h5(filepath)
        
        # Should handle empty groups gracefully
        self.assertIn('srate', EEG)
        self.assertEqual(EEG['srate'], 500.0)

    def test_reference_handling(self):
        """Test handling of HDF5 references."""
        filepath = os.path.join(self.temp_dir, 'test_refs.h5')
        
        with h5py.File(filepath, 'w') as f:
            # Create referenced data
            ref_data = f.create_dataset('referenced_data', data=np.array([42, 43, 44]))
            
            # Create group with references
            ref_group = f.create_group('ref_group')
            ref_group.create_dataset('ref1', data=ref_data.ref)
            ref_group.create_dataset('ref2', data=ref_data.ref)
        
        EEG = pop_loadset_h5(filepath)
        
        # Should handle references by dereferencing them
        self.assertIn('ref_group', EEG)
        ref_group = EEG['ref_group']
        self.assertIsInstance(ref_group, np.ndarray)
        
        # Check that references were dereferenced
        for item in ref_group:
            self.assertTrue(np.array_equal(item['ref1'], np.array([42, 43, 44])))
            self.assertTrue(np.array_equal(item['ref2'], np.array([42, 43, 44])))

    def test_nested_array_handling(self):
        """Test handling of nested arrays."""
        filepath = os.path.join(self.temp_dir, 'test_nested.h5')
        
        with h5py.File(filepath, 'w') as f:
            # Create nested array structure
            nested_group = f.create_group('nested')
            nested_group.create_dataset('array1', data=np.array([[1, 2], [3, 4]]))
            nested_group.create_dataset('array2', data=np.array([[5, 6], [7, 8]]))
        
        EEG = pop_loadset_h5(filepath)
        
        # Should handle nested arrays
        self.assertIn('nested', EEG)
        nested = EEG['nested']
        
        self.assertIn('array1', nested.dtype.names)
        self.assertIn('array2', nested.dtype.names)
        
        np.testing.assert_array_equal(nested['array1'][0], np.array([[1, 2], [3, 4]]))
        np.testing.assert_array_equal(nested['array2'][0], np.array([[5, 6], [7, 8]]))

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

    def test_large_dataset(self):
        """Test loading of large dataset."""
        filepath = os.path.join(self.temp_dir, 'test_large.h5')
        
        with h5py.File(filepath, 'w') as f:
            # Create large data array
            large_data = np.random.randn(64, 10000, 100).astype(np.float32)
            f.create_dataset('data', data=large_data)
            
            # Create corresponding metadata
            f.create_dataset('srate', data=np.array([[1000.0]]))
            f.create_dataset('pnts', data=np.array([[10000]]))
            f.create_dataset('nbchan', data=np.array([[64]]))
            f.create_dataset('trials', data=np.array([[100]]))
            f.create_dataset('xmin', data=np.array([[-5.0]]))
            f.create_dataset('xmax', data=np.array([[5.0]]))
        
        EEG = pop_loadset_h5(filepath)
        
        # Should load large dataset successfully
        self.assertIn('data', EEG)
        self.assertEqual(EEG['data'].shape, (64, 10000, 100))
        self.assertEqual(EEG['data'].dtype, np.float32)
        self.assertEqual(EEG['srate'], 1000.0)
        self.assertEqual(EEG['pnts'], 10000)
        self.assertEqual(EEG['nbchan'], 64)
        self.assertEqual(EEG['trials'], 100)

    def test_mixed_data_types(self):
        """Test loading of mixed data types."""
        filepath = os.path.join(self.temp_dir, 'test_mixed.h5')
        
        with h5py.File(filepath, 'w') as f:
            # Create various data types
            f.create_dataset('int_data', data=np.array([[42]]))
            f.create_dataset('float_data', data=np.array([[3.14]]))
            f.create_dataset('bool_data', data=np.array([[True]]))
            f.create_dataset('complex_data', data=np.array([[1 + 2j]]))
            f.create_dataset('string_data', data=np.array([b'mixed_types'], dtype='S'))
        
        EEG = pop_loadset_h5(filepath)
        
        # Should handle mixed data types
        self.assertEqual(EEG['int_data'], 42)
        self.assertEqual(EEG['float_data'], 3.14)
        self.assertEqual(EEG['bool_data'], True)
        self.assertEqual(EEG['complex_data'], 1 + 2j)
        self.assertEqual(EEG['string_data'], 'mixed_types')

    def test_unicode_strings(self):
        """Test handling of Unicode strings."""
        filepath = os.path.join(self.temp_dir, 'test_unicode.h5')
        
        with h5py.File(filepath, 'w') as f:
            # Create Unicode string data
            unicode_ascii = np.array([104, 101, 108, 108, 111, 32, 240, 159, 146, 150], dtype=np.uint16)  # "hello 👖"
            f.create_dataset('unicode_string', data=unicode_ascii)
        
        EEG = pop_loadset_h5(filepath)
        
        # Should handle Unicode strings
        self.assertEqual(EEG['unicode_string'], 'hello 👖')

    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        filepath = os.path.join(self.temp_dir, 'test_memory.h5')
        
        with h5py.File(filepath, 'w') as f:
            # Create moderately large dataset
            data = np.random.randn(128, 5000, 50).astype(np.float32)
            f.create_dataset('data', data=data)
            
            # Create metadata
            f.create_dataset('srate', data=np.array([[500.0]]))
            f.create_dataset('pnts', data=np.array([[5000]]))
            f.create_dataset('nbchan', data=np.array([[128]]))
            f.create_dataset('trials', data=np.array([[50]]))
        
        # Should load without memory issues
        EEG = pop_loadset_h5(filepath)
        
        self.assertIn('data', EEG)
        self.assertEqual(EEG['data'].shape, (128, 5000, 50))
        self.assertEqual(EEG['data'].dtype, np.float32)

    def test_roundtrip_compatibility(self):
        """Test compatibility with EEGLAB round-trip operations."""
        filepath = self.create_test_h5_file('test_roundtrip.h5')
        
        # Load the file
        EEG = pop_loadset_h5(filepath)
        
        # Check that the structure is compatible with EEGLAB operations
        self.assertIsInstance(EEG, dict)
        self.assertIn('data', EEG)
        self.assertIn('srate', EEG)
        self.assertIn('nbchan', EEG)
        self.assertIn('pnts', EEG)
        self.assertIn('trials', EEG)
        
        # Check that data can be accessed in EEGLAB-compatible way
        data_shape = EEG['data'].shape
        self.assertEqual(data_shape[0], EEG['nbchan'])
        self.assertEqual(data_shape[1], EEG['pnts'])
        self.assertEqual(data_shape[2], EEG['trials'])
        
        # Check that channel locations are properly structured
        if 'chanlocs' in EEG:
            self.assertIsInstance(EEG['chanlocs'], np.ndarray)
            self.assertEqual(len(EEG['chanlocs']), EEG['nbchan'])


if __name__ == '__main__':
    unittest.main()
