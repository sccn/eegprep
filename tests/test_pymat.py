"""
Test suite for pymat.py - Python-MATLAB data conversion utilities.

This module tests the py2mat and mat2py functions which convert between
Python data structures and MATLAB-compatible formats.
"""

import unittest
import sys
import numpy as np
import scipy.io
import os
import tempfile

# Add src to path for imports
sys.path.insert(0, 'src')
from eegprep.pymat import py2mat, mat2py, default_empty
from eegprep.utils.testing import DebuggableTestCase


class TestPy2Mat(DebuggableTestCase):
    """Test cases for py2mat function."""

    def test_empty_input(self):
        """Test py2mat with empty inputs."""
        # Test None input
        result = py2mat(None)
        self.assertEqual(result.size, 0)
        self.assertEqual(result.dtype, object)
        
        # Test empty list
        result = py2mat([])
        self.assertEqual(result.size, 0)

    def test_single_dict_input(self):
        """Test py2mat with single dictionary input."""
        single_dict = {'a': 1, 'b': 'hello', 'c': 3.14}
        result = py2mat(single_dict)
        
        # Should return structured array with one element
        self.assertEqual(len(result), 1)
        self.assertIn('a', result.dtype.names)
        self.assertIn('b', result.dtype.names)
        self.assertIn('c', result.dtype.names)
        
        # Check values
        self.assertEqual(result[0]['a'], 1)
        self.assertEqual(result[0]['b'], 'hello')
        self.assertAlmostEqual(result[0]['c'], 3.14)

    def test_list_of_dicts_basic(self):
        """Test py2mat with basic list of dictionaries."""
        dicts = [
            {'a': 1, 'b': 'hello'},
            {'a': 2, 'b': 'world'},
            {'a': 3, 'b': 'test'}
        ]
        result = py2mat(dicts)
        
        # Check structure
        self.assertEqual(len(result), 3)
        self.assertIn('a', result.dtype.names)
        self.assertIn('b', result.dtype.names)
        
        # Check values
        for i, expected in enumerate(dicts):
            self.assertEqual(result[i]['a'], expected['a'])
            self.assertEqual(result[i]['b'], expected['b'])

    def test_mixed_data_types(self):
        """Test py2mat with mixed data types."""
        dicts = [
            {'int_val': 42, 'float_val': 3.14, 'str_val': 'hello', 'bool_val': True},
            {'int_val': 24, 'float_val': 2.71, 'str_val': 'world', 'bool_val': False}
        ]
        result = py2mat(dicts)
        
        # Check types are handled correctly
        self.assertEqual(result[0]['int_val'], 42)
        self.assertAlmostEqual(result[0]['float_val'], 3.14)
        self.assertEqual(result[0]['str_val'], 'hello')
        self.assertTrue(result[0]['bool_val'])
        
        self.assertEqual(result[1]['int_val'], 24)
        self.assertAlmostEqual(result[1]['float_val'], 2.71)
        self.assertEqual(result[1]['str_val'], 'world')
        self.assertFalse(result[1]['bool_val'])

    def test_numpy_arrays(self):
        """Test py2mat with numpy arrays as values."""
        dicts = [
            {'data': np.array([1, 2, 3]), 'matrix': np.array([[1, 2], [3, 4]])},
            {'data': np.array([4, 5, 6]), 'matrix': np.array([[5, 6], [7, 8]])}
        ]
        result = py2mat(dicts)
        
        # Arrays should be preserved
        np.testing.assert_array_equal(result[0]['data'], np.array([1, 2, 3]))
        np.testing.assert_array_equal(result[0]['matrix'], np.array([[1, 2], [3, 4]]))
        np.testing.assert_array_equal(result[1]['data'], np.array([4, 5, 6]))
        np.testing.assert_array_equal(result[1]['matrix'], np.array([[5, 6], [7, 8]]))

    def test_nested_dictionaries(self):
        """Test py2mat with nested dictionaries."""
        dicts = [
            {
                'id': 1,
                'config': {'enabled': True, 'threshold': 0.8},
                'metadata': {'name': 'test1', 'version': 1.0}
            },
            {
                'id': 2,
                'config': {'enabled': False, 'threshold': 0.9},
                'metadata': {'name': 'test2', 'version': 2.0}
            }
        ]
        result = py2mat(dicts)
        
        # Check structure
        self.assertEqual(len(result), 2)
        self.assertIn('id', result.dtype.names)
        self.assertIn('config', result.dtype.names)
        self.assertIn('metadata', result.dtype.names)
        
        # Check nested values
        self.assertEqual(result[0]['id'], 1)
        # Nested dictionaries are converted to structured arrays or objects
        self.assertTrue(isinstance(result[0]['config'], (np.ndarray, np.void)))
        self.assertTrue(isinstance(result[0]['metadata'], (np.ndarray, np.void)))

    def test_lists_as_values(self):
        """Test py2mat with lists as values."""
        dicts = [
            {'tags': ['tag1', 'tag2', 'tag3'], 'numbers': [1, 2, 3]},
            {'tags': ['tag4', 'tag5'], 'numbers': [4, 5, 6, 7]}
        ]
        result = py2mat(dicts)
        
        # Lists should be converted to numpy arrays
        self.assertTrue(isinstance(result[0]['tags'], np.ndarray))
        self.assertTrue(isinstance(result[0]['numbers'], np.ndarray))
        
        # Check values
        np.testing.assert_array_equal(result[0]['tags'], np.array(['tag1', 'tag2', 'tag3']))
        np.testing.assert_array_equal(result[0]['numbers'], np.array([1, 2, 3]))

    def test_none_values(self):
        """Test py2mat with None values."""
        dicts = [
            {'a': 1, 'b': None, 'c': 'hello'},
            {'a': 2, 'b': 'world', 'c': None}
        ]
        result = py2mat(dicts)
        
        # None values should be handled appropriately based on field type
        self.assertEqual(result[0]['a'], 1)
        self.assertEqual(result[1]['a'], 2)
        
        # Check that None values are converted to appropriate defaults
        self.assertTrue(isinstance(result[0]['c'], str))
        self.assertTrue(isinstance(result[1]['c'], str))

    def test_empty_dict_values(self):
        """Test py2mat with empty dictionaries as values."""
        dicts = [
            {'id': 1, 'empty_dict': {}, 'data': [1, 2, 3]},
            {'id': 2, 'empty_dict': {}, 'data': [4, 5, 6]}
        ]
        result = py2mat(dicts)
        
        # Empty dicts should be handled gracefully
        self.assertEqual(result[0]['id'], 1)
        self.assertEqual(result[1]['id'], 2)
        self.assertTrue(isinstance(result[0]['empty_dict'], np.ndarray))

    def test_string_length_handling(self):
        """Test py2mat handles varying string lengths correctly."""
        dicts = [
            {'short': 'hi', 'long': 'this is a very long string'},
            {'short': 'bye', 'long': 'short'}
        ]
        result = py2mat(dicts)
        
        # All strings should fit in their fields
        self.assertEqual(result[0]['short'], 'hi')
        self.assertEqual(result[0]['long'], 'this is a very long string')
        self.assertEqual(result[1]['short'], 'bye')
        self.assertEqual(result[1]['long'], 'short')

    def test_list_of_dicts_as_values(self):
        """Test py2mat with lists of dictionaries as values."""
        dicts = [
            {
                'id': 1,
                'measurements': [
                    {'sensor': 'A', 'value': 1.2},
                    {'sensor': 'B', 'value': 2.3}
                ]
            },
            {
                'id': 2,
                'measurements': [
                    {'sensor': 'C', 'value': 3.4}
                ]
            }
        ]
        result = py2mat(dicts)
        
        # Nested list of dicts should be converted to structured arrays
        self.assertEqual(result[0]['id'], 1)
        self.assertEqual(result[1]['id'], 2)
        self.assertTrue(isinstance(result[0]['measurements'], np.ndarray))
        self.assertTrue(isinstance(result[1]['measurements'], np.ndarray))

    def test_numpy_array_of_dicts(self):
        """Test py2mat with numpy arrays of dictionaries."""
        dict_array = np.array([
            {'name': 'sensor1', 'value': 1.1},
            {'name': 'sensor2', 'value': 2.2}
        ], dtype=object)
        
        dicts = [
            {'id': 'device1', 'sensors': dict_array},
            {'id': 'device2', 'sensors': np.array([{'name': 'sensor3', 'value': 3.3}], dtype=object)}
        ]
        result = py2mat(dicts)
        
        # Should handle numpy arrays of dicts
        self.assertEqual(result[0]['id'], 'device1')
        self.assertEqual(result[1]['id'], 'device2')
        self.assertTrue(isinstance(result[0]['sensors'], np.ndarray))
        self.assertTrue(isinstance(result[1]['sensors'], np.ndarray))

    def test_non_dict_input(self):
        """Test py2mat with non-dictionary inputs."""
        # Test with regular list
        regular_list = [1, 2, 3, 'hello']
        result = py2mat(regular_list)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.dtype, object)
        
        # Test with mixed list (not all dicts)
        mixed_list = [{'a': 1}, 'not_a_dict', {'b': 2}]
        result = py2mat(mixed_list)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.dtype, object)

    def test_empty_lists_and_arrays(self):
        """Test py2mat with empty lists and arrays."""
        dicts = [
            {'empty_list': [], 'empty_array': np.array([])},
            {'empty_list': [], 'empty_array': np.array([])}
        ]
        result = py2mat(dicts)
        
        # Empty containers should be handled gracefully
        self.assertTrue(isinstance(result[0]['empty_list'], np.ndarray))
        self.assertTrue(isinstance(result[0]['empty_array'], np.ndarray))

    def test_complex_nested_structure(self):
        """Test py2mat with complex nested structures."""
        complex_dict = {
            'metadata': {
                'version': '1.0',
                'authors': ['Alice', 'Bob'],
                'config': {
                    'enabled': True,
                    'parameters': [1.0, 2.0, 3.0]
                }
            },
            'data': np.random.randn(3, 4),
            'results': [
                {'test': 'A', 'score': 0.95, 'details': {'passed': True}},
                {'test': 'B', 'score': 0.87, 'details': {'passed': True}}
            ]
        }
        
        result = py2mat(complex_dict)
        
        # Should handle complex nesting
        self.assertEqual(len(result), 1)
        self.assertIn('metadata', result.dtype.names)
        self.assertIn('data', result.dtype.names)
        self.assertIn('results', result.dtype.names)


class TestMat2Py(DebuggableTestCase):
    """Test cases for mat2py function."""

    def test_empty_array(self):
        """Test mat2py with empty arrays."""
        empty_array = np.array([])
        result = mat2py(empty_array)
        # Both should be empty arrays, compare using array_equal
        np.testing.assert_array_equal(result, default_empty)

    def test_scalar_values(self):
        """Test mat2py with scalar values."""
        # Test various scalar types
        self.assertEqual(mat2py(42), 42)
        self.assertEqual(mat2py(3.14), 3.14)
        self.assertEqual(mat2py('hello'), 'hello')
        self.assertEqual(mat2py(True), True)

    def test_numeric_arrays(self):
        """Test mat2py with numeric arrays."""
        # Single element array
        single_int = np.array([42])
        self.assertEqual(mat2py(single_int), 42)
        
        single_float = np.array([3.14])
        self.assertEqual(mat2py(single_float), 3.14)
        
        # Multi-element array
        multi_array = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(mat2py(multi_array), multi_array)

    def test_string_arrays(self):
        """Test mat2py with string arrays."""
        # Single string
        single_str = np.array(['hello'])
        self.assertEqual(mat2py(single_str), 'hello')
        
        # Multiple strings
        multi_str = np.array(['hello', 'world', 'test'])
        self.assertEqual(mat2py(multi_str), ['hello', 'world', 'test'])

    def test_structured_arrays(self):
        """Test mat2py with structured arrays."""
        # Create structured array
        dtype = [('name', 'U10'), ('age', 'i4'), ('weight', 'f4')]
        data = np.array([('Alice', 25, 55.0), ('Bob', 30, 70.5)], dtype=dtype)
        
        result = mat2py(data)
        
        # Should return list of dictionaries
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        
        # Check first record
        self.assertEqual(result[0]['name'], 'Alice')
        self.assertEqual(result[0]['age'], 25)
        self.assertAlmostEqual(result[0]['weight'], 55.0)
        
        # Check second record
        self.assertEqual(result[1]['name'], 'Bob')
        self.assertEqual(result[1]['age'], 30)
        self.assertAlmostEqual(result[1]['weight'], 70.5)

    def test_single_structured_element(self):
        """Test mat2py with single-element structured array."""
        dtype = [('x', 'i4'), ('y', 'f4')]
        data = np.array([(10, 3.14)], dtype=dtype)
        
        result = mat2py(data)
        
        # Should return single dictionary
        self.assertIsInstance(result, dict)
        self.assertEqual(result['x'], 10)
        self.assertAlmostEqual(result['y'], 3.14, places=5)  # Less strict precision

    def test_nested_arrays(self):
        """Test mat2py with nested arrays."""
        # Array containing arrays
        nested = np.array([np.array([1, 2, 3]), np.array([4, 5, 6])], dtype=object)
        result = mat2py(nested)
        
        # Should handle nested structure
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_dictionary_input(self):
        """Test mat2py with dictionary input."""
        test_dict = {
            'a': np.array([1, 2, 3]),
            'b': 'hello',
            'c': {
                'nested': np.array([42]),
                'value': 3.14
            }
        }
        
        result = mat2py(test_dict)
        
        # Should recursively process dictionary
        self.assertIsInstance(result, dict)
        np.testing.assert_array_equal(result['a'], np.array([1, 2, 3]))
        self.assertEqual(result['b'], 'hello')
        self.assertIsInstance(result['c'], dict)
        self.assertEqual(result['c']['nested'], 42)  # Single element extracted
        self.assertEqual(result['c']['value'], 3.14)

    def test_list_input(self):
        """Test mat2py with list input."""
        # Empty list
        empty_list = []
        np.testing.assert_array_equal(mat2py(empty_list), default_empty)
        
        # Non-empty list
        test_list = [np.array([1]), 'hello', np.array([2, 3, 4])]
        result = mat2py(test_list)
        
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], 1)  # Single element extracted
        self.assertEqual(result[1], 'hello')
        np.testing.assert_array_equal(result[2], np.array([2, 3, 4]))

    def test_complex_nested_extraction(self):
        """Test mat2py with complex nested single-element extraction."""
        # Test case from the function: obj[0] contains another array with single element
        nested_single = np.array([np.array([42])])
        result = mat2py(nested_single)
        self.assertEqual(result, 42)
        
        # Test case: obj[0] contains array with multiple elements
        nested_multi = np.array([np.array([1, 2, 3])])
        result = mat2py(nested_multi)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_object_with_attributes(self):
        """Test mat2py with objects having attributes."""
        # Create a simple object with attributes
        class TestObject:
            def __init__(self):
                self.public_attr = 42
                self.string_attr = 'hello'
                self._private_attr = 'should_be_ignored'
        
        obj = TestObject()
        result = mat2py(obj)
        
        # Should convert to dictionary with public attributes only
        self.assertIsInstance(result, dict)
        self.assertEqual(result['public_attr'], 42)
        self.assertEqual(result['string_attr'], 'hello')
        self.assertNotIn('_private_attr', result)

    def test_recarray_input(self):
        """Test mat2py with numpy record arrays."""
        # Create a record array
        dtype = [('name', 'U10'), ('value', 'f4')]
        rec_array = np.rec.array([('test', 1.5), ('example', 2.5)], dtype=dtype)
        
        result = mat2py(rec_array)
        
        # Should handle record arrays
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_fallback_cases(self):
        """Test mat2py fallback behavior."""
        # Test with object that doesn't match any conversion rules
        class UnknownObject:
            pass
        
        unknown = UnknownObject()
        result = mat2py(unknown)
        
        # Should return object as-is or convert to dict if it has attributes
        self.assertTrue(isinstance(result, (UnknownObject, dict)))

    def test_single_element_preservation(self):
        """Test that single elements are properly extracted in various contexts."""
        # Test [[5, 8]] case mentioned in mat2py function
        nested_array = np.array([[5, 8]])
        result = mat2py(nested_array)
        np.testing.assert_array_equal(result, np.array([5, 8]))
        
        # Test single element in single element array
        single_nested = np.array([[42]])
        result = mat2py(single_nested)
        self.assertEqual(result, 42)


class TestRoundTripConversion(DebuggableTestCase):
    """Test round-trip conversion between py2mat and mat2py."""

    def test_basic_roundtrip(self):
        """Test basic round-trip conversion."""
        original = [
            {'a': 1, 'b': 'hello'},
            {'a': 2, 'b': 'world'}
        ]
        
        # Convert to MATLAB format and back
        mat_format = py2mat(original)
        converted_back = mat2py(mat_format)
        
        # Should be similar (but not necessarily identical due to type changes)
        self.assertEqual(len(converted_back), len(original))
        for i in range(len(original)):
            self.assertEqual(converted_back[i]['a'], original[i]['a'])
            self.assertEqual(converted_back[i]['b'], original[i]['b'])

    def test_numeric_roundtrip(self):
        """Test round-trip with numeric data."""
        original = [
            {'int_val': 42, 'float_val': 3.14, 'array': np.array([1, 2, 3])},
            {'int_val': 24, 'float_val': 2.71, 'array': np.array([4, 5, 6])}
        ]
        
        mat_format = py2mat(original)
        converted_back = mat2py(mat_format)
        
        # Check numeric values
        self.assertEqual(converted_back[0]['int_val'], 42)
        self.assertAlmostEqual(converted_back[0]['float_val'], 3.14, places=5)
        np.testing.assert_array_equal(converted_back[0]['array'], np.array([1, 2, 3]))

    def test_string_roundtrip(self):
        """Test round-trip with string data."""
        original = [
            {'short': 'hi', 'long': 'this is a longer string'},
            {'short': 'bye', 'long': 'another string'}
        ]
        
        mat_format = py2mat(original)
        converted_back = mat2py(mat_format)
        
        # Strings should be preserved
        self.assertEqual(converted_back[0]['short'], 'hi')
        self.assertEqual(converted_back[0]['long'], 'this is a longer string')
        self.assertEqual(converted_back[1]['short'], 'bye')
        self.assertEqual(converted_back[1]['long'], 'another string')


class TestFileOperations(DebuggableTestCase):
    """Test file I/O operations with MATLAB files."""

    def setUp(self):
        """Set up temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_load_cycle(self):
        """Test saving and loading MATLAB files."""
        original_data = [
            {'name': 'test1', 'value': 1.5, 'array': np.array([1, 2, 3])},
            {'name': 'test2', 'value': 2.5, 'array': np.array([4, 5, 6])}
        ]
        
        # Convert to MATLAB format
        mat_struct = py2mat(original_data)
        
        # Save to .mat file
        mat_file = os.path.join(self.temp_dir, 'test.mat')
        scipy.io.savemat(mat_file, {'data': mat_struct})
        
        # Load from .mat file
        loaded_data = scipy.io.loadmat(mat_file)
        loaded_struct = loaded_data['data'][0]  # Extract from cell array
        
        # Convert back to Python
        converted_data = mat2py(loaded_struct)
        
        # Check that data is preserved through the cycle
        self.assertIsInstance(converted_data, list)
        self.assertEqual(len(converted_data), 2)
        
        # Values might have type changes but should be equivalent
        self.assertEqual(converted_data[0]['name'], 'test1')
        self.assertAlmostEqual(float(converted_data[0]['value']), 1.5)

    def test_complex_structure_file_cycle(self):
        """Test file cycle with complex nested structures."""
        complex_data = {
            'metadata': {
                'version': '1.0',
                'config': {'param1': 1.0, 'param2': 'value'}
            },
            'results': [
                {'test': 'A', 'score': 0.95},
                {'test': 'B', 'score': 0.87}
            ],
            'matrix': np.random.randn(3, 4)
        }
        
        # Convert and save
        mat_struct = py2mat(complex_data)
        mat_file = os.path.join(self.temp_dir, 'complex_test.mat')
        scipy.io.savemat(mat_file, {'complex_data': mat_struct})
        
        # Load and convert back
        loaded = scipy.io.loadmat(mat_file)
        converted_back = mat2py(loaded['complex_data'][0])
        
        # Check structure is preserved
        self.assertIn('metadata', converted_back)
        self.assertIn('results', converted_back)
        self.assertIn('matrix', converted_back)


class TestEdgeCases(DebuggableTestCase):
    """Test edge cases and error conditions."""

    def test_py2mat_edge_cases(self):
        """Test py2mat with edge cases."""
        # Very long strings
        long_string_dict = [{'long_field': 'a' * 1000}]
        result = py2mat(long_string_dict)
        self.assertEqual(result[0]['long_field'], 'a' * 1000)
        
        # Mixed types in same field across records
        mixed_types = [
            {'field': 42},
            {'field': 'string'},  # This forces object type
            {'field': 3.14}
        ]
        result = py2mat(mixed_types)
        # Should handle mixed types by using object dtype
        self.assertEqual(result[0]['field'], 42)
        self.assertEqual(result[1]['field'], 'string')

    def test_mat2py_edge_cases(self):
        """Test mat2py with edge cases."""
        # Very large arrays
        large_array = np.random.randn(100, 100)
        result = mat2py(large_array)
        np.testing.assert_array_equal(result, large_array)
        
        # Arrays with NaN values
        nan_array = np.array([1.0, np.nan, 3.0])
        result = mat2py(nan_array)
        np.testing.assert_array_equal(result, nan_array)

    def test_error_handling(self):
        """Test error handling in conversion functions."""
        # Test with objects that might cause issues
        try:
            # Object with problematic attributes
            class ProblematicObject:
                @property
                def bad_property(self):
                    raise Exception("This property always fails")
                
                def __init__(self):
                    self.good_attr = 42
            
            obj = ProblematicObject()
            result = mat2py(obj)
            
            # Should handle gracefully and include accessible attributes
            self.assertIsInstance(result, dict)
            self.assertEqual(result.get('good_attr'), 42)
            self.assertNotIn('bad_property', result)
            
        except Exception as e:
            # If any exception occurs, it should be handled gracefully
            self.fail(f"mat2py should handle problematic objects gracefully, but raised: {e}")

    def test_memory_efficiency(self):
        """Test that conversions don't create excessive memory usage."""
        # Test with moderately large data
        large_data = []
        for i in range(100):
            large_data.append({
                'id': i,
                'data': np.random.randn(50),
                'metadata': {'name': f'item_{i}', 'value': float(i)}
            })
        
        # Should be able to convert without issues
        mat_format = py2mat(large_data)
        self.assertEqual(len(mat_format), 100)
        
        converted_back = mat2py(mat_format)
        self.assertEqual(len(converted_back), 100)

    def test_type_preservation(self):
        """Test that types are preserved as much as possible."""
        original = [
            {
                'int8': np.int8(42),
                'int16': np.int16(1000),
                'int32': np.int32(100000),
                'int64': np.int64(10000000000),
                'float32': np.float32(3.14),
                'float64': np.float64(2.718281828),
                'complex64': np.complex64(1 + 2j),
                'complex128': np.complex128(3 + 4j)
            }
        ]
        
        mat_format = py2mat(original)
        converted_back = mat2py(mat_format)
        
        # Check that numeric values are preserved (types might change)
        # converted_back might be a dict (single element) or list, handle both
        if isinstance(converted_back, dict):
            data = converted_back
        else:
            data = converted_back[0]
            
        self.assertEqual(int(data['int8']), 42)
        self.assertEqual(int(data['int16']), 1000)
        self.assertAlmostEqual(float(data['float32']), 3.14, places=5)
        self.assertAlmostEqual(float(data['float64']), 2.718281828, places=8)


class TestCompatibilityWithEEGLAB(DebuggableTestCase):
    """Test compatibility with EEGLAB-style data structures."""

    def test_eeglab_style_structure(self):
        """Test conversion of EEGLAB-style data structures."""
        # Simulate simplified EEGLAB structure
        eeglab_style = {
            'data': np.random.randn(32, 1000, 10),  # channels x timepoints x epochs
            'nbchan': 32,
            'pnts': 1000,
            'trials': 10,
            'srate': 500,
            'chanlocs': [
                {'labels': f'Ch{i}', 'X': np.cos(i), 'Y': np.sin(i), 'Z': 0.0}
                for i in range(32)
            ],
            'event': [
                {'latency': 100, 'type': 'stimulus'},
                {'latency': 200, 'type': 'response'}
            ]
        }
        
        # Should be able to convert
        mat_format = py2mat(eeglab_style)
        converted_back = mat2py(mat_format)
        
        # Check key fields are preserved
        self.assertEqual(converted_back['nbchan'], 32)
        self.assertEqual(converted_back['pnts'], 1000)
        self.assertEqual(converted_back['trials'], 10)
        self.assertEqual(converted_back['srate'], 500)
        
        # Check that nested structures are handled
        self.assertIsInstance(converted_back['chanlocs'], list)
        self.assertIsInstance(converted_back['event'], list)

    def test_channel_locations_structure(self):
        """Test conversion of channel location structures."""
        chanlocs = [
            {
                'labels': 'Fp1',
                'X': 0.8090,
                'Y': 0.5878,
                'Z': -0.0000,
                'theta': 36.0,
                'radius': 1.0,
                'sph_theta': 36.0,
                'sph_phi': 0.0,
                'sph_radius': 85.0
            },
            {
                'labels': 'Fp2',
                'X': -0.8090,
                'Y': 0.5878,
                'Z': -0.0000,
                'theta': 144.0,
                'radius': 1.0,
                'sph_theta': 144.0,
                'sph_phi': 0.0,
                'sph_radius': 85.0
            }
        ]
        
        mat_format = py2mat(chanlocs)
        converted_back = mat2py(mat_format)
        
        # Check structure
        self.assertIsInstance(converted_back, list)
        self.assertEqual(len(converted_back), 2)
        
        # Check specific values
        self.assertEqual(converted_back[0]['labels'], 'Fp1')
        self.assertAlmostEqual(converted_back[0]['X'], 0.8090, places=4)
        self.assertEqual(converted_back[1]['labels'], 'Fp2')


if __name__ == '__main__':
    unittest.main()
