import unittest
import numpy as np
import tempfile
import os
from unittest.mock import MagicMock, patch

from eegprep.pymat import py2mat, mat2py, default_empty


class TestPy2Mat(unittest.TestCase):
    
    def test_py2mat_none_input(self):
        """Test py2mat with None input."""
        result = py2mat(None)
        expected = np.array([], dtype=object)
        np.testing.assert_array_equal(result, expected)
    
    def test_py2mat_single_dict(self):
        """Test py2mat with single dictionary input."""
        input_dict = {'a': 'hello', 'b': 42}
        result = py2mat(input_dict)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['a'], 'hello')
        self.assertEqual(result[0]['b'], 42)
    
    def test_py2mat_list_of_dicts(self):
        """Test py2mat with list of dictionaries."""
        input_dicts = [
            {'name': 'item1', 'value': 1.5},
            {'name': 'item2', 'value': 2.5}
        ]
        result = py2mat(input_dicts)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['name'], 'item1')
        self.assertEqual(result[0]['value'], 1.5)
        self.assertEqual(result[1]['name'], 'item2')
        self.assertEqual(result[1]['value'], 2.5)
    
    def test_py2mat_non_dict_list(self):
        """Test py2mat with non-dictionary list returns object array."""
        input_list = [1, 2, 'hello', 3.14]
        result = py2mat(input_list)
        
        expected = np.array([1, 2, 'hello', 3.14], dtype=object)
        np.testing.assert_array_equal(result, expected)
    
    def test_py2mat_mixed_list(self):
        """Test py2mat with mixed list (not all dicts) returns object array."""
        input_list = [{'a': 1}, 'not_a_dict', {'b': 2}]
        result = py2mat(input_list)
        
        expected = np.array([{'a': 1}, 'not_a_dict', {'b': 2}], dtype=object)
        np.testing.assert_array_equal(result, expected)
    
    def test_py2mat_nested_dict(self):
        """Test py2mat with nested dictionary."""
        input_dict = {
            'name': 'parent',
            'child': {'nested_name': 'child', 'nested_value': 100}
        }
        result = py2mat(input_dict)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['name'], 'parent')
        # Nested dict should be converted - check that it's structured
        child_field = result[0]['child']
        self.assertTrue(hasattr(child_field, 'dtype') or isinstance(child_field, tuple))
    
    def test_py2mat_empty_dict(self):
        """Test py2mat with empty dictionary."""
        input_dict = {'empty': {}}
        result = py2mat(input_dict)
        
        self.assertEqual(len(result), 1)
        # Empty dict should become empty object array
        empty_field = result[0]['empty']
        self.assertIsInstance(empty_field, np.ndarray)
        self.assertEqual(empty_field.dtype, object)
        self.assertEqual(empty_field.size, 0)
    
    def test_py2mat_numpy_array_values(self):
        """Test py2mat with numpy arrays as values."""
        input_dict = {
            'data': np.array([1, 2, 3]),
            'matrix': np.array([[1, 2], [3, 4]])
        }
        result = py2mat(input_dict)
        
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0]['data'], np.array([1, 2, 3]))
        np.testing.assert_array_equal(result[0]['matrix'], np.array([[1, 2], [3, 4]]))
    
    def test_py2mat_numpy_array_of_dicts(self):
        """Test py2mat with numpy array containing dictionaries."""
        dict_array = np.array([
            {'sensor': 'A', 'reading': 1.2},
            {'sensor': 'B', 'reading': 2.3}
        ], dtype=object)
        
        input_dict = {'measurements': dict_array}
        result = py2mat(input_dict)
        
        self.assertEqual(len(result), 1)
        # Should convert the numpy array of dicts to struct array
        measurements = result[0]['measurements']
        self.assertIsInstance(measurements, np.ndarray)
    
    def test_py2mat_list_and_tuple_conversion(self):
        """Test py2mat converts lists and tuples to numpy arrays."""
        input_dict = {
            'list_data': [1, 2, 3],
            'tuple_data': (4, 5, 6),
            'empty_list': [],
            'nested_list': [[1, 2], [3, 4]]
        }
        result = py2mat(input_dict)
        
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0]['list_data'], np.array([1, 2, 3]))
        np.testing.assert_array_equal(result[0]['tuple_data'], np.array([4, 5, 6]))
        
        # Empty list should become empty object array
        empty_field = result[0]['empty_list']
        self.assertEqual(empty_field.size, 0)
        self.assertEqual(empty_field.dtype, object)
    
    def test_py2mat_string_handling(self):
        """Test py2mat handles strings correctly."""
        input_dicts = [
            {'short': 'hi', 'long': 'this is a longer string'},
            {'short': 'bye', 'long': 'another string'}
        ]
        result = py2mat(input_dicts)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['short'], 'hi')
        self.assertEqual(result[0]['long'], 'this is a longer string')
        self.assertEqual(result[1]['short'], 'bye')
        self.assertEqual(result[1]['long'], 'another string')
    
    def test_py2mat_none_value_handling(self):
        """Test py2mat handles None values appropriately."""
        input_dicts = [
            {'string_field': 'hello', 'int_field': 42, 'float_field': 3.14, 'bool_field': True, 'none_field': None},
            {'string_field': None, 'int_field': None, 'float_field': None, 'bool_field': None, 'none_field': 'not_none'}
        ]
        result = py2mat(input_dicts)
        
        self.assertEqual(len(result), 2)
        
        # First row - check proper values
        self.assertEqual(result[0]['string_field'], 'hello')
        self.assertEqual(result[0]['int_field'], 42)
        self.assertEqual(result[0]['float_field'], 3.14)
        self.assertEqual(result[0]['bool_field'], True)
        
        # Second row - check None handling
        self.assertEqual(result[1]['string_field'], '')  # None -> empty string
        self.assertEqual(result[1]['int_field'], 0)      # None -> 0
        self.assertTrue(np.isnan(result[1]['float_field']))  # None -> NaN
        self.assertEqual(result[1]['bool_field'], False)  # None -> False
    
    def test_py2mat_type_consistency(self):
        """Test py2mat maintains type consistency across records."""
        input_dicts = [
            {'mixed_field': 42},
            {'mixed_field': 'string'},  # Different type - should become object
        ]
        result = py2mat(input_dicts)
        
        self.assertEqual(len(result), 2)
        # Both should be stored as objects due to type inconsistency
        # Note: the implementation might convert everything to string to maintain array consistency
        # Let's just check that both values are preserved in some form
        self.assertTrue(str(result[0]['mixed_field']) == '42' or result[0]['mixed_field'] == 42)
        self.assertEqual(result[1]['mixed_field'], 'string')
    
    def test_py2mat_inhomogeneous_list(self):
        """Test py2mat handles inhomogeneous lists."""
        input_dict = {'mixed_list': [1, 'string', [2, 3], {'nested': 'dict'}]}
        result = py2mat(input_dict)
        
        self.assertEqual(len(result), 1)
        mixed_field = result[0]['mixed_list']
        self.assertIsInstance(mixed_field, np.ndarray)
        self.assertEqual(mixed_field.dtype, object)


class TestMat2Py(unittest.TestCase):
    
    def test_mat2py_dict(self):
        """Test mat2py with dictionary input."""
        input_dict = {'a': np.array([1, 2, 3]), 'b': 'hello'}
        result = mat2py(input_dict)
        
        self.assertIsInstance(result, dict)
        self.assertIn('a', result)
        self.assertIn('b', result)
        self.assertEqual(result['b'], 'hello')
    
    def test_mat2py_empty_list(self):
        """Test mat2py with empty list."""
        result = mat2py([])
        np.testing.assert_array_equal(result, default_empty)
    
    def test_mat2py_list(self):
        """Test mat2py with regular list."""
        input_list = [1, 2, 3]
        result = mat2py(input_list)
        
        self.assertEqual(result, [1, 2, 3])
    
    def test_mat2py_empty_array(self):
        """Test mat2py with empty numpy array."""
        input_array = np.array([])
        result = mat2py(input_array)
        
        np.testing.assert_array_equal(result, default_empty)
    
    def test_mat2py_numeric_array_single_element(self):
        """Test mat2py with single-element numeric array."""
        input_array = np.array([42])
        result = mat2py(input_array)
        
        self.assertEqual(result, 42)
    
    def test_mat2py_numeric_array_multiple_elements(self):
        """Test mat2py with multi-element numeric array."""
        input_array = np.array([1, 2, 3, 4])
        result = mat2py(input_array)
        
        np.testing.assert_array_equal(result, input_array)
    
    def test_mat2py_string_array_single(self):
        """Test mat2py with single-element string array."""
        input_array = np.array(['hello'])
        result = mat2py(input_array)
        
        self.assertEqual(result, 'hello')
    
    def test_mat2py_string_array_multiple(self):
        """Test mat2py with multi-element string array."""
        input_array = np.array(['hello', 'world'])
        result = mat2py(input_array)
        
        self.assertEqual(result, ['hello', 'world'])
    
    def test_mat2py_nested_array_single(self):
        """Test mat2py with single nested array."""
        input_array = np.array([np.array([1, 2, 3])])
        result = mat2py(input_array)
        
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))
    
    def test_mat2py_nested_array_multiple(self):
        """Test mat2py with multiple nested arrays."""
        inner1 = np.array([1, 2])
        inner2 = np.array([3, 4])
        input_array = np.array([inner1, inner2])
        result = mat2py(input_array)
        
        # The actual behavior returns the array as-is for multi-element numeric arrays
        np.testing.assert_array_equal(result, input_array)
    
    def test_mat2py_structured_array_single(self):
        """Test mat2py with single-element structured array."""
        dtype = [('name', 'U10'), ('value', 'f4')]
        input_array = np.array([('test', 3.14)], dtype=dtype)
        result = mat2py(input_array)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['name'], 'test')
        # Use approximate equality for float comparison
        self.assertAlmostEqual(result['value'], 3.14, places=5)
    
    def test_mat2py_structured_array_multiple(self):
        """Test mat2py with multi-element structured array."""
        dtype = [('name', 'U10'), ('value', 'f4')]
        input_array = np.array([('test1', 1.0), ('test2', 2.0)], dtype=dtype)
        result = mat2py(input_array)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['name'], 'test1')
        self.assertEqual(result[1]['name'], 'test2')
    
    def test_mat2py_scalar(self):
        """Test mat2py with scalar values."""
        self.assertEqual(mat2py(42), 42)
        self.assertEqual(mat2py(3.14), 3.14)
        self.assertEqual(mat2py('hello'), 'hello')
        self.assertEqual(mat2py(True), True)
    
    def test_mat2py_nested_single_element_extraction(self):
        """Test mat2py properly extracts nested single elements."""
        # Test the specific case mentioned in the code: [[5, 8]]
        input_array = np.array([np.array([5])])
        result = mat2py(input_array)
        self.assertEqual(result, 5)
        
        # But preserve multi-element inner arrays
        input_array2 = np.array([np.array([5, 8])])
        result2 = mat2py(input_array2)
        np.testing.assert_array_equal(result2, np.array([5, 8]))
    
    def test_mat2py_recarray(self):
        """Test mat2py with recarray input."""
        # Create a simple recarray
        dtype = [('x', 'i4'), ('y', 'f8')]
        data = [(1, 2.5), (2, 3.5)]
        recarray = np.rec.fromrecords(data, dtype=dtype)
        
        result = mat2py(recarray)
        # Should convert to list of dicts
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
    
    def test_mat2py_scipy_mat_struct(self):
        """Test mat2py with scipy.io.matlab.mat_struct."""
        # Create a mock class that behaves like mat_struct
        class MockMatStruct:
            def __init__(self):
                self._fieldnames = ['field1', 'field2', 'tracking']
                self.field1 = 'value1'
                self.field2 = 42
                self.tracking = 'should_be_unsupported'
        
        mock_mat_struct = MockMatStruct()
        
        # Patch the isinstance check for scipy.io.matlab.mat_struct
        with patch('eegprep.pymat.scipy.io.matlab.mat_struct', MockMatStruct):
            result = mat2py(mock_mat_struct)
            
            self.assertIsInstance(result, dict)
            self.assertEqual(result['field1'], 'value1')
            self.assertEqual(result['field2'], 42)
            self.assertEqual(result['tracking'], '<unsupported>')
    
    def test_mat2py_object_with_attributes(self):
        """Test mat2py with object that has attributes."""
        class TestObject:
            def __init__(self):
                self.public_attr = 'public_value'
                self.another_attr = 123
                self._private_attr = 'should_be_ignored'
            
            def method(self):
                return 'should_be_ignored'
        
        test_obj = TestObject()
        result = mat2py(test_obj)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['public_attr'], 'public_value')
        self.assertEqual(result['another_attr'], 123)
        self.assertNotIn('_private_attr', result)
        self.assertNotIn('method', result)
    
    def test_mat2py_object_with_no_accessible_attributes(self):
        """Test mat2py with object that has no accessible attributes."""
        class EmptyObject:
            def __init__(self):
                self._private_only = 'private'
            
            def _private_method(self):
                pass
        
        empty_obj = EmptyObject()
        result = mat2py(empty_obj)
        
        # Should return the original object if no public attributes found
        self.assertEqual(result, empty_obj)
    
    def test_mat2py_fallback(self):
        """Test mat2py fallback for unknown types."""
        class UnknownType:
            pass
        
        unknown = UnknownType()
        result = mat2py(unknown)
        
        # Should return the object as-is
        self.assertEqual(result, unknown)


class TestRoundTrip(unittest.TestCase):
    """Test round-trip conversions where possible."""
    
    def test_simple_dict_roundtrip(self):
        """Test simple dictionary round-trip conversion."""
        original = [{'name': 'test', 'value': 42, 'flag': True}]
        
        # py2mat -> mat2py
        mat_result = py2mat(original)
        py_result = mat2py(mat_result)
        
        # mat2py returns a single dict for single-element struct arrays
        self.assertIsInstance(py_result, dict)
        self.assertEqual(py_result['name'], 'test')
        self.assertEqual(py_result['value'], 42)
        self.assertEqual(py_result['flag'], True)
    
    def test_numeric_array_roundtrip(self):
        """Test numeric array preservation in round-trip."""
        original = [{'data': np.array([1, 2, 3, 4])}]
        
        mat_result = py2mat(original)
        py_result = mat2py(mat_result)
        
        # mat2py returns a single dict for single-element struct arrays
        self.assertIsInstance(py_result, dict)
        np.testing.assert_array_equal(py_result['data'], np.array([1, 2, 3, 4]))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_py2mat_non_list_non_dict_input(self):
        """Test py2mat with input that's neither list nor dict."""
        result = py2mat("just a string")
        self.assertEqual(result, "just a string")
        
        result = py2mat(42)
        self.assertEqual(result, 42)
    
    def test_py2mat_empty_list_input(self):
        """Test py2mat with empty list input."""
        result = py2mat([])
        self.assertEqual(len(result), 0)
    
    def test_mat2py_complex_dtype(self):
        """Test mat2py with complex number arrays."""
        complex_array = np.array([1+2j, 3+4j])
        result = mat2py(complex_array)
        
        np.testing.assert_array_equal(result, complex_array)
    
    def test_mat2py_attribute_access_error(self):
        """Test mat2py handles attribute access errors gracefully."""
        class ProblematicObject:
            @property
            def problematic_attr(self):
                raise RuntimeError("Cannot access this attribute")
            
            def normal_attr(self):
                return "normal"
            
            # Add a real accessible attribute
            def __init__(self):
                self.accessible_attr = "accessible"
        
        prob_obj = ProblematicObject()
        result = mat2py(prob_obj)
        
        # Should handle the error and continue with accessible attributes
        self.assertIsInstance(result, dict)
        # The problematic attribute should be skipped
        self.assertNotIn('problematic_attr', result)
        # But accessible attributes should be included
        self.assertEqual(result['accessible_attr'], 'accessible')
    
    def test_default_empty_constant(self):
        """Test that default_empty is properly defined."""
        self.assertIsInstance(default_empty, np.ndarray)
        self.assertEqual(default_empty.size, 0)


class TestTypeHandling(unittest.TestCase):
    """Test specific type handling cases."""
    
    def test_py2mat_numpy_scalar_types(self):
        """Test py2mat with numpy scalar types."""
        input_dict = {
            'np_int': np.int32(42),
            'np_float': np.float64(3.14),
            'np_bool': np.bool_(True)
        }
        result = py2mat(input_dict)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['np_int'], 42)
        self.assertEqual(result[0]['np_float'], 3.14)
        self.assertEqual(result[0]['np_bool'], True)
    
    def test_mat2py_uint16_strings(self):
        """Test mat2py with uint16 string arrays (mentioned in requirements)."""
        # Create a uint16 array that could represent string data
        uint16_array = np.array([72, 101, 108, 108, 111], dtype=np.uint16)  # "Hello" in ASCII
        result = mat2py(uint16_array)
        
        # Should be treated as numeric array
        np.testing.assert_array_equal(result, uint16_array)
    
    def test_py2mat_nan_values(self):
        """Test py2mat with NaN values."""
        input_dict = {'nan_field': np.nan, 'regular_field': 42}
        result = py2mat(input_dict)
        
        self.assertEqual(len(result), 1)
        self.assertTrue(np.isnan(result[0]['nan_field']))
        self.assertEqual(result[0]['regular_field'], 42)


if __name__ == '__main__':
    unittest.main()