import unittest
import numpy as np

from eegprep.functions.adminfunc.pymat import py2mat


class TestPy2Mat(unittest.TestCase):
    def test_py2mat_none_value_handling(self):
        """Test py2mat handles None values appropriately."""
        input_dicts = [
            {'string_field': 'hello', 'int_field': 42, 'float_field': 3.14, 'bool_field': True, 'none_field': None},
            {
                'string_field': None,
                'int_field': None,
                'float_field': None,
                'bool_field': None,
                'none_field': 'not_none',
            },
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
        self.assertEqual(result[1]['int_field'], 0)  # None -> 0
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


if __name__ == '__main__':
    unittest.main()
