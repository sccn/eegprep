from io import BytesIO
import unittest
import numpy as np
from scipy.io import loadmat, savemat

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
        self.assertEqual(result[0]['none_field'], '')
        self.assertEqual(result[1]['none_field'], 'not_none')

        # Second row - check None handling
        self.assertEqual(result[1]['string_field'], '')  # None -> empty string
        self.assertEqual(result[1]['int_field'], 0)  # None -> 0
        self.assertTrue(np.isnan(result[1]['float_field']))  # None -> NaN
        self.assertEqual(result[1]['bool_field'], False)  # None -> False

    def test_py2mat_type_consistency(self):
        """Mixed struct fields retain their MATLAB types regardless of record order."""
        for values in (
            [42, 'string'],
            [42.0, 'string'],
            [7.0, 'boundary', 8.0],
            ['boundary', 7.0, 8.0],
            [7.0, 8.0, 'boundary'],
        ):
            with self.subTest(values=values):
                result = py2mat([{'mixed_field': value} for value in values])
                buffer = BytesIO()
                savemat(buffer, {'records': result})
                buffer.seek(0)
                records = loadmat(buffer, mat_dtype=True)['records'].ravel()

                self.assertEqual(len(records), len(values))
                for record, value in zip(records, values):
                    field = record['mixed_field']
                    self.assertEqual(field.dtype, np.asarray(value).dtype)
                    self.assertEqual(field.item(), value)


if __name__ == '__main__':
    unittest.main()
