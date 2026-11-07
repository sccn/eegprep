"""Stream 9 regressions for F15/F26/F34 and issues #497/#561."""

import copy

import h5py
import numpy as np
import pytest
import scipy.io

from eegprep import pop_loadbv, pop_loadset, pop_saveset
from eegprep.functions.popfunc.pop_loadset_h5 import pop_loadset_h5
from tests.test_pop_loadbv import _write_binary_brainvision
from tests.test_pop_loadset import _singleton_eeg


def _h5_value(group, name, value):
    if isinstance(value, str):
        codes = np.frombuffer(value.encode('utf-16-le'), dtype='<u2').reshape(-1, 1)
        ds = group.create_dataset(name, data=codes)
        ds.attrs['MATLAB_class'] = np.bytes_('char')
    else:
        ds = group.create_dataset(name, data=np.asarray(value).T)
        ds.attrs['MATLAB_class'] = np.bytes_('double')
    return ds


def _h5_dataset(path, data, *, events=True):
    with h5py.File(path, 'w', userblock_size=512) as file:
        eeg = file.create_group('EEG')
        refs = file.create_group('#refs#')
        for key, value in _singleton_eeg(data).items():
            if key in {'data', 'nbchan', 'pnts', 'trials', 'srate', 'xmin', 'xmax'}:
                _h5_value(eeg, key, value)
        for field, records in [
            ('chanlocs', [{'labels': f'EEG{n}µ'} for n in range(data.shape[0])]),
            ('event', [{'type': 'stimulus', 'latency': 2.0}] if events else []),
        ]:
            if not records:
                ds = eeg.create_dataset(field, data=np.array([0, 0], dtype='uint64'))
                ds.attrs['MATLAB_empty'] = 1
                continue
            group = eeg.create_group(field)
            group.attrs['MATLAB_class'] = np.bytes_('struct')
            for key in records[0]:
                pointers = group.create_dataset(key, (len(records), 1), dtype=h5py.ref_dtype)
                for i, record in enumerate(records):
                    pointers[i, 0] = _h5_value(refs, f'{field}_{key}_{i}', record[key]).ref
        etc = eeg.create_group('etc')
        cell = etc.create_dataset('notes', (2, 1), dtype=h5py.ref_dtype)
        cell.attrs['MATLAB_class'] = np.bytes_('cell')
        cell[0, 0] = _h5_value(refs, 'note', 'complete text µ').ref
        cell[1, 0] = _h5_value(refs, 'vector', [[1, 2, 3]]).ref


@pytest.mark.parametrize('shape', [(1, 5), (3, 3), (2, 5, 2)])
@pytest.mark.parametrize('events', [False, True])
def test_f15_h5_dereferences_strings_vectors_and_empty_structs(tmp_path, shape, events):
    data = np.arange(np.prod(shape), dtype=float).reshape(shape)
    path = tmp_path / 'v73.set'
    _h5_dataset(path, data, events=events)
    for loader in (pop_loadset, pop_loadset_h5):
        eeg = copy.deepcopy(loader(path))
        np.testing.assert_array_equal(eeg['data'], data)
        assert eeg['chanlocs'][0]['labels'] == 'EEG0µ'
        assert eeg['etc']['notes'][0] == 'complete text µ'
        np.testing.assert_array_equal(eeg['etc']['notes'][1], [1, 2, 3])
        assert len(eeg['event']) == int(events)
        if events:
            assert eeg['event'][0]['type'] == 'stimulus'


@pytest.mark.parametrize('field', ['data', 'datfile'])
@pytest.mark.parametrize(
    'reference', ['../outside.fdt', '/tmp/outside.fdt', r'C:\outside.fdt', r'..\outside.fdt', 'link.fdt']
)
def test_561_loadset_refuses_escaping_sidecars(tmp_path, field, reference):
    eeg = _singleton_eeg(np.zeros((1, 4)))
    eeg.update(data=reference if field == 'data' else 'in set file', datfile=reference)
    (tmp_path / 'link.fdt').symlink_to(tmp_path.parent / 'outside.fdt')
    path = tmp_path / 'unsafe.set'
    scipy.io.savemat(path, eeg)
    with pytest.raises(ValueError, match=r'pop_loadset.*directory'):
        pop_loadset(path)


@pytest.mark.parametrize('field', ['DataFile', 'MarkerFile'])
@pytest.mark.parametrize(
    'reference', ['../outside.dat', '/tmp/outside.dat', r'C:\outside.dat', r'..\outside.dat', 'link.dat']
)
def test_561_loadbv_refuses_escaping_references_before_fallback(tmp_path, field, reference):
    path = _write_binary_brainvision(tmp_path, 'test', np.zeros((2, 8)))
    (tmp_path / 'link.dat').symlink_to(tmp_path.parent / 'outside.dat')
    old = 'test.dat' if field == 'DataFile' else 'test.vmrk'
    path.write_text(path.read_text().replace(f'{field}={old}', f'{field}={reference}'))
    with pytest.raises(ValueError, match=r'pop_loadbv.*directory'):
        pop_loadbv(path)


def test_f26_save_as_preserves_two_files_and_adds_extension(tmp_path):
    eeg = _singleton_eeg(np.arange(12).reshape(2, 6))
    pop_saveset(eeg, tmp_path / 'original.set', savemode='twofiles')
    loaded = pop_loadset(tmp_path / 'original.set')
    pop_saveset(loaded, tmp_path / 'copy')
    assert (tmp_path / 'copy.fdt').is_file()
    assert loaded['filename'] == 'copy.set'
    np.testing.assert_array_equal(pop_loadset(tmp_path / 'copy.set')['data'], eeg['data'])
    pop_saveset(loaded, tmp_path / 'embedded', savemode='onefile')
    assert not (tmp_path / 'embedded.fdt').exists()


def test_f34_corrupt_dataset_explains_recovery(tmp_path):
    path = tmp_path / 'corrupt.set'
    path.write_bytes(b'not a MATLAB dataset' * 20)
    with pytest.raises(ValueError, match=r'pop_loadset.*valid.*\.set.*[Rr]e-export'):
        pop_loadset(path)


def test_f34_missing_sidecar_names_file_and_recovery(tmp_path):
    eeg = _singleton_eeg(np.zeros((1, 4)))
    eeg['data'] = 'missing.fdt'
    scipy.io.savemat(tmp_path / 'missing.set', eeg)
    with pytest.raises(FileNotFoundError, match=r'pop_loadset.*missing.fdt.*[Rr]estore'):
        pop_loadset(tmp_path / 'missing.set')
