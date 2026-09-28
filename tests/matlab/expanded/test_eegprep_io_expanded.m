function tests = test_eegprep_io_expanded
% Real-fixture I/O contracts added after the pinned native coverage baseline.
% BrainVision supplied the BVA fixtures (their README.txt); the epoch fixture
% is EEGLAB sample_data/eeglab_data_epochs_ica.set plus its original FDT.
% All data comparisons cover the complete file/epoch array, not a header stub.
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
testCase.TestData.eeglab = fileparts(which('eeglab'));
% TestRunner changes cwd to the test file's directory; the frozen runtime
% remains directly inside its fixture suite, unlike this additional test file.
testCase.TestData.suite = fileparts(testCase.TestData.eeglab);
eeglab_options;
testCase.assertEqual(option_scaleicarms, 1);
testCase.assertEqual(option_storedisk, 0);
testCase.assertEqual(option_savetwofiles, 1);
end

function setup(testCase)
folder = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture('PreservingOnFailure', true));
testCase.TestData.output = folder.Folder;
testCase.applyFixture(matlab.unittest.fixtures.CurrentFolderFixture(folder.Folder));
end

function testBrainVisionMultiplexedCalibrationAndSubset(testCase)
verify_brainvision(testCase, 'multiplexed');
end

function testBrainVisionVectorizedCalibrationAndSubset(testCase)
verify_brainvision(testCase, 'vectorized');
end

function verify_brainvision(testCase, orientation)
directory = fullfile(testCase.TestData.suite, 'unittesting_binary', 'testfiles', 'BVA');
stem = ['brainvision_genericdataformat_binary' orientation '_int16'];
fid = fopen(fullfile(directory, [stem '.dat']), 'r', 'ieee-le');
assert(fid >= 0);
cleanup = onCleanup(@() fclose(fid));
codes = fread(fid, Inf, '*int16');
testCase.verifyNumElements(codes, 32*2112);
if strcmp(orientation, 'multiplexed')
    codes = reshape(codes, 32, 2112);
else
    codes = reshape(codes, 2112, 32)';
end
% The original VHDR specifies this microvolt resolution for every channel.
expected = single(codes) * 0.00045777764;
labels = {'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2', ...
    'F7','F8','T7','T8','P7','P8','Fz','Cz','Pz','FC1','FC2', ...
    'CP1','CP2','FC5','FC6','CP5','CP6','TP9','TP10','Eog','Ekg1','Ekg2'};
EEG = pop_loadbv(directory, [stem '.vhdr']);
testCase.verifyEqual(EEG.data, expected);
testCase.verifyEqual([EEG.nbchan EEG.pnts EEG.trials EEG.srate], [32 2112 1 200]);
testCase.verifyEqual([EEG.xmin EEG.xmax], [0 2111/200]);
testCase.verifyEqual(EEG.times, (0:2111)*5);
testCase.verifyEqual({EEG.chanlocs.labels}, labels);
stimuli = EEG.event(strcmp({EEG.event.code}, 'Stimulus'));
testCase.verifyEqual([stimuli.latency], [108 265 282 455 629 803 811 977 1151 1325 1357 1499 1673 1847 1903 2021]);
testCase.verifyEqual({stimuli.type}, {'S  4','S  1','S  4','S  4','S  4','S  4','S  2','S  4', ...
    'S  4','S  4','S  3','S  4','S  4','S  4','S  1','S  4'});
subset = pop_loadbv(directory, [stem '.vhdr'], [108 811], [32 1 17]);
testCase.verifyEqual(subset.data, expected([32 1 17],108:811));
testCase.verifyEqual([subset.nbchan subset.pnts subset.trials subset.srate], [3 704 1 200]);
testCase.verifyEqual(subset.times, (0:703)*5);
testCase.verifyEqual({subset.chanlocs.labels}, labels([32 1 17]));
testCase.verifyEqual([subset.event.latency], [1 158 175 348 522 696 704]);
testCase.verifyEqual({subset.event.type}, {'S  4','S  1','S  4','S  4','S  4','S  4','S  2'});
testCase.verifyEqual([subset.urevent.latency], [subset.event.latency]);
end

function testImportFloat32LittleEndianEpochs(testCase)
verify_import(testCase, 'float32le');
end

function testImportFloat32BigEndianEpochs(testCase)
verify_import(testCase, 'float32be');
end

function testImportMatlabEpochs(testCase)
verify_import(testCase, 'matlab');
end

function testImportTransposedAsciiEpochs(testCase)
verify_import(testCase, 'ascii');
end

function verify_import(testCase, format)
source = original_epochs(testCase);
filename = fullfile(testCase.TestData.output, ['epochs-' format '.dat']);
data = source.data;
switch format
    case 'matlab'
        save(filename, 'data', '-v7');
    case 'ascii'
        % Every original sample is retained; text rows are samples, not channels.
        writematrix(reshape(double(data),32,[])', filename, 'Delimiter', 'tab');
    otherwise
        if strcmp(format, 'float32le'), endian = 'ieee-le'; else, endian = 'ieee-be'; end
        fid = fopen(filename, 'w', endian);
        assert(fid >= 0);
        cleanup = onCleanup(@() fclose(fid));
        fwrite(fid, data, 'float32');
        clear cleanup;
end
EEG = pop_importdata('dataformat', format, 'nbchan', 32, 'data', filename, ...
    'pnts', 384, 'srate', 128, 'xmin', -1, 'chanlocs', source.chanlocs, ...
    'setname', 'Real epoch import', 'subject', 'S01', 'session', 2, ...
    'condition', 'targets', 'group', 'Control', 'ref', 'average', ...
    'comments', 'Original EEGLAB sample epochs; no sample reduction');
testCase.verifyEqual(double(EEG.data), double(source.data));
testCase.verifyEqual([EEG.nbchan EEG.pnts EEG.trials EEG.srate], [32 384 80 128]);
testCase.verifyEqual([EEG.xmin EEG.xmax], [-1 255/128]);
testCase.verifyEqual(EEG.times, (-128:255)*1000/128);
testCase.verifyEqual({EEG.chanlocs.labels}, {source.chanlocs.labels});
testCase.verifyEqual(EEG.setname, 'Real epoch import');
testCase.verifyEqual(EEG.subject, 'S01');
testCase.verifyEqual(EEG.session, 2);
testCase.verifyEqual(EEG.condition, 'targets');
testCase.verifyEqual(EEG.group, 'Control');
testCase.verifyEqual(EEG.ref, 'average');
testCase.verifyEqual(EEG.comments, 'Original EEGLAB sample epochs; no sample reduction');
end

function testSingleFileSaveVersionsAndReload(testCase)
source = original_epochs(testCase);
for version = {'6','7','7.3'}
    filename = ['inline-' version{1} '.set'];
    saved = pop_saveset(source, 'filename', filename, 'filepath', testCase.TestData.output, ...
        'savemode', 'onefile', 'version', version{1}, 'check', 'on');
    file = fullfile(testCase.TestData.output, filename);
    testCase.verifyEqual(H5F.is_hdf5(file), double(strcmp(version{1}, '7.3')));
    disk = read_set(file);
    testCase.verifyEqual(disk.data, source.data);
    testCase.verifyEqual(disk.icaweights, saved.icaweights);
    testCase.verifyEqual(disk.icawinv, saved.icawinv);
    verify_metadata(testCase, disk, source);
    testCase.verifyEmpty(dir(fullfile(testCase.TestData.output, ['inline-' version{1} '.fdt'])));
    testCase.verifyEqual(saved.filename, filename);
    loaded = pop_loadset('filename', filename, 'filepath', testCase.TestData.output);
    testCase.verifyEqual(loaded.data, source.data);
    verify_metadata(testCase, loaded, source);
    reloaded = pop_loadset('eeg', loaded);
    testCase.verifyEqual(reloaded.data, source.data);
    verify_metadata(testCase, reloaded, source);
end
end

function testTwoFileInfoChannelLoadAndMetadataResave(testCase)
source = original_epochs(testCase);
saved = pop_saveset(source, 'filename', 'external.set', 'filepath', testCase.TestData.output, ...
    'savemode', 'twofiles', 'version', '7');
file = fullfile(testCase.TestData.output, 'external.set');
disk = read_set(file);
testCase.verifyEqual(disk.data, 'external.fdt');
testCase.verifyEqual(disk.datfile, 'external.fdt');
testCase.verifyEqual(disk.icaweights, saved.icaweights);
testCase.verifyEqual(disk.icawinv, saved.icawinv);
verify_metadata(testCase, disk, source);
before = read_float_file(fullfile(testCase.TestData.output, 'external.fdt'));
testCase.verifyEqual(before, source.data(:));
info = pop_loadset('filename', 'external.set', 'filepath', testCase.TestData.output, 'loadmode', 'info');
testCase.verifyEqual(info.data, 'external.fdt');
verify_metadata(testCase, info, source);
channels = [1 17 32];
subset = pop_loadset('filename', 'external.set', 'filepath', testCase.TestData.output, 'loadmode', channels);
testCase.verifyEqual(subset.data, source.data(channels,:,:));
testCase.verifyEqual(subset.nbchan, 3);
testCase.verifyEqual({subset.chanlocs.labels}, {source.chanlocs(channels).labels});
testCase.verifyEmpty(subset.icaweights);
testCase.verifyEmpty(subset.icasphere);
testCase.verifyEmpty(subset.icawinv);
info.setname = 'Metadata-only resave';
info.saved = 'no';
pop_saveset(info, 'savemode', 'resave', 'version', '7');
disk = read_set(file);
testCase.verifyEqual(disk.setname, 'Metadata-only resave');
testCase.verifyEqual(disk.data, 'external.fdt');
testCase.verifyEqual(read_float_file(fullfile(testCase.TestData.output, 'external.fdt')), before);
loaded = pop_loadset('filename', 'external.set', 'filepath', testCase.TestData.output);
testCase.verifyEqual(loaded.data, source.data);
verify_metadata(testCase, loaded, source);
end

function testNonmonotonicChannelLoadPreservesDataLabelOrder(testCase)
% Unsuppressed native regression: eeg_getdatact's intersect sorts the data
% indices, while pop_loadset retains the requested order for channel labels.
source = original_epochs(testCase);
pop_saveset(source, 'filename', 'reordered.set', 'filepath', testCase.TestData.output, ...
    'savemode', 'twofiles', 'version', '7');
channels = [32 1 17];
subset = pop_loadset('filename', 'reordered.set', 'filepath', testCase.TestData.output, 'loadmode', channels);
testCase.verifyEqual(subset.data, source.data(channels,:,:));
testCase.verifyEqual({subset.chanlocs.labels}, {source.chanlocs(channels).labels});
testCase.verifyEqual([subset.nbchan subset.pnts subset.trials], [3 384 80]);
end

function testMultipleDatasetLoadAndResave(testCase)
source = original_epochs(testCase);
for name = {'first','second'}
    source.setname = name{1};
    pop_saveset(source, 'filename', [name{1} '.set'], 'filepath', testCase.TestData.output, ...
        'savemode', 'twofiles', 'version', '7');
end
EEG = pop_loadset('filename', {'first.set','second.set'}, 'filepath', testCase.TestData.output);
testCase.verifySize(EEG, [1 2]);
testCase.verifyEqual({EEG.setname}, {'first','second'});
for index = 1:2
    testCase.verifyEqual(EEG(index).data, source.data);
    verify_metadata(testCase, EEG(index), source);
    EEG(index).setname = ['Updated ' num2str(index)];
    EEG(index).saved = 'no';
end
pop_saveset(EEG, 'savemode', 'resave');
for index = 1:2
    disk = read_set(fullfile(testCase.TestData.output, EEG(index).filename));
    testCase.verifyEqual(disk.setname, ['Updated ' num2str(index)]);
    testCase.verifyEqual(disk.icaweights, EEG(index).icaweights);
    testCase.verifyEqual(disk.icawinv, EEG(index).icawinv);
    verify_metadata(testCase, disk, source);
    testCase.verifyEqual(read_float_file(fullfile(testCase.TestData.output, disk.datfile)), source.data(:));
end
end

function testExportAllEpochSamplesCsv(testCase)
source = original_epochs(testCase);
filename = fullfile(testCase.TestData.output, 'epochs.csv');
pop_export(source, filename, 'transpose', 'on', 'separator', ',', 'precision', 9, 'timeunit', 1);
fid = fopen(filename);
assert(fid >= 0);
cleanup = onCleanup(@() fclose(fid));
header = fgetl(fid);
testCase.verifyEqual(strsplit(header, ','), [{'Time'} {source.chanlocs.labels}]);
values = readmatrix(filename, 'NumHeaderLines', 1);
expected = [repmat((-128:255)'/128,80,1) reshape(double(source.data),32,[])'];
testCase.verifySize(values, [384*80 33]);
% Nine fractional digits round by at most half a decimal unit.
testCase.verifyEqual(values, expected, 'AbsTol', 5.1e-10);
end

function testExportErpExpressionWithoutLabelsOrTime(testCase)
source = original_epochs(testCase);
filename = fullfile(testCase.TestData.output, 'erp.txt');
pop_export(source, filename, 'erp', 'on', 'time', 'off', 'elec', 'off', ...
    'expr', 'x = 2*x', 'precision', 9);
values = readmatrix(filename, 'FileType', 'text');
expected = double(2*mean(source.data,3));
testCase.verifySize(values, [32 384]);
testCase.verifyEqual(values, expected, 'AbsTol', 5.1e-10);
end

function EEG = original_epochs(testCase)
directory = fullfile(testCase.TestData.eeglab, 'sample_data');
EEG = read_set(fullfile(directory, 'eeglab_data_epochs_ica.set'));
EEG.data = reshape(read_float_file(fullfile(directory, EEG.data)), EEG.nbchan, EEG.pnts, EEG.trials);
end

function EEG = read_set(filename)
EEG = load(filename, '-mat');
if isfield(EEG, 'EEG'), EEG = EEG.EEG; end
end

function data = read_float_file(filename)
fid = fopen(filename, 'r', 'ieee-le');
assert(fid >= 0);
cleanup = onCleanup(@() fclose(fid));
data = fread(fid, Inf, '*single');
end

function verify_metadata(testCase, actual, expected)
testCase.verifyEqual([actual.nbchan actual.pnts actual.trials actual.srate], [32 384 80 128]);
testCase.verifyEqual([actual.xmin actual.xmax], [-1 255/128]);
testCase.verifyEqual(actual.times, expected.times);
testCase.verifyEqual({actual.chanlocs.labels}, {expected.chanlocs.labels});
testCase.verifyEqual([actual.event.latency], [expected.event.latency]);
testCase.verifyEqual({actual.event.type}, {expected.event.type});
testCase.verifyEqual([actual.event.urevent], [expected.event.urevent]);
testCase.verifyEqual(actual.icasphere, expected.icasphere);
% With option_scaleicarms=1, eeg_checkset legitimately rescales ICA weights.
% Verify unit mixing-column RMS and unchanged reconstruction independently of
% the writer's returned metadata. 1e-12 bounds double 32x32 roundoff, not data.
testCase.verifyEqual(sqrt(mean(actual.icawinv.^2)), ones(1,32), 'AbsTol', 1e-12);
testCase.verifyEqual(actual.icawinv*(actual.icaweights*actual.icasphere), ...
    expected.icawinv*(expected.icaweights*expected.icasphere), 'AbsTol', 1e-12);
end
