function varargout = eegprep_test_transport(mode, varargin)
% Independent sentinels for the Python test harness, not scientific behavior.
switch mode
    case 'echo'
        value = varargin{1};
        assert(strcmp(class(value), varargin{2}), 'Input dtype changed');
        assert(isequal(size(value), varargin{3}), 'Input shape changed');
        varargout = {value};
    case 'eeg'
        EEG = varargin{1};
        assert(strcmp(class(EEG.data), 'single'));
        assert(isequal(EEG.data, reshape(single(1:numel(EEG.data)), size(EEG.data))));
        assert(size(EEG.data, 1) == EEG.nbchan);
        assert(size(EEG.data, 2) == EEG.pnts);
        assert(size(EEG.data, 3) == EEG.trials);
        assert(EEG.event(1).latency == 1);
        assert(EEG.event(2).latency == EEG.pnts * EEG.trials);
        assert(strcmp(EEG.event(1).type, 'start'));
        assert(strcmp(EEG.event(2).type, 'end'));
        assert(isstruct(EEG.event) && isstruct(EEG.urevent) && isstruct(EEG.chanlocs));
        assert(strcmp(EEG.chanlocs(2).labels, 'Pz'));
        varargout = {EEG};
    case 'fixture'
        value.numeric = reshape(single(1:6), 2, 3);
        value.complex = complex(single([1 2]), single([3 -4]));
        value.logical = logical([1 0]);
        value.empty = zeros(0, 3, 'uint16');
        value.nested = struct('label', 'nested', 'column', int16([7; 8]));
        value.cells = {int8(1), 'two'; [], struct('last', uint32(4))};
        value.structs = struct('index', {int8(1), int8(2)});
        value.cell_structs = {struct('index', int8(1)), struct('index', int8(2))};
        varargout = {value};
    case 'nested'
        value = varargin{1};
        assert(isstruct(value.structs));
        assert(iscell(value.cell_structs));
        assert(isstruct(value.cell_structs{1}));
        assert(isstruct(value.cells{2, 2}));
        assert(value.structs(2).index == 2);
        varargout = {value};
    case 'unassigned'
        value = cell(2, 2);
        value{1, 1} = 1;
        value{2, 2} = zeros(1, 0);
        nested(1).first = 1;
        nested(2).second = 2;
        value{1, 2} = nested;
        varargout = {value};
    case 'figure_visibility'
        figure_handle = figure;
        cleanup = onCleanup(@() close(figure_handle));
        varargout = {char(get(figure_handle, 'Visible'))};
    case 'pair'
        assert(numel(varargin) == 2, 'Equal-shaped arguments were expanded');
        varargout = {varargin{1}, varargin{2}};
    case 'path'
        assert(isfile(varargin{1}), 'Path was not transported intact');
        varargout = {fileread(varargin{1})};
    case 'no_output'
        assert(nargout == 0);
    otherwise
        error('eegprep:test:UnknownTransportMode', 'Unknown mode');
end
end
