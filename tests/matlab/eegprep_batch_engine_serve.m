function eegprep_batch_engine_serve(directory)
% Serve matlab.engine style requests from files, inside a matlab-batch session.
% Loops until DIRECTORY/quit exists. Request DIRECTORY/req_*.mat holds kind
% ('call'|'eval'), name, nargout and args (1xN cell); the answer res_*.mat holds
% outputs (1xnargout cell) or error (identifier, message, stack). Everything
% runs in the base workspace, like the Engine, so eval statements share state.
% See tests/matlab_batch_engine.py for the Python side.
quit_file = fullfile(directory, 'quit');
if isfile(quit_file), delete(quit_file); end
fclose(fopen(fullfile(directory, 'ready'), 'w'));
while ~isfile(quit_file)
    pending = dir(fullfile(directory, 'req_*.mat'));
    if isempty(pending), pause(0.01); continue; end
    for name = sort({pending.name})
        request_file = fullfile(directory, name{1});
        response_file = strrep(request_file, 'req_', 'res_');
        try
            response = serve_one(load(request_file));
        catch err
            response = struct('error', describe_error(err));
        end
        save([response_file '.part'], '-struct', 'response', '-v7');
        movefile([response_file '.part'], response_file);
        delete(request_file);
    end
end
delete(quit_file);
end

function response = serve_one(request)
n = double(request.nargout);
assignin('base', 'eegprep_rpc_name__', char(request.name));
if strcmp(char(request.kind), 'eval')
    call = 'eval(eegprep_rpc_name__)';
else
    args = request.args;
    if ~iscell(args), args = {args}; end
    if isempty(request.args), args = {}; end
    assignin('base', 'eegprep_rpc_args__', restore_arguments(args));
    call = 'feval(eegprep_rpc_name__, eegprep_rpc_args__{:})';
end
try
    if n == 0
        evalin('base', [call ';']);
        outputs = cell(1, 0);
    else
        evalin('base', sprintf('eegprep_rpc_out__ = cell(1, %d); [eegprep_rpc_out__{:}] = %s;', n, call));
        outputs = evalin('base', 'eegprep_rpc_out__');
    end
    response = struct('outputs', {prepare_outputs(outputs)});
catch err
    response = struct('error', describe_error(err));
end
evalin('base', 'clear eegprep_rpc_name__ eegprep_rpc_args__ eegprep_rpc_out__');
end

function args = restore_arguments(args)
% savemat cannot write an empty cell, [] or a logical scalar reliably; Python tags them.
for i = 1:numel(args)
    v = args{i};
    if isstruct(v) && isscalar(v) && isfield(v, 'eegprep_rpc_tag__')
        switch char(v.eegprep_rpc_tag__)
            case 'empty',   args{i} = [];
            case 'logical', args{i} = logical(v.value);
            case 'cell'
                inner = v.value;
                if ~iscell(inner), inner = num2cell(inner); end
                args{i} = restore_arguments(inner);
        end
    end
end
end

function outputs = prepare_outputs(outputs)
% MAT v7 cannot hold every class; keep what Python decodes.
for i = 1:numel(outputs)
    v = outputs{i};
    if isa(v, 'function_handle'), outputs{i} = func2str(v);
    elseif isstring(v), outputs{i} = char(v);
    elseif isobject(v), outputs{i} = sprintf('<%s>', class(v));
    end
end
end

function d = describe_error(err)
stack = '';
for i = 1:min(numel(err.stack), 10)
    stack = sprintf('%s  %s (line %d)\n', stack, err.stack(i).name, err.stack(i).line);
end
d = struct('identifier', err.identifier, 'message', err.message, 'stack', stack);
end
