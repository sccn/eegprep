function eegprep_test_tutorial_video(operation, varargin)
% MATLAB VideoWriter is a language/library object, not an EEGPrep function.
persistent writer;
switch operation
    case 'open'
        writer = VideoWriter(varargin{:});
        open(writer);
    case 'write'
        writeVideo(writer, varargin{1});
    case 'capture'
        currFrame = getframe(gcf);
        writeVideo(writer, currFrame);
    case 'close'
        if ~isempty(writer)
            close(writer);
            writer = [];
        end
end
end
