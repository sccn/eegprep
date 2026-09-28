function handle = eegprep_test_tutorial_figure(varargin)
% Finish native figure activation before the next Python-owned plotting call.
figure_handle = figure(varargin{:});
drawnow;
set(groot, 'CurrentFigure', figure_handle);
handle = double(figure_handle);
end
