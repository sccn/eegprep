function success = generate_ica_comparison_plot(mat_file, py_file, output_file, n_comps, correlations)
% Generate ICA component scalp map comparison plot using MATLAB's topoplot.
%
% This function creates a comparison plot showing MATLAB and Python ICA
% component scalp maps side by side, using MATLAB's topoplot function.
%
% Inputs:
%   mat_file: Path to MATLAB-generated .set file
%   py_file: Path to Python-generated .set file (with reordered components)
%   output_file: Path to save the output PNG file
%   n_comps: Number of components to plot (default: 10)
%   correlations: Vector of correlations for each matched component pair
%
% Output:
%   success: 1 if successful, 0 otherwise
%   Also saves a PNG file with the comparison plot

if nargin < 4
    n_comps = 10;
end
if nargin < 5
    correlations = [];
end

% Load EEG structures (convert paths to MATLAB format if needed)
mat_file = strrep(mat_file, '\', '/');  % Convert backslashes to forward slashes
py_file = strrep(py_file, '\', '/');
output_file = strrep(output_file, '\', '/');

try
    EEG_mat = pop_loadset('filename', mat_file);
    EEG_py = pop_loadset('filename', py_file);
catch ME
    error('Failed to load EEG files: %s', ME.message);
end

% Limit number of components to plot
n_comps = min(n_comps, size(EEG_mat.icawinv, 2));

% Create figure
fig = figure('Visible', 'off', 'Position', [100, 100, 300*n_comps, 600]);

% Create subplots: MATLAB on top row, Python on bottom row
for i = 1:n_comps
    % MATLAB scalp map (top row)
    subplot(2, n_comps, i);
    topoplot(EEG_mat.icawinv(:, i), EEG_mat.chanlocs, 'noplot', 'off', 'electrodes', 'off');
    if ~isempty(correlations) && i <= length(correlations)
        title(sprintf('MAT IC%d\n(r=%.3f)', i, correlations(i)), 'FontSize', 10);
    else
        title(sprintf('MAT IC%d', i), 'FontSize', 10);
    end
    axis off;
    
    % Python scalp map (bottom row)
    subplot(2, n_comps, n_comps + i);
    topoplot(EEG_py.icawinv(:, i), EEG_py.chanlocs, 'noplot', 'off', 'electrodes', 'off');
    if ~isempty(correlations) && i <= length(correlations)
        title(sprintf('Py IC%d\n(r=%.3f)', i, correlations(i)), 'FontSize', 10);
    else
        title(sprintf('Py IC%d', i), 'FontSize', 10);
    end
    axis off;
end

% Add overall title
sgtitle('ICA Component Scalp Maps: MATLAB (top) vs Python (bottom) - MATLAB Topoplot', ...
    'FontSize', 12, 'FontWeight', 'bold');

% Save figure
try
    print(fig, output_file, '-dpng', '-r150');
    close(fig);
    success = 1;
catch ME
    close(fig);
    error('Failed to save figure: %s', ME.message);
end

end

