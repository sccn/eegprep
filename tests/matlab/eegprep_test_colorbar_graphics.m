function varargout = eegprep_test_colorbar_graphics(action)
% Only native figure lifetime and property queries; Python owns every test.
persistent previous_figures previous_visibility other_axes
switch action
    case 'setup'
        previous_figures = findall(groot, 'Type', 'figure');
        previous_visibility = get(groot, 'DefaultFigureVisible');
        set(groot, 'DefaultFigureVisible', 'off');
        figure('Visible', 'off');
    case 'teardown'
        delete(setdiff(findall(groot, 'Type', 'figure'), previous_figures));
        set(groot, 'DefaultFigureVisible', previous_visibility);
        previous_figures = [];
        other_axes = [];
    case 'colorbar'
        bar = findall(gcf, 'Type', 'axes', 'Tag', 'cbar');
        varargout = {numel(bar), get(bar, 'YTick'), cellstr(get(bar, 'YTickLabel')), get(bar, 'YLim')};
    case 'unrelated_axes'
        other_axes = axes('YTick', [10 20 30]);
        figure('Visible', 'off');
    case 'unrelated_ticks'
        varargout = {get(other_axes, 'YTick')};
end
end
