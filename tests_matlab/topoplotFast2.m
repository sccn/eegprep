% topoplot() - plot a topographic map of a scalp data field in a 2-D circular view
%              (looking down at the top of the head) using interpolation on a fine
%              cartesian grid. Can also show specified channnel location(s), or return
%              an interpolated value at an arbitrary scalp location (see 'noplot').
%              By default, channel locations below head center (arc_length 0.5) are
%              shown in a 'skirt' outside the cartoon head (see 'plotrad' and 'headrad'
%              options below). Nose is at top of plot; left is left; right is right.
%              Using option 'plotgrid', the plot may be one or more rectangular grids.
% Usage:
%        >>  topoplot(datavector, EEG.chanlocs);   % plot a map using an EEG chanlocs structure
%        >>  [h grid_or_val plotrad_or_grid, xmesh, ymesh]= ...
%                           topoplot(datavector, chan_locs, 'Input1','Value1', ...);
% Required Inputs:
%   datavector        - single vector of channel values. Else, if a vector of selected subset
%                       (int) channel numbers -> mark their location(s) using 'style' 'blank'.
%   chan_locs         - name of an EEG electrode position file (>> topoplot example).
%                       Else, an EEG.chanlocs structure (>> help readlocs or >> topoplot example)
% Optional inputs:
%   'noplot'          - ['on'|'off'|[rad theta]] do not plot (but return interpolated data).
%                       Else, if [rad theta] are coordinates of a (possibly missing) channel,
%                       returns interpolated value for channel location.  For more info,
%                       see >> topoplot 'example' {default: 'off'}
%                       {default: 'k' = black}.
% Outputs:
%                   h - handle of the colored surface. If no surface is plotted,
%                       return "gca", the handle of the current plot.
%         grid_or_val - [matrix] the interpolated data image (with off-head points = NaN).
%                       Else, single interpolated value at the specified 'noplot' arg channel
%                       location ([rad theta]), if any.
%     plotrad_or_grid - IF grid image returned above, then the 'plotrad' radius of the grid.
%                       Else, the grid image
%     xmesh, ymesh    - x and y values of the returned grid (above)
%
% Authors: Andy Spydell, Colin Humphries, Arnaud Delorme & Scott Makeig
%          CNL / Salk Institute, 8/1996-/10/2001; SCCN/INC/UCSD, Nov. 2001 -

% Copyright (C) Colin Humphries & Scott Makeig, CNL / Salk Institute, Aug, 1996
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

function [handle,Zi,grid,Xi,Yi] = topoplotFast2(Values,loc_file,varargin)

%
%%%%%%%%%%%%%%%%%%%%%%%% Set defaults %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
plotgrid = 'off';
plotchans = [];
noplot  = 'off';
handle = [];
Zi = [];
chanval = NaN;
rmax = 0.5;             % actual head radius - Don't change this!
INTERPLIMITS = 'head';  % head, electrodes
INTSQUARE = 'on';       % default, interpolate electrodes located though the whole square containing
% the plotting disk
default_intrad = 1;     % indicator for (no) specified intrad
ELECTRODES = [];        % default 'electrodes': on|off|label - set below
MAXDEFAULTSHOWLOCS = 64;% if more channels than this, don't show electrode locations by default
intrad       = [];      % default interpolation square is to outermost electrode (<=1.0)
plotrad      = [];      % plotting radius ([] = auto, based on outermost channel location)
headrad      = [];      % default plotting radius for cartoon head is 0.5
squeezefac = 1.0;
ContourVals = Values;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%
%%%%%%%%%%%%%%%%%%%%%%% Handle arguments %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
if nargin< 1
    help topoplot;
    return
end

nargs = nargin;

if nargs > 2
    if ~(round(nargs/2) == nargs/2)
        error('Odd number of input arguments??')
    end
    for i = 1:2:length(varargin)
        Param = varargin{i};
        Value = varargin{i+1};
        if ~isstr(Param)
            error('Flag arguments must be strings')
        end
        Param = lower(Param);
        switch Param
            case 'noplot'
                noplot = Value;
                if ~isstr(noplot)
                    if length(noplot) ~= 2
                        error('''noplot'' location should be [radius, angle]')
                    else
                        chanrad = noplot(1);
                        chantheta = noplot(2);
                        noplot = 'on';
                    end
                end
            otherwise
                error(['Unknown input parameter ''' Param ''' ???'])
        end
    end
end

cmap = jet(64);
cmaplen = size(cmap,1);
GRID_SCALE = 32;

%
%%%%%%%%%%%%%%%%%%%%%%%%%%% misc arg tests %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
if isempty(ELECTRODES)                     % if electrode labeling not specified
    if length(Values) > MAXDEFAULTSHOWLOCS   % if more channels than default max
        ELECTRODES = 'off';                    % don't show electrodes
    else                                     % else if fewer chans,
        ELECTRODES = 'on';                     % do
    end
end

[r,c] = size(Values);
Values = Values(:); % make Values a column vector
ContourVals = ContourVals(:); % values for contour

%
%%%%%%%%%%%%%%%%%%%% Read the channel location information %%%%%%%%%%%%%%%%%%%%%%%%
%
tmpeloc = loc_file; % assum all channels have coordinates
labels = { tmpeloc.labels };
indices = ~cellfun(@isempty, {tmpeloc.theta});
Th = { tmpeloc.theta };
Rd = { tmpeloc.radius };
Th(~indices) = { NaN };
Rd(~indices) = { NaN };
Th = [ Th{:} ];
Rd = [ Rd{:} ];
indices = find(indices);

Th = pi/180*Th;                              % convert degrees to radians
allchansind = 1:length(Th);

if ~isempty(plotchans)
    if max(plotchans) > length(Th)
        error('''plotchans'' values must be <= max channel index');
    end
end
plotchans = indices;

%
%%%%%%%%%%%%%%%%%%% last channel is reference? %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
if length(tmpeloc) == length(Values) + 1 % remove last channel if necessary
    % (common reference channel)
    if plotchans(end) == length(tmpeloc)
        plotchans(end) = [];
    end
end

%
%%%%%%%%%%%%%%%%%%% remove infinite and NaN values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
if length(Values) > 1
    inds          = union(find(isnan(Values)), find(isinf(Values))); % NaN and Inf values
    plotchans     = setdiff(plotchans, inds);
end

[x,y]     = pol2cart(Th,Rd);  % transform electrode locations from polar to cartesian coordinates
plotchans = abs(plotchans);   % reverse indicated channel polarities
allchansind = allchansind(plotchans);
Th        = Th(plotchans);
Rd        = Rd(plotchans);
x         = x(plotchans);
y         = y(plotchans);
labels    = labels(plotchans); % remove labels for electrodes without locations
labels    = strvcat(labels); % make a label string matrix
if ~isempty(Values) & length(Values) > 1
    Values      = Values(plotchans);
    ContourVals = ContourVals(plotchans);
end

%
%%%%%%%%%%%%%%%%%% Read plotting radius from chanlocs  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
if isempty(plotrad)
    plotrad = min(1.0,max(Rd)*1.02);            % default: just outside the outermost electrode location
    plotrad = max(plotrad,0.5);                 % default: plot out to the 0.5 head boundary
end                                           % don't plot channels with Rd > 1 (below head)
if isempty(intrad)
    default_intrad = 1;     % indicator for (no) specified intrad
    intrad = min(1.0,max(Rd)*1.02);             % default: just outside the outermost electrode location
else
    default_intrad = 0;                         % indicator for (no) specified intrad
    if plotrad > intrad
        plotrad = intrad;
    end
end                                           % don't interpolate channels with Rd > 1 (below head)

%
%%%%%%%%%%%%%%%%%%%%% Find plotting channels  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
pltchans = find(Rd <= plotrad); % plot channels inside plotting circle

if strcmpi(INTSQUARE,'on') % interpolate channels in the radius intrad square
    intchans = find(x <= intrad & y <= intrad); % interpolate and plot channels inside interpolation square
else
    intchans = find(Rd <= intrad); % interpolate channels in the radius intrad circle only
end

%
%%%%%%%%%%%%%%%%%%%%% Eliminate channels not plotted  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%

allx      = x;
ally      = y;
intchans; % interpolate using only the 'intchans' channels
pltchans; % plot using only indicated 'plotchans' channels

if length(pltchans) < length(Rd) & strcmpi(VERBOSE, 'on')
    fprintf('Interpolating %d and plotting %d of the %d scalp electrodes.\n', ...
        length(intchans),length(pltchans),length(Rd));
end

if ~isempty(Values)
    if length(Values) == length(Th)  % if as many map Values as channel locs
        intValues      = Values(intchans);
        intContourVals = ContourVals(intchans);
        Values         = Values(pltchans);
        ContourVals    = ContourVals(pltchans);
    end
end   % now channel parameters and values all refer to plotting channels only

allchansind = allchansind(pltchans);
intTh = Th(intchans);           % eliminate channels outside the interpolation area
intRd = Rd(intchans);
intx  = x(intchans);
inty  = y(intchans);
Th    = Th(pltchans);              % eliminate channels outside the plotting area
Rd    = Rd(pltchans);
x     = x(pltchans);
y     = y(pltchans);

labels= labels(pltchans,:);
%
%%%%%%%%%%%%%%% Squeeze channel locations to <= rmax %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%

squeezefac = rmax/plotrad;
intRd = intRd*squeezefac; % squeeze electrode arc_lengths towards the vertex
Rd = Rd*squeezefac;       % squeeze electrode arc_lengths towards the vertex
% to plot all inside the head cartoon
intx = intx*squeezefac;
inty = inty*squeezefac;
x    = x*squeezefac;
y    = y*squeezefac;
allx    = allx*squeezefac;
ally    = ally*squeezefac;
% Note: Now outermost channel will be plotted just inside rmax

%
%%%%%%%%%%%%%%%% Find limits for interpolation %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
if default_intrad % if no specified intrad
    if strcmpi(INTERPLIMITS,'head') % intrad is 'head'
        xmin = min(-rmax,min(intx)); xmax = max(rmax,max(intx));
        ymin = min(-rmax,min(inty)); ymax = max(rmax,max(inty));

    else % INTERPLIMITS = rectangle containing electrodes -- DEPRECATED OPTION!
        xmin = max(-rmax,min(intx)); xmax = min(rmax,max(intx));
        ymin = max(-rmax,min(inty)); ymax = min(rmax,max(inty));
    end
else % some other intrad specified
    xmin = -intrad*squeezefac; xmax = intrad*squeezefac;   % use the specified intrad value
    ymin = -intrad*squeezefac; ymax = intrad*squeezefac;
end
%
%%%%%%%%%%%%%%%%%%%%%%% Interpolate scalp map data %%%%%%%%%%%%%%%%%%%%%%%%
%
xi = linspace(xmin,xmax,GRID_SCALE);   % x-axis description (row vector)
yi = linspace(ymin,ymax,GRID_SCALE);   % y-axis description (row vector)
[yi,xi] = meshgrid(yi,xi);
[Xi,Yi,Zi] = griddata(double(inty),double(intx),double(intValues),double(yi),double(xi),'v4'); % interpolate data

%
%%%%%%%%%%%%%%%%%%%%%%% Mask out data outside the head %%%%%%%%%%%%%%%%%%%%%
%
mask = (sqrt(Xi.^2 + Yi.^2) <= rmax); % mask outside the plotting circle
ii = find(mask == 0);
Zi(ii)  = NaN;                         % mask non-plotting voxels with NaNs
grid = plotrad;                       % unless 'noplot', then 3rd output arg is plotrad

