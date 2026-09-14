.. _api_plotting_and_topography:

Plotting and Topography
=======================

Scalp maps, curve summaries, figure-layout helpers, logarithmic image displays,
event rasters, artifact review plots, and replayable scalp movies. Movie frames
are returned as ``uint8`` RGB arrays so they can be inspected in notebooks,
tested headlessly, encoded by downstream tools, or replayed with ``seemovie``.
The ``movieframes`` argument follows EEGLAB's 1-based public indexing
convention.

.. autosummary::
   :toctree: generated/

   eegprep.hist2
   eegprep.cbar
   eegprep.copyaxis
   eegprep.forcelocs
   eegprep.headplot
   eegprep.plotcurve
   eegprep.sbplot
   eegprep.slider
   eegprep.gradmap
   eegprep.gradplot
   eegprep.imagesclogy
   eegprep.imagescloglog
   eegprep.show_events
   eegprep.setfont
   eegprep.eegmovie
   eegprep.headmovie
   eegprep.seemovie
   eegprep.topoplot
   eegprep.loc_subsets
   eegprep.vis_artifacts
   eegprep.vis_artifacts_diagnostics

Low-level EEGLAB plotting workflows
-----------------------------------

The low-level helpers retain EEGLAB's familiar names while returning ordinary
Matplotlib objects. ``cbar`` draws full or one-based partial colormap ranges;
``copyaxis`` copies the current plot into a standalone figure; and ``sbplot``
creates axes that may span opposite corners of a one-based subplot grid.
``slider`` returns its widget handles so scripts can pan or remove a magnified
viewport without relying on hidden callback strings.

``plotcurve`` accepts data with curves in rows or columns, plots individual and
mean traces, and supports confidence-limit highlighting. A two-value
``maskarray=[low, high]`` marks samples whose comparison values fall outside
that interval. ``highlightmode="background"`` shades the curve axes, while
``highlightmode="bottom"`` puts the significance runs in a narrow lower axes.

``forcelocs`` rotates X/Z or Y/Z channel coordinates and refreshes the
spherical and topographic fields after every rotation. ``headplot`` renders
reusable 3-D spline maps; ``headplot("example")`` and
``headplot("cartesian")`` return and print the corresponding electrode-file
examples for interactive use.
