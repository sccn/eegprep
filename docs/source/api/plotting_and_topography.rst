.. _api_plotting_and_topography:

Plotting and Topography
=======================

Scalp maps, curve summaries, figure-layout helpers, logarithmic image displays,
event rasters, artifact review plots, and replayable scalp movies. Movie frames
are returned as ``uint8`` RGB arrays so they can be inspected in notebooks,
tested headlessly, encoded by downstream tools, or replayed with ``seemovie``.
The ``movieframes`` argument follows EEGLAB's 1-based public indexing
convention.

``savefig`` writes a Matplotlib figure to a MATLAB ``.fig`` file that
``openfig`` can open. The file stores the figure, its axes, and the lines,
images, and text on those axes, including data, limits, colors, scales, and
labels. Figure text, such as a suptitle, is stored on the first axes.
``figure.savefig("name.fig")`` uses the same writer. Scatter plots,
bars, patches, and legends are omitted. The figure colormap is the first
image's colormap. Characters outside ASCII are stored as ASCII, and mathtext
tick labels are stored as the plain tick value, so MATLAB can read the file.

.. autosummary::
   :toctree: generated/

   eegprep.savefig
   eegprep.hist2
   eegprep.cbar
   eegprep.copyaxis
   eegprep.erpimage
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

.. _erpimage_api:

ERP images
----------

``erpimage`` images single trials of one channel or component, sorted and
smoothed across trials, with the ERP and, optionally, wavelet amplitude and
inter-trial coherence traces underneath. It takes EEGLAB's arguments in
EEGLAB's order, ``erpimage(data, sortvar, times, title, avewidth, decimate)``,
where ``data`` is ``(frames, trials)`` (for example ``EEG["data"][ch]``),
``sortvar`` holds one value per trial and ``times`` holds latencies in ms.
Options follow either as EEGLAB tokens or as keywords; these two calls are
equivalent:

.. code-block:: python

   from eegprep import erpimage

   result = erpimage(data, rt, times, "Pz", 10, 1, "erp", "cbar", "phasesort", [300, 25, 10])
   result = erpimage(data, rt, times, "Pz", 10, 1, erp=True, cbar=True, phasesort=[300, 25, 10])

The result unpacks like EEGLAB's fifteen outputs, followed by the figure:
``outdata, outvar, outtrials, limits, axhndls, erp, amps, cohers, cohsig,
ampsig, outamps, phsangls, phsamp, sortidx, erpsig, figure = result``, and the
same values are available as attributes (``result.outdata``). ``outdata`` is
``(frames, smoothed_trials)``; outputs that were not computed are ``None``.
Pass ``target=`` a Matplotlib Figure or SubFigure to draw into it, ``NoShow=True``
to compute without drawing, and ``rng=`` a NumPy random generator to make the
coherence bootstrap and the ``erpalpha`` permutations reproducible
(``numpy.random.RandomState(seed)`` gives the same bootstrap draws as MATLAB's
``rng(seed, 'twister')``).

The numbers come from ``eegprep.functions.sigprocfunc.erpimage_engine``, a
module that needs only NumPy. Its outputs match EEGLAB's ``erpimage`` run on
MATLAB R2025b to a relative tolerance of ``1e-12`` across 42 recorded
scenarios (sorting, smoothing and decimation, phase and amplitude sorting,
coherence and its significance, amplitude images, time warping, percentile
ERPs and axis limits).

Differences from EEGLAB
~~~~~~~~~~~~~~~~~~~~~~~

Where EEGLAB's behavior contradicts its own help text, EEGPrep follows the help:

* ``sortidx`` holds 0-based indices into the input trials. EEGLAB returns
  1-based indices into the trials left after removing NaN ``sortvar`` values.
* ``'noplot'`` hides only the sortvar trace. In EEGLAB it suppresses the whole
  figure.
* ``'erpalpha'`` draws unbiased random signs for its permutation test. EEGLAB
  draws ``sign(randn - 0.5)``, which is +1 only about 31% of the time (mean sign
  -0.38), so its thresholds are inflated
  (`sccn/eeglab#976 <https://github.com/sccn/eeglab/issues/976>`_). EEGPrep's
  ``erpalpha`` bands therefore differ slightly from MATLAB's; the rest of the
  computation is unchanged.
* ``'signif'`` with ``'coher'`` uses the given thresholds. On MATLAB R2025b this
  combination stops with an error (a vector operand to ``||``).
* Per-trial ``'timewarp'`` events and per-trial ``'vert'`` latencies stay with
  their trials through sorting and trial rejection; EEGLAB applies them by
  position after sorting.

Other behavior to know when moving from EEGLAB:

* Invalid inputs raise ``ValueError`` with EEGLAB's message, where EEGLAB
  prints the message and returns.
* Boxcar ``avewidth`` must be a whole number of trials, as on MATLAB R2025b,
  which rejects non-integer window sizes.
* A three-sample ``times`` vector is a time vector, not EEGLAB's compact
  ``[start_ms, frames, srate]`` form, when the data also have three frames.
* ``'timewarp'`` cannot be combined with ``'phasesort'``, ``'ampsort'`` or
  ``'plotamps'``; EEGLAB applies the warp marks inconsistently there.

