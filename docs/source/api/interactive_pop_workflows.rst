.. _api_interactive_pop_workflows:

Interactive Pop Workflows
=========================

User-facing ``pop_*`` wrappers. Each accepts ``return_com=True`` and returns a replayable history command.

Loading and Saving
------------------

``pop_biosig`` and ``pop_fileio`` accept ``blockrange=[start, stop]`` for
half-open, seconds-based reads of continuous EDF, BDF, and other MNE-backed
formats. ``pop_fileio`` also accepts EEGLAB's 1-based ``channels`` selection
and inclusive ``samples`` and ``trials`` ranges. ``pop_loadset`` supports
metadata-only and 1-based channel loading through ``loadmode``.
``pop_importpres`` recognizes tab-delimited Presentation headers and also
accepts EEGLAB's positional event-type, time, and duration field names.
``pop_read_erpss`` imports uncompressed and delta-compressed ERPSS ``.RAW``
and ``.RDF`` recordings. Pass a sampling rate only when timing is absent from
the recording header.
``pop_readegi`` imports continuous or equal-length segmented EGI Simple Binary
RAW files, including event channels and segment categories. ``pop_readsegegi``
joins a numbered continuous series ending in ``001.RAW``, ``002.RAW``, and so
on, and validates that acquisition headers agree before concatenating samples.
``pop_importegimat`` reads EGI Net Station MATLAB exports. Segment variables
named ``<condition>_Segment<number>`` become trials with one condition event
per trial; continuous exports are read from the ``Session`` variable by
default. An embedded ``samplingRate`` takes precedence over the supplied rate,
and ``latpoint0`` is expressed in milliseconds.

.. autosummary::
   :toctree: generated/

   eegprep.pop_biosig
   eegprep.pop_chanedit
   eegprep.pop_editset
   eegprep.pop_expevents
   eegprep.pop_export
   eegprep.pop_exportbids
   eegprep.importevent
   eegprep.pop_importdata
   eegprep.pop_importegimat
   eegprep.pop_importepoch
   eegprep.pop_importevent
   eegprep.pop_load_frombids
   eegprep.pop_loadset
   eegprep.pop_loadset_h5
   eegprep.pop_read_erpss
   eegprep.pop_readlocs
   eegprep.pop_readegi
   eegprep.pop_readsegegi
   eegprep.pop_saveh
   eegprep.pop_saveset
   eegprep.pop_writeeeg

Event and Epoch Tables
----------------------

``importevent`` converts standalone event tables into event dictionaries using
the same field, delimiter, time-unit, alignment, and append rules as
``pop_importevent``. ``pop_importevent`` accepts text files or record sequences
and stores the result on an EEG dataset. Imported latencies
use seconds by default, ``timeunit=1e-3`` selects milliseconds, and
``timeunit=numpy.nan`` selects sample positions. Existing events are appended
unless ``append="no"`` is supplied. Alignment can anchor imported rows to the
existing event stream and optionally compensate for small clock-rate drift;
``indices`` uses 1-based event numbers when updating selected rows.

``pop_importepoch`` accepts one row per epoch. It preserves the row metadata in
``EEG["epoch"]`` and creates time-locking and latency-field events in
``EEG["event"]``. Event latencies are stored as 1-based absolute samples, and
durations are stored as sample counts.

Preprocessing
-------------

.. autosummary::
   :toctree: generated/

   eegprep.pop_autorej
   eegprep.pop_clean_rawdata
   eegprep.pop_eegfilt
   eegprep.pop_eegfiltnew
   eegprep.pop_eegthresh
   eegprep.pop_epoch
   eegprep.pop_firws
   eegprep.pop_firwsord
   eegprep.pop_interp
   eegprep.pop_jointprob
   eegprep.pop_rejchan
   eegprep.pop_rejchanspec
   eegprep.pop_rejcont
   eegprep.pop_rejepoch
   eegprep.pop_rejkurt
   eegprep.pop_rejspec
   eegprep.pop_rejtrend
   eegprep.pop_reref
   eegprep.pop_resample
   eegprep.pop_rmbase
   eegprep.pop_select
   eegprep.pop_selectcomps
   eegprep.pop_selectevent
   eegprep.pop_xfirws

ICA and Components
------------------

.. autosummary::
   :toctree: generated/

   eegprep.pop_expica
   eegprep.pop_icathresh
   eegprep.pop_iclabel
   eegprep.pop_prop
   eegprep.pop_prop_extended
   eegprep.pop_runica
   eegprep.pop_subcomp
   eegprep.pop_viewprops

Plotting and Review
-------------------

.. autosummary::
   :toctree: generated/

   eegprep.pop_chanplot
   eegprep.pop_compareerps
   eegprep.pop_crossf
   eegprep.pop_dipplot
   eegprep.pop_eegplot
   eegprep.pop_envtopo
   eegprep.pop_erpimage
   eegprep.pop_headplot
   eegprep.pop_newcrossf
   eegprep.pop_newtimef
   eegprep.pop_plotdata
   eegprep.pop_plottopo
   eegprep.pop_spectopo
   eegprep.pop_timtopo
   eegprep.pop_topochansel
   eegprep.pop_topoplot

The low-level ``erpimage`` helper accepts either one time value per sample or
EEGLAB's compact ``[start_ms, frames, sampling_rate]`` time specification.
``newtimef`` accepts EEGLAB's default ``outputformat='plot'`` option; other
output layouts are not yet implemented.

STUDY
-----

.. autosummary::
   :toctree: generated/

   eegprep.pop_clust
   eegprep.pop_clustedit
   eegprep.pop_loadstudy
   eegprep.pop_preclust
   eegprep.pop_precomp
   eegprep.pop_savestudy
   eegprep.pop_study
   eegprep.pop_studydesign
   eegprep.pop_studyerp
   eegprep.pop_studywizard

Other
-----

.. autosummary::
   :toctree: generated/

   eegprep.pop_addindepvar
   eegprep.pop_adjustevents
   eegprep.pop_averef
   eegprep.pop_chancenter
   eegprep.pop_chancoresp
   eegprep.pop_chanevent
   eegprep.pop_chansel
   eegprep.pop_comments
   eegprep.pop_comperp
   eegprep.pop_copyset
   eegprep.pop_delset
   eegprep.pop_dipfit_gridsearch
   eegprep.pop_dipfit_headmodel
   eegprep.pop_dipfit_loreta
   eegprep.pop_dipfit_nonlinear
   eegprep.pop_dipfit_settings
   eegprep.pop_editeventfield
   eegprep.pop_editeventvals
   eegprep.pop_editoptions
   eegprep.pop_eventstat
   eegprep.pop_fileio
   eegprep.pop_fileio_brainvision_mat
   eegprep.pop_findmatchingcomps
   eegprep.pop_firma
   eegprep.pop_firpm
   eegprep.pop_firpmord
   eegprep.pop_fusechanrej
   eegprep.pop_icflag
   eegprep.pop_importbids
   eegprep.pop_importerplab
   eegprep.pop_importgroupvar
   eegprep.pop_importpres
   eegprep.pop_kaiserbeta
   eegprep.pop_leadfield
   eegprep.pop_limo
   eegprep.pop_limoresults
   eegprep.pop_listfactors
   eegprep.pop_loadbci
   eegprep.pop_mergeset
   eegprep.pop_multifit
   eegprep.pop_newset
   eegprep.pop_read_erpss
   eegprep.pop_rejmenu
   eegprep.pop_rmdat
   eegprep.pop_runscript
   eegprep.pop_signalstat
   eegprep.pop_snapread
   eegprep.pop_timef
   eegprep.pop_writelocs
