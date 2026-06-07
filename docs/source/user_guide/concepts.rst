.. _concepts:

==============
Concepts Guide
==============

EEGPrep keeps EEGLAB's user-facing vocabulary because it is the most useful
mental model for many EEG researchers. The runtime is Python, but the main
objects, command names, history behavior, and menu flow are designed so EEGLAB
users can predict what will happen.

Workspace Names
===============

``eegprep-console --full`` opens a Qt GUI and an IPython console against one
shared ``EEGPrepSession``. The console starts with the familiar EEGLAB
workspace names:

.. list-table::
   :header-rows: 1

   * - Name
     - Meaning in EEGPrep
   * - ``EEG``
     - The current EEG dataset dictionary.
   * - ``ALLEEG``
     - List of loaded EEG dataset dictionaries.
   * - ``CURRENTSET``
     - Current dataset selection, exposed as ``0`` when empty, a scalar for one
       selected dataset, or a list for multi-dataset selection.
   * - ``LASTCOM``
     - Last replayable command string returned by a GUI or ``pop_*`` action.
   * - ``ALLCOM``
     - Ordered command history for the interactive session.
   * - ``STUDY``
     - Current group-level STUDY dictionary.
   * - ``CURRENTSTUDY``
     - ``1`` when a STUDY is active, otherwise ``0``.

These are normal Python names in the console namespace, not hidden globals.
GUI actions update them through the shared session. Console calls update the GUI
when they go through ``eegprep-console`` wrappers.

The EEG Dataset Dictionary
==========================

EEGPrep datasets are plain dictionaries with EEGLAB field names. Continuous data
is channel-major, usually ``(nbchan, pnts)``. Epoched data is usually
``(nbchan, pnts, trials)``.

Core fields include:

.. list-table::
   :header-rows: 1

   * - Field
     - Purpose
   * - ``data``
     - NumPy array, memory-mapped array, or retrieved in-memory data.
   * - ``nbchan``, ``pnts``, ``trials``
     - Channel count, points per trial, and trial count.
   * - ``srate``, ``xmin``, ``xmax``, ``times``
     - Sampling rate, time limits in seconds, and time vector in milliseconds.
   * - ``chanlocs``
     - Channel metadata such as labels, types, and locations.
   * - ``event`` and ``urevent``
     - Event records and original event references.
   * - ``epoch``
     - Per-epoch metadata for epoched datasets.
   * - ``icaweights``, ``icasphere``, ``icawinv``, ``icaact``
     - ICA decomposition fields and activations.
   * - ``icachansind``
     - Channels used for ICA. Stored internally as Python zero-based indices.
   * - ``reject``
     - Manual, automatic, and component rejection marks.
   * - ``history``
     - Dataset-level command history.
   * - ``etc``
     - Structured extension metadata, including ICLabel classifications.

Events, Epochs, and Indexing
============================

EEGLAB event latencies are user-facing sample positions. EEGPrep preserves that
public convention in command strings and event records. Python array slicing is
still zero-based internally.

Practical rules:

* Use event ``latency`` values as EEGLAB-facing sample positions when reading
  or writing event metadata.
* Use Python zero-based indexing when slicing ``EEG["data"]``.
* Most ``pop_*`` command strings print EEGLAB-facing values so history can be
  understood by EEGLAB users.
* ``icachansind`` is normalized to zero-based integer indices after loading a
  MATLAB ``.set`` file, then converted back to one-based values when saving.

For epoched data, ``EEG["data"][:, :, trial_index]`` is zero-based Python
indexing. User-facing epoch and component selectors in GUI dialogs use
EEGLAB-style one-based values.

Channel Locations
=================

``chanlocs`` is a list-like channel table. Common keys are ``labels``, ``type``,
``theta``, ``radius``, ``X``, ``Y``, ``Z``, ``sph_theta``, ``sph_phi``, and
``sph_radius``. Channel location readers and editors keep the EEGLAB field
names so topographic plotting, interpolation, DIPFIT, ICLabel features, and
STUDY channel measures can share the same metadata.

Use ``pop_chanedit`` or ``pop_readlocs`` when working through EEGLAB-style
dialogs. Use ``readlocs`` or direct dictionary updates only when scripting and
you already know the channel table shape you need.

ICA and ICLabel Fields
======================

ICA workflows write the same core fields EEGLAB users expect:

* ``icaweights`` and ``icasphere`` store the unmixing transform.
* ``icawinv`` stores scalp maps.
* ``icaact`` can be cached or recomputed from the decomposition.
* ``reject.gcompreject`` stores component rejection flags.
* ``etc.ic_classification.ICLabel.classifications`` stores ICLabel class
  probabilities in the order Brain, Muscle, Eye, Heart, Line Noise, Channel
  Noise, Other.

Run ``pop_runica`` before ``pop_iclabel``. Use ``pop_viewprops`` or
``pop_prop_extended`` to review component properties, labels, and DIPFIT
projections when available.

STUDY Structure
===============

EEGPrep STUDY dictionaries cover the implemented standalone group-analysis
surface:

* ``datasetinfo`` records dataset membership and metadata.
* ``design`` records independent variables and selected designs.
* ``changrp`` stores channel measure caches such as ERP, spectrum, ERSP, ITC,
  and PAC.
* ``cluster`` stores component measure caches and clustering metadata.

The current implementation keeps measure caches in EEGPrep-owned JSON-safe
structures unless a workflow explicitly writes sidecar data. MATLAB-only LIMO
result files and external FieldTrip STUDY source statistics are explicit
backend boundaries; they are not silently emulated.

History and Replay
==================

The most important EEGLAB workflow is still available: run something from the
GUI, inspect the command, then reuse it in a script.

.. code-block:: python

   EEG, LASTCOM = pop_resample(EEG, 64, return_com=True)
   print(LASTCOM)
   ALLCOM.append(LASTCOM)

Inside ``eegprep-console``, bare calls such as ``pop_resample(EEG, 64)`` can
auto-store the returned dataset in the shared session. Normal Python scripts do
not do that; assign returned values explicitly.

Standalone Runtime Boundary
===========================

The installed EEGPrep package must work without ``src/eegprep/eeglab``. The
vendored EEGLAB tree is a development reference for parity work, not a runtime
dependency. Packaged help, bundled plugin data, montages, ICLabel assets, and
Sphinx documentation are EEGPrep-owned resources.

EEGLAB References Used
======================

The manual structure follows the EEGLAB tutorial site's emphasis on quickstart,
concepts, importing, preprocessing, plotting, STUDY, scripting, and data
structures, while rewriting the content for EEGPrep's Python, Qt GUI, and
``eegprep-console`` behavior.
