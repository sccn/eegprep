.. _study_workflows:

===============
STUDY Workflows
===============

EEGPrep includes standalone STUDY/session surfaces for common group-level
workflows. These APIs mirror EEGLAB's STUDY-facing names while storing cached
measure data in EEGPrep-owned JSON-safe structures.

Implemented Workflows
=====================

Create a STUDY from loaded datasets:

.. code-block:: python

   from eegprep import pop_study

   STUDY, ALLEEG, com = pop_study(None, ALLEEG, name="My study", return_com=True)

Create a simple ERP STUDY:

.. code-block:: python

   from eegprep import pop_studyerp

   STUDY, ALLEEG, com = pop_studyerp(ALLEEG, return_com=True)

Build and inspect design variables:

.. code-block:: python

   from eegprep import pop_importgroupvar, pop_listfactors, std_builddesignmat

   STUDY, com = pop_importgroupvar(
       STUDY,
       1,
       variable="age_group",
       values={"S01": "young", "S02": "older"},
       return_com=True,
   )
   factors = pop_listfactors(STUDY, constant="off")
   design_matrix, labels, categorical = std_builddesignmat(
       STUDY["design"][0],
       [{"condition": "target", "rt": 300.0}, {"condition": "standard", "rt": 400.0}],
       expanding=True,
   )

Load and save EEGPrep ``.study`` files:

.. code-block:: python

   from eegprep import pop_loadstudy, pop_savestudy

   STUDY, com = pop_savestudy(STUDY, EEG, "analysis.study", return_com=True)
   STUDY, ALLEEG, com = pop_loadstudy("analysis.study", return_com=True)

Precompute and plot STUDY measures:

.. code-block:: python

   from eegprep import pop_chanplot, pop_precomp, std_erpplot, std_readitc

   STUDY, ALLEEG, com = pop_precomp(
       STUDY,
       ALLEEG,
       "channels",
       erp="on",
       spec="on",
       return_com=True,
   )
   STUDY, com, fig = pop_chanplot(STUDY, ALLEEG, measure="erp", return_com=True)
   STUDY, erpdata, erptimes, fig = std_erpplot(STUDY, ALLEEG, channels=[1])
   STUDY, itcdata, itctimes, itcfreqs = std_readitc(STUDY, ALLEEG, channels=[1])

Channel measures are stored in ``STUDY.changrp``. Component measures are stored
on the parent ``STUDY.cluster[0]`` entry so preclustering can read the same
cached arrays. Cached measure fields follow EEGLAB names such as ``erpdata``,
``specdata``, ``erspdata``, and ``itcdata``. The selected ``design`` is recorded
in each measure group's metadata. EEGPrep stores dataset-level averages in the
current standalone cache rather than EEGLAB sidecar measure files.

Use ``std_checkfiles``, ``std_checkdatasession``, ``std_uniformfiles``, and
``std_uniformsetinds`` to audit loaded dataset consistency and cached measure
shapes before saving or plotting group-level results. ``std_savedat`` writes
explicit EEGPrep-owned JSON or MATLAB-compatible measure sidecars when a
workflow needs a durable array file outside the ``.study`` JSON.

Select datasets or trials from STUDY metadata:

.. code-block:: python

   from eegprep import std_getindvar, std_maketrialinfo, std_selectdataset

   STUDY, trialinfo = std_maketrialinfo(STUDY, ALLEEG)
   factors, factor_values, subjects, paired = std_getindvar(STUDY)
   dataset_indices, trial_indices = std_selectdataset(
       STUDY,
       ALLEEG,
       "condition",
       ["target"],
   )

These helpers return EEGLAB-facing 1-based dataset and trial indices. Use
``std_substudy`` or ``std_rmdat`` when a workflow needs to remove datasets;
EEGPrep remaps STUDY references and invalidates cached measure arrays after
membership changes.

``std_findsameica`` groups matching ICA decompositions within each subject.
This preserves the subject boundary used by STUDY designs instead of merging
identical test fixtures across subjects.

Precluster and cluster ICA components:

.. code-block:: python

   from eegprep import pop_clust, pop_clustedit, pop_preclust

   STUDY, ALLEEG, com = pop_preclust(
       STUDY,
       ALLEEG,
       preproc=[{"measure": "scalp", "npca": 3, "norm": 1, "weight": 1}],
       return_com=True,
   )
   STUDY, com = pop_clust(STUDY, ALLEEG, clus_num=4, random_state=0, return_com=True)
   STUDY, com, fig = pop_clustedit(STUDY, ALLEEG, action="plot", return_com=True)

Session Synchronization
=======================

The GUI and ``eegprep-console`` share ``STUDY`` and ``CURRENTSTUDY`` through
``EEGPrepSession``. Creating or loading a STUDY from the GUI sets
``CURRENTSTUDY`` to ``1``. Retrieving a dataset from the Datasets menu returns
``CURRENTSTUDY`` to ``0`` and records that transition in history.

Integration Notes
=================

Component ERP, spectrum, ERSP, and ITC arrays are cached on the parent
``STUDY.cluster[0]`` entry. Preclustering reads those cached component arrays
and can also build scalp-map features directly from loaded ICA maps. MATLAB
parity checks focus on deterministic metadata and cluster structure; exact
numeric clustering labels can differ because EEGPrep uses deterministic
in-package k-means helpers rather than MATLAB's Statistics Toolbox.

Limitations
===========

EEGPrep does not silently emulate EEGLAB's external LIMO toolbox. The
``pop_limo``, ``pop_limoresults``, ``std_limo*``, and ``std_readfilelimo``
entry points raise clear ``NotImplementedError`` messages.

Core EEGPrep also does not implement standalone phase-amplitude coupling
analysis. The ``pac``, ``pac_cont``, ``std_pac``, and ``std_pacplot`` entry
points raise clear ``NotImplementedError`` messages until there is a tested
EEGPrep-owned PAC backend. ``std_readpac`` only returns data when an explicit
EEGPrep-owned ``pacdata`` cache is present; external PAC or LIMO result files
are not interpreted as if they were native EEGPrep outputs.

See the :ref:`interactive_console` guide for mixed GUI plus console usage and
the :ref:`gui_help_menus` guide for menu inventory behavior.
