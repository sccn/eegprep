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

Load and save EEGPrep ``.study`` files:

.. code-block:: python

   from eegprep import pop_loadstudy, pop_savestudy

   STUDY, com = pop_savestudy(STUDY, EEG, "analysis.study", return_com=True)
   STUDY, ALLEEG, com = pop_loadstudy("analysis.study", return_com=True)

Precompute and plot STUDY measures:

.. code-block:: python

   from eegprep import pop_chanplot, pop_precomp, std_erpplot

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

Channel measures are stored in ``STUDY.changrp``. Component measures are stored
on the parent ``STUDY.cluster[0]`` entry so preclustering can read the same
cached arrays. Cached measure fields follow EEGLAB names such as ``erpdata``,
``specdata``, ``erspdata``, and ``itcdata``. The selected ``design`` is recorded
in each measure group's metadata. EEGPrep stores dataset-level averages in the
current standalone cache rather than EEGLAB sidecar measure files.

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
scikit-learn k-means rather than MATLAB's implementation.

See the :ref:`interactive_console` guide for mixed GUI plus console usage and
the :ref:`gui_help_menus` guide for menu inventory behavior.
