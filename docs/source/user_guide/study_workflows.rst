.. _study_workflows:

===============
STUDY Workflows
===============

EEGPrep includes the first standalone STUDY/session surfaces needed for
group-level workflows. These surfaces are intentionally small on this branch so
they can coexist with the Phase 5 STUDY workers.

Implemented Workflows
=====================

Create a STUDY from loaded datasets:

.. code-block:: python

   from eegprep import pop_study

   STUDY, ALLEEG, com = pop_study(None, ALLEEG, name="My study")

Create a simple ERP STUDY:

.. code-block:: python

   from eegprep import pop_studyerp

   STUDY, ALLEEG, com = pop_studyerp(ALLEEG)

Load and save EEGPrep ``.study`` files:

.. code-block:: python

   from eegprep import pop_loadstudy, pop_savestudy

   STUDY, com = pop_savestudy(STUDY, EEG, "analysis.study")
   STUDY, ALLEEG, com = pop_loadstudy("analysis.study")

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
on ``STUDY.cluster[0]`` until preclustering and cluster editing land in the next
STUDY phase. Cached measure fields follow EEGLAB names such as ``erpdata``,
``specdata``, ``erspdata``, and ``itcdata``. The selected ``design`` is recorded
in each measure group's metadata, but Phase 5b arrays are dataset-level averages
and are not split into EEGLAB design cells yet.

Session Synchronization
=======================

The GUI and ``eegprep-console`` share ``STUDY`` and ``CURRENTSTUDY`` through
``EEGPrepSession``. Creating or loading a STUDY from the GUI sets
``CURRENTSTUDY`` to ``1``. Retrieving a dataset from the Datasets menu returns
``CURRENTSTUDY`` to ``0`` and records that transition in history.

Pending Phase 5 Surfaces
========================

Preclustering, clustering, and cluster-edit actions remain Phase 5 placeholders
until those branches land. Help text for implemented STUDY actions is packaged
now; pending actions should receive expanded help resources when their
implementation merges.

See the :ref:`interactive_console` guide for mixed GUI plus console usage and
the :ref:`gui_help_menus` guide for menu inventory behavior.
