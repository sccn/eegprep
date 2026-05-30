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

Plot currently supported STUDY channel measures:

.. code-block:: python

   from eegprep import pop_chanplot

   STUDY, com, fig = pop_chanplot(STUDY, ALLEEG, measure="erp", return_com=True)

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

Phase 5 Coordination
====================

STUDY precompute and full measure plotting remain coordinated with Phase 5b.
Phase 5c reads component ERP, spectrum, ERSP, and ITC clustering inputs from
``STUDY["etc"]["eegprep"]["component_measures"]`` and can build scalp-map
features directly from loaded ICA maps.

See the :ref:`interactive_console` guide for mixed GUI plus console usage and
the :ref:`gui_help_menus` guide for menu inventory behavior.
