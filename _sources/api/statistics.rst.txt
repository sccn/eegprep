.. _api_statistics:

====================
Statistics Functions
====================

EEGPrep statistics helpers mirror the useful parts of EEGLAB's
``functions/statistics`` package while using explicit NumPy contracts. Unless
noted otherwise, condition arrays use their final axis for cases, subjects, or
surrogate replications.

Condition Tests
===============

.. autofunction:: eegprep.functions.statistics.statcond
   :no-index:

.. autofunction:: eegprep.functions.statistics.ttest_cell
   :no-index:

.. autofunction:: eegprep.functions.statistics.ttest2_cell
   :no-index:

.. autofunction:: eegprep.functions.statistics.anova1_cell
   :no-index:

.. autofunction:: eegprep.functions.statistics.anova1rm_cell
   :no-index:

.. autofunction:: eegprep.functions.statistics.anova2_cell
   :no-index:

.. autofunction:: eegprep.functions.statistics.anova2rm_cell
   :no-index:

Multiple Comparisons And Surrogates
===================================

.. autofunction:: eegprep.functions.statistics.fdr
   :no-index:

.. autofunction:: eegprep.functions.statistics.stat_surrogate_pvals
   :no-index:

.. autofunction:: eegprep.functions.statistics.stat_surrogate_ci
   :no-index:

.. autofunction:: eegprep.functions.statistics.surrogdistrib
   :no-index:

Data Helpers
============

.. autofunction:: eegprep.functions.statistics.concatdata
   :no-index:

.. autofunction:: eegprep.functions.statistics.corrcoef_cell
   :no-index:

.. autofunction:: eegprep.functions.statistics.teststat
   :no-index:
