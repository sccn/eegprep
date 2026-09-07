.. _api_statistics:

==========
Statistics
==========

EEGPrep statistics helpers mirror the useful parts of EEGLAB's
``functions/statistics`` package while using explicit NumPy contracts. Unless
noted otherwise, condition arrays use their final axis for cases, subjects, or
surrogate replications.

Condition Tests
===============

.. autosummary::
   :toctree: generated/

   eegprep.functions.statistics.statcond
   eegprep.functions.statistics.ttest_cell
   eegprep.functions.statistics.ttest2_cell
   eegprep.functions.statistics.anova1_cell
   eegprep.functions.statistics.anova1rm_cell
   eegprep.functions.statistics.anova2_cell
   eegprep.functions.statistics.anova2rm_cell

Multiple Comparisons and Surrogates
===================================

.. autosummary::
   :toctree: generated/

   eegprep.functions.statistics.fdr
   eegprep.functions.statistics.stat_surrogate_pvals
   eegprep.functions.statistics.stat_surrogate_ci
   eegprep.functions.statistics.surrogdistrib

Data Helpers
============

.. autosummary::
   :toctree: generated/

   eegprep.functions.statistics.concatdata
   eegprep.functions.statistics.corrcoef_cell
   eegprep.functions.statistics.teststat
