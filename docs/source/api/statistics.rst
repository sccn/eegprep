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

.. autofunction:: eegprep.functions.statistics.ttest_cell

.. autofunction:: eegprep.functions.statistics.ttest2_cell

.. autofunction:: eegprep.functions.statistics.anova1_cell

.. autofunction:: eegprep.functions.statistics.anova1rm_cell

.. autofunction:: eegprep.functions.statistics.anova2_cell

.. autofunction:: eegprep.functions.statistics.anova2rm_cell

Multiple Comparisons And Surrogates
===================================

.. autofunction:: eegprep.functions.statistics.fdr

.. autofunction:: eegprep.functions.statistics.stat_surrogate_pvals

.. autofunction:: eegprep.functions.statistics.stat_surrogate_ci

.. autofunction:: eegprep.functions.statistics.surrogdistrib

Data Helpers
============

.. autofunction:: eegprep.functions.statistics.concatdata

.. autofunction:: eegprep.functions.statistics.corrcoef_cell

.. autofunction:: eegprep.functions.statistics.teststat
