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

``statcond(..., return_resampling_array=True)`` returns a
``SurrogateDistribution``. With ``arraycomp="on"`` (the default), it contains
``naccu`` condition grids; every condition keeps its original feature and case
shape. ``arraycomp="off"`` returns one grid and is compatible with EEGLAB's
incremental resampling contract. For ordinary statistical results, the off
mode still computes all ``naccu`` statistics while holding one resampled grid
at a time.

.. autosummary::
   :toctree: generated/

   eegprep.functions.statistics.statcond
   eegprep.functions.statistics.ttest_cell
   eegprep.functions.statistics.ttest2_cell
   eegprep.functions.statistics.anova1_cell
   eegprep.functions.statistics.anova1rm_cell
   eegprep.functions.statistics.anova2_cell
   eegprep.functions.statistics.anova2rm_cell

Two-way factor order
--------------------

``statcond`` represents every two-way result computed from condition arrays as
``TwoWayEffects(rows, columns, interaction)``. Attribute access and iteration
use that order for paired and unpaired designs, including statistics, degrees
of freedom, p-values, and nonparametric outputs.

This deliberately resolves an inconsistency in EEGLAB's MATLAB implementation.
Although ``statcond`` documents rows, columns, and interaction, its unpaired
branch forwards ``anova2_cell`` outputs in columns, rows, and interaction order.
When translating positional unpaired MATLAB results, EEGPrep's ``rows`` value
therefore corresponds to the second MATLAB cell and ``columns`` to the first.

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

Clustering
==========

``kmeans_st`` and ``kmeanscluster`` operate on observations in rows and return
zero-based labels. Supply ``random_state`` for reproducible random starts.

.. autosummary::
   :toctree: generated/

   eegprep.kmeans_st
   eegprep.kmeanscluster

Effective Dimensionality
========================

``numdim`` evaluates zero eigenvalue entropy terms by their analytic limit,
so exactly rank-deficient inputs return a finite effective dimension.

.. autosummary::
   :toctree: generated/

   eegprep.functions.miscfunc.numdim.numdim
