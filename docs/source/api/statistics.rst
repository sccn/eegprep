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
   eegprep.functions.statistics.statcondfieldtrip
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

FieldTrip-style inference
-------------------------

``statcondfieldtrip`` is a standalone Python backend for the scientifically
active behavior in EEGLAB's FieldTrip wrapper. It accepts condition arrays
directly and does not require MATLAB, EEGLAB, or FieldTrip at runtime. Cases
occupy the final axis by default; all preceding feature axes are preserved in
the statistic, p-value, and mask.

Supported designs are paired or equal-variance unpaired two-condition t-tests
and unpaired one-way ANOVA. Analytic inference and seeded Monte Carlo
permutation inference are available. Multiple-comparison options are ``none``,
``bonferroni``, ``holm``, ``fdr``, and Monte Carlo ``max`` correction;
``bonferoni`` and ``holms`` remain accepted migration spellings.

As in FieldTrip, Bonferroni, Holm, and FDR leave the reported pointwise
``pvalue`` unchanged and apply their correction to ``mask``. Monte Carlo
probabilities use a plus-one estimate, so finite randomization runs cannot
report zero probability. Max-statistic correction returns family-wise
corrected probabilities and masks; two-condition tests use absolute t values
for conventional two-sided inference. This fulfills the EEGLAB wrapper's
documented two-tailed output contract directly; FieldTrip's internal
``correcttail='alpha'`` representation instead pairs a one-tail probability
with a halved alpha threshold.

Paired one-way and two-way ANOVA are rejected because the maintained EEGLAB
test disables those FieldTrip paths. Cluster correction and spatial-neighbour
inputs are also rejected: faithful cluster inference requires an explicit
adjacency graph and cluster-forming/statistic policy. Use ``statcond`` for the
supported paired and two-way ANOVA designs without FieldTrip correction.

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
