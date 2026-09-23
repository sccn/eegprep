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
``bonferroni``, ``holm``, ``fdr``, and Monte Carlo ``max`` or ``cluster``
correction; ``bonferoni`` and ``holms`` remain accepted migration spellings.

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
test disables those FieldTrip paths. Use ``statcond`` for the supported paired
and two-way ANOVA designs without FieldTrip correction.

Cluster correction requires ``method="montecarlo"`` and an explicit
``neighbours`` adjacency matrix. The matrix is square and symmetric over every
feature in C-order-flattened result space; it may be a dense NumPy array or a
SciPy sparse matrix. Its diagonal is ignored. EEGPrep does not infer sensor,
time, or frequency neighbours because bare condition arrays contain no axis
metadata. Construct the complete adjacency graph before calling the function.
For channel-only statistics, ``std_prepare_neighbors`` already returns the
usable matrix as ``limostruct["channeighbstructmat"]``. Add time/frequency
connectivity explicitly when those dimensions are present.

The supported cluster policy is a parametric cluster-forming threshold and
FieldTrip's default ``clusterstatistic="maxsum"``. Set ``clusteralpha`` to
derive the critical t or F value, or provide a positive ``clustercritval``
directly. Positive and negative t-value clusters are formed separately. The
largest absolute cluster mass across both signs is retained for every
permutation, giving one direct two-sided family-wise null distribution; F tests
use their right tail. This is intentionally more explicit than FieldTrip's
internal representation using separate one-tail probabilities and a corrected
alpha threshold.

Paired two-condition permutations swap labels independently within each case;
unpaired permutations pool case labels and split them back into the original
group sizes. Integer ``naccu`` values use a seedable Monte Carlo sample and the
plus-one probability estimate. ``naccu="all"`` enumerates the complete null
space for designs requiring at most 100,000 assignments and reports exact
frequencies. Cluster-corrected ``pvalue`` is constant within each observed
cluster and one outside clusters. ``cluster_labels``, ``clusters``,
``cluster_null``, and ``cluster_critical_value`` expose the inference details.
Automatic channel geometry, nonparametric cluster-forming thresholds,
max-size/weighted cluster statistics, minimum-neighbour pruning, and TFCE are
explicitly unsupported.

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
