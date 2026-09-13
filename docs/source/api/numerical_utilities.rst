.. _api_numerical_utilities:

===================
Numerical Utilities
===================

EEGPrep includes the small numerical building blocks used by its signal and
component workflows. They are standalone NumPy/SciPy functions and do not need
MATLAB or an EEGLAB checkout.

Indexing and compatibility
==========================

Array indices are zero-based. This applies to peak frames returned by
``abspeak``, assignments returned by ``hungarian`` and ``matcorr``, permutation
vectors, ``matsel`` selections, and ``shuffle`` axes. Kernel peak coordinates
remain one-based because they describe the sampled coordinate system in the
EEGLAB formulas rather than Python array indices.

``covary`` deliberately retains EEGLAB's grand-mean centering before computing
column second moments; use NumPy's variance functions when ordinary per-column
centering is intended. ``quantile`` retains the midpoint empirical-probability
rule used by the current EEGLAB tests. ``vectdata`` supports linear, cubic, and
nearest-neighbor interpolation. MATLAB's legacy biharmonic ``griddata``
``v4`` mode has no well-defined one-dimensional SciPy equivalent and raises
``NotImplementedError`` rather than substituting a different interpolator.

Peaks, summaries, and transforms
================================

.. autosummary::
   :toctree: generated/

   eegprep.abspeak
   eegprep.averef
   eegprep.covary
   eegprep.datlim
   eegprep.eucl
   eegprep.means
   eegprep.nan_mean
   eegprep.nan_std
   eegprep.quantile
   eegprep.vectdata

Kernels
-------

.. autosummary::
   :toctree: generated/

   eegprep.gauss
   eegprep.gauss2d
   eegprep.gauss3d
   eegprep.gabor2d
   eegprep.laplac2d

Matching and component projections
==================================

.. autosummary::
   :toctree: generated/

   eegprep.hungarian
   eegprep.mapcorr
   eegprep.matcorr
   eegprep.matperm
   eegprep.pcsquash
   eegprep.pcexpand
   eegprep.perminv
   eegprep.uniquef

Low-level compatibility helpers
===============================

.. autosummary::
   :toctree: generated/

   eegprep.celltomat
   eegprep.eyelike
   eegprep.fastif
   eegprep.matsel
   eegprep.mattocell
   eegprep.scanfold
   eegprep.shuffle
