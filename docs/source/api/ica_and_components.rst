.. _api_ica_and_components:

ICA and Components
==================

Decomposition, component classification, and component measures.

Low-level decomposition helpers accept channel/component-major matrices. The
``runica_ml2``, ``runica_mlb``, and ``runicalowmem`` compatibility names all
delegate to the maintained ``runica`` implementation, preventing numerical
drift between historical copies of the infomax engine. ``varsort`` and ``zica``
return zero-based component and sample indices.

.. autosummary::
   :toctree: generated/

   eegprep.ICL_feature_extractor
   eegprep.ICADefaults
   eegprep.compvar
   eegprep.eeg_amica
   eegprep.eeg_autocorr
   eegprep.eeg_autocorr_fftw
   eegprep.eeg_autocorr_welch
   eegprep.eeg_getica
   eegprep.eeg_icalabelstat
   eegprep.eeg_icflag
   eegprep.eeg_picard
   eegprep.eeg_pv
   eegprep.eeg_pvaf
   eegprep.eeg_rpsd
   eegprep.eeg_runica
   eegprep.icaact
   eegprep.icadefs
   eegprep.icaproj
   eegprep.icavar
   eegprep.iclabel
   eegprep.iclabel_async
   eegprep.optimal_kmeans
   eegprep.picard
   eegprep.posact
   eegprep.promax
   eegprep.robust_kmeans
   eegprep.runica
   eegprep.runica_ml2
   eegprep.runica_mlb
   eegprep.runicalowmem
   eegprep.runpca
   eegprep.runpca2
   eegprep.varimax
   eegprep.varsort
   eegprep.zica
