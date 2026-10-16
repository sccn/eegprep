.. _api_ica_and_components:

ICA and Components
==================

Decomposition, component classification, and component measures.

Low-level decomposition helpers accept channel/component-major matrices. The
``runica_ml2``, ``runica_mlb``, and ``runicalowmem`` compatibility names all
delegate to the maintained ``runica`` implementation, preventing numerical
drift between historical copies of the infomax engine. ``runica`` and
``runicalowmem`` accept EEGLAB positional name/value pairs, such as
``runica(data, "extended", 1)``, in addition to keywords. ``runica.m`` line 205
sets ``DEFAULT_RESETRANDOMSEED = true`` (clock reseed at lines 816-817), while
the help at lines 48-50 says the default has been ``'off'`` since 2015.
EEGPrep follows the help text: an omitted ``rndreset`` is off and uses
``RandomState(5489)``, the same MT19937 stream as ``rng(5489, 'twister')``
(``pythoncompat``, line 814), rather than the legacy ``rand('state', 0)``
stream at line 819. ``varsort`` and ``zica`` return
zero-based component and sample indices.

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
   eegprep.optimal_kmeans
   eegprep.picard
   eegprep.posact
   eegprep.promax
   eegprep.robust_kmeans
   eegprep.spher
   eegprep.runamica
   eegprep.runica
   eegprep.runica_ml2
   eegprep.runica_mlb
   eegprep.runicalowmem
   eegprep.runpca
   eegprep.runpca2
   eegprep.varimax
   eegprep.varsort
   eegprep.zica
