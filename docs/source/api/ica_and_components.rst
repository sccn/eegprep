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
stream at line 819. A seeded ``pop_runica`` therefore matches EEGLAB
``pop_runica`` with ``pythoncompat`` on. ``pop_runica`` and ``eeg_runica``
pass ``lrate`` 0.001 unless the caller supplies one (``pop_runica.m`` lines
503 and 523). The history command records that value only when the caller
supplied it. ``varsort`` and ``zica`` return
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
   eegprep.eeg_mir
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

.. _mutual_information_reduction:

Mutual information reduction
----------------------------

``eeg_mir`` rates a decomposition by its mutual information reduction (MIR):
the mutual information that ``icaweights`` removes from the sphered data
``icasphere @ data``, in nats (Delorme et al., 2012,
`doi:10.1371/journal.pone.0030135 <https://doi.org/10.1371/journal.pone.0030135>`_).
It returns the MIR and the variance of that estimate:

.. code-block:: python

   from eegprep import eeg_mir

   mir, variance = eeg_mir(EEG)

With ``W = icaweights``, ``x = icasphere @ data`` and ``y = W @ x``, MIR is
``log|det W| + sum(h(x_i)) - sum(h(y_i))``, where ``h`` is a marginal
differential entropy. Identity weights give 0, and rescaling, reordering or
flipping the sign of components leaves MIR unchanged. Larger is better.

This is equation 17 of Delorme et al. (2012) with ``x`` taken after sphering,
so MIR rates what the unmixing adds to the decorrelation that sphering does.
The paper measures MIR from the recorded channels with the full unmixing and
reports it in kbits/s, and the commented-out ``mir`` of NEMAR's
``eeg_nemar_dataqual.m`` passes the channel data with
``icaweights @ icasphere``. Their values include the reduction that sphering
achieves, so they are not comparable with the values of ``eeg_mir``.

Each entropy is the binned estimator of ``getMIR.m``, as pAMICA ports it in
``pamica.metrics.mir``, with a fixed 256 bins between a row's minimum and
maximum. ``getMIR.m`` takes ``round(3 * log2(1 + N / 10))`` bins for ``N``
samples, a count that grows with the recording. On two unit-variance uniform
sources rotated by 45 degrees, whose MIR is ``1 - ln 2`` (0.3069) nats, that
rule reads 0.046 nats low at 5,000 samples and 0.026 nats low at 1.2 million,
while 256 bins read 0.005 nats low at every length. On the 32 components of
``sample_data/eeglab_data_epochs_ica.set`` (30,720 samples), the length-based
count of 35 bins gives 0.851 nats, 256 bins give 0.993, and 512 bins give 0.994.
``eeg_mir`` needs no optional dependency, and it returns the numbers of
``pamica.metrics.mir(icaweights, icasphere @ data, nbins=256)`` to within
rounding in the last bit.

Epochs are concatenated, and only the channels in ``icachansind`` are used.
``icaact`` is not read: the components are recomputed from ``data``, a block
of 8,192 samples at a time, so the memory ``eeg_mir`` needs beyond the dataset
stays near 18 MiB for 32 components and 73 MiB for 128, whatever the length.
``icasphere`` may have fewer rows than channels, as after AMICA's ``pcakeep``,
and MIR is then measured in that subspace. ``icaweights`` must be square, so a
``runica`` decomposition reduced by PCA, which keeps the reduction in
``icaweights``, raises ``ValueError``. A singular ``icaweights`` or
``icasphere``, NaN or infinite values, and a constant sphered row or component
also raise ``ValueError``.
